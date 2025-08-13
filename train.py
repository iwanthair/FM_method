import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
import torch.nn.functional as F
import torchdiffeq
from PIL import Image
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import copy
import time
import numpy as np
from tqdm import tqdm
from torch.amp.grad_scaler import GradScaler
from torch.amp.autocast_mode import autocast

# Conditional Flow Matching
from torchcfm.conditional_flow_matching import (
    ConditionalFlowMatcher,
    ExactOptimalTransportConditionalFlowMatcher,
    TargetConditionalFlowMatcher,
    VariancePreservingConditionalFlowMatcher,
    SchrodingerBridgeConditionalFlowMatcher
)

from torchcfm.models.unet.unet import UNetModelWrapper
from utils import ema
from dataloader import FloorPlanDataset

WARMUP = 2000
TIME = time.strftime("%Y%m%d-%H%M%S")

def warmup_lr(step):
    return min(step, WARMUP) / WARMUP


def build_model(cfm_type, image_size, device, sigma=0.0):
    in_ch = 1 + 4
    model = UNetModelWrapper(
        dim=(in_ch, image_size, image_size),
        num_channels=64,
        # num_res_blocks=3,
        num_res_blocks=2,
        channel_mult=[1,2,4,8],
        num_heads=4,
        num_head_channels=32,
        attention_resolutions="16",
        # attention_resolutions="32,16,8",
        dropout=0.1,
    ).to(device)

    if cfm_type == 'otcfm':
        matcher = ExactOptimalTransportConditionalFlowMatcher(sigma=sigma)
    elif cfm_type == 'icfm':
        matcher = ConditionalFlowMatcher(sigma=sigma)
    elif cfm_type == 'fm':
        matcher = TargetConditionalFlowMatcher(sigma=sigma)
    elif cfm_type == 'si':
        matcher = VariancePreservingConditionalFlowMatcher(sigma=sigma)
    elif cfm_type == 'sb-cfm':
        matcher = SchrodingerBridgeConditionalFlowMatcher(sigma=sigma)
    else:
        raise ValueError(f"Unknown model {cfm_type}")
    return model, matcher


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, f'_{TIME}_{args.model}'))
    loader = DataLoader(
        FloorPlanDataset(
            heatmap_dir=args.heat_dir,
            traj_dir=args.traj_dir,
            target_dir=args.floor_dir,
            image_size=(args.image_size, args.image_size)
        ),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True,
        pin_memory=True
    )
    val_loader = DataLoader(
        FloorPlanDataset(
            heatmap_dir=args.val_heat_dir,
            traj_dir=args.val_traj_dir,
            target_dir=args.val_floor_dir,
            image_size=(args.image_size, args.image_size)
        ),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=False,
        pin_memory=True
    )

    model, matcher = build_model(
        cfm_type=args.model,
        image_size=args.image_size,
        device=device,
        sigma=0.05
    )
    ema_model = copy.deepcopy(model).to(device).float()
    for p in ema_model.parameters():
        p.requires_grad_(False)


    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-3
    )

    total_steps  = args.epochs * len(loader)
    # warmup_steps = total_steps * 0.3    # 5000，or total_steps * 0.1
    warmup_steps = min(WARMUP, total_steps//10)
    cosine_steps = total_steps - warmup_steps
    print(f"Total steps: {total_steps}, Warmup steps: {warmup_steps}, Cosine steps: {cosine_steps}")

    sched_warmup = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: min(step, warmup_steps) / warmup_steps
    )
    sched_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cosine_steps,
        eta_min=1e-6
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[sched_warmup, sched_cosine],
        milestones=[warmup_steps]
    )

    start_epoch = 0
    # resume from checkpoint
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model_state'])
        ema_model.load_state_dict(ckpt['ema_state'])
        optimizer.load_state_dict(ckpt['optim_state'])
        if ckpt.get('sched_state') is not None:
            scheduler.load_state_dict(ckpt['sched_state'])
        start_epoch = ckpt.get('epoch', 0)
        print(f"Resumed from epoch {start_epoch}")
    else:
        print("Training from scratch.")

    # Create output directories
    model.train()
    for epoch in range(start_epoch, args.epochs):
        epoch_loss = 0.0
        for batch_idx, (cond, floor, _) in enumerate(tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}")):
            floor, cond = floor.to(device), cond.to(device)
            x1 = floor
            x0 = torch.randn_like(x1)
            t, xt, ut = matcher.sample_location_and_conditional_flow(x0, x1)
            x_t_img = xt 
            xt = torch.cat([xt, cond], dim=1)

            vt = model(t, xt)

            w = (t.view(-1,1,1,1) ** args.alpha) * ((1 - t.view(-1,1,1,1)) ** args.beta)
            # w = 1.0
            loss_map = (vt[:,0:1] - ut)**2
            flow_loss = torch.mean(loss_map * w)
            scale = (1.0 - t).reshape(-1, 1, 1, 1)
            x_pred = x_t_img + scale * vt[:,0:1]
            x_pred = x_pred.clamp(-1, 1)

            # mse loss
            # img_loss  = torch.mean((x_pred - x1) ** 2)
            # mae loss
            # img_loss  = torch.mean(torch.abs(x_pred - x1))
            # L1 loss
            img_loss = F.l1_loss(x_pred, x1)

            lam_rec = args.img_loss_weight

            loss = flow_loss + lam_rec * img_loss
            # loss = img_loss

            epoch_loss += loss.item()
            global_step = epoch * len(loader) + batch_idx
            writer.add_scalar('Loss/flow', flow_loss.item(), global_step)
            writer.add_scalar('Loss/image', img_loss.item(), global_step)
            writer.add_scalar('Loss/train_step', loss.item(), global_step)
            writer.add_scalar('LR', scheduler.get_last_lr()[0], global_step)
            # backward and update
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            # update scheduler and EMA
            scheduler.step()
            ema(model, ema_model, args.ema_decay)

        # log epoch loss
        avg_loss = epoch_loss / len(loader)
        writer.add_scalar('Loss/train_epoch', avg_loss, epoch+1)
        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {avg_loss:.4f}")

        if (epoch + 1) % args.save_epoch == 0:
            ckpt = {
                'epoch': epoch + 1,
                'model_state': model.state_dict(),
                'ema_state': ema_model.state_dict(),
                'optim_state': optimizer.state_dict(),
                'sched_state': scheduler.state_dict(),
                # 'scaler_state': scaler.state_dict(),  # 若启用 AMP 再加
                # 'ema_updates': ema_updates,
                'args': vars(args),
            }
            torch.save(ckpt, os.path.join(args.output_dir, 'checkpoints', f'ckpt_epoch_{epoch+1}.pt'))
            print(f"Saved checkpoint at epoch {epoch+1}")
            
        if (epoch + 1) % args.val_interval == 0:
            val_dir = os.path.join(args.output_dir, 'val', str(epoch+1))
            os.makedirs(val_dir, exist_ok=True)
            model_eval = ema_model if args.use_ema else model
            model_eval.eval().float()
            total_val_loss = 0.0


            with torch.no_grad():
                num_samples = 0
                for idx, (val_cond, val_floor, fname) in enumerate(tqdm(val_loader, desc=f"Val {epoch+1}")):
                    x1, cond_map = val_floor.to(device), val_cond.to(device)
                    x0 = torch.randn_like(x1)
                    B = x1.shape[0]
                    def ode_fn(t, x):
                        tt = t * torch.ones(B, device=device)
                        inp = torch.cat([x, cond_map], dim=1)
                        vt = model_eval(tt, inp)
                        dx = vt[:, 0:1]
                        return dx

                    t_span = torch.linspace(0.0, 1.0, args.ode_steps, device=device)
                    traj = torchdiffeq.odeint(
                        ode_fn,
                        x0,
                        t_span,
                        atol=1e-4,
                        rtol=1e-4,
                        method="dopri5",
                    )
                    x_pred = traj[-1]

                    # mse loss
                    # loss = torch.mean((x_pred - x1) ** 2)
                    # mae loss
                    loss = torch.mean(torch.abs(x_pred - x1))
                    
                    mae_loss = loss.item() * B
                    total_val_loss += mae_loss
                    num_samples += B

                    if idx < 5:
                        # write the x1 and cond map to tensorboard
                        writer.add_images('Val/x1', (x1 * 0.5 + 0.5).clamp(0, 1), epoch+1)
                        # four channel cond map
                        writer.add_images('Val/heat_map', (cond_map[:, :3] * 0.5 + 0.5).clamp(0, 1), epoch+1)
                        writer.add_images('Val/traj_map', (cond_map[:, 3:4] * 0.5 + 0.5).clamp(0, 1), epoch+1)
                        writer.add_images('Val/pred', (x_pred * 0.5 + 0.5).clamp(0, 1), epoch+1)
                        for b in range(5):
                            pred_img = (x_pred[b].cpu() * 0.5 + 0.5).clamp(0,1)
                            gt_img   = (x1[b].cpu()      * 0.5 + 0.5).clamp(0,1)
                            save_image(pred_img, os.path.join(val_dir, f'pred_{fname[b]}'))
                            save_image(gt_img,   os.path.join(val_dir, f'gt_{fname[b]}'))

            avg_val_loss = total_val_loss / num_samples
            writer.add_scalar('Loss/val_epoch', avg_val_loss, epoch+1)
            print(f"Validation Epoch {epoch+1}, Loss: {avg_val_loss:.4f}")

            model_eval.train()
            torch.cuda.empty_cache()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--floor_dir', default='Dataset_Scale100_SExPE/train/Target/')
    parser.add_argument('--heat_dir', default='Dataset_Scale100_SExPE/train/Condition_1/')
    parser.add_argument('--traj_dir', default='Dataset_Scale100_SExPE/train/Condition_2/')
    parser.add_argument('--val_floor_dir', default='Dataset_Scale100_SExPE/test/Target/')
    parser.add_argument('--val_heat_dir', default='Dataset_Scale100_SExPE/test/Condition_1/')
    parser.add_argument('--val_traj_dir', default='Dataset_Scale100_SExPE/test/Condition_2/')
    parser.add_argument('--output_dir', default=f'./results/{TIME}_sexpe/')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--image_size', type=int, default=128)
    parser.add_argument('--total_steps', type=int, default=40000)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--ema_decay', type=float, default=0.9999)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--save_epoch', type=int, default=10)
    parser.add_argument('--val_interval', type=int, default=50)
    parser.add_argument('--model', choices=['otcfm','icfm','sb-cfm', 'fm', 'si'], default='icfm',)
    parser.add_argument('--sigma', type=float, default=0.05, help='sigma for noise in CFM')
    parser.add_argument('--alpha', type=float, default=1.0, help='alpha for t weighting')
    parser.add_argument('--beta', type=float, default=0.5, help='beta for t weighting')
    parser.add_argument('--img_loss_weight', type=float, default=0.5, help='weight for image loss')
    parser.add_argument('--ode_steps', type=int, default=50, help='number of steps for ODE solver')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--use_ema', action='store_true', default=True, help='use EMA model for evaluation')
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'val'), exist_ok=True)
    train(args)


if __name__ == "__main__":
    main()
