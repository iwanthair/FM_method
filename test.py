# test.py
import os
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import torchdiffeq
from tqdm import tqdm
import numpy as np


from torchcfm.conditional_flow_matching import (
    ConditionalFlowMatcher,
    ExactOptimalTransportConditionalFlowMatcher,
    TargetConditionalFlowMatcher,
    VariancePreservingConditionalFlowMatcher,
    SchrodingerBridgeConditionalFlowMatcher
)
from torchcfm.models.unet.unet import UNetModelWrapper
from dataloader import FloorPlanDataset
import time

TIME = time.strftime("%Y%m%d-%H%M%S")
CHECKPOINT = "results/20250813-010347_sepe/checkpoints/ckpt_epoch_150.pt"
TYPE = "sepe"
EXP = 1

@torch.no_grad()
def build_model(cfm_type, image_size, device, sigma=0.0):
    in_ch = 1 + 4  # floor(1) + cond(4)
    model = UNetModelWrapper(
        dim=(in_ch, image_size, image_size),
        num_channels=64,
        num_res_blocks=2,
        channel_mult=[1, 2, 4, 8],
        num_heads=4,
        num_head_channels=32,
        attention_resolutions="16",
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

@torch.no_grad()
def run_test(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    os.makedirs(args.output_dir, exist_ok=True)
    out_dir = os.path.join(args.output_dir, "test")
    os.makedirs(out_dir, exist_ok=True)

    # Data
    test_loader = DataLoader(
        FloorPlanDataset(
            heatmap_dir=args.test_heat_dir,
            traj_dir=args.test_traj_dir,
            target_dir=args.test_floor_dir,
            image_size=(args.image_size, args.image_size)
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        drop_last=False,
        pin_memory=True
    )

    # model, _ = build_model(args.model, args.image_size, device, sigma=args.sigma)
    # ema_model = copy.deepcopy(model).to(device).float()
    # for p in ema_model.parameters():
    #     p.requires_grad_(False)

    # eval_model = load_checkpoint(args.checkpoint, device, model, True, args.use_ema_for_eval)
    # eval_model.eval().float()

    model, _ = build_model(args.model, args.image_size, device, sigma=args.sigma)
    ckpt = torch.load(args.checkpoint, map_location=device)

    if args.use_ema_for_eval and ("ema_state" in ckpt):
        model.load_state_dict(ckpt["ema_state"])
    else:
        model.load_state_dict(ckpt["model_state"])

    eval_model = model.to(device).eval().float()


    # ODE schedule
    t_span = torch.linspace(0.0, 1.0, args.ode_steps, device=device, dtype=torch.float32)

    pred_dir = os.path.join(out_dir, "pred")
    gt_dir   = os.path.join(out_dir, "gt")
    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(gt_dir,   exist_ok=True)
    save_num = 0
    for idx, (cond_map, x1, fnames) in enumerate(tqdm(test_loader, desc="Test")):
        cond_map = cond_map.to(device).float()
        x1 = x1.to(device).float()
        # create noise images
        x0 = torch.randn_like(x1)

        def ode_fn(t, x):
            B  = x.shape[0]
            tt = t.expand(B).to(x.dtype).to(x.device)
            inp = torch.cat([x, cond_map], dim=1)
            vt  = eval_model(tt, inp).to(x.dtype)
            return vt[:, 0:1]

        traj   = torchdiffeq.odeint(ode_fn, x0, t_span, atol=1e-4, rtol=1e-4, method="dopri5")
        x_pred = traj[-1]

        if args.save_images:
            vis_pred = (x_pred * 0.5 + 0.5).clamp(0, 1)
            for b in range(x_pred.size(0)):
                save_image(vis_pred[b], os.path.join(pred_dir, f"{fnames[b]}"))
                save_num += 1
                if save_num >= 50:
                    args.save_images = False
                    break

    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_floor_dir', default=None)
    parser.add_argument('--test_heat_dir',  default=f'Dataset_rw/{TYPE}_exp{EXP}_gmap/Condition_1')
    parser.add_argument('--test_traj_dir',  default=f'Dataset_rw/{TYPE}_exp{EXP}_gmap/Condition_2')

    parser.add_argument('--output_dir', default=f'./results_test/rw_exp{EXP}_{TYPE}_{TIME}/')
    parser.add_argument('--checkpoint', type=str, default=CHECKPOINT, help='path to ckpt .pt')
    parser.add_argument('--use_ema_for_eval', action='store_true', default=True)
    parser.add_argument('--ode_steps', type=int, default=50)
    parser.add_argument('--save_images', action='store_true', default=True)

    parser.add_argument('--model', choices=['otcfm','icfm','sb-cfm','fm','si'], default='icfm')
    parser.add_argument('--sigma', type=float, default=0.05)
    parser.add_argument('--image_size', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=1)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    run_test(args)

if __name__ == "__main__":
    main()
