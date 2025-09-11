import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class FloorPlanDataset(Dataset):
    def __init__(self, heatmap_dir, traj_dir, target_dir=None, image_size=(128, 128)):
        self.heatmap_dir = heatmap_dir
        self.traj_dir = traj_dir
        self.target_dir = target_dir
        self.image_size = image_size

        # have same number of images in all directories
        if target_dir is not None:
            print(f"length of target_dir: {len(os.listdir(self.target_dir))}")
        if target_dir is None:
            assert len(os.listdir(self.heatmap_dir)) == len(os.listdir(self.traj_dir))
        else:
            assert len(os.listdir(self.heatmap_dir)) == len(os.listdir(self.traj_dir)) == len(os.listdir(self.target_dir))

        self.filenames = sorted([
            fname for fname in os.listdir(self.heatmap_dir)
            if fname.endswith('.png')
        ])

        # transforms
        self.transform_heatmap = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.transform_gray = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, index):
        fname = self.filenames[index]

        heatmap_path = os.path.join(self.heatmap_dir, fname)
        traj_path = os.path.join(self.traj_dir, fname)

        heatmap_img = Image.open(heatmap_path).convert('RGB')  # 3 channel
        traj_img = Image.open(traj_path).convert('L')          # 1 channel
        

        heatmap_tensor = self.transform_heatmap(heatmap_img)
        traj_tensor = self.transform_gray(traj_img)
        

        cond_image = torch.cat([heatmap_tensor, traj_tensor], dim=0)

        if self.target_dir is not None:
            target_path = os.path.join(self.target_dir, fname)
            target_img = Image.open(target_path).convert('L')
            target_tensor = self.transform_gray(target_img)
        else:
            target_tensor = torch.zeros_like(traj_tensor)

        return cond_image, target_tensor, fname
    


class FloorPlanDatasetTrajOnly(Dataset):
    def __init__(self, traj_dir, target_dir=None, image_size=(128, 128)):
        self.traj_dir = traj_dir
        self.target_dir = target_dir
        self.image_size = image_size

        # have same number of images in all directories
        if target_dir is not None:
            print(f"length of target_dir: {len(os.listdir(self.target_dir))}")
            assert len(os.listdir(self.traj_dir)) == len(os.listdir(self.target_dir))
            
        self.filenames = sorted([
            fname for fname in os.listdir(self.traj_dir)
            if fname.endswith('.png')
        ])

        # transforms
        self.transform_gray = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, index):
        fname = self.filenames[index]

        traj_path = os.path.join(self.traj_dir, fname)
        traj_img = Image.open(traj_path).convert('L')          # 1 channel

        traj_tensor = self.transform_gray(traj_img)

        if self.target_dir is not None:
            target_path = os.path.join(self.target_dir, fname)
            target_img = Image.open(target_path).convert('L')
            target_tensor = self.transform_gray(target_img)
        else:
            target_tensor = torch.zeros_like(traj_tensor)

        return traj_tensor, target_tensor, fname
    

class FloorPlanDatasetHeatMapOnly(Dataset):
    def __init__(self, heatmap_dir, target_dir=None, image_size=(128, 128)):
        self.heatmap_dir = heatmap_dir
        self.target_dir = target_dir
        self.image_size = image_size

        # have same number of images in all directories
        if target_dir is not None:
            print(f"length of target_dir: {len(os.listdir(self.target_dir))}")
            assert len(os.listdir(self.heatmap_dir)) == len(os.listdir(self.target_dir))

        self.filenames = sorted([
            fname for fname in os.listdir(self.heatmap_dir)
            if fname.endswith('.png')
        ])

        # transforms
        self.transform_heatmap = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.transform_gray = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, index):
        fname = self.filenames[index]

        heatmap_path = os.path.join(self.heatmap_dir, fname)
        heatmap_img = Image.open(heatmap_path).convert('RGB')  # 3 channel
        
        heatmap_tensor = self.transform_heatmap(heatmap_img)

        if self.target_dir is not None:
            target_path = os.path.join(self.target_dir, fname)
            target_img = Image.open(target_path).convert('L')
            target_tensor = self.transform_gray(target_img)
        else:
            target_tensor = torch.zeros_like(heatmap_tensor[0:1, :, :])  # make it 1 channel

        return heatmap_tensor, target_tensor, fname