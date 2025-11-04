"""
Dataset, transforms, and dataloaders for fire classification.
"""
import os
from pathlib import Path
from typing import Dict, Optional, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# Disable PIL image size limit to avoid decompression bomb warnings
Image.MAX_IMAGE_PIXELS = None


class FireDataset(Dataset):
    """Dataset for fire vs non-fire binary classification."""
    
    def __init__(
        self,
        split: str,
        root: str,
        img_size: int = 224,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
        augment_cfg: Optional[Dict] = None,
        need_dct: bool = False
    ):
        """
        Args:
            split: 'train' or 'val'
            root: root directory containing train/val subdirectories
            img_size: target image size
            mean: RGB normalization mean
            std: RGB normalization std
            augment_cfg: augmentation config dict
            need_dct: if True, also return grayscale image for DCT
        """
        self.split = split
        self.root = Path(root)
        self.img_size = img_size
        self.need_dct = need_dct
        
        # Class mapping: nonfire=0, fire=1
        # Support both 'nonfire' and 'non_fire' directory names
        self.classes = ['nonfire', 'fire']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Load image paths
        split_dir = self.root / split
        self.samples = []
        self.class_counts = {}
        
        # Determine actual directory names (support both nonfire and non_fire)
        # Check which directory name exists
        if (split_dir / 'nonfire').exists():
            class_dirs = ['nonfire', 'fire']
        elif (split_dir / 'non_fire').exists():
            class_dirs = ['non_fire', 'fire']
        else:
            class_dirs = ['nonfire', 'fire']  # Default, will skip if doesn't exist
        
        for idx, class_name in enumerate(self.classes):
            dir_name = class_dirs[idx]
            class_dir = split_dir / dir_name
            if not class_dir.exists():
                self.class_counts[class_name] = 0
                continue
            
            label = self.class_to_idx[class_name]
            count = 0
            for img_path in class_dir.glob('*'):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    self.samples.append((str(img_path), label))
                    count += 1
            self.class_counts[class_name] = count
        
        # Build transforms
        if split == 'train' and augment_cfg:
            transform_list = []
            if augment_cfg.get('random_resized_crop', False):
                transform_list.append(transforms.RandomResizedCrop(img_size))
            else:
                transform_list.append(transforms.Resize((img_size, img_size)))
            
            if augment_cfg.get('horizontal_flip', False):
                transform_list.append(transforms.RandomHorizontalFlip())
            
            if augment_cfg.get('color_jitter', False):
                transform_list.append(transforms.ColorJitter())
            
            transform_list.extend([
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
            self.transform = transforms.Compose(transform_list)
        else:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
        
        # Grayscale transform for DCT (no normalization, just to tensor)
        self.gray_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor()
        ])
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_path, label = self.samples[idx]
        
        # Load RGB image
        img = Image.open(img_path).convert('RGB')
        image = self.transform(img)  # [3, H, W]
        
        result = {
            'image': image,
            'label': label
        }
        
        # Optionally compute grayscale for DCT
        if self.need_dct:
            dct_gray = self.gray_transform(img)  # [1, H, W], values in [0, 1]
            result['dct_gray'] = dct_gray
        
        return result


def build_dataloaders(
    root: str,
    img_size: int = 224,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    augment_cfg: Optional[Dict] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    need_dct: bool = False
) -> Tuple[DataLoader, DataLoader]:
    """
    Build train and validation dataloaders.
    
    Returns:
        train_loader, val_loader
    """
    train_dataset = FireDataset(
        split='train',
        root=root,
        img_size=img_size,
        mean=mean,
        std=std,
        augment_cfg=augment_cfg,
        need_dct=need_dct
    )
    
    val_dataset = FireDataset(
        split='val',
        root=root,
        img_size=img_size,
        mean=mean,
        std=std,
        augment_cfg=None,  # No augmentation for val
        need_dct=need_dct
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None,
        drop_last=False
    )
    
    # Print beautiful file counts
    print("\n" + "="*50)
    print("Dataset Statistics")
    print("="*50)
    
    # Train statistics
    print(f"\ntrain:")
    for class_name in ['fire', 'nonfire']:
        count = train_dataset.class_counts.get(class_name, 0)
        print(f"  {class_name}: {count}")
    
    # Validation statistics
    print(f"\nval:")
    for class_name in ['fire', 'nonfire']:
        count = val_dataset.class_counts.get(class_name, 0)
        print(f"  {class_name}: {count}")
    
    print("="*50 + "\n")
    
    return train_loader, val_loader

