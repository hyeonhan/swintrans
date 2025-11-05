"""
Dataset and data loading utilities for fire/nonfire classification.

Implements FireDataset with proper transforms, normalization, and DCT band extraction.
"""

import os
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
from collections import defaultdict

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np

# Disable decompression bomb warning for large images
Image.MAX_IMAGE_PIXELS = None

from dct_utils import create_dct_basis, band_split_idct


class FireDataset(Dataset):
    """
    Fire/nonfire classification dataset with DCT band extraction.
    
    Expected structure:
        root_dir/
            train/{fire,nonfire}/
            val/{fire,nonfire}/
    """
    
    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        img_size: int = 224,
        augment: bool = True,
        dct_gray: bool = True,
        class_to_idx: Optional[Dict[str, int]] = None,
        mode: str = "rgbresnet_dctswin",
        dct_block: int = 8,
        use_gray_for_dct: bool = True,
    ):
        """
        Args:
            root_dir: Root directory containing train/val folders
            split: "train" or "val"
            img_size: Target image size (must be multiple of 8)
            augment: Whether to apply data augmentation (for training)
            dct_gray: Whether to compute DCT bands from grayscale
            class_to_idx: Optional class name to index mapping
        """
        assert img_size % 8 == 0, f"img_size must be multiple of 8, got {img_size}"
        
        self.root_dir = Path(root_dir)
        self.split = split
        self.img_size = img_size
        self.augment = augment
        self.dct_gray = dct_gray
        self.mode = mode
        self.dct_block = dct_block
        self.use_gray_for_dct = use_gray_for_dct
        
        # Determine if DCT is needed
        self.needs_dct = "dctswin" in mode
        
        # Find class directories
        split_dir = self.root_dir / split
        if not split_dir.exists():
            raise ValueError(f"Split directory not found: {split_dir}")
        
        # Discover classes (sorted alphabetically for consistency)
        class_dirs = sorted([d.name for d in split_dir.iterdir() if d.is_dir()])
        if len(class_dirs) == 0:
            raise ValueError(f"No class directories found in {split_dir}")
        
        # Build class mapping
        if class_to_idx is None:
            self.class_to_idx = {name: idx for idx, name in enumerate(class_dirs)}
        else:
            self.class_to_idx = class_to_idx
        self.idx_to_class = {idx: name for name, idx in self.class_to_idx.items()}
        
        # Collect image paths
        self.samples = []
        for class_name in class_dirs:
            if class_name not in self.class_to_idx:
                continue
            class_idx = self.class_to_idx[class_name]
            class_dir = split_dir / class_name
            for img_path in class_dir.glob("*.jpg"):
                self.samples.append((str(img_path), class_idx))
            for img_path in class_dir.glob("*.jpeg"):
                self.samples.append((str(img_path), class_idx))
            for img_path in class_dir.glob("*.png"):
                self.samples.append((str(img_path), class_idx))
        
        # RGB transforms
        if self.augment:
            self.rgb_transform = transforms.Compose([
                transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.ToTensor(),
            ])
        else:
            self.rgb_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
            ])
        
        # RGB normalization (ImageNet stats)
        self.rgb_normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        # DCT basis (shared across all samples) - only create if needed
        if self.needs_dct:
            self.D = create_dct_basis(N=dct_block)
        else:
            self.D = None
        
        # Running stats for band map normalization (simple per-image z-score)
        # We'll normalize each band map independently
        self.band_normalize = True
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Optional[torch.Tensor], int]:
        """
        Returns:
            Tuple of (rgb, dct, label):
                - rgb: [3, H, W] normalized RGB tensor
                - dct: [3, H, W] DCT tensor (or None if mode doesn't need DCT)
                - label: class index (int)
        """
        img_path, label = self.samples[idx]
        
        # Load RGB image
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise ValueError(f"Failed to load image {img_path}: {e}")
        
        # Apply RGB transforms
        rgb = self.rgb_transform(img)  # [3, H, W], values in [0, 1]
        rgb_normalized = self.rgb_normalize(rgb)
        
        # Compute DCT if needed
        dct = None
        if self.needs_dct:
            # Convert to grayscale for DCT (luma: 0.299R + 0.587G + 0.114B)
            gray = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]  # [H, W]
            gray = gray.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
            
            # Compute initial band maps with default c1=2.0, c2=4.0
            # The model will recompute these with learnable c1, c2 during forward for gradients
            c1_init = torch.tensor(2.0, device=gray.device)
            c2_init = torch.tensor(4.0, device=gray.device)
            D_device = self.D.to(device=gray.device, dtype=gray.dtype)
            band_low, band_mid, band_high = band_split_idct(
                gray, c1_init, c2_init, D_device, k=50.0
            )
            
            # Remove batch dimension: [1, 1, H, W] -> [1, H, W]
            band_low = band_low.squeeze(0)
            band_mid = band_mid.squeeze(0)
            band_high = band_high.squeeze(0)
            
            # Simple normalization: z-score per image
            if self.band_normalize:
                for band in [band_low, band_mid, band_high]:
                    band_mean = band.mean()
                    band_std = band.std() + 1e-6
                    band.sub_(band_mean).div_(band_std)
            
            # Create DCT tensor based on use_gray_for_dct
            if self.use_gray_for_dct:
                # Single-channel DCT map (use low band as representative) repeated to 3 channels
                # This preserves ImageNet pretrained weights in Swin (in_chans=3)
                dct_single = band_low.squeeze(0)  # [H, W]
                dct = dct_single.unsqueeze(0).repeat(3, 1, 1)  # [3, H, W]
            else:
                # Three-channel DCT: stack low/mid/high bands
                dct = torch.stack([band_low.squeeze(0), band_mid.squeeze(0), band_high.squeeze(0)], dim=0)  # [3, H, W]
            
            # Ensure DCT is normalized to [0, 1] range
            dct_min = dct.min()
            dct_max = dct.max()
            if dct_max > dct_min:
                dct = (dct - dct_min) / (dct_max - dct_min + 1e-6)
        
        return rgb_normalized, dct, label


def collate_fn(batch):
    """
    Custom collate function to handle None DCT values.
    
    Args:
        batch: List of (rgb, dct, label) tuples
        
    Returns:
        Tuple of (rgb_batch, dct_batch, label_batch) where dct_batch can be None
    """
    rgb_list, dct_list, label_list = zip(*batch)
    
    # Stack RGB tensors
    rgb_batch = torch.stack(rgb_list, dim=0)
    
    # Stack DCT tensors if any are not None
    if any(d is not None for d in dct_list):
        # Filter out None values and stack
        dct_valid = [d for d in dct_list if d is not None]
        if len(dct_valid) == len(dct_list):
            dct_batch = torch.stack(dct_list, dim=0)
        else:
            # If some are None, we shouldn't reach here in normal operation
            # But handle it gracefully
            dct_batch = None
    else:
        dct_batch = None
    
    # Stack labels
    label_batch = torch.tensor(label_list, dtype=torch.long)
    
    return rgb_batch, dct_batch, label_batch


def compute_dataset_stats(root: str) -> None:
    """
    Compute and print dataset statistics (class counts for train/val).
    
    Args:
        root: Root directory containing train/val folders
    """
    root_path = Path(root)
    
    print("\n" + "="*60)
    print("Dataset Statistics")
    print("="*60)
    
    for split in ["train", "val"]:
        split_dir = root_path / split
        if not split_dir.exists():
            print(f"\n{split.upper()}: Directory not found")
            continue
        
        # Count samples per class
        class_counts = defaultdict(int)
        total = 0
        
        for class_dir in sorted(split_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            class_name = class_dir.name
            count = len(list(class_dir.glob("*.jpg"))) + \
                    len(list(class_dir.glob("*.jpeg"))) + \
                    len(list(class_dir.glob("*.png")))
            class_counts[class_name] = count
            total += count
        
        print(f"\n{split.upper()}:")
        for class_name in sorted(class_counts.keys()):
            count = class_counts[class_name]
            print(f"  {class_name}: {count}")
        print(f"  Total: {total}")
    
    print("="*60 + "\n")

