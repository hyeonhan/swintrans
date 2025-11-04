"""Dataset and DataLoader utilities with ImageFolder-like behavior.

Automatically discovers classes from train split (alphabetical), and applies the
same mapping to val/test. Provides standard train/val/test transforms.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _is_image_file(path: str) -> bool:
    _, ext = os.path.splitext(path)
    return ext.lower() in IMG_EXTENSIONS


def _discover_classes(train_dir: str) -> List[str]:
    if not os.path.isdir(train_dir):
        raise FileNotFoundError(f"Train directory not found: {train_dir}")
    classes = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
    classes = sorted(classes)
    if not classes:
        raise RuntimeError(f"No class subfolders found under: {train_dir}")
    return classes


def _gather_samples(root_split: str, class_to_idx: Dict[str, int]) -> List[Tuple[str, int]]:
    samples: List[Tuple[str, int]] = []
    if not os.path.isdir(root_split):
        return samples
    for cls in sorted(class_to_idx.keys()):
        cdir = os.path.join(root_split, cls)
        if not os.path.isdir(cdir):
            continue
        for r, _, files in os.walk(cdir):
            for fn in files:
                fp = os.path.join(r, fn)
                if _is_image_file(fp):
                    samples.append((fp, class_to_idx[cls]))
    return samples


class GaussianNoise:
    """Apply Gaussian noise to tensor images.
    
    Args:
        std: Standard deviation of the Gaussian noise. If a tuple (min, max), 
             randomly sample std from this range.
        p: Probability of applying the transform. Default: 1.0
    """
    
    def __init__(self, std: float = 0.1, p: float = 1.0) -> None:
        self.std = std
        self.p = p
    
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() > self.p:
            return img
        
        if isinstance(self.std, (tuple, list)):
            std = torch.empty(1).uniform_(self.std[0], self.std[1]).item()
        else:
            std = self.std
        
        noise = torch.randn_like(img) * std
        return torch.clamp(img + noise, 0.0, 1.0)


class SimpleImageDataset(Dataset):
    """A minimal ImageFolder-like dataset with fixed class_to_idx mapping.
    
    If triple_dataset is True, each image is returned 3 times with different augmentations:
    - Version 0: Original (with standard augmentations)
    - Version 1: With Gaussian blur
    - Version 2: With Gaussian noise
    """

    def __init__(self, samples: List[Tuple[str, int]], transform: Optional[Callable] = None, 
                 triple_dataset: bool = False, blur_transform: Optional[Callable] = None,
                 noise_transform: Optional[Callable] = None) -> None:
        self.triple_dataset = triple_dataset
        if triple_dataset:
            # Expand samples: create 3 entries per image (original, blur, noise)
            expanded_samples: List[Tuple[str, int, int]] = []
            for path, target in samples:
                expanded_samples.append((path, target, 0))  # Original
                expanded_samples.append((path, target, 1))  # Blur
                expanded_samples.append((path, target, 2))  # Noise
            self.samples = expanded_samples
        else:
            # Keep original format: (path, target)
            self.samples = [(path, target, 0) for path, target in samples]
        self.transform = transform
        self.blur_transform = blur_transform
        self.noise_transform = noise_transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        if self.triple_dataset:
            path, target, aug_type = self.samples[idx]
        else:
            path, target, _ = self.samples[idx]
            aug_type = 0
        
        with Image.open(path) as img:
            img = img.convert("RGB")
        
        # Apply appropriate transform based on augmentation type
        if aug_type == 1 and self.blur_transform is not None:
            img = self.blur_transform(img)
        elif aug_type == 2 and self.noise_transform is not None:
            img = self.noise_transform(img)
        elif self.transform is not None:
            img = self.transform(img)
        
        return img, target


def build_transforms(cfg: Dict[str, Any]) -> Tuple[Callable, Callable, Optional[Callable], Optional[Callable]]:
    """Create train and eval torchvision transforms from config.
    
    Returns:
        Tuple of (standard_transform, eval_transform, blur_transform, noise_transform)
        blur_transform and noise_transform are None if tripling is disabled.
    """
    size = int(cfg["data"]["img_size"])
    mean = cfg["data"]["normalize"]["mean"]
    std = cfg["data"]["normalize"]["std"]
    aug = cfg["data"]["augment"]
    triple_dataset = aug.get("triple_dataset", False)

    # Base transforms (applied to all versions)
    t_base = [
        transforms.RandomResizedCrop(size),
    ]
    if aug.get("random_horizontal_flip", True):
        t_base.append(transforms.RandomHorizontalFlip())
    if aug.get("color_jitter", False):
        t_base.append(transforms.ColorJitter(0.4, 0.4, 0.4, 0.1))
    
    # ToTensor and Normalize (applied to all)
    t_tensor_norm = [
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]

    # Standard transform (original version)
    t_train = t_base + t_tensor_norm
    
    # Blur transform (for version 1)
    blur_transform = None
    if triple_dataset:
        blur_kernel_size = aug.get("gaussian_blur", {}).get("kernel_size", 3)
        blur_sigma_config = aug.get("gaussian_blur", {}).get("sigma", (0.1, 2.0))
        # If sigma is a list/tuple, use a fixed value (e.g., midpoint) or first value
        if isinstance(blur_sigma_config, (list, tuple)):
            blur_sigma = (blur_sigma_config[0] + blur_sigma_config[1]) / 2.0 if len(blur_sigma_config) == 2 else blur_sigma_config[0]
        else:
            blur_sigma = blur_sigma_config
        t_blur = [
            transforms.GaussianBlur(kernel_size=blur_kernel_size, sigma=blur_sigma),
        ] + t_base + t_tensor_norm
        blur_transform = transforms.Compose(t_blur)
    
    # Noise transform (for version 2)
    noise_transform = None
    if triple_dataset:
        noise_std_config = aug.get("gaussian_noise", {}).get("std", 0.1)
        # If std is a list/tuple, use a fixed value (e.g., midpoint) or first value
        if isinstance(noise_std_config, (list, tuple)):
            noise_std = (noise_std_config[0] + noise_std_config[1]) / 2.0 if len(noise_std_config) == 2 else noise_std_config[0]
        else:
            noise_std = noise_std_config
        t_noise = t_base + [
            transforms.ToTensor(),
            GaussianNoise(std=noise_std, p=1.0),  # Always apply noise when this transform is used
        ] + [transforms.Normalize(mean=mean, std=std)]
        noise_transform = transforms.Compose(t_noise)
    
    # If not tripling but old style augmentations enabled, keep backward compatibility
    if not triple_dataset:
        if aug.get("gaussian_blur", {}).get("enable", False):
            blur_kernel_size = aug["gaussian_blur"].get("kernel_size", 3)
            blur_sigma = aug["gaussian_blur"].get("sigma", (0.1, 2.0))
            blur_p = aug["gaussian_blur"].get("p", 0.5)
            # Wrap in RandomApply to support probability
            blur_transform_temp = transforms.GaussianBlur(kernel_size=blur_kernel_size, sigma=blur_sigma)
            t_train = [transforms.RandomApply([blur_transform_temp], p=blur_p)] + t_train
        
        if aug.get("gaussian_noise", {}).get("enable", False):
            noise_std = aug["gaussian_noise"].get("std", 0.1)
            noise_p = aug["gaussian_noise"].get("p", 0.5)
            # Insert noise before normalize
            t_train_tensor = t_train[:-1]  # All except normalize
            t_train_normalize = t_train[-1:]  # Just normalize
            t_train = t_train_tensor + [GaussianNoise(std=noise_std, p=noise_p)] + t_train_normalize

    t_eval = transforms.Compose([
        transforms.Resize(int(size * 256 / 224)),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    return transforms.Compose(t_train), t_eval, blur_transform, noise_transform


@dataclass
class DataModule:
    """Build DataLoaders for train/val/test with discovered classes."""

    cfg: Dict[str, Any]
    class_to_idx: Dict[str, int]
    idx_to_class: List[str]
    train: Optional[SimpleImageDataset]
    val: Optional[SimpleImageDataset]
    test: Optional[SimpleImageDataset]

    @classmethod
    def from_config(cls, cfg: Dict[str, Any]) -> "DataModule":
        # Configure PIL max image pixels from config (avoid DecompressionBomb warnings)
        max_pix = cfg["data"].get("max_image_pixels", None)
        if max_pix is None:
            Image.MAX_IMAGE_PIXELS = None
        else:
            try:
                mp = int(max_pix)
                Image.MAX_IMAGE_PIXELS = mp if mp > 0 else None
            except Exception:
                Image.MAX_IMAGE_PIXELS = None

        root = cfg["data"]["root"]
        train_dir = os.path.join(root, "train")
        val_dir = os.path.join(root, "val")
        test_dir = os.path.join(root, "test")

        classes = _discover_classes(train_dir)
        class_to_idx = {c: i for i, c in enumerate(classes)}
        idx_to_class = classes

        train_samples = _gather_samples(train_dir, class_to_idx)
        val_samples = _gather_samples(val_dir, class_to_idx)
        test_samples = _gather_samples(test_dir, class_to_idx)

        if len(train_samples) == 0:
            raise RuntimeError("No training images found. Ensure train/<class> folders contain images.")
        if len(val_samples) == 0:
            print("[Warning] No validation images found. Training will run without validation.")
        if not os.path.isdir(test_dir):
            print("[Info] No test split found. Skipping test.")

        t_train, t_eval, blur_transform, noise_transform = build_transforms(cfg)
        triple_dataset = cfg["data"]["augment"].get("triple_dataset", False)

        train_ds = SimpleImageDataset(
            train_samples, 
            transform=t_train,
            triple_dataset=triple_dataset,
            blur_transform=blur_transform,
            noise_transform=noise_transform
        )
        val_ds = SimpleImageDataset(val_samples, transform=t_eval) if len(val_samples) > 0 else None
        test_ds = SimpleImageDataset(test_samples, transform=t_eval) if len(test_samples) > 0 else None
        
        if triple_dataset:
            print(f"[Dataset] Tripling enabled: {len(train_samples)} images -> {len(train_ds)} samples")

        return cls(cfg=cfg, class_to_idx=class_to_idx, idx_to_class=idx_to_class, train=train_ds, val=val_ds, test=test_ds)

    def loaders(self) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader]]:
        bs = int(self.cfg["data"]["batch_size"])
        nw = int(self.cfg["data"]["num_workers"])
        pin = bool(self.cfg["data"].get("pin_memory", True))
        pw = bool(self.cfg["data"].get("persistent_workers", True))

        train_loader = DataLoader(
            self.train,
            batch_size=bs,
            shuffle=True,
            num_workers=nw,
            pin_memory=pin,
            persistent_workers=pw if nw > 0 else False,
        )
        val_loader = None
        test_loader = None
        if self.val is not None:
            val_loader = DataLoader(
                self.val,
                batch_size=bs,
                shuffle=False,
                num_workers=nw,
                pin_memory=pin,
                persistent_workers=pw if nw > 0 else False,
            )
        if self.test is not None:
            test_loader = DataLoader(
                self.test,
                batch_size=bs,
                shuffle=False,
                num_workers=nw,
                pin_memory=pin,
                persistent_workers=pw if nw > 0 else False,
            )
        return train_loader, val_loader, test_loader



