"""
Dataset and data loading utilities for fire/nonfire classification.

Implements FireDataset with proper transforms, normalization, and DCT band extraction.
"""

import os
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
from collections import defaultdict
import random
import warnings

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image, ImageFilter
import numpy as np

# Disable decompression bomb warning for large images
Image.MAX_IMAGE_PIXELS = None

from dct_utils import create_dct_basis, band_split_idct


# ============================================================================
# Fog Augmentation Functions (Koschmieder Model)
# ============================================================================

def _normalize01(x: np.ndarray) -> np.ndarray:
    """Normalize array to [0, 1] range."""
    x_min = x.min()
    x_max = x.max()
    if x_max > x_min:
        return (x - x_min) / (x_max - x_min + 1e-6)
    return x


def _build_depth_map(
    rgb: np.ndarray,
    mode: str,
    flat_depth: float = 0.5,
    grad_angle_deg: float = 0.0,
    contrast_radius: int = 11,
    contrast_gain: float = 1.2,
    depth_npy_path: Optional[str] = None,
    rng: Optional[random.Random] = None,
) -> np.ndarray:
    """
    Build depth map from RGB image using various strategies.
    
    Args:
        rgb: Input RGB image [H, W, 3] in float32 [0, 1]
        mode: "contrast", "gradient", or "flat"
        flat_depth: Depth value for flat mode
        grad_angle_deg: Gradient angle in degrees for gradient mode
        contrast_radius: Radius for contrast-based depth
        contrast_gain: Gain factor for contrast enhancement
        depth_npy_path: Optional path to load precomputed depth (unused in this impl)
        rng: Random number generator
        
    Returns:
        depth: Depth map [H, W] in float32, normalized to [0, 1]
    """
    H, W = rgb.shape[:2]
    
    if mode == "flat":
        depth = np.full((H, W), flat_depth, dtype=np.float32)
    
    elif mode == "gradient":
        # Create linear gradient based on angle
        angle_rad = np.deg2rad(grad_angle_deg)
        center_x, center_y = W / 2.0, H / 2.0
        
        # Create coordinate grids
        y_coords, x_coords = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        x_coords = x_coords.astype(np.float32) - center_x
        y_coords = y_coords.astype(np.float32) - center_y
        
        # Rotate coordinates
        x_rot = x_coords * np.cos(angle_rad) - y_coords * np.sin(angle_rad)
        
        # Normalize to [0, 1]
        x_rot_min = x_rot.min()
        x_rot_max = x_rot.max()
        if x_rot_max > x_rot_min:
            depth = (x_rot - x_rot_min) / (x_rot_max - x_rot_min)
        else:
            depth = np.full((H, W), 0.5, dtype=np.float32)
    
    elif mode == "contrast":
        # Convert to grayscale
        gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
        
        # Apply Gaussian blur for contrast-based depth
        gray_pil = Image.fromarray((gray * 255).astype(np.uint8))
        blurred_pil = gray_pil.filter(ImageFilter.GaussianBlur(radius=contrast_radius))
        blurred = np.array(blurred_pil, dtype=np.float32) / 255.0
        
        # Compute local contrast (abs difference from blurred)
        contrast = np.abs(gray - blurred)
        
        # Enhance contrast
        contrast = contrast * contrast_gain
        
        # Normalize to [0, 1] and invert (high contrast = close, low contrast = far)
        depth = _normalize01(contrast)
        depth = 1.0 - depth  # Invert: high contrast (close) -> low depth value
    
    else:
        raise ValueError(f"Unknown depth mode: {mode}")
    
    # Ensure depth is in [0, 1]
    depth = np.clip(depth, 0.0, 1.0)
    return depth


def _apply_koschmieder(
    rgb: np.ndarray,
    d: np.ndarray,
    beta: float,
    airlight: Union[float, Tuple[float, float, float]],
) -> np.ndarray:
    """
    Apply Koschmieder fog model: I_fog = J·t + A·(1-t), where t = exp(-β·d).
    
    Args:
        rgb: Original RGB image [H, W, 3] in float32 [0, 1]
        d: Depth map [H, W] in float32 [0, 1]
        beta: Attenuation coefficient (higher = thicker fog)
        airlight: Airlight scalar or RGB tuple (3,) in [0, 1]
        
    Returns:
        fogged: Fogged RGB image [H, W, 3] in float32 [0, 1]
    """
    # Compute transmission: t = exp(-β·d)
    t = np.exp(-beta * d)
    t = np.clip(t, 0.0, 1.0)
    
    # Expand t to match RGB channels: [H, W] -> [H, W, 1]
    t = np.expand_dims(t, axis=2)
    
    # Handle airlight: scalar or RGB tuple
    if isinstance(airlight, (int, float)):
        A = np.array([airlight, airlight, airlight], dtype=np.float32)
    else:
        A = np.array(airlight, dtype=np.float32)
    
    # Reshape A to [1, 1, 3] for broadcasting
    A = A.reshape(1, 1, 3)
    
    # Apply Koschmieder: I_fog = J·t + A·(1-t)
    fogged = rgb * t + A * (1.0 - t)
    
    # Clip to [0, 1]
    fogged = np.clip(fogged, 0.0, 1.0)
    return fogged


def _add_bloom(
    rgb: np.ndarray,
    strength: float = 0.0,
    threshold: float = 0.75,
    radius: int = 9,
) -> np.ndarray:
    """
    Add bloom effect (glow) to bright areas.
    
    Args:
        rgb: RGB image [H, W, 3] in float32 [0, 1]
        strength: Bloom strength (0.0 = no bloom)
        threshold: Brightness threshold for bloom
        radius: Gaussian blur radius
        
    Returns:
        bloomed: RGB image with bloom [H, W, 3] in float32 [0, 1]
    """
    if strength <= 0.0:
        return rgb
    
    # Convert to PIL for Gaussian blur
    rgb_uint8 = (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)
    rgb_pil = Image.fromarray(rgb_uint8)
    
    # Compute brightness mask (bright areas above threshold)
    gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    mask = (gray > threshold).astype(np.float32)
    
    # Apply Gaussian blur to bright areas
    mask_pil = Image.fromarray((mask * 255).astype(np.uint8))
    blurred_mask_pil = mask_pil.filter(ImageFilter.GaussianBlur(radius=radius))
    blurred_mask = np.array(blurred_mask_pil, dtype=np.float32) / 255.0
    
    # Create bloom by blurring bright RGB channels
    bloom_pil = rgb_pil.filter(ImageFilter.GaussianBlur(radius=radius))
    bloom = np.array(bloom_pil, dtype=np.float32) / 255.0
    
    # Blend: original + bloom * strength * mask
    bloomed = rgb + bloom * strength * np.expand_dims(blurred_mask, axis=2)
    
    # Clip to [0, 1]
    bloomed = np.clip(bloomed, 0.0, 1.0)
    return bloomed


def _sample_fog_params(cfg: Dict, rng: random.Random) -> Dict:
    """
    Sample fog augmentation parameters from config.
    
    Args:
        cfg: Fog config dictionary
        rng: Random number generator
        
    Returns:
        Dictionary with sampled parameters
    """
    # Validate and normalize depth_mode_probs
    depth_probs = cfg.get("depth_mode_probs", {"contrast": 0.6, "gradient": 0.3, "flat": 0.1})
    prob_sum = sum(depth_probs.values())
    if abs(prob_sum - 1.0) > 0.01:
        warnings.warn(f"depth_mode_probs sum to {prob_sum}, normalizing to 1.0")
        depth_probs = {k: v / prob_sum for k, v in depth_probs.items()}
    
    # Sample depth mode
    modes = list(depth_probs.keys())
    probs = [depth_probs[m] for m in modes]
    depth_mode = rng.choices(modes, weights=probs)[0]
    
    # Sample beta (attenuation coefficient)
    beta_range = cfg.get("beta_range", [0.03, 0.10])
    if len(beta_range) != 2 or beta_range[0] >= beta_range[1]:
        raise ValueError(f"Invalid beta_range: {beta_range}, must be [low, high] with low < high")
    beta = rng.uniform(beta_range[0], beta_range[1])
    
    # Sample airlight
    use_rgb_airlight = cfg.get("use_rgb_airlight", False)
    if use_rgb_airlight:
        airlight_rgb_low = cfg.get("airlight_rgb_low", [0.90, 0.92, 0.94])
        airlight_rgb_high = cfg.get("airlight_rgb_high", [0.98, 0.99, 1.00])
        if len(airlight_rgb_low) != 3 or len(airlight_rgb_high) != 3:
            raise ValueError("airlight_rgb_low and airlight_rgb_high must be 3-element lists")
        airlight = tuple(rng.uniform(airlight_rgb_low[i], airlight_rgb_high[i]) for i in range(3))
    else:
        airlight_range = cfg.get("airlight_scalar_range", [0.9, 1.0])
        if len(airlight_range) != 2 or airlight_range[0] >= airlight_range[1]:
            warnings.warn(f"Invalid airlight_scalar_range: {airlight_range}, defaulting to 0.95")
            airlight = 0.95
        else:
            airlight = rng.uniform(airlight_range[0], airlight_range[1])
    
    # Sample gradient parameters
    grad_angle_range = cfg.get("grad_angle_range", [0, 180])
    grad_angle = rng.uniform(grad_angle_range[0], grad_angle_range[1])
    
    # Sample contrast parameters
    contrast_radius_range = cfg.get("contrast_radius_range", [7, 15])
    contrast_radius = rng.randint(int(contrast_radius_range[0]), int(contrast_radius_range[1]) + 1)
    
    contrast_gain_range = cfg.get("contrast_gain_range", [0.8, 1.6])
    contrast_gain = rng.uniform(contrast_gain_range[0], contrast_gain_range[1])
    
    # Sample bloom parameters
    bloom_strength_range = cfg.get("bloom_strength_range", [0.0, 0.3])
    bloom_strength = rng.uniform(bloom_strength_range[0], bloom_strength_range[1])
    
    bloom_th = cfg.get("bloom_th", 0.75)
    bloom_radius = cfg.get("bloom_radius", 9)
    
    return {
        "beta": beta,
        "airlight": airlight,
        "depth_mode": depth_mode,
        "grad_angle": grad_angle,
        "contrast_radius": contrast_radius,
        "contrast_gain": contrast_gain,
        "bloom_strength": bloom_strength,
        "bloom_th": bloom_th,
        "bloom_radius": bloom_radius,
    }


def _apply_fog_rgb(rgb_np_hwc_float01: np.ndarray, cfg: Dict, rng: random.Random) -> np.ndarray:
    """
    Apply fog augmentation to RGB image.
    
    Args:
        rgb_np_hwc_float01: RGB image [H, W, 3] in float32 [0, 1]
        cfg: Fog config dictionary
        rng: Random number generator
        
    Returns:
        fogged: Fogged RGB image [H, W, 3] in float32 [0, 1]
    """
    # Sample fog parameters
    params = _sample_fog_params(cfg, rng)
    
    # Build depth map
    depth = _build_depth_map(
        rgb_np_hwc_float01,
        mode=params["depth_mode"],
        flat_depth=0.5,
        grad_angle_deg=params["grad_angle"],
        contrast_radius=params["contrast_radius"],
        contrast_gain=params["contrast_gain"],
        rng=rng,
    )
    
    # Apply Koschmieder fog
    fogged = _apply_koschmieder(
        rgb_np_hwc_float01,
        depth,
        beta=params["beta"],
        airlight=params["airlight"],
    )
    
    # Add bloom effect
    fogged = _add_bloom(
        fogged,
        strength=params["bloom_strength"],
        threshold=params["bloom_th"],
        radius=params["bloom_radius"],
    )
    
    return fogged


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
        cfg: Optional[Dict] = None,
    ):
        """
        Args:
            root_dir: Root directory containing train/val folders
            split: "train" or "val"
            img_size: Target image size (must be multiple of 8)
            augment: Whether to apply data augmentation (for training)
            dct_gray: Whether to compute DCT bands from grayscale
            class_to_idx: Optional class name to index mapping
            cfg: Optional full config dictionary (for fog augmentation)
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
        
        # Extract fog config with robust fallbacks
        self.fog_cfg = {}
        if cfg is not None:
            fog_section = cfg.get("augment", {}).get("fog", {})
            if fog_section:
                self.fog_cfg = fog_section
        
        # Extract DCT config with robust fallbacks
        self.dct_cfg = {}
        if cfg is not None:
            dct_section = cfg.get("dct", {})
            if dct_section:
                self.dct_cfg = dct_section
        
        # Determine if fog duplication is enabled
        # When enabled, creates 4 variants: original, fog, flip, fog+flip
        fog_enabled = self.fog_cfg.get("enabled", False)
        fog_duplicate = self.fog_cfg.get("duplicate", False)
        # Apply to both train and val if enabled
        self.duplication_enabled = fog_enabled and fog_duplicate
        
        # Store original length before duplication
        self._orig_len = None  # Will be set after samples are collected
        
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
        
        # Store original length (before duplication)
        self._orig_len = len(self.samples)
        
        # RGB transforms - split into geometric transforms and ToTensor/Normalize
        # This allows us to apply fog augmentation before ToTensor
        if self.augment:
            # Geometric transforms (before fog)
            # When duplication is enabled, we explicitly control horizontal flip,
            # so remove RandomHorizontalFlip from the transform pipeline
            if self.duplication_enabled:
                self.rgb_geom_transform = transforms.Compose([
                    transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
                    # Note: Horizontal flip is handled explicitly in __getitem__
                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                ])
            else:
                self.rgb_geom_transform = transforms.Compose([
                    transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                ])
        else:
            self.rgb_geom_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(img_size),
            ])
        
        # ToTensor and Normalize (after fog)
        self.rgb_to_tensor = transforms.ToTensor()
        
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
        
        # Debug dump tracking (set of indices already dumped)
        self._debug_dumped_indices = set()
        
        # Log fog duplication status
        if self.duplication_enabled:
            apply_to = self.fog_cfg.get("apply_to", "both")
            print(f"Fog Duplication active ({self.split}): length x4 (original, fog, flip, fog+flip), apply_to={apply_to}")
    
    def __len__(self) -> int:
        if self.duplication_enabled:
            return 4 * self._orig_len  # 4 variants: original, fog, flip, fog+flip
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Optional[torch.Tensor], int]:
        """
        Returns:
            Tuple of (rgb, dct, label):
                - rgb: [3, H, W] normalized RGB tensor
                - dct: [3, H, W] DCT tensor (or None if mode doesn't need DCT)
                - label: class index (int)
        """
        # Handle duplication: map idx to source index and variant type
        # Variants: 0=original, 1=fog, 2=flip, 3=fog+flip
        if self.duplication_enabled:
            src_idx = idx % self._orig_len
            variant = idx // self._orig_len  # 0, 1, 2, or 3
            apply_flip = variant in [2, 3]  # Variants 2 and 3 are flipped
            apply_fog = variant in [1, 3]   # Variants 1 and 3 have fog
        else:
            src_idx = idx
            variant = 0
            apply_flip = False
            apply_fog = False
            fog_enabled = self.fog_cfg.get("enabled", False)
            # If fog enabled but duplication disabled, apply fog randomly (old behavior)
            if fog_enabled and self.split == "train":
                apply_fog = True
        
        img_path, label = self.samples[src_idx]
        
        # Load RGB image
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise ValueError(f"Failed to load image {img_path}: {e}")
        
        # Apply geometric transforms (before fog and explicit flip)
        img_geom = self.rgb_geom_transform(img)
        
        # Apply explicit horizontal flip if needed (when duplication enabled)
        if self.duplication_enabled and apply_flip:
            img_geom = img_geom.transpose(Image.FLIP_LEFT_RIGHT)
        
        # Apply fog augmentation if needed (apply_fog already determined above)
        
        if apply_fog:
            # Convert PIL to numpy float32 [0, 1] HWC
            rgb_np = np.asarray(img_geom, dtype=np.float32) / 255.0
            
            # Create deterministic RNG per source index
            # Note: To make fog change every epoch, XOR with epoch (requires setter)
            fog_seed = self.fog_cfg.get("seed", 1337)
            rng = random.Random(fog_seed ^ src_idx)
            
            # Apply fog augmentation
            rgb_np = _apply_fog_rgb(rgb_np, self.fog_cfg, rng)
            
            # Convert back to PIL for ToTensor
            rgb_np_uint8 = (np.clip(rgb_np, 0.0, 1.0) * 255).astype(np.uint8)
            img_final = Image.fromarray(rgb_np_uint8)
        else:
            img_final = img_geom
        
        # Debug dump (only once per source index and variant)
        debug_n = self.fog_cfg.get("debug_dump_first_n", 0)
        if debug_n > 0 and src_idx < debug_n:
            debug_dir = Path("runs/fog_debug") / self.split
            debug_dir.mkdir(parents=True, exist_ok=True)
            
            variant_names = ["orig", "fog", "flip", "fogflip"]
            if self.duplication_enabled:
                variant_name = variant_names[variant]
                dump_key = f"{src_idx}_{variant}"
                
                if dump_key not in self._debug_dumped_indices:
                    dump_path = debug_dir / f"idx_{src_idx:05d}_{variant_name}.png"
                    img_final.save(dump_path)
                    self._debug_dumped_indices.add(dump_key)
            else:
                # Old behavior: save original and fogged separately
                if not apply_fog and src_idx not in self._debug_dumped_indices:
                    orig_path = debug_dir / f"idx_{src_idx:05d}_orig.png"
                    img_geom.save(orig_path)
                    self._debug_dumped_indices.add(src_idx)
                
                if apply_fog:
                    fog_path = debug_dir / f"idx_{src_idx:05d}_fog.png"
                    img_final.save(fog_path)
        
        # Convert to tensor: [3, H, W], values in [0, 1]
        rgb = self.rgb_to_tensor(img_final)
        
        # Normalize RGB
        rgb_normalized = self.rgb_normalize(rgb)
        
        # Compute DCT if needed
        dct = None
        if self.needs_dct:
            # Determine if DCT should use processed image (fogged/flipped)
            apply_to = self.fog_cfg.get("apply_to", "both")
            
            # When duplication is enabled and apply_to="both", all variants should use processed RGB
            # When duplication disabled, use processed RGB only if fog is applied and apply_to="both"
            if self.duplication_enabled:
                # All variants use processed RGB when apply_to="both"
                use_processed_for_dct = (apply_to == "both")
            else:
                # Only use processed RGB if fog is applied and apply_to="both"
                use_processed_for_dct = (apply_to == "both") and apply_fog
            
            if use_processed_for_dct:
                # Use the same RGB (already processed) for DCT computation
                rgb_for_dct = rgb
            else:
                # Use original RGB (recompute from geometric transform only, no flip/fog)
                rgb_for_dct = self.rgb_to_tensor(img_geom)
            
            # Convert to grayscale for DCT (luma: 0.299R + 0.587G + 0.114B)
            gray = 0.299 * rgb_for_dct[0] + 0.587 * rgb_for_dct[1] + 0.114 * rgb_for_dct[2]  # [H, W]
            gray = gray.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
            
            # Read DCT parameters from config (with fallback defaults)
            c1_init_val = self.dct_cfg.get("c1_init", 2.0)
            c2_init_val = self.dct_cfg.get("c2_init", 4.0)
            dct_k_val = self.dct_cfg.get("k", 50.0)
            
            # Compute initial band maps with config values
            # The model will recompute these with learnable c1, c2 during forward for gradients
            c1_init = torch.tensor(c1_init_val, device=gray.device)
            c2_init = torch.tensor(c2_init_val, device=gray.device)
            D_device = self.D.to(device=gray.device, dtype=gray.dtype)
            band_low, band_mid, band_high = band_split_idct(
                gray, c1_init, c2_init, D_device, k=dct_k_val
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

