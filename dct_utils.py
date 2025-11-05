"""
DCT utilities for block-based DCT/IDCT and frequency band splitting.

Implements orthonormal DCT-II basis, block-wise DCT/IDCT operations,
and learnable soft band masks for splitting frequency bands.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple


def create_dct_basis(N=8) -> torch.Tensor:
    """
    Create orthonormal DCT-II basis matrix D of size N×N.
    
    DCT-II formula: D[u,v] = sqrt(2/N) * cos((2v+1)uπ/(2N)) for u>0,
                     sqrt(1/N) for u=0.
    
    Args:
        N: Block size (default 8)
        
    Returns:
        D: Orthonormal DCT basis matrix [N, N]
    """
    D = torch.zeros(N, N)
    sqrt_2_N = math.sqrt(2.0 / N)
    sqrt_1_N = math.sqrt(1.0 / N)
    
    for u in range(N):
        for v in range(N):
            if u == 0:
                D[u, v] = sqrt_1_N
            else:
                D[u, v] = sqrt_2_N * math.cos((2 * v + 1) * u * math.pi / (2 * N))
    
    return D


def block_dct(x: torch.Tensor, D: torch.Tensor) -> torch.Tensor:
    """
    Apply block-wise DCT to input tensor.
    
    For each 8×8 block X: Y = D @ X @ D^T
    
    Args:
        x: Input tensor [B, 1, H, W] (single channel)
        D: DCT basis matrix [N, N]
        
    Returns:
        y: DCT coefficients [B, 1, H, W]
    """
    B, C, H, W = x.shape
    assert C == 1, "Input must be single channel"
    N = D.shape[0]
    assert H % N == 0 and W % N == 0, f"Height and width must be multiples of {N}"
    
    device = x.device
    dtype = x.dtype
    D = D.to(device=device, dtype=dtype)
    
    # Reshape to [B, 1, H//N, N, W//N, N]
    x_blocks = x.view(B, 1, H // N, N, W // N, N)
    
    # Transpose to [B, 1, H//N, W//N, N, N] for easier matrix multiplication
    x_blocks = x_blocks.permute(0, 1, 2, 4, 3, 5).contiguous()
    x_blocks = x_blocks.view(B, 1, H // N, W // N, N, N)
    
    # Apply DCT: Y = D @ X @ D^T for each block
    # x_blocks: [B, 1, H//N, W//N, N, N] - 6 dimensions
    # For each block [N, N] at position (j,k): Y = D @ block @ D^T
    # We need to contract over the spatial dimensions within each block
    # Using batch matrix multiplication: D @ block @ D^T
    # Reshape to use bmm: [B*H//N*W//N, N, N] for easier computation
    B_blocks = H // N * W // N
    x_flat = x_blocks.view(B * B_blocks, N, N)  # [B*blocks, N, N]
    
    # D @ X: [B*blocks, N, N] @ [N, N] -> [B*blocks, N, N]
    # We need to do: for each block, compute D @ block
    # Using einsum: 'bij,op->bio' where i=N, j=N, o=N, p=N
    # Actually, simpler: use matmul with broadcasting
    y_intermediate = torch.matmul(D.unsqueeze(0), x_flat)  # [1, N, N] @ [B*blocks, N, N] -> [B*blocks, N, N]
    
    # Now (D @ X) @ D^T: [B*blocks, N, N] @ [N, N]^T
    y_flat = torch.matmul(y_intermediate, D.unsqueeze(0).transpose(-2, -1))  # [B*blocks, N, N] @ [1, N, N] -> [B*blocks, N, N]
    
    # Reshape back
    y_blocks = y_flat.view(B, 1, H // N, W // N, N, N)
    
    # Reshape back to [B, 1, H, W]
    # Permute: [B, 1, H//N, W//N, N, N] -> [B, 1, H//N, N, W//N, N]
    y = y_blocks.permute(0, 1, 2, 4, 3, 5).contiguous()
    y = y.view(B, 1, H, W)
    
    return y


def block_idct(y: torch.Tensor, D: torch.Tensor) -> torch.Tensor:
    """
    Apply block-wise IDCT to DCT coefficients.
    
    For each 8×8 block Y: X = D^T @ Y @ D
    
    Args:
        y: DCT coefficients [B, 1, H, W]
        D: DCT basis matrix [N, N]
        
    Returns:
        x: Reconstructed image [B, 1, H, W]
    """
    B, C, H, W = y.shape
    assert C == 1, "Input must be single channel"
    N = D.shape[0]
    assert H % N == 0 and W % N == 0, f"Height and width must be multiples of {N}"
    
    device = y.device
    dtype = y.dtype
    D = D.to(device=device, dtype=dtype)
    
    # Reshape to blocks
    y_blocks = y.view(B, 1, H // N, N, W // N, N)
    y_blocks = y_blocks.permute(0, 1, 2, 4, 3, 5).contiguous()
    y_blocks = y_blocks.view(B, 1, H // N, W // N, N, N)
    
    # Apply IDCT: X = D^T @ Y @ D
    # y_blocks: [B, 1, H//N, W//N, N, N] - 6 dimensions (frequency domain)
    # For each block: X = D^T @ block @ D
    B_blocks = H // N * W // N
    y_flat = y_blocks.view(B * B_blocks, N, N)  # [B*blocks, N, N]
    
    # D^T @ Y: [N, N]^T @ [B*blocks, N, N]
    x_intermediate = torch.matmul(D.unsqueeze(0).transpose(-2, -1), y_flat)  # [1, N, N]^T @ [B*blocks, N, N] -> [B*blocks, N, N]
    
    # (D^T @ Y) @ D: [B*blocks, N, N] @ [N, N]
    x_flat = torch.matmul(x_intermediate, D.unsqueeze(0))  # [B*blocks, N, N] @ [1, N, N] -> [B*blocks, N, N]
    
    # Reshape back
    x_blocks = x_flat.view(B, 1, H // N, W // N, N, N)
    
    # Reshape back
    x = x_blocks.permute(0, 1, 2, 4, 3, 5).contiguous()
    x = x.view(B, 1, H, W)
    
    return x


def compute_radial_frequency_grid(N=8) -> torch.Tensor:
    """
    Compute radial frequency grid r[u,v] = sqrt(u^2 + v^2) for N×N block.
    
    Args:
        N: Block size
        
    Returns:
        r: Radial frequency grid [N, N]
    """
    u = torch.arange(N, dtype=torch.float32).view(N, 1)
    v = torch.arange(N, dtype=torch.float32).view(1, N)
    r = torch.sqrt(u**2 + v**2)
    return r


def band_split_idct(
    gray: torch.Tensor,
    c1: torch.Tensor,
    c2: torch.Tensor,
    D: torch.Tensor,
    k: float = 50.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Split grayscale image into three frequency bands via DCT and IDCT.
    
    Pipeline:
        1. Convert RGB to grayscale (if needed)
        2. Apply block-DCT
        3. Create soft band masks (low/mid/high)
        4. Apply masks to DCT coefficients
        5. Apply block-IDCT to get band maps
    
    Args:
        gray: Grayscale image [B, 1, H, W]
        c1: Learnable threshold for low/high boundary (scalar tensor)
        c2: Learnable threshold for mid/high boundary (scalar tensor)
        D: DCT basis matrix [N, N]
        k: Slope parameter for sigmoid masks (default 50.0)
        
    Returns:
        low_img: Low-frequency band map [B, 1, H, W]
        mid_img: Mid-frequency band map [B, 1, H, W]
        high_img: High-frequency band map [B, 1, H, W]
    """
    device = gray.device
    dtype = gray.dtype
    N = D.shape[0]
    
    # Ensure c1 < c2 and clamp to valid range
    # Clamp: 0.5 <= c1 <= c2 - 0.1, c2 <= max_r (≈10.63 for 8×8)
    max_r = math.sqrt(2 * (N - 1)**2)  # sqrt(2 * 7^2) ≈ 9.90
    c1_clamped = torch.clamp(c1, min=0.5, max=max_r - 0.1)
    
    # max_r를 c2와 같은 device/dtype의 Tensor로 변환
    max_r_tensor = torch.tensor(max_r, device=c2.device, dtype=c2.dtype)
    c2_clamped = torch.clamp(c2, min=c1_clamped + 0.1, max=max_r_tensor)
    
    # Compute radial frequency grid
    r = compute_radial_frequency_grid(N).to(device=device, dtype=dtype)
    r = r.unsqueeze(0).unsqueeze(0)  # [1, 1, N, N]
    
    # Create soft band masks
    # low = sigmoid(k*(c1 - r))
    low_mask = torch.sigmoid(k * (c1_clamped - r))
    
    # high = sigmoid(k*(r - c2))
    high_mask = torch.sigmoid(k * (r - c2_clamped))
    
    # mid = sigmoid(k*(c2 - r)) - sigmoid(k*(c1 - r))
    mid_mask = torch.sigmoid(k * (c2_clamped - r)) - torch.sigmoid(k * (c1_clamped - r))
    
    # Ensure masks sum to ~1.0 (numerical stability)
    total_mask = low_mask + mid_mask + high_mask
    total_mask = torch.clamp(total_mask, min=1e-6)
    low_mask = low_mask / total_mask
    mid_mask = mid_mask / total_mask
    high_mask = high_mask / total_mask
    
    # Apply block-DCT
    dct_coeffs = block_dct(gray, D)  # [B, 1, H, W]
    
    B, C, H, W = dct_coeffs.shape
    
    # Reshape DCT coefficients to blocks [B, 1, H//N, N, W//N, N]
    dct_blocks = dct_coeffs.view(B, C, H // N, N, W // N, N)
    
    # Apply masks to each block
    # Expand masks to match block structure: [1, 1, 1, N, 1, N]
    low_mask_expanded = low_mask.view(1, 1, 1, N, 1, N)
    mid_mask_expanded = mid_mask.view(1, 1, 1, N, 1, N)
    high_mask_expanded = high_mask.view(1, 1, 1, N, 1, N)
    
    dct_low = dct_blocks * low_mask_expanded
    dct_mid = dct_blocks * mid_mask_expanded
    dct_high = dct_blocks * high_mask_expanded
    
    # Reshape back to [B, 1, H, W]
    dct_low = dct_low.view(B, C, H, W)
    dct_mid = dct_mid.view(B, C, H, W)
    dct_high = dct_high.view(B, C, H, W)
    
    # Apply block-IDCT
    low_img = block_idct(dct_low, D)
    mid_img = block_idct(dct_mid, D)
    high_img = block_idct(dct_high, D)
    
    return low_img, mid_img, high_img

