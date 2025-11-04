"""Block-wise 8x8 DCT/IDCT utilities implemented with torch (no SciPy).

Provides batched block DCT/IDCT and normalization utilities for RGB images.
"""
from __future__ import annotations

from typing import Optional, Tuple

import math
import torch
import torch.nn.functional as F


def _dct_basis_8(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Return orthonormal 8x8 DCT-II basis matrix C.

    X_dct = C @ X @ C^T, with C orthonormal.
    """
    C = torch.zeros((8, 8), dtype=dtype, device=device)
    for u in range(8):
        for x in range(8):
            alpha = math.sqrt(1.0 / 8.0) if u == 0 else math.sqrt(2.0 / 8.0)
            C[u, x] = alpha * math.cos(((2 * x + 1) * u * math.pi) / 16.0)
    return C


def _unfold_blocks(x: torch.Tensor) -> torch.Tensor:
    """Unfold (B,C,H,W) into 8x8 blocks -> (B,C,Hb,Wb,8,8)."""
    B, C, H, W = x.shape
    assert H % 8 == 0 and W % 8 == 0, "Input height and width must be multiples of 8"
    Hb = H // 8
    Wb = W // 8
    # Use F.unfold to extract blocks, then reshape
    patches = F.unfold(x, kernel_size=8, stride=8)  # (B, C*64, Hb*Wb)
    patches = patches.transpose(1, 2)  # (B, Hb*Wb, C*64)
    patches = patches.view(B, Hb, Wb, C, 8, 8)
    patches = patches.permute(0, 3, 1, 2, 4, 5)  # (B,C,Hb,Wb,8,8)
    return patches


def _fold_blocks(blocks: torch.Tensor) -> torch.Tensor:
    """Fold blocks (B,C,Hb,Wb,8,8) back to (B,C,H,W)."""
    B, C, Hb, Wb, _, _ = blocks.shape
    H = Hb * 8
    W = Wb * 8
    blocks = blocks.permute(0, 2, 3, 1, 4, 5).contiguous().view(B, Hb * Wb, C * 64)
    blocks = blocks.transpose(1, 2)  # (B, C*64, Hb*Wb)
    x = F.fold(blocks, output_size=(H, W), kernel_size=8, stride=8)
    return x


def block_dct(x: torch.Tensor, standardize: bool = True, clip_sigma: Optional[float] = 3.0) -> torch.Tensor:
    """Compute 8x8 block-wise DCT for an RGB image batch.

    Args:
        x: Input tensor (B,3,H,W) with H,W multiples of 8.
        standardize: If True, standardize coefficients per (batch,channel) using mean/std.
        clip_sigma: If not None, clip to +/- clip_sigma * std after standardization.

    Returns:
        Tensor of shape (B, Hb, Wb, 3*64) where Hb=H/8, Wb=W/8.
    """
    assert x.dim() == 4 and x.size(1) == 3, "Input must be BCHW with 3 channels"
    device = x.device
    dtype = x.dtype
    C = _dct_basis_8(device, dtype)
    blocks = _unfold_blocks(x)  # (B,3,Hb,Wb,8,8)
    # DCT: Y[u,v] = sum_{x,y} C[u,x] * X[x,y] * C[v,y]
    Y = torch.einsum('ux,bchwxy->bchwuy', C, blocks)
    Y = torch.einsum('vy,bchwuy->bchwuv', C, Y)
    # Flatten 8x8 -> 64 per channel
    Bsz, Cc, Hb, Wb, _, _ = Y.shape
    Y = Y.reshape(Bsz, Cc, Hb, Wb, 64)
    # Move channels last and merge color
    Y = Y.permute(0, 2, 3, 1, 4).contiguous().view(Bsz, Hb, Wb, Cc * 64)

    if standardize:
        mean = Y.mean(dim=(1, 2), keepdim=True)
        std = Y.std(dim=(1, 2), keepdim=True) + 1e-6
        Y = (Y - mean) * (1.0 / std)
        if clip_sigma is not None and clip_sigma > 0:
            Y = torch.clamp(Y, -clip_sigma, clip_sigma)
    return Y


def block_idct(coeffs: torch.Tensor, orig_hw: Tuple[int, int]) -> torch.Tensor:
    """Inverse block-wise DCT from flattened 3*64 to (B,3,H,W).

    Args:
        coeffs: (B,Hb,Wb,3*64)
        orig_hw: (H, W) spatial size
    """
    B, Hb, Wb, C64 = coeffs.shape
    assert C64 == 3 * 64, "Expected 3*64 coefficients"
    C = _dct_basis_8(coeffs.device, coeffs.dtype)
    # reshape back to (B,3,Hb,Wb,8,8)
    Y = coeffs.view(B, Hb, Wb, 3, 64).permute(0, 3, 1, 2, 4)  # (B,3,Hb,Wb,64)
    Y = Y.view(B, 3, Hb, Wb, 8, 8)
    # X = C^T @ Y @ C  (inverse for orthonormal DCT)
    Ct = C.t()
    X = torch.einsum('xu,bchwuv->bchwxv', Ct, Y)
    X = torch.einsum('yv,bchwxv->bchwxy', Ct, X)
    X = _fold_blocks(X)  # (B,3,H,W)
    H, W = orig_hw
    return X[:, :, :H, :W]


class LearnableMask(torch.nn.Module):
    """Learnable per-color DCT coefficient mask of length 64 (sigmoid)."""

    def __init__(self, init: float = 0.0) -> None:
        super().__init__()
        self.mask_r = torch.nn.Parameter(torch.full((64,), init))
        self.mask_g = torch.nn.Parameter(torch.full((64,), init))
        self.mask_b = torch.nn.Parameter(torch.full((64,), init))

    def forward(self, coeffs: torch.Tensor) -> torch.Tensor:
        """Apply sigmoid(mask) elementwise to coeffs per color.

        Args:
            coeffs: (B, Hb, Wb, 3*64)
        Returns:
            Tensor (B, Hb, Wb, 3*64)
        """
        B, Hb, Wb, C64 = coeffs.shape
        assert C64 == 3 * 64
        r = coeffs[..., 0:64] * torch.sigmoid(self.mask_r)
        g = coeffs[..., 64:128] * torch.sigmoid(self.mask_g)
        b = coeffs[..., 128:192] * torch.sigmoid(self.mask_b)
        return torch.cat([r, g, b], dim=-1)



