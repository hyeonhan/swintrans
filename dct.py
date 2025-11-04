"""
Pure PyTorch DCT utilities for 2D DCT/IDCT, block DCT, and energy maps.
No SciPy dependency - uses separable cosine basis matrices.
"""
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional


class DCT2D(nn.Module):
    """2D DCT-II transform using separable cosine matrices."""
    
    def __init__(self):
        super().__init__()
        self._cosine_cache: Dict[Tuple[int, int], torch.Tensor] = {}
    
    def _get_cosine_matrix(self, size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Compute or retrieve cached cosine basis matrix for DCT-II."""
        cache_key = (size, device.type)
        if cache_key not in self._cosine_cache:
            # DCT-II: C[u,v] = sqrt(2/N) * cos(pi * (2k+1) * u / (2N)) for u>0
            # For u=0: sqrt(1/N)
            n = size
            k = torch.arange(n, device=device, dtype=dtype).unsqueeze(0)  # [1, n]
            u = torch.arange(n, device=device, dtype=dtype).unsqueeze(1)  # [n, 1]
            
            # Scale factors
            scale = torch.ones(n, device=device, dtype=dtype)
            scale[0] = 1.0 / torch.sqrt(torch.tensor(n, dtype=dtype, device=device))
            scale[1:] = torch.sqrt(torch.tensor(2.0 / n, dtype=dtype, device=device))
            
            # Cosine matrix: C[u,k] = scale[u] * cos(pi * (2k+1) * u / (2n))
            cos_matrix = scale.unsqueeze(1) * torch.cos(
                torch.pi * (2 * k + 1) * u / (2 * n)
            )
            self._cosine_cache[cache_key] = cos_matrix
        
        return self._cosine_cache[cache_key]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply 2D DCT-II to input [B, C, H, W] or [B, 1, H, W].
        Returns coefficients [B, C, H, W].
        """
        if x.dim() == 3:
            x = x.unsqueeze(1)  # [B, H, W] -> [B, 1, H, W]
        
        B, C, H, W = x.shape
        device, dtype = x.device, x.dtype
        
        # Get cosine matrices
        cos_h = self._get_cosine_matrix(H, device, dtype)  # [H, H]
        cos_w = self._get_cosine_matrix(W, device, dtype)  # [W, W]
        
        # Separable 2D DCT: DCT2(x) = cos_h @ x @ cos_w^T
        # Reshape: [B, C, H, W] -> [B*C, H, W]
        x_flat = x.view(B * C, H, W)
        
        # Apply DCT along height: [B*C, H, W] -> [B*C, H, W]
        dct_h = torch.einsum('ij,bjk->bik', cos_h, x_flat)
        
        # Apply DCT along width: [B*C, H, W] -> [B*C, H, W]
        dct_2d = torch.einsum('bij,jk->bik', dct_h, cos_w.t())
        
        # Reshape back: [B*C, H, W] -> [B, C, H, W]
        return dct_2d.view(B, C, H, W)


class IDCT2D(nn.Module):
    """2D IDCT-III (inverse of DCT-II) using separable cosine matrices."""
    
    def __init__(self):
        super().__init__()
        self._cosine_cache: Dict[Tuple[int, int], torch.Tensor] = {}
    
    def _get_cosine_matrix(self, size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Compute or retrieve cached cosine basis matrix for IDCT-III."""
        cache_key = (size, device.type)
        if cache_key not in self._cosine_cache:
            # IDCT-III: C[k,u] = sqrt(2/N) * cos(pi * (2k+1) * u / (2N)) for u>0
            # For u=0: sqrt(1/N)
            n = size
            k = torch.arange(n, device=device, dtype=dtype).unsqueeze(0)  # [1, n]
            u = torch.arange(n, device=device, dtype=dtype).unsqueeze(1)  # [n, 1]
            
            # Scale factors (same as DCT-II for orthonormal transform)
            scale = torch.ones(n, device=device, dtype=dtype)
            scale[0] = 1.0 / torch.sqrt(torch.tensor(n, dtype=dtype, device=device))
            scale[1:] = torch.sqrt(torch.tensor(2.0 / n, dtype=dtype, device=device))
            
            # Cosine matrix: C[k,u] = scale[u] * cos(pi * (2k+1) * u / (2n))
            cos_matrix = scale.unsqueeze(1) * torch.cos(
                torch.pi * (2 * k + 1) * u / (2 * n)
            )
            self._cosine_cache[cache_key] = cos_matrix
        
        return self._cosine_cache[cache_key]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply 2D IDCT-III to input [B, C, H, W].
        Returns reconstructed image [B, C, H, W].
        """
        if x.dim() == 3:
            x = x.unsqueeze(1)
        
        B, C, H, W = x.shape
        device, dtype = x.device, x.dtype
        
        # Get cosine matrices
        cos_h = self._get_cosine_matrix(H, device, dtype)  # [H, H]
        cos_w = self._get_cosine_matrix(W, device, dtype)  # [W, W]
        
        # Separable 2D IDCT: IDCT2(x) = cos_h^T @ x @ cos_w
        x_flat = x.view(B * C, H, W)
        
        # Apply IDCT along height
        idct_h = torch.einsum('ij,bjk->bik', cos_h.t(), x_flat)
        
        # Apply IDCT along width
        idct_2d = torch.einsum('bij,jk->bik', idct_h, cos_w)
        
        return idct_2d.view(B, C, H, W)


def block_dct_8x8(gray: torch.Tensor) -> torch.Tensor:
    """
    Apply 8x8 block DCT to grayscale image.
    
    Args:
        gray: [B, 1, H, W] grayscale image
        
    Returns:
        coeff: [B, H//8, W//8, 8, 8] DCT coefficients per block
    """
    B, C, H, W = gray.shape
    assert C == 1, "Expected single channel grayscale"
    assert H % 8 == 0 and W % 8 == 0, f"Image size ({H}, {W}) must be divisible by 8"
    
    dct2d = DCT2D()
    
    # Reshape into blocks: [B, 1, H, W] -> [B, H//8, 8, W//8, 8]
    blocks = gray.view(B, 1, H // 8, 8, W // 8, 8)
    blocks = blocks.permute(0, 2, 4, 1, 3, 5).contiguous()  # [B, H//8, W//8, 1, 8, 8]
    
    # Flatten block dimension: [B, H//8, W//8, 1, 8, 8] -> [B*H//8*W//8, 1, 8, 8]
    n_blocks = B * (H // 8) * (W // 8)
    blocks_flat = blocks.view(n_blocks, 1, 8, 8)
    
    # Apply DCT to each block
    coeff_flat = dct2d(blocks_flat)  # [n_blocks, 1, 8, 8]
    
    # Reshape back: [n_blocks, 1, 8, 8] -> [B, H//8, W//8, 8, 8]
    coeff = coeff_flat.view(B, H // 8, W // 8, 8, 8)
    
    return coeff


def select_coeffs(coeff: torch.Tensor, selection: str = 'topk', k: int = 5) -> torch.Tensor:
    """
    Select coefficients from 8x8 DCT blocks.
    
    Args:
        coeff: [B, H//8, W//8, 8, 8] DCT coefficients
        selection: 'topk' (by magnitude) or 'lowfirst' (low-frequency order)
        k: number of coefficients to select (including DC)
        
    Returns:
        selected: [B, H//8, W//8, k] selected coefficients (DC always first)
    """
    B, h, w, block_h, block_w = coeff.shape
    assert block_h == 8 and block_w == 8
    
    # Flatten block: [B, h, w, 8, 8] -> [B, h, w, 64]
    coeff_flat = coeff.view(B, h, w, 64)
    
    if selection == 'lowfirst':
        # Low-frequency order: zigzag pattern
        # DC is [0,0], then [0,1], [1,0], [2,0], [1,1], [0,2], ...
        # Create zigzag order indices for 8x8 block
        zigzag_order = [
            0,  1,  8, 16,  9,  2,  3, 10,
            17, 24, 32, 25, 18, 11,  4,  5,
            12, 19, 26, 33, 40, 48, 41, 34,
            27, 20, 13,  6,  7, 14, 21, 28,
            35, 42, 49, 56, 57, 50, 43, 36,
            29, 22, 15, 23, 30, 37, 44, 51,
            58, 59, 52, 45, 38, 31, 39, 46,
            53, 60, 61, 54, 47, 55, 62, 63
        ]
        
        # Take first k indices from zigzag order
        selected_indices = zigzag_order[:k]
        indices_tensor = torch.tensor(selected_indices, dtype=torch.long, device=coeff.device)
        indices = indices_tensor.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(B, h, w, k)
        
        selected = torch.gather(coeff_flat, dim=3, index=indices)
        
    else:  # 'topk'
        # Select DC + top (k-1) by magnitude
        dc = coeff_flat[:, :, :, 0:1]  # [B, h, w, 1]
        ac = coeff_flat[:, :, :, 1:]   # [B, h, w, 63]
        
        # Get top-k-1 by magnitude
        ac_mag = torch.abs(ac)  # [B, h, w, 63]
        _, top_indices = torch.topk(ac_mag, k=k-1, dim=3)  # [B, h, w, k-1]
        
        # Gather top AC coefficients
        top_ac = torch.gather(ac, dim=3, index=top_indices)  # [B, h, w, k-1]
        
        # Concatenate DC + top AC
        selected = torch.cat([dc, top_ac], dim=3)  # [B, h, w, k]
    
    return selected


def band_energy_maps(coeff: torch.Tensor, bands_cfg: Dict) -> torch.Tensor:
    """
    Compute band energy maps from 8x8 DCT coefficients.
    
    Args:
        coeff: [B, H//8, W//8, 8, 8] DCT coefficients
        bands_cfg: dict with 'low', 'mid', 'high' band definitions
        
    Returns:
        energy_maps: [B, 3, H, W] low/mid/high energy maps (upsampled to original size)
    """
    B, h, w, block_h, block_w = coeff.shape
    assert block_h == 8 and block_w == 8
    
    H, W = h * 8, w * 8
    
    # Parse band indices
    low_bands = set(tuple(b) for b in bands_cfg.get('low', []))
    mid_bands = set(tuple(b) for b in bands_cfg.get('mid', []))
    
    # Compute squared magnitudes
    coeff_sq = coeff ** 2  # [B, h, w, 8, 8]
    
    # Initialize energy maps
    low_energy = torch.zeros(B, h, w, device=coeff.device, dtype=coeff.dtype)
    mid_energy = torch.zeros(B, h, w, device=coeff.device, dtype=coeff.dtype)
    high_energy = torch.zeros(B, h, w, device=coeff.device, dtype=coeff.dtype)
    
    # Aggregate energies per band
    for u in range(8):
        for v in range(8):
            if (u, v) in low_bands:
                low_energy += coeff_sq[:, :, :, u, v]
            elif (u, v) in mid_bands:
                mid_energy += coeff_sq[:, :, :, u, v]
            else:
                high_energy += coeff_sq[:, :, :, u, v]
    
    # Stack: [B, h, w] -> [B, 3, h, w]
    energy_maps = torch.stack([low_energy, mid_energy, high_energy], dim=1)
    
    # Upsample to original size using nearest neighbor
    energy_maps = torch.nn.functional.interpolate(
        energy_maps, size=(H, W), mode='nearest'
    )
    
    return energy_maps


# Module instances for reuse
_dct2d_instance = DCT2D()
_idct2d_instance = IDCT2D()


def dct2(x: torch.Tensor) -> torch.Tensor:
    """Convenience function for 2D DCT."""
    return _dct2d_instance(x)


def idct2(x: torch.Tensor) -> torch.Tensor:
    """Convenience function for 2D IDCT."""
    return _idct2d_instance(x)

