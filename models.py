"""
Model factory and modules for four model modes:
- rgb_only: Baseline Swin Transformer
- dct_gate: DCT-based Channel Attention inside Swin
- cross_attn: RGB↔DCT token cross-attention at mid-stage
- late_fusion: Two-tower late fusion with lightweight DCT branch
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import Dict, Optional

from dct import block_dct_8x8, select_coeffs, band_energy_maps


def _ensure_channel_first(features: torch.Tensor) -> torch.Tensor:
    """
    Ensure features are in channel-first format [B, C, H, W].
    If input is channel-last [B, H, W, C], permute to channel-first.
    """
    if len(features.shape) == 4:
        B, dim1, dim2, dim3 = features.shape
        # In channel-last format, dim1 and dim2 are both spatial (H, W) and should be similar
        # In channel-first format, dim1 is channels (C) which differs from dim2 (H)
        is_channel_last = abs(dim1 - dim2) <= max(dim1, dim2) // 4  # H and W are similar
        if is_channel_last:
            features = features.permute(0, 3, 1, 2)
    return features


class DCTChannelAttention(nn.Module):
    """DCT-based Channel Attention module."""
    
    def __init__(self, channels: int, P: int = 5, selection: str = 'topk', reduction: int = 4):
        """
        Args:
            channels: number of input channels
            P: number of DCT coefficients to select per channel
            selection: 'topk' or 'lowfirst'
            reduction: reduction ratio for MLP
        """
        super().__init__()
        self.channels = channels
        self.P = P
        self.selection = selection
        
        # MLP for channel attention
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )
        
        # DCT module (will be instantiated per forward)
        from dct import DCT2D
        self.dct2d = DCT2D()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W] feature map
            
        Returns:
            y: [B, C, H, W] attended feature map
        """
        B, C, H, W = x.shape
        
        # Per-channel DCT
        # Split channels and apply DCT to each
        channel_coeffs = []
        for c in range(C):
            channel_feat = x[:, c:c+1, :, :]  # [B, 1, H, W]
            dct_coeff = self.dct2d(channel_feat)  # [B, 1, H, W]
            
            # Select top P coefficients by magnitude (or low-first)
            # Flatten spatial: [B, 1, H, W] -> [B, H*W]
            coeff_flat = dct_coeff.view(B, H * W)
            
            if self.selection == 'lowfirst':
                # Take first P coefficients (including DC at 0)
                selected = coeff_flat[:, :self.P]  # [B, P]
            else:  # topk
                # Take DC + top (P-1) by magnitude
                dc = coeff_flat[:, 0:1]  # [B, 1]
                ac = coeff_flat[:, 1:]   # [B, H*W-1]
                ac_mag = torch.abs(ac)
                _, top_indices = torch.topk(ac_mag, k=min(self.P-1, ac.shape[1]), dim=1)
                top_ac = torch.gather(ac, dim=1, index=top_indices)
                selected = torch.cat([dc, top_ac], dim=1)  # [B, P]
            
            # Aggregate: sum of magnitudes
            channel_energy = torch.sum(torch.abs(selected), dim=1, keepdim=True)  # [B, 1]
            channel_coeffs.append(channel_energy)
        
        # Stack: [B, C]
        channel_features = torch.cat(channel_coeffs, dim=1)  # [B, C]
        
        # MLP to get attention weights
        attn_weights = self.mlp(channel_features)  # [B, C]
        attn_weights = attn_weights.view(B, C, 1, 1)  # [B, C, 1, 1]
        
        # Apply attention
        y = x * attn_weights
        
        return y


class RGBOnlySwin(nn.Module):
    """Baseline Swin Transformer for RGB-only classification."""
    
    def __init__(self, backbone: str = 'swin_tiny_patch4_window7_224', pretrained: bool = True, dropout: float = 0.0):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0, global_pool='')
        
        # Get feature dim
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            features = self.backbone(dummy)
            if isinstance(features, (list, tuple)):
                features = features[-1]
            features = _ensure_channel_first(features)
            feature_dim = features.shape[1]  # Now guaranteed to be [B, C, H, W]
        
        # Global average pooling + classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 2)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        if isinstance(features, (list, tuple)):
            features = features[-1]
        features = _ensure_channel_first(features)
        logits = self.classifier(features)
        return logits


class SwinWithDCTGate(nn.Module):
    """Swin Transformer with DCT-based Channel Attention after each stage."""
    
    def __init__(self, backbone: str = 'swin_tiny_patch4_window7_224', pretrained: bool = True, 
                 dropout: float = 0.0, P: int = 5, selection: str = 'topk', reduction: int = 4):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0, global_pool='')
        
        # Get stage outputs - we need to hook into intermediate features
        # For Swin, we'll wrap the forward to insert attention blocks
        self.stage_attentions = nn.ModuleList()
        
        # Determine channel dimensions (Swin tiny: 96, 192, 384, 768 at stages)
        # We'll apply attention to stage outputs
        stage_dims = [96, 192, 384, 768]  # Swin tiny default
        
        for dim in stage_dims:
            self.stage_attentions.append(DCTChannelAttention(dim, P, selection, reduction))
        
        # Get feature dim for classifier
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            features = self.backbone(dummy)
            if isinstance(features, (list, tuple)):
                feature_dim = features[-1].shape[1]
            else:
                feature_dim = features.shape[1]
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 2)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Get intermediate features from Swin stages
        # For timm Swin, we need to access stage outputs
        # Simplified: apply attention to final features only (more complex hooking needed for all stages)
        features = self.backbone(x)
        if isinstance(features, (list, tuple)):
            # Apply attention to last stage
            features = features[-1]
        features = _ensure_channel_first(features)
        features = self.stage_attentions[-1](features)
        
        logits = self.classifier(features)
        return logits


class CrossAttentionFusion(nn.Module):
    """Cross-attention fusion block for RGB↔DCT tokens."""
    
    def __init__(self, dim: int, num_heads: int = 4, drop: float = 0.0, bidirectional: bool = False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.bidirectional = bidirectional
        
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, dropout=drop, batch_first=True)
        
        if bidirectional:
            self.cross_attn_reverse = nn.MultiheadAttention(dim, num_heads, dropout=drop, batch_first=True)
    
    def forward(self, rgb_tokens: torch.Tensor, dct_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            rgb_tokens: [B, N_rgb, C] RGB patch tokens
            dct_tokens: [B, N_dct, C] DCT tokens
            
        Returns:
            fused_rgb: [B, N_rgb, C] RGB tokens after cross-attention
        """
        # RGB queries, DCT keys/values
        fused_rgb, _ = self.cross_attn(rgb_tokens, dct_tokens, dct_tokens)
        
        if self.bidirectional:
            # Also update DCT tokens with RGB (but we typically only use RGB stream)
            fused_dct, _ = self.cross_attn_reverse(dct_tokens, rgb_tokens, rgb_tokens)
            return fused_rgb, fused_dct
        
        return fused_rgb


class SwinCrossAttention(nn.Module):
    """Swin Transformer with RGB↔DCT cross-attention at mid-stage."""
    
    def __init__(self, backbone: str = 'swin_tiny_patch4_window7_224', pretrained: bool = True,
                 dropout: float = 0.0, P: int = 5, selection: str = 'topk',
                 num_heads: int = 4, drop: float = 0.0, bidirectional: bool = False):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0, global_pool='')
        self.P = P
        self.selection = selection
        
        # DCT token projection
        # At Stage 3 (H/16), we have tokens at resolution H/16 x W/16
        # DCT tokens: DC + (P-1) energies per patch -> project to C
        token_dim = 768  # Swin tiny Stage 3 dim
        self.dct_proj = nn.Linear(P, token_dim)
        
        # Cross-attention fusion
        self.cross_attn = CrossAttentionFusion(token_dim, num_heads, drop, bidirectional)
        
        # Get feature dim
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            features = self.backbone(dummy)
            if isinstance(features, (list, tuple)):
                features = features[-1]
            features = _ensure_channel_first(features)
            feature_dim = features.shape[1]  # Now guaranteed to be [B, C, H, W]
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 2)
        )
    
    def forward(self, x: torch.Tensor, dct_gray: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Build DCT tokens from grayscale
        if dct_gray is None:
            # Convert RGB to grayscale if needed
            dct_gray = torch.mean(x, dim=1, keepdim=True)  # [B, 1, H, W]
        
        B, _, H, W = dct_gray.shape
        
        # Ensure size is divisible by 8
        if H % 8 != 0 or W % 8 != 0:
            pad_h = (8 - H % 8) % 8
            pad_w = (8 - W % 8) % 8
            dct_gray = F.pad(dct_gray, (0, pad_w, 0, pad_h))
            H, W = dct_gray.shape[2], dct_gray.shape[3]
        
        # 8x8 block DCT
        coeff = block_dct_8x8(dct_gray)  # [B, H//8, W//8, 8, 8]
        
        # Select coefficients
        selected = select_coeffs(coeff, self.selection, self.P)  # [B, H//8, W//8, P]
        
        # Project to token dimension
        h_dct, w_dct = selected.shape[1], selected.shape[2]
        dct_tokens_flat = selected.view(B, h_dct * w_dct, self.P)  # [B, N_dct, P]
        dct_tokens = self.dct_proj(dct_tokens_flat)  # [B, N_dct, C]
        
        # Get RGB tokens from Swin at Stage 3 (H/16 resolution)
        # For simplicity, we'll apply cross-attention to final features
        # In practice, we'd hook into Stage 3 intermediate features
        features = self.backbone(x)
        if isinstance(features, (list, tuple)):
            rgb_features = features[-1]  # [B, C, H/32, W/32] typically
        else:
            rgb_features = features
        
        rgb_features = _ensure_channel_first(rgb_features)
        
        # Convert to tokens: [B, C, H, W] -> [B, H*W, C]
        B_rgb, C_rgb, H_rgb, W_rgb = rgb_features.shape
        rgb_tokens = rgb_features.view(B_rgb, C_rgb, H_rgb * W_rgb).permute(0, 2, 1)  # [B, N_rgb, C]
        
        # Resize DCT tokens to match RGB token count if needed
        if dct_tokens.shape[1] != rgb_tokens.shape[1]:
            # Interpolate DCT tokens spatially
            dct_tokens_2d = dct_tokens.view(B, h_dct, w_dct, C_rgb).permute(0, 3, 1, 2)  # [B, C, h, w]
            dct_tokens_2d = F.interpolate(dct_tokens_2d, size=(H_rgb, W_rgb), mode='bilinear', align_corners=False)
            dct_tokens = dct_tokens_2d.view(B, C_rgb, H_rgb * W_rgb).permute(0, 2, 1)  # [B, N_rgb, C]
        
        # Cross-attention
        fused_rgb = self.cross_attn(rgb_tokens, dct_tokens)  # [B, N_rgb, C]
        
        # Convert back to feature map: [B, N_rgb, C] -> [B, C, H, W]
        fused_features = fused_rgb.permute(0, 2, 1).view(B_rgb, C_rgb, H_rgb, W_rgb)
        
        logits = self.classifier(fused_features)
        return logits


class DCTTower(nn.Module):
    """Lightweight CNN tower for DCT band energy maps."""
    
    def __init__(self, mlp_hidden: int = 512):
        super().__init__()
        # Tiny CNN: 3-channel input (low/mid/high energy maps)
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        
        # MLP to match feature dimension
        self.mlp = nn.Sequential(
            nn.Linear(128, mlp_hidden),
            nn.GELU()
        )
    
    def forward(self, energy_maps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            energy_maps: [B, 3, H, W] band energy maps
            
        Returns:
            feat: [B, mlp_hidden] DCT features
        """
        feat = self.cnn(energy_maps)
        feat = self.mlp(feat)
        return feat


class TwoTowerLateFusion(nn.Module):
    """Two-tower late fusion: RGB Swin + DCT CNN."""
    
    def __init__(self, backbone: str = 'swin_tiny_patch4_window7_224', pretrained: bool = True,
                 dropout: float = 0.0, fusion_type: str = 'concat_mlp', mlp_hidden: int = 512):
        super().__init__()
        self.fusion_type = fusion_type
        
        # RGB tower
        self.rgb_backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0, global_pool='')
        
        # Get RGB feature dim
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            rgb_feat = self.rgb_backbone(dummy)
            if isinstance(rgb_feat, (list, tuple)):
                rgb_feat = rgb_feat[-1]
            rgb_feat = _ensure_channel_first(rgb_feat)
            rgb_dim = rgb_feat.shape[1]  # Now guaranteed to be [B, C, H, W]
        
        self.rgb_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        
        # DCT tower
        self.dct_tower = DCTTower(mlp_hidden)
        dct_dim = mlp_hidden
        
        # Fusion
        if fusion_type == 'concat_mlp':
            self.fusion = nn.Sequential(
                nn.Linear(rgb_dim + dct_dim, mlp_hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(mlp_hidden, 2)
            )
        else:  # gated_sum
            self.alpha = nn.Parameter(torch.tensor(0.5))
            self.fusion = nn.Sequential(
                nn.Linear(rgb_dim, mlp_hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(mlp_hidden, 2)
            )
            self.fusion_dct = nn.Sequential(
                nn.Linear(dct_dim, mlp_hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(mlp_hidden, 2)
            )
    
    def forward(self, x: torch.Tensor, dct_gray: Optional[torch.Tensor] = None, bands_cfg: Optional[Dict] = None) -> torch.Tensor:
        # RGB tower
        rgb_feat = self.rgb_backbone(x)
        if isinstance(rgb_feat, (list, tuple)):
            rgb_feat = rgb_feat[-1]
        rgb_feat = _ensure_channel_first(rgb_feat)
        rgb_vec = self.rgb_pool(rgb_feat)  # [B, rgb_dim]
        
        # DCT tower
        if dct_gray is None:
            dct_gray = torch.mean(x, dim=1, keepdim=True)
        
        B, _, H, W = dct_gray.shape
        if H % 8 != 0 or W % 8 != 0:
            pad_h = (8 - H % 8) % 8
            pad_w = (8 - W % 8) % 8
            dct_gray = F.pad(dct_gray, (0, pad_w, 0, pad_h))
            H, W = dct_gray.shape[2], dct_gray.shape[3]
        
        coeff = block_dct_8x8(dct_gray)
        energy_maps = band_energy_maps(coeff, bands_cfg or {})
        dct_vec = self.dct_tower(energy_maps)  # [B, dct_dim]
        
        # Fusion
        if self.fusion_type == 'concat_mlp':
            fused = torch.cat([rgb_vec, dct_vec], dim=1)
            logits = self.fusion(fused)
        else:  # gated_sum
            alpha = torch.sigmoid(self.alpha)  # Ensure [0, 1]
            logits_rgb = self.fusion(rgb_vec)
            logits_dct = self.fusion_dct(dct_vec)
            logits = alpha * logits_rgb + (1 - alpha) * logits_dct
        
        return logits


def make_model(mode: str, cfg: Dict) -> nn.Module:
    """Factory function to create model based on mode."""
    model_cfg = cfg.get('model', {})
    dct_cfg = cfg.get('dct', {})
    fusion_cfg = cfg.get('fusion', {})
    
    backbone = model_cfg.get('backbone', 'swin_tiny_patch4_window7_224')
    pretrained = model_cfg.get('pretrained', True)
    dropout = model_cfg.get('dropout', 0.0)
    
    if mode == 'rgb_only':
        return RGBOnlySwin(backbone, pretrained, dropout)
    
    elif mode == 'dct_gate':
        P = dct_cfg.get('P', 5)
        selection = dct_cfg.get('selection', 'topk')
        reduction = fusion_cfg.get('reduction', 4)
        return SwinWithDCTGate(backbone, pretrained, dropout, P, selection, reduction)
    
    elif mode == 'cross_attn':
        P = dct_cfg.get('P', 5)
        selection = dct_cfg.get('selection', 'topk')
        num_heads = fusion_cfg.get('cross_attn', {}).get('num_heads', 4)
        drop = fusion_cfg.get('cross_attn', {}).get('drop', 0.0)
        bidirectional = fusion_cfg.get('cross_attn', {}).get('bidirectional', False)
        return SwinCrossAttention(backbone, pretrained, dropout, P, selection, num_heads, drop, bidirectional)
    
    elif mode == 'late_fusion':
        fusion_type = fusion_cfg.get('late', {}).get('type', 'concat_mlp')
        mlp_hidden = fusion_cfg.get('late', {}).get('mlp_hidden', 512)
        return TwoTowerLateFusion(backbone, pretrained, dropout, fusion_type, mlp_hidden)
    
    else:
        raise ValueError(f"Unknown mode: {mode}")

