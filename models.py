"""
Multi-branch image classification model with RGB ResNet-50 and DCT-based Swin-Tiny branches.

Architecture:
    - RGB branch: ResNet-50 → global feature
    - Three DCT band branches (Low/Mid/High): Swin-Tiny → global features
    - Late fusion head (scalar/channel_gate/concat_mlp)
    - Classifier
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import timm
import math
from typing import Optional, Literal, List, Tuple

from dct_utils import create_dct_basis, band_split_idct


class RGBBranch(nn.Module):
    """RGB branch using ResNet-50 backbone."""
    
    def __init__(self, feature_dim: int = 512):
        super().__init__()
        # Load pretrained ResNet-50
        resnet = models.resnet50(weights="IMAGENET1K_V1")
        
        # Remove classifier and avgpool
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Feature projection
        self.norm = nn.LayerNorm(2048)
        self.proj = nn.Linear(2048, feature_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: RGB image [B, 3, H, W]
        Returns:
            features: [B, feature_dim]
        """
        x = self.backbone(x)  # [B, 2048, H', W']
        x = self.global_pool(x)  # [B, 2048, 1, 1]
        x = x.flatten(1)  # [B, 2048]
        x = self.norm(x)
        x = self.proj(x)  # [B, feature_dim]
        return x


class SwinBranch(nn.Module):
    """Swin-Tiny branch for DCT band maps."""
    
    def __init__(self, feature_dim: int = 512, shared_backbone: Optional[nn.Module] = None):
        super().__init__()
        if shared_backbone is None:
            # Create new Swin-Tiny model
            self.backbone = timm.create_model(
                "swin_tiny_patch4_window7_224",
                pretrained=True,
                num_classes=0,  # Remove classifier
                global_pool="",  # Return features before pooling
            )
            # Swin-Tiny outputs 768-dim features
            backbone_dim = 768
        else:
            # Use shared backbone
            self.backbone = shared_backbone
            backbone_dim = 768
        
        # Feature projection
        self.norm = nn.LayerNorm(backbone_dim)
        self.proj = nn.Linear(backbone_dim, feature_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Band map [B, 1, H, W] (will be replicated to 3 channels)
        Returns:
            features: [B, feature_dim]
        """
        # Replicate single channel to 3 channels for Swin
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)  # [B, 3, H, W]
        
        # Forward through Swin
        x = self.backbone(x)  # [B, 768] (or [B, N, 768] or [B, H, W, 768] if global_pool="")
        
        # Handle different output shapes
        if x.dim() == 4:
            # [B, H, W, C] or [B, C, H, W] -> global average pool
            if x.shape[-1] == 768:
                # Channels last: [B, H, W, C]
                x = x.mean(dim=(1, 2))  # [B, C]
            else:
                # Channels first: [B, C, H, W]
                x = x.mean(dim=(2, 3))  # [B, C]
        elif x.dim() == 3:
            # [B, N, 768] -> global average pool
            x = x.mean(dim=1)  # [B, 768]
        elif x.dim() == 2:
            # Already [B, 768]
            pass
        else:
            raise ValueError(f"Unexpected Swin output shape: {x.shape}")
        
        x = self.norm(x)
        x = self.proj(x)  # [B, feature_dim]
        return x


class ScalarFusionHead(nn.Module):
    """Scalar fusion: learnable softmax weights."""
    
    def __init__(self, num_branches: int = 4):
        super().__init__()
        # Learnable logits for softmax weights
        self.logits = nn.Parameter(torch.zeros(num_branches))
        
    def forward(self, features: List[torch.Tensor], branch_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            features: List of branch features, each [B, D]
            branch_mask: Optional [B, num_branches] mask (0 = dropped branch)
        Returns:
            fused: [B, D]
            alpha: [B, num_branches] fusion weights
        """
        B, D = features[0].shape
        num_branches = len(features)
        
        # Stack features: [B, num_branches, D]
        feat_stack = torch.stack(features, dim=1)  # [B, num_branches, D]
        
        # Compute alpha weights
        if branch_mask is not None:
            # Mask out dropped branches
            logits_masked = self.logits.unsqueeze(0).expand(B, -1)  # [B, num_branches]
            logits_masked = logits_masked.masked_fill(branch_mask == 0, float('-inf'))
            alpha = F.softmax(logits_masked, dim=1)  # [B, num_branches]
        else:
            alpha = F.softmax(self.logits, dim=0).unsqueeze(0).expand(B, -1)  # [B, num_branches]
        
        # Weighted sum
        alpha_expanded = alpha.unsqueeze(-1)  # [B, num_branches, 1]
        fused = (feat_stack * alpha_expanded).sum(dim=1)  # [B, D]
        
        return fused, alpha


class ChannelGateFusionHead(nn.Module):
    """Channel-wise gating fusion."""
    
    def __init__(self, feature_dim: int, num_branches: int = 4):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_branches = num_branches
        
        # MLP for each branch to produce gates
        self.gate_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim, feature_dim // 2),
                nn.ReLU(),
                nn.Linear(feature_dim // 2, feature_dim),
            ) for _ in range(num_branches)
        ])
        
    def forward(self, features: List[torch.Tensor], branch_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            features: List of branch features, each [B, D]
            branch_mask: Optional [B, num_branches] mask
        Returns:
            fused: [B, D]
            alpha: [B, num_branches, D] gate values (averaged for logging)
        """
        B, D = features[0].shape
        
        # Compute gates for each branch
        gates = []
        for i, feat in enumerate(features):
            gate = torch.sigmoid(self.gate_mlps[i](feat))  # [B, D]
            if branch_mask is not None:
                gate = gate * branch_mask[:, i:i+1]  # Mask out dropped branches
            gates.append(gate)
        
        # Element-wise fusion: sum(g_i ⊙ h_i)
        fused = sum(g * f for g, f in zip(gates, features))  # [B, D]
        
        # Return average gate values per branch for logging
        alpha = torch.stack([g.mean(dim=1) for g in gates], dim=1)  # [B, num_branches]
        
        return fused, alpha


class ConcatMLPFusionHead(nn.Module):
    """Concatenation + MLP fusion."""
    
    def __init__(self, feature_dim: int, num_branches: int = 4):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_branches = num_branches
        
        # MLP on concatenated features
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim * num_branches, feature_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim * 2, feature_dim),
        )
        
    def forward(self, features: List[torch.Tensor], branch_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            features: List of branch features, each [B, D]
            branch_mask: Optional [B, num_branches] mask
        Returns:
            fused: [B, D]
            alpha: [B, num_branches] dummy ones (for compatibility)
        """
        # Mask out dropped branches by zeroing
        if branch_mask is not None:
            features = [f * m.unsqueeze(-1) for f, m in zip(features, branch_mask.unbind(1))]
        
        # Concatenate
        concat = torch.cat(features, dim=1)  # [B, D * num_branches]
        
        # MLP
        fused = self.mlp(concat)  # [B, D]
        
        # Return dummy alpha (ones) for compatibility
        B = features[0].shape[0]
        alpha = torch.ones(B, self.num_branches, device=fused.device) / self.num_branches
        
        return fused, alpha


class MultiBranchClassifier(nn.Module):
    """
    Multi-branch classifier with RGB ResNet-50 and three DCT band Swin-Tiny branches.
    """
    
    def __init__(
        self,
        feature_dim: int = 512,
        num_classes: int = 2,
        swin_weight_sharing: bool = True,
        fusion: Literal["scalar", "channel_gate", "concat_mlp"] = "scalar",
        branch_dropout: float = 0.15,
        freeze_backbone_epochs: int = 0,
        dct_c1_init: float = 2.0,
        dct_c2_init: float = 4.0,
        dct_k: float = 50.0,
    ):
        """
        Args:
            feature_dim: Dimension of branch features
            num_classes: Number of classes
            swin_weight_sharing: Whether to share Swin backbone across band branches
            fusion: Fusion type ("scalar", "channel_gate", or "concat_mlp")
            branch_dropout: Dropout probability for entire branches during training
            freeze_backbone_epochs: Number of epochs to freeze backbone layers
            dct_c1_init: Initial value for c1 threshold
            dct_c2_init: Initial value for c2 threshold
            dct_k: Slope parameter for sigmoid masks
        """
        super().__init__()
        
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.fusion_type = fusion
        self.branch_dropout = branch_dropout
        self.freeze_backbone_epochs = freeze_backbone_epochs
        self.current_epoch = 0
        
        # DCT basis
        self.register_buffer("D", create_dct_basis(N=8))
        
        # Learnable DCT thresholds (with reparameterization for constraints)
        self.c1_raw = nn.Parameter(torch.tensor(dct_c1_init))
        self.c2_raw = nn.Parameter(torch.tensor(dct_c2_init))
        self.dct_k = dct_k
        
        # RGB branch
        self.rgb_branch = RGBBranch(feature_dim)
        
        # Swin branches
        if swin_weight_sharing:
            # Shared Swin backbone
            shared_swin = timm.create_model(
                "swin_tiny_patch4_window7_224",
                pretrained=True,
                num_classes=0,
                global_pool="",
            )
            self.swin_low = SwinBranch(feature_dim, shared_backbone=shared_swin)
            self.swin_mid = SwinBranch(feature_dim, shared_backbone=shared_swin)
            self.swin_high = SwinBranch(feature_dim, shared_backbone=shared_swin)
        else:
            # Separate Swin backbones
            self.swin_low = SwinBranch(feature_dim)
            self.swin_mid = SwinBranch(feature_dim)
            self.swin_high = SwinBranch(feature_dim)
        
        # Fusion head
        if fusion == "scalar":
            self.fusion_head = ScalarFusionHead(num_branches=4)
        elif fusion == "channel_gate":
            self.fusion_head = ChannelGateFusionHead(feature_dim, num_branches=4)
        elif fusion == "concat_mlp":
            self.fusion_head = ConcatMLPFusionHead(feature_dim, num_branches=4)
        else:
            raise ValueError(f"Unknown fusion type: {fusion}")
        
        # Classifier
        self.classifier = nn.Linear(feature_dim, num_classes)
        
    def get_dct_params(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get clamped c1, c2 parameters."""
        max_r = (8 - 1) * math.sqrt(2)  # ≈ 9.90
        # Clamp c1 first
        c1_clamped = torch.clamp(self.c1_raw, min=0.5, max=max_r - 0.1)
        # Clamp c2 ensuring it's >= c1 + 0.1
        # Convert max_r to tensor to match the 'min' argument's type (Tensor)
        max_r_tensor = torch.tensor(max_r, device=self.c2_raw.device, dtype=self.c2_raw.dtype)
        c2_clamped = torch.clamp(self.c2_raw, min=c1_clamped + 0.1, max=max_r_tensor)
        return c1_clamped, c2_clamped
    
    def set_epoch(self, epoch: int):
        """Set current epoch for freezing/unfreezing backbones."""
        self.current_epoch = epoch
        if epoch < self.freeze_backbone_epochs:
            # Freeze early layers of ResNet (first 6 modules = conv1, bn1, relu, maxpool, layer1)
            for param in list(self.rgb_branch.backbone.children())[:6]:
                for p in param.parameters():
                    p.requires_grad = False
            
            # Freeze Swin patch embedding and first layer
            if hasattr(self.swin_low.backbone, 'patch_embed'):
                for p in self.swin_low.backbone.patch_embed.parameters():
                    p.requires_grad = False
            if hasattr(self.swin_low.backbone, 'layers') and len(self.swin_low.backbone.layers) > 0:
                for p in self.swin_low.backbone.layers[0].parameters():
                    p.requires_grad = False
        else:
            # Unfreeze all
            for param in self.parameters():
                param.requires_grad = True
    
    def forward(
        self,
        rgb: torch.Tensor,
        band_low: torch.Tensor,
        band_mid: torch.Tensor,
        band_high: torch.Tensor,
        gray: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            rgb: RGB image [B, 3, H, W]
            band_low: Low-frequency band map [B, 1, H, W] (used if gray is None)
            band_mid: Mid-frequency band map [B, 1, H, W]
            band_high: High-frequency band map [B, 1, H, W]
            gray: Optional grayscale [B, 1, H, W] to recompute bands with learnable c1, c2
            
        Returns:
            logits: [B, num_classes]
            alpha: [B, 4] fusion weights for logging
        """
        B = rgb.shape[0]
        
        # Recompute band maps with learnable c1, c2 if gray is provided
        if gray is not None:
            # Ensure gray has batch dimension: [B, 1, H, W]
            if gray.dim() == 3:
                gray = gray.unsqueeze(0)  # [1, H, W] -> [1, 1, H, W]
            c1, c2 = self.get_dct_params()
            band_low, band_mid, band_high = band_split_idct(
                gray, c1, c2, self.D, k=self.dct_k
            )
            # Output is [B, 1, H, W], which is correct
        
        # Branch dropout mask (training only)
        branch_mask = None
        if self.training and self.branch_dropout > 0:
            branch_mask = (torch.rand(B, 4, device=rgb.device) > self.branch_dropout).float()
            # Ensure at least one branch is active
            if branch_mask.sum(dim=1).min() == 0:
                branch_mask = torch.ones_like(branch_mask)
        
        # Forward through branches
        feat_rgb = self.rgb_branch(rgb)  # [B, D]
        feat_low = self.swin_low(band_low)  # [B, D]
        feat_mid = self.swin_mid(band_mid)  # [B, D]
        feat_high = self.swin_high(band_high)  # [B, D]
        
        # Apply branch dropout
        if branch_mask is not None:
            feat_rgb = feat_rgb * branch_mask[:, 0:1]
            feat_low = feat_low * branch_mask[:, 1:2]
            feat_mid = feat_mid * branch_mask[:, 2:3]
            feat_high = feat_high * branch_mask[:, 3:4]
        
        # Fusion
        features = [feat_rgb, feat_low, feat_mid, feat_high]
        fused, alpha = self.fusion_head(features, branch_mask)  # [B, D], [B, 4]
        
        # Classification
        logits = self.classifier(fused)  # [B, num_classes]
        
        return logits, alpha

