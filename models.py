"""
Multi-branch image classification models supporting 6 experiment modes:
1. RGB-ResNet only
2. RGB-Swin only
3. DCT-Swin only (DCT-IDCT-Swin transformer branch)
4. RGB-Swin + DCT-Swin
5. RGB-ResNet + DCT-Swin
6. Swin Transformer + FPN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import timm
import math
from typing import Optional, Literal, List, Tuple, Dict

from dct_utils import create_dct_basis, band_split_idct


def get_backbone_features(backbone_name: str, pretrained: bool = True):
    """
    Get backbone feature extractor.
    
    Args:
        backbone_name: Name of backbone (e.g., "resnet50", "swin_tiny_patch4_window7_224")
        pretrained: Whether to use pretrained weights
        
    Returns:
        backbone: Feature extractor
        feature_dim: Dimension of output features
    """
    if "resnet" in backbone_name.lower():
        if "18" in backbone_name:
            resnet = models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
            feature_dim = 512
        elif "50" in backbone_name:
            resnet = models.resnet50(weights="IMAGENET1K_V1" if pretrained else None)
            feature_dim = 2048
        else:
            resnet = models.resnet50(weights="IMAGENET1K_V1" if pretrained else None)
            feature_dim = 2048
        
        # Remove classifier and avgpool
        backbone = nn.Sequential(*list(resnet.children())[:-2])
        return backbone, feature_dim
    
    elif "swin" in backbone_name.lower():
        # Create Swin model without classifier
        backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classifier
            global_pool="",  # Return features before pooling
        )
        # Swin-Tiny outputs 768-dim features
        feature_dim = 768
        return backbone, feature_dim
    
    else:
        raise ValueError(f"Unknown backbone: {backbone_name}")


def extract_features_from_backbone(backbone, x: torch.Tensor, backbone_type: str) -> torch.Tensor:
    """
    Extract features from backbone.
    
    Args:
        backbone: Backbone model
        x: Input tensor [B, C, H, W]
        backbone_type: "resnet" or "swin"
        
    Returns:
        features: [B, D] feature vector
    """
    if backbone_type == "resnet":
        x = backbone(x)  # [B, D, H', W']
        x = F.adaptive_avg_pool2d(x, 1)  # [B, D, 1, 1]
        x = x.flatten(1)  # [B, D]
        return x
    
    elif backbone_type == "swin":
        # Swin with num_classes=0 and global_pool="" returns features
        x = backbone(x)  # Can be [B, N, C] or [B, C, H, W] or [B, H, W, C]
        
        # Handle different output shapes
        if x.dim() == 4:
            # [B, C, H, W] or [B, H, W, C]
            if x.shape[-1] == 768:  # Channels last
                x = x.mean(dim=(1, 2))  # [B, C]
            else:  # Channels first
                x = F.adaptive_avg_pool2d(x, 1).flatten(1)  # [B, C]
        elif x.dim() == 3:
            # [B, N, C] (sequence of patches)
            x = x.mean(dim=1)  # [B, C]
        elif x.dim() == 2:
            # Already [B, C]
            pass
        else:
            raise ValueError(f"Unexpected Swin output shape: {x.shape}")
        
        return x
    
    else:
        raise ValueError(f"Unknown backbone_type: {backbone_type}")


class RGBResNetOnly(nn.Module):
    """RGB-ResNet only model."""
    
    def __init__(
        self,
        rgb_backbone: str = "resnet50",
        num_classes: int = 2,
        pretrained: bool = True,
    ):
        super().__init__()
        self.backbone_name = rgb_backbone
        self.backbone, backbone_dim = get_backbone_features(rgb_backbone, pretrained)
        self.backbone_type = "resnet"
        
        # Classifier
        self.classifier = nn.Linear(backbone_dim, num_classes)
        
    def forward(self, images: torch.Tensor, dct: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            images: RGB images [B, 3, H, W]
            dct: Ignored (for compatibility)
        Returns:
            logits: [B, num_classes]
        """
        features = extract_features_from_backbone(self.backbone, images, self.backbone_type)
        logits = self.classifier(features)
        return logits


class RGBSwinOnly(nn.Module):
    """RGB-Swin only model."""
    
    def __init__(
        self,
        rgb_backbone: str = "swin_tiny_patch4_window7_224",
        num_classes: int = 2,
        pretrained: bool = True,
    ):
        super().__init__()
        self.backbone_name = rgb_backbone
        self.backbone, backbone_dim = get_backbone_features(rgb_backbone, pretrained)
        self.backbone_type = "swin"
        
        # Classifier
        self.classifier = nn.Linear(backbone_dim, num_classes)
        
    def forward(self, images: torch.Tensor, dct: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            images: RGB images [B, 3, H, W]
            dct: Ignored (for compatibility)
        Returns:
            logits: [B, num_classes]
        """
        features = extract_features_from_backbone(self.backbone, images, self.backbone_type)
        logits = self.classifier(features)
        return logits


class DCTSwinOnly(nn.Module):
    """DCT-Swin only model (uses only DCT-IDCT-Swin transformer branch)."""
    
    def __init__(
        self,
        dct_backbone: str = "swin_tiny_patch4_window7_224",
        num_classes: int = 2,
        pretrained: bool = True,
    ):
        super().__init__()
        self.backbone_name = dct_backbone
        self.backbone, backbone_dim = get_backbone_features(dct_backbone, pretrained)
        self.backbone_type = "swin"
        
        # Classifier
        self.classifier = nn.Linear(backbone_dim, num_classes)
        
    def forward(self, images: torch.Tensor, dct: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            images: RGB images [B, 3, H, W] (ignored, for compatibility)
            dct: DCT tensor [B, 3, H, W] (required for this model)
        Returns:
            logits: [B, num_classes]
        """
        assert dct is not None, "DCT input is required for DCTSwinOnly model"
        features = extract_features_from_backbone(self.backbone, dct, self.backbone_type)
        logits = self.classifier(features)
        return logits


class FusionSwinSwin(nn.Module):
    """RGB-Swin + DCT-Swin fusion model."""
    
    def __init__(
        self,
        rgb_backbone: str = "swin_tiny_patch4_window7_224",
        dct_backbone: str = "swin_tiny_patch4_window7_224",
        fusion_head_dim: int = 512,
        num_classes: int = 2,
        pretrained: bool = True,
        freeze_stages_for_dct: int = 0,
    ):
        super().__init__()
        self.freeze_stages_for_dct = freeze_stages_for_dct
        
        # RGB branch
        self.rgb_backbone, rgb_dim = get_backbone_features(rgb_backbone, pretrained)
        self.rgb_backbone_type = "swin"
        
        # DCT branch
        self.dct_backbone, dct_dim = get_backbone_features(dct_backbone, pretrained)
        self.dct_backbone_type = "swin"
        
        # Print feature sizes at startup
        print(f"RGB feat: {rgb_dim}, DCT feat: {dct_dim}, fusion in: {rgb_dim + dct_dim}")
        
        # Late fusion MLP head
        self.fusion_head = nn.Sequential(
            nn.Linear(rgb_dim + dct_dim, fusion_head_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(fusion_head_dim, num_classes),
        )
        
    def forward(self, images: torch.Tensor, dct: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            images: RGB images [B, 3, H, W]
            dct: DCT tensor [B, 3, H, W] (required for this model)
        Returns:
            logits: [B, num_classes]
        """
        assert dct is not None, "DCT input is required for FusionSwinSwin model"
        assert dct.shape == images.shape, f"DCT shape {dct.shape} must match RGB shape {images.shape}"
        
        # Extract features
        feat_rgb = extract_features_from_backbone(self.rgb_backbone, images, self.rgb_backbone_type)
        feat_dct = extract_features_from_backbone(self.dct_backbone, dct, self.dct_backbone_type)
        
        # Concatenate and fuse
        feat_concat = torch.cat([feat_rgb, feat_dct], dim=1)  # [B, rgb_dim + dct_dim]
        logits = self.fusion_head(feat_concat)
        
        return logits


class FusionResNetSwin(nn.Module):
    """RGB-ResNet + DCT-Swin fusion model (current default)."""
    
    def __init__(
        self,
        rgb_backbone: str = "resnet50",
        dct_backbone: str = "swin_tiny_patch4_window7_224",
        fusion_head_dim: int = 512,
        num_classes: int = 2,
        pretrained: bool = True,
        freeze_stages_for_dct: int = 0,
        dct_c1_init: float = 2.0,
        dct_c2_init: float = 4.0,
        dct_k: float = 50.0,
    ):
        super().__init__()
        self.freeze_stages_for_dct = freeze_stages_for_dct
        
        # RGB branch
        self.rgb_backbone, rgb_dim = get_backbone_features(rgb_backbone, pretrained)
        self.rgb_backbone_type = "resnet"
        
        # DCT branch
        self.dct_backbone, dct_dim = get_backbone_features(dct_backbone, pretrained)
        self.dct_backbone_type = "swin"
        
        # DCT basis for learnable band splitting
        self.register_buffer("D", create_dct_basis(N=8))
        self.c1_raw = nn.Parameter(torch.tensor(dct_c1_init))
        self.c2_raw = nn.Parameter(torch.tensor(dct_c2_init))
        self.dct_k = dct_k
        
        # Print feature sizes at startup
        print(f"RGB feat: {rgb_dim}, DCT feat: {dct_dim}, fusion in: {rgb_dim + dct_dim}")
        
        # Late fusion MLP head
        self.fusion_head = nn.Sequential(
            nn.Linear(rgb_dim + dct_dim, fusion_head_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(fusion_head_dim, num_classes),
        )
        
    def get_dct_params(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get clamped c1, c2 parameters."""
        max_r = (8 - 1) * math.sqrt(2)  # ≈ 9.90
        c1_clamped = torch.clamp(self.c1_raw, min=0.5, max=max_r - 0.1)
        max_r_tensor = torch.tensor(max_r, device=self.c2_raw.device, dtype=self.c2_raw.dtype)
        c2_clamped = torch.clamp(self.c2_raw, min=c1_clamped + 0.1, max=max_r_tensor)
        return c1_clamped, c2_clamped
        
    def forward(self, images: torch.Tensor, dct: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            images: RGB images [B, 3, H, W]
            dct: DCT tensor [B, 3, H, W] (required for this model)
        Returns:
            logits: [B, num_classes]
        """
        assert dct is not None, "DCT input is required for FusionResNetSwin model"
        assert dct.shape == images.shape, f"DCT shape {dct.shape} must match RGB shape {images.shape}"
        
        # Extract features
        feat_rgb = extract_features_from_backbone(self.rgb_backbone, images, self.rgb_backbone_type)
        feat_dct = extract_features_from_backbone(self.dct_backbone, dct, self.dct_backbone_type)
        
        # Concatenate and fuse
        feat_concat = torch.cat([feat_rgb, feat_dct], dim=1)  # [B, rgb_dim + dct_dim]
        logits = self.fusion_head(feat_concat)
        
        return logits


class SwinFPNClassifier(nn.Module):
    """Swin Transformer backbone with FPN neck and classification head."""
    
    def __init__(
        self,
        rgb_backbone: str = "swin_tiny_patch4_window7_224",
        num_classes: int = 2,
        pretrained: bool = True,
        fpn_dim: int = 256,
        out_indices: Tuple[int, ...] = (1, 2, 3),
    ):
        super().__init__()
        self.backbone_name = rgb_backbone
        self.fpn_dim = fpn_dim
        
        # Create Swin backbone with features_only=True
        self.backbone = timm.create_model(
            rgb_backbone,
            pretrained=pretrained,
            features_only=True,
            out_indices=out_indices,
        )
        
        # Get channel dimensions from feature_info
        channels = self.backbone.feature_info.channels()
        print(f"SwinFPN: backbone={rgb_backbone}, channels={channels}, fpn_dim={fpn_dim}")
        
        # Lateral convolutions: 1x1 convs to unify channel dimensions
        self.laterals = nn.ModuleList([
            nn.Conv2d(c, fpn_dim, kernel_size=1) for c in channels
        ])
        
        # Smoothing convolutions: 3x3 convs with GELU and optional dropout
        self.smooths = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(fpn_dim, fpn_dim, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Dropout(0.05),  # Small dropout as specified
            ) for _ in channels
        ])
        
        # Classification head: GAP + concat + Linear
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(3 * fpn_dim, num_classes),
        )
        
    def _build_pyramid(self, feats: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Build top-down FPN pyramid.
        
        Args:
            feats: List of feature maps [C3, C4, C5] from backbone
                   NOTE: TIMM Swin returns [B, H, W, C] (channels-last)
            
        Returns:
            List of pyramid features [P3, P4, P5] (channels-first)
        """
        # --- START: 수정된 부분 ---
        # Permute from [B, H, W, C] to [B, C, H, W]
        # timm 백본이 채널-라스트 형식으로 반환하므로 채널-퍼스트로 변경합니다.
        try:
            feats_ch_first = [f.permute(0, 3, 1, 2) for f in feats]
        except Exception as e:
            # permute가 실패할 경우를 대비한 디버깅 로그
            print(f"Error permuting features. Shapes: {[f.shape for f in feats]}")
            raise e
        # --- END: 수정된 부분 ---

        # feats[0] = C3, feats[1] = C4, feats[2] = C5
        # Start from top (P5)
        # 수정: feats_ch_first 사용
        p5 = self.laterals[2](feats_ch_first[2])  # C5 -> P5
        
        # Top-down path: P4 = upsample(P5) + lateral(C4)
        p5_up = F.interpolate(p5, scale_factor=2, mode="nearest")
        # 수정: feats_ch_first 사용
        p4 = p5_up + self.laterals[1](feats_ch_first[1])  # P4
        
        # Top-down path: P3 = upsample(P4) + lateral(C3)
        p4_up = F.interpolate(p4, scale_factor=2, mode="nearest")
        # 수정: feats_ch_first 사용
        p3 = p4_up + self.laterals[0](feats_ch_first[0])  # P3
        
        # Apply smoothing convolutions
        p3 = self.smooths[0](p3)
        p4 = self.smooths[1](p4)
        p5 = self.smooths[2](p5)
        
        return [p3, p4, p5]
        
    def forward(self, images: torch.Tensor, dct: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            images: RGB images [B, 3, H, W]
            dct: Ignored (for compatibility with existing interface)
        Returns:
            logits: [B, num_classes]
        """
        # Extract multi-scale features from backbone
        feats = self.backbone(images)  # List of [B, C_i, H_i, W_i]
        
        # Build FPN pyramid
        p3, p4, p5 = self._build_pyramid(feats)
        
        # Global average pooling on each pyramid level
        v3 = torch.mean(p3, dim=(2, 3))  # [B, fpn_dim]
        v4 = torch.mean(p4, dim=(2, 3))  # [B, fpn_dim]
        v5 = torch.mean(p5, dim=(2, 3))  # [B, fpn_dim]
        
        # Concatenate pooled features
        v = torch.cat([v3, v4, v5], dim=1)  # [B, 3*fpn_dim]
        
        # Classification
        logits = self.classifier(v)
        
        return logits


def create_model_from_cfg(mode: str, cfg: Dict, num_classes: int) -> nn.Module:
    """
    Create model based on experiment mode and config.
    
    Args:
        mode: Experiment mode ("rgb_resnet", "rgb_swin", "dct_swin", "rgbswin_dctswin", "rgbresnet_dctswin", "swin_fpn")
        cfg: Configuration dictionary
        num_classes: Number of classes
        
    Returns:
        model: PyTorch model
    """
    model_cfg = cfg.get("model", {})
    pretrained = model_cfg.get("pretrained", True)
    
    if mode == "rgb_resnet":
        rgb_backbone = model_cfg.get("rgb_backbone", "resnet50")
        model = RGBResNetOnly(
            rgb_backbone=rgb_backbone,
            num_classes=num_classes,
            pretrained=pretrained,
        )
    
    elif mode == "rgb_swin":
        rgb_backbone = model_cfg.get("rgb_backbone", "swin_tiny_patch4_window7_224")
        model = RGBSwinOnly(
            rgb_backbone=rgb_backbone,
            num_classes=num_classes,
            pretrained=pretrained,
        )
    
    elif mode == "dct_swin":
        dct_backbone = model_cfg.get("dct_backbone", "swin_tiny_patch4_window7_224")
        model = DCTSwinOnly(
            dct_backbone=dct_backbone,
            num_classes=num_classes,
            pretrained=pretrained,
        )
    
    elif mode == "rgbswin_dctswin":
        rgb_backbone = model_cfg.get("rgb_backbone", "swin_tiny_patch4_window7_224")
        dct_backbone = model_cfg.get("dct_backbone", "swin_tiny_patch4_window7_224")
        fusion_head_dim = model_cfg.get("fusion_head_dim", 512)
        freeze_stages = model_cfg.get("freeze_stages_for_dct", 0)
        model = FusionSwinSwin(
            rgb_backbone=rgb_backbone,
            dct_backbone=dct_backbone,
            fusion_head_dim=fusion_head_dim,
            num_classes=num_classes,
            pretrained=pretrained,
            freeze_stages_for_dct=freeze_stages,
        )
    
    elif mode == "rgbresnet_dctswin":
        rgb_backbone = model_cfg.get("rgb_backbone", "resnet50")
        dct_backbone = model_cfg.get("dct_backbone", "swin_tiny_patch4_window7_224")
        fusion_head_dim = model_cfg.get("fusion_head_dim", 512)
        freeze_stages = model_cfg.get("freeze_stages_for_dct", 0)
        dct_cfg = cfg.get("dct", {})
        model = FusionResNetSwin(
            rgb_backbone=rgb_backbone,
            dct_backbone=dct_backbone,
            fusion_head_dim=fusion_head_dim,
            num_classes=num_classes,
            pretrained=pretrained,
            freeze_stages_for_dct=freeze_stages,
            dct_c1_init=dct_cfg.get("c1_init", 2.0),
            dct_c2_init=dct_cfg.get("c2_init", 4.0),
            dct_k=dct_cfg.get("k", 50.0),
        )
    
    elif mode == "swin_fpn":
        rgb_backbone = model_cfg.get("rgb_backbone", "swin_tiny_patch4_window7_224")
        fpn_cfg = cfg.get("fpn", {})
        fpn_dim = fpn_cfg.get("fpn_dim", 256)
        out_indices = tuple(fpn_cfg.get("out_indices", [1, 2, 3]))
        model = SwinFPNClassifier(
            rgb_backbone=rgb_backbone,
            num_classes=num_classes,
            pretrained=pretrained,
            fpn_dim=fpn_dim,
            out_indices=out_indices,
        )
    
    else:
        raise ValueError(f"Unknown mode: {mode}. Must be one of: rgb_resnet, rgb_swin, dct_swin, rgbswin_dctswin, rgbresnet_dctswin, swin_fpn")
    
    return model
