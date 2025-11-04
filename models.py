"""Model definitions: RGB Swin-T backbone, DCT branches, fusion head, and system."""
from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

import dct as dct_utils


class RGBBackbone(nn.Module):
    """Swin-Tiny backbone producing pooled features and logits."""

    def __init__(self, num_classes: int, pretrained: bool = True) -> None:
        super().__init__()
        self.backbone = timm.create_model(
            "swin_tiny_patch4_window7_224", pretrained=pretrained, num_classes=0
        )
        feat_dim = self.backbone.num_features
        self.head = nn.Linear(feat_dim, num_classes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feats = self.backbone.forward_features(x)  # may be (B,C), (B,L,C), (B,C,H,W), or (B,H,W,C)

        # 1) Prefer timm's forward_head(pre_logits=True) if available
        if hasattr(self.backbone, "forward_head"):
            try:
                feats_head = self.backbone.forward_head(feats, pre_logits=True)  # (B, C)
                if isinstance(feats_head, torch.Tensor) and feats_head.dim() == 2:
                    feats = feats_head
            except Exception:
                pass  # fall back to manual pooling

        # 2) Manual robust pooling if still not (B, C)
        if feats.dim() == 4:
            feat_dim_exp = self.head.in_features  # expected channels, e.g., 768
            if feats.shape[1] == feat_dim_exp:            # (B,C,H,W)
                feats = feats.mean(dim=(2, 3))
            elif feats.shape[-1] == feat_dim_exp:        # (B,H,W,C)
                feats = feats.permute(0, 3, 1, 2).mean(dim=(2, 3))
            else:
                cax = next((ax for ax in (1, 2, 3) if feats.shape[ax] == feat_dim_exp), None)
                if cax is not None:
                    order = [0, cax] + [ax for ax in (1, 2, 3) if ax != cax]
                    feats = feats.permute(*order).mean(dim=(2, 3))
                else:
                    feats = feats.flatten(2).mean(dim=2)  # last resort
        elif feats.dim() == 3:
            feat_dim_exp = self.head.in_features
            if feats.shape[-1] == feat_dim_exp:   # (B,L,C)
                feats = feats.mean(dim=1)
            elif feats.shape[1] == feat_dim_exp:  # (B,C,L)
                feats = feats.mean(dim=2)
            else:
                feats = feats.mean(dim=1)         # default

        # 3) Safety check
        if feats.dim() != 2 or feats.size(1) != self.head.in_features:
            raise RuntimeError(
                f"RGBBackbone: pooled feature dim mismatch: got {tuple(feats.shape)}, "
                f"expected (*,{self.head.in_features}). "
                f"Please check timm version / backbone outputs."
            )

        logits = self.head(feats)
        return logits, feats


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: int = 1) -> None:
        super().__init__()
        self.depth = nn.Conv2d(in_ch, in_ch, k, s, p, groups=in_ch, bias=False)
        self.point = nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=True)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depth(x)
        x = self.point(x)
        x = self.bn(x)
        return self.act(x)


class DCTBranchFreqCNN(nn.Module):
    """DCT frequency branch: compress 64->k per color and run lightweight CNN."""

    def __init__(self, num_classes: int, k: int = 8, standardize: bool = True, clip_sigma: float = 3.0) -> None:
        super().__init__()
        self.k = int(k)
        self.standardize = bool(standardize)
        self.clip_sigma = float(clip_sigma)

        # Per-color 1x1 conv: 64->k then concatenate to 3k channels
        self.compress_r = nn.Conv2d(64, self.k, kernel_size=1)
        self.compress_g = nn.Conv2d(64, self.k, kernel_size=1)
        self.compress_b = nn.Conv2d(64, self.k, kernel_size=1)

        in_ch = 3 * self.k
        hidden = max(32, in_ch)
        self.cnn = nn.Sequential(
            DepthwiseSeparableConv(in_ch, hidden),
            DepthwiseSeparableConv(hidden, hidden),
            DepthwiseSeparableConv(hidden, hidden),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(hidden, num_classes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, C, H, W = x.shape
        coeff = dct_utils.block_dct(x, standardize=self.standardize, clip_sigma=self.clip_sigma)
        Hb, Wb = coeff.shape[1], coeff.shape[2]
        # reshape to (B, 3*64, Hb, Wb)
        coeff = coeff.view(B, Hb, Wb, 3, 64).permute(0, 3, 4, 1, 2).contiguous()
        coeff = coeff.view(B, 3, 64, Hb, Wb)
        r = self.compress_r(coeff[:, 0])
        g = self.compress_g(coeff[:, 1])
        b = self.compress_b(coeff[:, 2])
        x2d = torch.cat([r, g, b], dim=1)
        feats_map = self.cnn(x2d)
        feats = self.pool(feats_map).flatten(1)
        logits = self.head(feats)
        return logits, feats


class DCTBranchIDCTCNN(nn.Module):
    """IDCT branch: learnable 64 masks per color -> fixed IDCT -> CNN."""

    def __init__(self, num_classes: int, standardize: bool = True, clip_sigma: float = 3.0) -> None:
        super().__init__()
        self.standardize = bool(standardize)
        self.clip_sigma = float(clip_sigma)
        self.mask = dct_utils.LearnableMask(0.0)

        hidden = 48
        self.cnn = nn.Sequential(
            nn.Conv2d(3, hidden, 3, 1, 1), nn.BatchNorm2d(hidden), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, 1, 1), nn.BatchNorm2d(hidden), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, 1, 1), nn.BatchNorm2d(hidden), nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(hidden, num_classes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, C, H, W = x.shape
        coeff = dct_utils.block_dct(x, standardize=self.standardize, clip_sigma=self.clip_sigma)
        coeff = self.mask(coeff)
        recon = dct_utils.block_idct(coeff, (H, W))
        feats_map = self.cnn(recon)
        feats = self.pool(feats_map).flatten(1)
        logits = self.head(feats)
        return logits, feats


class FusionHead(nn.Module):
    """Late fusion strategies for RGB and DCT logits/features."""

    def __init__(self, mode: str, num_classes: int, alpha_init: float = 0.3, dynamic_hidden: int = 256, alpha_min: Optional[float] = None, alpha_max: Optional[float] = None) -> None:
        super().__init__()
        self.mode = mode
        self.num_classes = num_classes
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.hidden = dynamic_hidden
        self.alpha_init = alpha_init

        if mode == "late_fusion_alpha_scalar":
            init = torch.tensor([alpha_init]).float().logit()
            self.alpha_param = nn.Parameter(init)
        elif mode == "late_fusion_alpha_classwise":
            init = torch.full((num_classes,), alpha_init).float().logit()
            self.alpha_param = nn.Parameter(init)
        elif mode == "late_fusion_dynamic":
            # Lazy MLP initialization - will be built on first forward
            self.mlp = None
            self.in_dim = None
            self.K = None
        else:
            self.alpha_param = None  # type: ignore

    def _infer_device(self, feat_rgb: Optional[torch.Tensor], feat_dct: Optional[torch.Tensor]) -> torch.device:
        """Infer device from features or fallback to parameters' device."""
        if feat_rgb is not None and isinstance(feat_rgb, torch.Tensor):
            return feat_rgb.device
        if feat_dct is not None and isinstance(feat_dct, torch.Tensor):
            return feat_dct.device
        # Fallback to parameters' device if any, else CPU
        for p in self.parameters():
            return p.device
        return torch.device("cpu")

    def _build_mlp(self, in_dim: int, K: int) -> None:
        """Build the MLP for dynamic fusion with given input dimension and number of classes."""
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, self.hidden, bias=True),
            nn.GELU(),
            nn.Linear(self.hidden, K, bias=True),
        )
        # Initialize bias for alpha_init
        if self.alpha_init is not None:
            with torch.no_grad():
                b = torch.logit(torch.tensor(float(self.alpha_init)))
                self.mlp[-1].bias.fill_(float(b))
        self.in_dim = in_dim
        self.K = K

    def ensure_ready(self, feat_rgb: Optional[torch.Tensor], feat_dct: Optional[torch.Tensor], K: int) -> None:
        """Ensure MLP is built and matches current feature dimensions."""
        d_r = int(feat_rgb.size(1)) if (feat_rgb is not None and feat_rgb.dim() == 2) else 0
        d_d = int(feat_dct.size(1)) if (feat_dct is not None and feat_dct.dim() == 2) else 0
        in_dim = d_r + d_d
        if self.mlp is None or self.in_dim != in_dim or self.K != K:
            self._build_mlp(in_dim, K)
            # Move freshly built MLP to the same device as features
            dev = self._infer_device(feat_rgb, feat_dct)
            if dev is not None and isinstance(dev, torch.device):
                self.mlp.to(dev)

    def forward(self, logits_rgb: torch.Tensor, logits_dct: torch.Tensor, feat_rgb: Optional[torch.Tensor] = None, feat_dct: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.mode == "rgb_only":
            return logits_rgb, None
        if self.mode == "dct_only":
            return logits_dct, None
        if self.mode == "late_fusion_avg":
            return (logits_rgb + logits_dct) * 0.5, None
        if self.mode == "late_fusion_alpha_scalar":
            a = torch.sigmoid(self.alpha_param)
            return (1 - a) * logits_rgb + a * logits_dct, a
        if self.mode == "late_fusion_alpha_classwise":
            a = torch.sigmoid(self.alpha_param).view(1, -1)
            return (1 - a) * logits_rgb + a * logits_dct, a
        if self.mode == "late_fusion_dynamic":
            assert feat_rgb is not None and feat_dct is not None
            # Ensure MLP is built and matches dimensions
            K = logits_rgb.size(1)
            self.ensure_ready(feat_rgb, feat_dct, K)
            x = torch.cat([feat_rgb, feat_dct], dim=1)
            # Extra safety: rebuild if unexpected dimension mismatch
            if (self.in_dim is None) or (x.size(1) != self.in_dim) or (self.mlp is None):
                self._build_mlp(x.size(1), K)
                self.mlp.to(x.device)
            else:
                # If already built, still make sure it's on the same device as x
                try:
                    if next(self.mlp.parameters()).device != x.device:
                        self.mlp.to(x.device)
                except StopIteration:
                    # no params found (unlikely), skip
                    pass
            a = torch.sigmoid(self.mlp(x))  # [B, K]
            if self.alpha_min is not None or self.alpha_max is not None:
                lo = self.alpha_min if self.alpha_min is not None else 0.0
                hi = self.alpha_max if self.alpha_max is not None else 1.0
                a = torch.clamp(a, lo, hi)
            z = (1.0 - a) * logits_rgb + a * logits_dct
            return z, a
        raise ValueError(f"Unknown fusion mode: {self.mode}")


class SwinDCTSystem(nn.Module):
    """Full system: RGB backbone, DCT branch, and late fusion."""

    def __init__(self, cfg: Mapping[str, Any], num_classes: int) -> None:
        super().__init__()
        self.cfg = cfg
        mode = cfg["models"]["dct"]["mode"]
        rgb_pretrained = bool(cfg["models"]["rgb"].get("pretrained", True))
        self.rgb = RGBBackbone(num_classes=num_classes, pretrained=rgb_pretrained)

        if mode == "freq_cnn":
            self.dct = DCTBranchFreqCNN(
                num_classes=num_classes,
                k=int(cfg["models"]["dct"].get("k", 8)),
                standardize=bool(cfg["models"]["dct"]["norm"].get("standardize", True)),
                clip_sigma=float(cfg["models"]["dct"]["norm"].get("clip_sigma", 3.0)),
            )
        elif mode == "idct_cnn":
            self.dct = DCTBranchIDCTCNN(
                num_classes=num_classes,
                standardize=bool(cfg["models"]["dct"]["norm"].get("standardize", True)),
                clip_sigma=float(cfg["models"]["dct"]["norm"].get("clip_sigma", 3.0)),
            )
        else:
            raise ValueError(f"Unknown DCT mode: {mode}")

        fusion_cfg = cfg["fusion"]
        self.fusion = FusionHead(
            mode=cfg["experiment"]["type"],
            num_classes=num_classes,
            alpha_init=float(fusion_cfg.get("alpha_init", 0.3)),
            dynamic_hidden=int(fusion_cfg.get("dynamic_hidden", 256)),
            alpha_min=fusion_cfg.get("alpha_min", None),
            alpha_max=fusion_cfg.get("alpha_max", None),
        )
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        mode = getattr(self.fusion, "mode", "late_fusion_dynamic")

        if mode == "rgb_only":
            logits_rgb, feat_rgb = self.rgb(x)
            return {
                "logits": logits_rgb,
                "logits_rgb": logits_rgb,
                "feat_rgb": feat_rgb,
            }

        if mode == "dct_only":
            logits_dct, feat_dct = self.dct(x)
            return {
                "logits": logits_dct,
                "logits_dct": logits_dct,
                "feat_dct": feat_dct,
            }

        # default: compute both and fuse
        logits_rgb, feat_rgb = self.rgb(x)
        logits_dct, feat_dct = self.dct(x)
        # Ensure fusion head is ready (for dynamic mode)
        if getattr(self.fusion, "mode", None) == "late_fusion_dynamic":
            if hasattr(self.fusion, "ensure_ready"):
                self.fusion.ensure_ready(feat_rgb, feat_dct, K=self.num_classes)
        fused, alpha = self.fusion(logits_rgb, logits_dct, feat_rgb, feat_dct)

        out: Dict[str, torch.Tensor] = {
            "logits": fused,
            "logits_rgb": logits_rgb,
            "logits_dct": logits_dct,
            "feat_rgb": feat_rgb,
            "feat_dct": feat_dct,
        }
        if alpha is not None:
            out["alpha"] = alpha
        return out



