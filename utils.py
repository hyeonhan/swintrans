"""Utility functions: config I/O, logging, metrics, optimizer/scheduler, seeding.

This module provides helper utilities used across training and evaluation.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.tensorboard import SummaryWriter


DEFAULT_CONFIG_TEMPLATE = """
experiment:
  output_dir: "./runs/exp1"
  type: "late_fusion_dynamic"   # ["rgb_only","dct_only","late_fusion_avg","late_fusion_alpha_scalar","late_fusion_alpha_classwise","late_fusion_dynamic"]
  save_best_metric: "f1_macro"  # or "auc","acc"
  seed: 42

data:
  root: "/data"
  img_size: 224
  num_workers: 8
  batch_size: 64
  persistent_workers: true
  pin_memory: true
  normalize:
    mean: [0.485, 0.456, 0.406]
    std:  [0.229, 0.224, 0.225]
  augment:
    color_jitter: false
    random_horizontal_flip: true

models:
  rgb:
    pretrained: true
  dct:
    mode: "freq_cnn"           # ["freq_cnn","idct_cnn"]
    k: 8                       # (freq_cnn) 64->k per color
    norm:
      standardize: true
      clip_sigma: 3.0

fusion:
  alpha_init: 0.3
  dynamic_hidden: 256
  alpha_min: 0.1
  alpha_max: 0.9

loss:
  lambda_branch: 0.2
  label_smoothing: 0.1

optim:
  lr: 3.0e-4
  weight_decay: 0.05
  betas: [0.9, 0.999]
  epochs: 50
  warmup_epochs: 5
  grad_clip_norm: 1.0
  amp: true
  ema:
    enable: false

eval:
  compute_auc: true
  confmat_normalize: "true"

log:
  use_tensorboard: true
  log_interval: 50
  save_confmat: true
  save_roc: true
"""


def load_config(path: str) -> Dict[str, Any]:
    """Load YAML config from disk.

    If file does not exist, create the default template and exit with a message.
    """
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(DEFAULT_CONFIG_TEMPLATE.strip() + "\n")
        print(f"Config template created at: {path}\nPlease edit the config and re-run.")
        sys.exit(0)
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def save_config_copy(cfg: Mapping[str, Any], out_dir: str) -> None:
    """Save a copy of the merged config to the run directory."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "config.effective.yaml")
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(dict(cfg), f, sort_keys=False)


def merge_overrides(cfg: MutableMapping[str, Any], overrides: Iterable[str]) -> Dict[str, Any]:
    """Merge CLI key=value overrides into the loaded config.

    Supports dotted keys: section.sub=value
    """
    for ov in overrides:
        if "=" not in ov:
            continue
        key, val = ov.split("=", 1)
        target = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            if p not in target or not isinstance(target[p], dict):
                target[p] = {}
            target = target[p]
        # Try to parse primitive types
        parsed: Any
        if val.lower() in {"true", "false"}:
            parsed = val.lower() == "true"
        else:
            try:
                if "." in val or "e" in val.lower():
                    parsed = float(val)
                else:
                    parsed = int(val)
            except ValueError:
                # try list
                if val.startswith("[") and val.endswith("]"):
                    try:
                        parsed = yaml.safe_load(val)
                    except Exception:
                        parsed = val
                else:
                    parsed = val
        target[parts[-1]] = parsed
    return cfg


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


class CSVLogger:
    """Lightweight CSV logger for metrics per epoch."""

    def __init__(self, path: str, fieldnames: List[str]) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.path = path
        self.fieldnames = fieldnames
        newfile = not os.path.exists(path)
        self._f = open(path, "a", newline="", encoding="utf-8")
        self._w = csv.DictWriter(self._f, fieldnames=fieldnames)
        if newfile:
            self._w.writeheader()

    def log(self, row: Mapping[str, Any]) -> None:
        self._w.writerow(row)
        self._f.flush()

    def close(self) -> None:
        self._f.close()


@dataclass
class Loggers:
    """Container for loggers used by the trainer."""
    writer: Optional[SummaryWriter]
    csv: CSVLogger


def setup_logging(out_dir: str, use_tensorboard: bool) -> Loggers:
    """Create logging sinks in the output directory."""
    os.makedirs(out_dir, exist_ok=True)
    writer = SummaryWriter(out_dir) if use_tensorboard else None
    csv_logger = CSVLogger(os.path.join(out_dir, "metrics.csv"), fieldnames=[
        "epoch",
        "split",
        "loss",
        "acc",
        "f1_macro",
        "f1_micro",
        "auc",
    ])
    return Loggers(writer=writer, csv=csv_logger)


def _one_hot(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    oh = torch.zeros((labels.numel(), num_classes), dtype=torch.float32, device=labels.device)
    oh.scatter_(1, labels.view(-1, 1), 1.0)
    return oh


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute top-1 accuracy."""
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


def precision_recall_f1(logits: torch.Tensor, targets: torch.Tensor, num_classes: int) -> Tuple[float, float, float, float]:
    """Compute macro/micro precision, recall, f1.

    Returns: (precision_macro, recall_macro, f1_macro, f1_micro)
    """
    preds = logits.argmax(dim=1)
    cm = confusion_matrix(preds, targets, num_classes)
    tp = torch.diag(cm).float()
    fp = cm.sum(dim=1).float() - tp
    fn = cm.sum(dim=0).float() - tp
    with torch.no_grad():
        precision_per_class = tp / (tp + fp + 1e-8)
        recall_per_class = tp / (tp + fn + 1e-8)
        f1_per_class = 2 * precision_per_class * recall_per_class / (precision_per_class + recall_per_class + 1e-8)
        precision_macro = precision_per_class.mean().item()
        recall_macro = recall_per_class.mean().item()
        f1_macro = f1_per_class.mean().item()
        # micro
        tp_micro = tp.sum()
        fp_micro = fp.sum()
        fn_micro = fn.sum()
        precision_micro = (tp_micro / (tp_micro + fp_micro + 1e-8)).item()
        recall_micro = (tp_micro / (tp_micro + fn_micro + 1e-8)).item()
        f1_micro = (2 * precision_micro * recall_micro / (precision_micro + recall_micro + 1e-8))
    return precision_macro, recall_macro, f1_macro, float(f1_micro)


def confusion_matrix(preds: torch.Tensor, targets: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Compute confusion matrix (num_classes x num_classes)."""
    cm = torch.zeros((num_classes, num_classes), dtype=torch.long, device=targets.device)
    for t, p in zip(targets.view(-1), preds.view(-1)):
        cm[p.long(), t.long()] += 1
    return cm


def binary_auc(scores_pos_class: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute ROC AUC for binary classification using trapezoidal rule.

    scores_pos_class: probabilities or logits for positive class (shape [N])
    targets: 0/1 labels (shape [N])
    """
    # Convert logits to probabilities if necessary
    if scores_pos_class.dim() != 1:
        raise ValueError("scores_pos_class must be 1D")
    s = scores_pos_class.detach().float().cpu().numpy()
    y = targets.detach().float().cpu().numpy()
    # Sort by score descending
    order = np.argsort(-s)
    s = s[order]
    y = y[order]
    P = y.sum()
    N = len(y) - P
    if P == 0 or N == 0:
        return float("nan")
    tp = 0.0
    fp = 0.0
    tpr_list = [0.0]
    fpr_list = [0.0]
    last_s = None
    for score, label in zip(s, y):
        if last_s is None or score != last_s:
            tpr_list.append(tp / P)
            fpr_list.append(fp / N)
            last_s = score
        if label > 0.5:
            tp += 1.0
        else:
            fp += 1.0
    tpr_list.append(tp / P)
    fpr_list.append(fp / N)
    # Compute AUC using trapezoidal rule
    return float(np.trapz(tpr_list, fpr_list))


def build_optimizer(model: nn.Module, cfg: Mapping[str, Any]) -> optim.Optimizer:
    """Create AdamW optimizer from cfg.optim."""
    opt_cfg = cfg["optim"]
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(
        params,
        lr=float(opt_cfg.get("lr", 3e-4)),
        betas=tuple(opt_cfg.get("betas", [0.9, 0.999])),
        weight_decay=float(opt_cfg.get("weight_decay", 0.05)),
    )
    return optimizer


class WarmupCosineLR(_LRScheduler):
    """Cosine scheduler with linear warmup over warmup_epochs."""

    def __init__(self, optimizer: optim.Optimizer, total_epochs: int, warmup_epochs: int = 0, last_epoch: int = -1):
        self.total_epochs = max(1, total_epochs)
        self.warmup_epochs = max(0, warmup_epochs)
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        epoch = self.last_epoch + 1
        lrs: List[float] = []
        for base_lr in self.base_lrs:
            if epoch < self.warmup_epochs:
                lr = base_lr * float(epoch + 1) / float(max(1, self.warmup_epochs))
            else:
                t = (epoch - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
                lr = 0.5 * base_lr * (1.0 + math.cos(math.pi * t))
            lrs.append(lr)
        return lrs


def build_scheduler(optimizer: optim.Optimizer, cfg: Mapping[str, Any]) -> _LRScheduler:
    """Build Warmup + Cosine scheduler from cfg.optim."""
    o = cfg["optim"]
    return WarmupCosineLR(
        optimizer,
        total_epochs=int(o.get("epochs", 50)),
        warmup_epochs=int(o.get("warmup_epochs", 5)),
    )


def pretty_print_experiment_summary(cfg: Mapping[str, Any], class_to_idx: Mapping[str, int], split_sizes: Mapping[str, int]) -> str:
    """Return a human-readable summary string of experiment configuration and data."""
    lines: List[str] = []
    lines.append("Experiment Summary")
    lines.append("- Output Dir: " + str(cfg["experiment"]["output_dir"]))
    lines.append("- Fusion Type: " + str(cfg["experiment"]["type"]))
    lines.append("- Seed: " + str(cfg["experiment"].get("seed", 42)))
    lines.append("- Classes (alphabetical):")
    for cls in sorted(class_to_idx.keys()):
        lines.append(f"    {class_to_idx[cls]} -> {cls}")
    lines.append("- Dataset sizes:")
    for split, size in split_sizes.items():
        lines.append(f"    {split}: {size}")
    return "\n".join(lines)



