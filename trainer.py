"""Training, validation, and evaluation loop with AMP, early stopping, and logging."""
from __future__ import annotations

import contextlib
import json
import os
import time
from typing import Any, Dict, List, Mapping, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Version-safe AMP imports
try:
    from torch.amp import autocast as _autocast, GradScaler as _GradScaler
    _AMP_BACKEND = "torch.amp"
except Exception:
    from torch.cuda.amp import autocast as _autocast, GradScaler as _GradScaler
    _AMP_BACKEND = "torch.cuda.amp"

import utils


class Trainer:
    """Encapsulates the full training/validation/testing routine."""

    def __init__(self, cfg: Mapping[str, Any], model: nn.Module, num_classes: int, out_dir: str) -> None:
        self.cfg = cfg
        self.model = model
        self.num_classes = num_classes
        self.out_dir = out_dir

        self.optimizer = utils.build_optimizer(model, cfg)
        self.scheduler = utils.build_scheduler(self.optimizer, cfg)
        self.amp_enabled = bool(cfg["optim"].get("amp", True))
        self.device_type = "cuda" if torch.cuda.is_available() else "cpu"

        # GradScaler: try new signature first, then fallback
        if self.amp_enabled:
            try:
                # Newer API (torch.amp) may accept device_type kw
                self.scaler = _GradScaler(device_type=self.device_type, enabled=True)
            except TypeError:
                # Older API (torch.cuda.amp) doesn't accept device_type
                try:
                    self.scaler = _GradScaler(enabled=True)
                except TypeError:
                    # Very old API: no enabled parameter
                    self.scaler = _GradScaler()
        else:
            self.scaler = None
        self.grad_clip = float(cfg["optim"].get("grad_clip_norm", 1.0))

        # Loss
        self.ce = nn.CrossEntropyLoss(label_smoothing=float(cfg["loss"].get("label_smoothing", 0.0)))
        self.lambda_branch = float(cfg["loss"].get("lambda_branch", 0.0))

        # Logging
        self.loggers = utils.setup_logging(out_dir, use_tensorboard=bool(cfg["log"].get("use_tensorboard", True)))
        self.best_metric_name = str(cfg["experiment"].get("save_best_metric", "f1_macro"))
        self.best_metric = -1e9
        self.best_path = os.path.join(out_dir, "best.pt")

    def _compute_loss(self, outputs: Dict[str, torch.Tensor], targets: torch.Tensor) -> torch.Tensor:
        loss_main = self.ce(outputs["logits"], targets)
        lam = self.lambda_branch
        if lam > 0:
            if "logits_rgb" in outputs:
                loss_main = loss_main + lam * self.ce(outputs["logits_rgb"], targets)
            if "logits_dct" in outputs:
                loss_main = loss_main + lam * self.ce(outputs["logits_dct"], targets)
        return loss_main

    def _metrics(self, logits: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        acc = utils.accuracy(logits, targets)
        p_mac, r_mac, f1_mac, f1_mic = utils.precision_recall_f1(logits, targets, self.num_classes)
        metrics = {
            "acc": acc,
            "f1_macro": f1_mac,
            "f1_micro": f1_mic,
        }
        if self.num_classes == 2 and bool(self.cfg["eval"].get("compute_auc", True)):
            probs = F.softmax(logits, dim=1)[:, 1]
            auc = utils.binary_auc(probs, targets)
            metrics["auc"] = auc
        else:
            metrics["auc"] = float("nan")
        return metrics

    def _should_improve(self, metrics: Mapping[str, float]) -> bool:
        name = self.best_metric_name
        val = metrics.get(name, float("nan"))
        return val == val and val > self.best_metric  # check for NaN, then compare

    def _save_ckpt(self, path: str, epoch: int) -> None:
        tosave = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "epoch": epoch,
            "cfg": dict(self.cfg),
        }
        torch.save(tosave, path)

    def _log_epoch(self, epoch: int, split: str, loss_val: float, metrics: Mapping[str, float]) -> None:
        row = {
            "epoch": epoch,
            "split": split,
            "loss": loss_val,
            "acc": metrics.get("acc", float("nan")),
            "f1_macro": metrics.get("f1_macro", float("nan")),
            "f1_micro": metrics.get("f1_micro", float("nan")),
            "auc": metrics.get("auc", float("nan")),
        }
        self.loggers.csv.log(row)
        if self.loggers.writer is not None:
            for k, v in row.items():
                if k in {"epoch", "split"}:
                    continue
                self.loggers.writer.add_scalar(f"{split}/{k}", float(v), epoch)

    def fit(self, train_loader, val_loader=None) -> None:
        # GPU detection and logging at training start
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory_allocated = torch.cuda.memory_allocated(0) / (1024**3)  # GB
            gpu_memory_reserved = torch.cuda.memory_reserved(0) / (1024**3)  # GB
            print(f"[Training] Starting training on GPU: {gpu_name}")
            print(f"[Training] GPU memory - Allocated: {gpu_memory_allocated:.2f} GB, Reserved: {gpu_memory_reserved:.2f} GB")
        else:
            print("[Training] Starting training on CPU")
        
        epochs = int(self.cfg["optim"].get("epochs", 50))
        patience = int(self.cfg["optim"].get("patience", 5))
        if patience < 0:
            print(f"[Training] Early stopping disabled (patience={patience})")
        else:
            print(f"[Training] Early stopping patience: {patience} epochs")
        best_epoch = -1
        no_improve = 0
        for epoch in range(epochs):
            self.model.train()
            t0 = time.time()
            total_loss = 0.0
            n = 0
            for i, (images, targets) in enumerate(train_loader):
                images = images.cuda(non_blocking=True) if torch.cuda.is_available() else images
                targets = targets.cuda(non_blocking=True) if torch.cuda.is_available() else targets
                self.optimizer.zero_grad(set_to_none=True)
                # autocast context with version-safe fallback
                amp_on = self.amp_enabled
                try:
                    # torch.amp.autocast usually accepts device_type first (or kw)
                    ctx = _autocast(self.device_type, enabled=amp_on)
                except TypeError:
                    # torch.cuda.amp.autocast: no device_type parameter
                    try:
                        ctx = _autocast(enabled=amp_on)
                    except TypeError:
                        # Very old API: no enabled parameter
                        if amp_on and torch.cuda.is_available():
                            ctx = _autocast()
                        else:
                            ctx = contextlib.nullcontext()
                with ctx:
                    outputs = self.model(images)
                    loss = self._compute_loss(outputs, targets)
                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                    if self.grad_clip and self.grad_clip > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    if self.grad_clip and self.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    self.optimizer.step()
                total_loss += loss.item() * images.size(0)
                n += images.size(0)
                if (i + 1) % int(self.cfg["log"].get("log_interval", 50)) == 0:
                    print(f"Epoch {epoch} Iter {i+1}: loss={loss.item():.4f}")

            train_loss = total_loss / max(1, n)
            self.scheduler.step()

            # Validation
            val_metrics = {"acc": float("nan"), "f1_macro": float("nan"), "f1_micro": float("nan"), "auc": float("nan")}
            val_loss = float("nan")
            if val_loader is not None:
                self.model.eval()
                with torch.no_grad():
                    total_vloss = 0.0
                    vn = 0
                    logits_all: List[torch.Tensor] = []
                    targets_all: List[torch.Tensor] = []
                    for images, targets in val_loader:
                        images = images.cuda(non_blocking=True) if torch.cuda.is_available() else images
                        targets = targets.cuda(non_blocking=True) if torch.cuda.is_available() else targets
                        outputs = self.model(images)
                        loss = self._compute_loss(outputs, targets)
                        total_vloss += loss.item() * images.size(0)
                        vn += images.size(0)
                        logits_all.append(outputs["logits"].detach())
                        targets_all.append(targets.detach())
                    val_loss = total_vloss / max(1, vn)
                    logits_cat = torch.cat(logits_all, dim=0)
                    targets_cat = torch.cat(targets_all, dim=0)
                    val_metrics = self._metrics(logits_cat, targets_cat)

            # Logging
            self._log_epoch(epoch, "train", train_loss, {"acc": float("nan"), "f1_macro": float("nan"), "f1_micro": float("nan"), "auc": float("nan")})
            if val_loader is not None:
                self._log_epoch(epoch, "val", val_loss, val_metrics)

            # Early stopping and checkpointing
            improved = self._should_improve(val_metrics if val_loader is not None else {self.best_metric_name: -float("inf")})
            if improved:
                self.best_metric = val_metrics[self.best_metric_name]
                best_epoch = epoch
                no_improve = 0
                self._save_ckpt(self.best_path, epoch)
                print(f"[Best] Epoch {epoch}: {self.best_metric_name}={self.best_metric:.4f}")
            else:
                no_improve += 1

            print(f"Epoch {epoch} done in {time.time()-t0:.1f}s, train_loss={train_loss:.4f}")
            if patience >= 0 and val_loader is not None and no_improve >= patience:
                print(f"Early stopping triggered (patience={patience}).")
                break

        # Save final checkpoint
        self._save_ckpt(os.path.join(self.out_dir, "last.pt"), best_epoch)
        if self.loggers.writer is not None:
            self.loggers.writer.flush()

    def evaluate(self, loader) -> Dict[str, float]:
        self.model.eval()
        with torch.no_grad():
            total_loss = 0.0
            n = 0
            logits_all: List[torch.Tensor] = []
            targets_all: List[torch.Tensor] = []
            for images, targets in loader:
                images = images.cuda(non_blocking=True) if torch.cuda.is_available() else images
                targets = targets.cuda(non_blocking=True) if torch.cuda.is_available() else targets
                outputs = self.model(images)
                loss = self._compute_loss(outputs, targets)
                total_loss += loss.item() * images.size(0)
                n += images.size(0)
                logits_all.append(outputs["logits"].detach())
                targets_all.append(targets.detach())
            loss_val = total_loss / max(1, n)
            logits_cat = torch.cat(logits_all, dim=0)
            targets_cat = torch.cat(targets_all, dim=0)
            metrics = self._metrics(logits_cat, targets_cat)
        # Save confusion matrix
        if bool(self.cfg["log"].get("save_confmat", True)):
            preds = logits_cat.argmax(dim=1)
            cm = utils.confusion_matrix(preds, targets_cat, self.num_classes).cpu().numpy()
            import numpy as np
            import csv
            with open(os.path.join(self.out_dir, "confusion_matrix.csv"), "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                for row in cm:
                    w.writerow([int(v) for v in row])
        # Save AUC if binary
        if self.num_classes == 2 and bool(self.cfg["log"].get("save_roc", True)):
            probs = F.softmax(logits_cat, dim=1)[:, 1]
            auc = utils.binary_auc(probs, targets_cat)
            with open(os.path.join(self.out_dir, "auc.txt"), "w", encoding="utf-8") as f:
                f.write(f"AUC: {auc:.6f}\n")
        return {"loss": loss_val, **metrics}



