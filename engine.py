"""
Training and validation engine with metrics, logging, and checkpointing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Optional, List
import os
from pathlib import Path
import time

from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
from rich.console import Console
from rich.table import Table

console = Console()


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1, 5), num_classes: Optional[int] = None) -> List[float]:
    """
    Compute top-k accuracy.
    
    Args:
        output: Logits [B, num_classes]
        target: Labels [B]
        topk: Tuple of k values
        num_classes: Number of classes (if None, inferred from output shape)
        
    Returns:
        List of accuracies for each k
    """
    if num_classes is None:
        num_classes = output.size(1)
    
    # Clamp k values to be at most num_classes
    maxk = min(max(topk), num_classes)
    batch_size = target.size(0)
    
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    
    res = []
    for k in topk:
        # If k > num_classes, use num_classes (which equals 100% accuracy)
        effective_k = min(k, num_classes)
        correct_k = correct[:effective_k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size).item())
    return res


def compute_metrics(output: torch.Tensor, target: torch.Tensor, num_classes: int) -> Dict[str, float]:
    """
    Compute classification metrics: acc@1, acc@5, precision, recall, F1 (macro and per-class).
    
    Args:
        output: Logits [B, num_classes]
        target: Labels [B]
        num_classes: Number of classes
        
    Returns:
        Dictionary of metrics
    """
    # Top-k accuracy
    acc1, acc5 = accuracy(output, target, topk=(1, 5), num_classes=num_classes)
    
    # Confusion matrix
    _, pred = output.max(1)
    correct = pred.eq(target)
    
    # Per-class metrics
    tp = torch.zeros(num_classes, device=output.device)
    fp = torch.zeros(num_classes, device=output.device)
    fn = torch.zeros(num_classes, device=output.device)
    
    for c in range(num_classes):
        tp[c] = ((pred == c) & (target == c)).sum().float()
        fp[c] = ((pred == c) & (target != c)).sum().float()
        fn[c] = ((pred != c) & (target == c)).sum().float()
    
    # Precision, Recall, F1 per class
    precision_per_class = tp / (tp + fp + 1e-6)
    recall_per_class = tp / (tp + fn + 1e-6)
    f1_per_class = 2 * precision_per_class * recall_per_class / (precision_per_class + recall_per_class + 1e-6)
    
    # Macro averages
    precision_macro = precision_per_class.mean().item()
    recall_macro = recall_per_class.mean().item()
    f1_macro = f1_per_class.mean().item()
    
    # Overall (micro) precision/recall/F1 (same as acc1 for balanced case)
    precision_micro = tp.sum() / (tp.sum() + fp.sum() + 1e-6)
    recall_micro = tp.sum() / (tp.sum() + fn.sum() + 1e-6)
    f1_micro = 2 * precision_micro * recall_micro / (precision_micro + recall_micro + 1e-6)
    
    return {
        "acc1": acc1,
        "acc5": acc5,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_micro": precision_micro.item(),
        "recall_micro": recall_micro.item(),
        "f1_micro": f1_micro.item(),
        "precision_per_class": precision_per_class.cpu().tolist(),
        "recall_per_class": recall_per_class.cpu().tolist(),
        "f1_per_class": f1_per_class.cpu().tolist(),
    }


class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance."""
    
    def __init__(self, gamma: float = 2.0, alpha: Optional[float] = None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(input, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.alpha is not None:
            alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
            focal_loss = alpha_t * focal_loss
        
        return focal_loss.mean()


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    log_interval: int,
    writer: Optional[SummaryWriter] = None,
    cfg: Optional[Dict] = None,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    if hasattr(model, 'set_epoch'):
        model.set_epoch(epoch)
    
    total_loss = 0.0
    all_outputs = []
    all_targets = []
    all_alphas = []
    
    num_batches = len(loader)
    
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(f"[cyan]Epoch {epoch} Training", total=num_batches)
        
        for batch_idx, batch in enumerate(loader):
            rgb = batch["rgb"].to(device)
            band_low = batch["band_low"].to(device)
            band_mid = batch["band_mid"].to(device)
            band_high = batch["band_high"].to(device)
            gray = batch.get("gray", None)
            if gray is not None:
                gray = gray.to(device)
            target = batch["label"].to(device)
            
            # Forward
            optimizer.zero_grad()
            logits, alpha = model(rgb, band_low, band_mid, band_high, gray=gray)
            
            # Loss
            loss = criterion(logits, target)
            
            # Backward
            loss.backward()
            optimizer.step()
            
            # Accumulate
            total_loss += loss.item()
            all_outputs.append(logits.detach())
            all_targets.append(target.detach())
            all_alphas.append(alpha.detach().mean(0))  # Average over batch
            
            # Logging
            if (batch_idx + 1) % log_interval == 0:
                current_loss = total_loss / (batch_idx + 1)
                progress.update(task, advance=1, description=f"[cyan]Epoch {epoch} Training - Loss: {current_loss:.4f}")
                
                if writer is not None:
                    writer.add_scalar("Train/Loss", current_loss, epoch * num_batches + batch_idx)
            
            progress.update(task, advance=1)
    
    # Compute metrics
    all_outputs = torch.cat(all_outputs, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    all_alphas = torch.stack(all_alphas, dim=0).mean(0)  # Average over all batches
    
    metrics = compute_metrics(all_outputs, all_targets, cfg["model"]["num_classes"] if cfg else 2)
    metrics["loss"] = total_loss / num_batches
    
    # Log alpha weights
    if writer is not None:
        for i, alpha_val in enumerate(all_alphas):
            writer.add_scalar(f"Train/Alpha_Branch_{i}", alpha_val.item(), epoch)
    
    return metrics


def validate_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    writer: Optional[SummaryWriter] = None,
    cfg: Optional[Dict] = None,
) -> Dict[str, float]:
    """Validate for one epoch."""
    model.eval()
    
    total_loss = 0.0
    all_outputs = []
    all_targets = []
    all_alphas = []
    
    with torch.no_grad():
        for batch in loader:
            rgb = batch["rgb"].to(device)
            band_low = batch["band_low"].to(device)
            band_mid = batch["band_mid"].to(device)
            band_high = batch["band_high"].to(device)
            gray = batch.get("gray", None)
            if gray is not None:
                gray = gray.to(device)
            target = batch["label"].to(device)
            
            # Forward
            logits, alpha = model(rgb, band_low, band_mid, band_high, gray=gray)
            
            # Loss
            loss = criterion(logits, target)
            
            # Accumulate
            total_loss += loss.item()
            all_outputs.append(logits)
            all_targets.append(target)
            all_alphas.append(alpha.mean(0))  # Average over batch
    
    # Compute metrics
    all_outputs = torch.cat(all_outputs, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    all_alphas = torch.stack(all_alphas, dim=0).mean(0)
    
    metrics = compute_metrics(all_outputs, all_targets, cfg["model"]["num_classes"] if cfg else 2)
    metrics["loss"] = total_loss / len(loader)
    metrics["alpha"] = all_alphas.cpu().tolist()  # Store alpha for printing
    
    # Log to tensorboard
    if writer is not None:
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and key != "alpha":
                writer.add_scalar(f"Val/{key}", value, epoch)
            elif key == "alpha":
                for i, alpha_val in enumerate(all_alphas):
                    writer.add_scalar(f"Val/Alpha_Branch_{i}", alpha_val.item(), epoch)
        
        # Log c1, c2
        if hasattr(model, 'get_dct_params'):
            c1, c2 = model.get_dct_params()
            writer.add_scalar("Val/DCT_c1", c1.item(), epoch)
            writer.add_scalar("Val/DCT_c2", c2.item(), epoch)
    
    return metrics


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_epochs: int,
    num_training_epochs: int,
    num_cycles: float = 0.5,
    last_epoch: int = -1,
):
    """Create cosine learning rate schedule with warmup."""
    from torch.optim.lr_scheduler import LambdaLR
    import math
    
    def lr_lambda(current_epoch):
        if current_epoch < num_warmup_epochs:
            return float(current_epoch) / float(max(1, num_warmup_epochs))
        progress = float(current_epoch - num_warmup_epochs) / float(max(1, num_training_epochs - num_warmup_epochs))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
    
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def train(cfg: Dict):
    """Main training loop."""
    # Setup
    set_seed(cfg["experiment"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print(f"[green]Using device: {device}")
    
    # Output directory
    output_dir = Path(cfg["experiment"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config copy
    import yaml
    with open(output_dir / "config.yaml.copy", "w") as f:
        yaml.dump(cfg, f)
    
    # TensorBoard
    writer = None
    if cfg["logging"]["tensorboard"]:
        writer = SummaryWriter(log_dir=str(output_dir))
    
    # Log file
    log_file = open(output_dir / "log.txt", "w")
    
    # Data
    from data import FireDataset, compute_dataset_stats
    
    compute_dataset_stats(cfg["data"]["root"])
    
    train_dataset = FireDataset(
        root_dir=cfg["data"]["root"],
        split="train",
        img_size=cfg["data"]["img_size"],
        augment=True,
    )
    val_dataset = FireDataset(
        root_dir=cfg["data"]["root"],
        split="val",
        img_size=cfg["data"]["img_size"],
        augment=False,
        class_to_idx=train_dataset.class_to_idx,  # Use same mapping
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["data"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["data"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
    )
    
    # Model
    from models import MultiBranchClassifier
    
    model = MultiBranchClassifier(
        feature_dim=cfg["model"]["feature_dim"],
        num_classes=cfg["model"]["num_classes"],
        swin_weight_sharing=cfg["model"]["swin_weight_sharing"],
        fusion=cfg["model"]["fusion"],
        branch_dropout=cfg["model"]["branch_dropout"],
        freeze_backbone_epochs=cfg["model"]["freeze_backbone_epochs"],
        dct_c1_init=cfg["dct"]["c1_init"],
        dct_c2_init=cfg["dct"]["c2_init"],
        dct_k=cfg["dct"]["k"],
    ).to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    console.print(f"[cyan]Total parameters: {total_params:,}")
    console.print(f"[cyan]Trainable parameters: {trainable_params:,}")
    
    # Loss
    if cfg["optim"]["loss"] == "focal":
        criterion = FocalLoss(
            gamma=cfg["optim"]["focal_gamma"],
            alpha=cfg["optim"]["focal_alpha"],
        )
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=cfg["optim"]["label_smoothing"])
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["optim"]["lr"],
        weight_decay=cfg["optim"]["weight_decay"],
        betas=tuple(cfg["optim"]["betas"]),
    )
    
    # Scheduler
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_epochs=cfg["optim"]["warmup_epochs"],
        num_training_epochs=cfg["optim"]["epochs"],
    )
    
    # Training loop
    best_metric = 0.0
    best_metric_name = cfg["logging"]["save_best_metric"]
    
    for epoch in range(1, cfg["optim"]["epochs"] + 1):
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch,
            cfg["logging"]["log_interval"], writer, cfg
        )
        
        # Validate
        val_metrics = validate_epoch(
            model, val_loader, criterion, device, epoch, writer, cfg
        )
        
        # Scheduler step
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # Get c1, c2
        c1, c2 = model.get_dct_params()
        
        # Print epoch summary
        alpha_str = ", ".join([f"{a:.3f}" for a in val_metrics.get('alpha', [0,0,0,0])])
        summary = (
            f"Epoch {epoch}/{cfg['optim']['epochs']} | "
            f"Train Loss: {train_metrics['loss']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Acc@1: {val_metrics['acc1']:.2f}% | "
            f"Val Acc@5: {val_metrics['acc5']:.2f}% | "
            f"Val F1-Macro: {val_metrics['f1_macro']:.4f} | "
            f"LR: {current_lr:.2e} | "
            f"c1: {c1.item():.3f} | c2: {c2.item():.3f} | "
            f"Alpha: [{alpha_str}]"
        )
        console.print(f"[yellow]{summary}")
        log_file.write(summary + "\n")
        log_file.flush()
        
        # Checkpoint
        metric_value = val_metrics.get(best_metric_name, val_metrics["acc1"])
        if metric_value > best_metric:
            best_metric = metric_value
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_metric": best_metric,
                "class_to_idx": train_dataset.class_to_idx,
                "c1": c1.item(),
                "c2": c2.item(),
            }
            torch.save(checkpoint, output_dir / "best.pt")
            console.print(f"[green]Saved best checkpoint (metric: {best_metric:.4f})")
        
        # Save last checkpoint
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_metric": best_metric,
            "class_to_idx": train_dataset.class_to_idx,
            "c1": c1.item(),
            "c2": c2.item(),
        }
        torch.save(checkpoint, output_dir / "last.pt")
    
    log_file.close()
    if writer is not None:
        writer.close()
    
    console.print(f"[green]Training completed! Best {best_metric_name}: {best_metric:.4f}")


def validate(cfg: Dict):
    """Validation only."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print(f"[green]Using device: {device}")
    
    # Data
    from data import FireDataset
    
    val_dataset = FireDataset(
        root_dir=cfg["data"]["root"],
        split="val",
        img_size=cfg["data"]["img_size"],
        augment=False,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["data"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
    )
    
    # Model
    from models import MultiBranchClassifier
    
    model = MultiBranchClassifier(
        feature_dim=cfg["model"]["feature_dim"],
        num_classes=cfg["model"]["num_classes"],
        swin_weight_sharing=cfg["model"]["swin_weight_sharing"],
        fusion=cfg["model"]["fusion"],
        branch_dropout=0.0,  # No dropout during validation
        freeze_backbone_epochs=0,
        dct_c1_init=cfg["dct"]["c1_init"],
        dct_c2_init=cfg["dct"]["c2_init"],
        dct_k=cfg["dct"]["k"],
    ).to(device)
    
    # Load checkpoint if available
    output_dir = Path(cfg["experiment"]["output_dir"])
    checkpoint_path = output_dir / "best.pt"
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        console.print(f"[green]Loaded checkpoint from epoch {checkpoint['epoch']}")
    else:
        console.print("[yellow]No checkpoint found, using random initialization")
    
    # Loss
    criterion = nn.CrossEntropyLoss()
    
    # Validate
    val_metrics = validate_epoch(model, val_loader, criterion, device, 0, None, cfg)
    
    # Print results
    console.print("\n[cyan]Validation Results:")
    console.print(f"  Acc@1: {val_metrics['acc1']:.2f}%")
    console.print(f"  Acc@5: {val_metrics['acc5']:.2f}%")
    console.print(f"  Precision (macro): {val_metrics['precision_macro']:.4f}")
    console.print(f"  Recall (macro): {val_metrics['recall_macro']:.4f}")
    console.print(f"  F1 (macro): {val_metrics['f1_macro']:.4f}")
    console.print(f"  Precision (micro): {val_metrics['precision_micro']:.4f}")
    console.print(f"  Recall (micro): {val_metrics['recall_micro']:.4f}")
    console.print(f"  F1 (micro): {val_metrics['f1_micro']:.4f}")

