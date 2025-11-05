"""
Training and validation engine with metrics, logging, and checkpointing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Optional, List, Tuple
import os
from pathlib import Path
import time
import numpy as np

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
    grad_accum_steps: int = 1,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    if hasattr(model, 'set_epoch'):
        model.set_epoch(epoch)
    
    total_loss = 0.0
    all_outputs = []
    all_targets = []
    
    num_batches = len(loader)
    optimizer.zero_grad()
    
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(f"[cyan]Epoch {epoch} Training", total=num_batches)
        
        for batch_idx, batch in enumerate(loader):
            # Handle tuple format: (rgb, dct, label)
            rgb, dct, target = batch
            rgb = rgb.to(device)
            dct = dct.to(device) if dct is not None else None
            target = target.to(device)
            
            # Sanity checks
            assert rgb.shape[0] == target.shape[0], f"Batch size mismatch: rgb {rgb.shape[0]} vs target {target.shape[0]}"
            assert rgb.shape[1] == 3, f"RGB should have 3 channels, got {rgb.shape[1]}"
            img_size = cfg["data"]["img_size"] if cfg else 224
            assert rgb.shape[2] == img_size and rgb.shape[3] == img_size, f"RGB shape should be [B, 3, {img_size}, {img_size}], got {rgb.shape}"
            if dct is not None:
                assert dct.shape == rgb.shape, f"DCT shape {dct.shape} must match RGB shape {rgb.shape}"
            
            # Forward
            logits = model(rgb, dct)
            
            # Loss (normalize by grad_accum_steps)
            loss = criterion(logits, target) / grad_accum_steps
            
            # Backward
            loss.backward()
            
            # Step optimizer every grad_accum_steps
            if (batch_idx + 1) % grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            # Accumulate
            total_loss += loss.item() * grad_accum_steps  # Scale back for logging
            all_outputs.append(logits.detach())
            all_targets.append(target.detach())
            
            # Logging
            if (batch_idx + 1) % log_interval == 0:
                current_loss = total_loss / (batch_idx + 1)
                progress.update(task, advance=1, description=f"[cyan]Epoch {epoch} Training - Loss: {current_loss:.4f}")
                
                if writer is not None:
                    writer.add_scalar("Train/Loss", current_loss, epoch * num_batches + batch_idx)
            
            progress.update(task, advance=1)
        
        # Handle remaining gradients if batch count is not divisible by grad_accum_steps
        if num_batches % grad_accum_steps != 0:
            optimizer.step()
            optimizer.zero_grad()
    
    # Compute metrics
    all_outputs = torch.cat(all_outputs, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    metrics = compute_metrics(all_outputs, all_targets, cfg["model"]["num_classes"] if cfg else 2)
    metrics["loss"] = total_loss / num_batches
    
    return metrics


def validate_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    writer: Optional[SummaryWriter] = None,
    cfg: Optional[Dict] = None,
    return_predictions: bool = False,
) -> Dict:
    """
    Validate for one epoch.
    
    Args:
        return_predictions: If True, also return predictions and probabilities
        
    Returns:
        Dictionary with metrics, and optionally predictions/probabilities
    """
    model.eval()
    
    total_loss = 0.0
    all_outputs = []
    all_targets = []
    
    with torch.no_grad():
        for batch in loader:
            # Handle tuple format: (rgb, dct, label)
            rgb, dct, target = batch
            rgb = rgb.to(device)
            dct = dct.to(device) if dct is not None else None
            target = target.to(device)
            
            # Sanity checks
            assert rgb.shape[0] == target.shape[0], f"Batch size mismatch: rgb {rgb.shape[0]} vs target {target.shape[0]}"
            assert rgb.shape[1] == 3, f"RGB should have 3 channels, got {rgb.shape[1]}"
            img_size = cfg["data"]["img_size"] if cfg else 224
            assert rgb.shape[2] == img_size and rgb.shape[3] == img_size, f"RGB shape should be [B, 3, {img_size}, {img_size}], got {rgb.shape}"
            if dct is not None:
                assert dct.shape == rgb.shape, f"DCT shape {dct.shape} must match RGB shape {rgb.shape}"
            
            # Forward
            logits = model(rgb, dct)
            
            # Loss
            loss = criterion(logits, target)
            
            # Accumulate
            total_loss += loss.item()
            all_outputs.append(logits)
            all_targets.append(target)
    
    # Compute metrics
    all_outputs = torch.cat(all_outputs, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    metrics = compute_metrics(all_outputs, all_targets, cfg["model"]["num_classes"] if cfg else 2)
    metrics["loss"] = total_loss / len(loader)
    
    # Log to tensorboard
    if writer is not None:
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                writer.add_scalar(f"Val/{key}", value, epoch)
        
        # Log c1, c2 if available
        if hasattr(model, 'get_dct_params'):
            c1, c2 = model.get_dct_params()
            writer.add_scalar("Val/DCT_c1", c1.item(), epoch)
            writer.add_scalar("Val/DCT_c2", c2.item(), epoch)
    
    result = metrics.copy()
    
    if return_predictions:
        probs = torch.softmax(all_outputs, dim=1)
        preds = all_outputs.argmax(dim=1)
        result["predictions"] = preds.cpu()
        result["probabilities"] = probs.cpu()
        result["targets"] = all_targets.cpu()
    
    return result


def compute_confusion_matrix(predictions: torch.Tensor, targets: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    Compute confusion matrix.
    
    Args:
        predictions: Predicted class indices [N]
        targets: True class indices [N]
        num_classes: Number of classes
        
    Returns:
        Confusion matrix [num_classes, num_classes] where cm[i, j] = count of class i predicted as class j
    """
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for i in range(num_classes):
        for j in range(num_classes):
            cm[i, j] = ((targets == i) & (predictions == j)).sum().item()
    return cm


def find_optimal_threshold(probabilities: torch.Tensor, targets: torch.Tensor, num_classes: int) -> Tuple[float, float]:
    """
    Find optimal threshold for binary classification by maximizing F1 score.
    
    Args:
        probabilities: Class probabilities [N, num_classes]
        targets: True class indices [N]
        num_classes: Number of classes (must be 2 for threshold finding)
        
    Returns:
        Tuple of (optimal_threshold, best_f1_score)
    """
    if num_classes != 2:
        # For multi-class, return default threshold of 0.5
        return 0.5, 0.0
    
    # Get probabilities for positive class (class 1)
    pos_probs = probabilities[:, 1].numpy()
    targets_np = targets.numpy()
    
    # Try different thresholds
    thresholds = np.linspace(0.0, 1.0, 101)
    best_threshold = 0.5
    best_f1 = 0.0
    
    for thresh in thresholds:
        preds = (pos_probs >= thresh).astype(int)
        
        # Compute F1 score
        tp = np.sum((preds == 1) & (targets_np == 1))
        fp = np.sum((preds == 1) & (targets_np == 0))
        fn = np.sum((preds == 0) & (targets_np == 1))
        
        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh
    
    return best_threshold, best_f1


def print_final_summary(
    metrics: Dict,
    predictions: torch.Tensor,
    targets: torch.Tensor,
    probabilities: torch.Tensor,
    num_classes: int,
    class_names: Optional[List[str]] = None,
):
    """
    Print comprehensive final summary including accuracy, precision, recall, F1, confusion matrix, and threshold.
    
    Args:
        metrics: Dictionary of computed metrics
        predictions: Predicted class indices [N]
        targets: True class indices [N]
        probabilities: Class probabilities [N, num_classes]
        num_classes: Number of classes
        class_names: Optional list of class names
    """
    if class_names is None:
        class_names = [f"Class {i}" for i in range(num_classes)]
    
    # Compute confusion matrix
    cm = compute_confusion_matrix(predictions, targets, num_classes)
    
    # Find optimal threshold (for binary classification)
    optimal_threshold, threshold_f1 = find_optimal_threshold(probabilities, targets, num_classes)
    
    console.print("\n" + "="*80)
    console.print("[bold cyan]FINAL TRAINING SUMMARY")
    console.print("="*80)
    
    # Overall metrics
    console.print("\n[bold yellow]Overall Metrics:")
    console.print(f"  Accuracy: {metrics['acc1']:.4f}%")
    console.print(f"  Precision (Macro): {metrics['precision_macro']:.4f}")
    console.print(f"  Recall (Macro): {metrics['recall_macro']:.4f}")
    console.print(f"  F1 Score (Macro): {metrics['f1_macro']:.4f}")
    console.print(f"  Precision (Micro): {metrics['precision_micro']:.4f}")
    console.print(f"  Recall (Micro): {metrics['recall_micro']:.4f}")
    console.print(f"  F1 Score (Micro): {metrics['f1_micro']:.4f}")
    
    # Per-class metrics
    console.print("\n[bold yellow]Per-Class Metrics:")
    precision_per_class = metrics.get('precision_per_class', [])
    recall_per_class = metrics.get('recall_per_class', [])
    f1_per_class = metrics.get('f1_per_class', [])
    
    for i in range(num_classes):
        console.print(f"\n  {class_names[i]}:")
        console.print(f"    Precision: {precision_per_class[i]:.4f}")
        console.print(f"    Recall: {recall_per_class[i]:.4f}")
        console.print(f"    F1 Score: {f1_per_class[i]:.4f}")
    
    # Confusion matrix
    console.print("\n[bold yellow]Confusion Matrix:")
    console.print("  Predicted →")
    
    # Header row
    header = "  Actual ↓"
    for j in range(num_classes):
        header += f"  {class_names[j]:>12}"
    console.print(header)
    console.print("  " + "-" * (13 * (num_classes + 1)))
    
    # Matrix rows
    for i in range(num_classes):
        row = f"  {class_names[i]:>10}"
        for j in range(num_classes):
            row += f"  {cm[i, j].item():>12}"
        console.print(row)
    
    # Threshold (for binary classification)
    if num_classes == 2:
        console.print("\n[bold yellow]Optimal Threshold:")
        console.print(f"  Threshold: {optimal_threshold:.4f}")
        console.print(f"  F1 Score at threshold: {threshold_f1:.4f}")
        console.print(f"  (Default threshold 0.5 gives F1: {metrics['f1_macro']:.4f})")
    else:
        console.print("\n[bold yellow]Threshold:")
        console.print("  (Threshold finding applies to binary classification only)")
        console.print(f"  Using default threshold: 0.5")
    
    console.print("\n" + "="*80 + "\n")


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
    
    # Get experiment mode
    mode = cfg.get("run", {}).get("mode", "rgbresnet_dctswin")
    
    train_dataset = FireDataset(
        root_dir=cfg["data"]["root"],
        split="train",
        img_size=cfg["data"]["img_size"],
        augment=True,
        mode=mode,
        dct_block=cfg["data"].get("dct_block", 8),
        use_gray_for_dct=cfg["data"].get("use_gray_for_dct", True),
        cfg=cfg,
    )
    val_dataset = FireDataset(
        root_dir=cfg["data"]["root"],
        split="val",
        img_size=cfg["data"]["img_size"],
        augment=False,
        class_to_idx=train_dataset.class_to_idx,  # Use same mapping
        mode=mode,
        dct_block=cfg["data"].get("dct_block", 8),
        use_gray_for_dct=cfg["data"].get("use_gray_for_dct", True),
        cfg=cfg,
    )
    
    # Print effective dataset lengths
    console.print(f"[cyan]Train dataset length: {len(train_dataset)}")
    console.print(f"[cyan]Val dataset length: {len(val_dataset)}")
    
    from data import collate_fn
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["data"]["batch_size"],
        shuffle=True,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["data"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
        collate_fn=collate_fn,
    )
    
    # Model
    from models import create_model_from_cfg
    
    num_classes = cfg["model"]["num_classes"]
    model = create_model_from_cfg(mode, cfg, num_classes).to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    console.print(f"[cyan]Total parameters: {total_params:,}")
    console.print(f"[cyan]Trainable parameters: {trainable_params:,}")
    
    # Loss
    label_smoothing = cfg["optim"].get("label_smoothing", 0.05)
    if cfg["optim"]["loss"] == "focal":
        criterion = FocalLoss(
            gamma=cfg["optim"]["focal_gamma"],
            alpha=cfg["optim"]["focal_alpha"],
        )
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["optim"]["lr"],
        weight_decay=cfg["optim"]["weight_decay"],
        betas=tuple(cfg["optim"]["betas"]),
    )
    
    # Scheduler
    warmup_epochs = cfg.get("train", {}).get("warmup_epochs", cfg["optim"].get("warmup_epochs", 5))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_epochs=warmup_epochs,
        num_training_epochs=cfg["optim"]["epochs"],
    )
    
    # Training loop
    best_metric = 0.0
    best_metric_name = cfg["logging"]["save_best_metric"]
    
    # Gradient accumulation steps
    grad_accum_steps = cfg.get("train", {}).get("grad_accum_steps", 1)
    
    for epoch in range(1, cfg["optim"]["epochs"] + 1):
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch,
            cfg["logging"]["log_interval"], writer, cfg, grad_accum_steps
        )
        
        # Validate
        val_metrics = validate_epoch(
            model, val_loader, criterion, device, epoch, writer, cfg
        )
        
        # Scheduler step
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # Get c1, c2 if available
        c1_str = ""
        c2_str = ""
        if hasattr(model, 'get_dct_params'):
            c1, c2 = model.get_dct_params()
            c1_str = f"c1: {c1.item():.3f} | "
            c2_str = f"c2: {c2.item():.3f} | "
        
        # Print epoch summary
        summary = (
            f"Epoch {epoch}/{cfg['optim']['epochs']} | "
            f"Train Loss: {train_metrics['loss']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Acc@1: {val_metrics['acc1']:.2f}% | "
            f"Val Acc@5: {val_metrics['acc5']:.2f}% | "
            f"Val F1-Macro: {val_metrics['f1_macro']:.4f} | "
            f"LR: {current_lr:.2e} | "
            f"{c1_str}{c2_str}"
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
                "mode": mode,
            }
            if hasattr(model, 'get_dct_params'):
                c1, c2 = model.get_dct_params()
                checkpoint["c1"] = c1.item()
                checkpoint["c2"] = c2.item()
            torch.save(checkpoint, output_dir / "best.pth")
            console.print(f"[green]Saved best checkpoint (metric: {best_metric:.4f})")
        
        # Save last checkpoint
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_metric": best_metric,
            "class_to_idx": train_dataset.class_to_idx,
            "mode": mode,
        }
        if hasattr(model, 'get_dct_params'):
            c1, c2 = model.get_dct_params()
            checkpoint["c1"] = c1.item()
            checkpoint["c2"] = c2.item()
        torch.save(checkpoint, output_dir / "last.pth")
    
    log_file.close()
    if writer is not None:
        writer.close()
    
    console.print(f"[green]Training completed! Best {best_metric_name}: {best_metric:.4f}")
    
    # Run final validation with predictions for comprehensive summary
    console.print("\n[cyan]Running final validation for comprehensive summary...")
    final_val_results = validate_epoch(
        model, val_loader, criterion, device, cfg["optim"]["epochs"], None, cfg, return_predictions=True
    )
    
    # Get class names from dataset
    class_names = [train_dataset.idx_to_class[i] for i in range(num_classes)]
    
    # Print final summary
    print_final_summary(
        metrics=final_val_results,
        predictions=final_val_results["predictions"],
        targets=final_val_results["targets"],
        probabilities=final_val_results["probabilities"],
        num_classes=num_classes,
        class_names=class_names,
    )
    
    # Also write summary to log file
    log_file = open(output_dir / "log.txt", "a")
    log_file.write("\n" + "="*80 + "\n")
    log_file.write("FINAL TRAINING SUMMARY\n")
    log_file.write("="*80 + "\n")
    log_file.write(f"Accuracy: {final_val_results['acc1']:.4f}%\n")
    log_file.write(f"Precision (Macro): {final_val_results['precision_macro']:.4f}\n")
    log_file.write(f"Recall (Macro): {final_val_results['recall_macro']:.4f}\n")
    log_file.write(f"F1 Score (Macro): {final_val_results['f1_macro']:.4f}\n")
    log_file.write(f"Precision (Micro): {final_val_results['precision_micro']:.4f}\n")
    log_file.write(f"Recall (Micro): {final_val_results['recall_micro']:.4f}\n")
    log_file.write(f"F1 Score (Micro): {final_val_results['f1_micro']:.4f}\n")
    
    # Confusion matrix
    cm = compute_confusion_matrix(
        final_val_results["predictions"], 
        final_val_results["targets"], 
        num_classes
    )
    log_file.write("\nConfusion Matrix:\n")
    log_file.write("  Predicted →\n")
    header = "  Actual ↓"
    for j in range(num_classes):
        header += f"  {class_names[j]:>12}"
    log_file.write(header + "\n")
    for i in range(num_classes):
        row = f"  {class_names[i]:>10}"
        for j in range(num_classes):
            row += f"  {cm[i, j].item():>12}"
        log_file.write(row + "\n")
    
    # Threshold
    if num_classes == 2:
        optimal_threshold, threshold_f1 = find_optimal_threshold(
            final_val_results["probabilities"],
            final_val_results["targets"],
            num_classes
        )
        log_file.write(f"\nOptimal Threshold: {optimal_threshold:.4f}\n")
        log_file.write(f"F1 Score at threshold: {threshold_f1:.4f}\n")
    
    log_file.close()


def validate(cfg: Dict):
    """Validation only."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print(f"[green]Using device: {device}")
    
    # Get experiment mode
    mode = cfg.get("run", {}).get("mode", "rgbresnet_dctswin")
    
    # Data
    from data import FireDataset
    
    val_dataset = FireDataset(
        root_dir=cfg["data"]["root"],
        split="val",
        img_size=cfg["data"]["img_size"],
        augment=False,
        mode=mode,
        dct_block=cfg["data"].get("dct_block", 8),
        use_gray_for_dct=cfg["data"].get("use_gray_for_dct", True),
        cfg=cfg,
    )
    
    from data import collate_fn
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["data"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
        pin_memory=True,
        collate_fn=collate_fn,
    )
    
    # Model
    from models import create_model_from_cfg
    
    num_classes = cfg["model"]["num_classes"]
    model = create_model_from_cfg(mode, cfg, num_classes).to(device)
    
    # Load checkpoint if available
    output_dir = Path(cfg["experiment"]["output_dir"])
    checkpoint_path = output_dir / "best.pth"
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
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

