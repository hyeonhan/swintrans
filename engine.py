"""
Training and evaluation engine with metrics, checkpointing, and CSV logging.
"""
import os
import csv
import inspect
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from tqdm import tqdm
from typing import Dict, Optional, Tuple
import numpy as np


def safe_torch_load(path, device):
    """Safely load checkpoint with weights_only if available, fallback otherwise."""
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Compute classification metrics."""
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    
    metrics = {
        'acc': acc,
        'prec': prec,
        'rec': rec,
        'f1': f1
    }
    
    # ROC-AUC if probabilities available
    if y_proba is not None:
        try:
            auc = roc_auc_score(y_true, y_proba)
            metrics['auc'] = auc
        except:
            metrics['auc'] = 0.0
    
    return metrics


def train_epoch(
    model: nn.Module,
    train_loader,
    optimizer,
    criterion,
    device: torch.device,
    scaler: Optional[GradScaler] = None,
    amp: bool = True,
    grad_clip: Optional[float] = None,
    label_smoothing: float = 0.0,
    need_dct: bool = False,
    bands_cfg: Optional[Dict] = None
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(train_loader, desc='Train')
    for batch in pbar:
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        
        with autocast(enabled=amp):
            if need_dct:
                dct_gray = batch.get('dct_gray', None)
                if dct_gray is not None:
                    dct_gray = dct_gray.to(device)
                    # Try to call with dct_gray if model supports it
                    sig = inspect.signature(model.forward)
                    params = list(sig.parameters.keys())
                    if len(params) >= 2 and 'dct_gray' in params:
                        if len(params) >= 3 and 'bands_cfg' in params:
                            logits = model(images, dct_gray, bands_cfg)
                        else:
                            logits = model(images, dct_gray)
                    else:
                        logits = model(images)
                else:
                    logits = model(images)
            else:
                logits = model(images)
            
            # Label smoothing
            if label_smoothing > 0:
                labels_onehot = F.one_hot(labels, num_classes=2).float()
                labels_smooth = (1 - label_smoothing) * labels_onehot + label_smoothing / 2
                loss = -torch.sum(labels_smooth * F.log_softmax(logits, dim=1), dim=1).mean()
            else:
                loss = criterion(logits, labels)
        
        if scaler is not None:
            scaler.scale(loss).backward()
            if grad_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        pbar.set_postfix({'loss': loss.item()})
    
    return total_loss / num_batches


def validate(
    model: nn.Module,
    val_loader,
    criterion,
    device: torch.device,
    amp: bool = True,
    need_dct: bool = False,
    bands_cfg: Optional[Dict] = None
) -> Tuple[float, Dict[str, float]]:
    """Validate and compute metrics."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_probas = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Val'):
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            with autocast(enabled=amp):
                if need_dct:
                    dct_gray = batch.get('dct_gray', None)
                    if dct_gray is not None:
                        dct_gray = dct_gray.to(device)
                        # Try to call with dct_gray if model supports it
                        sig = inspect.signature(model.forward)
                        params = list(sig.parameters.keys())
                        if len(params) >= 2 and 'dct_gray' in params:
                            if len(params) >= 3 and 'bands_cfg' in params:
                                logits = model(images, dct_gray, bands_cfg)
                            else:
                                logits = model(images, dct_gray)
                        else:
                            logits = model(images)
                    else:
                        logits = model(images)
                else:
                    logits = model(images)
                
                loss = criterion(logits, labels)
            
            total_loss += loss.item()
            
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probas.extend(probs[:, 1].cpu().numpy())  # Positive class probability
    
    avg_loss = total_loss / len(val_loader)
    metrics = compute_metrics(np.array(all_labels), np.array(all_preds), np.array(all_probas))
    
    return avg_loss, metrics


def train(
    model: nn.Module,
    train_loader,
    val_loader,
    cfg: Dict,
    device: torch.device,
    save_dir: str,
    need_dct: bool = False,
    bands_cfg: Optional[Dict] = None
) -> Dict[str, float]:
    """Main training loop."""
    train_cfg = cfg.get('training', {})
    
    # Optimizer
    optimizer_name = train_cfg.get('optimizer', 'adamw').lower()
    lr = train_cfg.get('lr', 3e-4)
    weight_decay = train_cfg.get('weight_decay', 0.05)
    
    if optimizer_name == 'adamw':
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    
    # Scheduler
    epochs = train_cfg.get('epochs', 30)
    warmup_epochs = train_cfg.get('warmup_epochs', 2)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs)
    
    # Criterion
    criterion = nn.CrossEntropyLoss()
    
    # AMP
    amp = train_cfg.get('amp', True)
    scaler = GradScaler() if amp else None
    
    # Training state
    best_f1 = 0.0
    best_epoch = 0
    patience = train_cfg.get('early_stop_patience', 8)
    patience_counter = 0
    
    # CSV logging
    csv_path = os.path.join(save_dir, 'metrics.csv')
    csv_file = open(csv_path, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['epoch', 'train_loss', 'val_loss', 'acc', 'prec', 'rec', 'f1', 'auc'])
    
    # Training loop
    for epoch in range(epochs):
        # Warmup
        if epoch < warmup_epochs:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr * (epoch + 1) / warmup_epochs
        
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device,
            scaler, amp, train_cfg.get('grad_clip'), train_cfg.get('label_smoothing', 0.0),
            need_dct, bands_cfg
        )
        
        # Validate
        val_loss, metrics = validate(model, val_loader, criterion, device, amp, need_dct, bands_cfg)
        
        # Step scheduler (after warmup)
        if epoch >= warmup_epochs:
            scheduler.step()
        
        # Log
        csv_writer.writerow([
            epoch + 1,
            train_loss,
            val_loss,
            metrics['acc'],
            metrics['prec'],
            metrics['rec'],
            metrics['f1'],
            metrics.get('auc', 0.0)
        ])
        csv_file.flush()
        
        print(f"Epoch {epoch+1}/{epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
              f"acc={metrics['acc']:.4f}, prec={metrics['prec']:.4f}, rec={metrics['rec']:.4f}, "
              f"f1={metrics['f1']:.4f}, auc={metrics.get('auc', 0.0):.4f}")
        
        # Save best
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            best_epoch = epoch + 1
            patience_counter = 0
            
            # Save best checkpoint
            best_path = os.path.join(save_dir, 'best.pt')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'f1': best_f1,
                'metrics': metrics
            }, best_path)
        else:
            patience_counter += 1
        
        # Save last checkpoint
        last_path = os.path.join(save_dir, 'last.pt')
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'f1': metrics['f1'],
            'metrics': metrics
        }, last_path)
        
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1} (patience={patience})")
            break
    
    csv_file.close()
    
    # Load best model
    best_path = os.path.join(save_dir, 'best.pt')
    if os.path.exists(best_path):
        checkpoint = safe_torch_load(best_path, device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded best model from epoch {best_epoch} with F1={best_f1:.4f}")
    
    return metrics

