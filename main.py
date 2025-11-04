"""
Main entry point: argument parsing, config loading, seeding, and launcher.
"""
import argparse
import os
import shutil
import random
import numpy as np
import torch
import yaml
from pathlib import Path

from data import build_dataloaders
from models import make_model
from engine import train, validate
from torch.cuda.amp import autocast


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def load_config(config_path: str) -> dict:
    """Load and validate config YAML."""
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Set defaults if missing
    cfg.setdefault('experiment_name', 'swin_dct_compare')
    cfg.setdefault('mode', 'auto')
    cfg.setdefault('seed', 1337)
    
    cfg.setdefault('data', {})
    cfg['data'].setdefault('root', './data')
    cfg['data'].setdefault('img_size', 224)
    cfg['data'].setdefault('mean', [0.485, 0.456, 0.406])
    cfg['data'].setdefault('std', [0.229, 0.224, 0.225])
    cfg['data'].setdefault('augment', {})
    
    cfg.setdefault('training', {})
    cfg['training'].setdefault('epochs', 30)
    cfg['training'].setdefault('batch_size', 32)
    cfg['training'].setdefault('optimizer', 'adamw')
    cfg['training'].setdefault('lr', 3e-4)
    cfg['training'].setdefault('weight_decay', 0.05)
    cfg['training'].setdefault('warmup_epochs', 2)
    cfg['training'].setdefault('label_smoothing', 0.0)
    cfg['training'].setdefault('amp', True)
    cfg['training'].setdefault('num_workers', 4)
    cfg['training'].setdefault('grad_clip', 1.0)
    cfg['training'].setdefault('early_stop_patience', 8)
    
    cfg.setdefault('model', {})
    cfg['model'].setdefault('backbone', 'swin_tiny_patch4_window7_224')
    cfg['model'].setdefault('pretrained', True)
    cfg['model'].setdefault('dropout', 0.0)
    
    cfg.setdefault('dct', {})
    cfg['dct'].setdefault('block', 8)
    cfg['dct'].setdefault('P', 5)
    cfg['dct'].setdefault('selection', 'topk')
    cfg['dct'].setdefault('bands', {
        'low': [[0,0],[0,1],[1,0],[1,1],[0,2],[2,0]],
        'mid': [[1,2],[2,1],[2,2],[0,3],[3,0],[1,3],[3,1]],
        'high': 'else'
    })
    
    cfg.setdefault('fusion', {})
    cfg['fusion'].setdefault('reduction', 4)
    cfg['fusion'].setdefault('cross_attn', {})
    cfg['fusion']['cross_attn'].setdefault('num_heads', 4)
    cfg['fusion']['cross_attn'].setdefault('bidirectional', False)
    cfg['fusion']['cross_attn'].setdefault('drop', 0.0)
    cfg['fusion'].setdefault('late', {})
    cfg['fusion']['late'].setdefault('type', 'concat_mlp')
    cfg['fusion']['late'].setdefault('mlp_hidden', 512)
    
    return cfg


def run_mode(mode: str, cfg: dict, device: torch.device) -> dict:
    """Train and evaluate a single mode."""
    print(f"\n{'='*60}")
    print(f"Training mode: {mode}")
    print(f"{'='*60}\n")
    
    # Determine if DCT is needed
    need_dct = mode in ['dct_gate', 'cross_attn', 'late_fusion']
    
    # Build dataloaders
    data_cfg = cfg['data']
    train_cfg = cfg['training']
    
    train_loader, val_loader = build_dataloaders(
        root=data_cfg['root'],
        img_size=data_cfg['img_size'],
        mean=tuple(data_cfg['mean']),
        std=tuple(data_cfg['std']),
        augment_cfg=data_cfg.get('augment'),
        batch_size=train_cfg['batch_size'],
        num_workers=train_cfg['num_workers'],
        need_dct=need_dct
    )
    
    # Create model
    model = make_model(mode, cfg)
    model = model.to(device)
    
    # Create save directory
    exp_name = cfg['experiment_name']
    save_dir = Path('runs') / exp_name / mode
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config copy
    config_copy_path = save_dir / 'config.yaml'
    with open(config_copy_path, 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False)
    
    # Train
    bands_cfg = cfg.get('dct', {}).get('bands', {})
    metrics = train(
        model, train_loader, val_loader, cfg, device,
        str(save_dir), need_dct, bands_cfg
    )
    
    # Final evaluation on validation set
    from engine import validate
    from torch.nn import CrossEntropyLoss
    
    criterion = CrossEntropyLoss()
    final_loss, final_metrics = validate(
        model, val_loader, criterion, device,
        train_cfg.get('amp', True), need_dct, bands_cfg
    )
    
    print(f"\nFinal metrics for {mode}:")
    print(f"  Loss: {final_loss:.4f}")
    print(f"  Acc:  {final_metrics['acc']:.4f}")
    print(f"  Prec: {final_metrics['prec']:.4f}")
    print(f"  Rec:  {final_metrics['rec']:.4f}")
    print(f"  F1:   {final_metrics['f1']:.4f}")
    if 'auc' in final_metrics:
        print(f"  AUC:  {final_metrics['auc']:.4f}")
    
    return final_metrics


def main():
    parser = argparse.ArgumentParser(description='Fire Classification with DCT Modes')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to config YAML file')
    args = parser.parse_args()
    
    # Load config
    cfg = load_config(args.config)
    
    # Set seed
    set_seed(cfg['seed'])
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Determine modes to run
    mode = cfg['mode']
    if mode == 'auto':
        modes = ['rgb_only', 'dct_gate', 'cross_attn', 'late_fusion']
    else:
        modes = [mode]
    
    # Run each mode
    all_results = {}
    for m in modes:
        try:
            metrics = run_mode(m, cfg, device)
            all_results[m] = metrics
        except Exception as e:
            print(f"Error training {m}: {e}")
            import traceback
            traceback.print_exc()
            all_results[m] = None
    
    # Print summary table if auto mode
    if mode == 'auto':
        print(f"\n{'='*80}")
        print("SUMMARY TABLE")
        print(f"{'='*80}")
        print(f"{'MODE':<15} {'ACC':<8} {'PREC':<8} {'REC':<8} {'F1':<8} {'AUC':<8}")
        print("-" * 80)
        
        best_f1 = 0.0
        best_mode = None
        
        for m in modes:
            if all_results[m] is not None:
                metrics = all_results[m]
                print(f"{m:<15} {metrics['acc']:<8.4f} {metrics['prec']:<8.4f} "
                      f"{metrics['rec']:<8.4f} {metrics['f1']:<8.4f} "
                      f"{metrics.get('auc', 0.0):<8.4f}")
                
                if metrics['f1'] > best_f1:
                    best_f1 = metrics['f1']
                    best_mode = m
        
        print("-" * 80)
        if best_mode:
            print(f"BEST = {best_mode} by F1 ({best_f1:.4f})")
            print(f"\nBest checkpoint saved at: runs/{cfg['experiment_name']}/{best_mode}/best.pt")
        print(f"{'='*80}\n")


if __name__ == '__main__':
    main()

