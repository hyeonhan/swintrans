"""Main entry for training/evaluation of Swin + DCT dual-branch with late fusion."""
from __future__ import annotations

import argparse
import json
import os
import sys
from copy import deepcopy
from typing import Any, Dict, Optional

import torch

import data as data_module
import models
import trainer as trainer_module
import utils


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Swin + DCT dual-branch trainer")
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml")
    parser.add_argument("--eval-only", action="store_true", help="Run evaluation only")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume")
    parser.add_argument("--smoke-all", action="store_true", help="Run all DCT×Fusion cases for a smoke test.")
    parser.add_argument("--smoke-epochs", type=int, default=1, help="Epochs per case when using --smoke-all (default: 1).")
    parser.add_argument("--smoke-warmup-epochs", type=int, default=0, help="Warmup epochs per case when using --smoke-all (default: 0).")
    # Accept free-form overrides after known args
    args, overrides = parser.parse_known_args()
    args.overrides = overrides
    return args


def run_once(cfg: Dict[str, Any], eval_only: bool = False, resume: Optional[str] = None) -> None:
    """Run a single training/evaluation session with the provided config."""
    # Setup run directory and seed
    out_dir = os.path.abspath(cfg["experiment"]["output_dir"])
    os.makedirs(out_dir, exist_ok=True)
    utils.set_seed(int(cfg["experiment"].get("seed", 42)))

    # Data
    dm = data_module.DataModule.from_config(cfg)
    train_loader, val_loader, test_loader = dm.loaders()

    # Model
    num_classes = len(dm.idx_to_class)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # GPU detection and logging
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
        print(f"[GPU] Using GPU: {gpu_name} ({gpu_memory:.2f} GB)")
        print(f"[GPU] Device count: {gpu_count}, Using device: {device}")
        if gpu_count > 1:
            print(f"[GPU] Note: Multiple GPUs detected but using single GPU mode")
    else:
        print("[GPU] CUDA not available, using CPU")
    
    model = models.SwinDCTSystem(cfg, num_classes=num_classes)
    model.to(device)

    # Trainer
    tr = trainer_module.Trainer(cfg, model, num_classes=num_classes, out_dir=out_dir)

    # Save config copy and class map
    utils.save_config_copy(cfg, out_dir)
    with open(os.path.join(out_dir, "classes.json"), "w", encoding="utf-8") as f:
        json.dump(dm.idx_to_class, f, ensure_ascii=False, indent=2)

    print(utils.pretty_print_experiment_summary(cfg, dm.class_to_idx, {
        "train": len(dm.train) if dm.train is not None else 0,
        "val": len(dm.val) if dm.val is not None else 0,
        "test": len(dm.test) if dm.test is not None else 0,
    }))

    # Resume
    if resume is not None and os.path.isfile(resume):
        print(f"Resuming from checkpoint: {resume}")
        ckpt = torch.load(resume, map_location="cpu")
        model.load_state_dict(ckpt["model"], strict=False)

    if eval_only:
        if val_loader is not None:
            val_metrics = tr.evaluate(val_loader)
            print("Validation:", val_metrics)
        if test_loader is not None:
            test_metrics = tr.evaluate(test_loader)
            print("Test:", test_metrics)
        return

    # Train
    tr.fit(train_loader, val_loader)

    # Evaluate best checkpoint if exists
    best_ckpt = os.path.join(out_dir, "best.pt")
    if os.path.isfile(best_ckpt):
        ckpt = torch.load(best_ckpt, map_location="cpu")
        model.load_state_dict(ckpt["model"], strict=False)
        if val_loader is not None:
            val_metrics = tr.evaluate(val_loader)
            print("Best Validation:", val_metrics)
        if test_loader is not None:
            test_metrics = tr.evaluate(test_loader)
            print("Best Test:", test_metrics)


def main() -> None:
    args = parse_args()
    cfg = utils.load_config(args.config)
    cfg = utils.merge_overrides(cfg, args.overrides)

    if args.smoke_all:
        dct_modes = ["freq_cnn", "idct_cnn"]
        fusions = [
            "rgb_only",
            "dct_only",
            "late_fusion_avg",
            "late_fusion_alpha_scalar",
            "late_fusion_alpha_classwise",
            "late_fusion_dynamic",
        ]
        base_out = cfg["experiment"]["output_dir"]
        for dm_mode in dct_modes:
            for fu in fusions:
                cfg_case = deepcopy(cfg)
                # override epochs/warmup for smoke
                cfg_case["optim"]["epochs"] = max(1, int(args.smoke_epochs))
                cfg_case["optim"]["warmup_epochs"] = max(0, int(args.smoke_warmup_epochs))
                # speed up logging for smoke
                cfg_case["log"]["use_tensorboard"] = False
                cfg_case["log"]["log_interval"] = max(100, cfg["log"].get("log_interval", 50))
                if "eval" in cfg_case and isinstance(cfg_case["eval"], dict):
                    cfg_case["eval"]["compute_auc"] = False
                # modes
                cfg_case["models"]["dct"]["mode"] = dm_mode
                cfg_case["experiment"]["type"] = fu
                # output directory per case
                cfg_case["experiment"]["output_dir"] = f"{base_out}/smoke_{dm_mode}_{fu}_e{cfg_case['optim']['epochs']}_w{cfg_case['optim']['warmup_epochs']}"
                print(f"[SMOKE] dct={dm_mode} fusion={fu} epochs={cfg_case['optim']['epochs']} warmup={cfg_case['optim']['warmup_epochs']}")
                run_once(cfg_case, eval_only=False, resume=None)
        print("[SMOKE] All cases finished.")
        return

    run_once(cfg, eval_only=args.eval_only, resume=args.resume)


if __name__ == "__main__":
    main()


