"""
Main entry point for training, validation, and inference.

Usage:
    # Train:
    python main.py --config config.yaml
    
    # Validate:
    python main.py --config config.yaml --mode validate
    
    # Override config:
    python main.py --config config.yaml --model.fusion=channel_gate --model.swin_weight_sharing=false --optim.epochs=50
    
    # Infer single image:
    python main.py --config config.yaml --mode infer_single --image path/to/img.jpg
"""

import argparse
import yaml
from pathlib import Path
from typing import Dict, Any, List
import torch
from torchvision import transforms
from PIL import Image
import sys

from rich.console import Console
from rich.table import Table

console = Console()

# Disable decompression bomb warning
Image.MAX_IMAGE_PIXELS = None


def parse_cli_overrides(args_list: List[str]) -> Dict[str, Any]:
    """
    Parse CLI overrides in format --key.subkey=value.
    
    Args:
        args_list: List of override strings like ["model.fusion=channel_gate"]
        
    Returns:
        Nested dictionary of overrides
    """
    overrides = {}
    for arg in args_list:
        if not arg.startswith("--"):
            continue
        arg = arg[2:]  # Remove "--"
        if "=" not in arg:
            continue
        key_path, value = arg.split("=", 1)
        keys = key_path.split(".")
        
        # Parse value (try int, float, bool, then string)
        try:
            if value.lower() == "true":
                parsed_value = True
            elif value.lower() == "false":
                parsed_value = False
            elif "e" in value.lower() or "E" in value:  # Scientific notation
                parsed_value = float(value)
            elif "." in value:
                parsed_value = float(value)
            else:
                parsed_value = int(value)
        except ValueError:
            parsed_value = value
        
        # Build nested dict
        current = overrides
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[keys[-1]] = parsed_value
    
    return overrides


def merge_dict(base: Dict, override: Dict) -> Dict:
    """Recursively merge override dict into base dict."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dict(result[key], value)
        else:
            result[key] = value
    return result


def print_config_table(cfg: Dict):
    """Print configuration in a nice table format."""
    table = Table(title="Configuration", show_header=True, header_style="bold magenta")
    table.add_column("Section", style="cyan")
    table.add_column("Key", style="yellow")
    table.add_column("Value", style="green")
    
    for section, values in cfg.items():
        if isinstance(values, dict):
            for key, value in values.items():
                table.add_row(section, key, str(value))
        else:
            table.add_row("", section, str(values))
    
    console.print(table)


def infer_single(cfg: Dict, image_path: str):
    """Run inference on a single image."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print(f"[green]Using device: {device}")
    
    # Load model
    from models import MultiBranchClassifier
    
    model = MultiBranchClassifier(
        feature_dim=cfg["model"]["feature_dim"],
        num_classes=cfg["model"]["num_classes"],
        swin_weight_sharing=cfg["model"]["swin_weight_sharing"],
        fusion=cfg["model"]["fusion"],
        branch_dropout=0.0,
        freeze_backbone_epochs=0,
        dct_c1_init=cfg["dct"]["c1_init"],
        dct_c2_init=cfg["dct"]["c2_init"],
        dct_k=cfg["dct"]["k"],
    ).to(device)
    
    # Load checkpoint
    output_dir = Path(cfg["experiment"]["output_dir"])
    checkpoint_path = output_dir / "best.pth"
    if not checkpoint_path.exists():
        checkpoint_path = output_dir / "last.pth"
    
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        class_to_idx = checkpoint.get("class_to_idx", None)
        console.print(f"[green]Loaded checkpoint from {checkpoint_path}")
    else:
        console.print("[yellow]No checkpoint found, using random initialization")
        class_to_idx = None
    
    # Build class_to_idx from data folder if not available
    if class_to_idx is None:
        data_root = Path(cfg["data"]["root"])
        # Try to read from train split first, then val
        for split in ["train", "val"]:
            split_dir = data_root / split
            if split_dir.exists():
                class_dirs = sorted([d.name for d in split_dir.iterdir() if d.is_dir()])
                if len(class_dirs) > 0:
                    class_to_idx = {name: idx for idx, name in enumerate(class_dirs)}
                    console.print(f"[cyan]Built class mapping from {split_dir}: {class_to_idx}")
                    break
        
        # Fallback to default if data folder doesn't exist
        if class_to_idx is None:
            class_to_idx = {"class0": 0, "class1": 1}
            console.print(f"[yellow]Using default class mapping: {class_to_idx}")
    
    idx_to_class = {idx: name for name, idx in class_to_idx.items()}
    
    # Load and preprocess image
    img = Image.open(image_path).convert("RGB")
    
    # RGB transform
    rgb_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(cfg["data"]["img_size"]),
        transforms.ToTensor(),
    ])
    rgb_normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    rgb = rgb_transform(img)  # [3, H, W]
    rgb_normalized = rgb_normalize(rgb).unsqueeze(0).to(device)  # [1, 3, H, W]
    
    # Grayscale for DCT
    gray = (0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]).unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, H, W]
    
    # Compute initial bands (will be recomputed in model with learnable c1, c2)
    from dct_utils import create_dct_basis, band_split_idct
    D = create_dct_basis(N=8).to(device)
    c1_init = torch.tensor(2.0, device=device)
    c2_init = torch.tensor(4.0, device=device)
    band_low, band_mid, band_high = band_split_idct(gray, c1_init, c2_init, D, k=50.0)
    band_low = band_low.squeeze(0)  # [1, H, W]
    band_mid = band_mid.squeeze(0)
    band_high = band_high.squeeze(0)
    
    # Normalize bands
    for band in [band_low, band_mid, band_high]:
        band_mean = band.mean()
        band_std = band.std() + 1e-6
        band.sub_(band_mean).div_(band_std)
    
    # Inference
    model.eval()
    with torch.no_grad():
        logits, alpha = model(
            rgb_normalized,
            band_low.unsqueeze(0),
            band_mid.unsqueeze(0),
            band_high.unsqueeze(0),
            gray=gray.squeeze(0),
        )
        
        probs = torch.softmax(logits, dim=1)
        pred_class_idx = logits.argmax(dim=1).item()
        pred_class_name = idx_to_class.get(pred_class_idx, f"class{pred_class_idx}")
        confidence = probs[0, pred_class_idx].item()
    
    # Get c1, c2
    c1, c2 = model.get_dct_params()
    
    # Print results
    console.print("\n[cyan]Inference Results:")
    console.print(f"  Image: {image_path}")
    console.print(f"  Predicted class: {pred_class_name} (confidence: {confidence:.4f})")
    console.print(f"  Probabilities: {probs[0].cpu().tolist()}")
    console.print(f"  Alpha weights: {alpha[0].cpu().tolist()}")
    console.print(f"  c1: {c1.item():.3f}, c2: {c2.item():.3f}")
    console.print()


def main():
    parser = argparse.ArgumentParser(description="Multi-branch fire classification")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML file")
    parser.add_argument("--mode", type=str, default=None, 
                       choices=["train", "validate", "infer_single", "rgb_resnet", "rgb_swin", "dct_swin", "rgbswin_dctswin", "rgbresnet_dctswin", "swin_fpn"],
                       help="Mode: train/validate/infer_single (task) or rgb_resnet/rgb_swin/dct_swin/rgbswin_dctswin/rgbresnet_dctswin/swin_fpn (experiment mode)")
    parser.add_argument("--image", type=str, help="Path to image for inference (required for infer_single mode)")
    parser.add_argument("--override", nargs="*", default=[], help="Config overrides in format --key.subkey=value")
    
    args, unknown = parser.parse_known_args()
    
    # Parse unknown args as overrides
    overrides = parse_cli_overrides(unknown)
    
    # Also parse --override args
    if args.override:
        overrides.update(parse_cli_overrides([f"--{o}" for o in args.override]))
    
    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        console.print(f"[red]Config file not found: {config_path}")
        sys.exit(1)
    
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    
    # Convert scientific notation strings to floats
    # YAML sometimes parses scientific notation as strings (e.g., "3e-4")
    def convert_scientific_notation(obj):
        """Recursively convert string scientific notation to floats."""
        if isinstance(obj, dict):
            return {k: convert_scientific_notation(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_scientific_notation(item) for item in obj]
        elif isinstance(obj, str):
            # Try to parse as float (handles scientific notation like "3e-4")
            try:
                return float(obj)
            except (ValueError, TypeError):
                return obj
        else:
            return obj
    
    cfg = convert_scientific_notation(cfg)
    
    # Handle mode argument: can be task mode (train/validate/infer_single) or experiment mode
    task_mode = "train"  # default task
    experiment_mode = None
    
    if args.mode:
        if args.mode in ["train", "validate", "infer_single"]:
            task_mode = args.mode
            # Use experiment mode from config if not overridden
            experiment_mode = cfg.get("run", {}).get("mode", "rgbresnet_dctswin")
        else:
            # It's an experiment mode
            experiment_mode = args.mode
            task_mode = "train"  # default to train when experiment mode is specified
    
    # Set experiment mode
    if experiment_mode:
        if "run" not in cfg:
            cfg["run"] = {}
        cfg["run"]["mode"] = experiment_mode
    
    # Resolve final experiment mode
    final_mode = cfg.get("run", {}).get("mode", "rgbresnet_dctswin")
    
    # Update experiment name to include mode
    base_name = cfg.get("experiment", {}).get("name", "baselines_4modes")
    cfg["experiment"]["name"] = f"{base_name}-{final_mode}"
    
    # Set output_dir if not already set
    if "output_dir" not in cfg.get("experiment", {}):
        cfg["experiment"]["output_dir"] = f"./runs/{cfg['experiment']['name']}"
    
    # Apply overrides
    if overrides:
        cfg = merge_dict(cfg, overrides)
        console.print("[yellow]Applied config overrides:")
        for key_path, value in overrides.items():
            console.print(f"  {key_path}: {value}")
    
    # Print config
    print_config_table(cfg)
    
    # Dataset stats
    from data import compute_dataset_stats
    compute_dataset_stats(cfg["data"]["root"])
    
    # Run mode
    if task_mode == "train":
        from engine import train
        train(cfg)
    elif task_mode == "validate":
        from engine import validate
        validate(cfg)
    elif task_mode == "infer_single":
        if not args.image:
            console.print("[red]--image is required for infer_single mode")
            sys.exit(1)
        infer_single(cfg, args.image)
    else:
        console.print(f"[red]Unknown task mode: {task_mode}")
        sys.exit(1)


if __name__ == "__main__":
    main()

