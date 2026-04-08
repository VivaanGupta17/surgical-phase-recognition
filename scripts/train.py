#!/usr/bin/env python3
"""
Training entrypoint for SurgPhase surgical phase recognition.

Usage
-----
# Basic Cholec80 training with default config:
    python scripts/train.py \\
        --config configs/cholec80_config.yaml \\
        --data-root /data/cholec80 \\
        --output-dir runs/experiment_01

# Resume from checkpoint:
    python scripts/train.py \\
        --config configs/cholec80_config.yaml \\
        --resume runs/experiment_01/checkpoints/epoch_010.pth

# Override config values via CLI:
    python scripts/train.py \\
        --config configs/cholec80_config.yaml \\
        --data-root /data/cholec80 \\
        --backbone efficientnet_b4 \\
        --temporal-model transformer \\
        --batch-size 8 \\
        --lr 5e-5 \\
        --epochs 80
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

# Ensure src is importable when running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.cholec_dataset import Cholec80Dataset, build_dataloader
from src.data.video_preprocessing import build_eval_transforms, build_train_transforms
from src.models.phase_classifier import SurgPhaseClassifier
from src.training.train_phase import SurgPhaseTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("surgphase.train")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train SurgPhase surgical phase recognition model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Core arguments
    parser.add_argument("--config", type=str, default="configs/cholec80_config.yaml",
                        help="Path to YAML configuration file")
    parser.add_argument("--data-root", type=str, default=None,
                        help="Cholec80 dataset root directory")
    parser.add_argument("--feature-dir", type=str, default=None,
                        help="Pre-extracted feature directory (speeds up temporal training)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory for checkpoints and logs")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint path")

    # Model overrides
    parser.add_argument("--backbone", type=str, default=None,
                        choices=["resnet50", "efficientnet_b0", "efficientnet_b4"],
                        help="Spatial encoder backbone")
    parser.add_argument("--temporal-model", type=str, default=None,
                        choices=["mstcn", "transformer", "lstm"],
                        help="Temporal modeling architecture")
    parser.add_argument("--causal", action="store_true", default=None,
                        help="Use causal (online) temporal modeling")
    parser.add_argument("--no-instrument-fusion", action="store_true",
                        help="Disable instrument feature fusion")

    # Training overrides
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument("--epochs", type=int, default=None, help="Total training epochs")
    parser.add_argument("--lr", type=float, default=None, help="Base learning rate")
    parser.add_argument("--workers", type=int, default=None, help="DataLoader workers")

    # Misc
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default=None,
                        help="CUDA device (e.g. 'cuda:0'). Auto-selects if omitted.")
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run 2 batches per epoch for testing")

    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load YAML configuration file.

    Args:
        config_path: Path to .yaml config file.

    Returns:
        Configuration dict.
    """
    with open(config_path) as f:
        return yaml.safe_load(f)


def apply_cli_overrides(config: dict, args: argparse.Namespace) -> dict:
    """Apply command-line argument overrides to config dict.

    Args:
        config: Base configuration dict from YAML.
        args: Parsed CLI arguments.

    Returns:
        Updated configuration dict.
    """
    if args.data_root:
        config["dataset"]["root"] = args.data_root
    if args.feature_dir:
        config["dataset"]["feature_dir"] = args.feature_dir
    if args.output_dir:
        config["output_dir"] = args.output_dir
    if args.backbone:
        config["model"]["backbone"] = args.backbone
    if args.temporal_model:
        config["model"]["temporal_model"] = args.temporal_model
    if args.causal is not None:
        config["model"]["causal"] = args.causal
    if args.no_instrument_fusion:
        config["model"]["fuse_instruments"] = False
    if args.batch_size:
        config["batch_size"] = args.batch_size
    if args.lr:
        config["learning_rate"] = args.lr
    if args.workers is not None:
        config["num_workers"] = args.workers
    if args.seed:
        config["seed"] = args.seed
    return config


def set_seed(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main() -> None:
    args = parse_args()

    # Load and merge config
    if not Path(args.config).exists():
        logger.error("Config file not found: %s", args.config)
        sys.exit(1)
    config = load_config(args.config)
    config = apply_cli_overrides(config, args)

    # Reproducibility
    set_seed(config.get("seed", 42))

    # Device
    if not torch.cuda.is_available():
        logger.warning("no CUDA found — falling back to CPU, this will be slow")
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("CUDA device: %s", torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        logger.warning("CUDA not available — training on CPU (slow).")
    logger.info("Using device: %s", device)

    # Output directory
    output_dir = Path(config.get("output_dir", "runs/")) / config.get("run_name", "experiment")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save effective config
    effective_config_path = output_dir / "config.yaml"
    with open(effective_config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    logger.info("Saved effective config to %s", effective_config_path)

    # -----------------------------------------------------------------------
    # Build datasets
    # -----------------------------------------------------------------------
    dataset_cfg = config["dataset"]
    img_size = config.get("augmentation", {}).get("img_size", 256)
    use_surgical_aug = config.get("augmentation", {}).get("use_surgical_aug", True)

    train_dataset = Cholec80Dataset(
        root=dataset_cfg["root"],
        split="train",
        feature_dir=dataset_cfg.get("feature_dir"),
        temporal_window=dataset_cfg.get("temporal_window", 64),
        stride=dataset_cfg.get("stride", 1),
        transform=build_train_transforms(img_size, use_surgical_aug),
        return_instruments=dataset_cfg.get("return_instruments", True),
        video_ids=dataset_cfg.get("split", {}).get("train_videos"),
    )

    val_dataset = Cholec80Dataset(
        root=dataset_cfg["root"],
        split="test",
        feature_dir=dataset_cfg.get("feature_dir"),
        temporal_window=dataset_cfg.get("temporal_window", 64),
        stride=dataset_cfg.get("stride", 1),
        transform=build_eval_transforms(img_size),
        return_instruments=dataset_cfg.get("return_instruments", True),
        video_ids=dataset_cfg.get("split", {}).get("test_videos"),
    )

    train_loader = build_dataloader(
        train_dataset,
        batch_size=config.get("batch_size", 4),
        num_workers=config.get("num_workers", 4),
        pin_memory=config.get("pin_memory", True),
        shuffle=True,
        balanced_sampling=dataset_cfg.get("balanced_sampling", True),
    )

    val_loader = build_dataloader(
        val_dataset,
        batch_size=config.get("batch_size", 4),
        num_workers=config.get("num_workers", 4),
        pin_memory=config.get("pin_memory", True),
        shuffle=False,
    )

    logger.info(
        "Datasets: train=%d windows, val=%d windows",
        len(train_dataset),
        len(val_dataset),
    )

    # -----------------------------------------------------------------------
    # Build model
    # -----------------------------------------------------------------------
    model_cfg = config["model"]
    model = SurgPhaseClassifier(
        backbone=model_cfg.get("backbone", "resnet50"),
        feature_dim=model_cfg.get("feature_dim", 2048),
        temporal_model=model_cfg.get("temporal_model", "mstcn"),
        num_phases=model_cfg.get("num_phases", 7),
        num_instruments=model_cfg.get("num_instruments", 7),
        causal=model_cfg.get("causal", True),
        use_crf=model_cfg.get("use_crf", True),
        fuse_instruments=model_cfg.get("fuse_instruments", True),
        pretrained_backbone=model_cfg.get("pretrained_backbone", True),
        tcn_stages=model_cfg.get("tcn_stages", 4),
        tcn_layers=model_cfg.get("tcn_layers", 10),
        tcn_filters=model_cfg.get("tcn_filters", 64),
        transformer_d_model=model_cfg.get("transformer_d_model", 256),
        transformer_heads=model_cfg.get("transformer_heads", 8),
        transformer_layers=model_cfg.get("transformer_layers", 6),
    )

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info("Model parameters: %.2fM", total_params)

    # -----------------------------------------------------------------------
    # Build trainer and run
    # -----------------------------------------------------------------------
    trainer = SurgPhaseTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        output_dir=output_dir,
        device=device,
        use_wandb=args.wandb,
    )

    if args.resume:
        trainer.load_checkpoint(args.resume)

    try:
        trainer.train()
    except KeyboardInterrupt:
        logger.info("Training interrupted by user.")

    logger.info("Training complete. Outputs saved to %s", output_dir)


if __name__ == "__main__":
    main()
