#!/usr/bin/env python3
"""
Evaluation entrypoint for SurgPhase surgical phase recognition.

Loads a trained checkpoint, runs inference on the Cholec80 test set,
and produces a comprehensive surgical metrics report.

Usage
-----
    # Full test set evaluation:
    python scripts/evaluate.py \\
        --checkpoint runs/experiment_01/checkpoints/best_model.pth \\
        --data-root /data/cholec80 \\
        --report-dir reports/experiment_01

    # Evaluate a single video:
    python scripts/evaluate.py \\
        --checkpoint runs/experiment_01/checkpoints/best_model.pth \\
        --video /data/cholec80/videos/video41.mp4 \\
        --annotation /data/cholec80/phase_annotations/video41-phase.txt
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.cholec_dataset import Cholec80Dataset, build_dataloader
from src.data.video_preprocessing import build_eval_transforms
from src.evaluation.surgical_metrics import SurgicalMetrics, compute_all_metrics
from src.models.phase_classifier import SurgPhaseClassifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("surgphase.evaluate")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate SurgPhase on the Cholec80 test set",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained model checkpoint (.pth)")
    parser.add_argument("--config", type=str, default=None,
                        help="Config YAML (read from checkpoint if omitted)")
    parser.add_argument("--data-root", type=str, required=True,
                        help="Cholec80 dataset root")
    parser.add_argument("--feature-dir", type=str, default=None,
                        help="Pre-extracted feature directory")
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "test", "all"],
                        help="Dataset split to evaluate")
    parser.add_argument("--report-dir", type=str, default="reports/",
                        help="Directory to write evaluation report files")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--video", type=str, default=None,
                        help="Evaluate single video file (requires --annotation)")
    parser.add_argument("--annotation", type=str, default=None,
                        help="Phase annotation file for single video evaluation")
    return parser.parse_args()


def load_checkpoint(checkpoint_path: str, device: torch.device) -> tuple:
    """Load model and config from checkpoint.

    Args:
        checkpoint_path: Path to .pth checkpoint.
        device: Target device.

    Returns:
        Tuple of (model, config_dict).
    """
    logger.info("Loading checkpoint: %s", checkpoint_path)
    ckpt = torch.load(checkpoint_path, map_location=device)
    config = ckpt.get("config", {})

    model_cfg = config.get("model", {})
    model = SurgPhaseClassifier(
        backbone=model_cfg.get("backbone", "resnet50"),
        feature_dim=model_cfg.get("feature_dim", 2048),
        temporal_model=model_cfg.get("temporal_model", "mstcn"),
        num_phases=model_cfg.get("num_phases", 7),
        num_instruments=model_cfg.get("num_instruments", 7),
        causal=model_cfg.get("causal", True),
        use_crf=model_cfg.get("use_crf", True),
        fuse_instruments=model_cfg.get("fuse_instruments", True),
        pretrained_backbone=False,  # Don't re-download when evaluating
        tcn_stages=model_cfg.get("tcn_stages", 4),
        tcn_layers=model_cfg.get("tcn_layers", 10),
        tcn_filters=model_cfg.get("tcn_filters", 64),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    logger.info("Checkpoint loaded (epoch %d)", ckpt.get("epoch", 0))
    return model, config


@torch.no_grad()
def evaluate_dataset(
    model: SurgPhaseClassifier,
    dataloader,
    device: torch.device,
) -> dict:
    """Run full dataset evaluation.

    Args:
        model: Trained SurgPhaseClassifier in eval mode.
        dataloader: DataLoader over test split.
        device: Inference device.

    Returns:
        Metrics dict from SurgicalMetrics.
    """
    metrics_tracker = SurgicalMetrics(num_phases=7)
    model.eval()

    for batch_idx, batch in enumerate(dataloader):
        frames = batch["frames"].to(device)
        labels = batch["phase_labels"].to(device)

        outputs = model(frames)

        if "phase_labels" in outputs:
            preds = outputs["phase_labels"]  # (B, T) from Viterbi
        elif "all_stage_logits" in outputs:
            final_logits = outputs["all_stage_logits"][-1]  # (B, C, T)
            preds = final_logits.argmax(dim=1)  # (B, T)
        else:
            logits = outputs["phase_logits"]
            if logits.dim() == 3 and logits.size(1) == 7:
                preds = logits.argmax(dim=1)
            else:
                preds = logits.argmax(dim=-1)

        metrics_tracker.update(preds, labels)

        if batch_idx % 20 == 0:
            logger.info("Evaluated %d / %d batches...", batch_idx, len(dataloader))

    results = metrics_tracker.compute()
    metrics_tracker.print_report(results)
    return results


def save_report(results: dict, report_dir: Path) -> None:
    """Save evaluation results to JSON and text report files.

    Args:
        results: Metrics dict from SurgicalMetrics.
        report_dir: Output directory.
    """
    report_dir.mkdir(parents=True, exist_ok=True)

    # JSON report (machine-readable)
    json_path = report_dir / "metrics.json"
    # Convert numpy scalars to Python floats for JSON serialisation
    def _to_serializable(obj):
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, dict):
            return {k: _to_serializable(v) for k, v in obj.items()}
        return obj

    with open(json_path, "w") as f:
        json.dump(_to_serializable(results), f, indent=2)
    logger.info("Metrics JSON saved: %s", json_path)

    # Text summary
    txt_path = report_dir / "summary.txt"
    with open(txt_path, "w") as f:
        f.write("SurgPhase Evaluation Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Overall Accuracy   : {results['accuracy']:.4f}\n")
        f.write(f"Mean Jaccard (mIoU): {results['mean_jaccard']:.4f}\n")
        f.write(f"Macro F1           : {results['macro_f1']:.4f}\n")
        f.write(f"Edit Distance      : {results['edit_distance']:.4f}\n")
        f.write(f"Relaxed Accuracy   : {results['relaxed_accuracy']:.4f}\n\n")

        f.write("Per-Phase F1 Scores:\n")
        for phase, m in results["per_phase"].items():
            f.write(f"  {phase:<35} F1={m['f1']:.4f}  Jac={m['jaccard']:.4f}\n")

        f.write("\nSegment-level F1:\n")
        for k, v in results["segment_f1"].items():
            f.write(f"  {k}: {v:.4f}\n")

    logger.info("Text summary saved: %s", txt_path)


def main() -> None:
    args = parse_args()

    # Device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logger.info("Evaluating on device: %s", device)

    # Load model
    model, config = load_checkpoint(args.checkpoint, device)
    model = model.to(device)

    report_dir = Path(args.report_dir)

    # Build dataset
    dataset = Cholec80Dataset(
        root=args.data_root,
        split=args.split,
        feature_dir=args.feature_dir,
        temporal_window=config.get("dataset", {}).get("temporal_window", 64),
        stride=config.get("dataset", {}).get("stride", 1),
        transform=build_eval_transforms(
            config.get("augmentation", {}).get("img_size", 256)
        ),
        return_instruments=True,
    )

    dataloader = build_dataloader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        shuffle=False,
    )

    logger.info("Evaluating %d windows from %s split...", len(dataset), args.split)

    # Run evaluation
    results = evaluate_dataset(model, dataloader, device)
    save_report(results, report_dir)

    logger.info(
        "Evaluation complete — Accuracy=%.4f  mJaccard=%.4f  Macro-F1=%.4f",
        results["accuracy"],
        results["mean_jaccard"],
        results["macro_f1"],
    )


if __name__ == "__main__":
    main()
