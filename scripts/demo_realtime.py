#!/usr/bin/env python3
"""
Real-time surgical phase recognition demo.

Runs the SurgPhase inference pipeline on a video file or live camera
stream, displaying an annotated overlay with phase prediction, confidence
bars, and FPS statistics.

Usage
-----
    # Run on a recorded surgical video:
    python scripts/demo_realtime.py \\
        --source /path/to/cholecystectomy.mp4 \\
        --checkpoint weights/surgphase_base.pth \\
        --display

    # Run on live RTSP stream (e.g. da Vinci endoscope):
    python scripts/demo_realtime.py \\
        --source rtsp://192.168.1.100:8554/endoscope \\
        --checkpoint weights/surgphase_base.pth \\
        --display \\
        --save-output output/inference_recording.mp4

    # Run with ONNX model (faster on edge hardware):
    python scripts/demo_realtime.py \\
        --source /path/to/video.mp4 \\
        --onnx weights/surgphase_base.onnx \\
        --display

    # Headless benchmark (measure throughput only):
    python scripts/demo_realtime.py \\
        --source /path/to/video.mp4 \\
        --checkpoint weights/surgphase_base.pth \\
        --max-frames 500 \\
        --no-display
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.inference.realtime_pipeline import RealtimeInferencePipeline, InferenceResult
from src.models.phase_classifier import SurgPhaseClassifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("surgphase.demo")

PHASE_NAMES = [
    "Preparation",
    "Calot Triangle Dissection",
    "Clipping & Cutting",
    "Gallbladder Dissection",
    "Gallbladder Packaging",
    "Cleaning & Coagulation",
    "Gallbladder Retraction",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Real-time surgical phase recognition demo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Input
    parser.add_argument("--source", type=str, required=True,
                        help="Video file path, RTSP URL, or webcam index (0, 1, ...)")

    # Model
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--checkpoint", type=str, default=None,
                       help="PyTorch checkpoint path (.pth)")
    group.add_argument("--onnx", type=str, default=None,
                       help="ONNX model path (.onnx)")

    # Display
    parser.add_argument("--display", action="store_true",
                        help="Show live annotated video window")
    parser.add_argument("--no-display", dest="display", action="store_false")
    parser.set_defaults(display=True)
    parser.add_argument("--save-output", type=str, default=None,
                        help="Save annotated output video to this path")
    parser.add_argument("--window-name", type=str, default="SurgPhase — Real-time Demo")

    # Pipeline settings
    parser.add_argument("--window-size", type=int, default=64,
                        help="Temporal context window (frames)")
    parser.add_argument("--img-size", type=int, default=256,
                        help="Input spatial resolution")
    parser.add_argument("--target-fps", type=float, default=25.0,
                        help="Target processing FPS (None = max speed)")
    parser.add_argument("--transition-threshold", type=int, default=5,
                        help="Consecutive frames before emitting phase transition")

    # Control
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Stop after N frames (default: run until end)")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--warmup", type=int, default=10,
                        help="Number of warmup frames before timing")

    return parser.parse_args()


def load_pytorch_model(
    checkpoint_path: str,
    device: torch.device,
) -> SurgPhaseClassifier:
    """Load a SurgPhaseClassifier from checkpoint.

    Args:
        checkpoint_path: Path to .pth checkpoint.
        device: Target device.

    Returns:
        Loaded model in eval mode.
    """
    logger.info("Loading PyTorch model from %s", checkpoint_path)
    ckpt = torch.load(checkpoint_path, map_location=device)
    config = ckpt.get("config", {})
    model_cfg = config.get("model", {})

    model = SurgPhaseClassifier(
        backbone=model_cfg.get("backbone", "resnet50"),
        feature_dim=model_cfg.get("feature_dim", 2048),
        temporal_model=model_cfg.get("temporal_model", "mstcn"),
        num_phases=model_cfg.get("num_phases", 7),
        causal=model_cfg.get("causal", True),
        use_crf=model_cfg.get("use_crf", True),
        fuse_instruments=model_cfg.get("fuse_instruments", True),
        pretrained_backbone=False,
        tcn_stages=model_cfg.get("tcn_stages", 4),
        tcn_layers=model_cfg.get("tcn_layers", 10),
        tcn_filters=model_cfg.get("tcn_filters", 64),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval().to(device)
    return model


def print_transition(result: InferenceResult) -> None:
    """Print a formatted phase transition event to stdout."""
    ts_min = result.timestamp_ms / 1000 / 60
    print(
        f"\n  [TRANSITION @ {ts_min:.1f} min]  "
        f"→ Phase {result.phase_id}: {result.phase_name}  "
        f"(conf={result.phase_probabilities[result.phase_id]:.2f})"
    )


def main() -> None:
    args = parse_args()

    # Device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("Using CUDA GPU: %s", torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        logger.warning("CUDA not available — running on CPU (expect low FPS).")

    # Build pipeline
    if args.onnx:
        pipeline = RealtimeInferencePipeline(
            model=args.onnx,
            device=device,
            window_size=args.window_size,
            img_size=args.img_size,
            target_fps=args.target_fps,
            transition_threshold=args.transition_threshold,
            use_onnx=True,
            show_overlay=args.display or bool(args.save_output),
        )
    else:
        model = load_pytorch_model(args.checkpoint, device)
        pipeline = RealtimeInferencePipeline(
            model=model,
            device=device,
            window_size=args.window_size,
            img_size=args.img_size,
            target_fps=args.target_fps,
            transition_threshold=args.transition_threshold,
            use_onnx=False,
            show_overlay=args.display or bool(args.save_output),
        )

    # Warmup
    if args.warmup > 0:
        logger.info("Warming up pipeline (%d frames)...", args.warmup)
        pipeline.warmup(args.warmup)

    # Setup display window
    if args.display:
        cv2.namedWindow(args.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(args.window_name, 1280, 720)

    # Try to determine video source type
    try:
        source_int = int(args.source)
    except (ValueError, TypeError):
        source_int = args.source

    logger.info("Starting real-time inference on: %s", args.source)
    print("\n" + "=" * 60)
    print("  SurgPhase Real-Time Surgical Phase Recognition")
    print("=" * 60)
    print(f"  Source:      {args.source}")
    print(f"  Window:      {args.window_size} frames")
    print(f"  Target FPS:  {args.target_fps}")
    print(f"  Device:      {device}")
    print("=" * 60)
    print("  Press 'q' to quit | 'r' to reset pipeline state\n")

    frame_count = 0
    t_start = time.perf_counter()

    try:
        for result in pipeline.process_video(
            source=source_int,
            callback=lambda r: print_transition(r) if r.is_transition else None,
            max_frames=args.max_frames,
            save_output=args.save_output,
        ):
            frame_count += 1

            # Display
            if args.display and result.annotated_frame is not None:
                cv2.imshow(args.window_name, result.annotated_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    logger.info("User quit.")
                    break
                elif key == ord("r"):
                    pipeline.reset()
                    logger.info("Pipeline state reset.")

            # Console progress (every 100 frames)
            if frame_count % 100 == 0:
                stats = pipeline.stats
                print(
                    f"  Frame {frame_count:6d} | "
                    f"FPS: {stats.fps:5.1f} | "
                    f"Latency: {stats.mean_latency_ms:5.1f} ms | "
                    f"Dropped: {stats.frames_dropped:4d} | "
                    f"Current: {result.phase_name}"
                )

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        if args.display:
            cv2.destroyAllWindows()

    # Final statistics
    total_time = time.perf_counter() - t_start
    stats = pipeline.stats
    print("\n" + "=" * 60)
    print("  INFERENCE SUMMARY")
    print("=" * 60)
    print(f"  Frames processed : {stats.frames_processed}")
    print(f"  Frames dropped   : {stats.frames_dropped}")
    print(f"  Drop rate        : {stats.drop_rate * 100:.1f}%")
    print(f"  Mean latency     : {stats.mean_latency_ms:.2f} ms")
    print(f"  Throughput       : {stats.fps:.1f} FPS")
    print(f"  Transitions      : {stats.transition_count}")
    print(f"  Wall clock time  : {total_time:.1f} s")
    print("=" * 60)

    if args.save_output:
        print(f"\n  Saved annotated video to: {args.save_output}")


if __name__ == "__main__":
    main()
