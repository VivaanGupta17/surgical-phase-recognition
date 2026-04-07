#!/usr/bin/env python3
"""
ONNX and TensorRT export script for SurgPhase deployment.

Exports a trained SurgPhaseClassifier to ONNX format for deployment
on edge hardware (NVIDIA Jetson AGX Orin, NVIDIA IGX) and cloud inference
(AWS Inferentia, Azure ONNX Runtime).

Optional TensorRT conversion for FP16 / INT8 optimisation.

Usage
-----
    # Export to ONNX (FP32, dynamic batch/sequence):
    python scripts/export_onnx.py \\
        --checkpoint runs/exp01/checkpoints/best_model.pth \\
        --output weights/surgphase_base.onnx

    # Export with ONNX Simplifier:
    python scripts/export_onnx.py \\
        --checkpoint runs/exp01/checkpoints/best_model.pth \\
        --output weights/surgphase_base_simplified.onnx \\
        --simplify

    # Export with TensorRT optimisation (requires CUDA + TensorRT):
    python scripts/export_onnx.py \\
        --checkpoint runs/exp01/checkpoints/best_model.pth \\
        --output weights/surgphase_base.onnx \\
        --tensorrt \\
        --precision fp16 \\
        --trt-output weights/surgphase_base_fp16.engine
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.phase_classifier import SurgPhaseClassifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("surgphase.export")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export SurgPhase model to ONNX / TensorRT",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Trained model checkpoint (.pth)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output ONNX file path (.onnx)")
    parser.add_argument("--opset", type=int, default=17,
                        help="ONNX opset version")
    parser.add_argument("--simplify", action="store_true",
                        help="Run onnx-simplifier after export")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size for export trace")
    parser.add_argument("--seq-len", type=int, default=64,
                        help="Sequence length for export trace")
    parser.add_argument("--img-size", type=int, default=256,
                        help="Spatial input resolution")
    parser.add_argument("--dynamic", action="store_true", default=True,
                        help="Enable dynamic batch and sequence length axes")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--fp16", action="store_true",
                        help="Export in FP16 precision (smaller, faster on supported hardware)")

    # TensorRT options
    parser.add_argument("--tensorrt", action="store_true",
                        help="Also build TensorRT engine from ONNX")
    parser.add_argument("--precision", type=str, default="fp16",
                        choices=["fp32", "fp16", "int8"],
                        help="TensorRT precision mode")
    parser.add_argument("--trt-output", type=str, default=None,
                        help="Output TensorRT engine file (.engine)")
    parser.add_argument("--workspace-gb", type=int, default=4,
                        help="TensorRT builder workspace in GB")
    return parser.parse_args()


def load_model(checkpoint_path: str, device: torch.device) -> tuple:
    """Load model from checkpoint for export.

    Args:
        checkpoint_path: Path to .pth file.
        device: Target device.

    Returns:
        Tuple of (model, config).
    """
    ckpt = torch.load(checkpoint_path, map_location=device)
    config = ckpt.get("config", {})
    model_cfg = config.get("model", {})

    model = SurgPhaseClassifier(
        backbone=model_cfg.get("backbone", "resnet50"),
        feature_dim=model_cfg.get("feature_dim", 2048),
        temporal_model=model_cfg.get("temporal_model", "mstcn"),
        num_phases=model_cfg.get("num_phases", 7),
        causal=model_cfg.get("causal", True),
        use_crf=False,  # CRF uses Viterbi — not ONNX-exportable
        fuse_instruments=model_cfg.get("fuse_instruments", True),
        pretrained_backbone=False,
        tcn_stages=model_cfg.get("tcn_stages", 4),
        tcn_layers=model_cfg.get("tcn_layers", 10),
        tcn_filters=model_cfg.get("tcn_filters", 64),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    logger.info("Loaded model from checkpoint (epoch %d)", ckpt.get("epoch", 0))
    return model, config


def export_onnx(
    model: torch.nn.Module,
    output_path: str,
    batch_size: int,
    seq_len: int,
    img_size: int,
    opset: int,
    dynamic: bool,
    fp16: bool,
    device: torch.device,
) -> str:
    """Export model to ONNX.

    Args:
        model: SurgPhaseClassifier in eval mode.
        output_path: Output .onnx path.
        batch_size: Static batch size for export trace.
        seq_len: Static sequence length for export trace.
        img_size: Spatial resolution.
        opset: ONNX opset version.
        dynamic: Enable dynamic axes.
        fp16: Cast model weights to FP16 before export.
        device: Export device.

    Returns:
        Absolute path to the written ONNX file.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    if fp16:
        model = model.half()
        dtype = torch.float16
    else:
        dtype = torch.float32

    dummy_input = torch.zeros(
        batch_size, seq_len, 3, img_size, img_size, dtype=dtype, device=device
    )

    dynamic_axes = None
    if dynamic:
        dynamic_axes = {
            "frames": {0: "batch_size", 1: "seq_len"},
            "phase_logits": {0: "batch_size", 1: "seq_len"},
        }

    logger.info(
        "Exporting ONNX: batch=%d seq=%d img=%dx%d opset=%d fp16=%s",
        batch_size, seq_len, img_size, img_size, opset, fp16,
    )

    with torch.no_grad():
        torch.onnx.export(
            model,
            (dummy_input,),
            output_path,
            input_names=["frames"],
            output_names=["phase_logits"],
            dynamic_axes=dynamic_axes,
            opset_version=opset,
            do_constant_folding=True,
            verbose=False,
        )

    size_mb = Path(output_path).stat().st_size / (1024 ** 2)
    logger.info("ONNX model saved: %s (%.1f MB)", output_path, size_mb)
    return output_path


def simplify_onnx(onnx_path: str) -> str:
    """Run ONNX Simplifier to reduce model size and operator count.

    Args:
        onnx_path: Path to original ONNX file.

    Returns:
        Path to simplified model (overwrites input).
    """
    try:
        import onnx
        import onnxsim

        logger.info("Running ONNX Simplifier...")
        model_onnx = onnx.load(onnx_path)
        model_simplified, check = onnxsim.simplify(model_onnx)
        if check:
            onnx.save(model_simplified, onnx_path)
            logger.info("ONNX Simplifier succeeded: %s", onnx_path)
        else:
            logger.warning("ONNX Simplifier check failed — keeping original.")
    except ImportError:
        logger.warning("onnxsim not installed. Skipping simplification.")
        logger.info("Install with: pip install onnx-simplifier")
    return onnx_path


def validate_onnx(onnx_path: str, dummy_input: torch.Tensor) -> None:
    """Validate ONNX model with ONNX Runtime.

    Args:
        onnx_path: Path to .onnx model.
        dummy_input: Example input tensor (CPU float32).
    """
    try:
        import onnxruntime as ort
        import numpy as np

        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        inputs = {sess.get_inputs()[0].name: dummy_input.numpy().astype(np.float32)}
        outputs = sess.run(None, inputs)
        logger.info(
            "ONNX Runtime validation passed. Output shape: %s",
            outputs[0].shape,
        )
    except ImportError:
        logger.warning("onnxruntime not installed — skipping validation.")
    except Exception as e:
        logger.error("ONNX Runtime validation FAILED: %s", e)


def build_tensorrt_engine(
    onnx_path: str,
    engine_path: str,
    precision: str,
    workspace_gb: int,
) -> None:
    """Build a TensorRT engine from an ONNX model.

    Requires NVIDIA TensorRT and tensorrt Python bindings.

    Args:
        onnx_path: Input ONNX model path.
        engine_path: Output .engine file path.
        precision: 'fp32', 'fp16', or 'int8'.
        workspace_gb: Builder workspace in GB.
    """
    try:
        import tensorrt as trt

        logger.info("Building TensorRT engine (%s precision)...", precision.upper())
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(TRT_LOGGER)
        network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(network_flags)
        parser = trt.OnnxParser(network, TRT_LOGGER)

        with open(onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                for i in range(parser.num_errors):
                    logger.error("TRT parser error: %s", parser.get_error(i))
                raise RuntimeError("Failed to parse ONNX model with TensorRT")

        config = builder.create_builder_config()
        config.set_memory_pool_limit(
            trt.MemoryPoolType.WORKSPACE, workspace_gb * (1 << 30)
        )

        if precision == "fp16" and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            logger.info("FP16 mode enabled")
        elif precision == "int8" and builder.platform_has_fast_int8:
            config.set_flag(trt.BuilderFlag.INT8)
            logger.warning(
                "INT8 mode requires a calibration data loader. "
                "Set config.int8_calibrator before building."
            )

        serialized_engine = builder.build_serialized_network(network, config)
        if serialized_engine is None:
            raise RuntimeError("TensorRT engine build failed.")

        Path(engine_path).parent.mkdir(parents=True, exist_ok=True)
        with open(engine_path, "wb") as f:
            f.write(serialized_engine)

        size_mb = Path(engine_path).stat().st_size / (1024 ** 2)
        logger.info("TensorRT engine saved: %s (%.1f MB)", engine_path, size_mb)

    except ImportError:
        logger.error(
            "TensorRT Python bindings not found. "
            "Install from: https://docs.nvidia.com/deeplearning/tensorrt/install-guide"
        )


def main() -> None:
    args = parse_args()

    # Device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logger.info("Export device: %s", device)

    # Load model
    model, config = load_model(args.checkpoint, device)
    model = model.to(device)

    # Export ONNX
    onnx_path = export_onnx(
        model=model,
        output_path=args.output,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        img_size=args.img_size,
        opset=args.opset,
        dynamic=args.dynamic,
        fp16=args.fp16,
        device=device,
    )

    if args.simplify:
        simplify_onnx(onnx_path)

    # Validate
    dummy_cpu = torch.zeros(args.batch_size, args.seq_len, 3, args.img_size, args.img_size)
    validate_onnx(onnx_path, dummy_cpu)

    # TensorRT (optional)
    if args.tensorrt:
        trt_out = args.trt_output or onnx_path.replace(".onnx", f"_{args.precision}.engine")
        build_tensorrt_engine(
            onnx_path=onnx_path,
            engine_path=trt_out,
            precision=args.precision,
            workspace_gb=args.workspace_gb,
        )

    logger.info("Export complete.")


if __name__ == "__main__":
    main()
