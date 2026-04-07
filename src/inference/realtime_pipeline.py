"""
Real-time surgical phase recognition inference pipeline.

Supports three input modes:
  - Video file: process an offline recording
  - RTSP stream: live laparoscope feed (da Vinci, Karl Storz, Stryker 1588)
  - Frame array: integrate into existing OR software via Python API

Key design decisions for real-time OR deployment:
  - Causal model only: no look-ahead, output latency = 1 frame
  - Sliding temporal window: maintain a rolling buffer of recent features
  - Phase transition smoothing: hysteresis to suppress flickering
  - FPS monitoring: drop frames gracefully when processing cannot keep up
  - ONNX Runtime: optional backend for TensorRT acceleration

Reference latency targets:
  - Soft real-time: < 500 ms (acceptable for most workflow alerts)
  - Near-real-time: < 100 ms (for phase-triggered robotic assistance)
  - Hard real-time: < 33 ms  (≥30 fps — not achievable on CPU without TRT)
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

PHASE_NAMES = [
    "Preparation",
    "Calot Triangle Dissection",
    "Clipping & Cutting",
    "Gallbladder Dissection",
    "Gallbladder Packaging",
    "Cleaning & Coagulation",
    "Gallbladder Retraction",
]

PHASE_COLORS_BGR = [
    (180, 119, 31),   # Preparation        — warm brown
    (14, 127, 255),   # Calot Dissection   — orange
    (44, 160, 44),    # Clipping           — green
    (40, 39, 214),    # GB Dissection      — red
    (189, 103, 148),  # GB Packaging       — purple
    (75, 86, 140),    # Cleaning           — maroon
    (194, 119, 227),  # GB Retraction      — pink
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class InferenceResult:
    """Single-frame inference result from the real-time pipeline.

    Attributes:
        frame_idx: Global frame index in the current video stream.
        timestamp_ms: Estimated timestamp in milliseconds.
        phase_id: Predicted surgical phase class index (0–6).
        phase_name: Human-readable phase name.
        phase_probabilities: Softmax probability vector (7 elements).
        instrument_presence: Binary vector of detected instruments.
        processing_time_ms: Wall-clock time for this frame's inference.
        is_transition: Whether a phase transition was detected at this frame.
    """

    frame_idx: int
    timestamp_ms: float
    phase_id: int
    phase_name: str
    phase_probabilities: np.ndarray
    instrument_presence: np.ndarray
    processing_time_ms: float
    is_transition: bool = False
    annotated_frame: Optional[np.ndarray] = None  # BGR uint8 with overlay


@dataclass
class PipelineStats:
    """Running statistics for the real-time pipeline."""

    frames_processed: int = 0
    frames_dropped: int = 0
    total_time_ms: float = 0.0
    phase_durations: Dict[int, float] = field(default_factory=dict)
    transition_count: int = 0

    @property
    def mean_latency_ms(self) -> float:
        if self.frames_processed == 0:
            return 0.0
        return self.total_time_ms / self.frames_processed

    @property
    def fps(self) -> float:
        if self.total_time_ms == 0:
            return 0.0
        return self.frames_processed / (self.total_time_ms / 1000.0)

    @property
    def drop_rate(self) -> float:
        total = self.frames_processed + self.frames_dropped
        return self.frames_dropped / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Real-time pipeline
# ---------------------------------------------------------------------------


class RealtimeInferencePipeline:
    """Streaming real-time surgical phase recognition pipeline.

    Wraps a SurgPhaseClassifier (or ONNX Runtime session) with:
      - Temporal feature buffer (rolling window)
      - Instrument-aware phase prediction
      - Phase transition hysteresis (debouncing)
      - FPS limiter and frame drop policy
      - Optional overlay rendering for display

    Args:
        model: ``SurgPhaseClassifier`` instance OR path to an ONNX model.
        device: Inference device (ignored for ONNX Runtime).
        window_size: Number of recent frames in the temporal buffer.
        img_size: Input resolution for the spatial encoder.
        target_fps: Target processing FPS. Frames are dropped if pipeline
                    falls behind. Set to None for maximum speed.
        transition_threshold: Minimum consecutive frames with new phase
                               prediction before emitting a transition event
                               (reduces flickering at boundaries).
        use_onnx: Force ONNX Runtime backend.
        show_overlay: Render phase overlay on output frames.
    """

    def __init__(
        self,
        model: Union[torch.nn.Module, str, Path],
        device: Optional[torch.device] = None,
        window_size: int = 64,
        img_size: int = 256,
        target_fps: Optional[float] = 25.0,
        transition_threshold: int = 5,
        use_onnx: bool = False,
        show_overlay: bool = True,
    ) -> None:
        self.window_size = window_size
        self.img_size = img_size
        self.target_fps = target_fps
        self.transition_threshold = transition_threshold
        self.show_overlay = show_overlay
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model
        if use_onnx or isinstance(model, (str, Path)):
            self._ort_session = self._load_onnx(str(model))
            self._model = None
        else:
            self._model = model.to(self.device).eval()
            self._ort_session = None

        # State
        self._feature_buffer: deque = deque(maxlen=window_size)
        self._phase_buffer: deque = deque(maxlen=transition_threshold + 5)
        self._lstm_hidden: Optional[Tuple] = None
        self._current_phase: int = -1
        self._phase_candidate: int = -1
        self._phase_candidate_count: int = 0
        self._stats = PipelineStats()

        # Preprocessing
        from torchvision import transforms
        self._preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        logger.info(
            "RealtimeInferencePipeline: window=%d, target_fps=%s, device=%s",
            window_size,
            target_fps,
            self.device,
        )

    def _load_onnx(self, model_path: str):
        """Load ONNX Runtime inference session.

        Args:
            model_path: Path to .onnx model file.

        Returns:
            ONNX Runtime InferenceSession.
        """
        try:
            import onnxruntime as ort

            providers = (
                ["CUDAExecutionProvider", "CPUExecutionProvider"]
                if torch.cuda.is_available()
                else ["CPUExecutionProvider"]
            )
            sess_opts = ort.SessionOptions()
            sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_opts.intra_op_num_threads = 4
            session = ort.InferenceSession(
                model_path, sess_options=sess_opts, providers=providers
            )
            logger.info("ONNX Runtime session loaded from %s", model_path)
            return session
        except ImportError:
            raise RuntimeError(
                "ONNX Runtime not installed. Run: pip install onnxruntime-gpu"
            )

    # ------------------------------------------------------------------
    # Frame preprocessing
    # ------------------------------------------------------------------

    def _preprocess_frame(self, frame_bgr: np.ndarray) -> torch.Tensor:
        """Convert a BGR uint8 frame to a normalised tensor.

        Args:
            frame_bgr: (H, W, 3) BGR uint8 numpy array from OpenCV.

        Returns:
            (1, 3, img_size, img_size) float tensor.
        """
        from PIL import Image

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(cv2.resize(rgb, (self.img_size, self.img_size)))
        return self._preprocess(pil).unsqueeze(0).to(self.device)

    # ------------------------------------------------------------------
    # Core inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def process_frame(self, frame_bgr: np.ndarray) -> InferenceResult:
        """Process a single frame and return the phase prediction.

        This is the main entry point for real-time integration. Call once
        per incoming frame from the endoscope feed.

        Args:
            frame_bgr: (H, W, 3) BGR uint8 frame.

        Returns:
            InferenceResult for this frame.
        """
        t0 = time.perf_counter()
        frame_idx = self._stats.frames_processed

        # Preprocess
        tensor = self._preprocess_frame(frame_bgr)  # (1, 3, H, W)

        # Build temporal window: (1, T, 3, H, W)
        self._feature_buffer.append(tensor.squeeze(0))
        window_tensors = list(self._feature_buffer)

        # Pad with first frame if buffer not full yet
        while len(window_tensors) < self.window_size:
            window_tensors.insert(0, window_tensors[0])

        frames_batch = torch.stack(window_tensors[-self.window_size:], dim=0).unsqueeze(0)
        # (1, window_size, 3, H, W)

        # Inference
        if self._ort_session is not None:
            phase_probs, inst_presence = self._ort_infer(frames_batch)
        else:
            phase_probs, inst_presence = self._torch_infer(frames_batch)

        # Current frame prediction = last time-step of window
        current_phase_probs = phase_probs[-1] if phase_probs.ndim == 2 else phase_probs
        phase_id = int(current_phase_probs.argmax())

        # Transition hysteresis
        is_transition = self._update_phase_state(phase_id)

        latency_ms = (time.perf_counter() - t0) * 1000

        # Update stats
        self._stats.frames_processed += 1
        self._stats.total_time_ms += latency_ms
        if is_transition:
            self._stats.transition_count += 1

        result = InferenceResult(
            frame_idx=frame_idx,
            timestamp_ms=frame_idx * (1000.0 / (self.target_fps or 25.0)),
            phase_id=self._current_phase,
            phase_name=PHASE_NAMES[self._current_phase] if self._current_phase >= 0 else "Unknown",
            phase_probabilities=current_phase_probs,
            instrument_presence=inst_presence,
            processing_time_ms=latency_ms,
            is_transition=is_transition,
        )

        if self.show_overlay:
            result.annotated_frame = self._render_overlay(frame_bgr.copy(), result)

        return result

    def _torch_infer(
        self, frames: torch.Tensor
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run inference with the PyTorch model.

        Args:
            frames: (1, T, 3, H, W) tensor.

        Returns:
            Tuple of (phase_probs, inst_presence) numpy arrays.
        """
        outputs = self._model(frames)

        # Phase probabilities — take last time-step
        if "phase_probs" in outputs:
            probs = outputs["phase_probs"].squeeze(0)  # (T, C)
        elif "all_stage_logits" in outputs:
            logits = outputs["all_stage_logits"][-1]  # (1, C, T)
            probs = F.softmax(logits, dim=1).squeeze(0).permute(1, 0)  # (T, C)
        else:
            logits = outputs["phase_logits"]
            if logits.dim() == 3:
                if logits.size(1) == 7:
                    probs = F.softmax(logits, dim=1).squeeze(0).permute(1, 0)
                else:
                    probs = F.softmax(logits, dim=-1).squeeze(0)
            else:
                probs = F.softmax(logits, dim=-1)

        probs_np = probs.cpu().numpy()

        # Instrument presence
        if "instrument_logits" in outputs:
            inst_logits = outputs["instrument_logits"].squeeze(0)[-1]  # last frame
            inst_np = torch.sigmoid(inst_logits).cpu().numpy()
        else:
            inst_np = np.zeros(7)

        return probs_np, inst_np

    def _ort_infer(
        self, frames: torch.Tensor
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run inference with ONNX Runtime.

        Args:
            frames: (1, T, 3, H, W) tensor.

        Returns:
            Tuple of (phase_probs, inst_presence) numpy arrays.
        """
        frames_np = frames.cpu().numpy().astype(np.float32)
        outputs = self._ort_session.run(None, {"frames": frames_np})
        phase_logits = outputs[0].squeeze(0)  # (T, C)
        probs = np.exp(phase_logits) / np.exp(phase_logits).sum(axis=-1, keepdims=True)
        inst_np = np.zeros(7)
        return probs, inst_np

    # ------------------------------------------------------------------
    # Transition smoothing
    # ------------------------------------------------------------------

    def _update_phase_state(self, new_phase: int) -> bool:
        """Apply hysteresis-based transition smoothing.

        A phase change is only committed after ``transition_threshold``
        consecutive frames predict the same new phase. This prevents
        flickering at phase boundaries.

        Args:
            new_phase: Raw per-frame phase prediction.

        Returns:
            True if a phase transition was committed this frame.
        """
        if self._current_phase == -1:
            self._current_phase = new_phase
            self._phase_candidate = new_phase
            return True

        if new_phase == self._phase_candidate:
            self._phase_candidate_count += 1
        else:
            self._phase_candidate = new_phase
            self._phase_candidate_count = 1

        if (
            self._phase_candidate != self._current_phase
            and self._phase_candidate_count >= self.transition_threshold
        ):
            self._current_phase = self._phase_candidate
            self._phase_candidate_count = 0
            logger.info(
                "Phase transition → %s (frame %d)",
                PHASE_NAMES[self._current_phase],
                self._stats.frames_processed,
            )
            return True

        return False

    # ------------------------------------------------------------------
    # Video stream processing
    # ------------------------------------------------------------------

    def process_video(
        self,
        source: Union[str, int],
        callback: Optional[Callable[[InferenceResult], None]] = None,
        max_frames: Optional[int] = None,
        save_output: Optional[str] = None,
    ) -> Iterator[InferenceResult]:
        """Process a video file or live camera stream.

        Args:
            source: Video file path, RTSP URL, or webcam index.
            callback: Optional callback called for each frame result.
            max_frames: Stop after this many frames (None = until end).
            save_output: Path to write annotated output video (optional).

        Yields:
            InferenceResult for each processed frame.
        """
        cap = cv2.VideoCapture(str(source) if not isinstance(source, int) else source)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {source}")

        src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_interval_s = 1.0 / (self.target_fps or src_fps)

        # Optional video writer
        writer = None
        if save_output:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(
                save_output, fourcc, src_fps, (width, height)
            )

        try:
            frame_count = 0
            last_process_time = time.perf_counter()

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if max_frames is not None and frame_count >= max_frames:
                    break

                # Frame drop policy: skip if processing has fallen behind
                now = time.perf_counter()
                elapsed = now - last_process_time
                if self.target_fps and elapsed < frame_interval_s * 0.8:
                    self._stats.frames_dropped += 1
                    continue

                result = self.process_frame(frame)
                last_process_time = time.perf_counter()

                if callback:
                    callback(result)

                if writer and result.annotated_frame is not None:
                    writer.write(result.annotated_frame)

                yield result
                frame_count += 1

        finally:
            cap.release()
            if writer:
                writer.release()

        logger.info(
            "Video processing complete. Stats: fps=%.1f latency=%.1fms drop=%.1f%%",
            self._stats.fps,
            self._stats.mean_latency_ms,
            self._stats.drop_rate * 100,
        )

    # ------------------------------------------------------------------
    # Overlay rendering
    # ------------------------------------------------------------------

    def _render_overlay(
        self, frame: np.ndarray, result: InferenceResult
    ) -> np.ndarray:
        """Render phase recognition overlay on the frame.

        Draws:
          - Phase name banner (top left)
          - Probability bar chart (bottom)
          - Instrument presence indicators
          - FPS counter

        Args:
            frame: (H, W, 3) BGR uint8 frame (will be modified in-place).
            result: InferenceResult for this frame.

        Returns:
            Annotated (H, W, 3) BGR frame.
        """
        h, w = frame.shape[:2]
        phase_id = result.phase_id
        if phase_id < 0 or phase_id >= len(PHASE_NAMES):
            return frame

        color = PHASE_COLORS_BGR[phase_id]

        # Semi-transparent phase banner
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 55), color, -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        cv2.putText(
            frame,
            f"Phase: {result.phase_name}",
            (10, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # FPS counter
        fps_text = f"{self._stats.fps:.1f} fps | {result.processing_time_ms:.1f} ms"
        cv2.putText(
            frame, fps_text, (w - 280, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA,
        )

        # Probability bars
        bar_h = 18
        bar_max_w = 160
        for i, prob in enumerate(result.phase_probabilities[-7:]):
            y_base = h - (7 - i) * (bar_h + 4) - 10
            bar_w = max(2, int(prob * bar_max_w))
            bar_color = PHASE_COLORS_BGR[i] if i < len(PHASE_COLORS_BGR) else (128, 128, 128)
            cv2.rectangle(frame, (10, y_base), (10 + bar_w, y_base + bar_h), bar_color, -1)
            label = PHASE_NAMES[i][:12] if i < len(PHASE_NAMES) else f"P{i}"
            cv2.putText(
                frame, f"{label[:14]} {prob:.2f}",
                (10 + bar_max_w + 5, y_base + bar_h - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1, cv2.LINE_AA,
            )

        return frame

    # ------------------------------------------------------------------
    # Statistics and export
    # ------------------------------------------------------------------

    @property
    def stats(self) -> PipelineStats:
        """Return running pipeline statistics."""
        return self._stats

    def reset(self) -> None:
        """Reset pipeline state (call between videos)."""
        self._feature_buffer.clear()
        self._phase_buffer.clear()
        self._lstm_hidden = None
        self._current_phase = -1
        self._phase_candidate = -1
        self._phase_candidate_count = 0
        self._stats = PipelineStats()

    def warmup(self, num_frames: int = 10) -> None:
        """Warm up the model to avoid JIT compilation latency on first frame.

        Args:
            num_frames: Number of dummy frames to process.
        """
        dummy = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        for _ in range(num_frames):
            self.process_frame(dummy)
        self.reset()
        logger.info("Pipeline warmed up with %d dummy frames.", num_frames)
