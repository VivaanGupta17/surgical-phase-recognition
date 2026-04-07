"""
Surgical-specific evaluation metrics for phase recognition and segmentation.

Implements the standard evaluation suite used in Cholec80 benchmarking:
  - Per-phase accuracy, precision, recall, F1
  - Mean phase Jaccard (IoU) — primary ranking metric in Cholec80
  - Phase transition detection accuracy (within ±τ frames)
  - Edit distance (Levenshtein on phase sequence) for temporal quality
  - Segment-level metrics (F1@k — segment overlap threshold)
  - Relaxed boundary evaluation (±10s tolerance)

References:
  [1] Twinanda et al., "EndoNet: A deep architecture for recognition tasks
      on laparoscopic videos", IEEE TMI 2017.
  [2] Czempiel et al., "TeCNO: Surgical phase recognition with multi-stage
      temporal convolutional networks", MICCAI 2020.
  [3] Lea et al., "Temporal Convolutional Networks for Action Segmentation
      and Detection", CVPR 2017 (edit distance, F1@k).
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)

NUM_PHASES = 7
PHASE_NAMES = [
    "Preparation",
    "CalotTriangleDissection",
    "ClippingCutting",
    "GallbladderDissection",
    "GallbladderPackaging",
    "CleaningCoagulation",
    "GallbladderRetraction",
]


# ---------------------------------------------------------------------------
# Core metric functions
# ---------------------------------------------------------------------------


def frame_level_accuracy(
    predictions: np.ndarray,
    targets: np.ndarray,
) -> float:
    """Overall frame-level phase accuracy.

    Args:
        predictions: (N,) predicted phase labels.
        targets: (N,) ground truth phase labels.

    Returns:
        Accuracy in [0, 1].
    """
    assert len(predictions) == len(targets)
    return float((predictions == targets).mean())


def per_phase_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    num_classes: int = NUM_PHASES,
    phase_names: Optional[List[str]] = None,
) -> Dict[str, Dict[str, float]]:
    """Compute per-phase precision, recall, F1, and Jaccard (IoU).

    Args:
        predictions: (N,) predicted phase labels.
        targets: (N,) ground-truth phase labels.
        num_classes: Number of phase classes.
        phase_names: Optional list of phase display names.

    Returns:
        Nested dict: {phase_name: {precision, recall, f1, jaccard, support}}.
    """
    if phase_names is None:
        phase_names = [f"Phase_{i}" for i in range(num_classes)]

    results: Dict[str, Dict[str, float]] = {}
    for p in range(num_classes):
        tp = int(((predictions == p) & (targets == p)).sum())
        fp = int(((predictions == p) & (targets != p)).sum())
        fn = int(((predictions != p) & (targets == p)).sum())
        support = int((targets == p).sum())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall + 1e-9)
        jaccard = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

        results[phase_names[p]] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "jaccard": jaccard,
            "support": support,
        }
    return results


def mean_jaccard(
    predictions: np.ndarray,
    targets: np.ndarray,
    num_classes: int = NUM_PHASES,
) -> float:
    """Mean phase Jaccard index (mean IoU) — primary Cholec80 metric.

    Args:
        predictions: (N,) predicted labels.
        targets: (N,) ground-truth labels.
        num_classes: Number of phase classes.

    Returns:
        Mean Jaccard score in [0, 1].
    """
    metrics = per_phase_metrics(predictions, targets, num_classes)
    jaccards = [v["jaccard"] for v in metrics.values() if v["support"] > 0]
    return float(np.mean(jaccards)) if jaccards else 0.0


def transition_detection_accuracy(
    pred_phases: np.ndarray,
    gt_phases: np.ndarray,
    tolerance_frames: int = 10,
) -> Dict[str, float]:
    """Evaluate phase transition detection within a tolerance window.

    A predicted transition is counted as correct if it occurs within
    ±tolerance_frames of the ground-truth transition boundary.

    Args:
        pred_phases: (N,) predicted phase label sequence.
        gt_phases: (N,) ground-truth phase label sequence.
        tolerance_frames: Half-window size in frames. Default 10 (= ~10s at 1fps).

    Returns:
        Dict with 'precision', 'recall', 'f1' for transition events.
    """
    gt_transitions = _get_transition_frames(gt_phases)
    pred_transitions = _get_transition_frames(pred_phases)

    # Match predicted transitions to GT within tolerance
    matched_gt = set()
    matched_pred = set()

    for pred_t in pred_transitions:
        for gt_t in gt_transitions:
            if abs(pred_t - gt_t) <= tolerance_frames and gt_t not in matched_gt:
                matched_gt.add(gt_t)
                matched_pred.add(pred_t)
                break

    tp = len(matched_pred)
    fp = len(pred_transitions) - tp
    fn = len(gt_transitions) - len(matched_gt)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall + 1e-9)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "num_gt_transitions": len(gt_transitions),
        "num_pred_transitions": len(pred_transitions),
    }


def _get_transition_frames(phases: np.ndarray) -> List[int]:
    """Extract frame indices where phase label changes.

    Args:
        phases: (N,) phase label sequence.

    Returns:
        List of frame indices (1-indexed) at which transitions occur.
    """
    transitions = []
    for i in range(1, len(phases)):
        if phases[i] != phases[i - 1]:
            transitions.append(i)
    return transitions


def edit_distance(
    pred_sequence: np.ndarray,
    gt_sequence: np.ndarray,
    normalise: bool = True,
) -> float:
    """Levenshtein edit distance between compressed phase sequences.

    Compresses consecutive duplicate phases (RLE) before computing
    edit distance, so the metric reflects sequence-level errors
    rather than frame-level duplications.

    Args:
        pred_sequence: (N,) predicted frame-level phase labels.
        gt_sequence: (N,) ground-truth frame-level phase labels.
        normalise: Normalise by the maximum of the two sequence lengths.

    Returns:
        Edit distance (float, normalised if requested).
    """
    pred_rle = _run_length_encode(pred_sequence)
    gt_rle = _run_length_encode(gt_sequence)

    dist = _levenshtein(pred_rle, gt_rle)
    if normalise:
        denom = max(len(pred_rle), len(gt_rle))
        return dist / denom if denom > 0 else 0.0
    return float(dist)


def _run_length_encode(sequence: np.ndarray) -> List[int]:
    """Compress a sequence by merging consecutive identical values."""
    if len(sequence) == 0:
        return []
    compressed = [sequence[0]]
    for v in sequence[1:]:
        if v != compressed[-1]:
            compressed.append(v)
    return compressed


def _levenshtein(a: List[int], b: List[int]) -> int:
    """Standard dynamic programming Levenshtein distance."""
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[j] = prev[j - 1]
            else:
                dp[j] = 1 + min(prev[j], dp[j - 1], prev[j - 1])
    return dp[n]


def segment_f1(
    predictions: np.ndarray,
    targets: np.ndarray,
    overlap_thresholds: Sequence[float] = (0.10, 0.25, 0.50),
    num_classes: int = NUM_PHASES,
) -> Dict[str, float]:
    """Compute F1@k segmentation metrics (Lea et al., 2017).

    A predicted segment is a true positive if its IoU with the matched
    GT segment exceeds the threshold ``k`` (expressed as a fraction).

    Args:
        predictions: (N,) predicted label sequence.
        targets: (N,) ground-truth label sequence.
        overlap_thresholds: IoU thresholds to evaluate (k ∈ [0, 1]).
        num_classes: Number of phase classes.

    Returns:
        Dict: {'f1@10': ..., 'f1@25': ..., 'f1@50': ...} (keys use percent).
    """
    results = {}
    for thresh in overlap_thresholds:
        tp = fp = fn = 0
        for c in range(num_classes):
            pred_segs = _extract_segments(predictions, c)
            gt_segs = _extract_segments(targets, c)
            matched = set()
            for p_start, p_end in pred_segs:
                best_iou = 0.0
                best_j = -1
                for j, (g_start, g_end) in enumerate(gt_segs):
                    if j in matched:
                        continue
                    intersection = max(0, min(p_end, g_end) - max(p_start, g_start))
                    union = (p_end - p_start) + (g_end - g_start) - intersection
                    iou = intersection / union if union > 0 else 0.0
                    if iou > best_iou:
                        best_iou = iou
                        best_j = j
                if best_iou >= thresh and best_j >= 0:
                    tp += 1
                    matched.add(best_j)
                else:
                    fp += 1
            fn += len(gt_segs) - len(matched)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall + 1e-9)
        key = f"f1@{int(thresh * 100)}"
        results[key] = f1
    return results


def _extract_segments(
    sequence: np.ndarray, label: int
) -> List[Tuple[int, int]]:
    """Extract (start, end) segments of a specific label in a sequence."""
    segments = []
    start = None
    for i, v in enumerate(sequence):
        if v == label and start is None:
            start = i
        elif v != label and start is not None:
            segments.append((start, i))
            start = None
    if start is not None:
        segments.append((start, len(sequence)))
    return segments


def relaxed_boundary_accuracy(
    predictions: np.ndarray,
    targets: np.ndarray,
    tolerance_frames: int = 10,
) -> float:
    """Phase accuracy with relaxed boundary evaluation.

    Frames within ±tolerance_frames of a ground-truth phase boundary
    are excluded from accuracy computation, reducing sensitivity to
    boundary annotation ambiguity (common in surgical video).

    Args:
        predictions: (N,) predicted labels.
        targets: (N,) ground-truth labels.
        tolerance_frames: Boundary exclusion half-window.

    Returns:
        Accuracy computed on non-boundary frames.
    """
    gt_transitions = set(_get_transition_frames(targets))
    boundary_frames = set()
    for t in gt_transitions:
        for delta in range(-tolerance_frames, tolerance_frames + 1):
            f = t + delta
            if 0 <= f < len(targets):
                boundary_frames.add(f)

    non_boundary = np.array(
        [i for i in range(len(targets)) if i not in boundary_frames]
    )
    if len(non_boundary) == 0:
        return 0.0
    return float((predictions[non_boundary] == targets[non_boundary]).mean())


# ---------------------------------------------------------------------------
# Inference latency measurement
# ---------------------------------------------------------------------------


class LatencyProfiler:
    """Measure model inference latency and throughput.

    Supports CPU and CUDA timing with warm-up to ensure stable measurements.

    Args:
        model: PyTorch model to profile.
        device: Inference device.
        num_warmup: Warmup iterations before measurement.
        num_runs: Measurement iterations.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        num_warmup: int = 10,
        num_runs: int = 100,
    ) -> None:
        self.model = model
        self.device = device
        self.num_warmup = num_warmup
        self.num_runs = num_runs

    @torch.no_grad()
    def measure(
        self,
        input_tensor: torch.Tensor,
        use_cuda_events: bool = True,
    ) -> Dict[str, float]:
        """Measure inference latency.

        Args:
            input_tensor: Example input batch.
            use_cuda_events: Use CUDA events for GPU timing (more accurate).

        Returns:
            Dict with 'mean_ms', 'std_ms', 'min_ms', 'max_ms', 'fps'.
        """
        self.model.eval()
        input_tensor = input_tensor.to(self.device)

        # Warm-up
        for _ in range(self.num_warmup):
            _ = self.model(input_tensor)

        if use_cuda_events and self.device.type == "cuda":
            torch.cuda.synchronize()
            latencies = self._cuda_timing(input_tensor)
        else:
            latencies = self._cpu_timing(input_tensor)

        latency_ms = np.array(latencies)
        return {
            "mean_ms": float(latency_ms.mean()),
            "std_ms": float(latency_ms.std()),
            "min_ms": float(latency_ms.min()),
            "max_ms": float(latency_ms.max()),
            "fps": float(1000.0 / latency_ms.mean()),
        }

    def _cuda_timing(self, x: torch.Tensor) -> List[float]:
        """CUDA event-based timing for accurate GPU latency measurement."""
        latencies = []
        for _ in range(self.num_runs):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            _ = self.model(x)
            end.record()
            torch.cuda.synchronize()
            latencies.append(start.elapsed_time(end))
        return latencies

    def _cpu_timing(self, x: torch.Tensor) -> List[float]:
        """Python time.perf_counter timing for CPU/non-CUDA devices."""
        latencies = []
        for _ in range(self.num_runs):
            t0 = time.perf_counter()
            _ = self.model(x)
            latencies.append((time.perf_counter() - t0) * 1000)
        return latencies


# ---------------------------------------------------------------------------
# Aggregated metric computation
# ---------------------------------------------------------------------------


class SurgicalMetrics:
    """Computes and stores all surgical phase recognition metrics.

    Accumulates predictions across an entire test set (multiple videos)
    then computes all metrics in a single call to ``compute()``.

    Usage::

        metrics = SurgicalMetrics(num_phases=7)
        for batch in dataloader:
            preds, labels = run_inference(batch)
            metrics.update(preds, labels)
        results = metrics.compute()
        metrics.print_report(results)
    """

    def __init__(self, num_phases: int = NUM_PHASES) -> None:
        self.num_phases = num_phases
        self.all_preds: List[int] = []
        self.all_labels: List[int] = []

    def update(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
    ) -> None:
        """Accumulate predictions for a batch or video.

        Args:
            predictions: (N,) or (B, T) predicted phase labels.
            labels: (N,) or (B, T) ground-truth phase labels.
        """
        preds_flat = predictions.cpu().numpy().flatten()
        labels_flat = labels.cpu().numpy().flatten()
        valid = labels_flat >= 0
        self.all_preds.extend(preds_flat[valid].tolist())
        self.all_labels.extend(labels_flat[valid].tolist())

    def compute(self) -> Dict:
        """Compute all metrics over accumulated predictions.

        Returns:
            Comprehensive dict of all surgical phase recognition metrics.
        """
        preds = np.array(self.all_preds, dtype=np.int64)
        labels = np.array(self.all_labels, dtype=np.int64)

        results = {
            "accuracy": frame_level_accuracy(preds, labels),
            "mean_jaccard": mean_jaccard(preds, labels, self.num_phases),
            "per_phase": per_phase_metrics(
                preds, labels, self.num_phases, PHASE_NAMES[: self.num_phases]
            ),
            "transition": transition_detection_accuracy(preds, labels),
            "edit_distance": edit_distance(preds, labels),
            "segment_f1": segment_f1(preds, labels, num_classes=self.num_phases),
            "relaxed_accuracy": relaxed_boundary_accuracy(preds, labels),
        }

        # Macro-averaged F1
        f1_scores = [v["f1"] for v in results["per_phase"].values() if v["support"] > 0]
        results["macro_f1"] = float(np.mean(f1_scores)) if f1_scores else 0.0

        return results

    def reset(self) -> None:
        """Clear all accumulated predictions."""
        self.all_preds = []
        self.all_labels = []

    def print_report(self, results: Dict) -> None:
        """Print a formatted metrics report to the logger.

        Args:
            results: Output of ``compute()``.
        """
        sep = "=" * 60
        logger.info(sep)
        logger.info("SURGICAL PHASE RECOGNITION — EVALUATION REPORT")
        logger.info(sep)
        logger.info("Overall Accuracy  : %.4f", results["accuracy"])
        logger.info("Mean Jaccard (mIoU): %.4f", results["mean_jaccard"])
        logger.info("Macro F1          : %.4f", results["macro_f1"])
        logger.info("Relaxed Accuracy  : %.4f", results["relaxed_accuracy"])
        logger.info("Edit Distance     : %.4f", results["edit_distance"])
        logger.info(sep)
        logger.info("Per-Phase Metrics:")
        logger.info("%-35s %8s %8s %8s %8s", "Phase", "F1", "Jaccard", "Prec", "Recall")
        for phase, m in results["per_phase"].items():
            logger.info(
                "  %-33s %8.4f %8.4f %8.4f %8.4f",
                phase,
                m["f1"],
                m["jaccard"],
                m["precision"],
                m["recall"],
            )
        logger.info(sep)
        logger.info("Segment-level F1:")
        for k, v in results["segment_f1"].items():
            logger.info("  %s: %.4f", k, v)
        logger.info(sep)
        logger.info("Transition Detection:")
        t = results["transition"]
        logger.info(
            "  Precision=%.4f  Recall=%.4f  F1=%.4f  (GT:%d  Pred:%d)",
            t["precision"],
            t["recall"],
            t["f1"],
            t["num_gt_transitions"],
            t["num_pred_transitions"],
        )
        logger.info(sep)


def compute_all_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    num_phases: int = NUM_PHASES,
) -> Dict:
    """Convenience function: compute all metrics from numpy arrays.

    Args:
        predictions: (N,) predicted labels.
        targets: (N,) ground-truth labels.
        num_phases: Number of phase classes.

    Returns:
        Full metrics dict (same as ``SurgicalMetrics.compute()``).
    """
    metrics = SurgicalMetrics(num_phases=num_phases)
    metrics.update(torch.tensor(predictions), torch.tensor(targets))
    return metrics.compute()
