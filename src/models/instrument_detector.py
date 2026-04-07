"""
Surgical instrument detection using YOLOv8 and multi-label classification.

Provides two complementary instrument recognition heads:
  1. ``SurgicalInstrumentDetector``: full bounding-box detection via YOLOv8-small,
     returning per-instrument boxes and confidence scores.
  2. ``InstrumentPresenceHead``: lightweight multi-label classification head for
     binary instrument presence prediction, sharing the spatial encoder backbone.

Instruments follow the Cholec80 / CholecT50 annotation schema (7 classes).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Instrument vocabulary
# ---------------------------------------------------------------------------

CHOLEC80_INSTRUMENTS = [
    "Grasper",
    "Bipolar",
    "Hook",
    "Scissors",
    "Clipper",
    "Irrigator",
    "SpecimenBag",
]

INSTRUMENT_TO_IDX: Dict[str, int] = {name: i for i, name in enumerate(CHOLEC80_INSTRUMENTS)}
NUM_INSTRUMENTS = len(CHOLEC80_INSTRUMENTS)  # 7


# ---------------------------------------------------------------------------
# Bounding-box detection via YOLOv8
# ---------------------------------------------------------------------------


class SurgicalInstrumentDetector(nn.Module):
    """YOLOv8-based surgical instrument detector.

    Wraps Ultralytics YOLOv8 for instrument detection in surgical frames,
    providing both bounding-box predictions and a feature embedding suitable
    for integration with the phase classifier via late fusion.

    The detector is trained / fine-tuned on CholecT50 bounding-box annotations
    and CholecDet (if available). At inference, detections are filtered by a
    minimum confidence threshold.

    Args:
        model_path: Path to a pre-trained YOLOv8 checkpoint (.pt), or 'yolov8s'
                    to load the small pretrained model from Ultralytics hub.
        num_classes: Number of instrument classes. Default 7.
        conf_threshold: Minimum confidence for a detection to be kept.
        iou_threshold: NMS IoU threshold.
        img_size: Inference image size (square crop).
        device: Torch device. If None, auto-selects CUDA when available.
    """

    def __init__(
        self,
        model_path: str = "yolov8s.pt",
        num_classes: int = NUM_INSTRUMENTS,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        img_size: int = 640,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.img_size = img_size
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = self._load_model(model_path, num_classes)
        logger.info(
            "SurgicalInstrumentDetector: model=%s, classes=%d, conf=%.2f",
            model_path,
            num_classes,
            conf_threshold,
        )

    def _load_model(self, model_path: str, num_classes: int) -> nn.Module:
        """Load YOLOv8 model from disk or Ultralytics hub.

        Args:
            model_path: Path or hub model name.
            num_classes: Number of target classes for fine-tuning head.

        Returns:
            YOLOv8 model instance.
        """
        try:
            from ultralytics import YOLO

            model = YOLO(model_path)
            # Verify or swap the detection head for our class count
            if model.model.nc != num_classes:
                logger.warning(
                    "Loaded model has %d classes; expected %d. "
                    "Fine-tuning head will be re-initialised.",
                    model.model.nc,
                    num_classes,
                )
                model.model.nc = num_classes
            return model
        except ImportError:
            logger.warning(
                "ultralytics not installed. Using stub detection head. "
                "Install with: pip install ultralytics"
            )
            return _StubYOLO(num_classes=num_classes)

    def forward(
        self, frames: torch.Tensor
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """Detect instruments in a batch of frames.

        Args:
            frames: (B, 3, H, W) float tensor in [0, 1].

        Returns:
            Tuple of:
                - detections: List of length B; each element is a (N, 6) tensor
                  [x1, y1, x2, y2, confidence, class_id] for the detected
                  instruments in that frame. N may be 0.
                - presence_vec: (B, num_classes) binary tensor indicating which
                  instrument classes appear in each frame (after NMS + threshold).
        """
        # Convert to uint8 numpy for YOLO (or pass tensors for stub)
        if hasattr(self.model, "predict"):
            results = self.model.predict(
                frames,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                imgsz=self.img_size,
                verbose=False,
            )
            detections = []
            presence_vec = torch.zeros(frames.size(0), self.num_classes, device=frames.device)
            for i, res in enumerate(results):
                if res.boxes is not None and len(res.boxes):
                    boxes = res.boxes.data  # (N, 6): x1 y1 x2 y2 conf cls
                    detections.append(boxes.to(frames.device))
                    classes = boxes[:, 5].long()
                    valid = classes < self.num_classes
                    presence_vec[i, classes[valid]] = 1.0
                else:
                    detections.append(torch.zeros(0, 6, device=frames.device))
        else:
            # Stub path
            detections, presence_vec = self.model(frames)

        return detections, presence_vec

    def get_instrument_features(self, frames: torch.Tensor) -> torch.Tensor:
        """Extract instrument presence feature vector for phase fusion.

        A 7-dim binary presence vector, optionally concatenated with
        confidence-weighted class scores, for use as additional input to
        the temporal phase model.

        Args:
            frames: (B, 3, H, W) float tensor.

        Returns:
            (B, num_classes) float tensor with confidence-weighted presence scores.
        """
        _, presence_vec = self.forward(frames)
        return presence_vec


# ---------------------------------------------------------------------------
# Lightweight multi-label presence classification head
# ---------------------------------------------------------------------------


class InstrumentPresenceHead(nn.Module):
    """Multi-label instrument presence head sharing the spatial encoder.

    A shallow MLP head that takes the 2048-dim spatial feature vector
    (from ResNet50 or EfficientNet) and predicts the binary presence of
    each of the 7 Cholec80 instruments.  This is the primary approach used
    in EndoNet and TeCNO, where full bounding-box annotation is unavailable.

    Args:
        feature_dim: Spatial encoder output dimension.
        num_instruments: Number of instrument classes. Default 7.
        hidden_dim: Hidden layer dimension.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        feature_dim: int = 2048,
        num_instruments: int = NUM_INSTRUMENTS,
        hidden_dim: int = 512,
        dropout: float = 0.4,
    ) -> None:
        super().__init__()
        self.num_instruments = num_instruments
        self.head = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_instruments),
        )
        logger.info(
            "InstrumentPresenceHead: feature_dim=%d, instruments=%d",
            feature_dim,
            num_instruments,
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Predict instrument presence from spatial features.

        Args:
            features: (B, feature_dim) or (B, T, feature_dim) spatial features.

        Returns:
            (B, num_instruments) or (B, T, num_instruments) logits.
            Apply sigmoid for inference-time binary predictions.
        """
        return self.head(features)

    def predict(self, features: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Binary presence prediction with configurable threshold.

        Args:
            features: (B, feature_dim) spatial features.
            threshold: Sigmoid threshold for positive detection.

        Returns:
            (B, num_instruments) bool tensor.
        """
        logits = self.forward(features)
        return torch.sigmoid(logits) >= threshold


# ---------------------------------------------------------------------------
# Multi-task detection + classification head
# ---------------------------------------------------------------------------


class MultiTaskInstrumentHead(nn.Module):
    """Combined detection + multi-label classification for surgical instruments.

    Integrates:
    - ``InstrumentPresenceHead``: presence classification from spatial features.
    - Detection aggregation: spatial instrument heatmaps for weakly-supervised
      localisation when full bounding boxes are unavailable.

    Args:
        feature_dim: Spatial encoder output dimension.
        num_instruments: Number of instrument classes.
        use_cam: Generate class activation maps for weakly-supervised localisation.
    """

    def __init__(
        self,
        feature_dim: int = 2048,
        num_instruments: int = NUM_INSTRUMENTS,
        use_cam: bool = True,
    ) -> None:
        super().__init__()
        self.use_cam = use_cam
        self.presence_head = InstrumentPresenceHead(
            feature_dim=feature_dim,
            num_instruments=num_instruments,
        )

        if use_cam:
            # GAP-based CAM: project feature map to instrument scores
            self.cam_conv = nn.Conv2d(
                feature_dim, num_instruments, kernel_size=1
            )

    def forward(
        self,
        features: torch.Tensor,
        feature_maps: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Run multi-task instrument head.

        Args:
            features: (B, feature_dim) pooled spatial features.
            feature_maps: (B, feature_dim, H, W) pre-pooled feature maps,
                          required when ``use_cam=True``.

        Returns:
            Dict with keys:
                - 'presence_logits': (B, num_instruments) presence logits.
                - 'cam': (B, num_instruments, H, W) class activation maps
                         if ``use_cam=True`` and feature_maps provided.
        """
        out: Dict[str, torch.Tensor] = {}
        out["presence_logits"] = self.presence_head(features)

        if self.use_cam and feature_maps is not None:
            out["cam"] = self.cam_conv(feature_maps)

        return out


# ---------------------------------------------------------------------------
# Stub YOLO for environments without ultralytics installed
# ---------------------------------------------------------------------------


class _StubYOLO(nn.Module):
    """Minimal stub that returns empty detections — used when ultralytics is absent."""

    def __init__(self, num_classes: int = NUM_INSTRUMENTS) -> None:
        super().__init__()
        self.num_classes = num_classes
        logger.warning(
            "Using _StubYOLO. Install ultralytics for real detection: "
            "pip install ultralytics"
        )

    def forward(
        self, frames: torch.Tensor
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        B = frames.size(0)
        detections = [torch.zeros(0, 6, device=frames.device) for _ in range(B)]
        presence_vec = torch.zeros(B, self.num_classes, device=frames.device)
        return detections, presence_vec
