"""
End-to-end surgical phase classifier combining spatial and temporal models.

``SurgPhaseClassifier`` is the top-level model that connects:
  1. Spatial encoder (ResNet50 / EfficientNet)
  2. Instrument presence head (multi-label classification)
  3. Temporal model (MS-TCN++ / Trans-SVNet / LSTM)
  4. Optional temporal CRF smoothing

Supports three inference modes:
  - ``online``  : causal, single-frame-at-a-time, stateful LSTM hidden
  - ``offline`` : full video batch, non-causal temporal context
  - ``window``  : sliding window with fixed context length (default for deployment)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .instrument_detector import InstrumentPresenceHead, MultiTaskInstrumentHead
from .spatial_encoder import SpatialEncoder, build_encoder
from .temporal_model import MSTCN, LSTMTemporalModel, TemporalCRF, TransSVNet

logger = logging.getLogger(__name__)

# Cholec80 canonical phase count
NUM_PHASES = 7
NUM_INSTRUMENTS = 7


class SurgPhaseClassifier(nn.Module):
    """End-to-end surgical phase recognition model.

    Architecture summary::

        frames (B, T, 3, H, W)
            │
            ▼ [spatial encoder — shared weights across time]
        features (B, T, feature_dim)
            │
            ├─▶ instrument_head → presence logits (B, T, 7)  [aux loss]
            │
            ▼ [concatenate features + instrument_presence → (B, T, feature_dim + 7)]
        fused_features (B, feature_dim + 7, T)   [TCN expects channel-first]
            │
            ▼ [temporal model]
        phase_logits (B, num_phases, T)  — list of stages for MS-TCN
            │
            ▼ [CRF smoothing — optional, inference only]
        smoothed_logits (B, num_phases, T)

    Args:
        backbone: Spatial encoder variant ('resnet50', 'efficientnet_b4', etc.).
        feature_dim: Spatial feature dimension.
        temporal_model: Temporal model type ('mstcn', 'transformer', 'lstm').
        num_phases: Number of surgical phases. Default 7.
        num_instruments: Number of instruments. Default 7.
        causal: Use causal temporal modeling (required for online inference).
        use_crf: Apply CRF temporal smoothing at inference. Default True.
        fuse_instruments: Concatenate instrument features with spatial features.
        pretrained_backbone: Load ImageNet weights for spatial encoder.

        # MS-TCN-specific
        tcn_stages: Number of TCN refinement stages (1–4).
        tcn_layers: Dilated layers per TCN stage.
        tcn_filters: TCN feature channels.

        # Transformer-specific
        transformer_d_model: Transformer embedding dim.
        transformer_heads: Number of attention heads.
        transformer_layers: Transformer encoder layers.
    """

    def __init__(
        self,
        backbone: str = "resnet50",
        feature_dim: int = 2048,
        temporal_model: str = "mstcn",
        num_phases: int = NUM_PHASES,
        num_instruments: int = NUM_INSTRUMENTS,
        causal: bool = True,
        use_crf: bool = True,
        fuse_instruments: bool = True,
        pretrained_backbone: bool = True,
        # MS-TCN params
        tcn_stages: int = 4,
        tcn_layers: int = 10,
        tcn_filters: int = 64,
        # Transformer params
        transformer_d_model: int = 256,
        transformer_heads: int = 8,
        transformer_layers: int = 6,
    ) -> None:
        super().__init__()
        self.num_phases = num_phases
        self.num_instruments = num_instruments
        self.temporal_model_type = temporal_model
        self.causal = causal
        self.use_crf = use_crf
        self.fuse_instruments = fuse_instruments

        # 1. Spatial encoder
        self.spatial_encoder: SpatialEncoder = build_encoder(
            backbone=backbone,
            feature_dim=feature_dim,
            pretrained=pretrained_backbone,
        )

        # 2. Instrument head (shared backbone)
        self.instrument_head = InstrumentPresenceHead(
            feature_dim=feature_dim,
            num_instruments=num_instruments,
        )

        # 3. Feature fusion dimension
        fused_dim = feature_dim + (num_instruments if fuse_instruments else 0)

        # 4. Temporal model
        self.temporal: nn.Module = self._build_temporal(
            temporal_model=temporal_model,
            fused_dim=fused_dim,
            num_phases=num_phases,
            causal=causal,
            tcn_stages=tcn_stages,
            tcn_layers=tcn_layers,
            tcn_filters=tcn_filters,
            transformer_d_model=transformer_d_model,
            transformer_heads=transformer_heads,
            transformer_layers=transformer_layers,
        )

        # 5. Optional CRF smoother
        self.crf: Optional[TemporalCRF] = TemporalCRF(num_classes=num_phases) if use_crf else None

        self._log_model_info(backbone, temporal_model, feature_dim, fused_dim)

    # ------------------------------------------------------------------
    # Builder helpers
    # ------------------------------------------------------------------

    def _build_temporal(
        self,
        temporal_model: str,
        fused_dim: int,
        num_phases: int,
        causal: bool,
        tcn_stages: int,
        tcn_layers: int,
        tcn_filters: int,
        transformer_d_model: int,
        transformer_heads: int,
        transformer_layers: int,
    ) -> nn.Module:
        """Instantiate the temporal model by name."""
        if temporal_model == "mstcn":
            return MSTCN(
                num_stages=tcn_stages,
                num_layers=tcn_layers,
                num_filters=tcn_filters,
                feature_dim=fused_dim,
                num_classes=num_phases,
                causal=causal,
            )
        elif temporal_model == "transformer":
            return TransSVNet(
                feature_dim=fused_dim,
                d_model=transformer_d_model,
                nhead=transformer_heads,
                num_layers=transformer_layers,
                num_classes=num_phases,
                causal=causal,
            )
        elif temporal_model == "lstm":
            return LSTMTemporalModel(
                feature_dim=fused_dim,
                hidden_size=512,
                num_layers=2,
                num_classes=num_phases,
                bidirectional=not causal,
            )
        else:
            raise ValueError(
                f"Unknown temporal_model {temporal_model!r}. "
                "Choose from: 'mstcn', 'transformer', 'lstm'."
            )

    def _log_model_info(
        self,
        backbone: str,
        temporal_model: str,
        feature_dim: int,
        fused_dim: int,
    ) -> None:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            "SurgPhaseClassifier: backbone=%s, temporal=%s, feature_dim=%d → "
            "fused_dim=%d, causal=%s | params: %.1fM total, %.1fM trainable",
            backbone,
            temporal_model,
            feature_dim,
            fused_dim,
            self.causal,
            total / 1e6,
            trainable / 1e6,
        )

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def encode_frames(self, frames: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode a temporal batch of frames.

        Args:
            frames: (B, T, 3, H, W) float tensor.

        Returns:
            Tuple of:
                - spatial_features: (B, T, feature_dim)
                - instrument_logits: (B, T, num_instruments)
        """
        B, T, C, H, W = frames.shape
        # Fold time into batch for efficient parallel encoding
        frames_flat = frames.view(B * T, C, H, W)
        feat_flat = self.spatial_encoder(frames_flat)  # (B*T, feature_dim)
        inst_logits_flat = self.instrument_head(feat_flat)  # (B*T, num_instruments)

        feature_dim = feat_flat.size(-1)
        spatial_features = feat_flat.view(B, T, feature_dim)
        instrument_logits = inst_logits_flat.view(B, T, self.num_instruments)
        return spatial_features, instrument_logits

    def fuse_features(
        self,
        spatial_features: torch.Tensor,
        instrument_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse spatial features with instrument presence predictions.

        Args:
            spatial_features: (B, T, feature_dim)
            instrument_logits: (B, T, num_instruments)

        Returns:
            (B, fused_dim, T) — channel-first for TCN / causal convolutions.
        """
        if self.fuse_instruments:
            inst_presence = torch.sigmoid(instrument_logits)  # soft presence
            fused = torch.cat([spatial_features, inst_presence], dim=-1)  # (B, T, F+7)
        else:
            fused = spatial_features  # (B, T, F)

        # Transpose to (B, F, T) for 1D temporal convolutions
        return fused.permute(0, 2, 1).contiguous()

    def forward(
        self,
        frames: torch.Tensor,
        lstm_hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Dict[str, Any]:
        """Full forward pass.

        Args:
            frames: (B, T, 3, H, W) batch of frame sequences.
            lstm_hidden: Optional LSTM hidden state for stateful online inference
                         (only used when ``temporal_model='lstm'``).

        Returns:
            Dict with keys:
                - 'phase_logits': Final phase logits.
                  - MS-TCN: (B, num_phases, T)
                  - Transformer/LSTM: (B, T, num_phases)
                - 'all_stage_logits': List of per-stage logits (MS-TCN only).
                - 'instrument_logits': (B, T, num_instruments) auxiliary output.
                - 'phase_probs': Softmax phase probabilities (inference-time).
                - 'phase_labels': (B, T) argmax phase labels (inference-time).
                - 'lstm_hidden': Updated hidden state (LSTM only).
        """
        # -- 1. Spatial encoding --
        spatial_features, instrument_logits = self.encode_frames(frames)

        # -- 2. Feature fusion --
        fused = self.fuse_features(spatial_features, instrument_logits)  # (B, F, T)

        # -- 3. Temporal modeling --
        out: Dict[str, Any] = {"instrument_logits": instrument_logits}

        if self.temporal_model_type == "mstcn":
            all_logits = self.temporal(fused)  # List[(B, C, T)]
            out["all_stage_logits"] = all_logits
            out["phase_logits"] = all_logits[-1]  # Final stage

        elif self.temporal_model_type == "transformer":
            # TransSVNet expects (B, T, F)
            fused_bt = fused.permute(0, 2, 1)  # (B, T, F)
            logits = self.temporal(fused_bt)  # (B, T, C)
            out["phase_logits"] = logits

        elif self.temporal_model_type == "lstm":
            fused_bt = fused.permute(0, 2, 1)  # (B, T, F)
            logits, new_hidden = self.temporal(fused_bt, lstm_hidden)  # (B, T, C)
            out["phase_logits"] = logits
            out["lstm_hidden"] = new_hidden

        # -- 4. CRF smoothing (inference only, not during training) --
        if not self.training:
            logits = out["phase_logits"]
            # Normalise to (B, T, C) for CRF
            if logits.dim() == 3 and logits.size(1) == self.num_phases:
                logits_bt = logits.permute(0, 2, 1)  # (B, T, C)
            else:
                logits_bt = logits

            log_probs = F.log_softmax(logits_bt, dim=-1)

            if self.crf is not None:
                phase_labels = self.crf.viterbi_decode(log_probs)
            else:
                phase_labels = log_probs.argmax(dim=-1)

            out["phase_probs"] = log_probs.exp()
            out["phase_labels"] = phase_labels

        return out

    # ------------------------------------------------------------------
    # Staged training helpers
    # ------------------------------------------------------------------

    def set_training_stage(self, stage: str) -> None:
        """Configure parameter freezing for staged training.

        Stage 1 — ``'backbone'``: Train only the spatial encoder and
            instrument head; freeze temporal model.
        Stage 2 — ``'temporal'``: Freeze backbone; train temporal model only.
        Stage 3 — ``'finetune'``: Unfreeze all parameters for joint fine-tuning.

        Args:
            stage: One of 'backbone', 'temporal', 'finetune'.
        """
        if stage == "backbone":
            # Freeze temporal; train spatial encoder + instrument head
            for param in self.temporal.parameters():
                param.requires_grad = False
            for param in self.spatial_encoder.parameters():
                param.requires_grad = True
            for param in self.instrument_head.parameters():
                param.requires_grad = True
            logger.info("Training stage: backbone + instrument head")

        elif stage == "temporal":
            # Freeze backbone; train temporal model
            for param in self.spatial_encoder.parameters():
                param.requires_grad = False
            for param in self.instrument_head.parameters():
                param.requires_grad = False
            for param in self.temporal.parameters():
                param.requires_grad = True
            if self.crf:
                for param in self.crf.parameters():
                    param.requires_grad = True
            logger.info("Training stage: temporal model")

        elif stage == "finetune":
            for param in self.parameters():
                param.requires_grad = True
            logger.info("Training stage: full fine-tuning (all parameters)")
        else:
            raise ValueError(f"Unknown training stage: {stage!r}")

    def get_parameter_groups(self, base_lr: float = 1e-4) -> List[Dict]:
        """Return parameter groups with differential learning rates.

        Backbone layers receive a lower LR (base_lr / 10) to preserve
        ImageNet representations during fine-tuning.

        Args:
            base_lr: Learning rate for temporal head.

        Returns:
            Parameter group list for ``torch.optim`` constructors.
        """
        return [
            {
                "params": list(self.spatial_encoder.parameters()),
                "lr": base_lr / 10,
                "name": "backbone",
            },
            {
                "params": list(self.instrument_head.parameters()),
                "lr": base_lr / 5,
                "name": "instrument_head",
            },
            {
                "params": list(self.temporal.parameters()),
                "lr": base_lr,
                "name": "temporal",
            },
        ]

    # ------------------------------------------------------------------
    # Export helpers
    # ------------------------------------------------------------------

    def export_onnx(
        self,
        output_path: str,
        sequence_length: int = 100,
        img_size: int = 256,
        opset: int = 17,
    ) -> None:
        """Export the model to ONNX format.

        Args:
            output_path: Destination .onnx file path.
            sequence_length: Fixed temporal length for the export trace.
            img_size: Spatial resolution of input frames.
            opset: ONNX opset version.
        """
        self.eval()
        dummy_input = torch.randn(1, sequence_length, 3, img_size, img_size)
        dynamic_axes = {
            "frames": {0: "batch", 1: "seq_len"},
            "phase_logits": {0: "batch", 1: "seq_len"},
        }
        torch.onnx.export(
            self,
            (dummy_input,),
            output_path,
            input_names=["frames"],
            output_names=["phase_logits"],
            dynamic_axes=dynamic_axes,
            opset_version=opset,
            do_constant_folding=True,
        )
        logger.info("Exported ONNX model to %s", output_path)
