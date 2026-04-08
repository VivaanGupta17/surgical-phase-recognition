"""
Temporal modeling architectures for surgical phase recognition.

Implements three complementary temporal models:
  - MS-TCN++: Multi-Stage Temporal Convolutional Network (Li et al., 2020)
  - Trans-SVNet: Transformer-based temporal model (Jin et al., 2021)
  - LSTMTemporalModel: Bidirectional / causal LSTM baseline

All models accept a sequence of per-frame spatial features and output
per-frame phase logits, supporting both causal (online) and non-causal modes.
"""

from __future__ import annotations

import logging
import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MS-TCN++ Multi-Stage Temporal Convolutional Network
# ---------------------------------------------------------------------------


class DilatedResidualLayer(nn.Module):
    """Single dilated residual layer used in each MS-TCN stage.

    Applies a 1-D dilated causal convolution followed by layer norm and ReLU,
    with a residual skip connection.

    Args:
        num_filters: Number of convolutional feature channels.
        kernel_size: Temporal kernel width (odd number).
        dilation: Dilation factor — doubles per layer to achieve exponential
                  receptive field growth.
        causal: If True, apply causal padding (no future context).
    """

    def __init__(
        self,
        num_filters: int,
        kernel_size: int = 3,
        dilation: int = 1,
        causal: bool = True,
    ) -> None:
        super().__init__()
        pad = (kernel_size - 1) * dilation
        if causal:
            # Left-only padding for causal operation
            self.padding = nn.ConstantPad1d((pad, 0), 0)
        else:
            # Symmetric padding for non-causal (offline) mode
            self.padding = nn.ConstantPad1d((pad // 2, pad - pad // 2), 0)

        self.conv = nn.Conv1d(
            num_filters, num_filters, kernel_size, dilation=dilation, padding=0
        )
        self.norm = nn.InstanceNorm1d(num_filters, track_running_stats=False)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply dilated residual layer.

        Args:
            x: (B, C, T) temporal feature sequence.

        Returns:
            (B, C, T) tensor — same shape as input.
        """
        residual = x
        out = self.padding(x)
        out = F.relu(self.norm(self.conv(out)))
        out = self.dropout(out)
        return out + residual


class SingleStageTCN(nn.Module):
    """Single stage of the MS-TCN architecture.

    Consists of a projection convolution followed by N dilated residual
    layers with exponentially increasing dilation factors.

    Args:
        in_channels: Input feature dimension (either feature_dim or num_filters
                     for subsequent stages).
        num_layers: Number of dilated residual layers. Each layer doubles
                    the dilation, giving 2^N temporal receptive field.
        num_filters: Internal feature channels.
        num_classes: Number of output phase classes.
        kernel_size: Temporal kernel size.
        causal: Use causal (online) convolutions.
    """

    def __init__(
        self,
        in_channels: int,
        num_layers: int,
        num_filters: int,
        num_classes: int,
        kernel_size: int = 3,
        causal: bool = True,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Conv1d(in_channels, num_filters, kernel_size=1)
        self.layers = nn.ModuleList(
            [
                DilatedResidualLayer(
                    num_filters=num_filters,
                    kernel_size=kernel_size,
                    dilation=2**i,
                    causal=causal,
                )
                for i in range(num_layers)
            ]
        )
        self.output_proj = nn.Conv1d(num_filters, num_classes, kernel_size=1)

    def forward(
        self, x: torch.Tensor, return_features: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass through single TCN stage.

        Args:
            x: (B, in_channels, T) input sequence.
            return_features: If True, also return pre-logit features.

        Returns:
            Tuple of (logits, features) where logits is (B, num_classes, T)
            and features is (B, num_filters, T) if ``return_features`` else None.
        """
        feat = self.input_proj(x)
        for layer in self.layers:
            feat = layer(feat)
        logits = self.output_proj(feat)
        return logits, feat if return_features else None


class MSTCN(nn.Module):
    """Multi-Stage Temporal Convolutional Network (MS-TCN++).

    Implements the multi-stage refinement architecture from:
      Li et al., "MS-TCN++: Multi-Stage Temporal Convolutional Network
      for Action Segmentation", IEEE TPAMI 2020.

    The first stage predicts initial phase labels from raw features.
    Each subsequent stage refines the predictions by also consuming the
    softmax-smoothed predictions of the previous stage, enabling iterative
    boundary sharpening without requiring future context.

    Args:
        num_stages: Number of TCN stages (typically 2–4).
        num_layers: Dilated layers per stage (typically 10).
        num_filters: Feature channels per stage (typically 64).
        feature_dim: Spatial feature dimensionality from the encoder.
        num_classes: Number of surgical phase classes (7 for Cholec80).
        kernel_size: Temporal kernel width.
        causal: Use causal convolutions (required for online inference).
    """

    def __init__(
        self,
        num_stages: int = 4,
        num_layers: int = 10,
        num_filters: int = 64,
        feature_dim: int = 2048,
        num_classes: int = 7,
        kernel_size: int = 3,
        causal: bool = True,
    ) -> None:
        super().__init__()
        self.num_stages = num_stages
        self.num_classes = num_classes
        self.causal = causal

        # Stage 1: takes raw spatial features
        self.stage1 = SingleStageTCN(
            in_channels=feature_dim,
            num_layers=num_layers,
            num_filters=num_filters,
            num_classes=num_classes,
            kernel_size=kernel_size,
            causal=causal,
        )

        # Refinement stages: take concatenated [features, prev_softmax]
        self.stages = nn.ModuleList(
            [
                SingleStageTCN(
                    in_channels=num_classes + feature_dim,
                    num_layers=num_layers,
                    num_filters=num_filters,
                    num_classes=num_classes,
                    kernel_size=kernel_size,
                    causal=causal,
                )
                for _ in range(num_stages - 1)
            ]
        )

        logger.info(
            "MSTCN: stages=%d, layers=%d, filters=%d, causal=%s",
            num_stages,
            num_layers,
            num_filters,
            causal,
        )

    def forward(
        self, features: torch.Tensor
    ) -> List[torch.Tensor]:
        """Run multi-stage temporal prediction.

        Args:
            features: (B, feature_dim, T) spatial feature sequence.

        Returns:
            List of logit tensors, one per stage: [(B, num_classes, T), ...].
            Training uses all stage outputs for deep supervision. Inference
            uses the final stage output.
        """
        all_logits = []

        # Stage 1
        logits, _ = self.stage1(features, return_features=False)
        all_logits.append(logits)

        # Refinement stages
        prev_logits = logits
        for stage in self.stages:
            stage_input = torch.cat(
                [F.softmax(prev_logits, dim=1), features], dim=1
            )
            logits, _ = stage(stage_input, return_features=False)
            all_logits.append(logits)
            prev_logits = logits

        return all_logits


# ---------------------------------------------------------------------------
# Transformer-based Temporal Model (Trans-SVNet)
# ---------------------------------------------------------------------------


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for temporal transformers.

    Args:
        d_model: Model embedding dimension.
        max_len: Maximum sequence length supported.
        dropout: Dropout applied to the output.
    """

    def __init__(self, d_model: int, max_len: int = 10000, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term[: d_model // 2])
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input sequence.

        Args:
            x: (B, T, d_model) input tensor.

        Returns:
            (B, T, d_model) tensor with positional encoding added.
        """
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class CausalTransformerEncoderLayer(nn.Module):
    """Causal (auto-regressive) Transformer encoder layer.

    Applies a causal attention mask so that each position can only attend
    to previous positions — enabling true online inference.

    Args:
        d_model: Embedding dimension.
        nhead: Number of attention heads.
        dim_feedforward: FFN hidden layer size.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, causal: bool = True
    ) -> torch.Tensor:
        """Forward pass with optional causal masking.

        Args:
            x: (B, T, d_model) input sequence.
            causal: Apply causal attention mask.

        Returns:
            (B, T, d_model) attended sequence.
        """
        T = x.size(1)
        attn_mask = None
        if causal:
            # Upper-triangular mask: future positions → -inf
            attn_mask = torch.triu(
                torch.full((T, T), float("-inf"), device=x.device), diagonal=1
            )
        attn_out, _ = self.self_attn(x, x, x, attn_mask=attn_mask)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.linear2(F.relu(self.linear1(x)))
        x = self.norm2(x + self.dropout(ffn_out))
        return x


class TransSVNet(nn.Module):
    """Trans-SVNet: Transformer-based surgical video temporal model.

    Adapted from Jin et al., "Trans-SVNet: Accurate phase recognition from
    surgical videos via hybrid embedding aggregation transformer", MICCAI 2021.

    The model projects spatial features into a lower-dimensional embedding
    space, adds positional encoding, runs N causal transformer encoder layers,
    and outputs per-frame phase logits.

    Args:
        feature_dim: Spatial encoder output dimension.
        d_model: Transformer embedding dimension.
        nhead: Attention heads.
        num_layers: Transformer encoder layers.
        dim_feedforward: FFN hidden size.
        num_classes: Number of phase classes.
        causal: Use causal attention masking.
        dropout: Dropout probability.
        max_seq_len: Maximum sequence length for positional encoding.
    """

    def __init__(
        self,
        feature_dim: int = 2048,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        num_classes: int = 7,
        causal: bool = True,
        dropout: float = 0.1,
        max_seq_len: int = 10000,
    ) -> None:
        super().__init__()
        self.causal = causal
        self.d_model = d_model

        self.input_proj = nn.Linear(feature_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=max_seq_len, dropout=dropout)
        self.encoder_layers = nn.ModuleList(
            [
                CausalTransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.output_proj = nn.Linear(d_model, num_classes)

        logger.info(
            "TransSVNet: d_model=%d, heads=%d, layers=%d, causal=%s",
            d_model,
            nhead,
            num_layers,
            causal,
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Run transformer temporal prediction.

        Args:
            features: (B, feature_dim, T) or (B, T, feature_dim) spatial features.
                      Both orderings are supported.

        Returns:
            (B, T, num_classes) logits.
        """
        # Normalise to (B, T, feature_dim)
        if features.dim() == 3 and features.size(1) != features.size(2):
            if features.size(1) > features.size(2):
                features = features.permute(0, 2, 1)  # (B, T, F)

        x = self.input_proj(features)  # (B, T, d_model)
        x = self.pos_enc(x)

        for layer in self.encoder_layers:
            x = layer(x, causal=self.causal)

        logits = self.output_proj(x)  # (B, T, num_classes)
        return logits


# ---------------------------------------------------------------------------
# LSTM Baseline
# ---------------------------------------------------------------------------


class LSTMTemporalModel(nn.Module):
    """Bidirectional / causal LSTM temporal baseline.

    Provides a competitive LSTM baseline for ablation studies. Supports
    bidirectional (offline) and unidirectional causal (online) modes.

    Args:
        feature_dim: Input feature dimension.
        hidden_size: LSTM hidden state size.
        num_layers: Stacked LSTM layers.
        num_classes: Number of output phase classes.
        bidirectional: Use BiLSTM (offline mode only — not causal).
        dropout: Dropout between LSTM layers.
    """

    def __init__(
        self,
        feature_dim: int = 2048,
        hidden_size: int = 512,
        num_layers: int = 2,
        num_classes: int = 7,
        bidirectional: bool = False,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if bidirectional:
            logger.warning(
                "Bidirectional LSTM uses future context — not suitable for online inference."
            )
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        out_size = hidden_size * (2 if bidirectional else 1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(out_size, num_classes),
        )

    def forward(
        self,
        features: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Run LSTM temporal prediction.

        Args:
            features: (B, T, feature_dim) feature sequence. If (B, F, T),
                      it will be transposed automatically.
            hidden: Optional initial hidden state (h_0, c_0).

        Returns:
            Tuple of:
                - logits: (B, T, num_classes)
                - hidden: updated (h_n, c_n) for stateful inference
        """
        if features.dim() == 3 and features.size(1) < features.size(2):
            features = features.permute(0, 2, 1)  # (B, T, F)

        lstm_out, hidden_out = self.lstm(features, hidden)  # (B, T, hidden*dirs)
        logits = self.classifier(lstm_out)  # (B, T, num_classes)
        return logits, hidden_out


# ---------------------------------------------------------------------------
# Temporal CRF Smoothing
# ---------------------------------------------------------------------------


class TemporalCRF(nn.Module):
    """Linear-chain CRF for temporal phase smoothing.

    Post-processes frame-level logits with a learned transition cost matrix
    to remove physically impossible phase micro-transitions (e.g., phase 6
    immediately following phase 0 without intermediate phases).

    Applies the Viterbi algorithm at inference time for globally optimal
    phase sequence decoding.

    Args:
        num_classes: Number of phase classes.
        init_transitions: Optional (num_classes, num_classes) matrix for
                          transition log-potentials initialisation.
    """

    def __init__(
        self,
        num_classes: int = 7,
        init_transitions: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes

        # Transition matrix: transitions[i, j] = log-potential of i→j
        if init_transitions is not None:
            self.transitions = nn.Parameter(init_transitions.float())
        else:
            # Initialise to slightly prefer self-transitions (temporal smoothness)
            t = torch.zeros(num_classes, num_classes)
            t.fill_diagonal_(1.0)
            self.transitions = nn.Parameter(t)

    def forward(self, emissions: torch.Tensor) -> torch.Tensor:
        """Apply CRF log-likelihood smoothing (training mode).

        Args:
            emissions: (B, T, num_classes) per-frame log-softmax scores.

        Returns:
            (B, T, num_classes) CRF-smoothed log-scores.
        """
        # During training, apply soft transition regularisation
        # Full CRF training uses CRF loss (neg log-likelihood)
        return emissions + self.transitions.unsqueeze(0).unsqueeze(0).mean(-1, keepdim=True)

    @torch.no_grad()
    def viterbi_decode(self, emissions: torch.Tensor) -> torch.Tensor:
        """Viterbi decoding for MAP sequence inference.

        Args:
            emissions: (B, T, num_classes) log-probabilities.

        Returns:
            (B, T) integer tensor of decoded phase labels.
        """
        B, T, C = emissions.shape
        # Viterbi DP
        viterbi = torch.full((B, T, C), -1e9, device=emissions.device)
        backptr = torch.zeros(B, T, C, dtype=torch.long, device=emissions.device)

        viterbi[:, 0] = emissions[:, 0]  # init

        for t in range(1, T):
            # (B, C_prev, C_next) = viterbi[:, t-1, :, None] + transitions[None, :, :]
            scores = viterbi[:, t - 1].unsqueeze(2) + self.transitions.unsqueeze(0)
            best_scores, best_prev = scores.max(dim=1)  # (B, C)
            viterbi[:, t] = emissions[:, t] + best_scores
            backptr[:, t] = best_prev

        # Backtrack
        sequences = torch.zeros(B, T, dtype=torch.long, device=emissions.device)
        sequences[:, -1] = viterbi[:, -1].argmax(dim=1)
        for t in range(T - 2, -1, -1):
            sequences[:, t] = backptr[
                torch.arange(B), t + 1, sequences[:, t + 1]
            ]

        return sequences

# Handle empty sequences gracefully
