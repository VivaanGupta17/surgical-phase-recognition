"""
Spatial feature extraction backbones for surgical video frame encoding.

Supports ResNet50, EfficientNet-B4, and a pluggable backbone interface.
All encoders output a fixed-dimensional feature vector per frame, suitable
for downstream temporal modeling.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import (
    EfficientNet_B0_Weights,
    EfficientNet_B4_Weights,
    ResNet50_Weights,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class SpatialEncoder(ABC, nn.Module):
    """Abstract base class for spatial feature encoders.

    All spatial encoders accept a batch of RGB frames (B, C, H, W) and
    return a batch of 1-D feature vectors (B, feature_dim).
    """

    def __init__(self, feature_dim: int, pretrained: bool = True) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.pretrained = pretrained

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a batch of frames.

        Args:
            x: Float tensor of shape (B, 3, H, W), values in [0, 1].

        Returns:
            Feature tensor of shape (B, feature_dim).
        """
        ...

    def freeze_backbone(self, num_frozen_layers: int = -1) -> None:
        """Freeze backbone layers for staged training.

        Args:
            num_frozen_layers: Number of early layers to freeze. -1 freezes all.
        """
        layers = list(self.backbone.parameters())
        if num_frozen_layers == -1:
            for param in layers:
                param.requires_grad = False
        else:
            for param in layers[:num_frozen_layers]:
                param.requires_grad = False
        frozen = sum(1 for p in self.backbone.parameters() if not p.requires_grad)
        total = sum(1 for _ in self.backbone.parameters())
        logger.info("Frozen %d / %d backbone parameters.", frozen, total)

    def unfreeze_backbone(self) -> None:
        """Unfreeze all backbone parameters for fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        logger.info("All backbone parameters unfrozen for fine-tuning.")

    def count_parameters(self) -> Dict[str, int]:
        """Return parameter counts for diagnostics."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable, "frozen": total - trainable}


# ---------------------------------------------------------------------------
# ResNet50 Encoder
# ---------------------------------------------------------------------------


class ResNetEncoder(SpatialEncoder):
    """ResNet50-based spatial feature extractor for surgical frames.

    Uses an ImageNet-pretrained ResNet50 backbone with the final classification
    head replaced by a learnable projection to ``feature_dim``.  Supports
    multi-scale pooling strategies (global average, GeM, concat).

    Args:
        feature_dim: Output feature dimensionality. Default 2048.
        pretrained: Load ImageNet weights. Default True.
        pooling: Pooling strategy — one of 'avg', 'max', 'gem', 'concat'.
        dropout: Dropout probability on the output projection. Default 0.3.
    """

    RESNET_FEATURE_DIM = 2048  # ResNet50 layer4 output channels

    def __init__(
        self,
        feature_dim: int = 2048,
        pretrained: bool = True,
        pooling: str = "avg",
        dropout: float = 0.3,
    ) -> None:
        super().__init__(feature_dim=feature_dim, pretrained=pretrained)
        self.pooling_type = pooling

        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        resnet = models.resnet50(weights=weights)

        # Remove the original FC and avgpool
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )

        backbone_out = self.RESNET_FEATURE_DIM
        if pooling == "concat":
            backbone_out = self.RESNET_FEATURE_DIM * 2  # avg + max

        self.pool = self._build_pooling(pooling)

        # Projection head — keeps feature_dim flexible
        self.projector = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(backbone_out, feature_dim),
            nn.LayerNorm(feature_dim),
        )

        logger.info(
            "ResNetEncoder: pretrained=%s, pooling=%s, feature_dim=%d",
            pretrained,
            pooling,
            feature_dim,
        )

    def _build_pooling(self, pooling: str) -> nn.Module:
        """Construct the spatial pooling layer."""
        if pooling == "avg":
            return nn.AdaptiveAvgPool2d((1, 1))
        elif pooling == "max":
            return nn.AdaptiveMaxPool2d((1, 1))
        elif pooling == "gem":
            return GeM(p=3.0)
        elif pooling == "concat":
            return ConcatPool2d()
        else:
            raise ValueError(f"Unknown pooling strategy: {pooling!r}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a batch of frames to feature vectors.

        Args:
            x: (B, 3, H, W) float tensor — frames normalised to [0, 1].

        Returns:
            (B, feature_dim) feature tensor.
        """
        feat_map = self.backbone(x)  # (B, 2048, h, w)
        pooled = self.pool(feat_map)  # (B, C, 1, 1) or (B, 2C, 1, 1)
        flat = pooled.flatten(1)  # (B, C)
        return self.projector(flat)  # (B, feature_dim)


# ---------------------------------------------------------------------------
# EfficientNet Encoder
# ---------------------------------------------------------------------------


class EfficientNetEncoder(SpatialEncoder):
    """EfficientNet-B4-based spatial feature extractor.

    Lighter-weight alternative to ResNet50 with higher accuracy on ImageNet.
    Suitable for edge deployment where model size and FLOPs are constrained.

    Args:
        variant: EfficientNet variant — 'b0' (lite) or 'b4' (full).
        feature_dim: Output feature dimensionality. Default 1792 (B4 native).
        pretrained: Load ImageNet weights. Default True.
        dropout: Dropout probability. Default 0.3.
    """

    _VARIANTS = {
        "b0": (models.efficientnet_b0, EfficientNet_B0_Weights.IMAGENET1K_V1, 1280),
        "b4": (models.efficientnet_b4, EfficientNet_B4_Weights.IMAGENET1K_V1, 1792),
    }

    def __init__(
        self,
        variant: str = "b4",
        feature_dim: int = 1792,
        pretrained: bool = True,
        dropout: float = 0.3,
    ) -> None:
        super().__init__(feature_dim=feature_dim, pretrained=pretrained)
        if variant not in self._VARIANTS:
            raise ValueError(f"EfficientNet variant must be one of {list(self._VARIANTS)}")

        builder, weights_cls, native_dim = self._VARIANTS[variant]
        weights = weights_cls if pretrained else None
        net = builder(weights=weights)

        # Keep feature extraction layers; strip classifier
        self.backbone = net.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.projector = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(native_dim, feature_dim),
            nn.LayerNorm(feature_dim),
        )

        self.native_dim = native_dim
        logger.info(
            "EfficientNetEncoder: variant=%s, pretrained=%s, feature_dim=%d",
            variant,
            pretrained,
            feature_dim,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode frames to feature vectors.

        Args:
            x: (B, 3, H, W) float tensor.

        Returns:
            (B, feature_dim) feature tensor.
        """
        feat_map = self.backbone(x)  # (B, native_dim, h, w)
        pooled = self.pool(feat_map).flatten(1)  # (B, native_dim)
        return self.projector(pooled)  # (B, feature_dim)


# ---------------------------------------------------------------------------
# Pooling utilities
# ---------------------------------------------------------------------------


class GeM(nn.Module):
    """Generalised Mean Pooling (Radenovic et al., 2019).

    A learnable pooling operation that generalises average (p→1) and
    max (p→∞) pooling. Typically p≈3 works well for recognition tasks.

    Args:
        p: Initial pooling exponent. Learnable if ``learnable=True``.
        eps: Clamp lower bound to avoid log(0).
    """

    def __init__(self, p: float = 3.0, eps: float = 1e-6, learnable: bool = True) -> None:
        super().__init__()
        self.eps = eps
        if learnable:
            self.p = nn.Parameter(torch.tensor(p))
        else:
            self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply GeM pooling.

        Args:
            x: (B, C, H, W) feature map.

        Returns:
            (B, C, 1, 1) pooled tensor.
        """
        p = self.p.clamp(min=1) if isinstance(self.p, torch.Tensor) else max(self.p, 1)
        return F.adaptive_avg_pool2d(x.clamp(min=self.eps).pow(p), 1).pow(1.0 / p)


class ConcatPool2d(nn.Module):
    """Concatenate global average and max pooling.

    Doubles the channel dimension, capturing both average activation
    and peak activation statistics.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = F.adaptive_avg_pool2d(x, 1)
        mx = F.adaptive_max_pool2d(x, 1)
        return torch.cat([avg, mx], dim=1)


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------


def build_encoder(
    backbone: str = "resnet50",
    feature_dim: int = 2048,
    pretrained: bool = True,
    **kwargs,
) -> SpatialEncoder:
    """Construct a spatial encoder by name.

    Args:
        backbone: One of 'resnet50', 'efficientnet_b0', 'efficientnet_b4'.
        feature_dim: Output feature dimension.
        pretrained: Whether to load ImageNet weights.
        **kwargs: Additional arguments passed to the encoder constructor.

    Returns:
        Configured SpatialEncoder instance.

    Raises:
        ValueError: If backbone name is not recognised.
    """
    registry: Dict[str, type] = {
        "resnet50": ResNetEncoder,
        "efficientnet_b0": lambda **kw: EfficientNetEncoder(variant="b0", **kw),
        "efficientnet_b4": lambda **kw: EfficientNetEncoder(variant="b4", **kw),
    }
    if backbone not in registry:
        raise ValueError(
            f"Unknown backbone {backbone!r}. Choose from: {list(registry)}"
        )
    return registry[backbone](feature_dim=feature_dim, pretrained=pretrained, **kwargs)
