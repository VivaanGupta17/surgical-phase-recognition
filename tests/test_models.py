"""
Unit tests for SurgPhase model components.

Run with: pytest tests/test_models.py -v
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Spatial encoder tests
# ---------------------------------------------------------------------------


class TestResNetEncoder:
    """Tests for ResNetEncoder."""

    def test_output_shape(self):
        """Encoder should output (B, feature_dim) tensors."""
        from src.models.spatial_encoder import ResNetEncoder

        encoder = ResNetEncoder(feature_dim=2048, pretrained=False)
        x = torch.randn(4, 3, 256, 256)
        out = encoder(x)
        assert out.shape == (4, 2048), f"Expected (4, 2048), got {out.shape}"

    def test_feature_dim_flexibility(self):
        """Encoder should respect custom feature_dim."""
        from src.models.spatial_encoder import ResNetEncoder

        for dim in [512, 1024, 2048]:
            encoder = ResNetEncoder(feature_dim=dim, pretrained=False)
            x = torch.randn(2, 3, 224, 224)
            out = encoder(x)
            assert out.shape[-1] == dim

    def test_pooling_strategies(self):
        """All pooling strategies should produce valid output."""
        from src.models.spatial_encoder import ResNetEncoder

        for pooling in ["avg", "max", "gem", "concat"]:
            encoder = ResNetEncoder(feature_dim=512, pretrained=False, pooling=pooling)
            x = torch.randn(2, 3, 224, 224)
            out = encoder(x)
            assert out.shape == (2, 512), f"Pooling={pooling}: got {out.shape}"

    def test_freeze_backbone(self):
        """Freezing backbone should prevent gradient flow."""
        from src.models.spatial_encoder import ResNetEncoder

        encoder = ResNetEncoder(pretrained=False)
        encoder.freeze_backbone()
        frozen = all(not p.requires_grad for p in encoder.backbone.parameters())
        assert frozen, "Backbone should be frozen"

        encoder.unfreeze_backbone()
        unfrozen = all(p.requires_grad for p in encoder.backbone.parameters())
        assert unfrozen, "Backbone should be unfrozen"


class TestEfficientNetEncoder:
    """Tests for EfficientNetEncoder."""

    @pytest.mark.parametrize("variant", ["b0", "b4"])
    def test_output_shape(self, variant):
        """EfficientNet encoder should produce valid feature vectors."""
        from src.models.spatial_encoder import EfficientNetEncoder

        encoder = EfficientNetEncoder(variant=variant, feature_dim=512, pretrained=False)
        x = torch.randn(2, 3, 256, 256)
        out = encoder(x)
        assert out.shape == (2, 512)

    def test_invalid_variant_raises(self):
        """Invalid variant name should raise ValueError."""
        from src.models.spatial_encoder import EfficientNetEncoder

        with pytest.raises(ValueError, match="EfficientNet variant"):
            EfficientNetEncoder(variant="b99", pretrained=False)


# ---------------------------------------------------------------------------
# Temporal model tests
# ---------------------------------------------------------------------------


class TestMSTCN:
    """Tests for MS-TCN temporal model."""

    def test_output_list_length(self):
        """MS-TCN should return one set of logits per stage."""
        from src.models.temporal_model import MSTCN

        model = MSTCN(num_stages=4, num_layers=4, num_filters=32, feature_dim=64, num_classes=7)
        features = torch.randn(2, 64, 100)  # (B, F, T)
        outputs = model(features)
        assert len(outputs) == 4, f"Expected 4 stage outputs, got {len(outputs)}"

    def test_output_shape(self):
        """Each stage output should be (B, num_classes, T)."""
        from src.models.temporal_model import MSTCN

        B, C, T = 2, 7, 50
        model = MSTCN(num_stages=2, num_layers=4, num_filters=32, feature_dim=64, num_classes=C)
        features = torch.randn(B, 64, T)
        outputs = model(features)
        for i, logits in enumerate(outputs):
            assert logits.shape == (B, C, T), f"Stage {i}: expected {(B, C, T)}, got {logits.shape}"

    def test_causal_vs_noncausal(self):
        """Both causal and non-causal modes should produce valid outputs."""
        from src.models.temporal_model import MSTCN

        for causal in [True, False]:
            model = MSTCN(num_stages=2, num_layers=3, num_filters=16, feature_dim=32, num_classes=7, causal=causal)
            features = torch.randn(1, 32, 30)
            outputs = model(features)
            assert len(outputs) == 2


class TestTransSVNet:
    """Tests for transformer temporal model."""

    def test_output_shape(self):
        """TransSVNet should return (B, T, num_classes)."""
        from src.models.temporal_model import TransSVNet

        model = TransSVNet(
            feature_dim=64, d_model=32, nhead=4, num_layers=2, num_classes=7, causal=True
        )
        features = torch.randn(2, 50, 64)  # (B, T, F)
        out = model(features)
        assert out.shape == (2, 50, 7), f"Got {out.shape}"


class TestLSTMTemporalModel:
    """Tests for LSTM temporal model."""

    def test_forward_stateless(self):
        """LSTM forward without hidden state should work."""
        from src.models.temporal_model import LSTMTemporalModel

        model = LSTMTemporalModel(feature_dim=64, hidden_size=32, num_classes=7)
        features = torch.randn(2, 50, 64)
        logits, hidden = model(features)
        assert logits.shape == (2, 50, 7)
        assert hidden is not None

    def test_stateful_inference(self):
        """LSTM hidden state should be carried between calls."""
        from src.models.temporal_model import LSTMTemporalModel

        model = LSTMTemporalModel(feature_dim=64, hidden_size=32, num_classes=7)
        features1 = torch.randn(1, 10, 64)
        features2 = torch.randn(1, 10, 64)
        _, hidden1 = model(features1, hidden=None)
        logits2, _ = model(features2, hidden=hidden1)
        assert logits2.shape == (1, 10, 7)


class TestTemporalCRF:
    """Tests for CRF temporal smoother."""

    def test_viterbi_decode(self):
        """Viterbi decoding should return valid sequence of class indices."""
        from src.models.temporal_model import TemporalCRF

        crf = TemporalCRF(num_classes=7)
        log_probs = torch.randn(2, 50, 7)  # (B, T, C) log probs
        sequence = crf.viterbi_decode(log_probs)
        assert sequence.shape == (2, 50)
        assert (sequence >= 0).all() and (sequence < 7).all()


# ---------------------------------------------------------------------------
# Phase classifier integration tests
# ---------------------------------------------------------------------------


class TestSurgPhaseClassifier:
    """Integration tests for the end-to-end SurgPhaseClassifier."""

    @pytest.fixture
    def model(self):
        """Create a small SurgPhaseClassifier for testing."""
        from src.models.phase_classifier import SurgPhaseClassifier

        return SurgPhaseClassifier(
            backbone="resnet50",
            feature_dim=256,  # Small for testing
            temporal_model="mstcn",
            num_phases=7,
            causal=True,
            use_crf=True,
            fuse_instruments=True,
            pretrained_backbone=False,
            tcn_stages=2,
            tcn_layers=4,
            tcn_filters=16,
        )

    def test_forward_shape(self, model):
        """End-to-end forward pass should produce phase logits."""
        frames = torch.randn(1, 8, 3, 64, 64)  # (B, T, C, H, W)
        model.eval()
        with torch.no_grad():
            outputs = model(frames)
        assert "phase_logits" in outputs or "all_stage_logits" in outputs
        assert "instrument_logits" in outputs

    def test_training_mode(self, model):
        """Training forward pass should include all stage logits for deep supervision."""
        frames = torch.randn(2, 8, 3, 64, 64)
        model.train()
        outputs = model(frames)
        assert "all_stage_logits" in outputs
        assert len(outputs["all_stage_logits"]) == 2  # 2 stages

    def test_inference_mode(self, model):
        """Eval forward pass should include Viterbi-decoded phase labels."""
        frames = torch.randn(1, 8, 3, 64, 64)
        model.eval()
        with torch.no_grad():
            outputs = model(frames)
        assert "phase_labels" in outputs
        assert "phase_probs" in outputs


# ---------------------------------------------------------------------------
# Metrics tests
# ---------------------------------------------------------------------------


class TestSurgicalMetrics:
    """Tests for surgical evaluation metrics."""

    def test_frame_accuracy(self):
        """Frame accuracy should be correct."""
        from src.evaluation.surgical_metrics import frame_level_accuracy
        import numpy as np

        preds = np.array([0, 0, 1, 1, 2, 2])
        labels = np.array([0, 1, 1, 1, 2, 3])
        acc = frame_level_accuracy(preds, labels)
        assert abs(acc - (4 / 6)) < 1e-6

    def test_edit_distance_identical(self):
        """Edit distance between identical sequences should be 0."""
        from src.evaluation.surgical_metrics import edit_distance
        import numpy as np

        seq = np.array([0, 0, 1, 1, 2, 3, 3])
        assert edit_distance(seq, seq) == 0.0

    def test_transition_detection(self):
        """Transition detection should find correct boundary frames."""
        from src.evaluation.surgical_metrics import transition_detection_accuracy
        import numpy as np

        gt = np.array([0, 0, 0, 1, 1, 1, 2, 2])
        pred = np.array([0, 0, 1, 1, 1, 2, 2, 2])
        result = transition_detection_accuracy(gt, pred, tolerance_frames=2)
        assert result["f1"] > 0.5

    def test_mean_jaccard_perfect(self):
        """Perfect predictions should yield Jaccard of 1."""
        from src.evaluation.surgical_metrics import mean_jaccard
        import numpy as np

        labels = np.array([0, 0, 1, 1, 2, 2, 3])
        assert abs(mean_jaccard(labels, labels) - 1.0) < 1e-6

    def test_surgical_metrics_accumulate(self):
        """SurgicalMetrics should accumulate across multiple batches."""
        from src.evaluation.surgical_metrics import SurgicalMetrics

        tracker = SurgicalMetrics(num_phases=7)
        for _ in range(5):
            preds = torch.randint(0, 7, (2, 20))
            labels = torch.randint(0, 7, (2, 20))
            tracker.update(preds, labels)
        results = tracker.compute()
        assert "accuracy" in results
        assert "mean_jaccard" in results
        assert 0.0 <= results["accuracy"] <= 1.0
