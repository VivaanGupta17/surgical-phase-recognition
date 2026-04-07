"""
Multi-stage training loop for surgical phase recognition.

Training Procedure
------------------
Stage 1 — Backbone pre-training (epochs 1–N₁):
  Train spatial encoder + instrument head only using cross-entropy on per-frame
  phase labels and binary cross-entropy on instrument presence. Temporal model
  frozen. Input: individual frames (no temporal context).

Stage 2 — Temporal model training (epochs N₁–N₂):
  Freeze backbone; train the full temporal model (MS-TCN++, Transformer, LSTM)
  on feature sequences. Uses TMSE loss (temporal mean squared error) to
  penalise over-segmentation, in addition to cross-entropy.

Stage 3 — Joint fine-tuning (epochs N₂–N₃):
  Unfreeze all parameters with differential learning rates. Final polishing.

Supports:
  - Mixed precision (AMP) with GradScaler
  - Gradient accumulation for effective large-batch training
  - Causal (online) vs non-causal (offline) training modes
  - Multi-GPU via torch.nn.parallel.DistributedDataParallel
  - Checkpoint saving with best-model tracking
  - Weights & Biases logging
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------


class TMSELoss(nn.Module):
    """Temporal Mean Squared Error loss for MS-TCN over-segmentation penalty.

    Penalises abrupt frame-to-frame changes in class probability predictions.
    Introduced by Farha & Gall (2019) in the original MS-TCN paper.

    .. math::
        \\mathcal{L}_{TMSE} = \\frac{1}{TC}\\sum_{t,c}
            \\Delta(\\tilde{y}_{t,c}, \\tilde{y}_{t-1,c})^2

    where :math:`\\tilde{y}_{t,c} = \\log \\text{softmax}(x_{t,c})` and
    :math:`\\Delta` is clipped to [-\\tau, \\tau]` to avoid exploding gradients.

    Args:
        tau: Clipping threshold. Default 4.0 (from original paper).
    """

    def __init__(self, tau: float = 4.0) -> None:
        super().__init__()
        self.tau = tau

    def forward(self, logits: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute TMSE loss.

        Args:
            logits: (B, num_classes, T) logit tensor.
            mask: (B, num_classes, T) binary mask — 0 for padded positions.

        Returns:
            Scalar loss tensor.
        """
        log_probs = F.log_softmax(logits, dim=1)  # (B, C, T)
        diff = log_probs[:, :, 1:] - log_probs[:, :, :-1]  # (B, C, T-1)
        diff = diff.clamp(-self.tau, self.tau)
        loss = diff.pow(2)
        if mask is not None:
            loss = loss * mask[:, :, 1:]
        return loss.mean()


class MSTCNLoss(nn.Module):
    """Combined cross-entropy + TMSE loss for multi-stage MS-TCN training.

    Applies cross-entropy and TMSE to each stage output with deep supervision,
    using the same loss coefficient across all stages.

    Args:
        num_classes: Number of phase classes.
        tmse_weight: Weight for the TMSE over-segmentation penalty.
        ignore_index: Label index to ignore in cross-entropy (e.g. -1 for padding).
    """

    def __init__(
        self,
        num_classes: int = 7,
        tmse_weight: float = 0.15,
        ignore_index: int = -1,
    ) -> None:
        super().__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.tmse = TMSELoss()
        self.tmse_weight = tmse_weight

    def forward(
        self,
        all_stage_logits: List[torch.Tensor],
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute multi-stage training loss.

        Args:
            all_stage_logits: List of (B, num_classes, T) logit tensors,
                              one per MS-TCN stage.
            targets: (B, T) ground truth phase labels (long).
            mask: (B, num_classes, T) optional TMSE mask.

        Returns:
            Tuple of (total_loss, loss_dict) where loss_dict contains
            per-stage and per-component loss values for logging.
        """
        total_loss = torch.tensor(0.0, device=all_stage_logits[0].device)
        loss_dict: Dict[str, float] = {}

        for i, logits in enumerate(all_stage_logits):
            # logits: (B, C, T) → CE expects (B*T, C) and (B*T,)
            B, C, T = logits.shape
            logits_flat = logits.permute(0, 2, 1).reshape(B * T, C)
            targets_flat = targets.reshape(B * T)

            ce_loss = self.ce(logits_flat, targets_flat)
            tmse_loss = self.tmse(logits, mask)

            stage_loss = ce_loss + self.tmse_weight * tmse_loss
            total_loss = total_loss + stage_loss

            loss_dict[f"stage{i+1}_ce"] = ce_loss.item()
            loss_dict[f"stage{i+1}_tmse"] = tmse_loss.item()

        total_loss = total_loss / len(all_stage_logits)
        loss_dict["total"] = total_loss.item()
        return total_loss, loss_dict


# ---------------------------------------------------------------------------
# Trainer class
# ---------------------------------------------------------------------------


class SurgPhaseTrainer:
    """Multi-stage trainer for surgical phase recognition models.

    Handles the full training lifecycle: multi-stage curriculum, optimiser
    scheduling, AMP mixed precision, gradient accumulation, checkpoint
    management, and optional W&B logging.

    Args:
        model: ``SurgPhaseClassifier`` instance.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        config: Training configuration dict (from YAML config file).
        output_dir: Directory for checkpoints and logs.
        device: Target training device.
        use_wandb: Enable Weights & Biases logging.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict,
        output_dir: Path,
        device: Optional[torch.device] = None,
        use_wandb: bool = False,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_wandb = use_wandb

        self.model = self.model.to(self.device)

        # AMP scaler
        self.scaler = GradScaler(enabled=config.get("mixed_precision", True))

        # Losses
        self.phase_loss = MSTCNLoss(
            num_classes=config.get("num_phases", 7),
            tmse_weight=config.get("tmse_weight", 0.15),
        )
        self.instrument_loss = nn.BCEWithLogitsLoss()

        # Training state
        self.current_epoch = 0
        self.best_val_accuracy = 0.0
        self.global_step = 0

        if use_wandb:
            self._init_wandb()

    def _init_wandb(self) -> None:
        """Initialise Weights & Biases run."""
        try:
            import wandb
            wandb.init(
                project=self.config.get("wandb_project", "surgical-phase-recognition"),
                name=self.config.get("run_name", "surgphase_run"),
                config=self.config,
            )
            self.wandb = wandb
        except ImportError:
            logger.warning("wandb not installed. Disabling W&B logging.")
            self.use_wandb = False

    def _build_optimizer(self, stage: str) -> torch.optim.Optimizer:
        """Build optimizer with per-group learning rates for current stage.

        Args:
            stage: Training stage — 'backbone', 'temporal', or 'finetune'.

        Returns:
            Configured AdamW optimizer.
        """
        base_lr = self.config.get("learning_rate", 1e-4)
        weight_decay = self.config.get("weight_decay", 1e-4)

        if hasattr(self.model, "get_parameter_groups"):
            param_groups = self.model.get_parameter_groups(base_lr=base_lr)
        else:
            param_groups = [{"params": self.model.parameters(), "lr": base_lr}]

        return AdamW(param_groups, lr=base_lr, weight_decay=weight_decay)

    def _build_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
        num_epochs: int,
        warmup_epochs: int = 5,
    ) -> torch.optim.lr_scheduler._LRScheduler:
        """Build cosine LR schedule with linear warmup.

        Args:
            optimizer: Optimizer to schedule.
            num_epochs: Total epochs for this stage.
            warmup_epochs: Linear warmup duration.

        Returns:
            SequentialLR combining warmup + cosine decay.
        """
        warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs)
        cosine = CosineAnnealingLR(optimizer, T_max=max(1, num_epochs - warmup_epochs), eta_min=1e-7)
        return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])

    def train(self) -> None:
        """Run the full multi-stage training procedure."""
        cfg = self.config
        stages = cfg.get(
            "training_stages",
            [
                {"name": "backbone", "epochs": 10},
                {"name": "temporal", "epochs": 20},
                {"name": "finetune", "epochs": 30},
            ],
        )

        for stage_cfg in stages:
            stage_name = stage_cfg["name"]
            n_epochs = stage_cfg["epochs"]

            logger.info("=" * 60)
            logger.info("Starting training stage: %s (%d epochs)", stage_name, n_epochs)
            logger.info("=" * 60)

            # Configure model training stage
            if hasattr(self.model, "set_training_stage"):
                self.model.set_training_stage(stage_name)

            optimizer = self._build_optimizer(stage_name)
            scheduler = self._build_scheduler(
                optimizer,
                n_epochs,
                warmup_epochs=cfg.get("warmup_epochs", 3),
            )

            for epoch in range(n_epochs):
                self.current_epoch += 1
                train_metrics = self._train_epoch(optimizer, stage_name)
                val_metrics = self._val_epoch()
                scheduler.step()

                self._log_metrics(
                    {**train_metrics, **{f"val_{k}": v for k, v in val_metrics.items()}},
                    epoch=self.current_epoch,
                )

                # Save checkpoint
                is_best = val_metrics.get("accuracy", 0.0) > self.best_val_accuracy
                if is_best:
                    self.best_val_accuracy = val_metrics.get("accuracy", 0.0)
                self._save_checkpoint(
                    optimizer,
                    scheduler,
                    val_metrics,
                    is_best=is_best,
                )

        logger.info("Training complete. Best validation accuracy: %.4f", self.best_val_accuracy)

    def _train_epoch(
        self,
        optimizer: torch.optim.Optimizer,
        stage: str,
    ) -> Dict[str, float]:
        """Run one training epoch.

        Args:
            optimizer: Active optimizer.
            stage: Current training stage (controls which losses are active).

        Returns:
            Dict of scalar metrics for this epoch.
        """
        self.model.train()
        cfg = self.config
        accum_steps = cfg.get("gradient_accumulation_steps", 1)
        inst_loss_weight = cfg.get("instrument_loss_weight", 0.3)

        total_loss = 0.0
        total_phase_correct = 0
        total_frames = 0
        num_batches = 0

        optimizer.zero_grad()
        t0 = time.time()

        for batch_idx, batch in enumerate(self.train_loader):
            frames = batch["frames"].to(self.device)          # (B, T, 3, H, W) or (B, T, F)
            phase_labels = batch["phase_labels"].to(self.device)   # (B, T)
            inst_labels = batch["instrument_labels"].to(self.device)  # (B, T, 7)

            # If frames are features (2D), add a dummy spatial dimension
            # Models handle both modes internally

            with autocast(enabled=cfg.get("mixed_precision", True)):
                outputs = self.model(frames)

                # --- Phase loss ---
                if "all_stage_logits" in outputs:
                    phase_loss, loss_info = self.phase_loss(
                        outputs["all_stage_logits"], phase_labels
                    )
                else:
                    # Transformer / LSTM: (B, T, C) → (B, C, T)
                    logits = outputs["phase_logits"]
                    if logits.dim() == 3 and logits.size(2) != self.config.get("num_phases", 7):
                        logits = logits.permute(0, 2, 1)
                    phase_loss, loss_info = self.phase_loss([logits], phase_labels)

                # --- Instrument loss (auxiliary) ---
                inst_logits = outputs["instrument_logits"]  # (B, T, 7)
                inst_loss = self.instrument_loss(inst_logits, inst_labels)

                # Stage-dependent weighting
                if stage == "backbone":
                    loss = phase_loss + inst_loss_weight * inst_loss
                elif stage == "temporal":
                    loss = phase_loss
                else:
                    loss = phase_loss + inst_loss_weight * inst_loss

                loss = loss / accum_steps

            self.scaler.scale(loss).backward()

            if (batch_idx + 1) % accum_steps == 0:
                self.scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                self.scaler.step(optimizer)
                self.scaler.update()
                optimizer.zero_grad()
                self.global_step += 1

            # Metrics accumulation
            total_loss += loss.item() * accum_steps
            if "all_stage_logits" in outputs:
                final_logits = outputs["all_stage_logits"][-1]  # (B, C, T)
                preds = final_logits.argmax(dim=1)  # (B, T)
            else:
                preds = outputs["phase_logits"].argmax(dim=-1)

            valid_mask = phase_labels >= 0
            total_phase_correct += (preds == phase_labels)[valid_mask].sum().item()
            total_frames += valid_mask.sum().item()
            num_batches += 1

            if batch_idx % 50 == 0:
                elapsed = time.time() - t0
                fps = (batch_idx + 1) * frames.size(0) * frames.size(1) / elapsed
                logger.info(
                    "Epoch %d [%d/%d] loss=%.4f acc=%.4f %.0f frames/s",
                    self.current_epoch,
                    batch_idx,
                    len(self.train_loader),
                    total_loss / num_batches,
                    total_phase_correct / max(1, total_frames),
                    fps,
                )

        return {
            "loss": total_loss / max(1, num_batches),
            "accuracy": total_phase_correct / max(1, total_frames),
        }

    @torch.no_grad()
    def _val_epoch(self) -> Dict[str, float]:
        """Run one validation epoch.

        Returns:
            Dict with 'loss', 'accuracy', and 'jaccard'.
        """
        self.model.eval()
        total_loss = 0.0
        all_preds: List[int] = []
        all_labels: List[int] = []
        num_batches = 0

        for batch in self.val_loader:
            frames = batch["frames"].to(self.device)
            phase_labels = batch["phase_labels"].to(self.device)
            inst_labels = batch["instrument_labels"].to(self.device)

            with autocast(enabled=self.config.get("mixed_precision", True)):
                outputs = self.model(frames)

                if "all_stage_logits" in outputs:
                    final_logits = outputs["all_stage_logits"][-1]
                    loss, _ = self.phase_loss(outputs["all_stage_logits"], phase_labels)
                else:
                    logits = outputs["phase_logits"]
                    if logits.dim() == 3 and logits.size(2) != self.config.get("num_phases", 7):
                        logits = logits.permute(0, 2, 1)
                    final_logits = logits
                    loss, _ = self.phase_loss([logits], phase_labels)

            total_loss += loss.item()

            if final_logits.dim() == 3 and final_logits.size(1) == self.config.get("num_phases", 7):
                preds = final_logits.argmax(dim=1).flatten()
            else:
                preds = final_logits.argmax(dim=-1).flatten()

            valid_mask = phase_labels.flatten() >= 0
            all_preds.extend(preds[valid_mask].cpu().tolist())
            all_labels.extend(phase_labels.flatten()[valid_mask].cpu().tolist())
            num_batches += 1

        # Compute metrics
        import numpy as np

        preds_arr = np.array(all_preds)
        labels_arr = np.array(all_labels)
        accuracy = (preds_arr == labels_arr).mean()

        # Per-phase Jaccard (IoU)
        num_phases = self.config.get("num_phases", 7)
        jaccards = []
        for p in range(num_phases):
            tp = ((preds_arr == p) & (labels_arr == p)).sum()
            fp = ((preds_arr == p) & (labels_arr != p)).sum()
            fn = ((preds_arr != p) & (labels_arr == p)).sum()
            denom = tp + fp + fn
            jaccards.append(tp / denom if denom > 0 else 0.0)
        mean_jaccard = float(np.mean(jaccards))

        metrics = {
            "loss": total_loss / max(1, num_batches),
            "accuracy": float(accuracy),
            "jaccard": mean_jaccard,
        }
        logger.info(
            "Val epoch %d: loss=%.4f acc=%.4f jac=%.4f",
            self.current_epoch,
            metrics["loss"],
            metrics["accuracy"],
            metrics["jaccard"],
        )
        return metrics

    def _save_checkpoint(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        val_metrics: Dict[str, float],
        is_best: bool = False,
    ) -> None:
        """Save model checkpoint to disk.

        Args:
            optimizer: Current optimizer state.
            scheduler: Current scheduler state.
            val_metrics: Validation metrics for this epoch.
            is_best: Whether this is the best model so far.
        """
        ckpt_dir = self.output_dir / "checkpoints"
        ckpt_dir.mkdir(exist_ok=True)

        state = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "val_metrics": val_metrics,
            "config": self.config,
        }

        ckpt_path = ckpt_dir / f"epoch_{self.current_epoch:03d}.pth"
        torch.save(state, ckpt_path)

        if is_best:
            best_path = ckpt_dir / "best_model.pth"
            torch.save(state, best_path)
            logger.info(
                "New best model saved: %.4f → %s",
                val_metrics.get("accuracy", 0.0),
                best_path,
            )

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Resume training from a checkpoint.

        Args:
            checkpoint_path: Path to a .pth checkpoint file.
        """
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.current_epoch = ckpt.get("epoch", 0)
        self.global_step = ckpt.get("global_step", 0)
        self.best_val_accuracy = ckpt.get("val_metrics", {}).get("accuracy", 0.0)
        logger.info(
            "Resumed from checkpoint %s (epoch %d, best_acc=%.4f)",
            checkpoint_path,
            self.current_epoch,
            self.best_val_accuracy,
        )

    def _log_metrics(self, metrics: Dict[str, float], epoch: int) -> None:
        """Log metrics to console and optionally W&B.

        Args:
            metrics: Dict of metric name → value.
            epoch: Current training epoch.
        """
        metric_str = "  ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        logger.info("Epoch %d — %s", epoch, metric_str)

        if self.use_wandb and hasattr(self, "wandb"):
            self.wandb.log(metrics, step=epoch)
