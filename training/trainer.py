"""
trainer.py
──────────
Training loop for GLCLAP.

Paper (Section 3.2):
    - Learning rate: 5e-4
    - Batch size: 64
    - Epochs: 100
    - Early stopping to prevent overfitting

Responsibilities:
    - Manage train / validation epoch loops
    - Call the GLCLAP model forward pass
    - Compute losses via losses.contrastive.glclap_loss
    - Run optimizer + scheduler steps
    - Apply gradient clipping
    - Handle AMP (mixed precision)
    - Log metrics and save checkpoints
    - Implement early stopping
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader

from losses.contrastive import glclap_loss
from models.glclap import GLCLAP
from training.scheduler import build_scheduler
from utils.checkpointing import save_checkpoint, load_checkpoint

logger = logging.getLogger(__name__)


class Trainer:
    """
    Manages the full training lifecycle for GLCLAP.

    Args:
        model:             GLCLAP model instance.
        train_loader:      DataLoader for training split.
        val_loader:        DataLoader for validation split (optional).
        output_dir:        Directory for checkpoints and logs.
        lr:                Peak learning rate (paper: 5e-4).
        weight_decay:      AdamW weight decay.
        temperature:       InfoNCE temperature.
        num_epochs:        Max training epochs (paper: 100).
        grad_clip:         Max gradient norm (0 = disabled).
        mixed_precision:   Whether to use torch.cuda.amp.
        early_stopping_patience: Stop if val loss does not improve for N epochs.
        warmup_steps:      LR scheduler warm-up steps.
        scheduler_type:    LR schedule variant (see scheduler.py).
        local_only:        If True, use simplified local-only contrastive
                           learning (subtext [B,D] vs pooled audio [B,D]).
                           Global branches are closed in the forward pass.
        device:            Target device string (e.g. "cuda", "cpu").
    """

    def __init__(
        self,
        model: GLCLAP,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        output_dir: str = "outputs/glclap",
        lr: float = 5e-4,
        weight_decay: float = 1e-2,
        temperature: float = 0.07,
        num_epochs: int = 100,
        grad_clip: float = 1.0,
        mixed_precision: bool = True,
        early_stopping_patience: int = 5,
        warmup_steps: int = 2000,
        scheduler_type: str = "cosine_with_warmup",
        local_only: bool = False,
        device: str = "cuda",
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.output_dir = Path(output_dir)
        if rank == 0:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        self.temperature = temperature
        self.num_epochs = num_epochs
        self.grad_clip = grad_clip
        self.mixed_precision = mixed_precision
        self.early_stopping_patience = early_stopping_patience
        self.local_only = local_only
        self.device = device
        self.rank = rank
        self.world_size = world_size
        self.is_main = rank == 0

        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.98),
        )

        # LR Scheduler
        total_steps = num_epochs * len(train_loader)
        self.scheduler = build_scheduler(
            self.optimizer,
            scheduler_type=scheduler_type,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
        )

        # AMP scaler
        self.scaler = GradScaler(enabled=mixed_precision)

        self._best_val_loss: float = float("inf")
        self._patience_counter: int = 0

    # ──────────────────────────────────────────────────────────────────────
    # Training loop
    # ──────────────────────────────────────────────────────────────────────

    def _unwrap_model(self) -> GLCLAP:
        """Return the underlying GLCLAP model (handles DDP wrapper)."""
        if isinstance(self.model, DDP):
            return self.model.module
        return self.model

    def train(self, resume_from: Optional[str] = None) -> None:
        """
        Run the full training loop.

        Args:
            resume_from: Path to a checkpoint to resume from (optional).
        """
        start_epoch = 0
        if resume_from is not None:
            start_epoch = load_checkpoint(
                resume_from, self._unwrap_model(), self.optimizer, self.scheduler
            )
            if self.is_main:
                logger.info(f"Resumed from {resume_from}, starting at epoch {start_epoch}")

        for epoch in range(start_epoch, self.num_epochs):
            # Set epoch for DistributedSampler
            if hasattr(self.train_loader.sampler, "set_epoch"):
                self.train_loader.sampler.set_epoch(epoch)

            train_loss = self._train_epoch(epoch)
            if self.is_main:
                logger.info(f"Epoch {epoch+1}/{self.num_epochs}  train_loss={train_loss:.4f}")

            val_loss = None
            if self.val_loader is not None:
                val_loss = self._val_epoch(epoch)
                if self.is_main:
                    logger.info(f"Epoch {epoch+1}/{self.num_epochs}  val_loss={val_loss:.4f}")

            # Checkpointing (main process only)
            if self.is_main:
                ckpt_path = self.output_dir / f"checkpoint_epoch{epoch+1:03d}.pt"
                save_checkpoint(
                    path=ckpt_path,
                    model=self._unwrap_model(),
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    epoch=epoch + 1,
                    val_loss=val_loss,
                )

                # Early stopping (on val loss if available, else train loss)
                monitor = val_loss if val_loss is not None else train_loss
                if monitor < self._best_val_loss:
                    self._best_val_loss = monitor
                    self._patience_counter = 0
                    # Save best model separately
                    save_checkpoint(
                        path=self.output_dir / "best_model.pt",
                        model=self._unwrap_model(),
                        optimizer=self.optimizer,
                        scheduler=self.scheduler,
                        epoch=epoch + 1,
                        val_loss=monitor,
                    )
                else:
                    self._patience_counter += 1
                    if self._patience_counter >= self.early_stopping_patience:
                        logger.info(f"Early stopping at epoch {epoch+1}.")
                        break

    # ──────────────────────────────────────────────────────────────────────
    # Epoch-level helpers
    # ──────────────────────────────────────────────────────────────────────

    def _train_epoch(self, epoch: int) -> float:
        """
        Run one training epoch.

        Returns:
            Mean training loss over all batches.
        """
        self.model.train()
        total_loss = 0.0

        for step, batch in enumerate(self.train_loader):
            batch = self._move_to_device(batch)

            self.optimizer.zero_grad()

            with autocast(enabled=self.mixed_precision):
                output = self.model(
                    text_input_ids=batch["text_input_ids"],
                    text_attention_mask=batch["text_attention_mask"],
                    subtext_input_ids=batch["subtext_input_ids"],
                    subtext_attention_mask=batch["subtext_attention_mask"],
                    waveform=batch["waveforms"].squeeze(1),   # [B, T_samples]
                    waveform_attention_mask=batch.get("waveform_attention_mask", None),
                    local_only=self.local_only,
                )
                # Standard mode:
                #   output.text_global:  [B, D]
                #   output.text_local:   [B, D]
                #   output.audio_global: [B, D]
                #   output.audio_local:  [B, T', D]
                # Local-only mode:
                #   output.text_local:   [B, D]
                #   output.audio_local:  [B, D]

                loss_dict = glclap_loss(
                    text_global=output.text_global,
                    text_local=output.text_local,
                    audio_global=output.audio_global,
                    audio_local=output.audio_local,
                    temperature=self.temperature,
                    local_only=self.local_only,
                )
                loss = loss_dict["loss"]

            self.scaler.scale(loss).backward()

            if self.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.scaler.step(self.optimizer)
            # PyTorch 2.0 AMP path may not increment optimizer._step_count,
            # which causes LambdaLR to emit a harmless first-step warning.
            # Manually ensure the counter is updated before scheduler.step().
            if self.optimizer._step_count == 0:
                self.optimizer._step_count = 1
            self.scheduler.step()
            self.scaler.update()

            total_loss += loss.item()

            if step % 50 == 0 and self.is_main:
                if self.local_only:
                    logger.info(
                        f"  [E{epoch+1} S{step}] loss={loss.item():.4f} "
                        f"(local-only contrastive)"
                    )
                else:
                    logger.info(
                        f"  [E{epoch+1} S{step}] loss={loss.item():.4f} "
                        f"Lg={loss_dict['loss_global'].item():.4f} "
                        f"Ll={loss_dict['loss_local'].item():.4f}"
                    )

        return total_loss / max(1, len(self.train_loader))

    @torch.no_grad()
    def _val_epoch(self, epoch: int) -> float:
        """
        Run one validation epoch (no gradient, no AMP scaler).

        Returns:
            Mean validation loss.
        """
        self.model.eval()
        total_loss = 0.0

        for batch in self.val_loader:
            batch = self._move_to_device(batch)

            with autocast(enabled=self.mixed_precision):
                output = self.model(
                    text_input_ids=batch["text_input_ids"],
                    text_attention_mask=batch["text_attention_mask"],
                    subtext_input_ids=batch["subtext_input_ids"],
                    subtext_attention_mask=batch["subtext_attention_mask"],
                    waveform=batch["waveforms"].squeeze(1),
                    waveform_attention_mask=batch.get("waveform_attention_mask", None),
                    local_only=self.local_only,
                )
                loss_dict = glclap_loss(
                    text_global=output.text_global,
                    text_local=output.text_local,
                    audio_global=output.audio_global,
                    audio_local=output.audio_local,
                    temperature=self.temperature,
                    local_only=self.local_only,
                )

            total_loss += loss_dict["loss"].item()

        return total_loss / max(1, len(self.val_loader))

    def _move_to_device(self, batch: dict) -> dict:
        """Move all tensor values in batch dict to self.device."""
        return {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
