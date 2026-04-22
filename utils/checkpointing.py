"""
checkpointing.py
────────────────
Utilities for saving and loading model checkpoints.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

logger = logging.getLogger(__name__)


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: Optional[Optimizer] = None,
    scheduler=None,
    epoch: int = 0,
    val_loss: Optional[float] = None,
    extra: Optional[dict] = None,
) -> None:
    """
    Save model (+ optional optimizer/scheduler) state to a .pt file.

    Args:
        path:       Destination file path.
        model:      PyTorch module to save.
        optimizer:  Optional optimizer state.
        scheduler:  Optional LR scheduler state.
        epoch:      Current epoch number (for resuming).
        val_loss:   Validation loss at this checkpoint.
        extra:      Any additional data to embed in the checkpoint dict.
    """
    state = {
        "epoch": epoch,
        "val_loss": val_loss,
        "model_state_dict": model.state_dict(),
    }
    if optimizer is not None:
        state["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        state["scheduler_state_dict"] = scheduler.state_dict()
    if extra:
        state.update(extra)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, str(path))
    logger.info(f"Checkpoint saved → {path}  (epoch={epoch}, val_loss={val_loss})")


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: Optional[Optimizer] = None,
    scheduler=None,
    strict: bool = True,
) -> int:
    """
    Load model (+ optional optimizer/scheduler) state from a checkpoint.

    Args:
        path:       Source checkpoint file path.
        model:      PyTorch module whose weights will be overwritten.
        optimizer:  Optional optimizer to restore.
        scheduler:  Optional LR scheduler to restore.
        strict:     If True, require exact key match in state_dict.

    Returns:
        epoch: The epoch number stored in the checkpoint (use as start_epoch).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    state = torch.load(str(path), map_location="cpu")

    model.load_state_dict(state["model_state_dict"], strict=strict)
    logger.info(f"Loaded model weights from {path}")

    if optimizer is not None and "optimizer_state_dict" in state:
        optimizer.load_state_dict(state["optimizer_state_dict"])
        logger.info("Restored optimizer state.")

    if scheduler is not None and "scheduler_state_dict" in state:
        scheduler.load_state_dict(state["scheduler_state_dict"])
        logger.info("Restored scheduler state.")

    epoch = state.get("epoch", 0)
    val_loss = state.get("val_loss", None)
    logger.info(f"Resuming from epoch {epoch}  (val_loss={val_loss})")
    return epoch
