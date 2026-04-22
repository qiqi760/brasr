"""
scheduler.py
────────────
Learning-rate scheduler factory for GLCLAP training.

Paper (Section 3.2): Learning rate = 5e-4.  No scheduler details are given.
Assumption: cosine annealing with linear warmup, following CLAP/CLIP convention.
"""

from __future__ import annotations

import math
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def build_scheduler(
    optimizer: Optimizer,
    scheduler_type: str = "cosine_with_warmup",
    warmup_steps: int = 2000,
    total_steps: int = 100_000,
) -> LambdaLR:
    """
    Build a learning-rate scheduler.

    Args:
        optimizer:      PyTorch optimizer.
        scheduler_type: One of {"cosine_with_warmup", "linear_with_warmup", "constant"}.
        warmup_steps:   Number of steps for linear warm-up phase.
        total_steps:    Total training steps (epochs × steps_per_epoch).

    Returns:
        LambdaLR scheduler.
    """
    if scheduler_type == "cosine_with_warmup":
        return _cosine_with_warmup(optimizer, warmup_steps, total_steps)
    elif scheduler_type == "linear_with_warmup":
        return _linear_with_warmup(optimizer, warmup_steps, total_steps)
    elif scheduler_type == "constant":
        return LambdaLR(optimizer, lambda _: 1.0)
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type!r}")


def _cosine_with_warmup(
    optimizer: Optimizer,
    warmup_steps: int,
    total_steps: int,
) -> LambdaLR:
    """Cosine decay after linear warmup."""
    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / max(1, warmup_steps)
        progress = float(current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


def _linear_with_warmup(
    optimizer: Optimizer,
    warmup_steps: int,
    total_steps: int,
) -> LambdaLR:
    """Linear decay after linear warmup (reaches 0 at total_steps)."""
    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / max(1, warmup_steps)
        return max(0.0, float(total_steps - current_step) / max(1, total_steps - warmup_steps))

    return LambdaLR(optimizer, lr_lambda)
