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

MODIFIED (2026-05-13):
    _val_epoch 不再仅在 val_loader 上计算 loss，而是仿照 eval_and_plot.py
    的测试思路：保存临时 checkpoint → 调用 evaluate.py 做 bias-word retrieval
    评估 → 解析 stdout 中的 top1_recall / precision / recall / f1 指标并打印。
    不绘图，只打印指标数据。返回 f1 作为 early stopping 的监控指标。
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional, Dict, Any

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
        val_eval_config:   Dict with evaluate.py args for per-epoch eval
                           (task, dataset, model_config, bias_list, etc.).
                           If provided, _val_epoch runs evaluate.py instead
                           of plain val loss.
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
        val_eval_config: Optional[Dict[str, Any]] = None,
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
        seed: Optional[int] = None,
        tokenizer: Optional[Any] = None,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.val_eval_config = val_eval_config
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
        self._seed = seed
        self.tokenizer = tokenizer

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

        self._best_val_metric: float = float("inf")  # 用 val_loss 做 early stopping，越低越好
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
                resume_from, self._unwrap_model(), self.optimizer, self.scheduler,
                strict=False,
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

            # ── val：使用 val_loader 计算 loss ─────────────────────────────
            val_metric = None
            if self.val_loader is not None:
                val_loss = self._val_epoch_loss(epoch)
                val_metric = val_loss
                if self.is_main:
                    logger.info(f"Epoch {epoch+1}/{self.num_epochs}  val_loss={val_loss:.4f}")
            # ──────────────────────────────────────────────────────────────

            # Checkpointing (main process only)
            should_stop = False
            if self.is_main:
                ckpt_path = self.output_dir / f"checkpoint_epoch{epoch+1:03d}.pt"
                save_checkpoint(
                    path=ckpt_path,
                    model=self._unwrap_model(),
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    epoch=epoch + 1,
                    val_loss=val_metric,
                )

                # Early stopping：val_loss 越低越好
                monitor = val_metric if val_metric is not None else train_loss
                improved = monitor < self._best_val_metric

                if improved:
                    self._best_val_metric = monitor
                    self._patience_counter = 0
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
                        should_stop = True

            # DDP：广播 early-stopping 信号到所有 rank，防止主进程 break 后
            # 其他 rank 继续执行导致 all_reduce 死锁。
            if self.world_size > 1:
                stop_tensor = torch.tensor(1 if should_stop else 0, device=self.device)
                dist.all_reduce(stop_tensor, op=dist.ReduceOp.MAX)
                should_stop = stop_tensor.item() > 0

            if should_stop:
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
        # Epoch-based seed for reproducible shuffling
        if self._seed is not None:
            if hasattr(self.train_loader.sampler, "set_epoch"):
                self.train_loader.sampler.set_epoch(epoch)
            else:
                torch.manual_seed(self._seed + epoch)

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
                    waveform=batch["waveforms"].squeeze(1),
                    waveform_attention_mask=batch.get("waveform_attention_mask", None),
                    local_only=self.local_only,
                    sample_ids=batch.get("sample_ids", None),  # MODIFIED (2026-05-14)
                )

                loss_dict = glclap_loss(
                    text_global=output.text_global,
                    text_local=output.text_local,
                    audio_global=output.audio_global,
                    audio_local=output.audio_local,
                    temperature=self.temperature,
                    local_only=self.local_only,
                    sample_ids=batch.get("sample_ids", None),
                )
                loss = loss_dict["loss"]

            self.scaler.scale(loss).backward()

            if self.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.scaler.step(self.optimizer)
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

    # ── 修改处：原 _val_epoch 改名为 _val_epoch_loss（保留回退）───────
    @torch.no_grad()
    def _val_epoch_loss(self, epoch: int) -> float:
        """
        Run one validation epoch (no gradient, no AMP scaler).
        在 val_loader 上计算平均 loss；DDP 下聚合所有 rank 的结果。
        """
        self.model.eval()
        if self.device.startswith("cuda"):
            torch.cuda.empty_cache()
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
                    sample_ids=batch.get("sample_ids", None),  # MODIFIED (2026-05-14)
                )
                loss_dict = glclap_loss(
                    text_global=output.text_global,
                    text_local=output.text_local,
                    audio_global=output.audio_global,
                    audio_local=output.audio_local,
                    temperature=self.temperature,
                    local_only=self.local_only,
                    sample_ids=batch.get("sample_ids", None),
                )

            total_loss += loss_dict["loss"].item()

        local_avg = total_loss / max(1, len(self.val_loader))

        # DDP：聚合所有 rank 的 val_loss 求全局平均
        if self.world_size > 1:
            loss_tensor = torch.tensor(local_avg, device=self.device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
            return loss_tensor.item()

        return local_avg

    # ── 修改处：_val_epoch_eval 直接在主进程内评估，不再调用子进程 ─────
    def _val_epoch_eval(self, epoch: int) -> float:
        """
        每轮 validation 时直接使用当前模型进行 bias-word retrieval 评估。
        不再保存临时 checkpoint 或调用 evaluate.py 子进程，避免显存竞争。
        返回 f1 作为 early stopping 监控指标。

        仅在主进程（rank 0）执行；其他进程直接返回 0.0。
        调用方（train()）已通过 dist.barrier() 保证所有 rank 同步进出。

        MODIFIED (2026-05-15): 改为 batch 推理加速。
            - 统一使用全局 bias list，不再逐条切换 per-sample bias（避免反复编码）
            - 音频从逐条 retrieve() 改为 batch retrieve_batch()，减少 forward 次数
            - eval_batch_size 默认 16，可从 val_eval_config 覆盖
        """
        if not self.is_main:
            return 0.0

        self.model.eval()

        # Release cached memory before evaluation to reduce fragmentation
        if self.device.startswith("cuda"):
            torch.cuda.empty_cache()

        cfg = self.val_eval_config
        manifest_path = f"data/{cfg['task']}/{cfg['dataset']}.jsonl"
        eval_batch_size = cfg.get("eval_batch_size", 16)

        # 获取 tokenizer（优先使用传入的，否则从模型推断）
        if self.tokenizer is not None:
            tokenizer = self.tokenizer
        else:
            from transformers import AutoTokenizer
            text_model_name = self._unwrap_model().text_encoder.bert.config._name_or_path
            tokenizer = AutoTokenizer.from_pretrained(text_model_name)

        # 加载全局 bias list
        bias_words = []
        if cfg.get("bias_list") and os.path.exists(cfg["bias_list"]):
            with open(cfg["bias_list"], encoding="utf-8") as f:
                bias_words = [line.strip() for line in f if line.strip()]

        from inference.retriever import BiasWordRetriever
        from eval.metrics import evaluate_retrieval
        from dataset.audio_utils import load_audio

        retriever = BiasWordRetriever(
            model=self._unwrap_model(),
            tokenizer=tokenizer,
            threshold=cfg.get("threshold", 0.5),
            top_k=cfg.get("top_k", 10),
            device=self.device,
        )
        if bias_words:
            retriever.set_bias_list(bias_words)

        # ── 收集所有 manifest 样本 ──
        samples: list[dict] = []
        with open(manifest_path, encoding="utf-8") as f:
            for line in f:
                samples.append(json.loads(line.strip()))

        all_predictions: list[list[str]] = []
        all_ground_truths: list[list[str]] = []
        top1_ground_truths: list[str] = []
        skipped_empty_gt = 0

        # ── batch 推理 ──
        for batch_start in range(0, len(samples), eval_batch_size):
            batch = samples[batch_start : batch_start + eval_batch_size]

            # 1) 加载 batch 音频到 CPU
            batch_wfs: list[torch.Tensor] = []
            for item in batch:
                wf, _ = load_audio(item["audio_path"], target_sr=16_000)
                batch_wfs.append(wf.squeeze(0))  # [T]

            # 2) pad 到相同长度
            max_len = max(wf.shape[0] for wf in batch_wfs)
            padded: list[torch.Tensor] = []
            masks: list[torch.Tensor] = []
            for wf in batch_wfs:
                pad_len = max_len - wf.shape[0]
                if pad_len > 0:
                    padded.append(torch.cat([wf, torch.zeros(pad_len, dtype=wf.dtype)]))
                    masks.append(torch.cat([torch.ones(wf.shape[0]), torch.zeros(pad_len)]))
                else:
                    padded.append(wf)
                    masks.append(torch.ones(wf.shape[0]))

            waveforms_batch = torch.stack(padded).to(self.device)      # [B, T]
            attention_mask = torch.stack(masks).to(self.device)        # [B, T]

            # 3) batch retrieve
            predictions = retriever.retrieve_batch(waveforms_batch, attention_mask)

            # 4) 收集结果
            for item, predicted in zip(batch, predictions):
                gt_entities = item.get("bias_entities", [])
                if not gt_entities:
                    skipped_empty_gt += 1
                    continue
                all_predictions.append(predicted)
                all_ground_truths.append(gt_entities)
                top1_ground_truths.append(gt_entities[0])

            # 显存清理
            if self.device.startswith("cuda"):
                torch.cuda.empty_cache()

        results = evaluate_retrieval(
            all_predictions=all_predictions,
            all_ground_truths=all_ground_truths,
            top1_ground_truths=top1_ground_truths,
        )

        if results:
            logger.info(f"  [Val Epoch {epoch+1}] ── Evaluation Results ──")
            if skipped_empty_gt > 0:
                logger.info(
                    f"    (Skipped {skipped_empty_gt} samples with empty bias_entities)"
                )
            for k, v in results.items():
                logger.info(f"    {k:<16}: {v*100:.2f}%")
            logger.info(f"  [Val Epoch {epoch+1}] ────────────────────────")

        return results.get("f1", 0.0)
    # ──────────────────────────────────────────────────────────────────

    def _move_to_device(self, batch: dict) -> dict:
        """Move all tensor values in batch dict to self.device."""
        return {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
