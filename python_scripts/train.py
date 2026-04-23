"""
train.py
────────
Main entry point for GLCLAP training.

Usage:
    python python_scripts/train.py \
        --task contrastive-learning \
        --dataset libri-960 \
        --model_config configs/model_config.yaml \
        --train_config configs/train_config.yaml \
        [--audio_root data/contrastive-learning/audio/] \
        [--resume outputs/glclap/checkpoint_epoch010.pt]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import yaml

# Ensure project root is on the Python path when running as a script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.dataset import build_dataloader
from models.glclap import GLCLAP
from training.trainer import Trainer
from transformers import AutoTokenizer
from utils.logging import setup_logging


def _resolve_local_model(model_name: str) -> str:
    """优先使用本地 pretrained_models/ 目录下的模型。

    transformers 的 from_pretrained 支持本地路径和 HF Hub ID。
    如果项目根目录的 pretrained_models/ 下存在同名模型文件夹
    （HF ID 中的 ``/`` 替换为 ``--``），则返回本地绝对路径；
    否则返回原始 model_name（会触发在线下载或 cache 查找）。
    """
    project_root = Path(__file__).parent.parent.resolve()
    local_dir = project_root / "pretrained_models" / model_name.replace("/", "--")
    if local_dir.exists():
        return str(local_dir)
    return model_name


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train GLCLAP")
    p.add_argument("--task", required=True, help="Task name, e.g. contrastive-learning")
    p.add_argument("--dataset", required=True, help="Dataset name, e.g. libri-960")
    p.add_argument("--model_config", default="configs/model_config.yaml")
    p.add_argument("--train_config", default="configs/train_config.yaml")
    p.add_argument("--audio_root", default=None,
                   help="Root directory for audio files. Default: data/{task}/audio/")
    p.add_argument("--resume", default=None, help="Checkpoint path to resume from")
    p.add_argument("--local_only", action="store_true",
                   help="Simplified local-only contrastive learning: close "
                        "global branches and train subtext [B,D] vs pooled "
                        "audio [B,D] with standard InfoNCE")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--local_rank", type=int, default=-1,
                   help="Local rank for DistributedDataParallel (auto-set by torchrun)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.model_config) as f:
        model_cfg = yaml.safe_load(f)
    with open(args.train_config) as f:
        train_cfg = yaml.safe_load(f)

    # ── Distributed setup ────────────────────────────────────────────────
    # torchrun / launch.py set LOCAL_RANK via env var (PyTorch 2.x+).
    # Fallback to the legacy --local_rank CLI arg for backward compat.
    local_rank_env = os.environ.get("LOCAL_RANK", "")
    if local_rank_env != "":
        local_rank = int(local_rank_env)
        distributed = True
    else:
        local_rank = args.local_rank
        distributed = args.local_rank != -1

    if distributed:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        device = args.device
        rank = 0
        world_size = 1

    output_dir = train_cfg["training"]["output_dir"]
    if rank == 0:
        setup_logging(log_dir=output_dir)

    # ── Build data paths from task + dataset ─────────────────────────────
    manifest_path = f"data/{args.task}/{args.dataset}.jsonl"
    audio_root = args.audio_root
    if audio_root is None:
        audio_root = f"data/{args.task}/audio/"

    # ── Resolve local model paths ────────────────────────────────────────
    text_model_name = _resolve_local_model(model_cfg["text_encoder"]["model_name"])
    audio_model_name = _resolve_local_model(model_cfg["audio_encoder"]["model_name"])

    # ── Tokeniser ────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(text_model_name)

    # ── Dataloaders ──────────────────────────────────────────────────────
    train_loader = build_dataloader(
        manifest_path=manifest_path,
        tokenizer=tokenizer,
        audio_root=audio_root,
        batch_size=train_cfg["training"]["batch_size"],
        num_workers=4,
        sample_rate=train_cfg["audio"]["sample_rate"],
        max_duration_sec=train_cfg["audio"]["max_duration_sec"],
        min_words=model_cfg["subtext"]["min_words"],
        max_words=model_cfg["subtext"]["max_words"],
        shuffle=True,
        seed=train_cfg["training"]["seed"],
        distributed=distributed,
    )

    # TODO: supply a separate validation manifest for early stopping
    val_loader = None

    # ── Model ────────────────────────────────────────────────────────────
    model = GLCLAP(
        text_model_name=text_model_name,
        audio_model_name=audio_model_name,
        embed_dim=model_cfg["projection"]["embed_dim"],
        text_freeze_layers=model_cfg["text_encoder"]["freeze_layers"],
        audio_freeze_layers=model_cfg["audio_encoder"]["freeze_layers"],
        detach_encoders=model_cfg.get("detach_encoders", False),
    )
    model = model.to(device)

    if distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                    find_unused_parameters=False)

    # ── Trainer ──────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        output_dir=output_dir,
        lr=train_cfg["optimizer"]["lr"],
        weight_decay=train_cfg["optimizer"]["weight_decay"],
        temperature=train_cfg["training"]["temperature"],
        num_epochs=train_cfg["training"]["num_epochs"],
        grad_clip=train_cfg["training"]["grad_clip"],
        mixed_precision=train_cfg["training"]["mixed_precision"],
        early_stopping_patience=train_cfg["training"]["early_stopping_patience"],
        warmup_steps=train_cfg["scheduler"]["warmup_steps"],
        scheduler_type=train_cfg["scheduler"]["type"],
        local_only=args.local_only,
        device=device,
        rank=rank,
        world_size=world_size,
    )

    trainer.train(resume_from=args.resume)

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
