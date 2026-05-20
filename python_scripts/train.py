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
        [--resume outputs/glclap/checkpoint_epoch010.pt] \
        [--output_dir exp/20250423-143022-libri-960-d2v-large-bert-multi-proj512-bs12/]
"""

from __future__ import annotations

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import datetime
from datetime import timedelta
import re
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


def _shorten_name(name: str) -> str:
    """Shorten a HuggingFace model name for directory naming."""
    name = name.replace("--", "/")
    name = name.split("/")[-1]
    # Common replacements
    name = name.replace("bert-base-multilingual-uncased", "bert-multi")
    name = name.replace("data2vec-audio-large-960h", "d2v-large")
    name = name.replace("data2vec-audio-base-960h", "d2v-base")
    name = name.replace("wav2vec2-large-960h", "w2v-large")
    name = name.replace("wav2vec2-base-960h", "w2v-base")
    name = name.replace("hubert-large-ls960-ft", "hubert-large")
    name = name.replace("hubert-base-ls960", "hubert-base")
    return name


def _build_exp_name(
    dataset: str,
    model_cfg: dict,
    train_cfg: dict,
    local_only: bool,
) -> str:
    """Build an experiment directory name from key hyper-parameters."""
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    audio_name = _shorten_name(model_cfg["audio_encoder"]["model_name"])
    text_name = _shorten_name(model_cfg["text_encoder"]["model_name"])
    embed_dim = model_cfg["projection"]["embed_dim"]
    batch_size = train_cfg["training"]["batch_size"]

    freeze_txt = model_cfg["text_encoder"].get("freeze_layers", 0)
    freeze_aud = model_cfg["audio_encoder"].get("freeze_layers", 0)

    parts = [
        ts,
        dataset,
        audio_name,
        text_name,
        f"proj{embed_dim}",
        f"bs{batch_size}",
    ]

    # Add freeze info only when meaningful
    if freeze_txt > 0 or freeze_aud > 0:
        parts.append(f"freeze{freeze_aud}")

    if local_only:
        parts.append("local")

    if model_cfg.get("detach_encoders", False):
        parts.append("detach")

    return "-".join(parts)


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
    p.add_argument("--output_dir", default=None,
                   help="Output directory for checkpoints and logs. "
                        "If not set, an auto-generated name under exp/ is used.")

    # ── 修改处：新增 flatten_subtexts 开关 ────────────────────────────
    p.add_argument("--flatten_subtexts", action="store_true",
                   help="Flatten subtexts in collate_fn to simulate larger "
                        "effective batch size (batch_size * num_subtexts).")
    # ──────────────────────────────────────────────────────────────────

    # ── 修改处：新增每轮 val 评估参数 ─────────────────────────────────
    p.add_argument("--val_dataset", default=None,
                   help="Validation dataset name (e.g. libri-960-dev). "
                        "If omitted, defaults to {dataset}-dev.")
    p.add_argument("--bias_list", default=None,
                   help="Global bias list file for per-epoch evaluation.")
    p.add_argument("--per_sample_bias_dir", default=None,
                   help="Per-sample bias list directory for per-epoch evaluation.")
    p.add_argument("--val_threshold", type=float, default=0.5,
                   help="Similarity threshold for per-epoch evaluation (default: 0.5).")
    p.add_argument("--val_top_k", type=int, default=10,
                   help="Top-k for per-epoch evaluation (default: 10).")
    # ──────────────────────────────────────────────────────────────────

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

    # Determine output directory
    if args.output_dir is not None:
        output_dir = args.output_dir
    else:
        exp_name = _build_exp_name(
            dataset=args.dataset,
            model_cfg=model_cfg,
            train_cfg=train_cfg,
            local_only=args.local_only,
        )
        output_dir = f"exp/{exp_name}"

    if rank == 0:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
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
    # ── 修改处：build_dataloader 传入 flatten_subtexts ───────────────
    train_loader = build_dataloader(
        manifest_path=manifest_path,
        tokenizer=tokenizer,
        audio_root=audio_root,
        batch_size=train_cfg["training"]["batch_size"],
        num_workers=0,
        sample_rate=train_cfg["audio"]["sample_rate"],
        max_duration_sec=train_cfg["audio"]["max_duration_sec"],
        min_words=model_cfg["subtext"]["min_words"],
        max_words=model_cfg["subtext"]["max_words"],
        shuffle=True,
        seed=train_cfg["training"]["seed"],
        distributed=distributed,
        flatten_subtexts=args.flatten_subtexts,
    )
    # ──────────────────────────────────────────────────────────────────

    # Validation loader (optional)
    # ── 修改处：支持自定义 val_dataset，传入 flatten_subtexts ─────────
    val_dataset_name = args.val_dataset or f"{args.dataset}-dev"
    val_manifest_path = f"data/{args.task}/{val_dataset_name}.jsonl"
    val_loader = None
    if os.path.exists(val_manifest_path):
        val_loader = build_dataloader(
            manifest_path=val_manifest_path,
            tokenizer=tokenizer,
            audio_root=audio_root,
            batch_size=train_cfg["training"]["batch_size"],
            num_workers=0,
            sample_rate=train_cfg["audio"]["sample_rate"],
            max_duration_sec=train_cfg["audio"]["max_duration_sec"],
            min_words=model_cfg["subtext"]["min_words"],
            max_words=model_cfg["subtext"]["max_words"],
            shuffle=False,
            seed=train_cfg["training"]["seed"],
            distributed=distributed,
            flatten_subtexts=args.flatten_subtexts,
        )
    # ──────────────────────────────────────────────────────────────────

    # ── Model ────────────────────────────────────────────────────────────
    model = GLCLAP(
        text_model_name=text_model_name,
        audio_model_name=audio_model_name,
        embed_dim=model_cfg["projection"]["embed_dim"],
        text_freeze_layers=model_cfg["text_encoder"]["freeze_layers"],
        audio_freeze_layers=model_cfg["audio_encoder"]["freeze_layers"],
        detach_encoders=model_cfg.get("detach_encoders", False),
        text_use_attention_pooling=model_cfg["text_encoder"].get("use_attention_pooling", False),
        audio_use_attention_pooling=model_cfg["audio_encoder"].get("use_attention_pooling", False),
    )
    model = model.to(device)

    if distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                    find_unused_parameters=True)

    # ── 修改处：构建 val_eval_config 传给 Trainer ────────────────────
    val_eval_config = None
    if args.bias_list or args.per_sample_bias_dir:
        val_eval_config = {
            "task": args.task,
            "dataset": val_dataset_name,
            "model_config": args.model_config,
            "bias_list": args.bias_list,
            "per_sample_bias_dir": args.per_sample_bias_dir,
            "threshold": args.val_threshold,
            "top_k": args.val_top_k,
            "eval_batch_size": 8,
        }
    # ──────────────────────────────────────────────────────────────────

    # ── Trainer ──────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        val_eval_config=val_eval_config,
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
        seed=train_cfg["training"]["seed"],
        tokenizer=tokenizer,
    )

    trainer.train(resume_from=args.resume)

    if distributed:
        dist.destroy_process_group()
        dist.init_process_group(backend="nccl", timeout=timedelta(hours=2))

if __name__ == "__main__":
    main()
