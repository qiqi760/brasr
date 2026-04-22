"""
train.py
────────
Main entry point for GLCLAP training.

Usage:
    python scripts/train.py \
        --model_config configs/model_config.yaml \
        --train_config configs/train_config.yaml \
        [--resume outputs/glclap/checkpoint_epoch010.pt]
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
import yaml

# Ensure project root is on the Python path when running as a script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import build_dataloader
from models.glclap import GLCLAP
from training.trainer import Trainer
from transformers import AutoTokenizer
from utils.logging import setup_logging


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train GLCLAP")
    p.add_argument("--model_config", default="configs/model_config.yaml")
    p.add_argument("--train_config", default="configs/train_config.yaml")
    p.add_argument("--resume", default=None, help="Checkpoint path to resume from")
    p.add_argument("--local_only", action="store_true",
                   help="Use only local (Ll) loss — trains LCLAP instead of GLCLAP")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.model_config) as f:
        model_cfg = yaml.safe_load(f)
    with open(args.train_config) as f:
        train_cfg = yaml.safe_load(f)

    output_dir = train_cfg["training"]["output_dir"]
    setup_logging(log_dir=output_dir)

    # ── Tokeniser ────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["text_encoder"]["model_name"]
    )

    # ── Dataloaders ──────────────────────────────────────────────────────
    # TODO: Split one of the manifests into train/val, or add a separate val manifest.
    # For now we concatenate all train manifests and use the first one as a val proxy.
    all_manifests = train_cfg["datasets"]
    train_loaders = []
    for ds in all_manifests:
        loader = build_dataloader(
            manifest_path=ds["manifest"],
            tokenizer=tokenizer,
            audio_root=ds.get("audio_root"),
            batch_size=train_cfg["training"]["batch_size"],
            num_workers=4,
            sample_rate=train_cfg["audio"]["sample_rate"],
            max_duration_sec=train_cfg["audio"]["max_duration_sec"],
            min_words=model_cfg["subtext"]["min_words"],
            max_words=model_cfg["subtext"]["max_words"],
            shuffle=True,
            seed=train_cfg["training"]["seed"],
        )
        train_loaders.append(loader)

    # For multi-dataset training, combine by round-robin or concatenation.
    # TODO: Implement CombinedDataLoader or use torch.utils.data.ConcatDataset.
    # For now, use only the first dataset loader as a placeholder.
    train_loader = train_loaders[0]

    # ── Model ────────────────────────────────────────────────────────────
    model = GLCLAP(
        text_model_name=model_cfg["text_encoder"]["model_name"],
        audio_model_name=model_cfg["audio_encoder"]["model_name"],
        embed_dim=model_cfg["projection"]["embed_dim"],
        text_freeze_layers=model_cfg["text_encoder"]["freeze_layers"],
        audio_freeze_layers=model_cfg["audio_encoder"]["freeze_layers"],
    )

    # ── Trainer ──────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=None,             # TODO: supply a validation DataLoader
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
        device=args.device,
    )

    trainer.train(resume_from=args.resume)


if __name__ == "__main__":
    main()
