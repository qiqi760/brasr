"""
evaluate.py
───────────
Evaluate a trained GLCLAP checkpoint for bias-word retrieval.

Usage:
    python python_scripts/evaluate.py \
        --task contrastive-learning \
        --dataset libri-960 \
        --checkpoint outputs/glclap/best_model.pt \
        --bias_list data/bias_lists/libri_bias.txt \
        --model_config configs/model_config.yaml \
        --threshold 0.5

The manifest format (JSONL) should include:
    {"audio_path": "...", "text": "...", "bias_entities": ["entity1", ...]}
The bias_list file is a plain text file with one phrase per line.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import yaml
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _resolve_local_model(model_name: str) -> str:
    """优先使用本地 pretrained_models/ 目录下的模型。"""
    project_root = Path(__file__).parent.parent.resolve()
    local_dir = project_root / "pretrained_models" / model_name.replace("/", "--")
    if local_dir.exists():
        return str(local_dir)
    return model_name

from dataset.audio_utils import load_audio
from eval.metrics import evaluate_retrieval
from inference.retriever import BiasWordRetriever
from models.glclap import GLCLAP
from utils.checkpointing import load_checkpoint
from utils.logging import setup_logging


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate GLCLAP bias-word retrieval")
    p.add_argument("--task", required=True, help="Task name, e.g. contrastive-learning")
    p.add_argument("--dataset", required=True, help="Dataset name, e.g. libri-960")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--bias_list", required=True, help="Plain text file: one bias phrase per line")
    p.add_argument("--model_config", default="configs/model_config.yaml")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--top_k", type=int, default=10)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging()

    with open(args.model_config) as f:
        model_cfg = yaml.safe_load(f)

    # ── Build manifest path from task + dataset ──────────────────────────
    manifest_path = f"data/{args.task}/{args.dataset}.jsonl"

    # ── Resolve local model paths ────────────────────────────────────────
    text_model_name = _resolve_local_model(model_cfg["text_encoder"]["model_name"])
    audio_model_name = _resolve_local_model(model_cfg["audio_encoder"]["model_name"])

    # ── Load bias list ────────────────────────────────────────────────────
    with open(args.bias_list, encoding="utf-8") as f:
        bias_words = [line.strip() for line in f if line.strip()]

    # ── Load model ────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(text_model_name)
    model = GLCLAP(
        text_model_name=text_model_name,
        audio_model_name=audio_model_name,
        embed_dim=model_cfg["projection"]["embed_dim"],
    )
    load_checkpoint(args.checkpoint, model, strict=True)

    retriever = BiasWordRetriever(
        model=model,
        tokenizer=tokenizer,
        threshold=args.threshold,
        top_k=args.top_k,
        device=args.device,
    )
    retriever.set_bias_list(bias_words)

    # ── Run evaluation ────────────────────────────────────────────────────
    all_predictions: list[list[str]] = []
    all_ground_truths: list[list[str]] = []
    top1_ground_truths: list[str] = []

    with open(manifest_path, encoding="utf-8") as f:
        for line in f:
            item = json.loads(line.strip())
            audio_path = item["audio_path"]
            gt_entities = item.get("bias_entities", [])

            waveform, _ = load_audio(audio_path, target_sr=16_000)
            waveform = waveform.squeeze(0)  # [T_samples]

            predicted = retriever.retrieve(waveform)

            all_predictions.append(predicted)
            all_ground_truths.append(gt_entities)
            top1_ground_truths.append(gt_entities[0] if gt_entities else "")

    results = evaluate_retrieval(
        all_predictions=all_predictions,
        all_ground_truths=all_ground_truths,
        top1_ground_truths=top1_ground_truths,
    )

    print("\n── Evaluation Results ──────────────────")
    for k, v in results.items():
        print(f"  {k:<16}: {v * 100:.2f}%")
    print("────────────────────────────────────────\n")


if __name__ == "__main__":
    main()
