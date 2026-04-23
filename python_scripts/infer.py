"""
infer.py
────────
Run GLCLAP bias-word retrieval on a single audio file.

Usage:
    python python_scripts/infer.py \
        --checkpoint outputs/glclap/best_model.pt \
        --audio /path/to/audio.wav \
        --bias_list data/bias_lists/phonecall.txt \
        --model_config configs/model_config.yaml \
        [--threshold 0.5] [--top_k 5]

Output:
    Prints selected bias words and their similarity scores.
    These can then be passed as prompts to Whisper or another ASR model.
"""

from __future__ import annotations

import argparse
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
from inference.retriever import BiasWordRetriever
from models.glclap import GLCLAP
from utils.checkpointing import load_checkpoint
from utils.logging import setup_logging


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GLCLAP bias-word inference")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--audio", required=True, help="Path to audio file")
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

    # ── Resolve local model paths ────────────────────────────────────────
    text_model_name = _resolve_local_model(model_cfg["text_encoder"]["model_name"])
    audio_model_name = _resolve_local_model(model_cfg["audio_encoder"]["model_name"])

    # ── Bias list ─────────────────────────────────────────────────────────
    with open(args.bias_list, encoding="utf-8") as f:
        bias_words = [line.strip() for line in f if line.strip()]
    print(f"Loaded {len(bias_words)} bias words.")

    # ── Model ─────────────────────────────────────────────────────────────
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

    # ── Audio ─────────────────────────────────────────────────────────────
    waveform, sr = load_audio(args.audio, target_sr=16_000)
    print(f"Audio: {args.audio}  ({waveform.shape[-1] / sr:.1f}s)")

    # ── Retrieve ──────────────────────────────────────────────────────────
    selected = retriever.retrieve(waveform.squeeze(0))  # [T_samples]

    print("\n── Retrieved Bias Words ────────────────")
    if selected:
        for w in selected:
            print(f"  → {w}")
    else:
        print("  (none above threshold)")
    print("────────────────────────────────────────")
    print(f"\nPrompt for ASR: {', '.join(selected)}\n")

    # ── Optional: full similarity matrix ─────────────────────────────────
    Sim = retriever.similarity_matrix(waveform.squeeze(0))
    print(f"Similarity matrix shape: {list(Sim.shape)}  [K={len(bias_words)}, T'=...]")


if __name__ == "__main__":
    main()
