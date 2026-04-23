"""
download_models.py
──────────────────
Download pretrained models to local disk for offline cluster usage.

Two download sources are supported:
    1. HuggingFace Hub   (requires VPN / overseas network)
    2. ModelScope (魔搭)  (domestic China mirror, usually faster)

Usage:
    # Option A: HuggingFace (default)
    python python_scripts/download_models.py --source huggingface --output_dir ./pretrained_models

    # Option B: ModelScope (recommended for users in mainland China)
    python python_scripts/download_models.py --source modelscope --output_dir ./pretrained_models

After downloading, copy the entire ./pretrained_models folder to your cluster
and update configs/model_config.yaml to point to these local paths.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download GLCLAP pretrained models for offline use")
    p.add_argument("--source", choices=["huggingface", "modelscope"], default="huggingface",
                   help="Download source: huggingface (default) or modelscope")
    p.add_argument("--output_dir", default="./pretrained_models",
                   help="Directory to save downloaded models")
    p.add_argument("--text_model", default="bert-base-multilingual-uncased",
                   help="HuggingFace model ID for text encoder (will be mapped to ModelScope ID if needed)")
    p.add_argument("--audio_model", default="facebook/data2vec-audio-large-960h",
                   help="HuggingFace model ID for audio encoder")
    return p.parse_args()


def download_from_huggingface(model_id: str, output_dir: Path) -> Path:
    """Download a model from HuggingFace Hub to a local directory."""
    from transformers import AutoTokenizer, AutoModel
    from transformers.utils import logging as transformers_logging
    transformers_logging.set_verbosity_info()

    local_path = output_dir / model_id.replace("/", "--")
    local_path.mkdir(parents=True, exist_ok=True)

    print(f"\n[HF] Downloading '{model_id}' -> {local_path}")

    # Download tokenizer + model weights
    print("  -> Downloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.save_pretrained(local_path)

    print("  -> Downloading model...")
    model = AutoModel.from_pretrained(model_id)
    model.save_pretrained(local_path)

    print(f"  -> Done: {local_path}")
    return local_path


# Mapping from HuggingFace ID -> ModelScope ID (for models without a direct HF namespace match)
HF_TO_MODELSCOPE: dict[str, str] = {
    "bert-base-multilingual-uncased": "google-bert/bert-base-multilingual-uncased",
}


def download_from_modelscope(model_id: str, output_dir: Path) -> Path:
    """Download a model from ModelScope to a local directory."""
    try:
        from modelscope import snapshot_download
    except ImportError:
        print("\nError: modelscope is not installed.")
        print("Please install it first:")
        print("    pip install modelscope")
        sys.exit(1)

    # Map HF ID to ModelScope ID if needed
    ms_model_id = HF_TO_MODELSCOPE.get(model_id, model_id)
    if ms_model_id != model_id:
        print(f"  -> Mapped HF '{model_id}' to ModelScope '{ms_model_id}'")

    local_path = output_dir / model_id.replace("/", "--")
    local_path.mkdir(parents=True, exist_ok=True)

    print(f"\n[ModelScope] Downloading '{ms_model_id}' -> {local_path}")

    # snapshot_download handles the full model repo
    downloaded = snapshot_download(ms_model_id, cache_dir=str(output_dir))
    print(f"  -> Downloaded to cache: {downloaded}")

    # For convenience, copy to our naming convention (using original HF ID as folder name)
    target = output_dir / model_id.replace("/", "--")
    if not target.exists():
        import shutil
        shutil.copytree(downloaded, target, dirs_exist_ok=True)
        print(f"  -> Copied to: {target}")
    else:
        print(f"  -> Already exists: {target}")

    return target


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(" GLCLAP Pretrained Model Downloader")
    print("=" * 60)
    print(f" Source     : {args.source}")
    print(f" Output dir : {output_dir}")
    print(f" Text model : {args.text_model}")
    print(f" Audio model: {args.audio_model}")
    print("=" * 60)

    if args.source == "huggingface":
        downloader = download_from_huggingface
    else:
        downloader = download_from_modelscope

    text_path = downloader(args.text_model, output_dir)
    audio_path = downloader(args.audio_model, output_dir)

    print("\n" + "=" * 60)
    print(" Download complete!")
    print("=" * 60)
    print(f"\nNext steps:")
    print(f"  1. Copy the folder to your cluster:")
    print(f"       rsync -avz {output_dir}/ <cluster>:/path/to/project/pretrained_models/")
    print(f"  2. Update configs/model_config.yaml:")
    print(f"       text_encoder:")
    print(f"         model_name: \"{text_path}\"")
    print(f"       audio_encoder:")
    print(f"         model_name: \"{audio_path}\"")
    print(f"  3. Set environment variable on cluster (optional but recommended):")
    print(f"       export TRANSFORMERS_OFFLINE=1")
    print("=" * 60)


if __name__ == "__main__":
    main()
