"""
dataset.py
──────────
PyTorch Dataset and DataLoader utilities for GLCLAP training.

Expected manifest format (JSONL, one sample per line):
    {"audio_path": "/path/to/audio.wav", "text": "full transcription"}

Each __getitem__ returns a dict with:
    - "waveform":      raw PCM tensor  [1, T_samples]  — for Data2Vec processor
    - "text":          full transcription string        (global branch)
    - "subtext":       randomly sampled subtext string  (local branch)
    - "audio_path":    original file path (for debugging)

Batching is handled by collate_fn which tokenises text/subtext via the
BERT tokeniser (padding handled here, not in the model).

Design decision:
    Tokenisation is done in collate_fn (not __getitem__) so that the
    DataLoader can be used with different tokenisers without touching the
    dataset class. The tokeniser is passed to the DataLoader as a callable
    collate_fn via functools.partial.
"""

from __future__ import annotations

import json
import random
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from .audio_utils import load_audio
from .subtext import sample_subtext


class GLCLAPDataset(Dataset):
    """
    Dataset for GLCLAP training.

    Each sample provides:
        - raw waveform (for the audio encoder)
        - full text transcription (global text branch)
        - randomly sampled subtext (local text branch)

    Args:
        manifest_path:    Path to JSONL manifest file.
        audio_root:       Optional base directory prepended to relative audio paths.
        sample_rate:      Target audio sample rate (Hz).
        max_duration_sec: Clips longer than this are truncated.
        min_words:        Minimum subtext span length (words).
        max_words:        Maximum subtext span length (words).
        seed:             Optional random seed for subtext sampling.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        audio_root: Optional[str | Path] = None,
        sample_rate: int = 16_000,
        max_duration_sec: float = 30.0,
        min_words: int = 1,
        max_words: int = 5,
        seed: Optional[int] = None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.audio_root = Path(audio_root) if audio_root else None
        self.sample_rate = sample_rate
        self.max_duration_sec = max_duration_sec
        self.min_words = min_words
        self.max_words = max_words
        self.rng = random.Random(seed)

        self.samples: List[Dict[str, str]] = self._load_manifest()

    def _load_manifest(self) -> List[Dict[str, str]]:
        """Load JSONL manifest; each line must have 'audio_path' and 'text'."""
        samples = []
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                assert "audio_path" in item and "text" in item, (
                    f"Manifest entry missing 'audio_path' or 'text': {item}"
                )
                samples.append(item)
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Returns:
            dict with keys:
                "waveform":   [1, T_samples]  float32 tensor
                "text":       str  — full transcription
                "subtext":    str  — randomly sampled word span
                "audio_path": str  — original file path
        """
        item = self.samples[idx]
        audio_path = item["audio_path"]
        if self.audio_root is not None:
            audio_path = str(self.audio_root / audio_path)

        waveform, _ = load_audio(
            audio_path,
            target_sr=self.sample_rate,
            max_duration_sec=self.max_duration_sec,
        )
        # waveform: [1, T_samples]

        text = item["text"]
        subtext = sample_subtext(
            text,
            min_words=self.min_words,
            max_words=self.max_words,
            rng=self.rng,
        )

        return {
            "waveform": waveform,    # [1, T_samples]
            "text": text,
            "subtext": subtext,
            "audio_path": audio_path,
        }


def collate_fn(
    batch: List[Dict[str, Any]],
    tokenizer: AutoTokenizer,
    max_text_length: int = 128,
    max_subtext_length: int = 32,
) -> Dict[str, Any]:
    """
    Custom collate function for DataLoader.

    Tokenises text and subtext with the BERT tokeniser.
    Pads waveforms to the longest in the batch.

    Args:
        batch:             List of dicts from GLCLAPDataset.__getitem__.
        tokenizer:         HuggingFace tokenizer (e.g. BertTokenizerFast).
        max_text_length:   Token length cap for full-text branch.
        max_subtext_length: Token length cap for subtext branch.

    Returns:
        dict with keys:
            "waveforms":          [B, 1, T_max]  — zero-padded waveforms
            "waveform_lengths":   [B]             — actual sample lengths
            "text_input_ids":     [B, N]          — tokenised full text
            "text_attention_mask":[B, N]
            "subtext_input_ids":  [B, N']         — tokenised subtext
            "subtext_attention_mask": [B, N']
            "audio_paths":        List[str]

    Shapes:
        B  = batch size
        N  = max_text_length (padded)
        N' = max_subtext_length (padded)
        T_max = longest waveform in batch (samples)
    """
    waveforms = [item["waveform"] for item in batch]          # each [1, T_i]
    texts   = [item["text"]    for item in batch]
    subtexts = [item["subtext"] for item in batch]
    paths   = [item["audio_path"] for item in batch]

    # Pad waveforms to max length in batch
    lengths = torch.tensor([w.shape[-1] for w in waveforms])   # [B]
    T_max = int(lengths.max().item())
    padded_waveforms = torch.zeros(len(waveforms), 1, T_max)   # [B, 1, T_max]
    for i, w in enumerate(waveforms):
        padded_waveforms[i, :, : w.shape[-1]] = w

    # Tokenise text (global branch)
    text_enc = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_text_length,
        return_tensors="pt",
    )
    # text_enc["input_ids"]: [B, N]  text_enc["attention_mask"]: [B, N]

    # Tokenise subtext (local branch)
    subtext_enc = tokenizer(
        subtexts,
        padding="max_length",
        truncation=True,
        max_length=max_subtext_length,
        return_tensors="pt",
    )
    # subtext_enc["input_ids"]: [B, N']

    return {
        "waveforms": padded_waveforms,                         # [B, 1, T_max]
        "waveform_lengths": lengths,                           # [B]
        "text_input_ids": text_enc["input_ids"],               # [B, N]
        "text_attention_mask": text_enc["attention_mask"],     # [B, N]
        "subtext_input_ids": subtext_enc["input_ids"],         # [B, N']
        "subtext_attention_mask": subtext_enc["attention_mask"],# [B, N']
        "audio_paths": paths,
    }


def build_dataloader(
    manifest_path: str | Path,
    tokenizer: AutoTokenizer,
    audio_root: Optional[str | Path] = None,
    batch_size: int = 64,
    num_workers: int = 4,
    sample_rate: int = 16_000,
    max_duration_sec: float = 30.0,
    min_words: int = 1,
    max_words: int = 5,
    max_text_length: int = 128,
    max_subtext_length: int = 32,
    shuffle: bool = True,
    seed: Optional[int] = None,
) -> DataLoader:
    """
    Convenience factory: build a DataLoader for one dataset split.

    Args:
        manifest_path:    Path to JSONL manifest.
        tokenizer:        HuggingFace tokenizer instance.
        audio_root:       Optional base directory for audio files.
        batch_size:       Mini-batch size (paper: 64).
        num_workers:      DataLoader worker processes.
        sample_rate:      Audio sample rate.
        max_duration_sec: Truncation threshold.
        min_words:        Subtext min span.
        max_words:        Subtext max span.
        max_text_length:  Max BERT token length for global text.
        max_subtext_length: Max BERT token length for local subtext.
        shuffle:          Whether to shuffle (True for training).
        seed:             RNG seed for subtext sampling.

    Returns:
        DataLoader yielding batches from collate_fn.
    """
    dataset = GLCLAPDataset(
        manifest_path=manifest_path,
        audio_root=audio_root,
        sample_rate=sample_rate,
        max_duration_sec=max_duration_sec,
        min_words=min_words,
        max_words=max_words,
        seed=seed,
    )

    _collate = partial(
        collate_fn,
        tokenizer=tokenizer,
        max_text_length=max_text_length,
        max_subtext_length=max_subtext_length,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=_collate,
        pin_memory=True,
        drop_last=True,
    )
