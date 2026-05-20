"""
dataset.py
──────────
PyTorch Dataset and DataLoader utilities for GLCLAP training.

Expected manifest format (JSONL, one sample per line):
    {"audio_path": "/path/to/audio.wav", "text": "full transcription"}

Each __getitem__ returns a dict with:
    - "waveform":      raw PCM tensor  [1, T_samples]  — for Data2Vec processor
    - "text":          full transcription string        (global branch)
    - "subtext":       randomly sampled subtext string  (local branch, 兼容)
    - "subtexts":      list of up to 10 constrained subtexts  (扩展)
    - "audio_path":    original file path (for debugging)

Batching is handled by collate_fn which tokenises text/subtext via the
BERT tokeniser (padding handled here, not in the model).

Design decision:
    Tokenisation is done in collate_fn (not __getitem__) so that the
    DataLoader can be used with different tokenisers without touching the
    dataset class. The tokeniser is passed to the DataLoader as a callable
    collate_fn via functools.partial.

MODIFIED (2026-05-13):
    1) 引入全局约束 subtext 预生成（generate_constrained_subtexts）。
       在 Dataset 初始化阶段一次性为全部文本预生成最多 10 个满足约束的
       subtexts：
           - 同文本 subtexts 之间词不重复
           - 全局正例隔离：一个词只能作为一条文本的正例

    2) collate_fn 新增 flatten_subtexts 模式：将 B 个样本展平为 B*K
       个虚拟样本（K = 每条文本的 subtext 数量，最多 10）。音频 tensor
       复制 K 次，subtexts 展平为一维列表，effective batch size 从 B
       放大到 B*K，直接模拟大 batch 的负例丰富度。
       标准 InfoNCE loss 无需修改即可使用（diagonal 自动匹配）。
"""

from __future__ import annotations

import json
import random
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from transformers import AutoTokenizer

from .audio_utils import load_audio, extract_mel
from .subtext import sample_subtext, generate_constrained_subtexts


class GLCLAPDataset(Dataset):
    """
    Dataset for GLCLAP training.

    Each sample provides:
        - raw waveform (for the audio encoder)
        - full text transcription (global text branch)
        - randomly sampled subtext (local text branch, 兼容)
        - list of constrained subtexts (扩展，最多 10 个)

    Args:
        manifest_path:         Path to JSONL manifest file.
        audio_root:            Optional base directory prepended to relative audio paths.
        sample_rate:           Target audio sample rate (Hz).
        max_duration_sec:      Clips longer than this are truncated.
        min_words:             Minimum subtext span length (words).
        max_words:             Maximum subtext span length (words).
        num_subtexts_per_text: 每条文本预生成的 subtext 数量（默认 10）。
        seed:                  Optional random seed for subtext sampling.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        audio_root: Optional[str | Path] = None,
        sample_rate: int = 16_000,
        max_duration_sec: float = 30.0,
        min_words: int = 1,
        max_words: int = 5,
        num_subtexts_per_text: int = 20,
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

        # ── 修改处：预生成满足全局约束的 subtexts ─────────────────
        texts = [s["text"] for s in self.samples]
        self.subtexts_per_sample: list[list[str]] = generate_constrained_subtexts(
            texts=texts,
            num_subtexts_per_text=num_subtexts_per_text,
            min_words=min_words,
            max_words=max_words,
            seed=seed,
        )
        # ──────────────────────────────────────────────────────────

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
                "subtext":    str  — randomly sampled subtext (兼容)
                "subtexts":   list[str] — up to 10 constrained subtexts
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

        text = item["text"]

        # ── 修改处：从预生成的 subtexts 中随机选 1 个 ──────────────
        candidate_subtexts = self.subtexts_per_sample[idx]
        subtext = self.rng.choice(candidate_subtexts)
        # ──────────────────────────────────────────────────────────

        return {
            "waveform": waveform,
            "text": text,
            "subtext": subtext,
            "subtexts": candidate_subtexts,
            "audio_path": audio_path,
        }


def collate_fn(
    batch: List[Dict[str, Any]],
    tokenizer: AutoTokenizer,
    max_text_length: int = 128,
    max_subtext_length: int = 32,
    flatten_subtexts: bool = False,
) -> Dict[str, Any]:
    """
    Custom collate function for DataLoader.

    Two modes:
      1) flatten_subtexts=False（默认，兼容模式）：
         每条样本只取 1 个 subtext，batch size = B，行为与原版完全一致。

      2) flatten_subtexts=True（展平模式，模拟大 batch）：
         将 B 条样本各自复制 K 次（K = 该样本的 subtext 数量，≤10），
         展平为 B*K 个虚拟样本。音频、文本、subtext 均相应复制/展平。
         Effective batch size 从 B 放大到 B*K，InfoNCE 的 in-batch
         负例数从 B-1 增加到 B*K-1，直接模拟大 batch 效果。
         标准 InfoNCE loss 无需修改（diagonal 自动匹配正例对）。

    MODIFIED (2026-05-14):
        新增 sample_ids 字段。展平模式下返回 [B*K] 分组标识，
        用于模型端对同一音频/全局文本进行去重编码，避免重复 forward
        浪费计算与显存。

    Args:
        batch:             List of dicts from GLCLAPDataset.__getitem__.
        tokenizer:         HuggingFace tokenizer.
        max_text_length:   Token length cap for full-text branch.
        max_subtext_length: Token length cap for subtext branch.
        flatten_subtexts:  是否展平 subtexts 以模拟大 batch（默认 False）。

    Returns:
        dict with keys（兼容模式）:
            "waveforms":          [B, 1, T_max]
            "waveform_lengths":   [B]
            "text_input_ids":     [B, N]
            "subtext_input_ids":  [B, N']
            ...
        或（展平模式）:
            "waveforms":          [B*K, 1, T_max]
            "waveform_lengths":   [B*K]
            "text_input_ids":     [B*K, N]
            "subtext_input_ids":  [B*K, N']
            ...
    """
    # ── 修改处：展平模式 ─────────────────────────────────────────
    if flatten_subtexts:
        # 将 B 个样本各自复制 K 次，展平为 B*K 个虚拟样本
        # 排列：sample0_sub0, sample0_sub1, ..., sample0_subK0-1,
        #       sample1_sub0, sample1_sub1, ..., sample1_subK1-1, ...
        flat_waveforms = []
        flat_texts = []
        flat_subtexts = []
        flat_paths = []
        sample_ids = []  # MODIFIED (2026-05-14): 记录每个展平样本属于哪个原始样本

        for i, item in enumerate(batch):
            subs = item["subtexts"]  # list of K subtexts
            K = len(subs)
            flat_waveforms.extend([item["waveform"]] * K)
            flat_texts.extend([item["text"]] * K)
            flat_subtexts.extend(subs)
            flat_paths.extend([item["audio_path"]] * K)
            sample_ids.extend([i] * K)

        waveforms = flat_waveforms
        texts = flat_texts
        subtexts = flat_subtexts
        paths = flat_paths
    else:
        # 原代码：兼容模式，每条样本只取 1 个 subtext
        waveforms = [item["waveform"] for item in batch]
        texts = [item["text"] for item in batch]
        subtexts = [item["subtext"] for item in batch]
        paths = [item["audio_path"] for item in batch]
    # ─────────────────────────────────────────────────────────────

    # Pad waveforms to max length in batch
    lengths = torch.tensor([w.shape[-1] for w in waveforms])
    T_max = int(lengths.max().item())
    padded_waveforms = torch.zeros(len(waveforms), 1, T_max)
    waveform_attention_mask = torch.zeros(len(waveforms), T_max, dtype=torch.long)
    for i, w in enumerate(waveforms):
        padded_waveforms[i, :, : w.shape[-1]] = w
        waveform_attention_mask[i, : w.shape[-1]] = 1

    # Tokenise text (global branch)
    text_enc = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_text_length,
        return_tensors="pt",
    )

    # Tokenise subtext (local branch)
    subtext_enc = tokenizer(
        subtexts,
        padding="max_length",
        truncation=True,
        max_length=max_subtext_length,
        return_tensors="pt",
    )

    result = {
        "waveforms": padded_waveforms,
        "waveform_lengths": lengths,
        "waveform_attention_mask": waveform_attention_mask,
        "text_input_ids": text_enc["input_ids"],
        "text_attention_mask": text_enc["attention_mask"],
        "subtext_input_ids": subtext_enc["input_ids"],
        "subtext_attention_mask": subtext_enc["attention_mask"],
        "audio_paths": paths,
    }
    # MODIFIED (2026-05-14): 展平模式下返回 sample_ids，供模型去重编码
    if flatten_subtexts:
        result["sample_ids"] = torch.tensor(sample_ids, dtype=torch.long)
    return result


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
    num_subtexts_per_text: int = 20,
    max_text_length: int = 128,
    max_subtext_length: int = 32,
    shuffle: bool = True,
    seed: Optional[int] = None,
    distributed: bool = False,
    flatten_subtexts: bool = False,
) -> DataLoader:
    """
    Convenience factory: build a DataLoader for one dataset split.

    Args:
        manifest_path:         Path to JSONL manifest.
        tokenizer:             HuggingFace tokenizer instance.
        audio_root:            Optional base directory for audio files.
        batch_size:            Mini-batch size（物理 batch size，展平前）。
        num_workers:           DataLoader worker processes.
        sample_rate:           Audio sample rate.
        max_duration_sec:      Truncation threshold.
        min_words:             Subtext min span.
        max_words:             Subtext max span.
        num_subtexts_per_text: 每条文本预生成的 subtext 数量（默认 10）。
        max_text_length:       Max BERT token length for global text.
        max_subtext_length:    Max BERT token length for subtext.
        shuffle:               Whether to shuffle (True for training).
        seed:                  RNG seed for subtext sampling.
        distributed:           Whether to use DistributedSampler.
        flatten_subtexts:      是否展平 subtexts 以模拟大 batch（默认 False）。
                               若 True，effective batch size = batch_size * K。

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
        num_subtexts_per_text=num_subtexts_per_text,
        seed=seed,
    )

    _collate = partial(
        collate_fn,
        tokenizer=tokenizer,
        max_text_length=max_text_length,
        max_subtext_length=max_subtext_length,
        flatten_subtexts=flatten_subtexts,
    )

    sampler = None
    if distributed:
        sampler = DistributedSampler(dataset, shuffle=shuffle, seed=seed)
        shuffle = False

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=_collate,
        pin_memory=True,
        drop_last=True,
    )
