"""
audio_utils.py
──────────────
Waveform loading and Mel-spectrogram extraction utilities.

Paper (Section 2.2):
    Audio encoder input: Xa ∈ R^{B × T × F}
        T = number of time frames
        F = number of Mel filterbank bins

The Data2Vec audio encoder (transformer-based) typically accepts raw waveform,
NOT mel spectrograms directly. We expose both interfaces:
  - extract_mel():  for models that consume spectrograms
  - load_audio():   raw waveform, which is what HuggingFace Data2Vec processors expect

TODO: Confirm whether the private pre-trained Data2Vec variant used by the paper
      consumes raw waveform (likely) or mel features.  The HF
      Data2VecAudioModel + Wav2Vec2Processor pipeline works with raw PCM.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T


def load_audio(
    path: str | Path,
    target_sr: int = 16_000,
    max_duration_sec: float = 30.0,
    mono: bool = True,
) -> Tuple[torch.Tensor, int]:
    """
    Load an audio file and resample to ``target_sr``.

    Args:
        path:             Path to audio file (wav, flac, mp3, …).
        target_sr:        Desired sample rate in Hz.
        max_duration_sec: Clips longer than this are silently truncated.
        mono:             If True, average channels to produce mono signal.

    Returns:
        waveform: Float tensor of shape [1, num_samples] (mono) or
                  [C, num_samples] (multi-channel).
        sample_rate: Integer sample rate after resampling (== target_sr).

    Shape:
        output waveform: [1, T_samples]
    """
    waveform, sr = torchaudio.load(str(path))  # [C, T_samples]

    if mono and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)  # [1, T_samples]

    if sr != target_sr:
        resampler = T.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)

    max_samples = int(max_duration_sec * target_sr)
    if waveform.shape[-1] > max_samples:
        waveform = waveform[..., :max_samples]

    return waveform, target_sr


def extract_mel(
    waveform: torch.Tensor,
    sample_rate: int = 16_000,
    n_mels: int = 128,
    n_fft: int = 1024,
    hop_length: int = 160,
    win_length: int = 400,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Compute log-Mel spectrogram from a raw waveform.

    Args:
        waveform:    Float tensor of shape [1, T_samples] or [T_samples].
        sample_rate: Sampling rate in Hz.
        n_mels:      Number of Mel filterbank channels (F).
        n_fft:       FFT size.
        hop_length:  Hop size in samples (≈10 ms at 16 kHz).
        win_length:  Window size in samples (≈25 ms at 16 kHz).
        normalize:   If True, apply log1p and global mean/std normalisation.

    Returns:
        mel: Float tensor of shape [T_frames, F] — ready for batching.

    Shape:
        input  waveform: [1, T_samples]
        output mel:      [T_frames, n_mels]
    """
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)  # [1, T_samples]

    mel_transform = T.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        n_mels=n_mels,
    )
    mel = mel_transform(waveform)  # [1, n_mels, T_frames]
    mel = mel.squeeze(0).T        # [T_frames, n_mels]
    mel = torch.log1p(mel)

    if normalize:
        mel = (mel - mel.mean()) / (mel.std() + 1e-8)

    return mel  # [T_frames, n_mels]


def pad_or_truncate_mel(
    mel: torch.Tensor,
    max_frames: int,
) -> torch.Tensor:
    """
    Pad (with zeros) or truncate a mel tensor to exactly ``max_frames`` frames.

    Args:
        mel:        [T_frames, F]
        max_frames: Target number of frames.

    Returns:
        mel padded/truncated to [max_frames, F].
    """
    T, F = mel.shape
    if T >= max_frames:
        return mel[:max_frames]
    pad = torch.zeros(max_frames - T, F, dtype=mel.dtype)
    return torch.cat([mel, pad], dim=0)  # [max_frames, F]
