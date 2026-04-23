"""
subtext.py
──────────
Subtext (local text) sampling for GLCLAP training.

Paper (Section 2.1):
    "We randomly extract sub-text from the original text annotations.
     For instance, if the original text transcription is
     'Have you ever heard Taylor Swift's songs',
     a randomly selected subtext could be 'Taylor Swift'."

Assumption (not fully specified in paper):
    A subtext is a contiguous word-span of length ∈ [min_words, max_words]
    sampled uniformly from the full word list.
    If the sentence is shorter than min_words, the full sentence is returned.
"""

from __future__ import annotations

import random
import re
from typing import Optional


def _split_words(text: str) -> list[str]:
    """Split text into words. Falls back to character-level for CJK text."""
    # CJK Unified Ideographs range: U+4E00–U+9FFF
    if re.search(r"[\u4e00-\u9fff]", text):
        return list(text)
    return text.split()


def sample_subtext(
    text: str,
    min_words: int = 1,
    max_words: int = 5,
    rng: Optional[random.Random] = None,
) -> str:
    """
    Sample a contiguous sub-span of words from ``text``.

    Args:
        text:       Full transcription string.
        min_words:  Minimum number of words in the sampled span.
        max_words:  Maximum number of words in the sampled span.
        rng:        Optional ``random.Random`` instance for reproducibility.

    Returns:
        Subtext string (subset of words from ``text``).

    Example:
        >>> sample_subtext("Have you ever heard Taylor Swift's songs", min_words=1, max_words=3)
        "Taylor Swift's"
    """
    if rng is None:
        rng = random

    words = _split_words(text)
    n = len(words)

    if n <= min_words:
        # Edge case: sentence too short → return full text as subtext
        return text

    span_len = rng.randint(min_words, min(max_words, n))
    start = rng.randint(0, n - span_len)
    return " ".join(words[start : start + span_len])


def sample_subtext_batch(
    texts: list[str],
    min_words: int = 1,
    max_words: int = 5,
    seed: Optional[int] = None,
) -> list[str]:
    """
    Vectorised wrapper: sample one subtext per element in ``texts``.

    Args:
        texts:     List of B full transcription strings.
        min_words: Minimum span length.
        max_words: Maximum span length.
        seed:      Optional seed for reproducibility.

    Returns:
        List of B subtext strings, shapes unchanged in terms of list length.
    """
    rng = random.Random(seed)
    return [sample_subtext(t, min_words, max_words, rng) for t in texts]
