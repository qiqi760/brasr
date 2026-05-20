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

MODIFIED (2026-05-13):
    新增全局约束 subtext 预生成函数 generate_constrained_subtexts()。
    由于 batch_size 受限导致 in-batch 负例不足，本修改在 Dataset 初始化阶段
    为每条文本一次性预生成最多 10 个满足全局约束的 subtexts：
        1. 同一条文本的 subtexts 之间词不重复（区间互不重叠）。
        2. 全局正例隔离：一个词只能作为一条文本的正例，绝不可出现在
           其他文本的 subtext 中（避免负例污染）。
    原代码（sample_subtext / sample_subtext_batch）完整保留并注释，
    供随机单条采样回退使用。
"""

from __future__ import annotations

import random
import re
from collections import defaultdict
from typing import Optional


def _split_words(text: str) -> list[str]:
    """Split text into words. Falls back to character-level for CJK text."""
    # CJK Unified Ideographs range: U+4E00–U+9FFF
    if re.search(r"[\u4e00-\u9fff]", text):
        return list(text)
    return text.split()


# ── 原代码：sample_subtext（已注释保留）────────────────────────────
# def sample_subtext(
#     text: str,
#     min_words: int = 1,
#     max_words: int = 5,
#     rng: Optional[random.Random] = None,
# ) -> str:
#     """
#     Sample a contiguous sub-span of words from ``text``.
#     """
#     if rng is None:
#         rng = random
#
#     words = _split_words(text)
#     n = len(words)
#
#     if n <= min_words:
#         return text
#
#     span_len = rng.randint(min_words, min(max_words, n))
#     start = rng.randint(0, n - span_len)
#     return " ".join(words[start : start + span_len])
# ──────────────────────────────────────────────────────────────────


# ── 修改处：保留原函数并直接扩展 ──────────────────────────────────
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


# ── 修改处：新增全局约束 subtext 预生成 ────────────────────────────
def generate_constrained_subtexts(
    texts: list[str],
    num_subtexts_per_text: int = 20,
    min_words: int = 1,
    max_words: int = 5,
    seed: Optional[int] = None,
) -> list[list[str]]:
    """
    为整个语料库中的每条文本预生成满足全局约束的 subtexts。

    约束（用户指定）：
        1. 无重复词：同一条文本生成的多个 subtexts 之间词集合互不相交
           （通过不重叠的连续区间实现）。
        2. 全局正例隔离：只要某词在某条文本中作为正例出现（被分配到该
           文本的 subtext 中），就绝不可出现在其他任何文本的 subtext 中。
           即每个词全局仅属于一条文本。

    实现步骤：
        Step 1 — 全局词归属分配：
            遍历全部文本建立倒排索引 word→{text_indices}。
            按词的出现频次升序处理；对于多文本共享词，贪心分配给当前
            已分配词数最少的那条文本。确保每个词仅归属一条文本。

        Step 2 — 单文本不重叠 subtext 抽取：
            对每条文本，按原文本顺序过滤出只属于该文本的词，得到
            filtered_words。在 filtered_words 上随机抽取互不重叠的
            连续子片段（长度∈[min_words, max_words]），最多抽
            num_subtexts_per_text 个。不足则有多少抽多少。

        Step 3 — 回退：
            若某文本完全无法抽出 subtext（极短或词全被分配走），
            直接以原句作为唯一 subtext，保证训练不中断。

    Args:
        texts:                 全部文本列表（长度 = 数据集大小）。
        num_subtexts_per_text: 每条文本目标抽取的 subtext 数量（默认 10）。
        min_words:             每个 subtext 的最短词数。
        max_words:             每个 subtext 的最长词数。
        seed:                  随机种子，保证可复现。

    Returns:
        list[list[str]]：外层长度 = len(texts)，内层长度 ≤ num_subtexts_per_text。
                         每个内层列表包含该文本满足约束的 subtext 字符串。
    """
    rng = random.Random(seed)
    all_words = [_split_words(t) for t in texts]

    # ── Step 1：全局词归属分配 ──────────────────────────────
    # 建立 word → 出现过的文本索引集合 的倒排索引
    word_to_indices: dict[str, set[int]] = defaultdict(set)
    for idx, words in enumerate(all_words):
        seen: set[str] = set()
        for w in words:
            if w not in seen:
                word_to_indices[w].add(idx)
                seen.add(w)

    # 每个词只能归属一条文本。贪心策略：
    #   - 按出现频次从少到多处理（减少高共享词的冲突）
    #   - 对共享词，分配给当前已分配词数最少的那条文本
    assigned_words_per_text: list[set[str]] = [set() for _ in range(len(texts))]

    # 先处理只出现一次的词（无冲突，直接分配）
    multi_occurrence: list[tuple[str, list[int]]] = []
    for word, indices in word_to_indices.items():
        if len(indices) == 1:
            idx = next(iter(indices))
            assigned_words_per_text[idx].add(word)
        else:
            multi_occurrence.append((word, sorted(indices)))

    # 再处理多文本共享词（按出现频次升序）
    multi_occurrence.sort(key=lambda x: len(x[1]))
    for word, indices in multi_occurrence:
        # 选择当前已分配词数最少、且包含该词的文本
        best_idx = min(indices, key=lambda i: (len(assigned_words_per_text[i]), i))
        assigned_words_per_text[best_idx].add(word)

    # ── Step 2：为每条文本生成不重叠 subtexts ───────────────
    result: list[list[str]] = []
    for idx, words in enumerate(all_words):
        assigned = assigned_words_per_text[idx]

        # 按原文本顺序过滤出只属于当前文本的词
        # 修改处：在单条文本内也去重，确保同一词不会出现在多个 subtext 中
        seen_filtered: set[str] = set()
        filtered: list[str] = []
        for w in words:
            if w in assigned and w not in seen_filtered:
                filtered.append(w)
                seen_filtered.add(w)
        n = len(filtered)

        subtexts: list[str] = []

        if n < min_words:
            # 回退：文本太短，直接返回原句
            subtexts.append(texts[idx])
            result.append(subtexts)
            continue

        used_positions: set[int] = set()
        max_attempts = num_subtexts_per_text * 100
        attempts = 0

        while len(subtexts) < num_subtexts_per_text and attempts < max_attempts:
            attempts += 1

            span_len = rng.randint(min_words, min(max_words, n))
            if n < span_len:
                break

            start = rng.randint(0, n - span_len)
            end = start + span_len

            # 检查是否与已抽取区间重叠
            overlap = any(p in used_positions for p in range(start, end))
            if overlap:
                continue

            subtext = " ".join(filtered[start:end])
            # 额外检查：subtext 内部不应出现重复词（如 "A A"）
            if len(set(filtered[start:end])) != len(filtered[start:end]):
                continue

            subtexts.append(subtext)
            for p in range(start, end):
                used_positions.add(p)

        # 若完全没抽到，回退到原句
        if not subtexts:
            subtexts.append(texts[idx])

        result.append(subtexts)

    return result
# ──────────────────────────────────────────────────────────────────
