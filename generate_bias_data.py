"""
generate_bias_data.py
─────────────────────
为 dev 集生成 per-sample 的 bias list 和包含 bias_entities 的新 manifest。

Usage:
    python generate_bias_data.py 
        --train_manifest data/contrastive-learning/libri-960.jsonl
        --dev_manifest data/contrastive-learning/libri-960-dev.jsonl 
        --output_dir data/contrastive-learning/per_sample_bias 
        --output_manifest data/contrastive-learning/libri-960-dev-bias.jsonl 
        --top_k_exclude 5000 
        --bias_list_size 100 
        --seed 42
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import List, Tuple

from tqdm import tqdm

# Ensure project root is on the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def find_trans_files(root_dir: str) -> List[str]:
    """递归查找所有 .trans.txt 文件"""
    trans_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname.endswith(".trans.txt"):
                trans_files.append(os.path.join(dirpath, fname))
    return trans_files


def load_texts_from_trans_files(trans_files: List[str]) -> List[str]:
    """从 .trans.txt 文件中读取每一行的转录文本"""
    texts = []
    for file_path in tqdm(trans_files, desc="Loading transcripts"):
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(maxsplit=1)
                if len(parts) == 2:
                    texts.append(parts[1])
    return texts


def load_texts_from_manifest(manifest_path: str) -> List[str]:
    """从 JSONL manifest 中读取所有 text 字段"""
    texts = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Loading manifest"):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            text = item.get("text", "")
            if text:
                texts.append(text)
    return texts


def preprocess_text(text: str) -> List[str]:
    """文本清洗：小写、去除非字母字符、分词"""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return [w for w in text.split() if w]


def compute_word_frequencies(texts: List[str]) -> Counter:
    """统计词频"""
    freq = Counter()
    for sent in tqdm(texts, desc="Counting word frequencies"):
        words = preprocess_text(sent)
        freq.update(words)
    return freq


def get_biased_words(
    sentence_words: List[str],
    high_freq_set: set,
    available_pool: List[str],
    # max_gt: int = 5,
) -> Tuple[List[str], List[str]]:
    """
    从句子中提取命中 available_pool 且不在高频词中的词作为 gtbias，
    同时返回剩余的候选池（用于填充干扰词）
    """
    candidates = [w for w in set(sentence_words) if w in available_pool and w not in high_freq_set]
    #gtbias = candidates[:max_gt]
    gtbias = candidates
    remaining = [w for w in available_pool if w not in gtbias]
    return gtbias, remaining


def construct_bias_list(
    sentence: str,
    high_freq_set: set,
    available_pool: List[str],
    total_size: int = 100,
    #max_gt_per_sentence: int = 5,
) -> Tuple[List[str], List[str]]:
    """
    为单句话构造 bias list，长度为 total_size，其中最多 max_gt_per_sentence 个真实命中词，
    其余从 available_pool 中随机抽取（可重复，若不足则用 <unk> 填充）。

    Returns:
        (bias_list, gtbias)  —  gtbias 是真实出现在句子中的 bias word
    """
    words = preprocess_text(sentence)
    #gtbias, remaining = get_biased_words(words, high_freq_set, available_pool, max_gt_per_sentence)
    gtbias, remaining = get_biased_words(words, high_freq_set, available_pool)
    bias_list = gtbias.copy()
    need = total_size - len(bias_list)
    if need > 0 and remaining:
        if need > len(remaining):
            additions = random.choices(remaining, k=need)
        else:
            additions = random.sample(remaining, need)
        bias_list.extend(additions)
    while len(bias_list) < total_size:
        bias_list.append("<unk>")
    return bias_list[:total_size], gtbias


# ═══════════════════════════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════════════════════════


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate per-sample bias lists and manifest")
    p.add_argument("--train_manifest", default="data/contrastive-learning/libri-960.jsonl",
                   help="Training set JSONL manifest (preferred). If provided, --train_root is ignored.")
    p.add_argument("--dev_manifest", default="data/contrastive-learning/libri-960-dev.jsonl")
    p.add_argument("--output_dir", default="data/contrastive-learning/per_sample_bias")
    p.add_argument("--output_manifest", default="data/contrastive-learning/libri-960-dev-bias.jsonl")
    p.add_argument("--top_k_exclude", type=int, default=5000, help="Exclude top-K high-frequency words")
    p.add_argument("--bias_list_size", type=int, default=100, help="Total size of per-sample bias list")
    #p.add_argument("--max_gt", type=int, default=5, help="Max ground-truth bias words per sentence")
    p.add_argument("--output_global_bias_list", default=None,
                   help="Path to write the global bias list (all available_pool words, one per line). "
                        "If omitted, defaults to {output_dir}/../bias-list.txt")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    # ── 1. 加载训练集转录并统计词频 ──────────────────────────────────────────
    print("[1/4] Loading training transcripts...")
    train_texts = load_texts_from_manifest(args.train_manifest)
    print(f"      Loaded {len(train_texts)} training sentences from manifest: {args.train_manifest}")

    print("[2/4] Computing word frequencies on training set...")
    word_freq = compute_word_frequencies(train_texts)
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    high_freq_set = {w for w, _ in sorted_words[: args.top_k_exclude]}
    available_pool = [w for w, _ in sorted_words[args.top_k_exclude :]]
    print(f"      Total unique words: {len(word_freq)}")
    print(
        f"      Removed top {args.top_k_exclude} high-frequency words, "
        f"left {len(available_pool)} candidate words for bias."
    )

    # ── 保存全局 bias list ───────────────────────────────────────────────────
    global_bias_path = args.output_global_bias_list
    if global_bias_path is None:
        global_bias_path = str(Path(args.output_dir).parent / "bias-list.txt")
    with open(global_bias_path, "w", encoding="utf-8") as f:
        for w in available_pool:
            f.write(w + "\n")
    print(f"      Global bias list saved to: {global_bias_path} ({len(available_pool)} words)")

    # ── 2. 读取 dev manifest ─────────────────────────────────────────────────
    print("[3/4] Loading dev manifest...")
    with open(args.dev_manifest, "r", encoding="utf-8") as f:
        dev_samples = [json.loads(line.strip()) for line in f if line.strip()]
    print(f"      Loaded {len(dev_samples)} dev samples.")

    # ── 3. 为每个 dev 样本生成 bias list ─────────────────────────────────────
    print("[4/4] Generating per-sample bias lists...")
    os.makedirs(args.output_dir, exist_ok=True)

    new_samples = []
    for idx, item in enumerate(tqdm(dev_samples, desc="Constructing bias lists")):
        text = item.get("text", "")
        bias_list, gtbias = construct_bias_list(
            text,
            high_freq_set,
            available_pool,
            total_size=args.bias_list_size,
           # max_gt_per_sentence=args.max_gt,
        )

        # 保存 per-sample bias list 文件
        # 使用音频文件名（不含扩展名）作为标识，保证唯一性
        audio_path = item.get("audio_path", "")
        audio_stem = Path(audio_path).stem
        bias_filename = f"{audio_stem}.txt"
        bias_filepath = os.path.join(args.output_dir, bias_filename)
        with open(bias_filepath, "w", encoding="utf-8") as f:
            for w in bias_list:
                f.write(w + "\n")

        # 构建新的 manifest 条目
        new_item = dict(item)  # 浅拷贝保留原有字段
        new_item["bias_entities"] = gtbias
        new_item["bias_list_file"] = bias_filename
        new_samples.append(new_item)

    # ── 4. 保存新的 manifest ─────────────────────────────────────────────────
    with open(args.output_manifest, "w", encoding="utf-8") as f:
        for item in new_samples:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\n✅ Done!")
    print(f"   Global bias list saved to: {global_bias_path}")
    print(f"   Per-sample bias lists saved to: {args.output_dir}")
    print(f"   New manifest saved to: {args.output_manifest}")

    # 简单统计
    total_gtbias = sum(len(item["bias_entities"]) for item in new_samples)
    avg_gtbias = total_gtbias / len(new_samples) if new_samples else 0
    print(f"   Average gtbias per sample: {avg_gtbias:.2f}")


if __name__ == "__main__":
    main()
