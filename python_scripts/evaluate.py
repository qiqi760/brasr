"""
evaluate.py
───────────
Evaluate a trained GLCLAP checkpoint for bias-word retrieval.

Usage (global bias list mode):
    python python_scripts/evaluate.py \
        --task contrastive-learning \
        --dataset libri-960 \
        --checkpoint outputs/glclap/best_model.pt \
        --bias_list data/bias_lists/libri_bias.txt \
        --model_config configs/model_config.yaml \
        --threshold 0.5

Usage (per-sample bias list mode — [新增]):
    python python_scripts/evaluate.py \
        --task contrastive-learning \
        --dataset libri-960-dev-bias \
        --checkpoint outputs/glclap/best_model.pt \
        --per_sample_bias_dir data/contrastive-learning/per_sample_bias \
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
from eval.metrics import compute_bwer, evaluate_retrieval
from inference.retriever import BiasWordRetriever
from models.glclap import GLCLAP
from utils.checkpointing import load_checkpoint
from utils.logging import setup_logging


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate GLCLAP bias-word retrieval")
    p.add_argument("--task", required=True, help="Task name, e.g. contrastive-learning")
    p.add_argument("--dataset", required=True, help="Dataset name, e.g. libri-960")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--bias_list", default=None, help="Plain text file: one bias phrase per line")
    # ── [新增] 支持 per-sample 的 bias list 目录 ────────────────────────────
    # 当提供 --per_sample_bias_dir 时，每个样本会从 manifest 中指定的
    # bias_list_file 加载独立的100词 bias list，用于模拟真实场景下的
    # contextual biasing（与 libri_process.py 的 construct_bias_list 逻辑对应）。
    p.add_argument("--per_sample_bias_dir", default=None,
                   help="Directory containing per-sample bias list files. "
                        "Manifest must contain 'bias_list_file' field.")
    # ── [新增结束] ──────────────────────────────────────────────────────────
    p.add_argument("--model_config", default="configs/model_config.yaml")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--top_k", type=int, default=10)
    p.add_argument("--device", default="cuda:3" if torch.cuda.is_available() else "cpu")
    # [新增] 输出检索结果到指定目录，格式与 per_sample_bias_960 一致
    p.add_argument("--output_retrieval_dir", default=None,
                   help="Directory to save per-sample retrieved bias words. "
                        "If provided, each sample will be written as a .txt file "
                        "with one word per line, matching the per_sample_bias format.")
    # [新增] 可选的 ASR hypothesis 文件，用于计算 B-WER
    p.add_argument("--asr_hypotheses_file", default=None,
                   help="Path to a plain text file with one ASR hypothesis per line "
                        "(ordered identically to the manifest). If provided, B-WER "
                        "(biased word error rate) will be computed alongside retrieval metrics.")

    args = p.parse_args()
    # ── [新增] 参数校验 ─────────────────────────────────────────────────────
    if args.bias_list is None and args.per_sample_bias_dir is None:
        p.error("Either --bias_list or --per_sample_bias_dir must be provided.")
    # ── [新增结束] ──────────────────────────────────────────────────────────
    return args


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
    # [原代码] 全局单 bias list 模式（保留，用于兼容原有调用方式）
    # with open(args.bias_list, encoding="utf-8") as f:
    #     bias_words = [line.strip() for line in f if line.strip()]
    # [原代码结束]
    #
    # [新增] 若使用全局 bias_list 模式，预先加载；per-sample 模式在循环中动态加载
    bias_words = []
    if args.bias_list is not None:
        with open(args.bias_list, encoding="utf-8") as f:
            bias_words = [line.strip() for line in f if line.strip()]
    # [新增结束]

    # ── Load model ────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(text_model_name)
    model = GLCLAP(
        text_model_name=text_model_name,
        audio_model_name=audio_model_name,
        embed_dim=model_cfg["projection"]["embed_dim"],
        text_freeze_layers=model_cfg["text_encoder"].get("freeze_layers", 0),
        audio_freeze_layers=model_cfg["audio_encoder"].get("freeze_layers", 0),
        detach_encoders=model_cfg.get("detach_encoders", False),
        text_use_attention_pooling=model_cfg["text_encoder"].get("use_attention_pooling", False),
        audio_use_attention_pooling=model_cfg["audio_encoder"].get("use_attention_pooling", False),
    )
    load_checkpoint(args.checkpoint, model, strict=False)

    retriever = BiasWordRetriever(
        model=model,
        tokenizer=tokenizer,
        threshold=args.threshold,
        top_k=args.top_k,
        device=args.device,
    )
    # [原代码] 全局单 bias list 模式（保留）
    # retriever.set_bias_list(bias_words)
    # [原代码结束]
    #
    # [新增] 仅在全局模式下预先设置 bias list；per-sample 模式在循环中动态设置
    if args.bias_list is not None:
        retriever.set_bias_list(bias_words)
    # [新增结束]

    # ── Load optional ASR hypotheses ──────────────────────────────────────
    asr_hypotheses: list[str] = []
    if args.asr_hypotheses_file is not None:
        with open(args.asr_hypotheses_file, encoding="utf-8") as hf:
            asr_hypotheses = [line.strip() for line in hf]

    # ── Run evaluation ────────────────────────────────────────────────────
    all_predictions: list[list[str]] = []
    all_ground_truths: list[list[str]] = []
    top1_ground_truths: list[str] = []
    skipped_empty_gt = 0  # [方案 A] 统计跳过的无 bias word 样本数

    # [新增] 收集 B-WER 所需的全部样本数据（不跳过空 ground-truth）
    all_references: list[str] = []
    all_bias_lists: list[list[str]] = []
    all_hypotheses: list[str] = []

    # [新增] 若指定输出目录，提前创建
    if args.output_retrieval_dir is not None:
        os.makedirs(args.output_retrieval_dir, exist_ok=True)

    with open(manifest_path, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            item = json.loads(line.strip())
            audio_path = item["audio_path"]
            gt_entities = item.get("bias_entities", [])

            # 显存不足时主动释放缓存
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()

            # ── [新增] per-sample bias list 支持 ──────────────────────────────
            if args.per_sample_bias_dir is not None:
                bias_list_file = item.get("bias_list_file", "")
                if not bias_list_file:
                    raise ValueError(
                        f"Manifest item missing 'bias_list_file' field: {item}"
                    )
                sample_bias_path = os.path.join(args.per_sample_bias_dir, bias_list_file)
                with open(sample_bias_path, encoding="utf-8") as bf:
                    sample_bias_words = [l.strip() for l in bf if l.strip()]
                retriever.set_bias_list(sample_bias_words)
            # ── [新增结束] ────────────────────────────────────────────────────

            waveform, _ = load_audio(audio_path, target_sr=16_000)
            waveform = waveform.squeeze(0)  # [T_samples]

            predicted = retriever.retrieve(waveform)

            # [新增] 输出检索结果到指定目录，格式与 per_sample_bias_960 一致
            if args.output_retrieval_dir is not None:
                out_filename = item.get("bias_list_file", "")
                if not out_filename:
                    # fallback: 从 audio_path 提取 basename 并改扩展名为 .txt
                    out_filename = os.path.splitext(os.path.basename(audio_path))[0] + ".txt"
                out_path = os.path.join(args.output_retrieval_dir, out_filename)
                with open(out_path, "w", encoding="utf-8") as out_f:
                    for w in predicted:
                        out_f.write(w + "\n")
            # [新增结束]

            """
            # [原代码] 保留所有样本，包括 bias_entities 为空的样本
            all_predictions.append(predicted)
            all_ground_truths.append(gt_entities)
            top1_ground_truths.append(gt_entities[0] if gt_entities else "")

            """ 
            # [新增] 收集 B-WER 数据（所有样本，含空 ground-truth）
            all_references.append(item.get("text", ""))
            if asr_hypotheses:
                if idx >= len(asr_hypotheses):
                    raise ValueError(
                        f"--asr_hypotheses_file has fewer lines ({len(asr_hypotheses)}) "
                        f"than manifest samples (at least {idx + 1})"
                    )
                all_hypotheses.append(asr_hypotheses[idx])
            if args.per_sample_bias_dir is not None:
                all_bias_lists.append(sample_bias_words)
            else:
                all_bias_lists.append(bias_words)
            # [新增结束]

            # [方案 A] 跳过不含任何 bias word 的样本，避免空 ground_truth
            # 导致 top1_recall 被恒定为 0、recall 被虚高等指标偏差问题。
            if not gt_entities:
                skipped_empty_gt += 1
                continue

            all_predictions.append(predicted)
            all_ground_truths.append(gt_entities)
            top1_ground_truths.append(gt_entities[0])

    # ── Validate ASR hypothesis count ──────────────────────────────────────
    if asr_hypotheses and len(asr_hypotheses) != len(all_references):
        raise ValueError(
            f"--asr_hypotheses_file has {len(asr_hypotheses)} lines, "
            f"but manifest has {len(all_references)} samples."
        )

    results = evaluate_retrieval(
        all_predictions=all_predictions,
        all_ground_truths=all_ground_truths,
        top1_ground_truths=top1_ground_truths,
    )

    # [新增] 计算 B-WER（如果提供了 ASR hypotheses）
    if all_hypotheses:
        bwer = compute_bwer(
            hypotheses=all_hypotheses,
            references=all_references,
            bias_lists=all_bias_lists,
        )
        results["bwer"] = bwer

    print("\n── Evaluation Results ──────────────────")
    if skipped_empty_gt > 0:
        print(f"  (Skipped {skipped_empty_gt} samples with empty bias_entities)")
    for k, v in results.items():
        print(f"  {k:<16}: {v * 100:.2f}%")
    print("────────────────────────────────────────\n")


if __name__ == "__main__":
    main()
