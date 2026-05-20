"""
batch_evaluate_and_plot.py
──────────────────────────
遍历指定目录下的所有 GLCLAP checkpoint，逐个调用 evaluate.py 进行 bias-word
retrieval 评估，最后绘制性能指标随 epoch 的变化曲线。

Usage:
    python python_scripts/batch_evaluate_and_plot.py 
        --checkpoint_dir  exp/20260506-162014-libri-960-d2v-large-bert-multi-proj512-bs16-freeze999-local-detach
        --task contrastive-learning 
        --dataset libri-960-dev-bias-new 
        --model_config configs/model_config.yaml 
        --per_sample_bias_dir data/contrastive-learning/per_sample_bias_960 
        --output_dir results/batch_eval 
        --threshold 0.5 
        --top_k 10
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

# Ensure project root is on the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch evaluate checkpoints and plot")
    p.add_argument("--checkpoint_dir", required=True, help="Directory containing .pt checkpoints")
    p.add_argument("--task", required=True, help="Task name, e.g. contrastive-learning")
    p.add_argument("--dataset", required=True, help="Dataset name (manifest stem), e.g. libri-960-dev-bias")
    p.add_argument("--model_config", default="configs/model_config.yaml")
    # [新增] per_sample_bias_dir 参数，与 evaluate.py 保持一致
    p.add_argument("--per_sample_bias_dir", default=None,
                   help="Directory containing per-sample bias list files")
    # 兼容旧的全局 bias_list 模式
    p.add_argument("--bias_list", default=None, help="Global bias list file (legacy mode)")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--top_k", type=int, default=10)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output_dir", default="results/batch_eval")
    p.add_argument("--save_format", default="png", choices=["png", "pdf", "svg"])
    # [新增] 支持输出检索结果到指定目录
    p.add_argument("--output_retrieval_dir", default=None,
                   help="Directory to save per-sample retrieved bias words. "
                        "Passed through to evaluate.py.")
    # [新增] 可选的 ASR hypothesis 文件，用于计算 B-WER
    p.add_argument("--asr_hypotheses_file", default=None,
                   help="Path to ASR hypotheses (one per line). "
                        "Passed through to evaluate.py for B-WER computation.")
    return p.parse_args()


def parse_checkpoint_info(checkpoint_path: str) -> tuple[int, str]:
    """
    从文件名解析 epoch标注标签。
    期望格式: model_epoch{epoch}.pt
    若包含 best，则标签为 "best"。
    若解析失败返回 (-1, "").
    """
    fname = os.path.basename(checkpoint_path)
    match = re.search(r"epoch(\d+)", fname)
    if match:
        epoch = int(match.group(1))
        label = f"{epoch:03d}"[-3:]
        return epoch, label
    if re.search(r"best", fname, re.IGNORECASE):
        return -1, "best"
    return -1, ""


def run_evaluate(
    checkpoint_path: str,
    task: str,
    dataset: str,
    model_config: str,
    per_sample_bias_dir: str | None,
    bias_list: str | None,
    threshold: float,
    top_k: int,
    device: str,
    output_retrieval_dir: str | None = None,
    asr_hypotheses_file: str | None = None,
) -> dict[str, float]:
    """
    调用 evaluate.py 对单个 checkpoint 进行评估，返回结果字典。
    """
    project_root = Path(__file__).parent.parent.resolve()
    eval_script = project_root / "python_scripts" / "evaluate.py"

    cmd = [
        sys.executable,
        str(eval_script),
        "--task", task,
        "--dataset", dataset,
        "--checkpoint", checkpoint_path,
        "--model_config", model_config,
        "--threshold", str(threshold),
        "--top_k", str(top_k),
        "--device", device,
    ]

    if per_sample_bias_dir is not None:
        cmd.extend(["--per_sample_bias_dir", per_sample_bias_dir])
    elif bias_list is not None:
        cmd.extend(["--bias_list", bias_list])
    else:
        raise ValueError("Either --per_sample_bias_dir or --bias_list must be provided.")

    if output_retrieval_dir is not None:
        cmd.extend(["--output_retrieval_dir", output_retrieval_dir])
    if asr_hypotheses_file is not None:
        cmd.extend(["--asr_hypotheses_file", asr_hypotheses_file])

    print(f"\nRunning: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"ERROR evaluating {checkpoint_path}:\n{result.stderr}")
        return {}

    # 从 stdout 中解析结果
    # evaluate.py 的输出格式:
    #   top1_recall    : 12.34%
    #   precision      : 56.78%
    #   ...
    metrics = {}
    for line in result.stdout.splitlines():
        for key in ["top1_recall", "precision", "recall", "f1", "bwer"]:
            if line.strip().startswith(key):
                try:
                    val_str = line.split(":")[-1].strip().replace("%", "")
                    metrics[key] = float(val_str) / 100.0
                except ValueError:
                    pass
    return metrics


def plot_results(
    results: list[dict],
    output_path: str,
    save_format: str,
) -> None:
    """
    绘制评估指标随 checkpoint 的变化曲线。
    横坐标为轮次标签（文件名中 epoch 后的三位数字），纵坐标为指标值。
    results: 列表，每个元素为 {
        "epoch": int,
        "epoch_label": str,
        "top1_recall": float,
        "precision": float,
        "recall": float,
        "f1": float,
    }
    """
    if not results:
        print("No results to plot.")
        return

    # 按 epoch 排序（best 放在最后）
    results = sorted(results, key=lambda x: (x["epoch"] == -1, x["epoch"]))
    labels = [r.get("epoch_label", "") for r in results]
    xs = np.arange(len(labels))
    top1_recalls = [r.get("top1_recall", 0.0) * 100 for r in results]
    precisions = [r.get("precision", 0.0) * 100 for r in results]
    recalls = [r.get("recall", 0.0) * 100 for r in results]
    f1s = [r.get("f1", 0.0) * 100 for r in results]
    has_bwer = any("bwer" in r for r in results)
    bwerrs = [r.get("bwer", 0.0) * 100 for r in results] if has_bwer else None

    # 如果包含 B-WER，使用 2x3 布局；否则保持 2x2
    if has_bwer:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle("GLCLAP Bias-Word Retrieval & ASR Performance across Checkpoints",
                     fontsize=14, fontweight="bold")
        plots = [
            (axes[0, 0], "Top-1 Recall", top1_recalls, "#1f77b4"),
            (axes[0, 1], "Precision", precisions, "#ff7f0e"),
            (axes[0, 2], "Recall", recalls, "#2ca02c"),
            (axes[1, 0], "F1 Score", f1s, "#d62728"),
            (axes[1, 1], "B-WER", bwerrs, "#9467bd"),
        ]
        # 隐藏最后一个空子图
        axes[1, 2].axis("off")
    else:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle("GLCLAP Bias-Word Retrieval Performance across Checkpoints",
                     fontsize=14, fontweight="bold")
        plots = [
            (axes[0, 0], "Top-1 Recall", top1_recalls, "#1f77b4"),
            (axes[0, 1], "Precision", precisions, "#ff7f0e"),
            (axes[1, 0], "Recall", recalls, "#2ca02c"),
            (axes[1, 1], "F1 Score", f1s, "#d62728"),
        ]

    for ax, title, values, color in plots:
        ax.plot(xs, values, marker="o", linestyle="-", color=color, linewidth=2, markersize=6)
        ax.set_xlabel("Checkpoint Epoch", fontsize=11)
        ax.set_ylabel(f"{title} (%)", fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, rotation=45, ha="right")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = f"{output_path}.{save_format}"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"\n📊 Plot saved to: {save_path}")
    plt.close()

    # 同时保存为单张综合对比图
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(xs, top1_recalls, marker="o", label="Top-1 Recall", linewidth=2)
    ax.plot(xs, precisions, marker="s", label="Precision", linewidth=2)
    ax.plot(xs, recalls, marker="^", label="Recall", linewidth=2)
    ax.plot(xs, f1s, marker="d", label="F1", linewidth=2)
    if has_bwer:
        ax.plot(xs, bwerrs, marker="v", label="B-WER", linewidth=2, color="#9467bd")
    ax.set_xlabel("Checkpoint Epoch", fontsize=12)
    ax.set_ylabel("Score (%)", fontsize=12)
    ax.set_title("All Metrics Comparison", fontsize=13, fontweight="bold")
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=45, ha="right")

    combined_path = f"{output_path}_combined.{save_format}"
    plt.savefig(combined_path, dpi=300, bbox_inches="tight")
    print(f"📊 Combined plot saved to: {combined_path}")
    plt.close()


def main() -> None:
    args = parse_args()

    if args.per_sample_bias_dir is None and args.bias_list is None:
        raise ValueError("Either --per_sample_bias_dir or --bias_list must be provided.")

    # 发现所有 checkpoint
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoints = sorted(
        [p for p in checkpoint_dir.glob("*.pt") if p.is_file()],
        key=lambda p: parse_checkpoint_info(str(p))[0],
    )
    if not checkpoints:
        print(f"No .pt checkpoints found in {checkpoint_dir}")
        return

    print(f"Found {len(checkpoints)} checkpoint(s) in {checkpoint_dir}:")
    for ckpt in checkpoints:
        print(f" {ckpt.name}")

    # 逐个评估
    all_results: list[dict] = []
    for ckpt in checkpoints:
        epoch, label = parse_checkpoint_info(str(ckpt))
        metrics = run_evaluate(
            checkpoint_path=str(ckpt),
            task=args.task,
            dataset=args.dataset,
            model_config=args.model_config,
            per_sample_bias_dir=args.per_sample_bias_dir,
            bias_list=args.bias_list,
            threshold=args.threshold,
            top_k=args.top_k,
            device=args.device,
            output_retrieval_dir=args.output_retrieval_dir,
            asr_hypotheses_file=args.asr_hypotheses_file,
        )
        if metrics:
            result = {
                "checkpoint": str(ckpt),
                "epoch": epoch,
                "epoch_label": label,
                **metrics,
            }
            all_results.append(result)
            msg = (f"  → Epoch {epoch}: top1_recall={metrics.get('top1_recall', 0)*100:.2f}%, "
                   f"precision={metrics.get('precision', 0)*100:.2f}%, "
                   f"recall={metrics.get('recall', 0)*100:.2f}%, "
                   f"f1={metrics.get('f1', 0)*100:.2f}%")
            if "bwer" in metrics:
                msg += f", bwer={metrics['bwer']*100:.2f}%"
            print(msg)

    # 保存原始结果为 JSON
    os.makedirs(args.output_dir, exist_ok=True)
    json_path = os.path.join(args.output_dir, "batch_eval_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n📄 Raw results saved to: {json_path}")

    # 绘制图像
    plot_base = os.path.join(args.output_dir, "performance_curve")
    plot_results(all_results, plot_base, args.save_format)

    print("\n✅ Batch evaluation and plotting completed!")


if __name__ == "__main__":
    main()
