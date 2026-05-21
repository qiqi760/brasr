"""
eval_and_plot.py
────────────────
遍历指定目录下的所有 GLCLAP checkpoint，逐个调用 evaluate.py 进行 bias-word
retrieval 评估，最后绘制性能指标随 epoch 的变化曲线。

Usage:
    python eval_and_plot.py 
        --checkpoint_dir exp/20260516-093900-libri-960-d2v-large-bert-multi-proj512-bs16-freeze999-local-detach
        --task contrastive-learning 
        --dataset libri-dev-clean-bias 
        --model_config configs/model_config.yaml 
        --per_sample_bias_dir data/contrastive-learning/per_sample_bias_dev 
        --output_dir results/eval_and_plot 
        --threshold 0.3 
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch evaluate checkpoints and plot")
    p.add_argument("--checkpoint_dir", required=True, help="Directory containing .pt checkpoints")
    p.add_argument("--task", required=True, help="Task name, e.g. contrastive-learning")
    p.add_argument("--dataset", required=True, help="Dataset name (manifest stem), e.g. libri-dev-clean-bias")
    p.add_argument("--model_config", default="configs/model_config.yaml")
    p.add_argument("--per_sample_bias_dir", default=None,
                   help="Directory containing per-sample bias list files")
    p.add_argument("--bias_list", default=None, help="Global bias list file (legacy mode)")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--top_k", type=int, default=10)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output_dir", default="results/eval_and_plot")
    p.add_argument("--save_format", default="png", choices=["png", "pdf", "svg"])
    return p.parse_args()


def parse_checkpoint_info(checkpoint_path: str) -> tuple[int, str]:
    """
    从文件名解析 epoch 标注标签。
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
) -> dict[str, float]:
    """
    调用 evaluate.py 对单个 checkpoint 进行评估，返回结果字典。
    """
    project_root = Path(__file__).parent.resolve()
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

    print(f"\nRunning: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"ERROR evaluating {checkpoint_path}:\n{result.stderr}")
        return {}

    metrics = {}
    for line in result.stdout.splitlines():
        for key in ["top1_recall", "precision", "recall", "f1"]:
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
    横坐标为轮次标签，纵坐标为指标值。
    """
    if not results:
        print("No results to plot.")
        return

    results = sorted(results, key=lambda x: (x["epoch"] == -1, x["epoch"]))
    labels = [r.get("epoch_label", "") for r in results]
    xs = np.arange(len(labels))
    top1_recalls = [r.get("top1_recall", 0.0) * 100 for r in results]
    precisions = [r.get("precision", 0.0) * 100 for r in results]
    recalls = [r.get("recall", 0.0) * 100 for r in results]
    f1s = [r.get("f1", 0.0) * 100 for r in results]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("GLCLAP Bias-Word Retrieval Performance across Checkpoints", fontsize=14, fontweight="bold")

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

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(xs, top1_recalls, marker="o", label="Top-1 Recall", linewidth=2)
    ax.plot(xs, precisions, marker="s", label="Precision", linewidth=2)
    ax.plot(xs, recalls, marker="^", label="Recall", linewidth=2)
    ax.plot(xs, f1s, marker="d", label="F1", linewidth=2)
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


def load_cached_results(output_dir: str) -> dict[str, dict]:
    """
    加载之前保存的 eval_results.json，返回以 checkpoint 绝对路径为 key 的结果字典。
    """
    json_path = os.path.join(output_dir, "eval_results.json")
    if not os.path.exists(json_path):
        return {}
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        cache = {}
        for item in data:
            ckpt_path = item.get("checkpoint", "")
            if ckpt_path:
                cache[str(Path(ckpt_path).resolve())] = item
        print(f"📂 加载缓存结果: {len(cache)} 个 checkpoint 已评估过（来自 {json_path}）")
        return cache
    except (json.JSONDecodeError, OSError) as e:
        print(f"⚠️ 读取缓存文件失败: {e}，将重新评估所有模型。")
        return {}


def main() -> None:
    args = parse_args()

    if args.per_sample_bias_dir is None and args.bias_list is None:
        raise ValueError("Either --per_sample_bias_dir or --bias_list must be provided.")

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
        print(f"  {ckpt.name}")

    # 加载缓存
    os.makedirs(args.output_dir, exist_ok=True)
    cache = load_cached_results(args.output_dir)

    all_results: list[dict] = []
    new_eval_count = 0
    cached_count = 0

    for ckpt in checkpoints:
        ckpt_abs = str(ckpt.resolve())
        epoch, label = parse_checkpoint_info(str(ckpt))

        if ckpt_abs in cache:
            # 复用缓存结果
            result = cache[ckpt_abs]
            # 确保 epoch/epoch_label 与当前 checkpoint 一致（防止路径相同但内容变化，这里信任缓存）
            result["epoch"] = epoch
            result["epoch_label"] = label
            all_results.append(result)
            cached_count += 1
            print(f"  ✓ Epoch {epoch}: 使用缓存结果  "
                  f"top1_recall={result.get('top1_recall', 0)*100:.2f}%, "
                  f"precision={result.get('precision', 0)*100:.2f}%, "
                  f"recall={result.get('recall', 0)*100:.2f}%, "
                  f"f1={result.get('f1', 0)*100:.2f}%")
            continue

        # 新模型，需要评估
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
        )
        if metrics:
            result = {
                "checkpoint": str(ckpt),
                "epoch": epoch,
                "epoch_label": label,
                **metrics,
            }
            all_results.append(result)
            new_eval_count += 1
            print(f"  → Epoch {epoch}: 新评估  "
                  f"top1_recall={metrics.get('top1_recall', 0)*100:.2f}%, "
                  f"precision={metrics.get('precision', 0)*100:.2f}%, "
                  f"recall={metrics.get('recall', 0)*100:.2f}%, "
                  f"f1={metrics.get('f1', 0)*100:.2f}%")

    print(f"\n📊 统计: {cached_count} 个从缓存复用, {new_eval_count} 个新评估")

    # 按 epoch 排序
    all_results = sorted(all_results, key=lambda x: (x["epoch"] == -1, x["epoch"]))

    json_path = os.path.join(args.output_dir, "eval_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n📄 Raw results saved to: {json_path}")

    plot_base = os.path.join(args.output_dir, "performance_curve")
    plot_results(all_results, plot_base, args.save_format)

    print("\n✅ Evaluation and plotting completed!")


if __name__ == "__main__":
    main()
