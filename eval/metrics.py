"""
metrics.py
──────────
Evaluation metrics for GLCLAP bias-word retrieval.

Paper (Section 3.2) — evaluation metrics:
    - Top-1 recall:  hit rate of the single best-matched bias word
    - F1 score:      harmonic mean of precision and recall over the retrieved set
    - WER:           Word Error Rate for the downstream ASR evaluation
                     (WER computation is left to an external ASR scoring tool,
                      e.g. jiwer; we provide a thin wrapper here)

Definitions used:
    For a single utterance with ground-truth bias entity gt ∈ {bias_list}:
        top1_recall = 1 if gt in retrieved_top1 else 0

    For set retrieval (F1):
        precision = |predicted ∩ ground_truth| / |predicted|
        recall    = |predicted ∩ ground_truth| / |ground_truth|
        F1        = 2 * P * R / (P + R)
"""

from __future__ import annotations

from typing import Optional


def top1_recall(
    predictions: list[str],
    ground_truth: str,
) -> int:
    """
    Top-1 recall for a single utterance.

    Args:
        predictions:  Ordered list of retrieved bias words (best first).
        ground_truth: The single correct bias entity for this utterance.

    Returns:
        1 if ground_truth is the top-1 prediction, 0 otherwise.

    Note:
        Paper Table 1 evaluates "top-1 recall", meaning the correct entity
        must be the highest-ranked retrieved word.
        TODO: Clarify whether it means "appears anywhere in the retrieved list"
              (recall@K) or strictly rank-1. We assume rank-1 here.
    """
    if not predictions:
        return 0
    return int(predictions[0] == ground_truth)


def precision_recall_f1(
    predicted_set: list[str],
    ground_truth_set: list[str],
) -> tuple[float, float, float]:
    """
    Compute precision, recall, and F1 for a single utterance.

    Args:
        predicted_set:    List of retrieved bias words.
        ground_truth_set: List of ground-truth bias words for this utterance.

    Returns:
        (precision, recall, f1) — floats in [0, 1].
    """
    pred = set(predicted_set)
    gt   = set(ground_truth_set)

    if len(pred) == 0 and len(gt) == 0:
        return 1.0, 1.0, 1.0
    if len(pred) == 0:
        return 0.0, 0.0, 0.0
    if len(gt) == 0:
        return 0.0, 1.0, 0.0

    tp = len(pred & gt)
    precision = tp / len(pred)
    recall    = tp / len(gt)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return precision, recall, f1


def evaluate_retrieval(
    all_predictions: list[list[str]],
    all_ground_truths: list[list[str]],
    top1_ground_truths: Optional[list[str]] = None,
) -> dict[str, float]:
    """
    Aggregate evaluation over the full test set.

    Args:
        all_predictions:    List of B predicted-word lists (one per utterance).
        all_ground_truths:  List of B ground-truth-word lists.
        top1_ground_truths: Optional list of B single ground-truth words for
                            top-1 recall computation. If None, uses the first
                            element of each ground-truth list.

    Returns:
        dict with keys:
            "top1_recall":  float  — mean top-1 recall  (Table 1 metric)
            "precision":    float  — mean precision
            "recall":       float  — mean recall
            "f1":           float  — mean F1  (Table 2 metric)

    Example:
        >>> evaluate_retrieval(
        ...     [["Taylor Swift"], ["FIFA"]],
        ...     [["Taylor Swift"], ["FIFA"]],
        ... )
        {'top1_recall': 1.0, 'precision': 1.0, 'recall': 1.0, 'f1': 1.0}
    """
    assert len(all_predictions) == len(all_ground_truths), (
        "predictions and ground_truths must have equal length"
    )
    N = len(all_predictions)

    if top1_ground_truths is None:
        top1_ground_truths = [
            gt[0] if gt else "" for gt in all_ground_truths
        ]

    top1_hits = []
    precisions, recalls, f1s = [], [], []

    for pred, gt, gt1 in zip(all_predictions, all_ground_truths, top1_ground_truths):
        top1_hits.append(top1_recall(pred, gt1))
        p, r, f = precision_recall_f1(pred, gt)
        precisions.append(p)
        recalls.append(r)
        f1s.append(f)

    return {
        "top1_recall": sum(top1_hits) / N,
        "precision":   sum(precisions) / N,
        "recall":      sum(recalls) / N,
        "f1":          sum(f1s) / N,
    }


def compute_wer(hypotheses: list[str], references: list[str]) -> float:
    """
    Compute Word Error Rate using the `jiwer` library.

    Args:
        hypotheses: List of ASR hypothesis strings.
        references: List of reference transcription strings.

    Returns:
        WER as a float in [0, ∞) (values > 1 possible due to insertions).

    Requires:
        pip install jiwer
    """
    try:
        import jiwer
    except ImportError as e:
        raise ImportError("Install jiwer: pip install jiwer") from e

    return jiwer.wer(references, hypotheses)


def compute_bwer(
    hypotheses: list[str],
    references: list[str],
    bias_lists: list[list[str]],
) -> float:
    """
    Compute Biased Word Error Rate (B-WER).

    B-WER measures the word error rate restricted to words that appear
    in the bias list. It is a standard metric for contextual biasing ASR
    evaluation (e.g. Interspeech 2025 papers).

    For each utterance, we align reference and hypothesis at word level
    via ``jiwer``, then count substitution / deletion / insertion errors
    that involve words present in the bias list.

    Args:
        hypotheses:   ASR hypothesis strings (one per utterance).
        references:   Reference transcription strings (one per utterance).
        bias_lists:   List of bias word/phrase lists (one per utterance).
                      Each phrase is split into individual words for matching.

    Returns:
        B-WER as a float in [0, ∞).  Returns 0.0 when no bias words
        occur in any reference.

    Raises:
        ImportError: if ``jiwer`` is not installed.
        ValueError:  if input lists have mismatched lengths.

    Example:
        >>> compute_bwer(
        ...     ["taylor swifty is singing"],
        ...     ["taylor swift is singing"],
        ...     [["taylor swift"]],
        ... )
        0.5   # "swift" substituted → 1 error / 2 bias words
    """
    try:
        import jiwer
    except ImportError as e:
        raise ImportError("Install jiwer: pip install jiwer") from e

    if not (len(hypotheses) == len(references) == len(bias_lists)):
        raise ValueError(
            "hypotheses, references, and bias_lists must have equal length"
        )
 
    # Build per-utterance bias-word sets (split phrases into individual words)
    utterance_bias_sets: list[set[str]] = []
    for bias_list in bias_lists:
        words: set[str] = set()
        for phrase in bias_list:
            for w in str(phrase).split():
                words.add(w.lower())
        utterance_bias_sets.append(words)

    # Batch word-level alignment with standard ASR preprocessing
    output = jiwer.process_words(
        references,
        hypotheses,
        reference_transform=jiwer.wer_standardize,
        hypothesis_transform=jiwer.wer_standardize,
    )

    total_bias_words = 0
    total_sub = 0
    total_del = 0
    total_ins = 0

    for align, ref_words, hyp_words, bias_set in zip(
        output.alignments,
        output.references,
        output.hypotheses,
        utterance_bias_sets,
    ):
        # Count how many reference words belong to the bias list
        for w in ref_words:
            if w.lower() in bias_set:
                total_bias_words += 1

        # Traverse alignment chunks and tally errors involving bias words
        for chunk in align:
            if chunk.type == "substitute":
                for i in range(chunk.ref_start_idx, chunk.ref_end_idx):
                    if ref_words[i].lower() in bias_set:
                        total_sub += 1
            elif chunk.type == "delete":
                for i in range(chunk.ref_start_idx, chunk.ref_end_idx):
                    if ref_words[i].lower() in bias_set:
                        total_del += 1
            elif chunk.type == "insert":
                for i in range(chunk.hyp_start_idx, chunk.hyp_end_idx):
                    if hyp_words[i].lower() in bias_set:
                        total_ins += 1

    if total_bias_words == 0:
        return 0.0

    return (total_sub + total_del + total_ins) / total_bias_words

#训练加acc（val-》测试）