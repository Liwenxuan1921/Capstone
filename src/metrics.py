from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _rankdata(values: Sequence[float]) -> List[float]:
    order = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    index = 0

    while index < len(order):
        end = index
        while end + 1 < len(order) and order[end + 1][1] == order[index][1]:
            end += 1

        average_rank = (index + end + 2) / 2.0
        for cursor in range(index, end + 1):
            original_index = order[cursor][0]
            ranks[original_index] = average_rank

        index = end + 1

    return ranks


def compute_auc(targets: Sequence[int], probabilities: Sequence[float]) -> Optional[float]:
    positives = sum(targets)
    negatives = len(targets) - positives

    if positives == 0 or negatives == 0:
        return None

    ranks = _rankdata(probabilities)
    positive_rank_sum = sum(rank for rank, target in zip(ranks, targets) if target == 1)
    auc = (positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)
    return float(auc)


def compute_binary_metrics(
    targets: Sequence[int],
    probabilities: Sequence[float],
    threshold: float = 0.5,
) -> Dict[str, Optional[float]]:
    predictions = [1 if probability >= threshold else 0 for probability in probabilities]

    tp = sum(1 for pred, target in zip(predictions, targets) if pred == 1 and target == 1)
    tn = sum(1 for pred, target in zip(predictions, targets) if pred == 0 and target == 0)
    fp = sum(1 for pred, target in zip(predictions, targets) if pred == 1 and target == 0)
    fn = sum(1 for pred, target in zip(predictions, targets) if pred == 0 and target == 1)

    accuracy = _safe_divide(tp + tn, len(targets))
    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    f1 = _safe_divide(2 * precision * recall, precision + recall)
    specificity = _safe_divide(tn, tn + fp)
    auc = compute_auc(targets, probabilities)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
        "sensitivity": recall,
        "specificity": specificity,
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
    }


def save_confusion_matrix_figure(metrics: Dict[str, Optional[float]], output_path: Path | str) -> None:
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    matrix = [
        [int(metrics["tn"] or 0), int(metrics["fp"] or 0)],
        [int(metrics["fn"] or 0), int(metrics["tp"] or 0)],
    ]

    figure, axis = plt.subplots(figsize=(4.2, 4.0))
    axis.imshow(matrix, cmap="Blues")
    axis.set_xticks([0, 1], labels=["Pred Normal", "Pred Abnormal"])
    axis.set_yticks([0, 1], labels=["True Normal", "True Abnormal"])
    axis.set_title("Confusion Matrix")

    for row_index, row in enumerate(matrix):
        for col_index, value in enumerate(row):
            axis.text(col_index, row_index, str(value), ha="center", va="center", color="black")

    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def save_roc_curve_figure(
    targets: Sequence[int],
    probabilities: Sequence[float],
    output_path: Path | str,
) -> None:
    import matplotlib.pyplot as plt

    thresholds = sorted(set(float(probability) for probability in probabilities), reverse=True)
    thresholds = [1.0] + thresholds + [0.0]

    tpr_values: List[float] = []
    fpr_values: List[float] = []

    for threshold in thresholds:
        metrics = compute_binary_metrics(targets, probabilities, threshold=threshold)
        tpr_values.append(float(metrics["recall"] or 0.0))
        fpr_values.append(1.0 - float(metrics["specificity"] or 0.0))

    auc = compute_auc(targets, probabilities)

    figure, axis = plt.subplots(figsize=(4.8, 4.0))
    axis.plot(fpr_values, tpr_values, label=f"ROC (AUC={auc:.4f})" if auc is not None else "ROC")
    axis.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.set_xlabel("False Positive Rate")
    axis.set_ylabel("True Positive Rate")
    axis.set_title("ROC Curve")
    axis.legend(loc="lower right")

    figure.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
