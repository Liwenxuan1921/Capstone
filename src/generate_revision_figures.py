from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import torch
from torch import nn
from torch.utils.data import DataLoader

from datasets import NIHBinaryChestXrayDataset, build_transforms
from metrics import compute_binary_metrics
from models import create_model


ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "outputs" / "figures" / "thesis_revision"
LOG_DIR = ROOT / "outputs" / "logs"
MODELS_DIR = ROOT / "outputs" / "models"
RESULTS_LOG = LOG_DIR / "experiment_log_template.csv"
PREDICTIONS_DIR = ROOT / "outputs" / "predictions"

TEST_CSV = ROOT / "data" / "processed" / "nih_chestxray14" / "test.csv"
IMAGES_ROOT = ROOT / "data" / "raw"

FULL_RUNS = {
    "ResNet50 scratch": LOG_DIR / "resnet50_scratch_full_v1" / "history.csv",
    "ResNet50 transfer": LOG_DIR / "resnet50_transfer_full_v1" / "history.csv",
    "DenseNet121 scratch": LOG_DIR / "densenet121_scratch_full_v1" / "history.csv",
    "DenseNet121 transfer": LOG_DIR / "densenet121_transfer_full_v1" / "history.csv",
}

TRANSFER_CHECKPOINTS = {
    "ResNet50 transfer": MODELS_DIR / "resnet50_transfer_full_v1" / "best_model.pt",
    "DenseNet121 transfer": MODELS_DIR / "densenet121_transfer_full_v1" / "best_model.pt",
}

PREDICTION_FILES = {
    "ResNet50 scratch": PREDICTIONS_DIR / "resnet50_scratch_full_v1_test_predictions.csv",
    "ResNet50 transfer": PREDICTIONS_DIR / "resnet50_transfer_full_v1_test_predictions.csv",
    "DenseNet121 scratch": PREDICTIONS_DIR / "densenet121_scratch_full_v1_test_predictions.csv",
    "DenseNet121 transfer": PREDICTIONS_DIR / "densenet121_transfer_full_v1_test_predictions.csv",
}

TEST_COUNTS = {
    "positive": 7741,
    "negative": 9078,
}


def read_history(path: Path) -> List[Dict[str, float]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [{key: float(value) for key, value in row.items()} for row in reader]


def read_experiment_summary() -> Dict[str, Dict[str, float]]:
    rows: Dict[str, Dict[str, float]] = {}
    with RESULTS_LOG.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if not row["test_accuracy"].strip():
                continue
            name = f"{row['model']} {'transfer' if row['training_strategy'] == 'transfer_learning' else 'scratch'}"
            rows[name] = {
                "accuracy": float(row["test_accuracy"]),
                "precision": float(row["test_precision"]),
                "recall": float(row["test_recall"]),
                "f1": float(row["test_f1"]),
                "auc": float(row["test_auc"]),
                "sensitivity": float(row["sensitivity"]),
                "specificity": float(row["specificity"]),
            }
    return rows


def read_prediction_file(path: Path) -> Tuple[List[int], List[float]]:
    targets: List[int] = []
    probabilities: List[float] = []

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            targets.append(int(row["true_label"]))
            probabilities.append(float(row["predicted_probability"]))

    return targets, probabilities


def save_workflow_diagram(output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(15.6, 3.8))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")

    boxes = [
        (0.03, 0.29, 0.14, 0.34, "NIH ChestXray14\nImages and Labels"),
        (0.23, 0.29, 0.14, 0.34, "Preprocessing\nResize, Normalize\nBinary Mapping"),
        (0.43, 0.29, 0.14, 0.34, "Train-Validation-\nTest Split"),
        (0.63, 0.29, 0.14, 0.34, "Four Experiments\nResNet50, DenseNet121\nScratch, Transfer"),
        (0.83, 0.29, 0.14, 0.34, "Evaluation\nMetrics, ROC\nGrad-CAM"),
    ]

    for x, y, width, height, label in boxes:
        axis.add_patch(
            FancyBboxPatch(
                (x, y),
                width,
                height,
                boxstyle="round,pad=0.006,rounding_size=0.018",
                linewidth=1.5,
                edgecolor="#1b4f72",
                facecolor="#eaf2f8",
            )
        )
        axis.text(
            x + width / 2,
            y + height / 2,
            label,
            ha="center",
            va="center",
            fontsize=12.2,
            linespacing=1.0,
        )

    arrow_y = 0.46
    box_edges = [(x, x + width) for x, _, width, _, _ in boxes]
    for (_, start_edge), (end_edge, _) in zip(box_edges[:-1], box_edges[1:]):
        axis.add_patch(
            FancyArrowPatch(
                (start_edge + 0.008, arrow_y),
                (end_edge - 0.008, arrow_y),
                arrowstyle="-|>",
                mutation_scale=13,
                linewidth=1.5,
                color="#34495e",
            )
        )

    axis.set_title("End-to-End System Workflow for the Thesis Pipeline", fontsize=16.5, pad=12)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_history_grid(
    histories: Dict[str, List[Dict[str, float]]],
    metric_pairs: Sequence[Tuple[str, str, str]],
    output_path: Path,
    title: str,
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=False)
    axes = axes.flatten()

    for axis, (name, rows) in zip(axes, histories.items()):
        epochs = [row["epoch"] for row in rows]
        for key, label, color in metric_pairs:
            axis.plot(epochs, [row[key] for row in rows], label=label, linewidth=2, color=color)
        axis.set_title(name, fontsize=11)
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.25)

    if "loss" in title.lower():
        for axis in axes:
            axis.set_ylabel("Loss")
    else:
        for axis in axes:
            axis.set_ylabel("AUC")
            axis.set_ylim(0.45, 0.9)

    handles, labels = axes[0].get_legend_handles_labels()
    figure.suptitle(title, fontsize=14, y=0.99)
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=len(metric_pairs),
        frameon=False,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.9))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def build_eval_loader(batch_size: int = 128, num_workers: int = 0) -> DataLoader:
    dataset = NIHBinaryChestXrayDataset(
        csv_path=TEST_CSV,
        transform=build_transforms(image_size=224)["eval"],
        images_root=IMAGES_ROOT,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def evaluate_checkpoint(model_name: str, checkpoint_path: Path, pretrained: bool) -> Tuple[List[int], List[float]]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = create_model(model_name=model_name, pretrained=pretrained).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    loader = build_eval_loader()
    targets: List[int] = []
    probabilities: List[float] = []
    criterion = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"].float().to(device).unsqueeze(1)
            logits = model(images)
            _ = criterion(logits, labels)
            targets.extend(labels.cpu().view(-1).int().tolist())
            probabilities.extend(torch.sigmoid(logits).cpu().view(-1).tolist())

    return targets, probabilities


def roc_points(targets: Sequence[int], probabilities: Sequence[float]) -> Tuple[List[float], List[float], float]:
    pairs = sorted(zip(probabilities, targets), key=lambda item: item[0], reverse=True)
    positives = sum(targets)
    negatives = len(targets) - positives

    if positives == 0 or negatives == 0:
        return [0.0, 1.0], [0.0, 1.0], 0.0

    tp = 0
    fp = 0
    last_score = None
    tpr_values: List[float] = [0.0]
    fpr_values: List[float] = [0.0]

    for score, target in pairs:
        if last_score is not None and score != last_score:
            tpr_values.append(tp / positives)
            fpr_values.append(fp / negatives)
        if target == 1:
            tp += 1
        else:
            fp += 1
        last_score = score

    tpr_values.append(tp / positives)
    fpr_values.append(fp / negatives)

    auc = float(compute_binary_metrics(targets, probabilities)["auc"] or 0.0)
    return fpr_values, tpr_values, auc


def auc_summary_curve(auc: float, points: int = 120) -> Tuple[List[float], List[float]]:
    """Create a smooth ROC-shaped summary curve with the requested AUC."""
    exponent = auc / max(1.0 - auc, 1e-6)
    fpr_values = [index / (points - 1) for index in range(points)]
    tpr_values = [1.0 - (1.0 - fpr) ** exponent for fpr in fpr_values]
    return fpr_values, tpr_values


def save_full_roc_comparison(output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(6.8, 5.4))

    configs = [
        ("ResNet50 scratch", "#7f7f7f", "--"),
        ("ResNet50 transfer", "#1f77b4", "-"),
        ("DenseNet121 scratch", "#2ca02c", "-"),
        ("DenseNet121 transfer", "#d62728", "-."),
    ]

    for label, color, linestyle in configs:
        targets, probabilities = read_prediction_file(PREDICTION_FILES[label])
        fpr, tpr, auc = roc_points(targets, probabilities)
        axis.plot(fpr, tpr, linewidth=2.2, linestyle=linestyle, color=color, label=f"{label} (AUC={auc:.4f})")

    axis.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1.2, label="Random classifier")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.set_xlabel("False Positive Rate")
    axis.set_ylabel("True Positive Rate")
    axis.set_title("Test ROC Curves for the Four Full Experiments")
    axis.grid(alpha=0.25)
    axis.legend(loc="lower right", frameon=False, fontsize=9)

    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def derive_confusion_from_rates(summary: Dict[str, float]) -> List[List[int]]:
    tp = round(summary["recall"] * TEST_COUNTS["positive"])
    fn = TEST_COUNTS["positive"] - tp
    tn = round(summary["specificity"] * TEST_COUNTS["negative"])
    fp = TEST_COUNTS["negative"] - tn
    return [[tn, fp], [fn, tp]]


def save_confusion_grid(summaries: Dict[str, Dict[str, float]], output_path: Path) -> None:
    order = [
        "ResNet50 scratch",
        "ResNet50 transfer",
        "DenseNet121 scratch",
        "DenseNet121 transfer",
    ]
    figure, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()

    for axis, name in zip(axes, order):
        matrix = derive_confusion_from_rates(summaries[name])
        axis.imshow(matrix, cmap="Blues")
        axis.set_title(name, fontsize=11)
        axis.set_xticks([0, 1], labels=["Pred Normal", "Pred Abnormal"])
        axis.set_yticks([0, 1], labels=["True Normal", "True Abnormal"])
        for row_index, row in enumerate(matrix):
            for col_index, value in enumerate(row):
                axis.text(col_index, row_index, str(value), ha="center", va="center", color="black", fontsize=12)

    figure.suptitle("Test Confusion Matrices Derived from the Final Experimental Results", fontsize=14, y=0.98)
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    histories = {name: read_history(path) for name, path in FULL_RUNS.items()}
    summaries = read_experiment_summary()

    save_workflow_diagram(OUTPUT_DIR / "workflow_diagram.png")
    plot_history_grid(
        histories,
        metric_pairs=(
            ("train_loss", "Train loss", "#1f77b4"),
            ("val_loss", "Validation loss", "#d62728"),
        ),
        output_path=OUTPUT_DIR / "full_training_loss_curves.png",
        title="Training and Validation Loss Curves for the Four Full Experiments",
    )
    plot_history_grid(
        histories,
        metric_pairs=(
            ("train_auc", "Train AUC", "#2ca02c"),
            ("val_auc", "Validation AUC", "#9467bd"),
        ),
        output_path=OUTPUT_DIR / "full_auc_curves.png",
        title="Training and Validation AUC Curves for the Four Full Experiments",
    )
    save_full_roc_comparison(OUTPUT_DIR / "full_roc_comparison.png")
    save_confusion_grid(summaries, OUTPUT_DIR / "full_confusion_matrices.png")


if __name__ == "__main__":
    main()
