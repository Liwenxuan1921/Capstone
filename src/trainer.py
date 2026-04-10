from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch import nn

from metrics import compute_binary_metrics


@dataclass
class TrainerConfig:
    epochs: int = 20
    patience: int = 5
    device: str = "cpu"
    monitor_metric: str = "auc"


@dataclass
class ResumeState:
    start_epoch: int = 1
    best_metric: float = float("-inf")
    best_epoch: int = -1
    epochs_without_improvement: int = 0
    history: Optional[List[Dict[str, float]]] = None


def _to_float(value: Optional[float]) -> float:
    if value is None:
        return 0.0
    return float(value)


def run_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
) -> Dict[str, object]:
    is_training = optimizer is not None
    model.train(mode=is_training)

    total_loss = 0.0
    sample_count = 0
    all_targets: List[int] = []
    all_probabilities: List[float] = []

    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].float().to(device).unsqueeze(1)

        if is_training:
            optimizer.zero_grad(set_to_none=True)

        logits = model(images)
        loss = criterion(logits, labels)

        if is_training:
            loss.backward()
            optimizer.step()

        probabilities = torch.sigmoid(logits).detach().cpu().view(-1).tolist()
        targets = labels.detach().cpu().view(-1).int().tolist()

        batch_size = len(targets)
        total_loss += float(loss.item()) * batch_size
        sample_count += batch_size
        all_targets.extend(targets)
        all_probabilities.extend(probabilities)

    metrics = compute_binary_metrics(all_targets, all_probabilities)
    metrics["loss"] = total_loss / max(sample_count, 1)

    return {
        "metrics": metrics,
        "targets": all_targets,
        "probabilities": all_probabilities,
    }


def _append_history(history_path: Path, row: Dict[str, float]) -> None:
    history_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = history_path.exists()

    with history_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def load_history(history_path: Path | str) -> List[Dict[str, float]]:
    history_path = Path(history_path)
    if not history_path.exists():
        return []

    with history_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows: List[Dict[str, float]] = []
        for row in reader:
            rows.append({key: float(value) for key, value in row.items()})
    return rows


def _save_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, object],
    best_metric: float,
    best_epoch: int,
    epochs_without_improvement: int,
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "best_metric": best_metric,
            "best_epoch": best_epoch,
            "epochs_without_improvement": epochs_without_improvement,
        },
        checkpoint_path,
    )


def train_model(
    model: nn.Module,
    train_loader,
    val_loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: TrainerConfig,
    checkpoint_path: Path | str,
    latest_checkpoint_path: Path | str,
    interrupt_checkpoint_path: Path | str,
    history_path: Path | str,
    resume_state: Optional[ResumeState] = None,
) -> Dict[str, object]:
    checkpoint_path = Path(checkpoint_path)
    latest_checkpoint_path = Path(latest_checkpoint_path)
    interrupt_checkpoint_path = Path(interrupt_checkpoint_path)
    history_path = Path(history_path)

    resume_state = resume_state or ResumeState()
    best_metric = resume_state.best_metric
    best_epoch = resume_state.best_epoch
    epochs_without_improvement = resume_state.epochs_without_improvement
    history: List[Dict[str, float]] = list(resume_state.history or [])

    for epoch in range(resume_state.start_epoch, config.epochs + 1):
        try:
            train_result = run_epoch(model, train_loader, criterion, config.device, optimizer=optimizer)
            val_result = run_epoch(model, val_loader, criterion, config.device, optimizer=None)
        except KeyboardInterrupt:
            # Save the partially updated model state and restart from this epoch.
            _save_checkpoint(
                checkpoint_path=interrupt_checkpoint_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch - 1,
                metrics={},
                best_metric=best_metric,
                best_epoch=best_epoch,
                epochs_without_improvement=epochs_without_improvement,
            )
            raise

        train_metrics = train_result["metrics"]
        val_metrics = val_result["metrics"]

        row = {
            "epoch": float(epoch),
            "train_loss": _to_float(train_metrics["loss"]),
            "train_accuracy": _to_float(train_metrics["accuracy"]),
            "train_precision": _to_float(train_metrics["precision"]),
            "train_recall": _to_float(train_metrics["recall"]),
            "train_f1": _to_float(train_metrics["f1"]),
            "train_auc": _to_float(train_metrics["auc"]),
            "val_loss": _to_float(val_metrics["loss"]),
            "val_accuracy": _to_float(val_metrics["accuracy"]),
            "val_precision": _to_float(val_metrics["precision"]),
            "val_recall": _to_float(val_metrics["recall"]),
            "val_f1": _to_float(val_metrics["f1"]),
            "val_auc": _to_float(val_metrics["auc"]),
        }
        history.append(row)
        _append_history(history_path, row)

        monitored_value = _to_float(val_metrics.get(config.monitor_metric))
        if monitored_value > best_metric:
            best_metric = monitored_value
            best_epoch = epoch
            epochs_without_improvement = 0
            _save_checkpoint(
                checkpoint_path=checkpoint_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics={"train": train_metrics, "val": val_metrics},
                best_metric=best_metric,
                best_epoch=best_epoch,
                epochs_without_improvement=epochs_without_improvement,
            )
        else:
            epochs_without_improvement += 1

        _save_checkpoint(
            checkpoint_path=latest_checkpoint_path,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            metrics={"train": train_metrics, "val": val_metrics},
            best_metric=best_metric,
            best_epoch=best_epoch,
            epochs_without_improvement=epochs_without_improvement,
        )

        if epochs_without_improvement >= config.patience:
            break

    return {
        "best_metric": best_metric,
        "best_epoch": best_epoch,
        "epochs_without_improvement": epochs_without_improvement,
        "history": history,
    }


def save_json(data: Dict[str, object], output_path: Path | str) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)


def save_history_figure(history: List[Dict[str, float]], output_path: Path | str) -> None:
    import matplotlib.pyplot as plt

    if not history:
        return

    epochs = [row["epoch"] for row in history]
    train_loss = [row["train_loss"] for row in history]
    val_loss = [row["val_loss"] for row in history]
    train_auc = [row["train_auc"] for row in history]
    val_auc = [row["val_auc"] for row in history]

    figure, axes = plt.subplots(1, 2, figsize=(9, 3.8))

    axes[0].plot(epochs, train_loss, label="Train Loss")
    axes[0].plot(epochs, val_loss, label="Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training and Validation Loss")
    axes[0].legend()

    axes[1].plot(epochs, train_auc, label="Train AUC")
    axes[1].plot(epochs, val_auc, label="Val AUC")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("AUC")
    axes[1].set_title("Training and Validation AUC")
    axes[1].legend()

    figure.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
