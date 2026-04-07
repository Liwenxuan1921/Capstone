from __future__ import annotations

import argparse
import json
import random
from datetime import datetime
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from datasets import NIHBinaryChestXrayDataset, build_transforms
from metrics import save_confusion_matrix_figure, save_roc_curve_figure
from models import create_model, get_trainable_parameter_count
from trainer import TrainerConfig, run_epoch, save_history_figure, save_json, train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train chest X-ray binary classification models.")
    parser.add_argument("--train-csv", type=Path, default=Path("data/processed/nih_chestxray14/train.csv"))
    parser.add_argument("--val-csv", type=Path, default=Path("data/processed/nih_chestxray14/val.csv"))
    parser.add_argument("--test-csv", type=Path, default=Path("data/processed/nih_chestxray14/test.csv"))
    parser.add_argument("--images-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--model", choices=["resnet50", "densenet121"], default="resnet50")
    parser.add_argument("--pretrained", action="store_true", help="Use ImageNet-pretrained weights.")
    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="Freeze the feature extractor and train only the classification head.",
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument("--experiment-name", type=str, default="")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_dataloader(dataset, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def make_experiment_name(args: argparse.Namespace) -> str:
    if args.experiment_name:
        return args.experiment_name

    training_mode = "transfer" if args.pretrained else "scratch"
    if args.freeze_backbone:
        training_mode = f"{training_mode}_frozen"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{args.model}_{training_mode}_{timestamp}"


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    transforms_map = build_transforms(image_size=args.image_size)
    train_dataset = NIHBinaryChestXrayDataset(
        csv_path=args.train_csv,
        transform=transforms_map["train"],
        images_root=args.images_root,
    )
    val_dataset = NIHBinaryChestXrayDataset(
        csv_path=args.val_csv,
        transform=transforms_map["eval"],
        images_root=args.images_root,
    )
    test_dataset = NIHBinaryChestXrayDataset(
        csv_path=args.test_csv,
        transform=transforms_map["eval"],
        images_root=args.images_root,
    )

    train_loader = build_dataloader(train_dataset, args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = build_dataloader(val_dataset, args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = build_dataloader(test_dataset, args.batch_size, shuffle=False, num_workers=args.num_workers)

    experiment_name = make_experiment_name(args)
    model_dir = args.output_root / "models" / experiment_name
    results_dir = args.output_root / "results" / experiment_name
    figure_dir = args.output_root / "figures" / experiment_name
    log_dir = args.output_root / "logs" / experiment_name
    checkpoint_path = model_dir / "best_model.pt"
    history_path = log_dir / "history.csv"

    model = create_model(
        model_name=args.model,
        pretrained=args.pretrained,
        freeze_backbone=args.freeze_backbone,
    ).to(args.device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    save_json(
        {
            "experiment_name": experiment_name,
            "model": args.model,
            "pretrained": args.pretrained,
            "freeze_backbone": args.freeze_backbone,
            "trainable_parameters": get_trainable_parameter_count(model),
            "device": args.device,
            "image_size": args.image_size,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "patience": args.patience,
            "seed": args.seed,
        },
        results_dir / "config.json",
    )

    training_summary = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        config=TrainerConfig(
            epochs=args.epochs,
            patience=args.patience,
            device=args.device,
            monitor_metric="auc",
        ),
        checkpoint_path=checkpoint_path,
        history_path=history_path,
    )

    checkpoint = torch.load(checkpoint_path, map_location=args.device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_result = run_epoch(model, test_loader, criterion, args.device, optimizer=None)
    test_metrics = test_result["metrics"]

    save_json(
        {
            "training_summary": training_summary,
            "test_metrics": test_metrics,
        },
        results_dir / "metrics.json",
    )

    save_history_figure(training_summary["history"], figure_dir / "training_history.png")
    save_confusion_matrix_figure(test_metrics, figure_dir / "test_confusion_matrix.png")
    save_roc_curve_figure(
        test_result["targets"],
        test_result["probabilities"],
        figure_dir / "test_roc_curve.png",
    )

    print(json.dumps({"experiment_name": experiment_name, "test_metrics": test_metrics}, indent=2))


if __name__ == "__main__":
    main()
