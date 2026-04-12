from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from datasets import build_transforms
from models import create_model


@dataclass
class SamplePrediction:
    image_name: str
    image_path: str
    patient_id: str
    original_labels: str
    true_label: int
    predicted_label: int
    probability_abnormal: float
    category: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Grad-CAM visualizations for chest X-ray models.")
    parser.add_argument("--model", choices=["resnet50", "densenet121"], default="resnet50")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("outputs/models/resnet50_transfer_full_v1/best_model.pt"),
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=Path("data/processed/nih_chestxray14/test.csv"),
    )
    parser.add_argument("--images-root", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/figures/resnet50_transfer_full_v1/gradcam"),
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_rows(csv_path: Path) -> List[Dict[str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def resolve_image_path(row: Dict[str, str], images_root: Path) -> Path:
    raw_path = row.get("image_path", "").strip()
    if raw_path:
        candidate = Path(raw_path)
        if candidate.exists():
            return candidate

    direct = images_root / row["image_name"]
    if direct.exists():
        return direct

    matches = list(images_root.rglob(row["image_name"]))
    if matches:
        return matches[0]

    raise FileNotFoundError(f"Could not locate image for {row['image_name']}")


def get_target_layer(model: torch.nn.Module, model_name: str):
    if model_name == "resnet50":
        return model.layer4[-1]
    if model_name == "densenet121":
        return model.features
    raise ValueError(f"Unsupported model name: {model_name}")


def compute_predictions(
    model: torch.nn.Module,
    rows: List[Dict[str, str]],
    transform,
    images_root: Path,
    device: str,
    threshold: float,
    batch_size: int,
) -> List[SamplePrediction]:
    predictions: List[SamplePrediction] = []
    model.eval()

    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start : start + batch_size]
        batch_tensors = []
        batch_paths: List[Path] = []

        for row in batch_rows:
            image_path = resolve_image_path(row, images_root)
            image = Image.open(image_path).convert("RGB")
            batch_tensors.append(transform(image))
            batch_paths.append(image_path)

        tensor = torch.stack(batch_tensors, dim=0).to(device)

        with torch.no_grad():
            logits = model(tensor)
            probabilities = torch.sigmoid(logits).view(-1).cpu().tolist()

        for row, image_path, probability in zip(batch_rows, batch_paths, probabilities):
            true_label = int(row["binary_label"])
            predicted_label = 1 if probability >= threshold else 0

            if true_label == 0 and predicted_label == 0:
                category = "correct_normal"
            elif true_label == 1 and predicted_label == 1:
                category = "correct_abnormal"
            elif true_label == 0 and predicted_label == 1:
                category = "false_positive"
            else:
                category = "false_negative"

            predictions.append(
                SamplePrediction(
                    image_name=row["image_name"],
                    image_path=str(image_path),
                    patient_id=row.get("patient_id", ""),
                    original_labels=row.get("original_labels", ""),
                    true_label=true_label,
                    predicted_label=predicted_label,
                    probability_abnormal=float(probability),
                    category=category,
                )
            )

    return predictions


def select_representative_samples(predictions: Iterable[SamplePrediction]) -> List[SamplePrediction]:
    grouped: Dict[str, List[SamplePrediction]] = {
        "correct_normal": [],
        "correct_abnormal": [],
        "false_positive": [],
        "false_negative": [],
    }
    for prediction in predictions:
        grouped[prediction.category].append(prediction)

    grouped["correct_normal"].sort(key=lambda item: 1.0 - item.probability_abnormal, reverse=True)
    grouped["correct_abnormal"].sort(key=lambda item: item.probability_abnormal, reverse=True)
    grouped["false_positive"].sort(key=lambda item: item.probability_abnormal, reverse=True)
    grouped["false_negative"].sort(key=lambda item: 1.0 - item.probability_abnormal, reverse=True)

    selected: List[SamplePrediction] = []
    for key in ["correct_normal", "correct_abnormal", "false_positive", "false_negative"]:
        if not grouped[key]:
            raise RuntimeError(f"No samples found for category: {key}")
        selected.append(grouped[key][0])
    return selected


def compute_gradcam(
    model: torch.nn.Module,
    tensor: torch.Tensor,
    model_name: str,
    target_label: int,
) -> np.ndarray:
    activations = []
    gradients = []
    target_layer = get_target_layer(model, model_name)

    def forward_hook(_, __, output):
        activations.append(output.detach())

    def backward_hook(_, grad_input, grad_output):
        gradients.append(grad_output[0].detach())

    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)

    model.zero_grad(set_to_none=True)
    logits = model(tensor)
    score = logits[:, 0] if target_label == 1 else -logits[:, 0]
    score.backward(torch.ones_like(score))

    forward_handle.remove()
    backward_handle.remove()

    activation = activations[0]
    gradient = gradients[0]
    weights = gradient.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * activation).sum(dim=1, keepdim=True))
    cam = torch.nn.functional.interpolate(
        cam,
        size=tensor.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )
    cam = cam.squeeze().cpu().numpy()
    cam = cam - cam.min()
    if cam.max() > 0:
        cam = cam / cam.max()
    return cam


def denormalize_image(tensor: torch.Tensor) -> np.ndarray:
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
    image = tensor.squeeze(0).cpu().numpy()
    image = (image * std) + mean
    image = np.clip(image, 0.0, 1.0)
    return np.transpose(image, (1, 2, 0))


def create_overlay(image: np.ndarray, cam: np.ndarray, alpha: float = 0.4) -> Tuple[np.ndarray, np.ndarray]:
    heatmap = cm.jet(cam)[..., :3]
    overlay = np.clip((1 - alpha) * image + alpha * heatmap, 0.0, 1.0)
    return heatmap, overlay


def save_case_figure(
    sample: SamplePrediction,
    image: np.ndarray,
    heatmap: np.ndarray,
    overlay: np.ndarray,
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(10, 3.8))
    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Original")
    axes[1].imshow(heatmap)
    axes[1].set_title("Grad-CAM")
    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")

    for axis in axes:
        axis.axis("off")

    title = (
        f"{sample.category.replace('_', ' ').title()} | "
        f"True={sample.true_label} Pred={sample.predicted_label} "
        f"P(abnormal)={sample.probability_abnormal:.4f}"
    )
    figure.suptitle(title, fontsize=10)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def save_overview_grid(cases: List[Tuple[SamplePrediction, np.ndarray]], output_path: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(8, 8))
    for axis, (sample, overlay) in zip(axes.flatten(), cases):
        axis.imshow(overlay)
        axis.set_title(
            f"{sample.category.replace('_', ' ').title()}\n"
            f"P={sample.probability_abnormal:.4f}",
            fontsize=10,
        )
        axis.axis("off")

    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    rows = load_rows(args.csv_path)
    transforms_map = build_transforms(image_size=args.image_size)

    model = create_model(model_name=args.model, pretrained=False).to(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    predictions = compute_predictions(
        model=model,
        rows=rows,
        transform=transforms_map["eval"],
        images_root=args.images_root,
        device=args.device,
        threshold=args.threshold,
        batch_size=args.batch_size,
    )
    selected_samples = select_representative_samples(predictions)

    overview_cases: List[Tuple[SamplePrediction, np.ndarray]] = []
    summary_rows: List[Dict[str, object]] = []

    for sample in selected_samples:
        image_path = Path(sample.image_path)
        image = Image.open(image_path).convert("RGB")
        tensor = transforms_map["eval"](image).unsqueeze(0).to(args.device)
        target_label = sample.predicted_label
        cam = compute_gradcam(model=model, tensor=tensor, model_name=args.model, target_label=target_label)
        denormalized = denormalize_image(tensor)
        heatmap, overlay = create_overlay(denormalized, cam)

        file_stem = f"{sample.category}_{sample.image_name.replace('.png', '')}"
        save_case_figure(sample, denormalized, heatmap, overlay, args.output_dir / f"{file_stem}.png")
        overview_cases.append((sample, overlay))
        summary_rows.append(
            {
                "category": sample.category,
                "image_name": sample.image_name,
                "image_path": sample.image_path,
                "patient_id": sample.patient_id,
                "original_labels": sample.original_labels,
                "true_label": sample.true_label,
                "predicted_label": sample.predicted_label,
                "probability_abnormal": sample.probability_abnormal,
            }
        )

    save_overview_grid(overview_cases, args.output_dir / "gradcam_overview.png")

    summary_path = args.output_dir / "gradcam_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary_rows, handle, indent=2)

    print(json.dumps({"output_dir": str(args.output_dir.resolve()), "cases": summary_rows}, indent=2))


if __name__ == "__main__":
    main()
