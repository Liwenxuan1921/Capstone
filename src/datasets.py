from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable, Dict, List, Optional

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class NIHBinaryChestXrayDataset(Dataset):
    def __init__(
        self,
        csv_path: Path | str,
        transform: Optional[Callable] = None,
        images_root: Optional[Path | str] = None,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.transform = transform
        self.images_root = Path(images_root) if images_root is not None else None
        self.rows = self._load_rows()

    def _load_rows(self) -> List[Dict[str, str]]:
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Dataset split file not found: {self.csv_path}")

        with self.csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = [row for row in reader]

        if not rows:
            raise ValueError(f"No samples found in dataset split file: {self.csv_path}")

        required_columns = {"image_name", "binary_label"}
        missing = required_columns.difference(rows[0].keys())
        if missing:
            raise KeyError(f"Missing required dataset columns in {self.csv_path}: {sorted(missing)}")

        return rows

    def _resolve_image_path(self, row: Dict[str, str]) -> Path:
        image_path = row.get("image_path", "").strip()
        if image_path:
            return Path(image_path)

        if self.images_root is None:
            raise FileNotFoundError(
                "CSV row does not contain image_path and no images_root was provided."
            )

        candidate = self.images_root / row["image_name"]
        if candidate.exists():
            return candidate

        matches = list(self.images_root.rglob(row["image_name"]))
        if matches:
            return matches[0]

        raise FileNotFoundError(f"Could not locate image file for: {row['image_name']}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Dict[str, object]:
        row = self.rows[index]
        image_path = self._resolve_image_path(row)

        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        return {
            "image": image,
            "label": float(row["binary_label"]),
            "image_name": row["image_name"],
            "patient_id": row.get("patient_id", ""),
            "image_path": str(image_path),
        }


def build_transforms(image_size: int = 224) -> Dict[str, transforms.Compose]:
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=7),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ]
    )

    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ]
    )

    return {
        "train": train_transform,
        "eval": eval_transform,
    }
