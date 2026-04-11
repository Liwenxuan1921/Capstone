from __future__ import annotations

import argparse
import csv
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare NIH ChestX-ray14 metadata and patient-level train/val/test splits."
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("data/raw/Data_Entry_2017.csv"),
        help="Path to the NIH ChestX-ray14 metadata CSV.",
    )
    parser.add_argument(
        "--images-root",
        type=Path,
        default=Path("data/raw"),
        help="Root directory containing image folders or image files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/nih_chestxray14"),
        help="Directory where processed metadata and split files will be written.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.70,
        help="Training split ratio.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Validation split ratio.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for patient-level split generation.",
    )
    parser.add_argument(
        "--skip-image-check",
        action="store_true",
        help="Do not verify that image files exist under --images-root.",
    )
    return parser.parse_args()


def normalize_name(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def infer_column(fieldnames: Iterable[str], candidates: Iterable[str]) -> str:
    normalized = {normalize_name(name): name for name in fieldnames}
    for candidate in candidates:
        key = normalize_name(candidate)
        if key in normalized:
            return normalized[key]
    raise KeyError(f"Could not infer required column from candidates: {list(candidates)}")


def build_image_index(images_root: Path) -> Dict[str, Path]:
    image_index: Dict[str, Path] = {}
    for root, _, files in os.walk(images_root, followlinks=True):
        root_path = Path(root)
        for name in files:
            path = root_path / name
            if path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                image_index[path.name] = path
    return image_index


def parse_labels(label_text: str) -> List[str]:
    return [label.strip() for label in label_text.split("|") if label.strip()]


def to_binary_label(labels: List[str]) -> Tuple[int, str]:
    if labels == ["No Finding"]:
        return 0, "Normal"
    return 1, "Abnormal"


def allocate_counts(total: int, train_ratio: float, val_ratio: float) -> Tuple[int, int, int]:
    train_count = int(total * train_ratio)
    val_count = int(total * val_ratio)
    test_count = total - train_count - val_count

    if total >= 3:
        if train_count == 0:
            train_count = 1
            test_count -= 1
        if val_count == 0:
            val_count = 1
            test_count -= 1
        if test_count == 0:
            test_count = 1
            if train_count > val_count:
                train_count -= 1
            else:
                val_count -= 1

    return train_count, val_count, test_count


def split_group(ids: List[str], rng: random.Random, train_ratio: float, val_ratio: float) -> Dict[str, str]:
    ids = list(ids)
    rng.shuffle(ids)
    train_count, val_count, _ = allocate_counts(len(ids), train_ratio, val_ratio)

    assignments: Dict[str, str] = {}
    for idx, patient_id in enumerate(ids):
        if idx < train_count:
            assignments[patient_id] = "train"
        elif idx < train_count + val_count:
            assignments[patient_id] = "val"
        else:
            assignments[patient_id] = "test"
    return assignments


def make_patient_split(rows: List[dict], train_ratio: float, val_ratio: float, seed: int) -> Dict[str, str]:
    patient_to_label: Dict[str, int] = {}
    for row in rows:
        patient_id = row["patient_id"]
        patient_to_label[patient_id] = max(patient_to_label.get(patient_id, 0), row["binary_label"])

    normal_patients = [pid for pid, label in patient_to_label.items() if label == 0]
    abnormal_patients = [pid for pid, label in patient_to_label.items() if label == 1]

    rng = random.Random(seed)
    assignments = {}
    assignments.update(split_group(normal_patients, rng, train_ratio, val_ratio))
    assignments.update(split_group(abnormal_patients, rng, train_ratio, val_ratio))
    return assignments


def write_csv(path: Path, rows: List[dict], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_summary(rows: List[dict], missing_images: List[dict], duplicates_removed: int) -> dict:
    summary = {
        "total_rows": len(rows),
        "duplicates_removed": duplicates_removed,
        "missing_images": len(missing_images),
        "split_counts": {},
        "class_counts": {},
        "patient_counts": {},
    }

    for split_name in ["train", "val", "test"]:
        split_rows = [row for row in rows if row["split"] == split_name]
        summary["split_counts"][split_name] = len(split_rows)
        summary["class_counts"][split_name] = dict(Counter(row["binary_label_name"] for row in split_rows))
        summary["patient_counts"][split_name] = len({row["patient_id"] for row in split_rows})

    return summary


def main() -> None:
    args = parse_args()

    if not args.metadata.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {args.metadata}. Place NIH ChestX-ray14 metadata in data/raw first."
        )

    if args.train_ratio <= 0 or args.val_ratio <= 0 or args.train_ratio + args.val_ratio >= 1:
        raise ValueError("train_ratio and val_ratio must be positive and sum to less than 1.")

    image_index: Dict[str, Path] = {}
    if not args.skip_image_check:
        image_index = build_image_index(args.images_root)
        if not image_index:
            raise FileNotFoundError(
                f"No image files found under {args.images_root}. "
                "Place NIH image folders there or rerun with --skip-image-check."
            )

    with args.metadata.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []

        image_col = infer_column(fieldnames, ["Image Index", "image_index", "image"])
        label_col = infer_column(fieldnames, ["Finding Labels", "finding_labels", "labels"])
        patient_col = infer_column(fieldnames, ["Patient ID", "patient_id", "patientid"])

        rows: List[dict] = []
        missing_images: List[dict] = []
        seen_images = set()
        duplicates_removed = 0

        for raw_row in reader:
            image_name = raw_row[image_col].strip()
            if image_name in seen_images:
                duplicates_removed += 1
                continue
            seen_images.add(image_name)

            labels = parse_labels(raw_row[label_col].strip())
            binary_label, binary_label_name = to_binary_label(labels)

            image_path = ""
            if not args.skip_image_check:
                matched_path = image_index.get(image_name)
                if matched_path is None:
                    missing_images.append(
                        {
                            "image_name": image_name,
                            "patient_id": raw_row[patient_col].strip(),
                            "original_labels": raw_row[label_col].strip(),
                        }
                    )
                    continue
                image_path = str(matched_path.resolve())

            rows.append(
                {
                    "image_name": image_name,
                    "image_path": image_path,
                    "patient_id": raw_row[patient_col].strip(),
                    "original_labels": raw_row[label_col].strip(),
                    "binary_label": binary_label,
                    "binary_label_name": binary_label_name,
                }
            )

    patient_split = make_patient_split(rows, args.train_ratio, args.val_ratio, args.seed)
    for row in rows:
        row["split"] = patient_split[row["patient_id"]]

    rows.sort(key=lambda row: (row["split"], row["patient_id"], row["image_name"]))

    args.output_dir.mkdir(parents=True, exist_ok=True)

    base_fields = [
        "image_name",
        "image_path",
        "patient_id",
        "original_labels",
        "binary_label",
        "binary_label_name",
        "split",
    ]

    write_csv(args.output_dir / "processed_metadata.csv", rows, base_fields)
    for split_name in ["train", "val", "test"]:
        split_rows = [row for row in rows if row["split"] == split_name]
        write_csv(args.output_dir / f"{split_name}.csv", split_rows, base_fields)

    if missing_images:
        write_csv(
            args.output_dir / "missing_images.csv",
            missing_images,
            ["image_name", "patient_id", "original_labels"],
        )

    summary = build_summary(rows, missing_images, duplicates_removed)
    with (args.output_dir / "split_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"Wrote processed dataset artifacts to: {args.output_dir}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
