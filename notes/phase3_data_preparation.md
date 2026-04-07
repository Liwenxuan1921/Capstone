# Phase 3 Data Preparation

## Goal

Prepare NIH ChestX-ray14 for the capstone experiments and generate reproducible patient-level train, validation, and test split files.

## Expected Raw Inputs

Place the raw dataset under:

- `data/raw/`

Expected items:

- metadata CSV:
  - `data/raw/Data_Entry_2017.csv`
- image files:
  - either directly under `data/raw/`
  - or inside subfolders under `data/raw/`

The script searches recursively for image files with these extensions:

- `.png`
- `.jpg`
- `.jpeg`

## What the Script Produces

After running the preparation script, these files will be written to:

- `data/processed/nih_chestxray14/`

Outputs:

- `processed_metadata.csv`
- `train.csv`
- `val.csv`
- `test.csv`
- `split_summary.json`
- `missing_images.csv` if any referenced images are not found

## Label Mapping

The binary task is defined as:

- `Normal`: label is exactly `No Finding`
- `Abnormal`: one or more pathology labels are present

## Split Strategy

The script performs:

- patient-level splitting
- target ratio of 70% train, 15% validation, 15% test
- stratified splitting at the patient level using normal vs. abnormal patient grouping

This reduces leakage where images from the same patient might otherwise appear in both training and evaluation sets.

## Run Command

From the project root:

```powershell
python src/prepare_nih_chestxray14.py
```

If your metadata or image folders are in non-default locations:

```powershell
python src/prepare_nih_chestxray14.py `
  --metadata data/raw/Data_Entry_2017.csv `
  --images-root data/raw `
  --output-dir data/processed/nih_chestxray14 `
  --seed 42
```

If you want to create split files before the images are fully available:

```powershell
python src/prepare_nih_chestxray14.py --skip-image-check
```

## Recommended Next Step

After the split files are generated:

1. inspect `split_summary.json`
2. verify class balance in train/val/test
3. use `train.csv`, `val.csv`, and `test.csv` to build the training pipeline in Phase 4
