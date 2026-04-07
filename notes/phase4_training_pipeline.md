# Phase 4: Baseline Training Pipeline

This note describes the training pipeline skeleton prepared for the capstone experiments.

## Source Files

- `src/datasets.py`
  - Loads `train.csv`, `val.csv`, and `test.csv`
  - Reads binary labels generated in Phase 3
  - Applies training and evaluation transforms
- `src/models.py`
  - Builds `ResNet50` or `DenseNet121`
  - Supports both scratch training and ImageNet transfer learning
  - Supports optional backbone freezing for head-only training
- `src/metrics.py`
  - Computes accuracy, precision, recall, F1-score, AUC, sensitivity, and specificity
  - Saves confusion matrix and ROC curve figures
- `src/trainer.py`
  - Runs train and validation epochs
  - Handles early stopping
  - Saves the best checkpoint
  - Writes epoch-level history logs
- `src/train.py`
  - Main experiment entrypoint
  - Creates data loaders
  - Builds model, loss, and optimizer
  - Saves outputs to `outputs/models`, `outputs/results`, `outputs/figures`, and `outputs/logs`

## Expected Inputs

Before running Phase 4, Phase 3 should have produced:

- `data/processed/nih_chestxray14/train.csv`
- `data/processed/nih_chestxray14/val.csv`
- `data/processed/nih_chestxray14/test.csv`

Each CSV is expected to include at least:

- `image_name`
- `image_path`
- `patient_id`
- `binary_label`
- `binary_label_name`
- `split`

## Example Commands

ResNet50 from scratch:

```powershell
python src/train.py --model resnet50
```

DenseNet121 from scratch:

```powershell
python src/train.py --model densenet121
```

ResNet50 with transfer learning:

```powershell
python src/train.py --model resnet50 --pretrained
```

DenseNet121 with transfer learning:

```powershell
python src/train.py --model densenet121 --pretrained
```

Optional head-only transfer learning warm-up:

```powershell
python src/train.py --model resnet50 --pretrained --freeze-backbone
```

## Default Training Settings

- image size: `224`
- batch size: `32`
- optimizer: `Adam`
- learning rate: `1e-3`
- weight decay: `1e-4`
- epochs: `20`
- early stopping patience: `5`
- loss: `BCEWithLogitsLoss`
- monitor metric: validation `AUC`
- random seed: `42`

These defaults match the current Methodology chapter draft and can be adjusted later after initial experiments.

## Output Structure

For each experiment run, the pipeline writes:

- `outputs/models/<experiment_name>/best_model.pt`
- `outputs/results/<experiment_name>/config.json`
- `outputs/results/<experiment_name>/metrics.json`
- `outputs/figures/<experiment_name>/training_history.png`
- `outputs/figures/<experiment_name>/test_confusion_matrix.png`
- `outputs/figures/<experiment_name>/test_roc_curve.png`
- `outputs/logs/<experiment_name>/history.csv`

## Current Status

The code skeleton is in place, but it has not been executed in this environment because:

- the raw NIH dataset has not been added to `data/raw`
- the processed split CSVs have not been generated yet
- Python execution in the current shell environment still needs to be confirmed

The next practical step is to finish Phase 3 by generating the split CSV files, then run the four core experiments.
