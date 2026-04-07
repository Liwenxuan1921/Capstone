# Phase 1 Project Setup

## Final Topic

**Deep Learning for Medical Image Processing: Improving Chest X-ray Diagnostic Classification with Transfer Learning and Explainable Deep Learning**

## One-Paragraph Problem Definition

This capstone investigates binary chest X-ray diagnostic classification using deep learning on the NIH ChestX-ray14 dataset. The project studies whether transfer learning can improve classification performance compared with training from scratch, and whether explainability methods can make model decisions easier to interpret. The classification task is defined as `Normal` versus `Abnormal`, where images labeled `No Finding` are treated as normal and images with any pathology label are treated as abnormal. Two convolutional neural network architectures, ResNet50 and DenseNet121, will be compared under both scratch-training and transfer-learning settings. Model behavior will be analyzed quantitatively through classification metrics and qualitatively through Grad-CAM visualizations.

## Research Questions

1. How do ResNet50 and DenseNet121 perform on binary chest X-ray classification when trained from scratch?
2. Does transfer learning improve the performance of these models on NIH ChestX-ray14?
3. Can Grad-CAM help explain correct predictions and common model errors in chest X-ray classification?

## Scope

Included in this project:

- one public dataset: `NIH ChestX-ray14`
- one binary classification task: `No Finding` vs. `Any Pathology`
- two base architectures: `ResNet50` and `DenseNet121`
- two training settings: `from scratch` and `transfer learning`
- one explainability method: `Grad-CAM`

Excluded from this project:

- lesion localization
- segmentation
- object detection
- federated learning
- clinical deployment
- multi-dataset benchmarking

## Dataset and Label Definition

Dataset:

- `NIH ChestX-ray14`

Binary labels:

- `Normal`: label is exactly `No Finding`
- `Abnormal`: one or more pathology labels are present

Important implementation rule:

- if patient IDs are available, splitting must be done at the patient level

## Planned Project Structure

- `data/raw/`: original downloaded dataset files
- `data/processed/`: cleaned metadata, split files, and processed artifacts
- `src/`: preprocessing, training, evaluation, and Grad-CAM scripts
- `outputs/models/`: trained checkpoints
- `outputs/figures/`: plots, confusion matrices, ROC curves, Grad-CAM images
- `outputs/results/`: CSV tables and summary metrics
- `outputs/logs/`: training logs and run notes
- `notes/`: planning documents and literature notes

## Evaluation Metrics

Primary metrics:

- Accuracy
- Precision
- Recall
- F1-score
- AUC

Secondary metrics:

- Sensitivity
- Specificity
- Confusion matrix
- Training time

## Experiment Matrix

| Experiment ID | Model | Training Strategy | Pretrained Weights | Goal |
| --- | --- | --- | --- | --- |
| E1 | ResNet50 | From scratch | No | Baseline comparison |
| E2 | ResNet50 | Transfer learning | ImageNet | Measure transfer benefit |
| E3 | DenseNet121 | From scratch | No | Baseline comparison |
| E4 | DenseNet121 | Transfer learning | ImageNet | Measure transfer benefit |

## Default Training Decisions

These are initial defaults and may be adjusted later:

- task type: binary classification
- image backbone inputs: convert grayscale chest X-rays to model-compatible tensor format
- split target: train 70%, validation 15%, test 15%
- checkpoint rule: save the best validation model
- comparison principle: keep preprocessing, split, and metrics fixed across all four experiments

## Phase 1 Deliverables

- finalized topic statement
- one-paragraph problem definition
- confirmed research questions
- fixed evaluation metrics
- fixed experiment matrix
- standard project folder structure
- experiment logging template

## Immediate Next Steps

1. Download NIH ChestX-ray14 into `data/raw/`
2. Inspect metadata and confirm label fields
3. Create the binary label mapping
4. Generate a reproducible train/validation/test split
5. Begin implementing the training pipeline in `src/`
