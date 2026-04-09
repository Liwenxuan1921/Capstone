# Capstone TODO

## Project Goal

Topic: **Deep Learning for Medical Image Processing: Improving Chest X-ray Diagnostic Classification with Transfer Learning and Explainable Deep Learning**

Fixed setup:

- Dataset: `NIH ChestX-ray14`
- Task: binary classification
- Labels:
  - `Normal` = `No Finding`
  - `Abnormal` = any pathology
- Models:
  - `ResNet50`
  - `DenseNet121`
- Comparisons:
  - from scratch
  - transfer learning
- Explainability:
  - `Grad-CAM`

## Phase 1: Project Setup

- [x] Confirm the final title in all project notes and thesis files
- [x] Write one-paragraph problem definition
- [x] Write final research questions
- [x] Define evaluation metrics: accuracy, precision, recall, F1-score, AUC
- [x] Create a folder structure for data, code, outputs, and figures
- [x] Decide how experiment logs will be recorded

Suggested folders:

- [x] `data/`
- [x] `data/raw/`
- [x] `data/processed/`
- [x] `src/`
- [x] `outputs/`
- [x] `outputs/models/`
- [x] `outputs/figures/`
- [x] `outputs/results/`
- [x] `notes/`

## Phase 2: Literature Review

- [x] Collect 20 to 30 relevant papers
- [x] Separate papers into:
  - chest X-ray classification
  - transfer learning in medical imaging
  - explainable AI in medical imaging
- [x] Build a literature review table
- [x] Record for each paper:
  - citation
  - dataset
  - task
  - model
  - metrics
  - strengths
  - limitations
- [x] Identify 3 to 5 key papers most related to your project
- [x] Write a short summary of the research gap
- [x] Write draft notes for Chapter 2

## Phase 3: Dataset Preparation

- [x] Prepare NIH ChestX-ray14 preprocessing and split-generation script
- [x] Document expected raw data structure and output files
- [x] Download `NIH ChestX-ray14`
- [x] Read the dataset documentation
- [x] Locate the image files and label file
- [x] Check whether patient IDs are available
- [x] Inspect class distribution
- [x] Define binary mapping:
  - `No Finding` -> `Normal`
  - all other findings -> `Abnormal`
- [x] Count how many samples are `Normal`
- [x] Count how many samples are `Abnormal`
- [x] Remove missing, broken, or duplicate entries if necessary
- [x] Perform patient-level splitting if possible
- [x] Create train/validation/test split
- [x] Save the split file for reproducibility

Recommended split:

- [x] Train 70%
- [x] Validation 15%
- [x] Test 15%

Preprocessing tasks:

- [ ] Resize images
- [ ] Normalize pixel values
- [ ] Convert grayscale format appropriately for the selected models
- [ ] Define training augmentations
- [ ] Define validation/test transforms
- [ ] Document every preprocessing step

Outputs:

- [x] Dataset statistics table
- [x] Split summary table
- [ ] Preprocessing description for thesis

## Phase 4: Baseline Training Pipeline

- [x] Confirm the Python interpreter path
- [x] Set up the deep learning environment
- [x] Confirm library versions
- [x] Implement data loader
- [x] Implement training loop
- [x] Implement validation loop
- [x] Implement test loop
- [x] Implement metric calculation
- [x] Implement model checkpoint saving
- [x] Implement training log saving
- [x] Implement confusion matrix generation
- [x] Implement ROC/AUC plotting

Training settings to define:

- [x] Batch size
- [x] Learning rate
- [x] Number of epochs
- [x] Optimizer
- [x] Loss function
- [x] Early stopping rule
- [x] Random seed

## Phase 5: Baseline Experiments

### ResNet50 from Scratch

- [x] Implement `ResNet50`
- [ ] Train `ResNet50` from scratch
- [ ] Save best checkpoint
- [ ] Evaluate on validation set
- [ ] Evaluate on test set
- [ ] Record training time
- [ ] Save metrics
- [ ] Save confusion matrix
- [ ] Save ROC curve

### DenseNet121 from Scratch

- [x] Implement `DenseNet121`
- [ ] Train `DenseNet121` from scratch
- [ ] Save best checkpoint
- [ ] Evaluate on validation set
- [ ] Evaluate on test set
- [ ] Record training time
- [ ] Save metrics
- [ ] Save confusion matrix
- [ ] Save ROC curve

## Phase 6: Transfer Learning Experiments

### ResNet50 with Transfer Learning

- [x] Load ImageNet pretrained weights
- [x] Replace final classification head
- [x] Train with frozen backbone
- [ ] Fine-tune upper layers or full model
- [ ] Save best checkpoint
- [ ] Evaluate on validation set
- [ ] Evaluate on test set
- [ ] Record training time
- [ ] Save metrics

### DenseNet121 with Transfer Learning

- [x] Load ImageNet pretrained weights
- [x] Replace final classification head
- [x] Train with frozen backbone
- [ ] Fine-tune upper layers or full model
- [ ] Save best checkpoint
- [ ] Evaluate on validation set
- [ ] Evaluate on test set
- [ ] Record training time
- [ ] Save metrics

Optional only if needed for imbalance:

- [ ] Try weighted loss
- [ ] Try oversampling
- [ ] Try stronger augmentation
- [ ] Compare whether imbalance handling changes results

## Phase 7: Evaluation and Comparison

- [ ] Build one final comparison table covering all four experiments
- [ ] Compare:
  - `ResNet50` scratch vs. transfer learning
  - `DenseNet121` scratch vs. transfer learning
  - `ResNet50` vs. `DenseNet121`
- [ ] Identify the best-performing model
- [ ] Identify the most computationally efficient model
- [ ] Analyze whether transfer learning improves performance consistently
- [ ] Analyze error types using confusion matrices
- [ ] Select final results for thesis tables

Metrics to report:

- [ ] Accuracy
- [ ] Precision
- [ ] Recall
- [ ] F1-score
- [ ] AUC
- [ ] Sensitivity if possible
- [ ] Specificity if possible

Figures to prepare:

- [ ] Training loss curves
- [ ] Validation curves
- [ ] Confusion matrices
- [ ] ROC curves

## Phase 8: Explainability with Grad-CAM

- [ ] Select the best-performing model
- [ ] Select representative test images
- [ ] Generate Grad-CAM for correct `Normal` predictions
- [ ] Generate Grad-CAM for correct `Abnormal` predictions
- [ ] Generate Grad-CAM for false positives
- [ ] Generate Grad-CAM for false negatives
- [ ] Compare whether highlighted regions look medically reasonable
- [ ] Write a short interpretation for each selected case
- [ ] Choose 4 to 8 final figures for the thesis

## Phase 9: Results Discussion

- [ ] Summarize which model performed best
- [ ] Explain why transfer learning helped or did not help
- [ ] Discuss whether DenseNet121 or ResNet50 was more suitable
- [ ] Discuss class imbalance effects
- [ ] Discuss limitations of NIH ChestX-ray14
- [ ] Discuss limitations of binary classification
- [ ] Discuss limitations of Grad-CAM interpretation
- [ ] Write the key findings section

## Phase 10: Thesis Writing

### Chapter 1: Introduction

- [x] Write research background
- [x] Write motivation
- [x] Write problem statement
- [x] Write research objectives
- [x] Write research questions
- [x] Write thesis structure

### Chapter 2: Literature Review

- [x] Write overview of deep learning in medical imaging
- [x] Write overview of chest X-ray classification studies
- [x] Write overview of transfer learning studies
- [x] Write overview of explainable AI studies
- [x] Write research gap section

### Chapter 3: Methodology

- [x] Describe NIH ChestX-ray14
- [x] Describe binary label mapping
- [x] Describe preprocessing
- [x] Describe train/validation/test split
- [x] Describe `ResNet50` and `DenseNet121`
- [x] Describe scratch training and transfer learning settings
- [x] Describe evaluation metrics
- [x] Describe Grad-CAM method

### Chapter 4: Experiments and Results

- [ ] Present experiment setup
- [ ] Present baseline results
- [ ] Present transfer learning results
- [ ] Present final comparison table
- [ ] Present Grad-CAM figures

### Chapter 5: Discussion

- [ ] Interpret the main results
- [ ] Explain observed strengths and weaknesses
- [ ] Discuss model trustworthiness
- [ ] Discuss limitations

### Chapter 6: Conclusion and Future Work

- [ ] Summarize the main findings
- [ ] State contributions
- [ ] Propose future work

## Phase 11: Figures, Tables, and References

- [ ] Finalize all figure filenames and captions
- [ ] Finalize all table titles and numbering
- [ ] Check consistency between text and figures
- [ ] Clean and format references
- [ ] Ensure all cited papers appear in references
- [ ] Ensure all references are cited in the text

## Phase 12: Defense Preparation

- [ ] Create defense slide outline
- [ ] Create final slides
- [ ] Include:
  - background
  - research questions
  - dataset
  - models
  - experiment design
  - results
  - Grad-CAM examples
  - conclusion
- [ ] Prepare answers for likely questions:
  - Why NIH ChestX-ray14?
  - Why binary classification?
  - Why ResNet50 and DenseNet121?
  - Why transfer learning?
  - Why Grad-CAM?
  - What are the limitations?
- [ ] Rehearse presentation
- [ ] Rehearse question answering

## Final Submission Checklist

- [ ] Dataset preprocessing is documented
- [ ] All four experiments are completed
- [ ] Final comparison table is complete
- [ ] Grad-CAM figures are complete
- [ ] Thesis is complete
- [ ] Thesis formatting is checked
- [ ] References are checked
- [ ] Defense slides are complete

## Priority Order If Time Becomes Tight

- [ ] Finish dataset preprocessing first
- [ ] Finish all four core experiments
- [ ] Finish evaluation tables and figures
- [ ] Finish Grad-CAM on the best model
- [ ] Finish thesis writing

If time is very limited, do not add extra models or extra datasets.
