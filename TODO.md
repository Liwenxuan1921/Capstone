# Capstone TODO

## Project Goal

Topic: **Improving Chest X-ray Diagnostic Classification with Transfer Learning and Explainable Deep Learning**

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

- [ ] Confirm the final title in all project notes and thesis files
- [ ] Write one-paragraph problem definition
- [ ] Write final research questions
- [ ] Define evaluation metrics: accuracy, precision, recall, F1-score, AUC
- [ ] Create a folder structure for data, code, outputs, and figures
- [ ] Decide how experiment logs will be recorded

Suggested folders:

- [ ] `data/`
- [ ] `data/raw/`
- [ ] `data/processed/`
- [ ] `src/`
- [ ] `outputs/`
- [ ] `outputs/models/`
- [ ] `outputs/figures/`
- [ ] `outputs/results/`
- [ ] `notes/`

## Phase 2: Literature Review

- [ ] Collect 20 to 30 relevant papers
- [ ] Separate papers into:
  - chest X-ray classification
  - transfer learning in medical imaging
  - explainable AI in medical imaging
- [ ] Build a literature review table
- [ ] Record for each paper:
  - citation
  - dataset
  - task
  - model
  - metrics
  - strengths
  - limitations
- [ ] Identify 3 to 5 key papers most related to your project
- [ ] Write a short summary of the research gap
- [ ] Write draft notes for Chapter 2

## Phase 3: Dataset Preparation

- [ ] Download `NIH ChestX-ray14`
- [ ] Read the dataset documentation
- [ ] Locate the image files and label file
- [ ] Check whether patient IDs are available
- [ ] Inspect class distribution
- [ ] Define binary mapping:
  - `No Finding` -> `Normal`
  - all other findings -> `Abnormal`
- [ ] Count how many samples are `Normal`
- [ ] Count how many samples are `Abnormal`
- [ ] Remove missing, broken, or duplicate entries if necessary
- [ ] Perform patient-level splitting if possible
- [ ] Create train/validation/test split
- [ ] Save the split file for reproducibility

Recommended split:

- [ ] Train 70%
- [ ] Validation 15%
- [ ] Test 15%

Preprocessing tasks:

- [ ] Resize images
- [ ] Normalize pixel values
- [ ] Convert grayscale format appropriately for the selected models
- [ ] Define training augmentations
- [ ] Define validation/test transforms
- [ ] Document every preprocessing step

Outputs:

- [ ] Dataset statistics table
- [ ] Split summary table
- [ ] Preprocessing description for thesis

## Phase 4: Baseline Training Pipeline

- [ ] Set up the deep learning environment
- [ ] Confirm library versions
- [ ] Implement data loader
- [ ] Implement training loop
- [ ] Implement validation loop
- [ ] Implement test loop
- [ ] Implement metric calculation
- [ ] Implement model checkpoint saving
- [ ] Implement training log saving
- [ ] Implement confusion matrix generation
- [ ] Implement ROC/AUC plotting

Training settings to define:

- [ ] Batch size
- [ ] Learning rate
- [ ] Number of epochs
- [ ] Optimizer
- [ ] Loss function
- [ ] Early stopping rule
- [ ] Random seed

## Phase 5: Baseline Experiments

### ResNet50 from Scratch

- [ ] Implement `ResNet50`
- [ ] Train `ResNet50` from scratch
- [ ] Save best checkpoint
- [ ] Evaluate on validation set
- [ ] Evaluate on test set
- [ ] Record training time
- [ ] Save metrics
- [ ] Save confusion matrix
- [ ] Save ROC curve

### DenseNet121 from Scratch

- [ ] Implement `DenseNet121`
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

- [ ] Load ImageNet pretrained weights
- [ ] Replace final classification head
- [ ] Train with frozen backbone
- [ ] Fine-tune upper layers or full model
- [ ] Save best checkpoint
- [ ] Evaluate on validation set
- [ ] Evaluate on test set
- [ ] Record training time
- [ ] Save metrics

### DenseNet121 with Transfer Learning

- [ ] Load ImageNet pretrained weights
- [ ] Replace final classification head
- [ ] Train with frozen backbone
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

- [ ] Write research background
- [ ] Write motivation
- [ ] Write problem statement
- [ ] Write research objectives
- [ ] Write research questions
- [ ] Write thesis structure

### Chapter 2: Literature Review

- [ ] Write overview of deep learning in medical imaging
- [ ] Write overview of chest X-ray classification studies
- [ ] Write overview of transfer learning studies
- [ ] Write overview of explainable AI studies
- [ ] Write research gap section

### Chapter 3: Methodology

- [ ] Describe NIH ChestX-ray14
- [ ] Describe binary label mapping
- [ ] Describe preprocessing
- [ ] Describe train/validation/test split
- [ ] Describe `ResNet50` and `DenseNet121`
- [ ] Describe scratch training and transfer learning settings
- [ ] Describe evaluation metrics
- [ ] Describe Grad-CAM method

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
