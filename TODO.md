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

## Must Finish Before Submission

### Thesis Core

- [x] Final title is fixed across thesis files
- [x] Abstract is written
- [ ] Acknowledgements are written
- [x] Chapter 1 is drafted
- [x] Chapter 2 is drafted
- [x] Chapter 3 is drafted
- [x] Chapter 4 includes experiment setup and completed transfer-learning results
- [x] Chapter 5 discussion is drafted
- [x] Chapter 6 conclusion and future work are drafted
- [ ] Final language cleanup across Chapters 1-6
- [ ] Final formatting check of the thesis PDF

### Experiments and Results

- [x] NIH ChestX-ray14 dataset is prepared
- [x] Reproducible train/validation/test split is saved
- [x] Transfer-learning experiment: `ResNet50`
- [x] Transfer-learning experiment: `DenseNet121`
- [x] Metrics are saved for completed runs
- [x] Confusion matrix and ROC outputs are generated for `ResNet50`
- [x] Final transfer-learning comparison table is inserted into Chapter 4
- [x] Best-performing model is identified

### Explainability

- [x] Best-performing model is selected for Grad-CAM
- [x] Representative test images are selected
- [x] Grad-CAM for correct `Normal` prediction is generated
- [x] Grad-CAM for correct `Abnormal` prediction is generated
- [x] Grad-CAM for false positive is generated
- [x] Grad-CAM for false negative is generated
- [x] Short interpretation is written for selected Grad-CAM cases
- [x] Grad-CAM figures are inserted into Chapter 4

### References and Consistency

- [ ] Check all in-text citations against the reference list
- [ ] Ensure all cited papers appear in references
- [ ] Ensure no important reference is listed but unused
- [ ] Check consistency between text, tables, and reported numbers

### Submission Readiness

- [ ] Thesis PDF compiles cleanly enough for submission
- [ ] Final `README` / project notes reflect the delivered scope
- [ ] Defense slide outline is prepared

## Optional If Time Allows

### Additional Experiments

- [ ] Train `ResNet50` from scratch
- [ ] Train `DenseNet121` from scratch
- [ ] Build the full four-experiment comparison table
- [ ] Record training time for every run
- [ ] Identify the most computationally efficient model

### Extra Analysis

- [ ] Analyze transfer learning against scratch training for both models
- [ ] Analyze confusion-matrix error types in more depth
- [ ] Try weighted loss
- [ ] Try oversampling
- [ ] Try stronger augmentation
- [ ] Compare whether imbalance handling changes results

### Thesis Enhancements

- [ ] Add more training-curve discussion
- [ ] Add more detailed Grad-CAM commentary
- [ ] Expand Chapter 5 with deeper comparison to literature
- [ ] Further reduce LaTeX overfull warnings

### Defense Preparation

- [ ] Create final defense slides
- [ ] Add Grad-CAM figures to slides
- [ ] Prepare answers for likely questions
- [ ] Rehearse presentation

## Progress by Phase

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

- [x] Resize images
- [x] Normalize pixel values
- [x] Convert grayscale format appropriately for the selected models
- [x] Define training augmentations
- [x] Define validation/test transforms
- [x] Document every preprocessing step

Outputs:

- [x] Dataset statistics table
- [x] Split summary table
- [x] Preprocessing description for thesis

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
- [x] Fine-tune upper layers or full model
- [x] Save best checkpoint
- [x] Evaluate on validation set
- [x] Evaluate on test set
- [ ] Record training time
- [x] Save metrics

### DenseNet121 with Transfer Learning

- [x] Load ImageNet pretrained weights
- [x] Replace final classification head
- [x] Train with frozen backbone
- [x] Fine-tune upper layers or full model
- [x] Save best checkpoint
- [x] Evaluate on validation set
- [x] Evaluate on test set
- [ ] Record training time
- [x] Save metrics

## Phase 7: Evaluation and Comparison

- [ ] Build one final comparison table covering all four experiments
- [ ] Compare:
  - `ResNet50` scratch vs. transfer learning
  - `DenseNet121` scratch vs. transfer learning
  - `ResNet50` vs. `DenseNet121`
- [x] Identify the best-performing model
- [ ] Identify the most computationally efficient model
- [ ] Analyze whether transfer learning improves performance consistently
- [ ] Analyze error types using confusion matrices
- [x] Select final results for thesis tables

Metrics to report:

- [x] Accuracy
- [x] Precision
- [x] Recall
- [x] F1-score
- [x] AUC
- [x] Sensitivity if possible
- [x] Specificity if possible

Figures to prepare:

- [x] Training loss curves
- [x] Validation curves
- [x] Confusion matrices
- [x] ROC curves

## Phase 8: Explainability with Grad-CAM

- [x] Select the best-performing model
- [x] Select representative test images
- [x] Generate Grad-CAM for correct `Normal` predictions
- [x] Generate Grad-CAM for correct `Abnormal` predictions
- [x] Generate Grad-CAM for false positives
- [x] Generate Grad-CAM for false negatives
- [x] Compare whether highlighted regions look medically reasonable
- [x] Write a short interpretation for each selected case
- [x] Choose 4 to 8 final figures for the thesis

## Phase 9: Results Discussion

- [x] Summarize which model performed best
- [x] Explain why transfer learning helped or did not help
- [x] Discuss whether DenseNet121 or ResNet50 was more suitable
- [ ] Discuss class imbalance effects
- [x] Discuss limitations of NIH ChestX-ray14
- [x] Discuss limitations of binary classification
- [x] Discuss limitations of Grad-CAM interpretation
- [x] Write the key findings section

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

- [x] Present experiment setup
- [ ] Present baseline results
- [x] Present transfer learning results
- [x] Present final comparison table
- [x] Present Grad-CAM figures

### Chapter 5: Discussion

- [x] Interpret the main results
- [x] Explain observed strengths and weaknesses
- [x] Discuss model trustworthiness
- [x] Discuss limitations

### Chapter 6: Conclusion and Future Work

- [x] Summarize the main findings
- [x] State contributions
- [x] Propose future work

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
