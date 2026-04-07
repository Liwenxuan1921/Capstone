# Capstone README

## Topic

**Deep Learning for Medical Image Processing: Improving Chest X-ray Diagnostic Classification with Transfer Learning and Explainable Deep Learning**

This capstone focuses on a **single, manageable research problem**:

- Task: chest X-ray image classification
- Goal: improve diagnostic classification performance in a stable and explainable way
- Core methods:
  - baseline deep learning models
  - transfer learning
  - explainable AI

To keep the project realistic, do **not** expand into segmentation, detection, federated learning, diffusion models, or clinical deployment systems unless all core work is already complete.

## Finalized Project Configuration

The project decisions are fixed as follows:

- Dataset: `NIH ChestX-ray14`
- Task: binary chest X-ray classification
- Binary label definition:
  - `Normal`: images labeled `No Finding`
  - `Abnormal`: images with any disease finding
- Baseline models:
  - `ResNet50`
  - `DenseNet121`
- Comparison setting:
  - training from scratch
  - transfer learning
- Explainability method:
  - `Grad-CAM`

This configuration is intentionally conservative. It is suitable for a stable undergraduate capstone because it keeps the scope narrow while still allowing meaningful experiments and discussion.

## Research Scope

### Main research question

How can transfer learning and explainable deep learning improve the performance and trustworthiness of chest X-ray diagnostic classification?

### Sub-questions

1. How do baseline CNN models perform on chest X-ray classification?
2. Does transfer learning improve accuracy, F1-score, and AUC compared with training from scratch?
3. Can explainability methods such as Grad-CAM help interpret model decisions and analyze errors?

### Minimum viable project

To finish the capstone safely, the minimum target is:

- 1 public chest X-ray dataset
- 2 baseline models
- 1 transfer learning experiment
- 1 explainability method
- 1 complete thesis

## Recommended Technical Setup

### Dataset

Selected dataset:

- `NIH ChestX-ray14`

### Classification target

Selected target:

- binary classification: `No Finding` vs. `Any Pathology`

Reason:

- this is the simplest and safest label setting for a capstone
- it supports a clean comparison between models
- it avoids the added instability of multi-label classification

### Models

Selected model set:

- Baseline 1: `ResNet50`
- Baseline 2: `DenseNet121`
- Improvement condition: transfer learning on both models

### Explainability

Selected explainability method:

- `Grad-CAM`

## Project Steps

### Step 1: Finalize the problem definition

Complete these decisions first:

- confirm the `No Finding` vs. `Any Pathology` label rule
- define the train/validation/test split strategy
- define the evaluation metrics
- confirm the experiment matrix

Output:

- a one-paragraph project definition
- a table of dataset, task, models, and metrics

### Step 2: Literature review

Read papers in these groups:

- chest X-ray classification
- transfer learning in medical imaging
- explainable AI in medical imaging

Target:

- 20 to 30 papers

For each paper, record:

- citation
- task
- dataset
- model
- metrics
- strengths
- limitations

Output:

- literature review notes
- comparison table
- summary of research gap

What your gap should look like:

- many papers report strong results, but fewer combine a clean baseline comparison, transfer learning improvement, and explainability analysis in one compact undergraduate capstone study

### Step 3: Prepare the dataset

Tasks:

1. download the dataset
2. identify images labeled `No Finding`
3. identify images with one or more disease labels
4. map labels into `Normal` and `Abnormal`
5. remove broken or duplicated records if needed
6. resize images
7. normalize images
8. split into train, validation, and test sets

Recommended split:

- train: 70%
- validation: 15%
- test: 15%

Important:

- keep patient-level separation if the dataset provides patient IDs
- record every preprocessing step for the methodology chapter

Output:

- dataset statistics table
- preprocessing pipeline description
- final split counts

### Step 4: Build baseline models

Train the two selected baseline models.

Model order:

1. `ResNet50`
2. `DenseNet121`

For each baseline, record:

- training loss
- validation loss
- accuracy
- precision
- recall
- F1-score
- AUC
- training time

Output:

- baseline result table
- training curves
- confusion matrix

### Step 5: Add transfer learning

This is the main improvement stage.

Experiments to run:

1. `ResNet50` from scratch
2. `ResNet50` with transfer learning
3. `DenseNet121` from scratch
4. `DenseNet121` with transfer learning

Recommended transfer learning strategy:

- initialize from ImageNet pretrained weights
- first train with a frozen backbone
- then fine-tune the upper or full backbone if needed

If class imbalance is a problem, also try:

- weighted loss
- oversampling
- data augmentation

Keep the design controlled:

- change one major factor at a time
- do not mix too many improvements in a single experiment

Output:

- experiment comparison table
- explanation of why transfer learning helps or does not help

### Step 6: Add explainability

Use Grad-CAM on the best-performing model and, if time allows, one weaker model.

Generate Grad-CAM for:

- correct predictions
- false positives
- false negatives

Questions to answer:

- does the model focus on medically relevant regions?
- are there cases where attention is obviously unreasonable?
- what kinds of errors appear repeatedly?

Output:

- 4 to 8 Grad-CAM figures
- one error analysis section

### Step 7: Evaluate results

Your final evaluation should compare:

- `from scratch` vs. `transfer learning`
- `ResNet50` vs. `DenseNet121`
- performance vs. computational cost

Metrics to report:

- accuracy
- precision
- recall
- F1-score
- AUC

If possible, also report:

- sensitivity
- specificity

Output:

- final comparison table
- final discussion of best model

### Step 8: Write the thesis

Map your work into the thesis structure already in this folder.

Suggested chapter plan:

1. Introduction
2. Literature Review
3. Methodology
4. Experiments and Results
5. Discussion
6. Conclusion and Future Work

### Chapter 1: Introduction

Write:

- background of chest X-ray diagnosis
- motivation for automated classification
- research problem
- objectives
- research questions
- thesis structure

### Chapter 2: Literature Review

Write:

- deep learning in medical imaging
- chest X-ray classification studies
- transfer learning studies
- explainable AI studies
- research gap

### Chapter 3: Methodology

Write:

- dataset description
- preprocessing steps
- model architectures
- transfer learning design
- explainability method
- evaluation metrics

### Chapter 4: Experiments and Results

Write:

- baseline results
- transfer learning results
- tables and figures
- Grad-CAM examples

### Chapter 5: Discussion

Write:

- why some models performed better
- effect of transfer learning
- explainability findings
- limitations

### Chapter 6: Conclusion and Future Work

Write:

- main findings
- research contribution
- practical meaning
- future extensions

### Step 9: Prepare the defense

Create a presentation with:

1. background
2. problem statement
3. research questions
4. dataset and methods
5. baseline and transfer learning results
6. explainability examples
7. conclusion
8. limitations and future work

Keep the defense focused on:

- what problem you solved
- how you designed the experiments
- why the results matter

## Suggested Timeline

### Week 1

- finalize dataset, labels, models, and metrics
- start literature review

### Week 2

- complete literature matrix
- prepare dataset and preprocessing

### Week 3

- train baseline model 1
- train baseline model 2

### Week 4

- analyze baseline performance
- run transfer learning experiments

### Week 5

- compare results
- generate figures and tables

### Week 6

- run Grad-CAM
- perform error analysis

### Week 7

- write Chapters 1 to 3

### Week 8

- write Chapters 4 to 6

### Week 9

- revise thesis
- fix formatting
- prepare slides

### Week 10

- rehearse defense
- finalize submission

## Deliverables Checklist

By the end of the capstone, you should have:

- a finalized topic statement
- a literature review table
- a cleaned and documented dataset split
- trained `ResNet50` and `DenseNet121` baselines
- `from scratch` and `transfer learning` comparison results
- Grad-CAM visualizations
- final result tables and figures
- complete thesis draft
- defense slides

## Risk Control

If you face time or hardware limitations, simplify in this order:

1. reduce to one dataset
2. reduce to two models only
3. keep only binary classification
4. keep only one explainability method

Do not simplify these away:

- baseline comparison
- transfer learning
- final evaluation
- thesis writing

## Recommended Final Positioning

Your capstone should present itself as:

> a focused comparative study on chest X-ray classification that evaluates whether transfer learning can improve performance and whether explainable AI can make model behavior easier to interpret

This positioning is realistic, academically defensible, and suitable for a stable capstone project.

## Next Actions

Do these next:

1. document the `No Finding` vs. `Any Pathology` label rule
2. start the literature review table
3. download and inspect NIH ChestX-ray14
4. build the train/validation/test split
5. implement `ResNet50` and `DenseNet121` baselines

If you stay within this scope, the project is very achievable.
