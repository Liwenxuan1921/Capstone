# Phase 2 Reading Queue

## Goal

Build a literature set of **20 to 30 papers** that directly supports Chapter 2.

## Category Targets

- Chest X-ray classification papers: 8 to 10
- Transfer learning in medical imaging: 5 to 7
- Explainable AI / Grad-CAM papers: 4 to 6
- Backbone / foundational model papers: 3 to 4
- Surveys and benchmark context papers: 2 to 3

## Read in This Order

### 1. Foundation and dataset context

- Wang et al. on ChestX-ray8 / ChestX-ray14
- Litjens et al. survey on deep learning in medical image analysis
- He et al. on ResNet
- Huang et al. on DenseNet

Why first:

- these papers define the dataset and model background your thesis depends on

### 2. Chest X-ray classification studies

- Rajpurkar et al. on CheXNet
- Irvin et al. on CheXpert
- additional papers using ChestX-ray14 or CheXpert for disease classification

Questions to answer while reading:

- what dataset was used?
- was the task binary, multi-class, or multi-label?
- what model backbone was used?
- what metrics were reported?
- what limitations remain?

### 3. Transfer learning studies

- Tajbakhsh et al. on transfer learning in medical imaging
- chest X-ray papers that compare pretrained vs. scratch training

Questions to answer while reading:

- how much improvement came from transfer learning?
- was ImageNet pretraining used?
- was the backbone frozen or fine-tuned?
- what conditions made transfer learning more useful?

### 4. Explainability studies

- Selvaraju et al. on Grad-CAM
- chest X-ray papers using Grad-CAM or saliency maps

Questions to answer while reading:

- what explainability method was used?
- how was interpretability evaluated?
- did the paper show both correct and incorrect predictions?
- what limitations of Grad-CAM were mentioned?

## Minimum Outputs for Phase 2

- 20 to 30 papers logged in the literature matrix
- 3 to 5 key papers marked as most relevant
- one research gap paragraph draft
- one Chapter 2 draft outline with evidence mapped to sections
