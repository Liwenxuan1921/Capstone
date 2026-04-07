# Chapter 2 Writing Outline

## Chapter Title

Literature Review

## Writing Goal

This chapter should not be a list of paper summaries. It should show:

- what is already known
- what methods are commonly used
- what is still missing
- why your capstone design is justified

## Section 2.1: Deep Learning for Chest X-ray Classification

Use papers from these categories:

- chest_xray_classification
- survey

What to write:

- why chest X-ray classification matters
- which public datasets are commonly used
- how problem settings differ: binary, multi-class, multi-label
- what performance trends appear in the literature
- what limitations still appear across studies

Target paragraph flow:

1. introduce chest X-ray classification as a core medical imaging task
2. summarize commonly used datasets and benchmarks
3. compare common modeling choices
4. note recurring limitations such as label noise and dataset bias

## Section 2.2: ResNet and DenseNet in Medical Imaging

Use papers from these categories:

- backbone_models
- chest_xray_classification

What to write:

- why ResNet is important
- why DenseNet is important
- why these architectures are reasonable baselines for your capstone
- how prior chest X-ray work has used them

Target paragraph flow:

1. summarize ResNet
2. summarize DenseNet
3. compare their strengths in medical image classification
4. justify why your thesis selects them

## Section 2.3: Transfer Learning in Medical Image Analysis

Use papers from these categories:

- transfer_learning
- chest_xray_classification

What to write:

- why transfer learning is common in medical imaging
- how ImageNet-pretrained weights are typically used
- when transfer learning helps and when it may not
- what gap remains in controlled comparisons for your chosen task

Target paragraph flow:

1. define transfer learning in your context
2. summarize empirical benefits found by prior work
3. discuss limitations and conditions
4. connect directly to your experiment matrix

## Section 2.4: Explainable AI for Chest X-ray Models

Use papers from these categories:

- explainability
- chest_xray_classification

What to write:

- why interpretability matters in medical AI
- what Grad-CAM does
- how prior papers use explainability in chest X-ray studies
- what limitations remain in explainability claims

Target paragraph flow:

1. motivate trustworthy AI in medicine
2. explain Grad-CAM at a high level
3. summarize how prior papers present visual explanations
4. explain why Grad-CAM is appropriate for your capstone

## Section 2.5: Research Gap

Use evidence from all sections above.

What to write:

- many papers show strong results, but they often focus on accuracy only
- fewer small-scope studies combine:
  - a clean binary chest X-ray task
  - two standard backbones
  - scratch vs. transfer learning comparison
  - focused Grad-CAM analysis

Your gap statement should lead directly to your methodology.

## Section 2.6: Summary

Keep this short.

What to include:

- chest X-ray classification is well studied but still limited by data quality and interpretability concerns
- ResNet50 and DenseNet121 are justified baselines
- transfer learning is important but should be tested on your exact setup
- Grad-CAM provides a practical explainability layer
- therefore your capstone design is justified
