# Phase 8: Grad-CAM Results

The best-performing model used for explainability was `ResNet50 + transfer learning` from `resnet50_transfer_full_v1`.

## Generated Files

- `outputs/figures/resnet50_transfer_full_v1/gradcam/correct_normal_00003468_005.png`
- `outputs/figures/resnet50_transfer_full_v1/gradcam/correct_abnormal_00015799_013.png`
- `outputs/figures/resnet50_transfer_full_v1/gradcam/false_positive_00021772_015.png`
- `outputs/figures/resnet50_transfer_full_v1/gradcam/false_negative_00004482_001.png`
- `outputs/figures/resnet50_transfer_full_v1/gradcam/gradcam_overview.png`
- `outputs/figures/resnet50_transfer_full_v1/gradcam/gradcam_summary.json`

## Selected Cases

- Correct normal: `00003468_005.png`, probability of abnormal = `0.0824`
- Correct abnormal: `00015799_013.png`, probability of abnormal = `0.8740`
- False positive: `00021772_015.png`, probability of abnormal = `0.8547`
- False negative: `00004482_001.png`, probability of abnormal = `0.0833`

## Short Interpretation

- Correct normal: the model focuses mainly on the central thoracic region, suggesting that it is relying on broad chest structure rather than on image borders or obvious artifacts.
- Correct abnormal: attention is concentrated in clinically relevant thoracic areas, with stronger activation over the right lung field where the abnormal pattern appears more prominent.
- False positive: the heatmap is diffuse and strongly weighted toward one hemithorax, indicating that the model may sometimes overreact to intensity patterns, positioning, or acquisition artifacts in otherwise normal images.
- False negative: the activation remains narrow and centered, which suggests that the model may overlook more subtle abnormal findings when they do not dominate the overall thoracic appearance.

## Use in Thesis

These four case figures are suitable for insertion into the Grad-CAM subsection of Chapter 4 because they cover:

- correct normal prediction
- correct abnormal prediction
- false positive
- false negative
