# Prediction Artifacts Summary

## Sanity Check

- All four generated prediction CSV files contain `16819` test samples.
- The test split matches `data/processed/nih_chestxray14/test.csv`.
- `true_label` is binary: `0 = Normal / No Finding`, `1 = Abnormal / any pathology`.
- `predicted_probability` is the sigmoid probability for the abnormal class.
- `predicted_label` uses threshold `0.5`.
- AUC values approximately match the thesis experiment table.

## Experiment Details

### resnet50_scratch_full_v1

- Checkpoint used: `outputs/models/resnet50_scratch_full_v1/best_model.pt`
- Prediction CSV: `outputs/predictions/resnet50_scratch_full_v1_test_predictions.csv`
- Number of test samples evaluated: `16819`
- AUC from prediction CSV: `0.722521`
- Accuracy: `0.673167`
- Precision: `0.674766`
- Recall: `0.559618`
- F1-score: `0.611821`
- Sensitivity: `0.559618`
- Specificity: `0.769993`
- Thesis reference AUC: `0.722521`
- Match status: exact match to the existing thesis metrics.

### resnet50_transfer_full_v1

- Checkpoint used: `outputs/models/resnet50_transfer_full_v1/best_model.pt`
- Prediction CSV: `outputs/predictions/resnet50_transfer_full_v1_test_predictions.csv`
- Number of test samples evaluated: `16819`
- AUC from prediction CSV: `0.745422`
- Accuracy: `0.695404`
- Precision: `0.684626`
- Recall: `0.627051`
- F1-score: `0.654575`
- Sensitivity: `0.627051`
- Specificity: `0.753690`
- Thesis reference AUC: `0.745423`
- Match status: close match to the existing thesis metrics; AUC differs by less than `0.000001`, and threshold metrics differ only at the one-sample level.

### densenet121_scratch_full_v1

- Checkpoint used: `outputs/models/densenet121_scratch_full_v1/best_model.pt`
- Prediction CSV: `outputs/predictions/densenet121_scratch_full_v1_test_predictions.csv`
- Number of test samples evaluated: `16819`
- AUC from prediction CSV: `0.745692`
- Accuracy: `0.691183`
- Precision: `0.691302`
- Recall: `0.594497`
- F1-score: `0.639255`
- Sensitivity: `0.594497`
- Specificity: `0.773629`
- Thesis reference AUC: `0.745692`
- Match status: exact match to the existing thesis metrics.

### densenet121_transfer_full_v1

- Checkpoint used: `outputs/models/densenet121_transfer_full_v1/best_model.pt`
- Prediction CSV: `outputs/predictions/densenet121_transfer_full_v1_test_predictions.csv`
- Number of test samples evaluated: `16819`
- AUC from prediction CSV: `0.742088`
- Accuracy: `0.671086`
- Precision: `0.722457`
- Recall: `0.463377`
- F1-score: `0.564615`
- Sensitivity: `0.463377`
- Specificity: `0.848204`
- Thesis reference AUC: `0.742088`
- Match status: exact match to the existing thesis metrics.
