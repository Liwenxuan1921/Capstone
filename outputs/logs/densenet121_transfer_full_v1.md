# DenseNet121 Transfer Learning Experiment Summary

- Experiment name: `densenet121_transfer_full_v1`
- Dataset: NIH ChestX-ray14
- Task: Binary classification (`No Finding` = Normal, all other findings = Abnormal)
- Model: DenseNet121 with ImageNet pretrained weights
- Resume point: `last_checkpoint.pt` after epoch 7
- Training stop: Early stopping after epoch 17 (`patience = 5`)
- Best validation epoch: 12
- Best validation AUC: 0.7348006639408574

## Final Test Metrics

- Accuracy: 0.6710862714786848
- Precision: 0.7224572004028197
- Recall: 0.4633768246996512
- F1-score: 0.5646151424523848
- AUC: 0.7420875998135154
- Specificity: 0.8482044503194536
- Loss: 0.627305783654331

## Notes

- Training was resumed successfully on the new machine from epoch 8.
- The run produced new `history.csv`, `last_checkpoint.pt`, `best_model.pt`, and `metrics.json` artifacts locally.
- Cross-platform fixes were applied so the same project can resume on Windows after preparing data under WSL-style paths.
