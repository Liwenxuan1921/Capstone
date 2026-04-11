# DenseNet121 Transfer Learning Experiment Summary

This experiment corresponds to the formal DenseNet121 transfer learning run for the NIH ChestX-ray14 binary classification task, where `No Finding` was treated as the Normal class and all other findings were grouped into the Abnormal class. The experiment name was `densenet121_transfer_full_v1`, and the model was initialized with ImageNet-pretrained DenseNet121 weights.

Training was resumed successfully from `last_checkpoint.pt` after epoch 7 on the new machine. The run continued from epoch 8 and stopped early at epoch 17 after reaching the configured early stopping criterion (`patience = 5`). The best validation performance was obtained at epoch 12, where the model achieved a validation AUC of `0.7348006639408574`.

## Final Test Metrics

- Accuracy: `0.6710862714786848`
- Precision: `0.7224572004028197`
- Recall: `0.4633768246996512`
- F1-score: `0.5646151424523848`
- AUC: `0.7420875998135154`
- Specificity: `0.8482044503194536`
- Loss: `0.627305783654331`

## Interpretation

The resumed DenseNet121 model delivered stable performance after migration to the new device and produced a slight improvement over the previously recorded validation checkpoint. The final test AUC of `0.7421` indicates that the transfer learning configuration remained effective for distinguishing Normal and Abnormal chest X-rays under the binary formulation used in this study.

## Run Notes

- Dataset: NIH ChestX-ray14
- Resume checkpoint: `outputs/models/densenet121_transfer_full_v1/last_checkpoint.pt`
- New artifacts generated during this run: `history.csv`, `last_checkpoint.pt`, `best_model.pt`, and `metrics.json`
- Cross-platform compatibility fixes were applied so the same experiment could be resumed correctly on Windows after data preparation under WSL-style paths
