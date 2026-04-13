# DenseNet121 From-Scratch Experiment Summary

This experiment corresponds to the formal DenseNet121 from-scratch run for the NIH ChestX-ray14 binary classification task, where `No Finding` was treated as the Normal class and all other findings were grouped into the Abnormal class. The experiment name was `densenet121_scratch_full_v1`, and the model was trained from randomly initialized weights without ImageNet pretraining.

Training completed on the new Windows machine with CUDA enabled and ran the full `20` epochs. The best validation performance was obtained at epoch `20`, where the model achieved a validation AUC of `0.7355830556110341`.

## Final Test Metrics

- Accuracy: `0.6911825911171889`
- Precision: `0.6913023884632717`
- Recall: `0.5944968350342333`
- F1-score: `0.6392554521461314`
- AUC: `0.7456921168842601`
- Specificity: `0.7736285525446134`
- Loss: `0.5983914279031927`

## Interpretation

The from-scratch DenseNet121 model produced the strongest scratch baseline among the completed experiments, reaching a best validation AUC of `0.7356` and a final test AUC of `0.7457`. This result provides a clean controlled comparison against both the from-scratch ResNet50 baseline and the DenseNet121 transfer-learning configuration in the thesis results chapter.

## Run Notes

- Dataset: NIH ChestX-ray14
- Device: CUDA on the new Windows machine
- Training mode: from scratch
- Best checkpoint: `outputs/models/densenet121_scratch_full_v1/best_model.pt`
- Latest checkpoint: `outputs/models/densenet121_scratch_full_v1/last_checkpoint.pt`
- Generated artifacts include `history.csv`, `best_model.pt`, `last_checkpoint.pt`, and `metrics.json`
