# ResNet50 From-Scratch Experiment Summary

This experiment corresponds to the formal ResNet50 from-scratch run for the NIH ChestX-ray14 binary classification task, where `No Finding` was treated as the Normal class and all other findings were grouped into the Abnormal class. The experiment name was `resnet50_scratch_full_v1`, and the model was trained from randomly initialized weights without ImageNet pretraining.

Training completed on the new Windows machine with CUDA enabled and ran the full `20` epochs. The best validation performance was obtained at epoch `17`, where the model achieved a validation AUC of `0.7160515455553557`.

## Final Test Metrics

- Accuracy: `0.6731672513229087`
- Precision: `0.6747663551401869`
- Recall: `0.5596176204624725`
- F1-score: `0.6118211990678625`
- AUC: `0.7225211240343667`
- Specificity: `0.7699933906146729`
- Loss: `0.6150795524650071`

## Interpretation

The from-scratch ResNet50 model achieved a best validation AUC of `0.7161` and a final test AUC of `0.7225`, showing that the baseline architecture can learn a useful binary Normal-versus-Abnormal decision boundary even without transfer learning. Compared with the transfer-learning configuration, this run provides the needed controlled scratch baseline for the thesis comparison table.

## Run Notes

- Dataset: NIH ChestX-ray14
- Device: CUDA on the new Windows machine
- Training mode: from scratch
- Best checkpoint: `outputs/models/resnet50_scratch_full_v1/best_model.pt`
- Latest checkpoint: `outputs/models/resnet50_scratch_full_v1/last_checkpoint.pt`
- Generated artifacts include `history.csv`, `best_model.pt`, `last_checkpoint.pt`, and `metrics.json`
