# Phase 5: Scratch Experiment Preparation

The training pipeline has been checked for both scratch models using the smoke subset at `data/processed/nih_chestxray14_smoke`.

## Smoke Test Status

- `resnet50_scratch_smoke`: completed successfully
- `densenet121_scratch_smoke`: completed successfully

These smoke runs confirm that:

- the dataset loader works with scratch training
- checkpoints, metrics, and figures are generated correctly
- the current CUDA environment can run both models

## Recommended Full Experiments

### 1. ResNet50 from scratch

```powershell
python src/train.py --model resnet50 --device cuda --batch-size 16 --num-workers 4 --epochs 20 --patience 5 --experiment-name resnet50_scratch_full_v1
```

### 2. DenseNet121 from scratch

```powershell
python src/train.py --model densenet121 --device cuda --batch-size 16 --num-workers 4 --epochs 20 --patience 5 --experiment-name densenet121_scratch_full_v1
```

## Expected Output Locations

For each experiment, the following should be created automatically:

- `outputs/models/<experiment>/best_model.pt`
- `outputs/models/<experiment>/last_checkpoint.pt`
- `outputs/logs/<experiment>/history.csv`
- `outputs/results/<experiment>/config.json`
- `outputs/results/<experiment>/metrics.json`
- `outputs/figures/<experiment>/training_history.png`
- `outputs/figures/<experiment>/test_confusion_matrix.png`
- `outputs/figures/<experiment>/test_roc_curve.png`

## Suggested Execution Order

1. Run `resnet50_scratch_full_v1`
2. Record metrics
3. Run `densenet121_scratch_full_v1`
4. Update Chapter 4 final comparison table

## Notes

- Scratch training is expected to be slower and less stable than transfer learning.
- If interruption is needed, resume from `last_checkpoint.pt`.
- Keep the same batch size, learning rate, and patience settings as the transfer-learning runs so the comparison remains controlled.
