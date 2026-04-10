# Remote Training Handoff

Use this when moving training to a second machine.

## What GitHub already contains

- Latest training code with resume support.
- Experiment log entry for `densenet121_transfer_full_v1`.
- The resume command and project structure.

## What you still need to carry separately

GitHub does not store the large checkpoint or dataset.

Copy these from the current machine to a USB drive or cloud storage:

- `data.zip`
- `outputs/models/densenet121_transfer_full_v1/last_checkpoint.pt`

Optional but useful:

- `outputs/models/densenet121_transfer_full_v1/best_model.pt`
- `outputs/logs/densenet121_transfer_full_v1/history.csv`

## Steps on the new machine

1. Clone the repo and switch to `master`.
2. Put `data.zip` under `data/raw/data.zip`.
3. Extract `data.zip` into `data/raw/`.
4. If needed, run:

```powershell
python src/prepare_nih_chestxray14.py
```

5. Copy `last_checkpoint.pt` to:

```text
outputs/models/densenet121_transfer_full_v1/last_checkpoint.pt
```

6. Install dependencies and GPU-enabled PyTorch.
7. Resume training:

```powershell
python src/train.py --model densenet121 --pretrained --device cuda --batch-size 16 --num-workers 4 --epochs 20 --patience 5 --experiment-name densenet121_transfer_full_v1 --resume-checkpoint outputs/models/densenet121_transfer_full_v1/last_checkpoint.pt
```

## Current DenseNet progress

- Experiment: `densenet121_transfer_full_v1`
- Paused after `epoch 7`
- Best validation AUC so far: `0.7297052754493267`
- Best epoch so far: `5`
- Resume checkpoint: `outputs/models/densenet121_transfer_full_v1/last_checkpoint.pt`
