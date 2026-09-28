# APTOS / EfficientNet-B0

Python 3.10, isolated `.venv-train`, NVIDIA GTX 1650 Ti (4 GB).
ImageNet initialization is fine-tuned on APTOS; this is not random initialization.
The previous TensorFlow/Xception notebook is retained as historical reference.

## Install

```powershell
python -m venv .venv-train
.\.venv-train\Scripts\python -m pip install --upgrade pip
.\.venv-train\Scripts\python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
.\.venv-train\Scripts\python -m pip install -r requirements-train.txt
```

## Prepare and train

```powershell
.\.venv-train\Scripts\python prepare_data.py --data C:\Users\erkan\Downloads\aptos2019-blindness-detection
.\.venv-train\Scripts\python train.py --smoke
.\.venv-train\Scripts\python train.py --epochs 20 --batch-size 8
# Resume only an interrupted run:
.\.venv-train\Scripts\python train.py --epochs 20 --batch-size 8 --resume
.\.venv-train\Scripts\python run.py
```

Set the PyCharm interpreter to `.venv-train/Scripts/python.exe`.
Use the project root as working directory. No paid/cloud training is used.

## Protocol

- Original downloads remain untouched. Audited 320px crops are cached under `data/`.
- RGB, dark-border cropping, square padding, ImageNet normalization; shared inference preprocessing.
- Exact decoded-pixel duplicates are removed; conflicting labels excluded.
- Stratified 70/15/15 split, seed 42, saved CSV manifests. Patient IDs are absent;
  same-patient and near-duplicate leakage cannot be ruled out.
- Augmentation only in training: horizontal flips, mild rotation, brightness/contrast.
- Head warmup for 2 epochs, then full fine-tuning; frozen batch-normalization statistics.
- Square-root inverse class-frequency weights, AdamW, gradient clipping, mixed precision.
- Best checkpoint selected by validation quadratic weighted kappa; early stopping.
- Temperature fitted on validation only. Fixed 0.8 threshold is exploratory, not a guarantee.
- Held-out test evaluated after selection, including macro F1, per-class recall,
  confusion matrix, Kappa, coverage and accepted-subset accuracy.
- Competition `test.csv` is unlabeled and never treated as labeled evaluation data.

## Artifacts

- `data/audit.json`: source validation and split counts.
- `runs/efficientnet_b0/history.jsonl`: per-epoch validation results.
- `runs/efficientnet_b0/last.pt`: interruption recovery.
- `runs/efficientnet_b0/best.pt`: best validation checkpoint.
- `runs/efficientnet_b0/test_metrics.json`: final held-out evaluation.
- `models/aptos_efficientnet_b0.pt`: application model, exported only after training.

Research prototype, not validated for clinical diagnosis. A high softmax score is
not proof of correctness. No comparative improvement is claimed before evaluation.
Training may take hours on this GPU. If interrupted, use `--resume`; exact RNG
continuation is not guaranteed, but data splits and model/optimizer states persist.
