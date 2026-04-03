# AI Image Verifier (Hackathon PS-7)

Detect **AI-generated** images vs **real** photos and produce **interpretable explanations** (Grad-CAM + lightweight artifact cues).

Dataset (recommended): [CIFAKE on Kaggle](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images).

## Layout

```
data/raw/
  train/REAL/
  train/FAKE/
  val/REAL/        (optional)
  val/FAKE/        (optional)
  test/REAL/       (optional, recommended for reporting)
  test/FAKE/

models/            # saved checkpoints
outputs/           # metrics plots + explanation exports
src/
app.py             # Streamlit demo
```

## Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Run locally (Windows)

From the repo folder in **PowerShell**:

| Command | What it does |
|--------|----------------|
| `.\run_local.ps1 setup` | Creates `venv`, runs `pip install -r requirements.txt` |
| `.\run_local.ps1 smoke` | Seeds tiny demo images (if needed), **1-epoch** `cnn` train, eval, explain — verifies the pipeline |
| `.\run_local.ps1 demo` | Starts the Streamlit app (needs `models/best_<model>.pth`) |
| `.\run_local.ps1 train -- --model convnext --epochs 25` | Full training (pass any `src/train.py` args after `--`) |
| `.\run_local.ps1 eval -- --model convnext --checkpoint models/best_convnext.pth --split test` | Metrics + plots (TTA on by default) |
| `.\run_local.ps1 explore` | Counts images per class under `data/raw` |

**Cmd.exe:** `run_local.bat setup` (same commands, e.g. `run_local.bat smoke`).

After placing the real CIFAKE data under `data/raw`, use `setup` once, then `train` with your preferred model.

## Dataset download (Kaggle API)

1. Place `kaggle.json` in `C:\Users\<you>\.kaggle\`
2. Download + unzip into `data/raw/`:

```bash
kaggle datasets download -d birdy654/cifake-real-and-ai-generated-synthetic-images -p data/raw --unzip
```

Expected result: `data/raw/train/REAL`, `data/raw/train/FAKE`, and usually `data/raw/test/...`.

## Quick exploration

```bash
python src/explore.py --data_root data/raw
```

## Train

**Models** (see `src/models.py`): `cnn`, `efficientnet`, **`efficientnetv2`**, **`convnext`**, `vit`.

Recommended for **higher accuracy** than a plain ResNet/EfficientNet-B4 run:

- **`convnext`** or **`efficientnetv2`** backbone
- **`--aug strong`** (default): RandAugment + random erasing
- **`--ema`** (default): exponential moving average of weights (often stabilizes validation)
- **`--mixup 0.2`** (default): mixup regularization
- Differential LR: **`--lr_backbone_ratio 0.1`** (default)

Example (retrain from scratch on CIFAKE):

```bash
python src/train.py --model convnext --epochs 25 --batch_size 24 --lr 2e-4 --data_root data/raw
```

Lighter / faster:

```bash
python src/train.py --model efficientnet --epochs 20 --batch_size 32 --lr 1e-4 --aug default --no-ema --mixup 0
```

Other options:

```bash
python src/train.py --model cnn --epochs 30 --batch_size 128 --lr 3e-4
python src/train.py --model vit --epochs 15 --batch_size 32 --lr 5e-5
```

Notes:
- If `data/raw/val/` does not exist, the trainer automatically creates a **random train/val split** from `train/` using `--val_ratio` (default `0.1`).
- Checkpoints save to `models/best_<model>.pth` by default (EMA weights when `--ema` is on).

## Evaluate (Accuracy / Precision / Recall / F1 + ROC)

**Test-time augmentation (TTA)** (horizontal flip average) is **on by default**; disable with `--no-tta` if you need speed.

```bash
python src/evaluate.py --model convnext --checkpoint models/best_convnext.pth --split test --data_root data/raw
```

Outputs land in `outputs/` (JSON metrics + confusion matrix + ROC PNG).

## Explanations (Grad-CAM + artifact cues)

```bash
python src/explain.py --model efficientnet --checkpoint models/best_efficientnet.pth --split test --num_samples 20 --data_root data/raw
```

Writes per-image overlays and a JSON report under `outputs/explain/`.

## Streamlit demo

```bash
streamlit run app.py
```

## Efficiency tips (Windows + GPU)

- Prefer `--amp` (default **on** for CUDA) for faster training.
- If you run out of VRAM: reduce `--batch_size` (especially for `efficientnet` and `vit`).
- For fastest iteration, start with `--model cnn`.
