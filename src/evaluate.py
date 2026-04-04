"""
evaluate.py — Evaluate a trained model on the CIFAKE test set.

Usage:
  python src/evaluate.py --model efficientnet --checkpoint models/best_efficientnet.pth
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve,
    classification_report,
)
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.dataset import get_dataloaders
from src.models import build_model

CLASS_NAMES = ["FAKE", "REAL"]


# ── CLI ──────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Evaluate real/fake classifier")
    p.add_argument("--model",      type=str, default="efficientnet",
                   choices=["cnn", "efficientnet", "vit"])
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--data_dir",   type=str, default="data/raw")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--img_size",   type=int, default=224)
    p.add_argument("--num_workers",type=int, default=4)
    p.add_argument("--output_dir", type=str, default="outputs")
    return p.parse_args()


# ── Inference pass ────────────────────────────────────────────────────────────
@torch.no_grad()
def run_inference(model, loader, device):
    model.eval()
    all_preds, all_probs, all_labels = [], [], []

    for imgs, labels in tqdm(loader, desc="Evaluating"):
        imgs = imgs.to(device, non_blocking=True)
        logits = model(imgs)
        probs  = torch.softmax(logits, dim=1)[:, 1]   # prob of REAL
        preds  = logits.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(labels.numpy())

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


# ── Plots ─────────────────────────────────────────────────────────────────────
def plot_confusion_matrix(y_true, y_pred, out_path: Path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[Eval] Confusion matrix saved → {out_path}")


def plot_roc_curve(y_true, y_prob, auc: float, out_path: Path):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve"); ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[Eval] ROC curve saved → {out_path}")


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device)
    model_name = ckpt.get("model_name", args.model)
    model = build_model(model_name).to(device)
    model.load_state_dict(ckpt["state_dict"])
    print(f"[Eval] Loaded '{model_name}' from {args.checkpoint}  (epoch {ckpt.get('epoch', '?')})")

    # Data
    _, _, test_loader = get_dataloaders(
        data_dir    = args.data_dir,
        batch_size  = args.batch_size,
        num_workers = args.num_workers,
        img_size    = args.img_size,
    )

    # Inference
    y_true, y_pred, y_prob = run_inference(model, test_loader, device)

    # Metrics
    acc   = accuracy_score(y_true, y_pred)
    prec  = precision_score(y_true, y_pred)
    rec   = recall_score(y_true, y_pred)
    f1    = f1_score(y_true, y_pred)
    auc   = roc_auc_score(y_true, y_prob)

    print("\n" + "=" * 50)
    print(f"  Model      : {model_name}")
    print(f"  Accuracy   : {acc:.4f} ({acc*100:.2f}%)")
    print(f"  Precision  : {prec:.4f}")
    print(f"  Recall     : {rec:.4f}")
    print(f"  F1-Score   : {f1:.4f}")
    print(f"  AUC-ROC    : {auc:.4f}")
    print("=" * 50)
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

    # Save results
    results = dict(model=model_name, accuracy=acc, precision=prec,
                   recall=rec, f1=f1, auc=auc)
    import json
    with open(out_dir / f"metrics_{model_name}.json", "w") as fp:
        json.dump(results, fp, indent=2)

    plot_confusion_matrix(y_true, y_pred, out_dir / f"confusion_matrix_{model_name}.png")
    plot_roc_curve(y_true, y_prob, auc, out_dir / f"roc_curve_{model_name}.png")

    print(f"\n[Eval] Done. Results in: {out_dir}/")


if __name__ == "__main__":
    main()
