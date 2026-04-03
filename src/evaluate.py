"""Evaluation entry point."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import build_dataloaders
from metrics import compute_binary_metrics
from models import build_model, list_models
from utils import ensure_dir, get_device, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate AI image classifier")
    parser.add_argument("--data_root", type=str, default="data/raw")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--model", choices=list_models(), default="efficientnet")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--tta",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Average softmax with horizontal flip (often +1–2% accuracy)",
    )
    parser.add_argument("--out_dir", type=str, default="outputs")
    return parser.parse_args()


@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    amp: bool,
    fake_idx: int,
    tta: bool,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    ys: list[np.ndarray] = []
    ps: list[np.ndarray] = []

    for x, y in tqdm(loader, desc="eval", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with autocast(enabled=amp and device.type == "cuda"):
            logits = model(x)
            if tta:
                xf = torch.flip(x, dims=[3])
                logits2 = model(xf)
                probs = 0.5 * (torch.softmax(logits, dim=1) + torch.softmax(logits2, dim=1))
            else:
                probs = torch.softmax(logits, dim=1)

        prob_fake = probs[:, fake_idx].detach().float().cpu().numpy()
        ys.append(y.detach().cpu().numpy())
        ps.append(prob_fake)

    return np.concatenate(ys, axis=0), np.concatenate(ps, axis=0)


def main() -> None:
    args = parse_args()
    device = get_device(prefer_cuda=True)
    amp = bool(args.amp) and device.type == "cuda"

    ckpt = torch.load(args.checkpoint, map_location=device)
    class_to_idx: dict[str, int] = ckpt["class_to_idx"]

    # Rebuild the same loaders, but we'll only use val or test.
    _, val_loader, test_loader, ds_class_to_idx = build_dataloaders(
        data_root=args.data_root,
        model_name=args.model,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    if dict(ds_class_to_idx) != dict(class_to_idx):
        raise ValueError("Checkpoint class_to_idx does not match dataset folders.")

    if args.split == "test":
        loader = test_loader
        if loader is None:
            raise FileNotFoundError("No data/raw/test split found. Create test/REAL and test/FAKE or evaluate on --split val.")
    else:
        loader = val_loader

    model = build_model(args.model, num_classes=len(class_to_idx), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)

    fake_idx = int(class_to_idx["FAKE"])
    y_true_raw, y_score = collect_predictions(
        model, loader, device, amp=amp, fake_idx=fake_idx, tta=bool(args.tta)
    )
    # Canonical binary labels for metrics: 0 = REAL, 1 = FAKE (independent of folder sort order).
    y_true = (y_true_raw == fake_idx).astype(np.int64)

    metrics = compute_binary_metrics(y_true, y_score, threshold=0.5)

    out_dir = ensure_dir(Path(args.out_dir))
    metrics_path = out_dir / f"metrics_{args.model}_{args.split}.json"
    save_json(
        str(metrics_path),
        {
            "model": args.model,
            "split": args.split,
            "checkpoint": args.checkpoint,
            "accuracy": metrics.accuracy,
            "precision_fake": metrics.precision,
            "recall_fake": metrics.recall,
            "f1_fake": metrics.f1,
            "roc_auc": metrics.roc_auc,
            "pr_auc": metrics.pr_auc,
            "confusion": {"tn": metrics.tn, "fp": metrics.fp, "fn": metrics.fn, "tp": metrics.tp},
        },
    )

    # Confusion matrix: rows/cols are canonical REAL=0, FAKE=1
    y_pred = (y_score >= 0.5).astype(int)
    cm = np.zeros((2, 2), dtype=np.int64)
    for t, p in zip(y_true.tolist(), y_pred.tolist()):
        cm[int(t), int(p)] += 1

    plt.figure(figsize=(4.5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=["REAL", "FAKE"], yticklabels=["REAL", "FAKE"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion matrix ({args.split})")
    plt.tight_layout()
    plt.savefig(out_dir / f"confusion_{args.model}_{args.split}.png", dpi=200)
    plt.close()

    # ROC curve
    if metrics.roc_auc is not None:
        from sklearn.metrics import roc_curve

        fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=1)
        plt.figure(figsize=(4.5, 4))
        plt.plot(fpr, tpr, label=f"AUC={metrics.roc_auc:.4f}")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
        plt.xlabel("False positive rate")
        plt.ylabel("True positive rate")
        plt.title(f"ROC ({args.split})")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(out_dir / f"roc_{args.model}_{args.split}.png", dpi=200)
        plt.close()

    print(
        f"{args.split}: acc={metrics.accuracy:.4f} p={metrics.precision:.4f} "
        f"r={metrics.recall:.4f} f1={metrics.f1:.4f} roc_auc={metrics.roc_auc} pr_auc={metrics.pr_auc}"
    )
    print(f"Wrote: {metrics_path}")


if __name__ == "__main__":
    main()
