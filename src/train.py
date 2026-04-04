"""
train.py — Training loop for Real vs AI-Generated image classification.

Usage:
  python src/train.py --model efficientnet --epochs 20 --batch_size 64 --lr 1e-4
  python src/train.py --model cnn          --epochs 30 --batch_size 128 --lr 3e-4
  python src/train.py --model vit          --epochs 15 --batch_size 32  --lr 5e-5
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import GradScaler, autocast
from sklearn.metrics import f1_score, accuracy_score

# local imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.dataset import get_dataloaders
from src.models import build_model, count_parameters


# ── CLI ──────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Train real/fake image classifier")
    p.add_argument("--model",       type=str,   default="efficientnet",
                   choices=["cnn", "efficientnet", "vit"])
    p.add_argument("--data_dir",    type=str,   default="data/raw")
    p.add_argument("--epochs",      type=int,   default=20)
    p.add_argument("--batch_size",  type=int,   default=64)
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--weight_decay",type=float, default=1e-4)
    p.add_argument("--img_size",    type=int,   default=224)
    p.add_argument("--num_workers", type=int,   default=4)
    p.add_argument("--val_split",   type=float, default=0.1)
    p.add_argument("--dropout",     type=float, default=0.4)
    p.add_argument("--save_dir",    type=str,   default="models")
    p.add_argument("--no_amp",      action="store_true", help="Disable mixed precision")
    return p.parse_args()


# ── Training utilities ───────────────────────────────────────────────────────
def run_epoch(model, loader, criterion, optimizer, scaler, device, train: bool):
    model.train(train)
    total_loss, all_preds, all_labels = 0.0, [], []

    for imgs, labels in loader:
        imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        with autocast("cuda", enabled=(scaler is not None)):
            logits = model(imgs)
            loss   = criterion(logits, labels)

        if train:
            optimizer.zero_grad(set_to_none=True)
            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    n    = len(loader.dataset)
    loss = total_loss / n
    acc  = accuracy_score(all_labels, all_preds)
    f1   = f1_score(all_labels, all_preds, average="binary")
    return loss, acc, f1


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Device: {device}  Model: {args.model}")

    # Directories
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    log_path = save_dir / f"{args.model}_history.json"

    # Data
    train_loader, val_loader, _ = get_dataloaders(
        data_dir    = args.data_dir,
        batch_size  = args.batch_size,
        num_workers = args.num_workers,
        val_split   = args.val_split,
        img_size    = args.img_size,
    )

    # Model
    model = build_model(args.model, dropout=args.dropout).to(device)
    print(f"[Model] Trainable parameters: {count_parameters(model):,}")

    # Optimizer & scheduler
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    scaler    = GradScaler("cuda") if (not args.no_amp and device.type == "cuda") else None

    best_f1   = 0.0
    history   = []
    ckpt_path = save_dir / f"best_{args.model}.pth"

    print(f"{'Epoch':>5} | {'Train Loss':>10} {'Train Acc':>9} {'Val Loss':>8} {'Val Acc':>7} {'Val F1':>6}")
    print("-" * 60)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        tr_loss, tr_acc, tr_f1 = run_epoch(
            model, train_loader, criterion, optimizer, scaler, device, train=True)
        val_loss, val_acc, val_f1 = run_epoch(
            model, val_loader, criterion, None, None, device, train=False)

        scheduler.step()
        elapsed = time.time() - t0

        row = dict(epoch=epoch, tr_loss=tr_loss, tr_acc=tr_acc, tr_f1=tr_f1,
                   val_loss=val_loss, val_acc=val_acc, val_f1=val_f1, time=elapsed)
        history.append(row)

        print(f"{epoch:>5} | {tr_loss:>10.4f} {tr_acc:>9.4f} {val_loss:>8.4f} {val_acc:>7.4f} {val_f1:>6.4f}  [{elapsed:.0f}s]")

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save({
                "epoch": epoch,
                "model_name": args.model,
                "state_dict": model.state_dict(),
                "val_f1": val_f1,
                "val_acc": val_acc,
                "args": vars(args),
            }, ckpt_path)
            print(f"  ✔ Saved best checkpoint (val_f1={val_f1:.4f}) → {ckpt_path}")

    # Save training history
    with open(log_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n[Done] Best Val F1: {best_f1:.4f}  Checkpoint: {ckpt_path}")
    print(f"[Done] Training history saved to: {log_path}")


if __name__ == "__main__":
    main()
