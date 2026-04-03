"""Training entry point."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import timm
import torch
import torch.nn as nn
from timm.data import resolve_data_config
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from dataset import build_dataloaders, timm_model_name
from models import build_model, list_models
from utils import ensure_dir, get_device, save_json, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train AI image classifier (real vs AI-generated)")
    parser.add_argument("--data_root", type=str, default="data/raw", help="Folder containing train/ (+ optional val/, test/)")
    parser.add_argument("--model", choices=list_models(), default="efficientnet")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Used only if data/raw/val does not exist")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True, help="Mixed precision (CUDA)")
    parser.add_argument("--early_stop", type=int, default=0, help="Stop if no val improvement for N epochs (0 disables)")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="",
        help="Where to save the best checkpoint (default: models/best_<model>.pth)",
    )
    return parser.parse_args()


@torch.no_grad()
def evaluate(model: nn.Module, loader, device: torch.device, *, amp: bool) -> tuple[float, float]:
    model.eval()
    ce = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with autocast(enabled=amp and device.type == "cuda"):
            logits = model(x)
            loss = ce(logits, y)

        preds = logits.argmax(dim=1)
        total_correct += int((preds == y).sum().item())
        total += int(y.numel())
        total_loss += float(loss.item()) * int(y.numel())

    return total_loss / max(total, 1), total_correct / max(total, 1)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = get_device(prefer_cuda=True)
    amp = bool(args.amp) and device.type == "cuda"

    train_loader, val_loader, test_loader, class_to_idx = build_dataloaders(
        data_root=args.data_root,
        model_name=args.model,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    model = build_model(args.model, num_classes=len(class_to_idx), pretrained=True)
    model.to(device)

    # Use timm preprocessing defaults for label smoothing-friendly training.
    cfg = resolve_data_config({}, model=timm.create_model(timm_model_name(args.model), pretrained=False))
    label_smoothing = float(cfg.get("label_smoothing", 0.0) or 0.0)

    ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # One-cycle is usually efficient and stable for transfer learning.
    steps_per_epoch = max(len(train_loader), 1)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optim,
        max_lr=args.lr,
        epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.1,
        div_factor=25.0,
        final_div_factor=1e4,
    )

    scaler = GradScaler(enabled=amp)

    ensure_dir("models")
    ckpt_path = Path(args.checkpoint_path) if args.checkpoint_path else Path("models") / f"best_{args.model}.pth"

    best_val_acc = -1.0
    best_epoch = -1
    epochs_without_improve = 0

    history: dict[str, list[float | int]] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()

        running_loss = 0.0
        running_correct = 0
        running_total = 0

        pbar = tqdm(train_loader, desc=f"train {epoch}/{args.epochs}", leave=False)
        for x, y in pbar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optim.zero_grad(set_to_none=True)

            with autocast(enabled=amp):
                logits = model(x)
                loss = ce(logits, y)

            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            scheduler.step()

            preds = logits.argmax(dim=1)
            running_correct += int((preds == y).sum().item())
            running_total += int(y.numel())
            running_loss += float(loss.item()) * int(y.numel())

            pbar.set_postfix(loss=float(loss.item()))

        train_loss = running_loss / max(running_total, 1)
        train_acc = running_correct / max(running_total, 1)

        val_loss, val_acc = evaluate(model, val_loader, device, amp=amp)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        dt = time.time() - t0
        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
            f"time={dt:.1f}s"
        )

        if val_acc > best_val_acc + 1e-6:
            best_val_acc = float(val_acc)
            best_epoch = epoch
            epochs_without_improve = 0

            payload = {
                "model_name": args.model,
                "class_to_idx": class_to_idx,
                "state_dict": model.state_dict(),
                "epoch": epoch,
                "val_acc": best_val_acc,
                "train_args": vars(args),
            }
            torch.save(payload, ckpt_path)
            meta = dict(vars(args))
            meta.update({"best_epoch": epoch, "best_val_acc": best_val_acc})
            save_json(str(Path("models") / f"{args.model}_train_meta.json"), meta)
        else:
            epochs_without_improve += 1

        if args.early_stop and epochs_without_improve >= args.early_stop:
            print(f"Early stopping: no val improvement for {epochs_without_improve} epochs")
            break

    print(f"Done. Best val_acc={best_val_acc:.4f} at epoch={best_epoch}. Saved: {ckpt_path}")

    if test_loader is not None:
        best = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(best["state_dict"])
        test_loss, test_acc = evaluate(model, test_loader, device, amp=amp)
        print(f"Held-out test (using best checkpoint): loss={test_loss:.4f} acc={test_acc:.4f}")
        ensure_dir("outputs")
        save_json(
            str(Path("outputs") / f"{args.model}_test_quick.json"),
            {"checkpoint": str(ckpt_path), "test_loss": test_loss, "test_acc": test_acc},
        )


if __name__ == "__main__":
    main()
