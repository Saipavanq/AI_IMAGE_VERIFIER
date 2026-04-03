"""Training entry point."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn as nn
from timm.data import resolve_data_config
from timm.utils import ModelEmaV2
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from dataset import build_dataloaders, timm_model_name
from models import build_model, list_models
from utils import ensure_dir, get_device, save_json, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train AI image classifier (real vs AI-generated)")
    parser.add_argument("--data_root", type=str, default="data/raw", help="Folder containing train/ (+ optional val/, test/)")
    parser.add_argument("--model", choices=list_models(), default="convnext", help="convnext / efficientnetv2 recommended for accuracy")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_backbone_ratio", type=float, default=0.1, help="LR multiplier for backbone vs classification head")
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--label_smoothing", type=float, default=0.05, help="Override timm default; 0 disables")
    parser.add_argument(
        "--aug",
        choices=["default", "strong"],
        default="strong",
        help="strong: RandAugment + erasing (better generalization; needs enough epochs)",
    )
    parser.add_argument("--mixup", type=float, default=0.2, help="Mixup alpha; 0 disables")
    parser.add_argument("--ema", action=argparse.BooleanOptionalAction, default=True, help="EMA weights for validation & checkpoint")
    parser.add_argument("--ema_decay", type=float, default=0.9999)
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


def split_head_backbone(model: nn.Module) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    """Split timm-style models into classification head vs rest (for differential LR)."""
    head: list[nn.Parameter] = []
    backbone: list[nn.Parameter] = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        n = name.lower()
        is_head = (
            n.startswith("head")
            or ".head." in n
            or "classifier" in n
            or n.startswith("fc.")
            or n in ("fc.weight", "fc.bias")
        )
        if is_head:
            head.append(p)
        else:
            backbone.append(p)
    return backbone, head


def mixup_batch(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, bool]:
    if alpha <= 0 or x.size(0) < 2:
        return x, y, y, 1.0, False
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(x.size(0), device=x.device)
    x_mix = lam * x + (1.0 - lam) * x[idx]
    return x_mix, y, y[idx], lam, True


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
        aug_strength=args.aug,
    )

    model = build_model(args.model, num_classes=len(class_to_idx), pretrained=True)
    model.to(device)

    cfg = resolve_data_config({}, model=timm.create_model(timm_model_name(args.model), pretrained=False))
    ls = float(cfg.get("label_smoothing", 0.0) or 0.0)
    if args.label_smoothing >= 0:
        ls = args.label_smoothing

    ce = nn.CrossEntropyLoss(label_smoothing=ls)

    backbone_params, head_params = split_head_backbone(model)
    if not head_params:
        optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        max_lrs = args.lr
    else:
        optim = torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": args.lr * args.lr_backbone_ratio},
                {"params": head_params, "lr": args.lr},
            ],
            weight_decay=args.weight_decay,
        )
        max_lrs = [args.lr * args.lr_backbone_ratio, args.lr]

    steps_per_epoch = max(len(train_loader), 1)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optim,
        max_lr=max_lrs,
        epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.1,
        div_factor=25.0,
        final_div_factor=1e4,
    )

    scaler = GradScaler(enabled=amp)

    ema: ModelEmaV2 | None = None
    if args.ema:
        ema = ModelEmaV2(model, decay=args.ema_decay, device=None)

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
        running_samples = 0
        running_correct = 0
        running_acc_samples = 0

        pbar = tqdm(train_loader, desc=f"train {epoch}/{args.epochs}", leave=False)
        for x, y in pbar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            mixed = False
            if args.mixup > 0:
                x, y_a, y_b, lam, mixed = mixup_batch(x, y, args.mixup)

            optim.zero_grad(set_to_none=True)

            with autocast(enabled=amp):
                logits = model(x)
                if mixed:
                    loss = lam * ce(logits, y_a) + (1.0 - lam) * ce(logits, y_b)
                else:
                    loss = ce(logits, y)

            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            scheduler.step()

            if ema is not None:
                ema.update(model)

            bs = int(x.size(0))
            running_loss += float(loss.item()) * bs
            running_samples += bs

            if not mixed:
                preds = logits.argmax(dim=1)
                running_correct += int((preds == y).sum().item())
                running_acc_samples += bs

            pbar.set_postfix(loss=float(loss.item()))

        train_loss = running_loss / max(running_samples, 1)
        train_acc = running_correct / max(running_acc_samples, 1) if running_acc_samples else 0.0

        eval_net = ema.module if ema is not None else model
        val_loss, val_acc = evaluate(eval_net, val_loader, device, amp=amp)

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

            save_dict = ema.module.state_dict() if ema is not None else model.state_dict()
            payload = {
                "model_name": args.model,
                "class_to_idx": class_to_idx,
                "state_dict": save_dict,
                "epoch": epoch,
                "val_acc": best_val_acc,
                "train_args": vars(args),
                "ema_used": bool(ema),
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
