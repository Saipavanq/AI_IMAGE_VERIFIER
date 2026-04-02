"""Training entry point."""

from __future__ import annotations

import argparse

from models import list_models


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train AI image classifier")
    parser.add_argument("--model", choices=list_models(), default="efficientnet")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"Training placeholder | model={args.model} epochs={args.epochs} batch_size={args.batch_size} lr={args.lr}")


if __name__ == "__main__":
    main()
