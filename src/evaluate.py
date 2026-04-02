"""Evaluation entry point."""

from __future__ import annotations

import argparse

from models import list_models


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate AI image classifier")
    parser.add_argument("--model", choices=list_models(), default="efficientnet")
    parser.add_argument("--checkpoint", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"Evaluation placeholder | model={args.model} checkpoint={args.checkpoint}")


if __name__ == "__main__":
    main()
