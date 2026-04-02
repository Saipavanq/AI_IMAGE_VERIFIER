"""Explanation/Grad-CAM entry point."""

from __future__ import annotations

import argparse

from gradcam import generate_gradcam_stub
from models import list_models


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate explanations")
    parser.add_argument("--model", choices=list_models(), default="efficientnet")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--num_samples", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = generate_gradcam_stub()
    print(
        "Explanation placeholder | "
        f"model={args.model} checkpoint={args.checkpoint} num_samples={args.num_samples} output={output_dir}"
    )


if __name__ == "__main__":
    main()
