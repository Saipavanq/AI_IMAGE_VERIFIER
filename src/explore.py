"""Lightweight dataset exploration (counts + quick class balance)."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Explore CIFAKE-style folder dataset")
    p.add_argument("--data_root", type=str, default="data/raw", help="Folder containing train/ (+ optional val/, test/)")
    return p.parse_args()


def count_images(class_dir: Path) -> int:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    n = 0
    for p in class_dir.iterdir():
        if p.is_file() and p.suffix.lower() in exts:
            n += 1
    return n


def main() -> None:
    args = parse_args()
    base = Path(args.data_root)

    for split in ["train", "val", "test"]:
        d = base / split
        if not d.exists():
            continue
        print(f"== {d} ==")
        for cls_dir in sorted([p for p in d.iterdir() if p.is_dir()]):
            total = count_images(cls_dir)
            print(f"{cls_dir.name}: {total}")
        print("")


if __name__ == "__main__":
    main()
