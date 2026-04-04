"""
dataset.py — CIFAKE DataLoader with augmentations.

Expected directory structure:
  data/raw/train/REAL/   data/raw/train/FAKE/
  data/raw/test/REAL/    data/raw/test/FAKE/
"""

import os
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode


# ── Constants ────────────────────────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
IMG_SIZE      = 224          # resize target (EfficientNet / ViT default)
CLASS_NAMES   = ["FAKE", "REAL"]   # alphabetical → matches ImageFolder label order


# ── Transforms ───────────────────────────────────────────────────────────────
def get_train_transform(img_size: int = IMG_SIZE) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.1),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_val_transform(img_size: int = IMG_SIZE) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ── Dataset builders ─────────────────────────────────────────────────────────
def get_datasets(data_dir: str = "data/raw", val_split: float = 0.1, img_size: int = IMG_SIZE):
    """Return (train_ds, val_ds, test_ds) PyTorch datasets."""
    data_dir = Path(data_dir)
    train_dir = data_dir / "train"
    test_dir  = data_dir / "test"

    # Two ImageFolders over the same directory — different transforms
    train_full = datasets.ImageFolder(root=str(train_dir), transform=get_train_transform(img_size))
    val_full   = datasets.ImageFolder(root=str(train_dir), transform=get_val_transform(img_size))

    # Split indices deterministically
    n         = len(train_full)
    val_size  = int(n * val_split)
    train_size = n - val_size
    indices   = torch.randperm(n, generator=torch.Generator().manual_seed(42)).tolist()
    train_idx = indices[:train_size]
    val_idx   = indices[train_size:]

    train_ds = torch.utils.data.Subset(train_full, train_idx)
    val_ds   = torch.utils.data.Subset(val_full,   val_idx)
    test_ds  = datasets.ImageFolder(root=str(test_dir), transform=get_val_transform(img_size))

    print(f"[Dataset] Train: {len(train_ds)}  Val: {len(val_ds)}  Test: {len(test_ds)}")
    print(f"[Dataset] Classes: {train_full.classes}  (0=FAKE, 1=REAL)")
    return train_ds, val_ds, test_ds


def get_dataloaders(
    data_dir: str = "data/raw",
    batch_size: int = 64,
    num_workers: int = 4,
    val_split: float = 0.1,
    img_size: int = IMG_SIZE,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Return (train_loader, val_loader, test_loader)."""
    train_ds, val_ds, test_ds = get_datasets(data_dir, val_split, img_size)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader
