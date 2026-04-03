"""Dataset utilities for AI image verifier."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import timm
import torch
from timm.data import resolve_data_config
from torch.utils.data import DataLoader, Subset, random_split
from torchvision.datasets import ImageFolder
from torchvision import transforms as T


@dataclass(frozen=True)
class DataRoots:
    train: Path
    val: Path | None
    test: Path | None


def default_data_roots(data_root: str | Path) -> DataRoots:
    root = Path(data_root)
    train = root / "train"
    val = root / "val"
    test = root / "test"
    return DataRoots(train=train, val=val if val.exists() else None, test=test if test.exists() else None)


def _assert_imagefolder(root: Path) -> None:
    if not root.exists():
        raise FileNotFoundError(f"Expected dataset folder not found: {root}")
    # ImageFolder requires class subfolders.
    subs = [p for p in root.iterdir() if p.is_dir()]
    if not subs:
        raise FileNotFoundError(f"No class subfolders found under: {root}")


def build_transforms(*, model_name: str, is_training: bool) -> T.Compose:
    """Build input transforms using timm's per-model preprocessing defaults."""
    # We only need cfg fields (mean/std/crop/interpolation); the model weights are irrelevant here.
    m = timm.create_model(timm_model_name(model_name), pretrained=False)
    cfg = resolve_data_config(m.pretrained_cfg, model=m)

    mean = cfg["mean"]
    std = cfg["std"]
    input_size = cfg["input_size"]  # (C, H, W)

    interpolation = cfg.get("interpolation", "bicubic")
    interp = T.InterpolationMode.BICUBIC
    if str(interpolation).lower() in {"bilinear"}:
        interp = T.InterpolationMode.BILINEAR

    crop_pct = float(cfg.get("crop_pct", 1.0))

    h = int(input_size[-2])
    w = int(input_size[-1])
    if h != w:
        # Extremely rare for these models; keep it simple by using the larger side as square.
        size = max(h, w)
    else:
        size = h

    if is_training:
        return T.Compose(
            [
                T.RandomResizedCrop(size, scale=(0.7, 1.0), interpolation=interp),
                T.RandomHorizontalFlip(),
                T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std),
            ]
        )

    # Eval transforms: resize shorter side then center crop (timm-style).
    resize = int(round(size / crop_pct))
    return T.Compose(
        [
            T.Resize(resize, interpolation=interp),
            T.CenterCrop(size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]
    )


def _make_loader(
    dataset: ImageFolder,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    persistent_workers: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers and num_workers > 0,
        drop_last=False,
    )


def build_dataloaders(
    *,
    data_root: str | Path,
    model_name: str,
    batch_size: int,
    num_workers: int,
    val_ratio: float,
    seed: int,
) -> tuple[DataLoader, DataLoader, DataLoader | None, dict[str, str]]:
    """Create train/val/test loaders from a CIFAKE-style folder layout."""
    roots = default_data_roots(data_root)

    _assert_imagefolder(roots.train)
    train_tf = build_transforms(model_name=model_name, is_training=True)
    eval_tf = build_transforms(model_name=model_name, is_training=False)

    train_ds = ImageFolder(str(roots.train), transform=train_tf)

    class_to_idx: dict[str, int] = dict(train_ds.class_to_idx)
    # ImageFolder sorts class names; we keep that mapping for checkpoints.

    if roots.val is not None:
        _assert_imagefolder(roots.val)
        val_ds = ImageFolder(str(roots.val), transform=eval_tf)
        # Ensure consistent label mapping with train.
        if dict(val_ds.class_to_idx) != class_to_idx:
            raise ValueError(
                "Train/val class folders do not match. "
                f"train={train_ds.class_to_idx} val={val_ds.class_to_idx}"
            )
        train_loader = _make_loader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
        )
        val_loader = _make_loader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
        )
    else:
        if not (0.0 < val_ratio < 0.5):
            raise ValueError("val_ratio must be between 0 and 0.5 when no val/ split is provided.")

        n_total = len(train_ds)
        n_val = int(round(n_total * val_ratio))
        n_train = n_total - n_val
        if n_train <= 0 or n_val <= 0:
            raise ValueError("Dataset too small for the requested val_ratio.")

        g = torch.Generator().manual_seed(seed)  # type: ignore[name-defined]
        train_subset, val_subset = random_split(train_ds, [n_train, n_val], generator=g)

        # Eval transforms on val: rebuild dataset by swapping transform on underlying samples.
        # random_split returns Subset; easiest path is separate ImageFolder with eval_tf for val split indices:
        # To keep it simple + efficient, we wrap subsets but replace transform on val by re-instantiating val_ds:
        # Here we use two ImageFolders (same root) but different transforms + Subset indices.
        train_only = ImageFolder(str(roots.train), transform=train_tf)
        val_only = ImageFolder(str(roots.train), transform=eval_tf)
        train_loader = _make_loader(
            Subset(train_only, train_subset.indices),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
        )
        val_loader = _make_loader(
            Subset(val_only, val_subset.indices),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
        )

    test_loader: DataLoader | None = None
    if roots.test is not None:
        _assert_imagefolder(roots.test)
        test_ds = ImageFolder(str(roots.test), transform=eval_tf)
        if dict(test_ds.class_to_idx) != class_to_idx:
            raise ValueError(
                "Train/test class folders do not match. "
                f"train={train_ds.class_to_idx} test={test_ds.class_to_idx}"
            )
        test_loader = _make_loader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
        )

    return train_loader, val_loader, test_loader, class_to_idx


def timm_model_name(model_name: str) -> str:
    # Local import to avoid circular imports with models.py in some scripts.
    from models import timm_model_name as _timm_name

    return _timm_name(model_name)


def describe_expected_layout() -> dict[str, str]:
    return {
        "train": "data/raw/train/{REAL,FAKE}",
        "val_optional": "data/raw/val/{REAL,FAKE}",
        "test_optional": "data/raw/test/{REAL,FAKE}",
    }


def get_dataset_info() -> dict:
    """Return basic dataset folder expectations (legacy helper)."""
    return describe_expected_layout()
