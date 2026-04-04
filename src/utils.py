from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed(seed: int, *, cudnn_benchmark: bool | None = None) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # cudnn.benchmark=True is usually faster on fixed input sizes; cudnn.benchmark=False is more reproducible.
    if cudnn_benchmark is None:
        cudnn_benchmark = torch.cuda.is_available()

    torch.backends.cudnn.benchmark = bool(cudnn_benchmark)
    torch.backends.cudnn.deterministic = not bool(cudnn_benchmark)


def get_device(prefer_cuda: bool = True) -> torch.device:
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def ensure_dir(path: str | os.PathLike[str]) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(path: str | os.PathLike[str], payload: dict[str, Any]) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    with p.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
