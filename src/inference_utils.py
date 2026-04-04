"""Shared inference helpers: TTA (test-time augmentation) for more stable predictions."""

from __future__ import annotations

import torch
from PIL import Image
from torch.cuda.amp import autocast
from torchvision import transforms as T


@torch.no_grad()
def predict_proba_batch(
    model: nn.Module,
    x: torch.Tensor,
    device: torch.device,
    *,
    amp: bool,
) -> torch.Tensor:
    x = x.to(device, non_blocking=True)
    with autocast(enabled=amp and device.type == "cuda"):
        logits = model(x)
    return torch.softmax(logits, dim=1)


def build_tta_batch(pil_rgb: Image.Image, base_transform: T.Compose) -> torch.Tensor:
    """Stack original + horizontal flip (+ optional 90 rot) as a batch for TTA."""
    tensors: list[torch.Tensor] = []
    tensors.append(base_transform(pil_rgb))
    flip = pil_rgb.transpose(Image.FLIP_LEFT_RIGHT)
    tensors.append(base_transform(flip))
    return torch.stack(tensors, dim=0)


@torch.no_grad()
def predict_proba_tta_single(
    model: nn.Module,
    pil_rgb: Image.Image,
    base_transform: T.Compose,
    device: torch.device,
    *,
    amp: bool,
    use_tta: bool,
) -> torch.Tensor:
    """Return softmax probabilities [C] for one image."""
    if not use_tta:
        x = base_transform(pil_rgb).unsqueeze(0)
        return predict_proba_batch(model, x, device, amp=amp)[0]

    x = build_tta_batch(pil_rgb, base_transform)
    probs = predict_proba_batch(model, x, device, amp=amp)
    return probs.mean(dim=0)
