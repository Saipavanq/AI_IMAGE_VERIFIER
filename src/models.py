"""Model definitions for AI image verifier."""

from __future__ import annotations

from dataclasses import dataclass

import timm
import torch.nn as nn

SUPPORTED_MODELS = ("cnn", "efficientnet", "vit")


def list_models() -> tuple[str, ...]:
    """Return supported model names."""
    return SUPPORTED_MODELS


@dataclass(frozen=True)
class ModelSpec:
    timm_name: str
    pretrained: bool = True


_MODEL_SPECS: dict[str, ModelSpec] = {
    # Fast, strong baseline CNN.
    "cnn": ModelSpec("resnet18", pretrained=True),
    # Strong transfer-learning backbone (timm default input size is typically 320x320).
    "efficientnet": ModelSpec("efficientnet_b4", pretrained=True),
    # ViT-B/16 @ 224 (matches common hackathon constraints).
    "vit": ModelSpec("vit_base_patch16_224", pretrained=True),
}


def build_model(model_name: str, num_classes: int = 2, *, pretrained: bool | None = None) -> nn.Module:
    if model_name not in _MODEL_SPECS:
        raise ValueError(f"Unknown model {model_name!r}. Choose one of: {', '.join(SUPPORTED_MODELS)}")

    spec = _MODEL_SPECS[model_name]
    use_pretrained = spec.pretrained if pretrained is None else pretrained

    model = timm.create_model(spec.timm_name, pretrained=use_pretrained, num_classes=num_classes)
    return model


def timm_model_name(model_name: str) -> str:
    if model_name not in _MODEL_SPECS:
        raise ValueError(f"Unknown model {model_name!r}. Choose one of: {', '.join(SUPPORTED_MODELS)}")
    return _MODEL_SPECS[model_name].timm_name
