from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch
import torch.nn as nn


@dataclass(frozen=True)
class GradCAMResult:
    cam: np.ndarray  # HxW float32 in [0,1]
    target_layer: str


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module, target_name: str) -> None:
        self.model = model
        self.target_layer = target_layer
        self.target_name = target_name

        self._acts: torch.Tensor | None = None
        self._grads: torch.Tensor | None = None

        self._fwd_handle = target_layer.register_forward_hook(self._forward_hook)
        self._bwd_handle = target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, _module, inputs, output):  # type: ignore[no-untyped-def]
        # Keep activations detached from graph? We need them attached for backward through activations.
        self._acts = output

    def _backward_hook(self, _module, grad_in, grad_out):  # type: ignore[no-untyped-def]
        g = grad_out[0]
        self._grads = g

    def close(self) -> None:
        self._fwd_handle.remove()
        self._bwd_handle.remove()

    @torch.enable_grad()
    def compute(self, input_tensor: torch.Tensor, class_idx: int) -> GradCAMResult:
        self.model.zero_grad(set_to_none=True)
        self._acts = None
        self._grads = None

        logits = self.model(input_tensor)
        if logits.ndim != 2 or logits.shape[1] < 2:
            raise ValueError(f"Expected logits [N,C] with C>=2, got {tuple(logits.shape)}")

        score = logits[0, int(class_idx)]
        score.backward()

        acts = self._acts
        grads = self._grads
        if acts is None or grads is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations/gradients.")

        # Global average pool gradients across spatial dims -> channel weights
        weights = grads.mean(dim=(2, 3), keepdim=True)  # [N,C,1,1]
        cam = (weights * acts).sum(dim=1, keepdim=True)  # [N,1,H,W]
        cam = torch.relu(cam)

        cam_np = cam[0, 0].detach().float().cpu().numpy()
        cam_np = _normalize_map(cam_np)
        return GradCAMResult(cam=cam_np.astype(np.float32), target_layer=self.target_name)


def _normalize_map(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    m, M = float(x.min()), float(x.max())
    if M - m < 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    return (x - m) / (M - m)


def overlay_heatmap_on_image(
    rgb_uint8_hwc: np.ndarray,
    heatmap_hw: np.ndarray,
    *,
    alpha: float = 0.45,
) -> np.ndarray:
    """Overlay a heatmap on an RGB uint8 image (HxWx3)."""
    if rgb_uint8_hwc.dtype != np.uint8:
        raise ValueError("rgb_uint8_hwc must be uint8")
    if rgb_uint8_hwc.ndim != 3 or rgb_uint8_hwc.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 RGB, got {rgb_uint8_hwc.shape}")

    h, w = heatmap_hw.shape[:2]
    if (h, w) != (rgb_uint8_hwc.shape[0], rgb_uint8_hwc.shape[1]):
        heatmap_hw = cv2.resize(heatmap_hw, (rgb_uint8_hwc.shape[1], rgb_uint8_hwc.shape[0]), interpolation=cv2.INTER_CUBIC)

    heat_u8 = np.uint8(np.clip(heatmap_hw, 0.0, 1.0) * 255.0)
    color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    out = (alpha * color.astype(np.float32) + (1.0 - alpha) * rgb_uint8_hwc.astype(np.float32)).astype(np.uint8)
    return out


def resolve_target_layer(model_name: str, model: nn.Module) -> tuple[nn.Module, str]:
    """Pick a reasonable Grad-CAM target for each supported architecture."""
    if model_name == "cnn":
        # ResNet family
        if not hasattr(model, "layer4"):
            raise ValueError("Expected a ResNet-like model with `.layer4` for CNN baseline.")
        return model.layer4, "layer4"

    if model_name == "efficientnet":
        if not hasattr(model, "conv_head"):
            raise ValueError("Expected EfficientNet-like model with `.conv_head`.")
        return model.conv_head, "conv_head"

    if model_name == "vit":
        # Use the last block output tokens (includes patch tokens + cls token); Grad-CAM uses conv feature maps,
        # so we treat token grid as 1x1 "spatial" maps by reshaping patch tokens to a 2D grid.
        if not hasattr(model, "blocks"):
            raise ValueError("Expected ViT-like model with `.blocks`.")
        return model.blocks[-1], "blocks[-1]"

    raise ValueError(f"Unknown model_name={model_name!r}")


def compute_vit_patch_gradcam(
    model: nn.Module,
    input_tensor: torch.Tensor,
    class_idx: int,
) -> GradCAMResult:
    """Special-case Grad-CAM for ViT by reshaping patch tokens into a spatial map."""
    block = model.blocks[-1]

    acts = None
    grads = None

    def fwd_hook(_m, _inp, out):  # type: ignore[no-untyped-def]
        nonlocal acts
        acts = out

    def bwd_hook(_m, grad_in, grad_out):  # type: ignore[no-untyped-def]
        nonlocal grads
        grads = grad_out[0]

    h1 = block.register_forward_hook(fwd_hook)
    h2 = block.register_full_backward_hook(bwd_hook)
    try:
        model.zero_grad(set_to_none=True)
        logits = model(input_tensor)
        score = logits[0, int(class_idx)]
        score.backward()

        if acts is None or grads is None:
            raise RuntimeError("ViT hooks failed to capture activations/gradients.")

        # acts/grads: [B, N, C] includes cls token at index 0
        a = acts[0, 1:, :]  # patch tokens
        g = grads[0, 1:, :]

        w = g.mean(dim=0, keepdim=True)  # [1,C] (global pooling across tokens)
        cam_tokens = (a * w).sum(dim=-1)  # [N_patches]
        cam_tokens = torch.relu(cam_tokens)

        n_tokens = cam_tokens.numel()
        side = int(round(n_tokens**0.5))
        if side * side != n_tokens:
            # Fallback: reshape as 1xN map then resize later
            cam_map = cam_tokens.view(1, 1, 1, n_tokens)
        else:
            cam_map = cam_tokens.view(1, 1, side, side)

        cam_np = cam_map[0, 0].detach().float().cpu().numpy()
        cam_np = _normalize_map(cam_np)
        return GradCAMResult(cam=cam_np.astype(np.float32), target_layer="vit.blocks[-1].patch_tokens")
    finally:
        h1.remove()
        h2.remove()


def run_gradcam_for_model(
    model_name: str,
    model: nn.Module,
    input_tensor: torch.Tensor,
    class_idx: int,
) -> GradCAMResult:
    if model_name == "vit":
        return compute_vit_patch_gradcam(model, input_tensor, class_idx)

    layer, name = resolve_target_layer(model_name, model)
    cam = GradCAM(model, layer, name)
    try:
        return cam.compute(input_tensor, class_idx)
    finally:
        cam.close()
