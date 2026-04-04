"""
gradcam.py — Grad-CAM for CNNs/EfficientNet + Attention Rollout for ViT.

Provides:
  GradCAM            — Classic Grad-CAM for convolutional models
  GuidedGradCAM      — GradCAM * guided backprop (sharper)
  AttentionRollout   — Attention rollout visualization for ViT
  overlay_heatmap    — Blend heatmap onto an image for display
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from PIL import Image


# ═══════════════════════════════════════════════════════════════════════════════
# Grad-CAM (for CNN / EfficientNet)
# ═══════════════════════════════════════════════════════════════════════════════
class GradCAM:
    """
    Generates Grad-CAM heatmaps for a target layer in a CNN model.

    Usage:
        gcam = GradCAM(model, target_layer=model.model.features[-1])
        heatmap = gcam(input_tensor, class_idx=None)  # None → uses predicted class
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model        = model
        self.target_layer = target_layer
        self._activations = None
        self._gradients   = None
        self._register_hooks()

    def _register_hooks(self):
        def fwd_hook(_, __, output):
            self._activations = output.detach()

        def bwd_hook(_, __, grad_output):
            self._gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(fwd_hook)
        self.target_layer.register_full_backward_hook(bwd_hook)

    def __call__(self, input_tensor: torch.Tensor, class_idx: int = None) -> np.ndarray:
        """
        Args:
            input_tensor: (1, C, H, W) tensor, normalized
            class_idx:    target class index; None → use argmax
        Returns:
            heatmap: (H, W) float32 numpy array in [0, 1]
        """
        self.model.eval()
        input_tensor = input_tensor.clone().requires_grad_(True)

        logits = self.model(input_tensor)

        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()

        self.model.zero_grad()
        score = logits[0, class_idx]
        score.backward()

        # Grad-CAM formula: global average pooling over gradients
        weights     = self._gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam         = (weights * self._activations).sum(dim=1).squeeze(0)  # (H, W)
        cam         = F.relu(cam)
        cam         = cam.cpu().numpy()

        # Normalise to [0, 1]
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# Attention Rollout (for ViT)
# ═══════════════════════════════════════════════════════════════════════════════
class AttentionRollout:
    """
    Attention Rollout for timm ViT models.
    Aggregates attention across all transformer layers to produce a
    (num_patches, num_patches) relevance map.

    Usage:
        rollout = AttentionRollout(model, head_fusion='mean', discard_ratio=0.9)
        mask    = rollout(input_tensor)   # (H, W) numpy array
    """

    def __init__(self, model: nn.Module, head_fusion: str = "mean",
                 discard_ratio: float = 0.9):
        self.model        = model
        self.head_fusion  = head_fusion
        self.discard_ratio = discard_ratio
        self._attn_weights = []

    def _get_attn_hook(self):
        def hook(module, inp, output):
            self._attn_weights.append(output.cpu())
        return hook

    def __call__(self, input_tensor: torch.Tensor) -> np.ndarray:
        self._attn_weights = []
        self.model.eval()

        hooks = []
        for block in self.model.model.blocks:
            # timm attention modules expose raw query/key product via attn_drop
            hooks.append(block.attn.attn_drop.register_forward_hook(self._get_attn_hook()))

        with torch.no_grad():
            _ = self.model(input_tensor)

        for h in hooks:
            h.remove()

        # Rollout
        result = torch.eye(self._attn_weights[0].size(-1))
        for attn in self._attn_weights:
            if self.head_fusion == "mean":
                attn_fused = attn.mean(dim=1)
            elif self.head_fusion == "max":
                attn_fused = attn.max(dim=1).values
            else:
                attn_fused = attn.min(dim=1).values

            flat = attn_fused.view(attn_fused.size(0), attn_fused.size(1), -1)

            # Discard low-attention tokens
            threshold = flat.quantile(self.discard_ratio, dim=-1, keepdim=True)
            flat[flat < threshold] = 0

            # Add residual connection and normalise
            flat = (flat + torch.eye(flat.size(-1)).unsqueeze(0)) / 2
            flat = flat / flat.sum(dim=-1, keepdim=True)

            result = torch.matmul(flat.mean(dim=0), result)

        # CLS token attends to all patch tokens (skip index 0)
        mask = result[0, 1:]   # (num_patches,)
        num_patches = int(mask.numel() ** 0.5)

        mask = mask.reshape(num_patches, num_patches).numpy()
        mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
        return mask.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# Utility: overlay heatmap on image
# ═══════════════════════════════════════════════════════════════════════════════
def overlay_heatmap(
    img_pil: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.45,
    colormap: int = cv2.COLORMAP_JET,
) -> Image.Image:
    """
    Blend a Grad-CAM / attention map over a PIL image.

    Args:
        img_pil  : original PIL image (RGB)
        heatmap  : float32 (H, W) in [0, 1]
        alpha    : heatmap opacity
        colormap : OpenCV colormap constant

    Returns:
        blended PIL image (RGB)
    """
    h, w = img_pil.size[1], img_pil.size[0]
    heatmap_u8 = (heatmap * 255).astype(np.uint8)
    heatmap_resized = cv2.resize(heatmap_u8, (w, h))
    colored = cv2.applyColorMap(heatmap_resized, colormap)   # BGR
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)

    img_np  = np.array(img_pil.convert("RGB")).astype(np.float32)
    blended = (1 - alpha) * img_np + alpha * colored.astype(np.float32)
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    return Image.fromarray(blended)


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """Convert a normalised CHW tensor back to (H, W, 3) uint8 for display."""
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img  = tensor.cpu().numpy().transpose(1, 2, 0)
    img  = img * std + mean
    img  = np.clip(img * 255, 0, 255).astype(np.uint8)
    return img
