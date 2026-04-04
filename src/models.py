"""
models.py — Three classifier architectures for Real vs AI-Generated detection.

  1. BaselineCNN       — Custom 5-block convolutional network
  2. EfficientNetClassifier — Fine-tuned EfficientNet-B4 (torchvision)
  3. ViTClassifier     — Fine-tuned ViT-B/16 (timm)
"""

import torch
import torch.nn as nn
import torchvision.models as tv_models
import timm

NUM_CLASSES = 2


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Baseline CNN
# ═══════════════════════════════════════════════════════════════════════════════
class ConvBlock(nn.Module):
    """Conv → BN → ReLU → MaxPool"""
    def __init__(self, in_ch: int, out_ch: int, pool: bool = True):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class BaselineCNN(nn.Module):
    """
    Lightweight 5-block CNN for binary classification.
    Input: (B, 3, 224, 224)  →  Output: (B, 2)
    """
    def __init__(self, num_classes: int = NUM_CLASSES, dropout: float = 0.4):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3,   32),    # → 112
            ConvBlock(32,  64),    # → 56
            ConvBlock(64,  128),   # → 28
            ConvBlock(128, 256),   # → 14
            ConvBlock(256, 512),   # → 7
        )
        self.pool       = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. EfficientNet-B4 (fine-tuned)
# ═══════════════════════════════════════════════════════════════════════════════
class EfficientNetClassifier(nn.Module):
    """
    EfficientNet-B4 pretrained on ImageNet, last classifier replaced.
    Recommended model — best accuracy / compute balance.
    """
    def __init__(self, num_classes: int = NUM_CLASSES, dropout: float = 0.4,
                 freeze_backbone: bool = False):
        super().__init__()
        weights = tv_models.EfficientNet_B4_Weights.DEFAULT
        backbone = tv_models.efficientnet_b4(weights=weights)

        if freeze_backbone:
            for p in backbone.parameters():
                p.requires_grad = False

        # Replace classifier head (in_features = 1792 for B4)
        in_features = backbone.classifier[1].in_features
        backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(in_features, num_classes),
        )
        self.model = backbone

    def forward(self, x):
        return self.model(x)

    def get_last_conv(self):
        """Return last convolutional layer — needed for Grad-CAM."""
        return self.model.features[-1]


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Vision Transformer — ViT-B/16 (timm)
# ═══════════════════════════════════════════════════════════════════════════════
class ViTClassifier(nn.Module):
    """
    ViT-B/16 pretrained on ImageNet-21k, fine-tuned head.
    Attention maps reveal artifact patterns across image patches.
    """
    def __init__(self, num_classes: int = NUM_CLASSES, dropout: float = 0.1,
                 freeze_backbone: bool = False):
        super().__init__()
        self.model = timm.create_model(
            "vit_base_patch16_224",
            pretrained=True,
            num_classes=num_classes,
            drop_rate=dropout,
        )
        if freeze_backbone:
            for name, p in self.model.named_parameters():
                if "head" not in name:
                    p.requires_grad = False

    def forward(self, x):
        return self.model(x)

    def get_attention_weights(self, x: torch.Tensor):
        """
        Forward pass collecting attention weights from every transformer block.
        Returns list of attention tensors: [(B, heads, N, N), ...]
        """
        attn_weights = []

        def _hook(module, inp, out):
            # timm stores raw attention before softmax in some versions;
            # we capture the output (post-softmax attention)
            attn_weights.append(out.detach())

        hooks = []
        for block in self.model.blocks:
            hooks.append(block.attn.register_forward_hook(_hook))

        _ = self.model(x)
        for h in hooks:
            h.remove()
        return attn_weights


# ═══════════════════════════════════════════════════════════════════════════════
# Factory
# ═══════════════════════════════════════════════════════════════════════════════
def build_model(name: str, **kwargs) -> nn.Module:
    """
    Build a model by name.  name ∈ {'cnn', 'efficientnet', 'vit'}
    """
    name = name.lower()
    if name == "cnn":
        return BaselineCNN(**kwargs)
    elif name == "efficientnet":
        return EfficientNetClassifier(**kwargs)
    elif name == "vit":
        return ViTClassifier(**kwargs)
    else:
        raise ValueError(f"Unknown model: '{name}'. Choose from cnn | efficientnet | vit")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
