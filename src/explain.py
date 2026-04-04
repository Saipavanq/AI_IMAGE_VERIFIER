"""
explain.py — Generate Grad-CAM artifact explanation reports for test images.

Usage:
  python src/explain.py --model efficientnet --checkpoint models/best_efficientnet.pth --num_samples 20
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import datasets
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.dataset import get_val_transform
from src.models import build_model
from src.gradcam import GradCAM, AttentionRollout, overlay_heatmap, denormalize


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
CLASS_NAMES   = ["FAKE", "REAL"]

# Heuristic artifact type labels based on Grad-CAM activation location
ARTIFACT_DESCRIPTIONS = {
    0: {  # predicted FAKE
        "center":  "Central texture anomaly — unnatural surface regularity",
        "edges":   "Edge artifacts — blurry or over-sharpened boundaries",
        "default": "AI generation artifact — inconsistent tonal distribution",
    },
    1: {  # predicted REAL
        "center":  "Natural subject detail — consistent with real photography",
        "edges":   "Natural edge gradients — camera lens / depth-of-field response",
        "default": "Real image characteristics — natural noise & lighting pattern",
    },
}


def infer_artifact_region(heatmap: np.ndarray) -> str:
    """
    Classify where the highest activation is concentrated:
    'center' or 'edges'.
    """
    h, w   = heatmap.shape
    center = heatmap[h // 4: 3 * h // 4, w // 4: 3 * w // 4].mean()
    border = heatmap.mean() - center
    return "center" if center >= border else "edges"


def parse_args():
    p = argparse.ArgumentParser(description="Generate Grad-CAM artifact reports")
    p.add_argument("--model",       type=str, default="efficientnet",
                   choices=["cnn", "efficientnet", "vit"])
    p.add_argument("--checkpoint",  type=str, required=True)
    p.add_argument("--data_dir",    type=str, default="data/raw")
    p.add_argument("--num_samples", type=int, default=20)
    p.add_argument("--img_size",    type=int, default=224)
    p.add_argument("--output_dir",  type=str, default="outputs/gradcam")
    p.add_argument("--seed",        type=int, default=42)
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load model ─────────────────────────────────────────────────────────
    ckpt = torch.load(args.checkpoint, map_location=device)
    model_name = ckpt.get("model_name", args.model)
    model = build_model(model_name).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(f"[Explain] Loaded '{model_name}' checkpoint.")

    # ── Explainer ──────────────────────────────────────────────────────────
    if model_name == "vit":
        explainer = AttentionRollout(model, head_fusion="mean", discard_ratio=0.9)
    else:
        # Get last conv layer automatically
        if model_name == "efficientnet":
            target_layer = model.model.features[-1]
        else:  # cnn
            target_layer = model.features[-1]
        explainer = GradCAM(model, target_layer=target_layer)

    # ── Dataset (test, random sample) ─────────────────────────────────────
    transform  = get_val_transform(args.img_size)
    test_ds    = datasets.ImageFolder(
        root=str(Path(args.data_dir) / "test"), transform=transform)

    rng     = np.random.default_rng(args.seed)
    indices = rng.choice(len(test_ds), size=min(args.num_samples, len(test_ds)), replace=False)

    print(f"[Explain] Generating reports for {len(indices)} images …")

    for i, idx in enumerate(tqdm(indices)):
        img_tensor, true_label = test_ds[idx]
        img_pil_path = test_ds.imgs[idx][0]
        img_pil      = Image.open(img_pil_path).convert("RGB").resize(
            (args.img_size, args.img_size), Image.BICUBIC)

        inp = img_tensor.unsqueeze(0).to(device)

        # Forward for prediction
        with torch.no_grad():
            logits = model(inp)
            probs  = torch.softmax(logits, dim=1)[0]
            pred   = logits.argmax(dim=1).item()
            conf   = probs[pred].item()

        # Heatmap
        if model_name == "vit":
            heatmap = explainer(inp)
        else:
            heatmap = explainer(inp, class_idx=pred)

        region      = infer_artifact_region(heatmap)
        description = ARTIFACT_DESCRIPTIONS[pred].get(region,
                      ARTIFACT_DESCRIPTIONS[pred]["default"])

        overlay = overlay_heatmap(img_pil, heatmap, alpha=0.45)

        # ── Plot ──────────────────────────────────────────────────────────
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        fig.patch.set_facecolor("#1a1a2e")

        for ax in axes:
            ax.set_facecolor("#16213e")
            ax.tick_params(colors="white")
            for spine in ax.spines.values():
                spine.set_edgecolor("#0f3460")

        axes[0].imshow(img_pil)
        axes[0].set_title("Original Image", color="white", fontsize=11, pad=8)
        axes[0].axis("off")

        axes[1].imshow(overlay)
        axes[1].set_title("Grad-CAM Overlay", color="white", fontsize=11, pad=8)
        axes[1].axis("off")

        # Heatmap colourbar
        im = axes[2].imshow(heatmap, cmap="hot", vmin=0, vmax=1)
        axes[2].set_title("Activation Map", color="white", fontsize=11, pad=8)
        axes[2].axis("off")
        cbar = plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
        cbar.ax.yaxis.set_tick_params(color="white")
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

        # Title with prediction
        true_str = CLASS_NAMES[true_label]
        pred_str = CLASS_NAMES[pred]
        verdict  = "✔ Correct" if pred == true_label else "✘ Wrong"
        suptitle = (f"True: {true_str}  |  Pred: {pred_str} ({conf*100:.1f}%)  "
                    f"|  {verdict}\n"
                    f"Artifact: {description}")
        fig.suptitle(suptitle, color="white", fontsize=10, y=1.01, wrap=True)
        plt.tight_layout(pad=1.5)

        out_path = out_dir / f"sample_{i:03d}_true{true_str}_pred{pred_str}.png"
        fig.savefig(out_path, dpi=120, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)

    print(f"\n[Explain] Done. {len(indices)} images saved → {out_dir}/")


if __name__ == "__main__":
    main()
