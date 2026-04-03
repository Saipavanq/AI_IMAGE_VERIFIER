"""Explanation/Grad-CAM entry point."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import timm
import cv2
from PIL import Image
from timm.data import resolve_data_config
from torch.cuda.amp import autocast
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from artifact_features import compute_artifact_signals, explain_signals
from dataset import build_transforms, timm_model_name
from gradcam import overlay_heatmap_on_image, run_gradcam_for_model
from models import build_model, list_models
from utils import ensure_dir, get_device, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate explanations")
    parser.add_argument("--data_root", type=str, default="data/raw")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--model", choices=list_models(), default="efficientnet")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--out_dir", type=str, default="outputs/explain")
    return parser.parse_args()


def _pick_image_paths(root: Path, n: int, seed: int) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    all_paths: list[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            all_paths.append(p)

    if not all_paths:
        raise FileNotFoundError(f"No images found under: {root}")

    rng = random.Random(seed)
    rng.shuffle(all_paths)
    return all_paths[: min(n, len(all_paths))]


def _load_rgb_uint8(path: Path) -> np.ndarray:
    im = Image.open(path).convert("RGB")
    return np.asarray(im, dtype=np.uint8)


def cv2_resize_heatmap(heatmap_hw: np.ndarray, size_xy: tuple[int, int]) -> np.ndarray:
    w, h = size_xy
    return cv2.resize(heatmap_hw, (w, h), interpolation=cv2.INTER_CUBIC)


def _tensor_to_rgb_chw(x_chw: torch.Tensor, *, model_name: str) -> np.ndarray:
    m = timm.create_model(timm_model_name(model_name), pretrained=False)
    cfg = resolve_data_config(m.pretrained_cfg, model=m)
    mean = torch.tensor(cfg["mean"], dtype=x_chw.dtype).view(3, 1, 1)
    std = torch.tensor(cfg["std"], dtype=x_chw.dtype).view(3, 1, 1)
    rgb = (x_chw * std + mean).clamp(0.0, 1.0).numpy()
    return rgb



def main() -> None:
    args = parse_args()
    device = get_device(prefer_cuda=True)
    amp = bool(args.amp) and device.type == "cuda"

    ckpt = torch.load(args.checkpoint, map_location=device)
    class_to_idx: dict[str, int] = ckpt["class_to_idx"]

    split_dir = Path(args.data_root) / args.split
    if not split_dir.exists():
        raise FileNotFoundError(f"Missing split folder: {split_dir}")

    paths = _pick_image_paths(split_dir, args.num_samples, args.seed)

    model = build_model(args.model, num_classes=len(class_to_idx), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()

    tfm = build_transforms(model_name=args.model, is_training=False)

    folder_ds = ImageFolder(str(split_dir))
    if dict(folder_ds.class_to_idx) != dict(class_to_idx):
        raise ValueError("Checkpoint class_to_idx does not match dataset folders.")

    out_root = ensure_dir(Path(args.out_dir))
    report: list[dict] = []

    for path in tqdm(paths, desc="explain"):
        pil = Image.open(path).convert("RGB")
        x = tfm(pil).unsqueeze(0).to(device, non_blocking=True)

        with autocast(enabled=amp):
            logits = model(x)

        prob = torch.softmax(logits, dim=1)[0].detach().float().cpu().numpy()
        pred_idx = int(np.argmax(prob))
        pred_label = [k for k, v in class_to_idx.items() if v == pred_idx][0]

        gradcam = run_gradcam_for_model(args.model, model, x, class_idx=pred_idx)

        # Resize heatmap to original image size for overlay
        rgb = _load_rgb_uint8(path)
        heat = cv2_resize_heatmap(gradcam.cam, (rgb.shape[1], rgb.shape[0]))
        overlay = overlay_heatmap_on_image(rgb, heat, alpha=0.45)

        # Artifact cues on the model input tensor (unnormalize for RGB CHW)
        # We approximate by denormalizing using timm cfg via inverse transforms on tensor.
        x_cpu = x.detach().float().cpu().squeeze(0)
        rgb_chw = _tensor_to_rgb_chw(x_cpu, model_name=args.model)

        signals = compute_artifact_signals(rgb_chw)
        notes = explain_signals(signals)

        stem = f"{path.parent.name}__{path.stem}"
        out_img = out_root / f"{stem}_gradcam.png"
        Image.fromarray(overlay).save(out_img)

        report.append(
            {
                "image": str(path),
                "pred_label": pred_label,
                "probs": {"REAL": float(prob[int(class_to_idx["REAL"])]), "FAKE": float(prob[int(class_to_idx["FAKE"])])},
                "gradcam": str(out_img),
                "artifact_signals": {
                    "hf_energy_ratio": signals.hf_energy_ratio,
                    "lap_var": signals.lap_var,
                    "grad_mag_mean": signals.grad_mag_mean,
                    "color_bgr_std_mean": signals.color_bgr_std_mean,
                },
                "artifact_notes": notes,
            }
        )

    save_json(str(out_root / f"explain_report_{args.model}_{args.split}.json"), {"items": report})
    print(f"Wrote {len(report)} explanations to: {out_root}")


if __name__ == "__main__":
    main()
