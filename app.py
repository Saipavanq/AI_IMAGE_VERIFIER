"""Streamlit demo: real vs AI-generated + Grad-CAM overlay + heuristic artifact cues."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import streamlit as st
import timm
import torch
import cv2
from PIL import Image
from timm.data import resolve_data_config

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from artifact_features import compute_artifact_signals, explain_signals  # noqa: E402
from dataset import build_transforms, timm_model_name  # noqa: E402
from gradcam import overlay_heatmap_on_image, run_gradcam_for_model  # noqa: E402
from inference_utils import predict_proba_tta_single  # noqa: E402
from models import build_model, list_models  # noqa: E402
from utils import get_device  # noqa: E402


@st.cache_resource
def load_model(model_name: str, checkpoint_path: str):
    device = get_device(prefer_cuda=True)
    ckpt = torch.load(checkpoint_path, map_location=device)
    class_to_idx = ckpt["class_to_idx"]

    model = build_model(model_name, num_classes=len(class_to_idx), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    return model, class_to_idx, device


def tensor_to_rgb_chw(x_chw: torch.Tensor, model_name: str) -> np.ndarray:
    m = timm.create_model(timm_model_name(model_name), pretrained=False)
    cfg = resolve_data_config(m.pretrained_cfg, model=m)
    mean = torch.tensor(cfg["mean"], dtype=x_chw.dtype).view(3, 1, 1)
    std = torch.tensor(cfg["std"], dtype=x_chw.dtype).view(3, 1, 1)
    rgb = (x_chw * std + mean).clamp(0.0, 1.0).numpy()
    return rgb


def main() -> None:
    st.set_page_config(page_title="AI Image Verifier", layout="wide")
    st.title("AI Image Verifier (PS-7)")
    st.caption("Upload an image to get a Real vs AI-generated prediction with Grad-CAM + heuristic artifact cues.")

    with st.sidebar:
        model_name = st.selectbox("Model", list(list_models()))
        checkpoint = st.text_input("Checkpoint path", value=str(ROOT / "models" / f"best_{model_name}.pth"))
        use_tta = st.checkbox("Test-time augmentation (TTA)", value=True, help="Average prediction with a horizontal flip — usually more accurate.")

    uploaded = st.file_uploader("Image", type=["png", "jpg", "jpeg", "webp", "bmp"])
    if not uploaded:
        st.info("Upload an image to begin.")
        return

    pil = Image.open(uploaded).convert("RGB")
    rgb = np.asarray(pil, dtype=np.uint8)

    model, class_to_idx, device = load_model(model_name, checkpoint)
    tfm = build_transforms(model_name=model_name, is_training=False)
    x = tfm(pil).unsqueeze(0).to(device)

    amp = device.type == "cuda"
    prob_t = predict_proba_tta_single(
        model, pil, tfm, device, amp=amp, use_tta=use_tta
    )
    prob = prob_t.detach().float().cpu().numpy()

    pred_idx = int(np.argmax(prob))
    pred = [k for k, v in class_to_idx.items() if v == pred_idx][0]

    gradcam = run_gradcam_for_model(model_name, model, x, class_idx=pred_idx)
    heat = gradcam.cam
    # Resize heatmap to original image size
    heat = cv2.resize(heat, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_CUBIC)
    overlay = overlay_heatmap_on_image(rgb, heat, alpha=0.45)

    x_cpu = x.detach().float().cpu().squeeze(0)
    rgb_chw = tensor_to_rgb_chw(x_cpu, model_name)
    signals = compute_artifact_signals(rgb_chw)
    notes = explain_signals(signals)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Input")
        st.image(pil, use_container_width=True)
    with c2:
        st.subheader("Grad-CAM (prediction-focused)")
        st.image(Image.fromarray(overlay), use_container_width=True)

    st.subheader("Prediction")
    st.write(
        {
            "predicted": pred,
            "P(REAL)": float(prob[int(class_to_idx["REAL"])]),
            "P(FAKE)": float(prob[int(class_to_idx["FAKE"])]),
        }
    )

    st.subheader("Heuristic artifact cues (interpretability helpers)")
    st.json(
        {
            "hf_energy_ratio": signals.hf_energy_ratio,
            "lap_var": signals.lap_var,
            "grad_mag_mean": signals.grad_mag_mean,
            "color_bgr_std_mean": signals.color_bgr_std_mean,
        }
    )
    for n in notes:
        st.write(f"- {n}")


if __name__ == "__main__":
    main()

