"""
app.py — Streamlit interactive demo for Real vs AI-Generated image detection.

Run:
  streamlit run app.py
"""

import sys
import io
import json
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import streamlit as st
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from src.models import build_model
from src.dataset import get_val_transform
from src.gradcam import GradCAM, AttentionRollout, overlay_heatmap

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title  = "AIShield — Real vs AI Image Detector",
    page_icon   = "🔍",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  .main { background: linear-gradient(135deg, #0f0c29, #302b63, #24243e); }

  .hero-title {
    font-size: 2.8rem; font-weight: 700;
    background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    text-align: center; margin-bottom: 0.2rem;
  }
  .hero-sub {
    text-align: center; color: #94a3b8; font-size: 1.05rem; margin-bottom: 2rem;
  }

  .result-card {
    border-radius: 16px; padding: 1.5rem;
    border: 1px solid rgba(255,255,255,0.1);
    backdrop-filter: blur(10px);
    margin-bottom: 1rem;
  }
  .real-card  { background: rgba(52, 211, 153, 0.12); border-color: rgba(52, 211, 153, 0.4); }
  .fake-card  { background: rgba(248, 113, 113, 0.12); border-color: rgba(248, 113, 113, 0.4); }

  .metric-label { font-size: 0.8rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.08em; }
  .metric-value { font-size: 2rem; font-weight: 700; }
  .real-val  { color: #34d399; }
  .fake-val  { color: #f87171; }

  .artifact-box {
    background: rgba(167, 139, 250, 0.1);
    border: 1px solid rgba(167, 139, 250, 0.3);
    border-radius: 12px; padding: 1rem; margin-top: 1rem;
  }
  .stButton>button {
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white; border: none; border-radius: 10px;
    padding: 0.6rem 2rem; font-weight: 600; width: 100%;
    transition: transform 0.1s;
  }
  .stButton>button:hover { transform: translateY(-1px); }
  .upload-hint { color: #64748b; font-size: 0.85rem; text-align: center; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
CLASS_NAMES        = ["FAKE", "REAL"]
IMG_SIZE           = 224
DEVICE             = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_MODEL_NAME = "cnn"
DEFAULT_CKPT       = "models/best_cnn.pth"

ARTIFACT_INFO = {
    "FAKE": {
        "center": ("Texture Anomaly",
                   "AI generators often produce unnatural surface regularity in central regions — "
                   "look for suspiciously uniform patterns, smeared details, or over-smooth skin."),
        "edges": ("Boundary Artifact",
                  "Edges between objects or between subject and background may be over-sharpened, "
                  "blurred, or show 'halo' effects common in diffusion models."),
    },
    "REAL": {
        "center": ("Natural Subject Detail",
                   "Activation focuses on the subject's key features — consistent with natural "
                   "photography where sensor noise and lens characteristics create organic detail."),
        "edges": ("Natural Gradient Response",
                  "Edge response follows natural camera optics — depth-of-field and chromatic "
                  "aberration create the gradual transitions seen here."),
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers (cached)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model …")
def load_model(model_name: str, ckpt_path: str):
    ckpt  = torch.load(ckpt_path, map_location=DEVICE)
    model = build_model(model_name).to(DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    if model_name == "vit":
        explainer = AttentionRollout(model, head_fusion="mean", discard_ratio=0.9)
    elif model_name == "efficientnet":
        explainer = GradCAM(model, target_layer=model.model.features[-1])
    else:
        explainer = GradCAM(model, target_layer=model.features[-1])

    return model, explainer, ckpt.get("val_f1", None), ckpt.get("val_acc", None)


def preprocess(img_pil: Image.Image, img_size: int = IMG_SIZE) -> torch.Tensor:
    transform = get_val_transform(img_size)
    return transform(img_pil.convert("RGB")).unsqueeze(0).to(DEVICE)


def get_heatmap(explainer, inp, model_name, pred):
    if model_name == "vit":
        return explainer(inp)
    return explainer(inp, class_idx=pred)


def activation_region(heatmap: np.ndarray) -> str:
    h, w   = heatmap.shape
    center = heatmap[h // 4: 3 * h // 4, w // 4: 3 * w // 4].mean()
    border = heatmap.mean() - center
    return "center" if center >= border else "edges"


def plot_heatmap(heatmap: np.ndarray) -> bytes:
    fig, ax = plt.subplots(figsize=(4, 4))
    fig.patch.set_facecolor("#1a1a2e")
    im = ax.imshow(heatmap, cmap="inferno", vmin=0, vmax=1)
    ax.axis("off")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(colors="white")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor=fig.get_facecolor(), dpi=120)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    model_choice = st.selectbox(
        "Model Architecture",
        options=["efficientnet", "cnn", "vit"],
        index=1,
        help="EfficientNet-B4 is recommended for best accuracy."
    )
    ckpt_path = st.text_input("Checkpoint Path", value=f"models/best_{model_choice}.pth")
    alpha     = st.slider("Heatmap Opacity", 0.1, 0.8, 0.45, step=0.05)

    st.markdown("---")
    st.markdown("### 📖 About")
    st.markdown("""
**AIShield** detects AI-generated images and explains *why* using Grad-CAM attention visualization.

**Models**: CNN · EfficientNet-B4 · ViT-B/16  
**Dataset**: CIFAKE (60K images)  
**Explainability**: Grad-CAM · Attention Rollout
    """)

    # Model info card
    if Path(ckpt_path).exists():
        try:
            _, _, val_f1, val_acc = load_model(model_choice, ckpt_path)
            if val_f1:
                st.markdown("---")
                st.markdown("### 📊 Model Performance")
                col1, col2 = st.columns(2)
                col1.metric("Val F1", f"{val_f1:.3f}")
                col2.metric("Val Acc", f"{val_acc:.3f}")
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# Main UI
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<h1 class="hero-title">🔍 AIShield</h1>', unsafe_allow_html=True)
st.markdown('<p class="hero-sub">Real vs AI-Generated Image Detector · Powered by Deep Learning & Grad-CAM Explainability</p>',
            unsafe_allow_html=True)

# Upload
uploaded = st.file_uploader(
    "Upload an image to analyze",
    type=["jpg", "jpeg", "png", "webp", "bmp"],
    label_visibility="collapsed",
)
st.markdown('<p class="upload-hint">Supports JPG · PNG · WEBP · BMP</p>', unsafe_allow_html=True)

if uploaded is not None:
    img_pil = Image.open(uploaded).convert("RGB")

    # Check checkpoint
    if not Path(ckpt_path).exists():
        st.error(f"❌ Checkpoint not found: `{ckpt_path}`\n\nPlease train a model first:\n```\npython src/train.py --model {model_choice} --epochs 20\n```")
        st.stop()

    # Load model
    model, explainer, _, _ = load_model(model_choice, ckpt_path)

    # Inference
    inp    = preprocess(img_pil)
    with torch.no_grad():
        logits = model(inp)
        probs  = torch.softmax(logits, dim=1)[0]
        pred   = logits.argmax(dim=1).item()
        conf   = probs[pred].item()

    # Heatmap
    with st.spinner("Generating Grad-CAM explanation …"):
        heatmap = get_heatmap(explainer, inp, model_choice, pred)
        overlay = overlay_heatmap(img_pil.resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC),
                                  heatmap, alpha=alpha)

    label  = CLASS_NAMES[pred]
    region = activation_region(heatmap)
    artifact_title, artifact_desc = ARTIFACT_INFO[label][region]

    # ── Results layout ────────────────────────────────────────────────────
    col_img, col_cam, col_heat = st.columns([1, 1, 1])

    with col_img:
        st.image(img_pil.resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC),
                 caption="Uploaded Image", use_column_width=True)

    with col_cam:
        st.image(overlay, caption="Grad-CAM Overlay", use_column_width=True)

    with col_heat:
        st.image(plot_heatmap(heatmap),
                 caption="Activation Heatmap", use_column_width=True)

    # ── Verdict card ──────────────────────────────────────────────────────
    card_class = "real-card" if label == "REAL" else "fake-card"
    val_class  = "real-val"  if label == "REAL" else "fake-val"
    icon       = "✅" if label == "REAL" else "🚨"

    st.markdown(f"""
<div class="result-card {card_class}">
  <div style="display:flex; align-items:center; gap:1rem; margin-bottom:1rem;">
    <span style="font-size:2.5rem">{icon}</span>
    <div>
      <div class="metric-label">Classification Result</div>
      <div class="metric-value {val_class}">{label}</div>
    </div>
    <div style="margin-left:auto; text-align:right;">
      <div class="metric-label">Confidence</div>
      <div class="metric-value {val_class}">{conf*100:.1f}%</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

    # ── Confidence bar ────────────────────────────────────────────────────
    st.markdown("#### Confidence Breakdown")
    c1, c2 = st.columns(2)
    c1.metric("🟢 REAL probability", f"{probs[1].item()*100:.1f}%")
    c2.metric("🔴 FAKE probability", f"{probs[0].item()*100:.1f}%")

    # ── Artifact explanation ──────────────────────────────────────────────
    st.markdown(f"""
<div class="artifact-box">
  <div style="font-weight:600; color:#a78bfa; margin-bottom:0.4rem;">
    🔬 Detected Artifact Type: {artifact_title}
  </div>
  <div style="color:#cbd5e1; font-size:0.95rem; line-height:1.6;">
    {artifact_desc}
  </div>
  <div style="margin-top:0.8rem; font-size:0.8rem; color:#64748b;">
    Activation region: <b style="color:#94a3b8">{region.capitalize()}</b> of image
    &nbsp;|&nbsp; Model: <b style="color:#94a3b8">{model_choice}</b>
  </div>
</div>
""", unsafe_allow_html=True)

else:
    # Landing placeholder
    st.markdown("""
---
<div style="text-align:center; padding:3rem 0; color:#475569;">
  <div style="font-size:4rem; margin-bottom:1rem;">🖼️</div>
  <div style="font-size:1.2rem; font-weight:600; color:#94a3b8;">Upload an image to get started</div>
  <div style="font-size:0.9rem; margin-top:0.5rem;">
    The model will classify it as <b>REAL</b> or <b>AI-GENERATED</b><br>
    and highlight the visual artifacts that influenced the decision.
  </div>
</div>
""", unsafe_allow_html=True)
