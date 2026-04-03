from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class ArtifactSignals:
    # All values are unitless ratios / normalized scalars intended for explanation, not ground-truth physics.
    hf_energy_ratio: float
    lap_var: float
    grad_mag_mean: float
    color_bgr_std_mean: float


def _to_bgr_uint8(rgb_chw: np.ndarray) -> np.ndarray:
    # rgb_chw: float32/float64 in [0,1] or [0,255]
    x = np.asarray(rgb_chw)
    if x.ndim != 3 or x.shape[0] != 3:
        raise ValueError(f"Expected CHW RGB with 3 channels, got shape={x.shape}")

    if x.max() <= 1.0 + 1e-3:
        x = (np.clip(x, 0.0, 1.0) * 255.0).astype(np.uint8)
    else:
        x = np.clip(x, 0.0, 255.0).astype(np.uint8)

    # CHW -> HWC
    x = np.transpose(x, (1, 2, 0))
    return cv2.cvtColor(x, cv2.COLOR_RGB2BGR)


def compute_artifact_signals(rgb_chw: np.ndarray, *, hf_radius_ratio: float = 0.12) -> ArtifactSignals:
    """Compute lightweight, interpretable signals correlated with many GAN/GenAI artifacts."""
    bgr = _to_bgr_uint8(rgb_chw)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    gray_f = gray.astype(np.float32) / 255.0

    # High-frequency energy ratio via FFT radial mask.
    h, w = gray_f.shape
    yy, xx = np.indices((h, w))
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    r_norm = r / (r.max() + 1e-8)

    spec = np.fft.fftshift(np.fft.fft2(gray_f))
    ps = (np.abs(spec) ** 2).astype(np.float32)
    total = float(ps.sum() + 1e-8)
    hf_mask = r_norm >= (1.0 - float(hf_radius_ratio))
    hf = float(ps[hf_mask].sum() / total)

    # Texture / edge "crispness" proxies.
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    lap_var = float(lap.var())

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    grad_mag_mean = float(mag.mean())

    # Simple lighting/color consistency proxy: mean channel std in BGR.
    b, g, _r = cv2.split(bgr)
    color_bgr_std_mean = float((b.std() + g.std() + _r.std()) / 3.0)

    return ArtifactSignals(
        hf_energy_ratio=hf,
        lap_var=lap_var,
        grad_mag_mean=grad_mag_mean,
        color_bgr_std_mean=color_bgr_std_mean,
    )


def explain_signals(s: ArtifactSignals) -> list[str]:
    """Turn raw signals into short, human-readable bullets (heuristic explanations)."""
    notes: list[str] = []

    # Heuristic thresholds: intentionally loose; this is not a second classifier.
    if s.hf_energy_ratio >= 0.35:
        notes.append(
            "Frequency-domain check: elevated high-frequency energy vs typical photos "
            "(often consistent with sharpening/upsampling or generative texture grain)."
        )
    elif s.hf_energy_ratio <= 0.18:
        notes.append(
            "Frequency-domain check: unusually low high-frequency energy "
            "(sometimes consistent with heavy smoothing or compression-like appearance)."
        )

    if s.lap_var >= 250.0:
        notes.append("Texture check: high local contrast variability (possible unnatural micro-texture patterns).")
    elif s.lap_var <= 80.0:
        notes.append("Texture check: unusually smooth micro-texture (possible oversmoothing or plastic skin/backgrounds).")

    if s.grad_mag_mean >= 28.0:
        notes.append("Edge check: strong edge density (possible boundary artifacts, hair/fence inconsistencies).")
    elif s.grad_mag_mean <= 14.0:
        notes.append("Edge check: weak edge structure (possible blur or painterly flattening).")

    if s.color_bgr_std_mean >= 48.0:
        notes.append("Lighting/color check: high per-channel variation (possible inconsistent lighting or color banding).")

    if not notes:
        notes.append("Heuristic artifact cues are mild; rely primarily on the model's Grad-CAM focus regions.")

    return notes
