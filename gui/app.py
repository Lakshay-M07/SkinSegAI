"""
SkinSegAI — Visual Analysis Workstation
========================================
Wide-layout Streamlit application for skin lesion segmentation,
single-pass uncertainty visualisation, GrabCut comparison,
morphology metrics, and downloadable reports.

Layout
------
Sidebar : checkpoint selector · image source · threshold presets ·
          overlay style · uncertainty toggles · disclaimer
Main    : Tab 1 — Visual Comparison
          Tab 2 — Uncertainty & Error Analysis
          Tab 3 — Morphology & Metrics
          + Training Analytics (sidebar nav)
          + About Project    (sidebar nav)
"""

from __future__ import annotations

import csv
import hashlib
import io
import logging
import sys
import time
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image, ImageDraw, UnidentifiedImageError


# ============================================================
# PROJECT ROOT / PATH BOOTSTRAP
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.segmentation.unet_segmenter import UNetSegmenter      # noqa: E402
from src.segmentation.grabcut_segmenter import GrabCutSegmenter  # noqa: E402
from src.services.segmentation_service import SegmentationService  # noqa: E402
from src.services.report_exporter import ReportExporter         # noqa: E402
from src.analysis.lesion_analyzer import LesionAnalyzer         # noqa: E402

logger = logging.getLogger(__name__)

# ============================================================
# CONSTANTS
# ============================================================

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_IMAGE_PIXELS = 12_000_000

# Curated validation samples (stem names — covers low-contrast,
# hairy-lesion, and high-contrast examples across the split)
CURATED_SAMPLES: tuple[str, ...] = (
    "ISIC_0000000",
    "ISIC_0000003",
    "ISIC_0000014",
    "ISIC_0000022",
    "ISIC_0000025",
)

ALGO_COLORS = {
    "U-Net":   "#06B6D4",
    "GrabCut": "#F59E0B",
}

THRESHOLD_PRESETS: dict[str, float] = {
    "High Sensitivity (0.35)": 0.35,
    "Balanced (0.50)":         0.50,
    "High Precision (0.65)":   0.65,
    "Manual":                  -1.0,   # sentinel — use slider
}

DISCLAIMER = (
    "Research use only — SkinSegAI is an academic segmentation tool and "
    "not a medical diagnostic system."
)


# ============================================================
# CACHED RESOURCE / DATA LOADERS
# ============================================================

@st.cache_resource
def load_segmentation_service(checkpoint_path: str):
    """Load U-Net + GrabCut once; re-use across reruns."""
    unet    = UNetSegmenter(checkpoint_path)
    grabcut = GrabCutSegmenter()
    service = SegmentationService([unet, grabcut])
    return unet, service


@st.cache_data(show_spinner=False)
def run_unet_inference(
    _segmenter,          # leading underscore → not hashed by Streamlit
    image_bytes: bytes,
    mc_passes: int,
    threshold: float,
) -> dict:
    """Single-pass (or MC Dropout) U-Net inference; cached by image+params."""
    _segmenter.threshold = threshold
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return _segmenter.predict_with_uncertainty(img, mc_passes=mc_passes)


@st.cache_data(show_spinner=False)
def run_grabcut_inference(image_bytes: bytes):
    """GrabCut segmentation; cached so only runs when image changes."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    grabcut = GrabCutSegmenter()
    t0 = time.perf_counter()
    mask = grabcut.segment(img)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return mask, elapsed_ms


@st.cache_data(show_spinner=False)
def load_csv(path_str: str) -> pd.DataFrame | None:
    path = Path(path_str)
    if not path.is_file():
        return None
    try:
        df = pd.read_csv(path)
        return df if not df.empty else None
    except (OSError, UnicodeError, pd.errors.ParserError,
            pd.errors.EmptyDataError):
        logger.exception("CSV loading failed: %s", path)
        return None


# ============================================================
# IMAGE HELPERS
# ============================================================

def decode_image(data: bytes, mode: str = "RGB") -> Image.Image:
    file_type = "ground-truth mask" if mode == "L" else "image"
    if not data:
        raise ValueError(f"The uploaded {file_type} is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise ValueError("File exceeds the 20 MB upload limit.")
    try:
        with Image.open(io.BytesIO(data)) as img:
            if img.format not in {"JPEG", "PNG"}:
                raise ValueError("Unsupported format. Upload JPEG or PNG.")
            if mode == "L" and img.format != "PNG":
                raise ValueError("Ground-truth masks must be PNG files.")
            if img.width * img.height > MAX_IMAGE_PIXELS:
                raise ValueError("Image exceeds the 12-megapixel limit.")
            img.load()
            return img.convert(mode)
    except Image.DecompressionBombError:
        raise ValueError("Image dimensions are too large.") from None
    except (UnidentifiedImageError, OSError, SyntaxError):
        raise ValueError(
            f"The {file_type} could not be opened. "
            + ("Upload a readable PNG mask." if mode == "L"
               else "Upload a readable JPEG or PNG.")
        ) from None


def image_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def pil_to_bytes(img: Image.Image, fmt: str = "PNG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def style_plotly_figure(fig):
    """Apply transparent dark clinical-slate styling to Plotly charts."""
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#F1F5F9"),
        margin=dict(l=24, r=16, t=32, b=24),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(color="#CBD5E1"),
        ),
    )
    fig.update_xaxes(
        gridcolor="rgba(255,255,255,0.06)",
        zerolinecolor="rgba(255,255,255,0.08)",
    )
    fig.update_yaxes(
        gridcolor="rgba(255,255,255,0.06)",
        zerolinecolor="rgba(255,255,255,0.08)",
    )
    return fig


def inject_custom_css():
    """Inject clinical-imaging workstation CSS."""
    st.markdown(
        """
        <style>
            /* Hide right-side Streamlit toolbar/menu/deploy chrome while keeping sidebar reopen control clickable */
            [data-testid="stToolbarActions"],
            [data-testid="stAppDeployButton"],
            [data-testid="stMainMenu"],
            [data-testid="stDecoration"],
            [data-testid="stStatusWidget"],
            [data-testid="stFooter"],
            footer,
            #MainMenu,
            .stDeployButton {
                display: none !important;
            }
            [data-testid="stHeader"],
            [data-testid="stToolbar"] {
                background: transparent !important;
                pointer-events: none !important;
                height: 2.25rem;
            }
            [data-testid="stSidebarCollapsedControl"],
            [data-testid="stSidebarCollapsedControl"] button,
            [data-testid="stExpandSidebarButton"],
            [data-testid="stExpandSidebarButton"] button {
                visibility: visible !important;
                pointer-events: auto !important;
                opacity: 1 !important;
            }
            /* Tighten top page padding for widescreen above-the-fold fit */
            .block-container {
                padding-top: 1.0rem !important;
                padding-bottom: 1.5rem !important;
                max-width: 1440px;
            }
            /* Elevated dark metric cards */
            [data-testid="stMetric"] {
                background: #131A2A;
                border: 1px solid rgba(255, 255, 255, 0.08);
                border-radius: 12px;
                padding: 0.85rem 1.05rem;
            }
            [data-testid="stMetricLabel"] {
                color: #94A3B8;
                font-size: 0.78rem;
                letter-spacing: 0.03em;
            }
            [data-testid="stMetricValue"] {
                font-variant-numeric: tabular-nums;
                color: #F1F5F9;
            }
            [data-testid="stMetricDelta"] {
                font-variant-numeric: tabular-nums;
            }
            /* Pill-style navigation tabs */
            [data-baseweb="tab-list"] {
                gap: 0.5rem;
                background: #131A2A;
                border: 1px solid rgba(255, 255, 255, 0.08);
                border-radius: 12px;
                padding: 0.3rem;
            }
            [data-baseweb="tab"] {
                border-radius: 8px !important;
                padding: 0.45rem 1rem !important;
                color: #94A3B8 !important;
                font-weight: 500;
                border: none !important;
            }
            [data-baseweb="tab"][aria-selected="true"] {
                background: rgba(6, 182, 212, 0.16) !important;
                color: #06B6D4 !important;
                border: 1px solid rgba(6, 182, 212, 0.35) !important;
            }
            [data-baseweb="tab-highlight"] {
                display: none !important;
            }
            /* Top header banner */
            .skinseg-banner {
                display: flex;
                flex-wrap: wrap;
                align-items: center;
                justify-content: space-between;
                gap: 1rem;
                background: #131A2A;
                border: 1px solid rgba(255, 255, 255, 0.08);
                border-radius: 12px;
                padding: 0.9rem 1.25rem;
                margin-bottom: 1rem;
            }
            .skinseg-banner-title {
                margin: 0;
                font-size: 1.35rem;
                font-weight: 700;
                color: #F1F5F9;
                letter-spacing: -0.01em;
            }
            .skinseg-banner-sub {
                margin: 0.15rem 0 0 0;
                font-size: 0.85rem;
                color: #94A3B8;
            }
            .skinseg-badges {
                display: flex;
                flex-wrap: wrap;
                gap: 0.5rem;
                align-items: center;
            }
            .skinseg-badge {
                display: inline-flex;
                align-items: center;
                gap: 0.35rem;
                padding: 0.28rem 0.68rem;
                border-radius: 999px;
                font-size: 0.76rem;
                font-weight: 600;
                font-variant-numeric: tabular-nums;
                background: rgba(255, 255, 255, 0.04);
                border: 1px solid rgba(255, 255, 255, 0.08);
                color: #CBD5E1;
            }
            .skinseg-badge--accent {
                background: rgba(6, 182, 212, 0.12);
                border-color: rgba(6, 182, 212, 0.32);
                color: #22D3EE;
            }
            /* Standardized uppercase panel headers & aligned image frames */
            .panel-card-header {
                font-size: 0.72rem;
                font-weight: 700;
                letter-spacing: 0.08em;
                text-transform: uppercase;
                color: #94A3B8;
                background: #131A2A;
                border: 1px solid rgba(255, 255, 255, 0.08);
                border-bottom: none;
                border-radius: 10px 10px 0 0;
                padding: 0.45rem 0.75rem;
            }
            .panel-card-header + div[data-testid="stImage"] {
                background: #131A2A;
                border: 1px solid rgba(255, 255, 255, 0.08);
                border-radius: 0 0 10px 10px;
                padding: 0.5rem;
            }
            .panel-card-header + div[data-testid="stImage"] img {
                height: 340px;
                width: 100%;
                object-fit: contain;
                border-radius: 6px;
            }
            /* Subtle footer callout */
            .skinseg-footer {
                margin-top: 1.5rem;
                padding: 0.7rem 1rem;
                border-top: 1px solid rgba(255, 255, 255, 0.08);
                color: #64748B;
                font-size: 0.78rem;
                text-align: center;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_top_banner(settings: dict):
    """Render the top clinical-imaging header banner with status badges."""
    unet_model = settings.get("unet_model")
    arch_label = getattr(unet_model, "arch", "unavailable") if unet_model else "unavailable"
    device_obj = getattr(unet_model, "device", None) if unet_model else None
    device_label = str(device_obj).upper() if device_obj is not None else "CPU"
    mode_label = settings.get("dataset_mode", "Sample Gallery")

    st.markdown(
        f"""
        <div class="skinseg-banner">
            <div>
                <div class="skinseg-banner-title">SkinSegAI</div>
                <div class="skinseg-banner-sub">
                    Clinical-Imaging Lesion Segmentation &amp; Uncertainty Workstation
                </div>
            </div>
            <div class="skinseg-badges">
                <span class="skinseg-badge skinseg-badge--accent">Model: {arch_label}</span>
                <span class="skinseg-badge">Device: {device_label}</span>
                <span class="skinseg-badge">Source: {mode_label}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# VISUALISATION HELPERS
# ============================================================

_ANALYZER = LesionAnalyzer()


def build_overlay(
    image: Image.Image,
    mask: np.ndarray,
    alpha: float,
    style: str,
    color_hex: str,
) -> Image.Image:
    """Return a filled-mask or contour-outline overlay."""
    r, g, b = int(color_hex[1:3], 16), int(color_hex[3:5], 16), int(color_hex[5:7], 16)
    if style == "Filled Translucent Mask":
        return _ANALYZER.create_overlay(image, mask, alpha=alpha, color=(r, g, b))
    # Crisp Contour Outline
    arr = np.array(image.convert("RGB"), dtype=np.uint8)
    binary = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(
        binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )
    thickness = max(2, round(min(arr.shape[:2]) / 128))
    cv2.drawContours(arr, contours, -1, (r, g, b), thickness)
    return Image.fromarray(arr)


def compute_error_map(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
) -> Image.Image:
    """TP=green  FP=red  FN=blue  TN=black."""
    h, w = pred_mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    pred = pred_mask.astype(bool)
    gt   = gt_mask.astype(bool)
    rgb[pred & gt]   = [0,   200,   0]   # TP  green
    rgb[pred & ~gt]  = [220,   0,   0]   # FP  red
    rgb[~pred & gt]  = [0,     0, 220]   # FN  blue
    return Image.fromarray(rgb)


def side_by_side_montage(
    images: list[tuple[str, Image.Image]],
    gap: int = 8,
) -> Image.Image:
    """Horizontal montage with captions for the downloadable PNG."""
    max_h = max(img.height for _, img in images)
    total_w = sum(img.width for _, img in images) + gap * (len(images) - 1)
    canvas = Image.new("RGB", (total_w, max_h + 24), (20, 30, 48))
    draw   = ImageDraw.Draw(canvas)
    x = 0
    for label, img in images:
        canvas.paste(img, (x, 24))
        draw.text((x + 4, 4), label, fill=(200, 220, 240))
        x += img.width + gap
    return canvas


# ============================================================
# CHECKPOINT DISCOVERY
# ============================================================

def discover_checkpoints() -> list[Path]:
    ckpt_dir = ROOT / "artifacts" / "checkpoints"
    if not ckpt_dir.is_dir():
        return []
    return sorted(ckpt_dir.glob("*.pth"))


@st.cache_resource
def resolve_default_checkpoint() -> str | None:
    """
    Automatically pick the best trained checkpoint in artifacts/.
    Prioritizes the latest resnet34_unet checkpoint if present,
    falling back to custom_unet (or best_model.pth / newest .pth).
    """
    candidates = discover_checkpoints()
    if not candidates:
        artifacts_dir = ROOT / "artifacts"
        if artifacts_dir.is_dir():
            candidates = sorted(artifacts_dir.rglob("*.pth"))
    if not candidates:
        return None

    import torch

    resnet_ckpts: list[tuple[float, Path]] = []
    custom_ckpts: list[tuple[float, Path]] = []
    other_ckpts: list[tuple[float, Path]] = []

    for path in candidates:
        mtime = path.stat().st_mtime
        try:
            meta = torch.load(str(path), map_location="cpu", weights_only=False)
            arch = meta.get("arch", "custom_unet") if isinstance(meta, dict) else "custom_unet"
        except Exception:
            arch = "custom_unet"

        if arch == "resnet34_unet":
            resnet_ckpts.append((mtime, path))
        elif arch == "custom_unet":
            custom_ckpts.append((mtime, path))
        else:
            other_ckpts.append((mtime, path))

    if resnet_ckpts:
        resnet_ckpts.sort(key=lambda item: item[0], reverse=True)
        return str(resnet_ckpts[0][1])

    best_default = ROOT / "artifacts" / "checkpoints" / "best_model.pth"
    if best_default.is_file():
        return str(best_default)

    if custom_ckpts:
        custom_ckpts.sort(key=lambda item: item[0], reverse=True)
        return str(custom_ckpts[0][1])

    other_ckpts.sort(key=lambda item: item[0], reverse=True)
    return str(other_ckpts[0][1])


# ============================================================
# SAMPLE IMAGE SELECTOR
# ============================================================

def _load_sample_bytes(stem: str) -> tuple[bytes | None, bytes | None]:
    """Return (image_bytes, mask_bytes) for a curated sample, or (None, None)."""
    img_path  = ROOT / "data/processed_grouped/validation/images" / f"{stem}.png"
    mask_path = ROOT / "data/processed_grouped/validation/masks"  / f"{stem}_segmentation.png"
    if not img_path.is_file():
        return None, None
    img_bytes  = img_path.read_bytes()
    mask_bytes = mask_path.read_bytes() if mask_path.is_file() else None
    return img_bytes, mask_bytes


# ============================================================
# SIDEBAR BUILDER
# ============================================================

def build_sidebar() -> dict:
    """
    Render the sidebar and return a settings dict consumed by the main view.
    Keys: checkpoint_path, image_bytes, mask_bytes, source_label, dataset_mode,
          threshold, opacity, overlay_style, overlay_color,
          show_uncertainty, mc_passes, unet_model (or None).
    """
    st.sidebar.title("SkinSegAI")
    st.sidebar.caption("Lesion Segmentation Controls")
    st.sidebar.divider()

    # ── Auto-load default model checkpoint (cached) ──────────
    checkpoint_path = resolve_default_checkpoint()
    unet_model = None
    if checkpoint_path is None:
        st.sidebar.error("No trained model checkpoint found in artifacts/.")
    else:
        try:
            unet_model, _ = load_segmentation_service(checkpoint_path)
        except Exception:
            logger.exception("Failed to load default checkpoint.")
            unet_model = None

    # ── 1. Image Source ──────────────────────────────────────
    st.sidebar.subheader("Image Source")
    source = st.sidebar.radio(
        "Source",
        ["Sample Gallery", "Custom Upload"],
        label_visibility="collapsed",
    )

    image_bytes: bytes | None = None
    mask_bytes:  bytes | None = None
    source_label = ""

    if source == "Custom Upload":
        up_img = st.sidebar.file_uploader(
            "Upload dermoscopic image",
            type=["jpg", "jpeg", "png"],
            key="img_upload",
        )
        up_gt = st.sidebar.file_uploader(
            "Ground-truth mask (optional PNG)",
            type=["png"],
            key="gt_upload",
        )
        if up_img is not None:
            image_bytes  = up_img.getvalue()
            source_label = up_img.name
        if up_gt is not None:
            mask_bytes = up_gt.getvalue()
        st.sidebar.caption("JPEG/PNG · max 20 MB · max 12 MP")
    else:
        available = [(s, _load_sample_bytes(s)) for s in CURATED_SAMPLES]
        available = [(s, pair) for s, pair in available if pair[0] is not None]
        if not available:
            st.sidebar.warning(
                "Sample images not found. "
                "Ensure data/processed_grouped/validation/images/ exists."
            )
        else:
            sample_names = [s for s, _ in available]
            sel_sample = st.sidebar.selectbox(
                "Select sample image",
                sample_names,
                format_func=lambda s: f"{s}  (256×256 validation)",
            )
            image_bytes, mask_bytes = dict(available)[sel_sample]
            source_label = sel_sample

    st.sidebar.divider()

    # ── 2. Model & Threshold Controls ────────────────────────
    st.sidebar.subheader("Model & Threshold Controls")
    preset_choice = st.sidebar.radio(
        "Threshold preset",
        list(THRESHOLD_PRESETS.keys()),
        index=1,  # Balanced default
    )
    if THRESHOLD_PRESETS[preset_choice] < 0:
        threshold = st.sidebar.slider(
            "Manual threshold",
            min_value=0.10,
            max_value=0.90,
            value=0.50,
            step=0.05,
        )
    else:
        threshold = THRESHOLD_PRESETS[preset_choice]
        st.sidebar.caption(f"Threshold set to **{threshold:.2f}**")

    st.sidebar.divider()

    # ── 3. Advanced Display & Uncertainty Settings ───────────
    with st.sidebar.expander("Advanced Display & Uncertainty Settings", expanded=False):
        overlay_style = st.radio(
            "Overlay style",
            ["Filled Translucent Mask", "Crisp Contour Outline"],
        )
        overlay_color = st.color_picker(
            "Mask / outline colour", value="#06B6D4"
        )
        opacity = st.slider(
            "Opacity / alpha",
            min_value=0.05,
            max_value=0.95,
            value=0.35,
            step=0.05,
        )
        show_uncertainty = st.toggle(
            "Show Uncertainty Heatmap",
            value=True,
            help=(
                "Display pixel-wise predictive entropy [0=certain, 1=ambiguous] "
                "and Border Uncertainty Score."
            ),
        )
        mc_passes = 1
        if unet_model is not None and unet_model.has_dropout:
            use_mc = st.toggle(
                "Enable MC Dropout",
                value=False,
                help=(
                    "Run N stochastic forward passes with Dropout active while "
                    "keeping BatchNorm frozen."
                ),
            )
            if use_mc:
                mc_passes = st.slider(
                    "MC Dropout passes", min_value=2, max_value=20, value=10, step=1
                )

    return {
        "checkpoint_path": checkpoint_path,
        "image_bytes":     image_bytes,
        "mask_bytes":      mask_bytes,
        "source_label":    source_label,
        "dataset_mode":    source,
        "threshold":       threshold,
        "opacity":         opacity,
        "overlay_style":   overlay_style,
        "overlay_color":   overlay_color,
        "show_uncertainty": show_uncertainty,
        "mc_passes":       mc_passes,
        "unet_model":      unet_model,
    }


# ============================================================
# TAB 1 — VISUAL COMPARISON
# ============================================================

def render_tab_visual(
    image: Image.Image,
    unet_result: dict,
    grabcut_mask: np.ndarray,
    settings: dict,
):
    alpha   = settings["opacity"]
    style   = settings["overlay_style"]
    color   = settings["overlay_color"]

    unet_mask = unet_result["binary_mask"]
    unet_overlay    = build_overlay(image, unet_mask,    alpha, style, color)
    grabcut_overlay = build_overlay(image, grabcut_mask, alpha, style, color)

    col_orig, col_unet, col_gc = st.columns(3)
    with col_orig:
        st.markdown('<div class="panel-card-header">ORIGINAL INPUT</div>', unsafe_allow_html=True)
        st.image(image, use_container_width=True)
    with col_unet:
        st.markdown('<div class="panel-card-header">U-NET PREDICTION</div>', unsafe_allow_html=True)
        st.image(unet_overlay, use_container_width=True)
    with col_gc:
        st.markdown('<div class="panel-card-header">GRABCUT BASELINE</div>', unsafe_allow_html=True)
        st.image(grabcut_overlay, use_container_width=True)

    # Store overlays for montage download
    st.session_state["_overlays"] = (unet_overlay, grabcut_overlay)


# ============================================================
# TAB 2 — UNCERTAINTY & ERROR ANALYSIS
# ============================================================

def render_tab_uncertainty(
    image: Image.Image,
    unet_result: dict,
    grabcut_mask: np.ndarray,
    gt_mask: np.ndarray | None,
    settings: dict,
):
    uncertainty_map = unet_result.get("uncertainty_map")
    unet_mask       = unet_result["binary_mask"]

    if not settings["show_uncertainty"] or uncertainty_map is None:
        st.info(
            "Uncertainty heatmap is disabled. "
            "Enable **Show Uncertainty Heatmap** in the sidebar."
        )
    else:
        st.subheader("Pixel-wise Predictive Entropy")
        heatmap_pil, heat_overlay_pil = _ANALYZER.create_uncertainty_heatmap(
            image, uncertainty_map, alpha=max(0.35, settings["opacity"])
        )
        border_score = _ANALYZER.compute_border_uncertainty(unet_mask, uncertainty_map)

        u1, u2 = st.columns(2)
        u1.image(heat_overlay_pil,
                 caption="Uncertainty overlay (inferno)",
                 use_container_width=True)
        u2.image(heatmap_pil,
                 caption="Entropy map  [0 = certain  →  1 = ambiguous]",
                 use_container_width=True)

        st.metric(
            "Border Uncertainty Score",
            f"{border_score:.4f}",
            help=(
                "Mean predictive entropy within a dilated ring around "
                "the predicted lesion boundary.  Lower = more confident border."
            ),
        )

        st.session_state["_heatmap"] = heatmap_pil

    st.divider()

    # ── TP / FP / FN error map ───────────────────────────────
    if gt_mask is None:
        st.info(
            "Upload a ground-truth mask to see the TP / FP / FN error map."
        )
    else:
        st.subheader("Error Map (U-Net vs Ground Truth)")
        error_img = compute_error_map(unet_mask, gt_mask)
        st.image(error_img, caption="🟢 TP  🔴 FP  🔵 FN", use_container_width=True)

        legend_md = (
            "**Green** = True Positive (correctly predicted lesion)  \n"
            "**Red** = False Positive (over-segmented background)  \n"
            "**Blue** = False Negative (missed lesion pixels)  \n"
            "**Black** = True Negative"
        )
        st.caption(legend_md)

        # Quick pixel counts
        pred = unet_mask.astype(bool)
        gt   = gt_mask.astype(bool)
        tp   = int(np.count_nonzero(pred & gt))
        fp   = int(np.count_nonzero(pred & ~gt))
        fn   = int(np.count_nonzero(~pred & gt))
        tn   = int(np.count_nonzero(~pred & ~gt))
        px_total = unet_mask.size

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("TP pixels", f"{tp:,}",  f"{100*tp/px_total:.1f}%")
        c2.metric("FP pixels", f"{fp:,}",  f"{100*fp/px_total:.1f}%")
        c3.metric("FN pixels", f"{fn:,}",  f"{100*fn/px_total:.1f}%")
        c4.metric("TN pixels", f"{tn:,}",  f"{100*tn/px_total:.1f}%")


# ============================================================
# TAB 3 — MORPHOLOGY & METRICS
# ============================================================

def render_tab_metrics(
    image: Image.Image,
    unet_result: dict,
    grabcut_mask: np.ndarray,
    unet_ms: float,
    grabcut_ms: float,
    gt_mask: np.ndarray | None,
    settings: dict,
):
    unet_mask = unet_result["binary_mask"]

    unet_m    = _ANALYZER.analyze(unet_mask)
    grabcut_m = _ANALYZER.analyze(grabcut_mask)

    # ── KPI cards ────────────────────────────────────────────
    if gt_mask is not None:
        st.subheader("Segmentation Quality (vs Ground Truth)")
        from src.segmentation.unet_segmenter import UNetSegmenter as _U
        # compute dice / iou / precision / recall inline
        def _metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
            pred_b = pred.astype(bool)
            gt_b   = gt.astype(bool)
            inter  = np.count_nonzero(pred_b & gt_b)
            pred_s = np.count_nonzero(pred_b)
            gt_s   = np.count_nonzero(gt_b)
            eps    = 1e-7
            dice      = (2 * inter + eps) / (pred_s + gt_s + eps)
            iou       = (inter + eps) / (pred_s + gt_s - inter + eps)
            precision = (inter + eps) / (pred_s + eps)
            recall    = (inter + eps) / (gt_s   + eps)
            return dict(dice=dice, iou=iou, precision=precision, recall=recall)

        um = _metrics(unet_mask, gt_mask)
        gm = _metrics(grabcut_mask, gt_mask)

        kpi_cols = st.columns(5)
        kpi_cols[0].metric("U-Net Dice",      f"{um['dice']:.4f}",
                           delta=f"{um['dice']-gm['dice']:+.4f} vs GrabCut")
        kpi_cols[1].metric("U-Net IoU",       f"{um['iou']:.4f}",
                           delta=f"{um['iou']-gm['iou']:+.4f} vs GrabCut")
        kpi_cols[2].metric("Precision",       f"{um['precision']:.4f}")
        kpi_cols[3].metric("Recall",          f"{um['recall']:.4f}")
        kpi_cols[4].metric("GrabCut Dice",    f"{gm['dice']:.4f}")

    # ── Inference time ────────────────────────────────────────
    st.subheader("Inference Latency")
    t1, t2 = st.columns(2)
    t1.metric("U-Net", f"{unet_ms:.1f} ms",
              delta=f"{unet_ms - grabcut_ms:+.1f} ms vs GrabCut",
              delta_color="inverse")
    t2.metric("GrabCut", f"{grabcut_ms:.1f} ms")

    st.divider()

    # ── Shape metrics table & Plotly chart ───────────────────
    st.subheader("Morphology Comparison")

    shape_rows = [
        {
            "Algorithm":    "U-Net",
            "Area (px²)":   unet_m["area_pixels"],
            "Coverage (%)": round(unet_m["area_percent"], 2),
            "Perimeter":    round(unet_m["perimeter_pixels"], 1),
            "Circularity":  round(unet_m["circularity"], 4) if unet_m["circularity"] else None,
            "Asymmetry":    round(unet_m["asymmetry"],   4) if unet_m["asymmetry"]   else None,
        },
        {
            "Algorithm":    "GrabCut",
            "Area (px²)":   grabcut_m["area_pixels"],
            "Coverage (%)": round(grabcut_m["area_percent"], 2),
            "Perimeter":    round(grabcut_m["perimeter_pixels"], 1),
            "Circularity":  round(grabcut_m["circularity"], 4) if grabcut_m["circularity"] else None,
            "Asymmetry":    round(grabcut_m["asymmetry"],   4) if grabcut_m["asymmetry"]   else None,
        },
    ]
    shape_df = pd.DataFrame(shape_rows)
    st.dataframe(shape_df, hide_index=True)

    # Plotly grouped bar — numeric shape metrics
    chart_vars = ["Coverage (%)", "Circularity", "Asymmetry"]
    chart_vars_avail = [
        v for v in chart_vars if shape_df[v].notna().any()
    ]
    if chart_vars_avail:
        melted = shape_df.melt(
            id_vars="Algorithm",
            value_vars=chart_vars_avail,
            var_name="Metric",
            value_name="Value",
        )
        fig = px.bar(
            melted,
            x="Metric",
            y="Value",
            color="Algorithm",
            barmode="group",
            color_discrete_map=ALGO_COLORS,
            text_auto=".3f",
            labels={"Value": "Score / Ratio"},
        )
        fig.update_layout(legend_title_text="Algorithm")
        st.plotly_chart(style_plotly_figure(fig), use_container_width=True)

    st.divider()

    # ── Downloads ─────────────────────────────────────────────
    st.subheader("Export")
    dl1, dl2 = st.columns(2)

    # Side-by-side PNG montage
    montage_parts = [("Original", image.convert("RGB"))]
    if "_overlays" in st.session_state:
        u_ov, g_ov = st.session_state["_overlays"]
        montage_parts += [("U-Net", u_ov.convert("RGB")),
                          ("GrabCut", g_ov.convert("RGB"))]
    if "_heatmap" in st.session_state:
        montage_parts.append(("Uncertainty", st.session_state["_heatmap"].convert("RGB")))
    montage = side_by_side_montage(montage_parts)
    dl1.download_button(
        "⬇ Download Visual Comparison PNG",
        data=pil_to_bytes(montage),
        file_name="SkinSegAI_comparison.png",
        mime="image/png",
    )

    # CSV metrics
    csv_rows = [
        {
            "algorithm": "U-Net",
            "inference_ms": round(unet_ms, 2),
            **{k: v for k, v in shape_rows[0].items() if k != "Algorithm"},
            **({"dice": round(um["dice"], 4), "iou": round(um["iou"], 4),
                "precision": round(um["precision"], 4),
                "recall":    round(um["recall"], 4)}
               if gt_mask is not None else {}),
        },
        {
            "algorithm": "GrabCut",
            "inference_ms": round(grabcut_ms, 2),
            **{k: v for k, v in shape_rows[1].items() if k != "Algorithm"},
            **({"dice": round(gm["dice"], 4), "iou": round(gm["iou"], 4),
                "precision": round(gm["precision"], 4),
                "recall":    round(gm["recall"], 4)}
               if gt_mask is not None else {}),
        },
    ]
    csv_str = pd.DataFrame(csv_rows).to_csv(index=False)
    dl2.download_button(
        "⬇ Download Metrics CSV",
        data=csv_str,
        file_name="SkinSegAI_metrics.csv",
        mime="text/csv",
    )


# ============================================================
# WORKSTATION PAGE  (main 3-tab view)
# ============================================================

def _workstation_page(settings: dict):
    image_bytes = settings["image_bytes"]
    if image_bytes is None:
        st.info("👈  Select an image source in the sidebar to begin.")
        return

    if settings["checkpoint_path"] is None:
        st.error("No checkpoint selected. Add a .pth file to artifacts/checkpoints/")
        return

    # ── Decode image + optional GT mask ─────────────────────
    try:
        image = decode_image(image_bytes)
        gt_mask: np.ndarray | None = None
        if settings["mask_bytes"] is not None:
            gt_pil  = decode_image(settings["mask_bytes"], mode="L")
            if gt_pil.size != image.size:
                st.error("Image and ground-truth mask dimensions must match.")
                return
            arr = np.asarray(gt_pil)
            if not set(np.unique(arr).tolist()).issubset({0, 1, 255}):
                st.error("Ground-truth mask must be binary (0/255).")
                return
            gt_mask = arr > 0
    except ValueError as e:
        st.error(str(e))
        return

    # ── Run inference (cached) ───────────────────────────────
    unet_model: UNetSegmenter = settings["unet_model"]
    if unet_model is None:
        st.error("Model could not be loaded. Check the checkpoint file.")
        return

    with st.spinner("Running U-Net inference…"):
        t0 = time.perf_counter()
        unet_result = run_unet_inference(
            unet_model,
            image_bytes,
            settings["mc_passes"],
            settings["threshold"],
        )
        unet_ms = (time.perf_counter() - t0) * 1000

    with st.spinner("Running GrabCut…"):
        cv2.setRNGSeed(42)
        grabcut_mask, grabcut_ms = run_grabcut_inference(image_bytes)

    # ── Preview strip ─────────────────────────────────────────
    with st.expander("Input image", expanded=False):
        c1, c2 = st.columns([2, 1])
        c1.image(image, caption=f"{image.width}×{image.height} px  ·  {settings['source_label']}")
        if gt_mask is not None:
            c2.image(
                gt_mask.astype(np.uint8) * 255,
                caption="Ground-truth mask",
            )

    # ── 3 main tabs ───────────────────────────────────────────
    tab_vis, tab_unc, tab_met = st.tabs([
        "🖼  Visual Comparison",
        "🌡  Uncertainty & Error",
        "📐  Morphology & Metrics",
    ])

    with tab_vis:
        render_tab_visual(image, unet_result, grabcut_mask, settings)

    with tab_unc:
        render_tab_uncertainty(image, unet_result, grabcut_mask, gt_mask, settings)

    with tab_met:
        render_tab_metrics(
            image, unet_result, grabcut_mask,
            unet_ms, grabcut_ms, gt_mask, settings,
        )

    # ── ZIP report (full export via ReportExporter) ──────────
    st.divider()
    try:
        exporter = ReportExporter()
        # Build minimal SegmentationResult objects for the exporter
        from src.core.segmentation_result import SegmentationResult
        unet_overlay_pil = st.session_state.get(
            "_overlays", (None, None)
        )[0] or build_overlay(
            image, unet_result["binary_mask"],
            settings["opacity"], settings["overlay_style"], settings["overlay_color"]
        )
        grabcut_overlay_pil = st.session_state.get(
            "_overlays", (None, None)
        )[1] or build_overlay(
            image, grabcut_mask,
            settings["opacity"], settings["overlay_style"], settings["overlay_color"]
        )
        results_for_export = [
            SegmentationResult(
                algorithm="U-Net",
                mask=unet_result["binary_mask"],
                overlay=unet_overlay_pil,
                boundary=_ANALYZER.draw_boundary(image, unet_result["binary_mask"]),
                measurements=_ANALYZER.analyze(unet_result["binary_mask"]),
                inference_ms=unet_ms,
                prob_map=unet_result.get("prob_map"),
                uncertainty_map=unet_result.get("uncertainty_map"),
                border_uncertainty=float(
                    _ANALYZER.compute_border_uncertainty(
                        unet_result["binary_mask"],
                        unet_result["uncertainty_map"],
                    )
                ) if unet_result.get("uncertainty_map") is not None else None,
            ),
            SegmentationResult(
                algorithm="GrabCut",
                mask=grabcut_mask,
                overlay=grabcut_overlay_pil,
                boundary=_ANALYZER.draw_boundary(image, grabcut_mask),
                measurements=_ANALYZER.analyze(grabcut_mask),
                inference_ms=grabcut_ms,
            ),
        ]
        zip_bytes = exporter.export_zip(results_for_export)
        st.download_button(
            "⬇ Download Complete Analysis Report (ZIP)",
            data=zip_bytes,
            file_name="SkinSegAI_Report.zip",
            mime="application/zip",
        )
    except Exception:
        logger.exception("ZIP report generation failed.")
        st.warning("ZIP report could not be generated for this image.")


# ============================================================
# TRAINING ANALYTICS PAGE
# ============================================================

def _training_page():
    st.title("U-Net Training Analytics")

    history_path = str(ROOT / "artifacts/checkpoints/training_history.csv")
    history = load_csv(history_path)

    if history is None:
        st.error(
            "Training history not found at artifacts/checkpoints/training_history.csv. "
            "Run a training job first."
        )
        return

    required = {"epoch", "val_dice", "val_iou"}
    if not required.issubset(history.columns):
        st.error("Training history is incomplete.")
        return

    if history["val_dice"].dropna().empty:
        st.error("No valid validation Dice scores found.")
        return

    best = history.loc[history["val_dice"].idxmax()]
    cols = st.columns(4)
    cols[0].metric("Completed epochs",        len(history))
    cols[1].metric("Best validation Dice",    f"{best['val_dice']:.2%}")
    cols[2].metric("Best checkpoint epoch",   int(best["epoch"]))
    cols[3].metric("Val IoU at best epoch",   f"{best['val_iou']:.2%}")

    st.divider()

    chart_specs = [
        ("Training & Validation Loss",    ["train_loss",      "val_loss"]),
        ("Dice Coefficient",              ["train_dice",      "val_dice"]),
        ("Intersection over Union",       ["train_iou",       "val_iou"]),
        ("Precision",                     ["train_precision", "val_precision"]),
        ("Recall",                        ["train_recall",    "val_recall"]),
        ("Learning Rate",                 ["learning_rate"]),
    ]
    for title, ycols in chart_specs:
        if not set(ycols).issubset(history.columns):
            continue
        st.subheader(title)
        fig = px.line(
            history, x="epoch", y=ycols, markers=True,
            labels={"epoch": "Epoch", "value": "Value", "variable": "Metric"},
        )
        st.plotly_chart(style_plotly_figure(fig), use_container_width=True)

    with st.expander("Complete training history"):
        st.dataframe(history.round(6), hide_index=True)

    st.download_button(
        "⬇ Download Training History CSV",
        data=history.to_csv(index=False),
        file_name="training_history.csv",
        mime="text/csv",
    )


# ============================================================
# PERFORMANCE COMPARISON PAGE
# ============================================================

def _comparison_page():
    st.title("Comparative Performance Analysis")

    summary_path = str(ROOT / "artifacts/metrics/comparison_summary.csv")
    results_path = str(ROOT / "artifacts/metrics/per_image_results.csv")

    summary = load_csv(summary_path)
    results = load_csv(results_path)

    if summary is None:
        st.error(
            "Benchmark summary not found. "
            "Run scripts/evaluate_models.py first."
        )
        return

    required_s = {"algorithm", "dice", "iou", "precision", "recall", "inference_ms"}
    if not required_s.issubset(summary.columns):
        st.error("Benchmark summary is incomplete.")
        return

    st.caption("Evaluation on the held-out ISIC 2018 test split.")
    st.dataframe(summary.round(4), hide_index=True)

    # ── KPI bar chart ─────────────────────────────────────────
    st.subheader("Metric comparison")
    melted = summary.melt(
        id_vars="algorithm",
        value_vars=["dice", "iou", "precision", "recall"],
        var_name="metric",
        value_name="score",
    )
    fig = px.bar(
        melted, x="metric", y="score",
        color="algorithm", barmode="group",
        color_discrete_map=ALGO_COLORS,
        text_auto=".3f",
        labels={"metric": "Metric", "score": "Score"},
    )
    fig.update_yaxes(range=[0, 1.05])
    st.plotly_chart(style_plotly_figure(fig), use_container_width=True)

    # ── Distribution (per-image) ──────────────────────────────
    if results is not None and {"algorithm", "dice", "iou"}.issubset(results.columns):
        st.subheader("Per-image distribution")
        sel = st.selectbox("Metric", ["dice", "iou", "precision", "recall"])
        box = px.box(
            results, x="algorithm", y=sel,
            color="algorithm", color_discrete_map=ALGO_COLORS,
            points="outliers",
        )
        box.update_yaxes(range=[0, 1.05])
        st.plotly_chart(style_plotly_figure(box), use_container_width=True)

        hist = px.histogram(
            results, x=sel, color="algorithm",
            barmode="overlay", opacity=0.65, nbins=30,
            color_discrete_map=ALGO_COLORS,
        )
        st.plotly_chart(style_plotly_figure(hist), use_container_width=True)

    # ── Inference time ────────────────────────────────────────
    st.subheader("Inference time")
    t_fig = px.bar(
        summary, x="algorithm", y="inference_ms",
        color="algorithm", color_discrete_map=ALGO_COLORS,
        text_auto=".2f",
        labels={"inference_ms": "Mean inference (ms)"},
    )
    st.plotly_chart(style_plotly_figure(t_fig), use_container_width=True)

    # ── Downloads ─────────────────────────────────────────────
    c1, c2 = st.columns(2)
    c1.download_button(
        "⬇ Comparison summary CSV",
        summary.to_csv(index=False),
        file_name="comparison_summary.csv",
        mime="text/csv",
    )
    if results is not None:
        c2.download_button(
            "⬇ Per-image results CSV",
            results.to_csv(index=False),
            file_name="per_image_results.csv",
            mime="text/csv",
        )


# ============================================================
# ABOUT PAGE
# ============================================================

def _about_page():
    st.title("About SkinSegAI")
    st.markdown("""
### Project objective
Develop an object-oriented application for skin lesion segmentation using
a U-Net neural network and compare it with the GrabCut baseline.

### Dataset
ISIC 2018 Task 1 — 2,594 dermoscopic images with ground-truth segmentation masks.

### Deep learning architecture
- **Custom U-Net** (1.9 M parameters, 4-level encoder-decoder, skip connections)
- **ResNet-34 U-Net** (pretrained ImageNet encoder via `segmentation-models-pytorch`)
- **EfficientNet-B2 U-Net** (pretrained ImageNet encoder)
- Composite loss: Dice (0.4) + BCE (0.3) + Boundary (0.3)
- AdamW optimizer · cosine LR schedule · early stopping

### Uncertainty estimation
Single-pass binary predictive entropy
`H(p) = -p·log₂(p) - (1-p)·log₂(1-p)` derived directly from the sigmoid
probability map; peaks where `p ≈ 0.5` (ambiguous lesion borders).
Optional MC Dropout for models with Dropout layers.

### Object-oriented design
**Abstraction** `BaseSegmenter` · **Inheritance** `UNetSegmenter / GrabCutSegmenter`
· **Polymorphism** unified `segment()` interface · **Encapsulation** internal
weights, thresholds, and preprocessing.

Design patterns: Strategy · Facade · Dependency Injection · Composition.

### Technologies
Python 3.13 · PyTorch 2.x (CUDA) · segmentation-models-pytorch · timm ·
OpenCV · NumPy · Pandas · Pillow · Plotly · Streamlit.

### Evaluation metrics
Dice · IoU · Precision · Recall · Border Uncertainty Score · Inference latency.

### Limitations
- Trained on dermoscopic images only; clinical use requires further validation.
- Geometric measurements are approximate pixel-based estimates.
- Segmentation output is not a cancer classification.
    """)
    st.warning(
        "This application is for academic and research demonstrations only. "
        "It is NOT a medical diagnostic tool."
    )


# ============================================================
# APPLICATION ENTRY POINT
# ============================================================

def main():
    st.set_page_config(
        page_title="SkinSegAI",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    inject_custom_css()
    settings = build_sidebar()
    render_top_banner(settings)

    # Secondary navigation at the bottom of the sidebar
    st.sidebar.divider()
    page = st.sidebar.radio(
        "Pages",
        ["Workstation", "Performance Comparison", "Training Analytics", "About"],
        label_visibility="collapsed",
    )

    if page == "Workstation":
        _workstation_page(settings)
    elif page == "Performance Comparison":
        _comparison_page()
    elif page == "Training Analytics":
        _training_page()
    else:
        _about_page()

    st.markdown(
        f'<div class="skinseg-footer">{DISCLAIMER}</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
