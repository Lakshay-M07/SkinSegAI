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
    "U-Net":   "#38BDF8",
    "GrabCut": "#2DD4BF",
}

THRESHOLD_PRESETS: dict[str, float] = {
    "High Sensitivity (0.35)": 0.35,
    "Balanced (0.50)":         0.50,
    "High Precision (0.65)":   0.65,
    "Manual":                  -1.0,   # sentinel — use slider
}

DISCLAIMER = (
    "⚠️ **Research only.** SkinSegAI is not a clinical diagnostic "
    "tool and cannot determine whether a lesion is benign or malignant."
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


def checkpoint_arch_label(ckpt_path: Path) -> str:
    """Read arch from checkpoint metadata without loading the full model."""
    try:
        import torch
        meta = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        arch = meta.get("arch", "custom_unet")
        ep   = meta.get("epoch", "?")
        dice = meta.get("val_dice", None)
        dice_str = f"  dice={dice:.4f}" if dice is not None else ""
        return f"{ckpt_path.name}  [{arch} · ep {ep}{dice_str}]"
    except Exception:
        return ckpt_path.name


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
    Keys: checkpoint_path, image_bytes, mask_bytes, source_label,
          threshold, opacity, overlay_style, overlay_color,
          show_uncertainty, mc_passes, unet_model (or None).
    """
    st.sidebar.title("🔬 SkinSegAI")
    st.sidebar.caption("Skin Lesion Segmentation Workstation")
    st.sidebar.markdown(DISCLAIMER)
    st.sidebar.divider()

    # ── Checkpoint selector ──────────────────────────────────
    st.sidebar.subheader("Model checkpoint")
    checkpoints = discover_checkpoints()

    if not checkpoints:
        st.sidebar.error("No *.pth checkpoints found in artifacts/checkpoints/")
        checkpoint_path = None
        unet_model      = None
    else:
        default_idx = next(
            (i for i, p in enumerate(checkpoints) if p.name == "best_model.pth"),
            0,
        )
        ckpt_labels = [checkpoint_arch_label(p) for p in checkpoints]
        sel_idx = st.sidebar.selectbox(
            "Select checkpoint",
            range(len(checkpoints)),
            index=default_idx,
            format_func=lambda i: ckpt_labels[i],
        )
        checkpoint_path = str(checkpoints[sel_idx])
        try:
            unet_model, _ = load_segmentation_service(checkpoint_path)
            st.sidebar.caption(f"Architecture: `{unet_model.arch}`")
        except Exception:
            logger.exception("Failed to load checkpoint for sidebar inspection.")
            unet_model = None

    st.sidebar.divider()

    # ── Image source ─────────────────────────────────────────
    st.sidebar.subheader("Image source")
    source = st.sidebar.radio(
        "Source",
        ["Custom upload", "Curated sample"],
        label_visibility="collapsed",
    )

    image_bytes: bytes | None = None
    mask_bytes:  bytes | None = None
    source_label = ""

    if source == "Custom upload":
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
        # Curated gallery — only show samples whose files exist
        available = [(s, _load_sample_bytes(s)) for s in CURATED_SAMPLES]
        available = [(s, pair) for s, pair in available if pair[0] is not None]
        if not available:
            st.sidebar.warning(
                "Curated samples not found. "
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

    # ── Threshold ────────────────────────────────────────────
    st.sidebar.subheader("Threshold")
    preset_choice = st.sidebar.radio(
        "Preset",
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
        st.sidebar.caption(f"Threshold locked to **{threshold}**")

    st.sidebar.divider()

    # ── Overlay style ────────────────────────────────────────
    st.sidebar.subheader("Overlay style")
    overlay_style = st.sidebar.radio(
        "Style",
        ["Filled Translucent Mask", "Crisp Contour Outline"],
        label_visibility="collapsed",
    )
    overlay_color = st.sidebar.color_picker(
        "Mask / outline colour", value="#38BDF8"
    )
    opacity = st.sidebar.slider(
        "Opacity / alpha",
        min_value=0.05,
        max_value=0.95,
        value=0.35,
        step=0.05,
    )

    st.sidebar.divider()

    # ── Uncertainty ──────────────────────────────────────────
    st.sidebar.subheader("Uncertainty")
    show_uncertainty = st.sidebar.toggle(
        "Show Uncertainty Heatmap",
        value=True,
        help=(
            "Display pixel-wise predictive entropy [0=certain, 1=ambiguous] "
            "and Border Uncertainty Score."
        ),
    )
    mc_passes = 1
    if unet_model is not None and unet_model.has_dropout:
        use_mc = st.sidebar.toggle(
            "Enable MC Dropout",
            value=False,
            help=(
                "Run N stochastic forward passes with Dropout active while "
                "keeping BatchNorm frozen."
            ),
        )
        if use_mc:
            mc_passes = st.sidebar.slider(
                "MC Dropout passes", min_value=2, max_value=20, value=10, step=1
            )

    st.sidebar.divider()
    st.sidebar.caption("ISIC 2018 · PyTorch · Streamlit")

    return {
        "checkpoint_path": checkpoint_path,
        "image_bytes":     image_bytes,
        "mask_bytes":      mask_bytes,
        "source_label":    source_label,
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
        st.image(image, caption="Original", use_container_width=True)
    with col_unet:
        st.image(unet_overlay, caption="U-Net Overlay", use_container_width=True)
    with col_gc:
        st.image(grabcut_overlay, caption="GrabCut Overlay", use_container_width=True)

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
        st.plotly_chart(fig, use_container_width=True)

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
    st.title("Visual Analysis Workstation")
    st.caption(
        "Single-pass U-Net · GrabCut baseline · Pixel-wise uncertainty · "
        "Morphology metrics · Error analysis"
    )

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
        st.plotly_chart(fig, use_container_width=True)

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
    st.plotly_chart(fig, use_container_width=True)

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
        st.plotly_chart(box, use_container_width=True)

        hist = px.histogram(
            results, x=sel, color="algorithm",
            barmode="overlay", opacity=0.65, nbins=30,
            color_discrete_map=ALGO_COLORS,
        )
        st.plotly_chart(hist, use_container_width=True)

    # ── Inference time ────────────────────────────────────────
    st.subheader("Inference time")
    t_fig = px.bar(
        summary, x="algorithm", y="inference_ms",
        color="algorithm", color_discrete_map=ALGO_COLORS,
        text_auto=".2f",
        labels={"inference_ms": "Mean inference (ms)"},
    )
    st.plotly_chart(t_fig, use_container_width=True)

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

    settings = build_sidebar()

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


if __name__ == "__main__":
    main()
