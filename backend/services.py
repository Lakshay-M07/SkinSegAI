"""Backend service layer wrapping src/ segmentation, morphology, and metrics."""

from __future__ import annotations

import base64
import io
import logging
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image, UnidentifiedImageError

from src.analysis.lesion_analyzer import LesionAnalyzer
from src.evaluation.metrics import MetricsEvaluator
from src.segmentation.grabcut_segmenter import GrabCutSegmenter
from src.segmentation.unet_segmenter import UNetSegmenter

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = ROOT / "artifacts"
SAMPLES_DIR = ROOT / "data" / "processed_grouped" / "validation" / "images"
SAMPLES_MASK_DIR = ROOT / "data" / "processed_grouped" / "validation" / "masks"
BENCHMARK_CSV = ARTIFACTS_DIR / "metrics" / "comparison_summary.csv"

CURATED_SAMPLE_STEMS: tuple[str, ...] = (
    "ISIC_0000000",
    "ISIC_0000003",
    "ISIC_0000014",
    "ISIC_0000022",
    "ISIC_0000025",
)

ARCH_PRIORITY: dict[str, int] = {
    "resnet34_unet": 0,
    "custom_unet": 1,
    "efficientnet_b2_unet": 2,
}

VALID_OVERLAY_STYLES = {"overlay", "contour", "mask"}


def encode_pil_to_base64(image: Image.Image) -> str:
    """Encode a PIL image as a base64 PNG string."""
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def decode_image_bytes(data: bytes, mode: str = "RGB") -> Image.Image:
    """Decode raw uploaded bytes into a validated PIL image."""
    if not data:
        raise ValueError("Uploaded image payload is empty.")
    try:
        with Image.open(io.BytesIO(data)) as img:
            img.load()
            return img.convert(mode)
    except (UnidentifiedImageError, OSError, SyntaxError) as exc:
        raise ValueError("Unable to decode image. Provide a valid PNG or JPEG.") from exc


def resolve_default_checkpoint(artifacts_dir: Path = ARTIFACTS_DIR) -> Path | None:
    """Pick the best available checkpoint, prioritizing resnet34_unet."""
    ckpt_dir = artifacts_dir / "checkpoints"
    candidates = sorted(ckpt_dir.glob("*.pth")) if ckpt_dir.is_dir() else []
    if not candidates and artifacts_dir.is_dir():
        candidates = sorted(artifacts_dir.rglob("*.pth"))
    if not candidates:
        return None

    ranked: list[tuple[int, int, float, Path]] = []
    for path in candidates:
        try:
            meta = torch.load(str(path), map_location="cpu", weights_only=False)
            arch = meta.get("arch", "custom_unet") if isinstance(meta, dict) else "custom_unet"
        except Exception:
            arch = "custom_unet"
        priority = ARCH_PRIORITY.get(arch, 99)
        is_best = 0 if path.name == "best_model.pth" else 1
        ranked.append((priority, is_best, -path.stat().st_mtime, path))

    ranked.sort(key=lambda item: (item[0], item[1], item[2]))
    return ranked[0][3]


class SkinSegBackendService:
    """Holds loaded models in memory and serves segmentation, samples, and metrics."""

    def __init__(self, checkpoint_path: Path | None = None) -> None:
        self.checkpoint_path = checkpoint_path or resolve_default_checkpoint()
        self.unet: UNetSegmenter | None = None
        if self.checkpoint_path is not None:
            self.unet = UNetSegmenter(self.checkpoint_path)

        self.grabcut = GrabCutSegmenter()
        self.analyzer = LesionAnalyzer()
        self.evaluator = MetricsEvaluator()

    @property
    def arch(self) -> str:
        return self.unet.arch if self.unet is not None else "unavailable"

    @property
    def device(self) -> str:
        if self.unet is not None:
            return str(self.unet._device)
        return (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )

    def list_samples(self) -> list[dict[str, Any]]:
        """List available curated sample images (plus any fallback images in data/)."""
        if not SAMPLES_DIR.is_dir():
            return []

        items: list[dict[str, Any]] = []
        seen: set[str] = set()
        for stem in CURATED_SAMPLE_STEMS:
            img_path = SAMPLES_DIR / f"{stem}.png"
            if img_path.is_file():
                gt_path = SAMPLES_MASK_DIR / f"{stem}_segmentation.png"
                items.append({"name": stem, "has_ground_truth": gt_path.is_file()})
                seen.add(stem)

        if not items:
            for img_path in sorted(SAMPLES_DIR.glob("*.png"))[:5]:
                stem = img_path.stem
                if stem not in seen:
                    gt_path = SAMPLES_MASK_DIR / f"{stem}_segmentation.png"
                    items.append({"name": stem, "has_ground_truth": gt_path.is_file()})
        return items

    def get_sample(self, name: str) -> dict[str, Any]:
        """Load a curated sample image (and optional ground-truth mask) as base64 PNGs."""
        safe_name = Path(name).stem
        img_path = SAMPLES_DIR / f"{safe_name}.png"
        if not img_path.is_file():
            raise FileNotFoundError(f"Sample '{name}' not found.")

        img = Image.open(img_path).convert("RGB")
        gt_path = SAMPLES_MASK_DIR / f"{safe_name}_segmentation.png"
        gt_b64: str | None = None
        if gt_path.is_file():
            gt_img = Image.open(gt_path).convert("L")
            gt_b64 = encode_pil_to_base64(gt_img)

        return {
            "name": safe_name,
            "image_base64": encode_pil_to_base64(img),
            "ground_truth_base64": gt_b64,
        }

    def _render_visual(
        self,
        image: Image.Image,
        mask: np.ndarray,
        overlay_style: str,
        opacity: float,
    ) -> Image.Image:
        if overlay_style == "overlay":
            return self.analyzer.create_overlay(image, mask, alpha=opacity)
        if overlay_style == "contour":
            return self.analyzer.draw_boundary(image, mask)
        return Image.fromarray((mask.astype(np.uint8) * 255), mode="L").convert("RGB")

    def _compute_metrics(
        self,
        mask: np.ndarray,
        gt_mask: np.ndarray | None,
    ) -> dict[str, float] | None:
        if gt_mask is None:
            return None
        pred_t = torch.from_numpy(mask[None, None, ...])
        gt_t = torch.from_numpy(gt_mask[None, None, ...])
        return self.evaluator.evaluate_masks(pred_t, gt_t)

    def _build_method_payload(
        self,
        algorithm: str,
        image: Image.Image,
        mask: np.ndarray,
        inference_ms: float,
        overlay_style: str,
        opacity: float,
        gt_mask: np.ndarray | None,
        uncertainty_map: np.ndarray | None = None,
    ) -> dict[str, Any]:
        visual = self._render_visual(image, mask, overlay_style, opacity)
        mask_img = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
        measurements = self.analyzer.analyze(mask)
        metrics = self._compute_metrics(mask, gt_mask)
        border_uncertainty = (
            self.analyzer.compute_border_uncertainty(mask, uncertainty_map)
            if uncertainty_map is not None
            else None
        )
        return {
            "algorithm": algorithm,
            "inference_ms": round(float(inference_ms), 3),
            "overlay_base64": encode_pil_to_base64(visual),
            "mask_base64": encode_pil_to_base64(mask_img),
            "measurements": measurements,
            "metrics": metrics,
            "border_uncertainty": border_uncertainty,
        }

    def segment(
        self,
        image: Image.Image,
        threshold: float = 0.5,
        overlay_style: str = "overlay",
        opacity: float = 0.35,
        include_uncertainty: bool = True,
        gt_mask: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Run U-Net (with single-pass uncertainty) and GrabCut on the input image."""
        if self.unet is None:
            raise RuntimeError("No trained U-Net checkpoint is loaded.")
        if not 0.0 < threshold < 1.0:
            raise ValueError("threshold must be strictly between 0 and 1.")
        if not 0.0 <= opacity <= 1.0:
            raise ValueError("opacity must be between 0 and 1.")
        if overlay_style not in VALID_OVERLAY_STYLES:
            raise ValueError(
                f"overlay_style must be one of {sorted(VALID_OVERLAY_STYLES)}."
            )

        rgb = image.convert("RGB")
        if gt_mask is not None and gt_mask.shape != (rgb.height, rgb.width):
            raise ValueError("Ground-truth mask dimensions must match the input image.")

        # ── U-Net inference (single-pass uncertainty) ─────────
        self.unet.threshold = threshold
        t0 = time.perf_counter()
        unet_pred = self.unet.predict_with_uncertainty(rgb, mc_passes=1)
        unet_ms = (time.perf_counter() - t0) * 1000.0

        unet_mask = unet_pred["binary_mask"]
        uncertainty_map = unet_pred["uncertainty_map"] if include_uncertainty else None

        # ── GrabCut baseline ──────────────────────────────────
        cv2.setRNGSeed(42)
        t1 = time.perf_counter()
        grabcut_mask = self.grabcut.segment(rgb)
        grabcut_ms = (time.perf_counter() - t1) * 1000.0

        heatmap_b64: str | None = None
        heat_overlay_b64: str | None = None
        if uncertainty_map is not None:
            heatmap_pil, heat_overlay_pil = self.analyzer.create_uncertainty_heatmap(
                rgb, uncertainty_map, alpha=max(0.35, opacity)
            )
            heatmap_b64 = encode_pil_to_base64(heatmap_pil)
            heat_overlay_b64 = encode_pil_to_base64(heat_overlay_pil)

        error_map_b64: str | None = None
        if gt_mask is not None:
            pred_b = unet_mask.astype(bool)
            gt_b = gt_mask.astype(bool)
            err_rgb = np.zeros((rgb.height, rgb.width, 3), dtype=np.uint8)
            err_rgb[pred_b & gt_b] = [16, 185, 129]    # TP green
            err_rgb[pred_b & ~gt_b] = [239, 68, 68]    # FP red
            err_rgb[~pred_b & gt_b] = [59, 130, 246]   # FN blue
            error_map_b64 = encode_pil_to_base64(Image.fromarray(err_rgb))

        return {
            "width": rgb.width,
            "height": rgb.height,
            "threshold": threshold,
            "overlay_style": overlay_style,
            "opacity": opacity,
            "original_base64": encode_pil_to_base64(rgb),
            "uncertainty_heatmap_base64": heatmap_b64,
            "uncertainty_overlay_base64": heat_overlay_b64,
            "error_map_base64": error_map_b64,
            "unet": self._build_method_payload(
                "U-Net", rgb, unet_mask, unet_ms,
                overlay_style, opacity, gt_mask, uncertainty_map,
            ),
            "grabcut": self._build_method_payload(
                "GrabCut", rgb, grabcut_mask, grabcut_ms,
                overlay_style, opacity, gt_mask, None,
            ),
        }

    def get_benchmark(self) -> dict[str, Any]:
        """Load saved benchmark metrics from artifacts/ with structured fallback."""
        if BENCHMARK_CSV.is_file():
            try:
                df = pd.read_csv(BENCHMARK_CSV)
                records = df.where(pd.notnull(df), None).to_dict(orient="records")
                if records:
                    return {
                        "source": str(BENCHMARK_CSV.relative_to(ROOT)),
                        "available": True,
                        "summary": records,
                    }
            except Exception:
                logger.exception("Failed to read benchmark CSV.")

        return {
            "source": "fallback",
            "available": False,
            "summary": [
                {
                    "algorithm": "U-Net",
                    "samples": 0,
                    "dice": 0.0,
                    "dice_std": 0.0,
                    "iou": 0.0,
                    "precision": 0.0,
                    "recall": 0.0,
                    "inference_ms": 0.0,
                    "median_inference_ms": 0.0,
                },
                {
                    "algorithm": "GrabCut",
                    "samples": 0,
                    "dice": 0.0,
                    "dice_std": 0.0,
                    "iou": 0.0,
                    "precision": 0.0,
                    "recall": 0.0,
                    "inference_ms": 0.0,
                    "median_inference_ms": 0.0,
                },
            ],
        }
