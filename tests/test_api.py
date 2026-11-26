"""Smoke tests for the SkinSegAI FastAPI backend."""

import base64
import io

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from backend.main import create_app


def _synthetic_png_bytes(width: int = 64, height: int = 64) -> bytes:
    arr = np.full((height, width, 3), 200, dtype=np.uint8)
    arr[16:48, 16:48] = [80, 45, 35]
    buf = io.BytesIO()
    Image.fromarray(arr, mode="RGB").save(buf, format="PNG")
    return buf.getvalue()


def _synthetic_mask_bytes(width: int = 64, height: int = 64) -> bytes:
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[16:48, 16:48] = 255
    buf = io.BytesIO()
    Image.fromarray(mask, mode="L").save(buf, format="PNG")
    return buf.getvalue()


def test_health_and_benchmark_endpoints():
    app = create_app()
    with TestClient(app) as client:
        health_res = client.get("/api/health")
        assert health_res.status_code == 200
        health_data = health_res.json()
        assert health_data["status"] == "ok"
        assert health_data["checkpoint_loaded"] is True
        assert health_data["arch"] in {"resnet34_unet", "custom_unet", "efficientnet_b2_unet"}
        assert isinstance(health_data["samples"], list)

        bench_res = client.get("/api/benchmark")
        assert bench_res.status_code == 200
        bench_data = bench_res.json()
        assert isinstance(bench_data["summary"], list)
        assert len(bench_data["summary"]) >= 2


def test_segment_endpoint_on_synthetic_rgb_image():
    app = create_app()
    img_bytes = _synthetic_png_bytes()
    gt_bytes = _synthetic_mask_bytes()

    with TestClient(app) as client:
        res = client.post(
            "/api/segment",
            files={
                "image": ("lesion.png", img_bytes, "image/png"),
                "ground_truth": ("mask.png", gt_bytes, "image/png"),
            },
            data={
                "threshold": "0.5",
                "overlay_style": "overlay",
                "opacity": "0.35",
                "include_uncertainty": "true",
            },
        )
        assert res.status_code == 200
        payload = res.json()
        assert payload["width"] == 64
        assert payload["height"] == 64
        assert payload["uncertainty_heatmap_base64"] is not None
        assert payload["unet"]["algorithm"] == "U-Net"
        assert payload["grabcut"]["algorithm"] == "GrabCut"
        assert payload["unet"]["metrics"] is not None
        assert "dice" in payload["unet"]["metrics"]
        assert payload["unet"]["border_uncertainty"] is not None

        # Verify base64 payloads decode to valid PNG images
        decoded = Image.open(io.BytesIO(base64.b64decode(payload["unet"]["overlay_base64"])))
        assert decoded.size == (64, 64)
