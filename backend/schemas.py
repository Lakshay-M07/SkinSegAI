"""Pydantic response schemas for the SkinSegAI FastAPI backend."""

from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"
    arch: str
    device: str
    checkpoint_loaded: bool
    samples: list[str] = Field(default_factory=list)


class SampleSummary(BaseModel):
    name: str
    has_ground_truth: bool


class SampleListResponse(BaseModel):
    samples: list[SampleSummary]


class SampleDetailResponse(BaseModel):
    name: str
    image_base64: str
    ground_truth_base64: str | None = None


class MethodSegmentationPayload(BaseModel):
    algorithm: str
    inference_ms: float
    overlay_base64: str
    mask_base64: str
    measurements: dict[str, Any]
    metrics: dict[str, float] | None = None
    border_uncertainty: float | None = None


class SegmentResponse(BaseModel):
    width: int
    height: int
    threshold: float
    overlay_style: str
    opacity: float
    original_base64: str
    uncertainty_heatmap_base64: str | None = None
    uncertainty_overlay_base64: str | None = None
    error_map_base64: str | None = None
    unet: MethodSegmentationPayload
    grabcut: MethodSegmentationPayload


class BenchmarkEntry(BaseModel):
    algorithm: str
    samples: int
    dice: float
    dice_std: float | None = None
    iou: float
    precision: float
    recall: float
    inference_ms: float
    median_inference_ms: float | None = None


class BenchmarkResponse(BaseModel):
    source: str
    available: bool
    summary: list[BenchmarkEntry]
