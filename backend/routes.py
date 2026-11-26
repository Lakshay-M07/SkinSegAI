"""API route definitions for SkinSegAI."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from PIL import Image

from backend.schemas import (
    BenchmarkResponse,
    HealthResponse,
    SampleDetailResponse,
    SampleListResponse,
    SegmentResponse,
)
from backend.services import (
    SAMPLES_DIR,
    SAMPLES_MASK_DIR,
    SkinSegBackendService,
    decode_image_bytes,
)

router = APIRouter(prefix="/api", tags=["SkinSegAI"])


def _get_service(request: Request) -> SkinSegBackendService:
    service = getattr(request.app.state, "service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="Backend service not initialized.")
    return service


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    service = _get_service(request)
    samples = [item["name"] for item in service.list_samples()]
    return HealthResponse(
        status="ok",
        arch=service.arch,
        device=service.device,
        checkpoint_loaded=service.unet is not None,
        samples=samples,
    )


@router.get("/samples", response_model=SampleListResponse)
def list_samples(request: Request) -> SampleListResponse:
    service = _get_service(request)
    return SampleListResponse(samples=service.list_samples())


@router.get("/samples/{name}", response_model=SampleDetailResponse)
def get_sample(name: str, request: Request) -> SampleDetailResponse:
    service = _get_service(request)
    try:
        return SampleDetailResponse(**service.get_sample(name))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/segment", response_model=SegmentResponse)
async def segment_image(
    request: Request,
    image: UploadFile | None = File(default=None),
    sample_id: str | None = Form(default=None),
    threshold: float = Form(default=0.5),
    overlay_style: str = Form(default="overlay"),
    opacity: float = Form(default=0.35),
    include_uncertainty: bool = Form(default=True),
    ground_truth: UploadFile | None = File(default=None),
) -> SegmentResponse:
    service = _get_service(request)
    gt_mask: np.ndarray | None = None

    try:
        if image is not None:
            raw = await image.read()
            pil_image = decode_image_bytes(raw, mode="RGB")
        elif sample_id:
            safe_stem = Path(sample_id).stem
            sample_path = SAMPLES_DIR / f"{safe_stem}.png"
            if not sample_path.is_file():
                raise HTTPException(
                    status_code=404, detail=f"Sample '{sample_id}' not found."
                )
            pil_image = Image.open(sample_path).convert("RGB")
            if ground_truth is None:
                gt_path = SAMPLES_MASK_DIR / f"{safe_stem}_segmentation.png"
                if gt_path.is_file():
                    gt_mask = np.asarray(Image.open(gt_path).convert("L")) > 0
        else:
            raise HTTPException(
                status_code=400,
                detail="Provide either an uploaded 'image' or a 'sample_id'.",
            )

        if ground_truth is not None:
            gt_raw = await ground_truth.read()
            gt_pil = decode_image_bytes(gt_raw, mode="L")
            gt_mask = np.asarray(gt_pil) > 0

        result = service.segment(
            image=pil_image,
            threshold=threshold,
            overlay_style=overlay_style,
            opacity=opacity,
            include_uncertainty=include_uncertainty,
            gt_mask=gt_mask,
        )
        return SegmentResponse(**result)
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/benchmark", response_model=BenchmarkResponse)
def get_benchmark(request: Request) -> BenchmarkResponse:
    service = _get_service(request)
    return BenchmarkResponse(**service.get_benchmark())
