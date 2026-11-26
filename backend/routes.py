"""API route definitions for SkinSegAI."""

from __future__ import annotations


import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from backend.schemas import (
    BenchmarkResponse,
    HealthResponse,
    SampleDetailResponse,
    SampleListResponse,
    SegmentResponse,
)
from backend.services import (
    SkinSegBackendService,
    encode_pil_to_base64,
)

from backend.image_io import (
    read_upload,
    decode_image,
    decode_reference,
    mask_to_image,
)

from backend.sample_io import load_sample

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
    _get_service(request)

    try:

        image, reference = load_sample(name)

        return SampleDetailResponse(
            name=name,
            image_base64=encode_pil_to_base64(image),
            ground_truth_base64=(
                encode_pil_to_base64(
                    mask_to_image(reference)
                )
                if reference is not None
                else None
            ),
        )

    except FileNotFoundError as exc:

        raise HTTPException(
            status_code=404,
            detail="Requested sample was not found.",
        ) from exc

    except ValueError as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc


@router.post("/segment", response_model=SegmentResponse)
def segment_image(
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
            pil_image = decode_image(
                read_upload(image)
            )
        elif sample_id:
            try:

                pil_image, gt_mask = load_sample(
                    sample_id,
                    include_reference=(
                        ground_truth is None
                    ),
                )

            except FileNotFoundError as exc:

                raise HTTPException(
                    status_code=404,
                    detail="Requested sample was not found.",
                ) from exc
        else:
            raise HTTPException(
                status_code=400,
                detail="Provide either an uploaded 'image' or a 'sample_id'.",
            )

        if ground_truth is not None:
            gt_mask = decode_reference(
                read_upload(ground_truth)
            )

        manager = getattr(
            request.app.state,
            "session_manager",
            None,
        )

        if manager is None:

            raise HTTPException(
                status_code=503,
                detail="Session manager is unavailable.",
            )

        with manager.legacy_lease():

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
