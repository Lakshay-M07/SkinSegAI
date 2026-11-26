"""Session-based segmentation API shared by manual and AI workflows."""

from contextlib import contextmanager

from fastapi import (
    APIRouter,
    File,
    Form,
    Header,
    HTTPException,
    Request,
    Response,
    UploadFile,
)

from backend.image_io import (
    read_upload,
    decode_image,
    encode_png,
    mask_to_image,
    create_error_map,
)

from backend.schemas_v2 import (
    ConfigurationRequest,
    ConfirmationRequest,
)

from backend.session_manager import (
    SessionNotFoundError,
    SessionCapacityError,
)

from src.analysis.lesion_analyzer import LesionAnalyzer

from src.assistant.core.configuration import (
    SegmentationConfig,
)

from src.assistant.knowledge.model_knowledge import (
    ModelKnowledgeBase,
)


router = APIRouter(
    prefix="/api/v2",
    tags=["SkinSegAI v2"],
)


MODEL_IDS = {
    "Custom U-Net": "custom_unet",
    "ResNet-34 U-Net": "resnet34_unet",
    "GrabCut": "grabcut",
}


def get_manager(request):

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

    return manager


@contextmanager
def session_agent(request, session_id):

    manager = get_manager(request)

    try:

        with manager.lease(session_id) as agent:
            yield agent

    except SessionNotFoundError as exc:

        raise HTTPException(
            status_code=404,
            detail="Session not found or expired.",
        ) from exc


def require_revision(agent, expected_revision):

    current = agent.get_context()["revision"]

    if current != expected_revision:
        raise HTTPException(
            status_code=409,
            detail="Session state has changed. Refresh and retry.",
        )


def context_payload(agent):

    return {
        "stage": agent.stage,
        "provider": agent.provider_name,
        "context": agent.get_context(),
    }


@router.post("/session", status_code=201)
def create_session(request: Request):

    manager = get_manager(request)

    try:

        session_id = manager.create()

    except SessionCapacityError as exc:

        raise HTTPException(
            status_code=429,
            detail="Session capacity reached.",
        ) from exc

    with session_agent(request, session_id) as agent:

        return {
            "session_id": session_id,
            **context_payload(agent),
        }


@router.delete("/session")
def delete_session(
    request: Request,
    session_id: str = Header(alias="X-Session-ID"),
):

    try:

        get_manager(request).delete(session_id)

    except SessionNotFoundError as exc:

        raise HTTPException(
            status_code=404,
            detail="Session not found or expired.",
        ) from exc

    return {"deleted": True}


@router.get("/models")
def available_models():

    return {
        "models": ModelKnowledgeBase().get_all_model_contexts(),
    }


@router.get("/context")
def get_context(
    request: Request,
    session_id: str = Header(alias="X-Session-ID"),
):

    with session_agent(request, session_id) as agent:

        return context_payload(agent)


@router.post("/image")
def upload_image(
    request: Request,
    image: UploadFile = File(...),
    expected_revision: int = Form(...),
    ground_truth: UploadFile | None = File(default=None),
    session_id: str = Header(alias="X-Session-ID"),
):

    with session_agent(request, session_id) as agent:

        require_revision(agent, expected_revision)

        try:

            original = decode_image(
                read_upload(image)
            )

            reference = None

            if ground_truth is not None:

                reference = decode_image(
                    read_upload(ground_truth),
                    reference_mask=True,
                )

            response = agent.upload_image(
                original,
                ground_truth=reference,
            )

        except (ValueError, TypeError) as exc:

            raise HTTPException(
                status_code=400,
                detail=str(exc),
            ) from exc

        return {
            **response,
            **context_payload(agent),
        }


@router.post("/configuration")
def propose_configuration(
    payload: ConfigurationRequest,
    request: Request,
    session_id: str = Header(alias="X-Session-ID"),
):

    with session_agent(request, session_id) as agent:

        require_revision(
            agent,
            payload.expected_revision,
        )

        if not agent.get_context()["image_uploaded"]:
            raise HTTPException(
                status_code=409,
                detail="Upload an image first.",
            )

        try:

            config = SegmentationConfig.from_dict(
                payload.model_dump(
                    exclude={"expected_revision"},
                )
            )

            proposal = agent.propose_configuration(config)

        except (ValueError, TypeError) as exc:

            raise HTTPException(
                status_code=400,
                detail=str(exc),
            ) from exc

        return {
            "proposal": proposal,
            **context_payload(agent),
        }


@router.post("/confirm")
def confirm_configuration(
    payload: ConfirmationRequest,
    request: Request,
    session_id: str = Header(alias="X-Session-ID"),
):

    with session_agent(request, session_id) as agent:

        try:

            response = agent.confirm(
                payload.confirmation_token,
                approved=payload.approved,
            )

        except Exception as exc:

            if agent.stage == "failed":

                raise HTTPException(
                    status_code=503,
                    detail="Segmentation could not be completed.",
                ) from exc

            if isinstance(exc, (ValueError, RuntimeError)):

                raise HTTPException(
                    status_code=409,
                    detail="Invalid or expired confirmation.",
                ) from exc

            raise

        return {
            **response,
            **context_payload(agent),
        }


def available_views(result, config, has_reference):

    views = [
        "mask",
        "overlay",
        "boundary",
    ]

    if (
        config["include_uncertainty"]
        and result.uncertainty_map is not None
    ):

        views.extend([
            "uncertainty",
            "uncertainty_overlay",
        ])

    if has_reference:

        views.append("error_map")

    return views


@router.get("/results")
def get_results(
    request: Request,
    session_id: str = Header(alias="X-Session-ID"),
):

    with session_agent(request, session_id) as agent:

        context = agent.get_context()

        if not context["analysis_completed"]:
            raise HTTPException(
                status_code=409,
                detail="No completed segmentation is available.",
            )

        summaries = {
            entry["algorithm"]: entry
            for entry in context["results"]
        }

        results = []

        for result in agent.get_results():

            results.append({
                "model_id": MODEL_IDS[result.algorithm],

                **summaries[result.algorithm],

                "available_views": available_views(
                    result,
                    context["configuration"],
                    context["ground_truth_available"],
                ),
            })

        return {
            "revision": context["revision"],
            "image": context["image"],
            "configuration": context["configuration"],
            "results": results,
            "limitations": context["limitations"],
        }


@router.get("/results/{model_id}/{view}")
def get_visualization(
    model_id: str,
    view: str,
    request: Request,
    expected_revision: int,
    session_id: str = Header(alias="X-Session-ID"),
):

    with session_agent(request, session_id) as agent:

        require_revision(agent, expected_revision)

        context = agent.get_context()

        if not context["analysis_completed"]:
            raise HTTPException(
                status_code=409,
                detail="No completed segmentation is available.",
            )

        result = next(
            (
                item
                for item in agent.get_results()
                if MODEL_IDS[item.algorithm] == model_id
            ),
            None,
        )

        if result is None:
            raise HTTPException(
                status_code=404,
                detail="Model result not found.",
            )

        config = context["configuration"]

        if view not in available_views(
            result,
            config,
            context["ground_truth_available"],
        ):

            raise HTTPException(
                status_code=404,
                detail="Visualization is unavailable.",
            )

        if view == "mask":

            image = mask_to_image(result.mask)

        elif view == "overlay":

            image = result.overlay

        elif view == "boundary":

            image = result.boundary

        elif view == "error_map":

            image = create_error_map(
                result.mask,
                agent.get_ground_truth(),
            )

        else:

            heatmap, overlay = (
                LesionAnalyzer().create_uncertainty_heatmap(
                    agent.get_image(),
                    result.uncertainty_map,
                    alpha=config["overlay_alpha"],
                )
            )

            image = (
                heatmap
                if view == "uncertainty"
                else overlay
            )

        return Response(
            content=encode_png(image),
            media_type="image/png",
            headers={
                "Cache-Control": "no-store",
            },
        )
