"""HTTP integration tests for session-based segmentation."""

from io import BytesIO

import pytest

from PIL import Image, ImageDraw

from fastapi.testclient import TestClient

from backend.main import create_app


def png_bytes(image):

    buffer = BytesIO()
    image.save(buffer, format="PNG")

    return buffer.getvalue()


@pytest.fixture
def client():

    with TestClient(create_app()) as instance:

        yield instance


def create_session(client):

    response = client.post("/api/v2/session")

    assert response.status_code == 201

    body = response.json()

    return (
        {"X-Session-ID": body["session_id"]},
        body["context"]["revision"],
    )


def synthetic_image():

    image = Image.new(
        "RGB",
        (320, 240),
        (210, 170, 145),
    )

    ImageDraw.Draw(image).ellipse(
        (100, 60, 220, 180),
        fill=(80, 45, 30),
    )

    reference = Image.new(
        "L",
        image.size,
        0,
    )

    ImageDraw.Draw(reference).ellipse(
        (100, 60, 220, 180),
        fill=255,
    )

    return image, reference


def upload(client, headers, revision, reference=True):

    image, mask = synthetic_image()

    files = {
        "image": (
            "image.png",
            png_bytes(image),
            "image/png",
        ),
    }

    if reference:

        files["ground_truth"] = (
            "mask.png",
            png_bytes(mask),
            "image/png",
        )

    response = client.post(
        "/api/v2/image",
        headers=headers,
        data={"expected_revision": revision},
        files=files,
    )

    assert response.status_code == 200, response.text

    return response.json()["context"]["revision"]


def propose(client, headers, revision, models):

    response = client.post(
        "/api/v2/configuration",
        headers=headers,
        json={
            "expected_revision": revision,
            "models": models,
            "threshold": 0.5,
        },
    )

    assert response.status_code == 200, response.text

    return response.json()["proposal"]


def confirm(client, headers, token, approved=True):

    return client.post(
        "/api/v2/confirm",
        headers=headers,
        json={
            "confirmation_token": token,
            "approved": approved,
        },
    )


def test_models_and_benchmarks(client):

    response = client.get("/api/v2/models")

    assert response.status_code == 200

    models = response.json()["models"]

    assert {
        model["model_id"]
        for model in models
    } == {
        "custom_unet",
        "resnet34_unet",
        "grabcut",
    }

    assert all(
        model["benchmark"]["samples"] == 390
        for model in models
    )


def test_session_lifecycle(client):

    headers, revision = create_session(client)

    response = client.get(
        "/api/v2/context",
        headers=headers,
    )

    assert response.status_code == 200

    assert not response.json()["context"]["image_uploaded"]

    response = client.delete(
        "/api/v2/session",
        headers=headers,
    )

    assert response.status_code == 200

    response = client.get(
        "/api/v2/context",
        headers=headers,
    )

    assert response.status_code == 404


def test_session_capacity(client):

    first, _ = create_session(client)
    second, _ = create_session(client)

    assert client.post(
        "/api/v2/session"
    ).status_code == 429

    assert client.delete(
        "/api/v2/session",
        headers=first,
    ).status_code == 200

    create_session(client)


def test_invalid_reference_mask(client):

    headers, revision = create_session(client)

    image, _ = synthetic_image()

    invalid = Image.new(
        "L",
        image.size,
        128,
    )

    response = client.post(
        "/api/v2/image",
        headers=headers,
        data={"expected_revision": revision},
        files={
            "image": (
                "image.png",
                png_bytes(image),
                "image/png",
            ),
            "ground_truth": (
                "mask.png",
                png_bytes(invalid),
                "image/png",
            ),
        },
    )

    assert response.status_code == 400

    context = client.get(
        "/api/v2/context",
        headers=headers,
    ).json()["context"]

    assert not context["image_uploaded"]
    assert context["revision"] == revision


def test_confirmation_and_session_isolation(client):

    first, revision = create_session(client)
    second, _ = create_session(client)

    revision = upload(client, first, revision)

    proposal = propose(
        client,
        first,
        revision,
        ["custom_unet"],
    )

    token = proposal["confirmation_token"]

    assert client.get(
        "/api/v2/results",
        headers=first,
    ).status_code == 409

    assert confirm(
        client,
        second,
        token,
    ).status_code == 409

    assert confirm(
        client,
        first,
        token,
        approved=False,
    ).status_code == 200

    assert confirm(
        client,
        first,
        token,
    ).status_code == 409


def test_stale_revision_and_confirmation(client):

    headers, revision = create_session(client)

    revision = upload(client, headers, revision)

    proposal = propose(
        client,
        headers,
        revision,
        ["custom_unet"],
    )

    token = proposal["confirmation_token"]

    new_revision = upload(
        client,
        headers,
        revision,
        reference=False,
    )

    assert new_revision != revision

    assert confirm(
        client,
        headers,
        token,
    ).status_code == 409

    response = client.post(
        "/api/v2/configuration",
        headers=headers,
        json={
            "expected_revision": revision,
            "models": ["custom_unet"],
        },
    )

    assert response.status_code == 409


def test_three_model_segmentation(client):

    headers, revision = create_session(client)

    revision = upload(
        client,
        headers,
        revision,
    )

    proposal = propose(
        client,
        headers,
        revision,
        [
            "custom_unet",
            "resnet34_unet",
            "grabcut",
        ],
    )

    response = confirm(
        client,
        headers,
        proposal["confirmation_token"],
    )

    assert response.status_code == 200, response.text

    response = client.get(
        "/api/v2/results",
        headers=headers,
    )

    assert response.status_code == 200

    body = response.json()

    assert len(body["results"]) == 3

    for result in body["results"]:

        assert result["ground_truth_metrics"] is not None

        assert 0 <= (
            result["ground_truth_metrics"]["dice"]
        ) <= 1

        for view in result["available_views"]:

            response = client.get(
                f"/api/v2/results/{result['model_id']}/{view}",
                headers=headers,
                params={
                    "expected_revision": body["revision"],
                },
            )

            assert response.status_code == 200
            assert response.headers["content-type"] == "image/png"

            with Image.open(
                BytesIO(response.content)
            ) as image:

                assert image.size == (320, 240)

                image.verify()

        if result["model_id"] == "grabcut":

            assert "uncertainty" not in result["available_views"]


def test_invalid_configuration(client):

    headers, revision = create_session(client)

    revision = upload(
        client,
        headers,
        revision,
    )

    for changes in (
        {"models": []},
        {"models": ["unknown"]},
        {"models": ["custom_unet", "custom_unet"]},
        {"threshold": 1.5},
        {"mc_passes": 21},
        {"include_uncertainty": "true"},
    ):

        payload = {
            "expected_revision": revision,
            "models": ["custom_unet"],
            **changes,
        }

        response = client.post(
            "/api/v2/configuration",
            headers=headers,
            json=payload,
        )

        assert response.status_code in (400, 422)

    context = client.get(
        "/api/v2/context",
        headers=headers,
    ).json()["context"]

    assert context["revision"] == revision
    assert not context["analysis_completed"]


# =====================================================
# ASSISTANT API REGRESSION TESTS
# =====================================================

def test_assistant_chat_and_history(client):

    headers, revision = create_session(client)

    response = client.post(
        "/api/v2/chat",
        headers=headers,
        json={
            "expected_revision": revision,
            "message": "models",
        },
    )

    assert response.status_code == 200, response.text

    body = response.json()

    assert body["configured_provider"] == "offline"
    assert "Custom U-Net" in body["message"]
    assert "ResNet-34" in body["message"]

    assert body["confirmation_required"] is False

    response = client.get(
        "/api/v2/history",
        headers=headers,
    )

    assert response.status_code == 200

    history = response.json()["history"]

    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[0]["content"] == "models"
    assert history[1]["role"] == "assistant"

    second, _ = create_session(client)

    response = client.get(
        "/api/v2/history",
        headers=second,
    )

    assert response.json()["history"] == []


def test_ollama_unavailable(client, monkeypatch):

    from src.assistant.providers.ollama_provider import (
        OllamaProvider,
    )

    monkeypatch.setattr(
        OllamaProvider,
        "is_available",
        lambda self: False,
    )

    response = client.post(
        "/api/v2/session?provider=ollama"
    )

    assert response.status_code == 503

    # Failed creation must not consume a session slot.
    create_session(client)
    create_session(client)


def test_chat_configuration_requires_approval(
    client,
    monkeypatch,
):

    from src.assistant.providers.ollama_provider import (
        OllamaProvider,
    )

    from src.assistant.providers.base_provider import (
        ProviderResponse,
        ToolCall,
    )

    monkeypatch.setattr(
        OllamaProvider,
        "is_available",
        lambda self: True,
    )

    def mock_chat(self, messages, *, tools=()):

        return ProviderResponse(
            tool_calls=(
                ToolCall(
                    "propose_configuration",
                    {
                        "models": ["custom_unet"],
                        "threshold": 0.6,
                    },
                ),
            )
        )

    monkeypatch.setattr(
        OllamaProvider,
        "chat",
        mock_chat,
    )

    response = client.post(
        "/api/v2/session?provider=ollama"
    )

    assert response.status_code == 201

    body = response.json()

    headers = {
        "X-Session-ID": body["session_id"],
    }

    revision = upload(
        client,
        headers,
        body["context"]["revision"],
    )

    response = client.post(
        "/api/v2/chat",
        headers=headers,
        json={
            "expected_revision": revision,
            "message": (
                "Use Custom U-Net with threshold 0.6."
            ),
        },
    )

    assert response.status_code == 200, response.text

    body = response.json()

    assert body["configured_provider"] == "ollama"
    assert body["used_fallback"] is False
    assert body["confirmation_required"] is True

    assert not body["context"]["analysis_completed"]

    assert (
        body["context"]["configuration"]["threshold"]
        == 0.5
    )

    proposal = body["proposal"]

    token = proposal["confirmation_token"]

    response = confirm(
        client,
        headers,
        token,
    )

    assert response.status_code == 200

    context = response.json()["context"]

    assert context["analysis_completed"]
    assert context["configuration"]["threshold"] == 0.6

    assert context["configuration"]["models"] == [
        "custom_unet",
    ]

    assert confirm(
        client,
        headers,
        token,
    ).status_code == 409


def test_advanced_analysis_and_report(client):

    import json

    from zipfile import ZipFile

    headers, revision = create_session(client)

    revision = upload(
        client,
        headers,
        revision,
    )

    proposal = propose(
        client,
        headers,
        revision,
        ["custom_unet", "resnet34_unet"],
    )

    response = confirm(
        client,
        headers,
        proposal["confirmation_token"],
    )

    assert response.status_code == 200

    before = response.json()["context"]

    revision = before["revision"]

    response = client.get(
        "/api/v2/analysis",
        headers=headers,
        params={"expected_revision": revision},
    )

    assert response.status_code == 200, response.text

    analysis = response.json()["analysis"]

    assert analysis["uncertainty"]["available"]
    assert analysis["model_agreement"]["available"]
    assert analysis["threshold_sensitivity"]["available"]
    assert analysis["image_quality"]["available"]

    assert analysis["llm_generated_statistics"] is False
    assert analysis["medical_diagnosis"] is False

    response = client.get(
        "/api/v2/report",
        headers=headers,
        params={"expected_revision": revision},
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/zip"

    with ZipFile(BytesIO(response.content)) as archive:

        assert archive.testzip() is None

        required = {
            "assistant/session_context.json",
            "assistant/uncertainty.json",
            "assistant/model_agreement.json",
            "assistant/threshold_sensitivity.json",
            "assistant/image_quality.json",
            "assistant/summary.txt",
        }

        assert required.issubset(archive.namelist())

        assert len(archive.namelist()) == len(
            set(archive.namelist())
        )

        context = json.loads(
            archive.read(
                "assistant/session_context.json"
            )
        )

        assert context["ground_truth_available"]
        assert context["revision"] == revision

    after = client.get(
        "/api/v2/context",
        headers=headers,
    ).json()["context"]

    assert before == after


def test_analysis_and_report_without_results(client):

    headers, revision = create_session(client)

    for endpoint in ("analysis", "report"):

        response = client.get(
            f"/api/v2/{endpoint}",
            headers=headers,
            params={"expected_revision": revision},
        )

        assert response.status_code == 409


def test_release_and_reset(client):

    headers, revision = create_session(client)

    revision = upload(
        client,
        headers,
        revision,
    )

    proposal = propose(
        client,
        headers,
        revision,
        ["custom_unet"],
    )

    response = confirm(
        client,
        headers,
        proposal["confirmation_token"],
    )

    assert response.status_code == 200

    before = response.json()["context"]

    revision = before["revision"]

    response = client.post(
        "/api/v2/release",
        headers=headers,
        json={"expected_revision": revision},
    )

    assert response.status_code == 200
    assert response.json()["released"]
    assert response.json()["context"] == before

    response = client.post(
        "/api/v2/chat",
        headers=headers,
        json={
            "expected_revision": revision,
            "message": "status",
        },
    )

    assert response.status_code == 200

    response = client.post(
        "/api/v2/reset",
        headers=headers,
        json={"expected_revision": revision},
    )

    assert response.status_code == 200

    context = response.json()["context"]

    assert not context["image_uploaded"]
    assert not context["ground_truth_available"]
    assert not context["analysis_completed"]

    assert context["revision"] != revision

    response = client.get(
        "/api/v2/history",
        headers=headers,
    )

    assert response.json()["history"] == []


def test_chat_validation_and_stale_revision(client):

    headers, revision = create_session(client)

    for message in ("", " ", "x" * 4001, 123):

        response = client.post(
            "/api/v2/chat",
            headers=headers,
            json={
                "expected_revision": revision,
                "message": message,
            },
        )

        assert response.status_code in (400, 422)

    new_revision = upload(
        client,
        headers,
        revision,
    )

    assert new_revision != revision

    response = client.post(
        "/api/v2/chat",
        headers=headers,
        json={
            "expected_revision": revision,
            "message": "status",
        },
    )

    assert response.status_code == 409

    response = client.post(
        "/api/v2/chat",
        headers=headers,
        json={
            "expected_revision": new_revision,
            "message": "status",
            "approved": True,
        },
    )

    assert response.status_code == 422


# =====================================================
# RESULT VERSIONING AND REPORT CONSISTENCY
# =====================================================

def test_repeated_segmentation_advances_revision(client):

    headers, revision = create_session(client)

    revision = upload(
        client,
        headers,
        revision,
    )

    # First approved segmentation.

    proposal = propose(
        client,
        headers,
        revision,
        ["custom_unet"],
    )

    response = confirm(
        client,
        headers,
        proposal["confirmation_token"],
    )

    assert response.status_code == 200

    first = response.json()["context"]

    first_revision = first["revision"]

    assert first["analysis_completed"]
    assert first_revision > revision

    # Preparing and cancelling identical settings
    # must not change the existing results.

    proposal = propose(
        client,
        headers,
        first_revision,
        ["custom_unet"],
    )

    response = confirm(
        client,
        headers,
        proposal["confirmation_token"],
        approved=False,
    )

    assert response.status_code == 200
    assert response.json()["context"] == first

    # Repeat segmentation using identical settings.

    proposal = propose(
        client,
        headers,
        first_revision,
        ["custom_unet"],
    )

    response = confirm(
        client,
        headers,
        proposal["confirmation_token"],
    )

    assert response.status_code == 200

    second = response.json()["context"]

    second_revision = second["revision"]

    assert second["analysis_completed"]

    assert second_revision > first_revision

    assert (
        first["configuration"]
        == second["configuration"]
    )

    assert first["image"] == second["image"]

    # Previous result requests must now be rejected.

    for endpoint in (
        "/results/custom_unet/mask",
        "/results/custom_unet/overlay",
        "/analysis",
        "/report",
    ):

        response = client.get(
            f"/api/v2{endpoint}",
            headers=headers,
            params={
                "expected_revision": first_revision,
            },
        )

        assert response.status_code == 409

    # Current result requests must remain valid.

    response = client.get(
        "/api/v2/results/custom_unet/mask",
        headers=headers,
        params={
            "expected_revision": second_revision,
        },
    )

    assert response.status_code == 200

    response = client.get(
        "/api/v2/results",
        headers=headers,
    )

    assert response.status_code == 200

    assert (
        response.json()["revision"]
        == second_revision
    )


def test_report_context_matches_api_context(client):

    import json

    from datetime import datetime
    from zipfile import ZipFile

    headers, revision = create_session(client)

    revision = upload(
        client,
        headers,
        revision,
    )

    proposal = propose(
        client,
        headers,
        revision,
        ["custom_unet", "resnet34_unet"],
    )

    response = confirm(
        client,
        headers,
        proposal["confirmation_token"],
    )

    assert response.status_code == 200

    context = response.json()["context"]

    revision = context["revision"]

    response = client.get(
        "/api/v2/report",
        headers=headers,
        params={
            "expected_revision": revision,
        },
    )

    assert response.status_code == 200

    with ZipFile(BytesIO(response.content)) as archive:

        assert archive.testzip() is None

        exported = json.loads(
            archive.read(
                "assistant/session_context.json"
            )
        )

        assert set(exported) - set(context) == {
            "report_version",
            "generated_at_utc",
        }

        assert exported.pop("report_version") == "1.0"

        generated = datetime.fromisoformat(
            exported.pop("generated_at_utc")
        )

        assert generated.tzinfo is not None

        assert exported == context

    # Report generation must not mutate session state.

    after = client.get(
        "/api/v2/context",
        headers=headers,
    ).json()["context"]

    assert after == context


def test_legacy_and_v2_resource_switching(
    client,
    monkeypatch,
):

    service = client.app.state.service
    manager = client.app.state.session_manager

    # Legacy model initially loaded during startup.

    assert service.unet is not None

    original_arch = service.arch

    headers, revision = create_session(client)

    # Entering v2 must release the legacy model.

    assert service.unet is None
    assert service.arch == original_arch

    revision = upload(
        client,
        headers,
        revision,
    )

    proposal = propose(
        client,
        headers,
        revision,
        ["custom_unet", "resnet34_unet"],
    )

    response = confirm(
        client,
        headers,
        proposal["confirmation_token"],
    )

    assert response.status_code == 200

    before = response.json()["context"]

    assert before["analysis_completed"]
    assert service.unet is None

    release_calls = []

    # Track resource release without mocking inference.

    with manager.lease(
        headers["X-Session-ID"]
    ) as agent:

        original_release = agent.release_models

        def tracked_release():

            release_calls.append(True)
            return original_release()

        monkeypatch.setattr(
            agent,
            "release_models",
            tracked_release,
        )

    image, reference = synthetic_image()

    response = client.post(
        "/api/segment",
        files={
            "image": (
                "image.png",
                png_bytes(image),
                "image/png",
            ),
            "ground_truth": (
                "reference.png",
                png_bytes(reference),
                "image/png",
            ),
        },
    )

    assert response.status_code == 200, response.text

    assert "unet" in response.json()
    assert "grabcut" in response.json()

    # Legacy inference must release assistant models
    # and restore its own model.

    assert len(release_calls) == 1
    assert service.unet is not None

    # Return to the v2 interface.

    response = client.get(
        "/api/v2/context",
        headers=headers,
    )

    assert response.status_code == 200

    after = response.json()["context"]

    # No image, settings, metrics or results were lost.

    assert before == after

    assert service.unet is None
    assert service.arch == original_arch

    response = client.get(
        "/api/v2/results",
        headers=headers,
    )

    assert response.status_code == 200
    assert len(response.json()["results"]) == 2

    response = client.get(
        "/api/v2/results/custom_unet/mask",
        headers=headers,
        params={
            "expected_revision": before["revision"],
        },
    )

    assert response.status_code == 200

    with Image.open(
        BytesIO(response.content)
    ) as mask:

        assert mask.size == image.size
        mask.verify()


# =====================================================
# CURATED SAMPLES AND SHARED IMAGE VALIDATION
# =====================================================

@pytest.fixture
def sample_catalogue(tmp_path, monkeypatch):

    from backend import services

    images = tmp_path / "images"
    masks = tmp_path / "masks"

    images.mkdir()
    masks.mkdir()

    image, reference = synthetic_image()

    image.save(images / "ISIC_0000000.png")

    reference.save(
        masks / "ISIC_0000000_segmentation.png"
    )

    # A second sample without a reference mask.

    image.save(images / "ISIC_0000003.png")

    monkeypatch.setattr(
        services,
        "SAMPLES_DIR",
        images,
    )

    monkeypatch.setattr(
        services,
        "SAMPLES_MASK_DIR",
        masks,
    )

    return images, masks


def test_curated_sample_segmentation(
    client,
    sample_catalogue,
):

    response = client.get("/api/v2/samples")

    assert response.status_code == 200

    samples = {
        item["name"]: item["has_ground_truth"]
        for item in response.json()["samples"]
    }

    assert samples == {
        "ISIC_0000000": True,
        "ISIC_0000003": False,
    }

    # Original sample preview remains compatible.

    response = client.get(
        "/api/samples/ISIC_0000000"
    )

    assert response.status_code == 200

    assert response.json()["ground_truth_base64"]

    headers, revision = create_session(client)

    response = client.post(
        "/api/v2/sample",
        headers=headers,
        json={
            "expected_revision": revision,
            "sample_id": "ISIC_0000000",
        },
    )

    assert response.status_code == 200, response.text

    context = response.json()["context"]

    assert context["image_uploaded"]
    assert context["ground_truth_available"]
    assert not context["analysis_completed"]

    revision = context["revision"]

    for view in ("original", "reference"):

        response = client.get(
            "/api/v2/image",
            headers=headers,
            params={
                "expected_revision": revision,
                "view": view,
            },
        )

        assert response.status_code == 200

        with Image.open(
            BytesIO(response.content)
        ) as visual:

            assert visual.size == (320, 240)
            visual.verify()

    proposal = propose(
        client,
        headers,
        revision,
        ["custom_unet"],
    )

    response = confirm(
        client,
        headers,
        proposal["confirmation_token"],
    )

    assert response.status_code == 200

    context = response.json()["context"]

    assert context["analysis_completed"]

    assert (
        context["results"][0]["ground_truth_metrics"]
        is not None
    )


def test_sample_switch_invalidates_state(
    client,
    sample_catalogue,
):

    headers, revision = create_session(client)

    response = client.post(
        "/api/v2/sample",
        headers=headers,
        json={
            "expected_revision": revision,
            "sample_id": "ISIC_0000000",
        },
    )

    assert response.status_code == 200

    revision = response.json()["context"]["revision"]

    proposal = propose(
        client,
        headers,
        revision,
        ["custom_unet"],
    )

    token = proposal["confirmation_token"]

    response = client.post(
        "/api/v2/sample",
        headers=headers,
        json={
            "expected_revision": revision,
            "sample_id": "ISIC_0000003",
        },
    )

    assert response.status_code == 200

    context = response.json()["context"]

    assert context["revision"] > revision

    assert context["image_uploaded"]
    assert not context["ground_truth_available"]
    assert not context["analysis_completed"]

    assert confirm(
        client,
        headers,
        token,
    ).status_code == 409

    response = client.get(
        "/api/v2/image",
        headers=headers,
        params={
            "expected_revision": revision,
            "view": "original",
        },
    )

    assert response.status_code == 409

    response = client.get(
        "/api/v2/image",
        headers=headers,
        params={
            "expected_revision": context["revision"],
            "view": "reference",
        },
    )

    assert response.status_code == 404


def test_invalid_sample_identifiers(
    client,
    sample_catalogue,
):

    headers, revision = create_session(client)

    before = client.get(
        "/api/v2/context",
        headers=headers,
    ).json()["context"]

    for name in (
        "../private",
        "/tmp/image",
        "sample.png",
        "",
        "nonexistent_sample",
    ):

        response = client.post(
            "/api/v2/sample",
            headers=headers,
            json={
                "expected_revision": revision,
                "sample_id": name,
            },
        )

        assert response.status_code in (
            400, 404, 422,
        )

    after = client.get(
        "/api/v2/context",
        headers=headers,
    ).json()["context"]

    assert before == after


def test_invalid_sample_reference(
    client,
    sample_catalogue,
):

    _, masks = sample_catalogue

    invalid_reference = Image.new(
        "L", (320, 240), 128
    )

    invalid_reference.save(
        masks / "ISIC_0000000_segmentation.png"
    )

    headers, revision = create_session(client)

    response = client.post(
        "/api/v2/sample",
        headers=headers,
        json={
            "expected_revision": revision,
            "sample_id": "ISIC_0000000",
        },
    )

    assert response.status_code == 400

    context = client.get(
        "/api/v2/context",
        headers=headers,
    ).json()["context"]

    assert context["revision"] == revision
    assert not context["image_uploaded"]

    response = client.get(
        "/api/samples/ISIC_0000000"
    )

    assert response.status_code == 400


def test_legacy_invalid_image_and_reference(client):

    image, _ = synthetic_image()

    invalid_masks = [
        Image.new("L", (320, 240), 128),
        Image.new("L", (100, 100), 255),
        Image.new("RGB", (320, 240), "white"),
    ]

    for mask in invalid_masks:

        response = client.post(
            "/api/segment",
            files={
                "image": (
                    "image.png",
                    png_bytes(image),
                    "image/png",
                ),
                "ground_truth": (
                    "mask.png",
                    png_bytes(mask),
                    "image/png",
                ),
            },
        )

        assert response.status_code == 400

    response = client.post(
        "/api/segment",
        files={
            "image": (
                "invalid.png",
                b"not an image",
                "image/png",
            ),
        },
    )

    assert response.status_code == 400


def test_legacy_binary_reference_encodings(client):

    image, reference = synthetic_image()

    reference_01 = reference.point(
        lambda value: 1 if value else 0
    )

    metrics = []

    for mask in (reference, reference_01):

        response = client.post(
            "/api/segment",
            files={
                "image": (
                    "image.png",
                    png_bytes(image),
                    "image/png",
                ),
                "ground_truth": (
                    "mask.png",
                    png_bytes(mask),
                    "image/png",
                ),
            },
        )

        assert response.status_code == 200, response.text

        body = response.json()

        assert body["unet"]["metrics"] is not None

        metrics.append(
            body["unet"]["metrics"]
        )

    # Binary 0/1 and 0/255 encodings must produce
    # equivalent reference-mask evaluation.

    assert metrics[0] == metrics[1]
