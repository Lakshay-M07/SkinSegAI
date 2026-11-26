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
