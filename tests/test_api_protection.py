"""Regression tests for API request and cache protection."""

import asyncio

import pytest

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from backend.middleware import APIProtectionMiddleware


@pytest.fixture
def protected_app():

    app = FastAPI()

    app.state.calls = 0

    app.add_middleware(
        APIProtectionMiddleware,
        upload_limit=1024,
        json_limit=128,
    )

    @app.post("/api/v2/chat")
    async def chat(request: Request):

        app.state.calls += 1

        data = await request.body()

        return {"size": len(data)}

    @app.post("/api/v2/image")
    async def upload(request: Request):

        app.state.calls += 1

        data = await request.body()

        return {"size": len(data)}

    @app.get("/api/v2/context")
    def context():

        return {"available": True}

    @app.get("/outside")
    def outside():

        return {"available": True}

    return app


def test_valid_request(protected_app):

    with TestClient(protected_app) as client:

        response = client.post(
            "/api/v2/chat",
            content=b"x" * 128,
        )

        assert response.status_code == 200
        assert response.json()["size"] == 128

        assert protected_app.state.calls == 1


def test_oversized_json_request(protected_app):

    with TestClient(protected_app) as client:

        response = client.post(
            "/api/v2/chat",
            content=b"x" * 129,
        )

        assert response.status_code == 413

        assert protected_app.state.calls == 0

        assert "no-store" in (
            response.headers["cache-control"]
        )


def test_upload_request_limit(protected_app):

    with TestClient(protected_app) as client:

        accepted = client.post(
            "/api/v2/image",
            content=b"x" * 1024,
        )

        rejected = client.post(
            "/api/v2/image",
            content=b"x" * 1025,
        )

        assert accepted.status_code == 200
        assert rejected.status_code == 413

        assert protected_app.state.calls == 1


def test_streamed_request_limit():

    called = []

    async def downstream(scope, receive, send):

        called.append(True)

    middleware = APIProtectionMiddleware(
        downstream,
        json_limit=128,
    )

    messages = iter([
        {
            "type": "http.request",
            "body": b"x" * 80,
            "more_body": True,
        },
        {
            "type": "http.request",
            "body": b"x" * 80,
            "more_body": False,
        },
    ])

    sent = []

    async def receive():

        return next(messages)

    async def send(message):

        sent.append(message)

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/api/v2/chat",
        "headers": [],
    }

    asyncio.run(
        middleware(scope, receive, send)
    )

    assert sent[0]["status"] == 413
    assert called == []


def test_cache_protection(protected_app):

    with TestClient(protected_app) as client:

        for path in (
            "/api/v2/context",
            "/api/v2/nonexistent",
        ):

            response = client.get(path)

            assert "no-store" in (
                response.headers["cache-control"]
            )

            assert (
                response.headers["pragma"]
                == "no-cache"
            )

        response = client.get("/outside")

        assert "cache-control" not in response.headers


def test_empty_request_body(protected_app):

    with TestClient(protected_app) as client:

        response = client.post(
            "/api/v2/chat",
            content=b"",
        )

        assert response.status_code == 200

        assert response.json()["size"] == 0


def test_real_api_cache_protection():

    from backend.main import create_app

    with TestClient(create_app()) as client:

        session = client.post("/api/v2/session")

        assert session.status_code == 201

        assert "no-store" in (
            session.headers["cache-control"]
        )

        headers = {
            "X-Session-ID": session.json()["session_id"],
        }

        response = client.get(
            "/api/v2/context",
            headers=headers,
        )

        assert response.status_code == 200

        assert "no-store" in (
            response.headers["cache-control"]
        )

        client.delete(
            "/api/v2/session",
            headers=headers,
        )
