"""FastAPI entrypoint for SkinSegAI."""

from __future__ import annotations

from contextlib import asynccontextmanager
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.routes import router
from backend.services import SkinSegBackendService


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the best default checkpoint once at startup and hold it in memory."""
    app.state.service = SkinSegBackendService()
    yield
    app.state.service = None


def create_app() -> FastAPI:
    app = FastAPI(
        title="SkinSegAI API",
        description="Lesion segmentation, uncertainty estimation, and morphology backend.",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:3000",
            "http://127.0.0.1:3000",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router)
    return app


app = create_app()
