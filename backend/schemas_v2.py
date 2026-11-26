"""Validated request schemas for the session-based API."""

from pydantic import BaseModel, ConfigDict, Field


class ConfigurationRequest(BaseModel):

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
    )

    expected_revision: int = Field(ge=0)

    models: list[str] = Field(
        min_length=1,
        max_length=3,
    )

    threshold: float = Field(
        default=0.5,
        ge=0.1,
        le=0.9,
    )

    overlay_alpha: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
    )

    mc_passes: int = Field(
        default=1,
        ge=1,
        le=20,
    )

    include_uncertainty: bool = True
    include_morphology: bool = True
    include_disagreement: bool = True
    include_image_quality: bool = True


class ConfirmationRequest(BaseModel):

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
    )

    confirmation_token: str = Field(
        min_length=1,
        max_length=128,
    )

    approved: bool
