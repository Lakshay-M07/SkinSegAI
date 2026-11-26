
"""Validated configuration for conversational segmentation."""

import math
from dataclasses import dataclass, replace
from numbers import Real


SUPPORTED_MODELS = (
    "custom_unet",
    "resnet34_unet",
    "grabcut",
)


@dataclass(frozen=True, slots=True)
class SegmentationConfig:
    """Immutable configuration for an assistant-controlled analysis."""

    models: tuple[str, ...] = (
        "custom_unet",
        "resnet34_unet",
    )

    threshold: float = 0.5
    overlay_alpha: float = 0.35

    mc_passes: int = 1

    include_uncertainty: bool = True
    include_morphology: bool = True
    include_disagreement: bool = True
    include_image_quality: bool = True

    def __post_init__(self):

        if not isinstance(self.models, (list, tuple)):
            raise TypeError(
                "Models must be a list or tuple."
            )

        models = tuple(self.models)

        object.__setattr__(self, "models", models)

        if not models:
            raise ValueError(
                "At least one segmentation model is required."
            )

        for model in models:
            if model not in SUPPORTED_MODELS:
                raise ValueError(
                    f"Unsupported model: {model}"
                )

        if len(set(models)) != len(models):
            raise ValueError(
                "Duplicate models are not allowed."
            )

        self._validate_number(
            "Threshold",
            self.threshold,
            0.1,
            0.9,
        )

        self._validate_number(
            "Overlay alpha",
            self.overlay_alpha,
            0.0,
            1.0,
        )

        if (
            type(self.mc_passes) is not int
            or not 1 <= self.mc_passes <= 20
        ):
            raise ValueError(
                "MC passes must be an integer between 1 and 20."
            )

        for field in (
            "include_uncertainty",
            "include_morphology",
            "include_disagreement",
            "include_image_quality",
        ):
            if type(getattr(self, field)) is not bool:
                raise TypeError(
                    f"{field} must be a boolean."
                )

    @staticmethod
    def _validate_number(name, value, minimum, maximum):

        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(value)
            or not minimum <= value <= maximum
        ):
            raise ValueError(
                f"{name} must be between "
                f"{minimum} and {maximum}."
            )

    @property
    def neural_models(self):
        return tuple(
            model for model in self.models
            if model != "grabcut"
        )

    @property
    def uses_grabcut(self):
        return "grabcut" in self.models

    @property
    def can_compare(self):
        return len(self.models) >= 2

    @property
    def supports_uncertainty(self):
        return bool(self.neural_models)

    def updated(self, **changes):
        """Return a new validated configuration."""

        return replace(self, **changes)

    def to_dict(self):
        """Return JSON-compatible configuration data."""

        return {
            "models": list(self.models),
            "threshold": self.threshold,
            "overlay_alpha": self.overlay_alpha,
            "mc_passes": self.mc_passes,
            "include_uncertainty": self.include_uncertainty,
            "include_morphology": self.include_morphology,
            "include_disagreement": self.include_disagreement,
            "include_image_quality": self.include_image_quality,
        }

    @classmethod
    def from_dict(cls, data):
        """Construct a validated configuration from tool arguments."""

        if not isinstance(data, dict):
            raise TypeError(
                "Configuration must be a dictionary."
            )

        return cls(**data)
