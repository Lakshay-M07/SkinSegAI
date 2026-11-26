
"""Verified model knowledge for SkinSeg Assistant."""

import csv
import math
from dataclasses import dataclass, asdict

from src.models.artifacts import ModelArtifactRegistry


@dataclass(frozen=True, slots=True)
class ModelKnowledge:
    """Verified information about a segmentation model."""

    model_id: str
    display_name: str
    description: str
    technical_description: str
    strengths: tuple[str, ...]
    limitations: tuple[str, ...]
    trainable_parameters: int | None
    requires_training: bool
    supports_threshold: bool
    supports_uncertainty: bool


class ModelKnowledgeBase:
    """Provide verified model information and benchmark metrics."""

    _MODELS = {

        "custom_unet": ModelKnowledge(
            model_id="custom_unet",
            display_name="Custom U-Net",
            description=(
                "A lightweight AI model that identifies "
                "which parts of an image belong to a skin lesion."
            ),
            technical_description=(
                "A custom encoder-decoder convolutional neural "
                "network with skip connections."
            ),
            strengths=(
                "Fast inference",
                "Low memory consumption",
                "Strong segmentation performance",
                "Suitable for interactive analysis",
            ),
            limitations=(
                "May struggle with complex lesion boundaries",
                "Predictions depend on training data",
                "Does not perform medical diagnosis",
            ),
            trainable_parameters=1_942_577,
            requires_training=True,
            supports_threshold=True,
            supports_uncertainty=True,
        ),

        "resnet34_unet": ModelKnowledge(
            model_id="resnet34_unet",
            display_name="ResNet-34 U-Net",
            description=(
                "An advanced AI segmentation model that uses "
                "a deeper image feature extractor to identify "
                "skin lesion boundaries."
            ),
            technical_description=(
                "U-Net architecture with an ImageNet-pretrained "
                "ResNet-34 encoder."
            ),
            strengths=(
                "Highest verified mean Dice score",
                "Strong feature extraction",
                "Lower observed Dice variability",
                "Suitable for detailed segmentation",
            ),
            limitations=(
                "Higher memory consumption",
                "More computationally expensive than Custom U-Net",
                "Does not perform medical diagnosis",
            ),
            trainable_parameters=24_436_369,
            requires_training=True,
            supports_threshold=True,
            supports_uncertainty=True,
        ),

        "grabcut": ModelKnowledge(
            model_id="grabcut",
            display_name="GrabCut",
            description=(
                "A traditional computer vision algorithm "
                "that separates a lesion from its surrounding "
                "background using image colours and patterns."
            ),
            technical_description=(
                "Iterative graph-cut segmentation using "
                "Gaussian mixture models."
            ),
            strengths=(
                "Does not require model training",
                "Useful traditional computer vision baseline",
                "Operates without neural network weights",
            ),
            limitations=(
                "Lower verified segmentation performance",
                "Sensitive to image contrast and initialization",
                "Does not provide neural probability maps",
                "Does not provide predictive entropy maps",
            ),
            trainable_parameters=None,
            requires_training=False,
            supports_threshold=False,
            supports_uncertainty=False,
        ),
    }

    _METRIC_FIELDS = (
        "dice",
        "dice_std",
        "iou",
        "precision",
        "recall",
        "inference_ms",
        "median_inference_ms",
    )

    def __init__(self, project_root=None):

        self._registry = ModelArtifactRegistry(
            project_root
        )

    def _validate_model(self, model_id):

        if model_id not in self._MODELS:
            raise ValueError(
                f"Unknown model: {model_id}"
            )

    def available_models(self):
        """Return models currently available for segmentation."""

        trained = set(
            self._registry.trained_models()
        )

        return [
            model_id
            for model_id, model in self._MODELS.items()
            if (
                not model.requires_training
                or model_id in trained
            )
        ]

    def get_model(self, model_id):
        """Retrieve descriptive model knowledge."""

        self._validate_model(model_id)

        return self._MODELS[model_id]

    def get_benchmark(self, model_id):
        """Read benchmark metrics from preserved CSV artifacts."""

        self._validate_model(model_id)

        if model_id == "grabcut":

            source_architecture = "custom_unet"
            algorithm = "GrabCut"

        else:

            source_architecture = model_id
            algorithm = "U-Net"

        artifacts = self._registry.get(
            source_architecture
        )

        path = artifacts.comparison_summary

        if not path.is_file():
            raise FileNotFoundError(
                f"Benchmark unavailable: {path}"
            )

        with path.open(
            "r",
            newline="",
            encoding="utf-8",
        ) as file:

            rows = [
                row
                for row in csv.DictReader(file)
                if row["algorithm"] == algorithm
            ]

        if len(rows) != 1:
            raise ValueError(
                f"Expected one benchmark record for "
                f"{algorithm}, found {len(rows)}."
            )

        row = rows[0]

        samples = int(row["samples"])

        if samples <= 0:
            raise ValueError(
                "Invalid benchmark sample count."
            )

        metrics = {}

        for field in self._METRIC_FIELDS:

            value = float(row[field])

            if not math.isfinite(value):
                raise ValueError(
                    f"Invalid benchmark metric: {field}"
                )

            if field.endswith("_ms"):

                if value < 0:
                    raise ValueError(
                        f"Invalid inference time: {field}"
                    )

            elif not 0 <= value <= 1:

                raise ValueError(
                    f"Metric outside valid range: {field}"
                )

            metrics[field] = value

        hardware = {
            "custom_unet": "Apple MPS",
            "resnet34_unet": "NVIDIA RTX 3050 CUDA",
            "grabcut": "CPU",
        }

        return {
            "model_id": model_id,
            "samples": samples,
            **metrics,
            "hardware": hardware[model_id],
            "source": (
                f"artifacts/metrics/"
                f"{source_architecture}/"
                "comparison_summary.csv"
            ),
            "evaluation_scope": (
                "Held-out ISIC 2018 test dataset"
            ),
            "is_image_specific": False,
        }

    def get_model_context(self, model_id):
        """Return JSON-compatible information for the assistant."""

        model = self.get_model(model_id)

        try:

            benchmark = self.get_benchmark(
                model_id
            )

        except FileNotFoundError:

            benchmark = None

        return {
            **asdict(model),
            "available": (
                model_id in self.available_models()
            ),
            "benchmark": benchmark,
        }

    def get_all_model_contexts(self):

        return [
            self.get_model_context(model_id)
            for model_id in self.available_models()
        ]

    def recommend(self, objective):
        """Recommend a model for a supported user objective."""

        available = self.available_models()

        if objective == "speed":

            preferred = ("custom_unet", "resnet34_unet", "grabcut")

        elif objective == "accuracy":

            preferred = ("resnet34_unet", "custom_unet", "grabcut")

        elif objective == "comparison":

            return available

        elif objective == "neural_comparison":

            return [
                model_id
                for model_id in available
                if self._MODELS[model_id].requires_training
            ]

        elif objective == "traditional":

            preferred = ("grabcut",)

        else:

            raise ValueError(
                f"Unsupported objective: {objective}"
            )

        for model_id in preferred:

            if model_id in available:

                return [model_id]

        raise RuntimeError(
            "No suitable segmentation model is available."
        )
