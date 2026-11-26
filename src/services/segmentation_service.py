
"""Multi-model segmentation service for SkinSegAI."""

import time
from collections.abc import Mapping

import numpy as np
import torch
from PIL import Image

from src.core.base_segmenter import BaseSegmenter
from src.core.segmentation_result import SegmentationResult

from src.models.artifacts import ModelArtifactRegistry
from src.segmentation.unet_segmenter import UNetSegmenter
from src.segmentation.grabcut_segmenter import GrabCutSegmenter

from src.analysis.lesion_analyzer import LesionAnalyzer
from src.evaluation.metrics import MetricsEvaluator


class SegmentationService:
    """Coordinate segmentation, analysis and evaluation."""

    def __init__(self, segmenters):

        self._segmenters = {}

        # Support both named mappings and legacy lists.
        if isinstance(segmenters, Mapping):
            entries = segmenters.items()
        else:
            entries = (
                (segmenter.name, segmenter)
                for segmenter in segmenters
                if self._validate_segmenter(segmenter)
            )

        for name, segmenter in entries:

            self._validate_segmenter(segmenter)

            if not isinstance(name, str) or not name.strip():
                raise ValueError("Invalid algorithm name.")

            if name in self._segmenters:
                raise ValueError(
                    f"Duplicate algorithm: {name}"
                )

            self._segmenters[name] = segmenter

        if not self._segmenters:
            raise ValueError("No segmenters provided.")

        self._analyzer = LesionAnalyzer()
        self._evaluator = MetricsEvaluator()

    @staticmethod
    def _validate_segmenter(segmenter):
        """Validate the segmentation strategy."""

        if not isinstance(segmenter, BaseSegmenter):
            raise TypeError("Invalid segmenter.")

        return True

    @classmethod
    def from_architectures(
        cls,
        architectures=("custom_unet", "resnet34_unet"),
        *,
        device=None,
        threshold=0.5,
        include_grabcut=True,
        project_root=None,
    ):
        """Initialize a service with selected trained models."""

        registry = ModelArtifactRegistry(project_root)

        if isinstance(architectures, str):
            architectures = (architectures,)
        else:
            architectures = tuple(architectures)

        if len(set(architectures)) != len(architectures):
            raise ValueError("Duplicate model architecture.")

        if not architectures and not include_grabcut:
            raise ValueError("No segmentation algorithms selected.")

        # Validate all checkpoints before loading any model.
        for arch in architectures:
            registry.require_checkpoint(arch)

        segmenters = {}

        for arch in architectures:

            segmenter = UNetSegmenter.from_architecture(
                arch,
                device=device,
                threshold=threshold,
                project_root=project_root,
            )

            name = registry.display_name(arch)

            segmenters[name] = segmenter

        if include_grabcut:
            segmenters["GrabCut"] = GrabCutSegmenter()

        return cls(segmenters)

    @property
    def available_algorithms(self):
        """Return registered segmentation algorithm names."""

        return list(self._segmenters.keys())

    def segment(
        self,
        image,
        algorithm,
        alpha=0.35,
        ground_truth=None,
        mc_passes=1,
    ):
        """Segment an image and produce a complete result."""

        # ----------------------------------------------- #
        # Input validation
        # ----------------------------------------------- #

        if not isinstance(image, Image.Image):
            raise TypeError("Expected a PIL image.")

        if algorithm not in self._segmenters:
            raise ValueError(
                f"Unknown segmentation algorithm: {algorithm}. "
                f"Available: {self.available_algorithms}"
            )

        if not 0 <= alpha <= 1:
            raise ValueError("Invalid overlay transparency.")

        if (
            isinstance(mc_passes, bool)
            or not isinstance(mc_passes, int)
            or mc_passes < 1
        ):
            raise ValueError(
                "mc_passes must be a positive integer."
            )

        image = image.convert("RGB")

        expected_shape = (
            image.height,
            image.width,
        )

        if ground_truth is not None:

            ground_truth = np.asarray(ground_truth)

            if ground_truth.shape != expected_shape:
                raise ValueError(
                    "Ground-truth dimensions mismatch."
                )

            if not np.all(
                (ground_truth == 0) | (ground_truth == 1)
            ):
                raise ValueError(
                    "Ground truth must be binary."
                )

        # ----------------------------------------------- #
        # Segmentation
        # ----------------------------------------------- #

        segmenter = self._segmenters[algorithm]

        prob_map = None
        uncertainty_map = None

        start = time.perf_counter()

        predict_with_uncertainty = getattr(
            segmenter,
            "predict_with_uncertainty",
            None,
        )

        if callable(predict_with_uncertainty):

            prediction = predict_with_uncertainty(
                image,
                mc_passes=mc_passes,
            )

            mask = prediction["binary_mask"]
            prob_map = prediction["prob_map"]
            uncertainty_map = prediction["uncertainty_map"]

        else:

            mask = segmenter.segment(image)

        inference_ms = (
            time.perf_counter() - start
        ) * 1000

        # ----------------------------------------------- #
        # Prediction validation
        # ----------------------------------------------- #

        if not isinstance(mask, np.ndarray):
            raise TypeError(
                "Expected a NumPy prediction mask."
            )

        if mask.shape != expected_shape:
            raise ValueError(
                "Invalid prediction dimensions."
            )

        if mask.dtype != np.bool_:
            raise TypeError(
                "Expected a boolean mask."
            )

        for map_name, array in (
            ("Probability map", prob_map),
            ("Uncertainty map", uncertainty_map),
        ):

            if array is None:
                continue

            if not isinstance(array, np.ndarray):
                raise TypeError(
                    f"{map_name} must be a NumPy array."
                )

            if array.shape != expected_shape:
                raise ValueError(
                    f"Invalid {map_name.lower()} dimensions."
                )

            if not np.issubdtype(array.dtype, np.floating):
                raise TypeError(
                    f"{map_name} must contain floating-point values."
                )

            if not np.isfinite(array).all():
                raise ValueError(
                    f"{map_name} contains non-finite values."
                )

            if np.any((array < 0) | (array > 1)):
                raise ValueError(
                    f"{map_name} must contain values "
                    "between 0 and 1."
                )

        # ----------------------------------------------- #
        # Morphological analysis
        # ----------------------------------------------- #

        measurements = self._analyzer.analyze(mask)

        overlay = self._analyzer.create_overlay(
            image,
            mask,
            alpha=alpha,
        )

        boundary = self._analyzer.draw_boundary(
            image,
            mask,
        )

        # ----------------------------------------------- #
        # Ground-truth evaluation
        # ----------------------------------------------- #

        metrics = None

        if ground_truth is not None:

            prediction_tensor = torch.from_numpy(
                mask.copy()
            ).unsqueeze(0)

            target_tensor = torch.from_numpy(
                ground_truth.astype(bool)
            ).unsqueeze(0)

            metrics = self._evaluator.evaluate_masks(
                prediction_tensor,
                target_tensor,
            )

        # ----------------------------------------------- #
        # Border uncertainty analysis
        # ----------------------------------------------- #

        border_uncertainty = None

        if (
            uncertainty_map is not None
            and mask.any()
            and not mask.all()
        ):

            border_uncertainty = (
                self._analyzer.compute_border_uncertainty(
                    mask,
                    uncertainty_map,
                )
            )

        # ----------------------------------------------- #
        # Construct unified segmentation result
        # ----------------------------------------------- #

        return SegmentationResult(
            algorithm=algorithm,
            mask=mask,
            overlay=overlay,
            boundary=boundary,
            measurements=measurements,
            inference_ms=inference_ms,
            metrics=metrics,
            prob_map=prob_map,
            uncertainty_map=uncertainty_map,
            border_uncertainty=border_uncertainty,
        )

    def compare(
        self,
        image,
        alpha=0.35,
        ground_truth=None,
        mc_passes=1,
    ):
        """Run all registered segmentation algorithms."""

        return [
            self.segment(
                image,
                algorithm,
                alpha=alpha,
                ground_truth=ground_truth,
                mc_passes=mc_passes,
            )
            for algorithm in self.available_algorithms
        ]
