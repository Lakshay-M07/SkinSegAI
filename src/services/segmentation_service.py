
import time

import numpy as np
import torch
from PIL import Image

from src.core.base_segmenter import BaseSegmenter
from src.core.segmentation_result import SegmentationResult

from src.analysis.lesion_analyzer import LesionAnalyzer
from src.evaluation.metrics import MetricsEvaluator


class SegmentationService:

    def __init__(self, segmenters):

        self._segmenters = {}

        for segmenter in segmenters:

            if not isinstance(segmenter, BaseSegmenter):
                raise TypeError("Invalid segmenter.")

            if segmenter.name in self._segmenters:
                raise ValueError("Duplicate algorithm.")

            self._segmenters[segmenter.name] = segmenter

        if not self._segmenters:
            raise ValueError("No segmenters provided.")

        self._analyzer = LesionAnalyzer()
        self._evaluator = MetricsEvaluator()

    @property
    def available_algorithms(self):

        return list(self._segmenters.keys())

    def segment(
        self,
        image,
        algorithm,
        alpha=0.35,
        ground_truth=None
    ):

        if not isinstance(image, Image.Image):
            raise TypeError("Expected a PIL image.")

        if algorithm not in self._segmenters:
            raise ValueError("Unknown segmentation algorithm.")

        if not 0 <= alpha <= 1:
            raise ValueError("Invalid overlay transparency.")

        image = image.convert("RGB")

        if ground_truth is not None:

            ground_truth = np.asarray(ground_truth)

            if ground_truth.shape != (
                image.height, image.width
            ):
                raise ValueError(
                    "Ground-truth dimensions mismatch."
                )

            if not np.all(
                (ground_truth == 0) | (ground_truth == 1)
            ):
                raise ValueError(
                    "Ground truth must be binary."
                )

        segmenter = self._segmenters[algorithm]

        start = time.perf_counter()

        mask = segmenter.segment(image)

        inference_ms = (
            time.perf_counter() - start
        ) * 1000

        if mask.shape != (image.height, image.width):
            raise ValueError("Invalid prediction dimensions.")

        if mask.dtype != np.bool_:
            raise TypeError("Expected a boolean mask.")

        measurements = self._analyzer.analyze(mask)

        overlay = self._analyzer.create_overlay(
            image,
            mask,
            alpha=alpha
        )

        boundary = self._analyzer.draw_boundary(
            image,
            mask
        )

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
                target_tensor
            )

        return SegmentationResult(
            algorithm=algorithm,
            mask=mask,
            overlay=overlay,
            boundary=boundary,
            measurements=measurements,
            inference_ms=inference_ms,
            metrics=metrics
        )

    def compare(
        self,
        image,
        alpha=0.35,
        ground_truth=None
    ):

        return [
            self.segment(
                image,
                algorithm,
                alpha=alpha,
                ground_truth=ground_truth
            )
            for algorithm in self.available_algorithms
        ]
