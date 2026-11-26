
"""Threshold sensitivity analysis for SkinSeg Assistant."""

import math
from numbers import Real

import numpy as np

from src.core.segmentation_result import SegmentationResult


class ThresholdAnalyzer:
    """Analyze prediction thresholds using existing probability maps."""

    DEFAULT_THRESHOLDS = (
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
    )

    MAX_THRESHOLDS = 20

    @staticmethod
    def _validate_threshold(threshold):

        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, Real)
            or not math.isfinite(threshold)
            or not 0.1 <= threshold <= 0.9
        ):
            raise ValueError(
                "Threshold must be between 0.1 and 0.9."
            )

        return float(threshold)

    @staticmethod
    def _validate_probability_map(result):

        if not isinstance(result, SegmentationResult):
            raise TypeError(
                "Expected a SegmentationResult."
            )

        if result.prob_map is None:
            return None

        probability = np.asarray(
            result.prob_map,
            dtype=np.float32,
        )

        if probability.ndim != 2 or probability.size == 0:
            raise ValueError(
                "Expected a non-empty 2D probability map."
            )

        if probability.shape != result.mask.shape:
            raise ValueError(
                "Probability map dimensions mismatch."
            )

        if not np.isfinite(probability).all():
            raise ValueError(
                "Probability map contains non-finite values."
            )

        if np.any(
            (probability < 0) | (probability > 1)
        ):
            raise ValueError(
                "Probability values must be between 0 and 1."
            )

        return probability

    @staticmethod
    def _validate_ground_truth(ground_truth, shape):

        if ground_truth is None:
            return None

        target = np.asarray(ground_truth)

        if target.shape != shape:
            raise ValueError(
                "Ground-truth dimensions mismatch."
            )

        if not np.all(
            (target == 0) | (target == 1)
        ):
            raise ValueError(
                "Ground truth must be binary."
            )

        return target.astype(bool)

    @staticmethod
    def _calculate_metrics(mask, target):

        tp = int(np.count_nonzero(mask & target))
        fp = int(np.count_nonzero(mask & ~target))
        fn = int(np.count_nonzero(~mask & target))

        precision = (
            tp / (tp + fp)
            if tp + fp > 0
            else None
        )

        recall = (
            tp / (tp + fn)
            if tp + fn > 0
            else None
        )

        dice = (
            2 * tp / (2 * tp + fp + fn)
            if 2 * tp + fp + fn > 0
            else None
        )

        iou = (
            tp / (tp + fp + fn)
            if tp + fp + fn > 0
            else None
        )

        return {
            "precision": precision,
            "recall": recall,
            "dice": dice,
            "iou": iou,
            "true_positive_pixels": tp,
            "false_positive_pixels": fp,
            "false_negative_pixels": fn,
        }

    def create_mask(self, result, threshold):

        threshold = self._validate_threshold(
            threshold
        )

        probability = (
            self._validate_probability_map(result)
        )

        if probability is None:
            raise ValueError(
                "This algorithm does not support "
                "probability thresholds."
            )

        return probability >= threshold

    def analyze(
        self,
        result,
        thresholds=None,
        *,
        current_threshold=0.5,
        ground_truth=None,
    ):

        probability = (
            self._validate_probability_map(result)
        )

        if probability is None:

            return {
                "algorithm": result.algorithm,
                "available": False,
                "reason": (
                    "This algorithm does not provide "
                    "a neural probability map."
                ),
            }

        current_threshold = (
            self._validate_threshold(
                current_threshold
            )
        )

        if thresholds is None:
            thresholds = self.DEFAULT_THRESHOLDS

        if not isinstance(thresholds, (list, tuple)):
            raise TypeError(
                "Thresholds must be a list or tuple."
            )

        if not 1 <= len(thresholds) <= self.MAX_THRESHOLDS:
            raise ValueError(
                "Provide between 1 and 20 thresholds."
            )

        thresholds = sorted({
            self._validate_threshold(value)
            for value in thresholds
        })

        target = self._validate_ground_truth(
            ground_truth,
            probability.shape,
        )

        current_mask = (
            probability >= current_threshold
        )

        current_area = float(
            current_mask.mean() * 100
        )

        results = []

        for threshold in thresholds:

            mask = probability >= threshold

            area = float(mask.mean() * 100)

            foreground_pixels = int(mask.sum())

            if not mask.any():
                status = "empty"

            elif mask.all():
                status = "full"

            else:
                status = "partial"

            entry = {
                "threshold": threshold,

                "predicted_lesion_pixels": (
                    foreground_pixels
                ),

                "predicted_lesion_area_percent": area,

                "area_change_percentage_points": (
                    area - current_area
                ),

                "changed_pixel_percent": float(
                    (mask != current_mask).mean() * 100
                ),

                "mask_status": status,

                "metrics": (
                    self._calculate_metrics(mask, target)
                    if target is not None
                    else None
                ),
            }

            results.append(entry)

        areas = [
            entry["predicted_lesion_area_percent"]
            for entry in results
        ]

        return {
            "algorithm": result.algorithm,

            "available": True,

            "current_threshold": current_threshold,

            "current_area_percent": current_area,

            "threshold_results": results,

            "area_sensitivity_percentage_points": (
                max(areas) - min(areas)
            ),

            "ground_truth_available": (
                target is not None
            ),

            "requires_model_inference": False,

            "is_accuracy_estimate": False,

            "interpretation": (
                "Increasing the threshold produces a "
                "smaller or equal foreground mask for "
                "a fixed probability map. Actual precision "
                "and recall require ground truth."
            ),
        }

    def analyze_all(
        self,
        results,
        thresholds=None,
        *,
        current_threshold=0.5,
        ground_truth=None,
    ):

        return [
            self.analyze(
                result,
                thresholds,
                current_threshold=current_threshold,
                ground_truth=ground_truth,
            )
            for result in results
        ]
