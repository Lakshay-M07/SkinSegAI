
"""Prediction uncertainty analysis for SkinSeg Assistant."""

import numpy as np

from src.core.segmentation_result import SegmentationResult


class ConfidenceAnalyzer:
    """Summarize predictive uncertainty without claiming accuracy."""

    HIGH_ENTROPY_THRESHOLD = 0.75

    @staticmethod
    def _region_mean(array, region):

        if not region.any():
            return None

        return float(array[region].mean())

    def analyze(self, result):

        if not isinstance(result, SegmentationResult):
            raise TypeError("Invalid segmentation result.")

        if (
            result.prob_map is None
            or result.uncertainty_map is None
        ):
            return {
                "algorithm": result.algorithm,
                "available": False,
                "reason": (
                    "This algorithm does not provide "
                    "neural predictive uncertainty maps."
                ),
            }

        mask = result.mask

        probability = np.asarray(
            result.prob_map,
            dtype=np.float32,
        )

        entropy = np.asarray(
            result.uncertainty_map,
            dtype=np.float32,
        )

        for array in (probability, entropy):

            if array.shape != mask.shape:
                raise ValueError(
                    "Prediction map dimensions mismatch."
                )

            if not np.isfinite(array).all():
                raise ValueError(
                    "Non-finite prediction values."
                )

            if np.any((array < 0) | (array > 1)):
                raise ValueError(
                    "Prediction values outside valid range."
                )

        foreground = mask
        background = ~mask

        high_entropy = (
            entropy >= self.HIGH_ENTROPY_THRESHOLD
        )

        warnings = []

        if not foreground.any():

            mask_status = "empty"

            warnings.append(
                "The model predicted no lesion pixels. "
                "This does not establish that no lesion exists."
            )

        elif foreground.all():

            mask_status = "full"

            warnings.append(
                "The model classified the entire image "
                "as foreground. Inspect the result carefully."
            )

        else:

            mask_status = "partial"

        if result.border_uncertainty is None:

            warnings.append(
                "Boundary uncertainty is unavailable."
            )

        return {
            "algorithm": result.algorithm,
            "available": True,

            "mask_status": mask_status,

            "mean_predictive_entropy": float(
                entropy.mean()
            ),

            "median_predictive_entropy": float(
                np.median(entropy)
            ),

            "foreground_mean_entropy": (
                self._region_mean(
                    entropy,
                    foreground,
                )
            ),

            "background_mean_entropy": (
                self._region_mean(
                    entropy,
                    background,
                )
            ),

            "mean_foreground_probability": (
                self._region_mean(
                    probability,
                    foreground,
                )
            ),

            "mean_background_probability": (
                self._region_mean(
                    1.0 - probability,
                    background,
                )
            ),

            "border_uncertainty": (
                float(result.border_uncertainty)
                if result.border_uncertainty is not None
                else None
            ),

            "high_entropy_pixel_percent": (
                float(high_entropy.mean() * 100)
            ),

            "high_entropy_threshold": (
                self.HIGH_ENTROPY_THRESHOLD
            ),

            "ground_truth_metrics": result.metrics,

            "is_calibrated_confidence": False,

            "is_accuracy_estimate": False,

            "warnings": warnings,

            "interpretation": (
                "Lower predictive entropy indicates "
                "less ambiguous model predictions. "
                "It does not guarantee that the "
                "segmentation is correct."
            ),
        }

    def analyze_all(self, results):

        return [
            self.analyze(result)
            for result in results
        ]
