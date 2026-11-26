
"""Pairwise segmentation agreement and disagreement analysis."""

from itertools import combinations

import numpy as np

from src.core.segmentation_result import SegmentationResult


class ModelDisagreementAnalyzer:
    """Compare predicted masks without requiring ground truth."""

    @staticmethod
    def _validate_pair(first, second):

        for result in (first, second):

            if not isinstance(result, SegmentationResult):
                raise TypeError(
                    "Expected segmentation results."
                )

            if result.mask.dtype != np.bool_:
                raise TypeError(
                    "Expected boolean masks."
                )

            if result.mask.ndim != 2 or result.mask.size == 0:
                raise ValueError(
                    "Expected non-empty, two-dimensional masks."
                )

        if first.mask.shape != second.mask.shape:
            raise ValueError(
                "Cannot compare masks with different dimensions."
            )

        if first.algorithm == second.algorithm:
            raise ValueError(
                "Cannot compare duplicate algorithms."
            )

    def compare_pair(self, first, second):

        self._validate_pair(first, second)

        a = first.mask
        b = second.mask

        intersection = a & b
        union = a | b

        only_a = a & ~b
        only_b = b & ~a

        disagreement = a ^ b

        a_pixels = int(a.sum())
        b_pixels = int(b.sum())

        intersection_pixels = int(
            intersection.sum()
        )

        union_pixels = int(union.sum())

        total_pixels = int(a.size)

        dice = (
            2 * intersection_pixels / (a_pixels + b_pixels)
            if a_pixels + b_pixels > 0
            else None
        )

        iou = (
            intersection_pixels / union_pixels
            if union_pixels > 0
            else None
        )

        warnings = []

        if union_pixels == 0:

            warnings.append(
                "Both masks are empty. Foreground "
                "agreement is undefined."
            )

        elif a_pixels == 0 or b_pixels == 0:

            warnings.append(
                "One model predicted no foreground."
            )

        if a.all() or b.all():

            warnings.append(
                "At least one model classified the "
                "entire image as foreground."
            )

        return {
            "first_model": first.algorithm,

            "second_model": second.algorithm,

            "dice_agreement": dice,

            "iou_agreement": iou,

            "dice_agreement_percent": (
                dice * 100
                if dice is not None
                else None
            ),

            "pixel_agreement_percent": (
                float((a == b).mean() * 100)
            ),

            "disagreement_pixel_percent": (
                float(disagreement.mean() * 100)
            ),

            "first_model_area_percent": (
                a_pixels / total_pixels * 100
            ),

            "second_model_area_percent": (
                b_pixels / total_pixels * 100
            ),

            "area_difference_percentage_points": (
                abs(a_pixels - b_pixels)
                / total_pixels * 100
            ),

            "shared_foreground_pixels": (
                intersection_pixels
            ),

            "first_model_only_pixels": int(
                only_a.sum()
            ),

            "second_model_only_pixels": int(
                only_b.sum()
            ),

            "disagreement_within_union_percent": (
                float(
                    disagreement.sum()
                    / union_pixels
                    * 100
                )
                if union_pixels > 0
                else None
            ),

            "is_accuracy_estimate": False,

            "warnings": warnings,

            "interpretation": (
                "Model agreement measures similarity "
                "between predictions, not correctness "
                "against ground truth."
            ),
        }

    def compare_all(self, results):

        results = tuple(results)

        if len(results) < 2:
            raise ValueError(
                "At least two models are required."
            )

        names = [
            result.algorithm
            for result in results
        ]

        if len(names) != len(set(names)):
            raise ValueError(
                "Duplicate model results."
            )

        return [
            self.compare_pair(first, second)
            for first, second in combinations(results, 2)
        ]

    def disagreement_map(self, first, second):
        """Return pixels on which the models disagree."""

        self._validate_pair(first, second)

        return first.mask ^ second.mask

    def regional_map(self, first, second):
        """Generate an integer-coded comparison map.

        0: Both predict background
        1: Only the first model predicts foreground
        2: Only the second model predicts foreground
        3: Both predict foreground
        """

        self._validate_pair(first, second)

        return (
            first.mask.astype(np.uint8)
            + 2 * second.mask.astype(np.uint8)
        )
