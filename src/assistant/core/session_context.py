
"""Session-specific image and segmentation context."""

import hashlib
import math

import numpy as np
from PIL import Image

from src.core.segmentation_result import SegmentationResult
from src.assistant.core.configuration import SegmentationConfig


class SessionContext:
    """Manage the current image, configuration and analysis results."""

    _MODEL_NAMES = {
        "custom_unet": "Custom U-Net",
        "resnet34_unet": "ResNet-34 U-Net",
        "grabcut": "GrabCut",
    }

    MAX_IMAGE_PIXELS = 12_000_000

    def __init__(self):

        self._image = None
        self._image_id = None
        self._ground_truth = None

        self._config = SegmentationConfig()

        self._results = {}

        self._revision = 0

    @property
    def has_image(self):
        return self._image is not None

    @property
    def has_results(self):
        return bool(self._results)

    @property
    def image_id(self):
        return self._image_id

    @property
    def revision(self):
        return self._revision

    @property
    def config(self):
        return self._config

    def set_image(self, image, *, ground_truth=None):
        """Register a new image and invalidate previous results."""

        if not isinstance(image, Image.Image):
            raise TypeError("Expected a PIL image.")

        width, height = image.size

        if width <= 0 or height <= 0:
            raise ValueError("Invalid image dimensions.")

        if width * height > self.MAX_IMAGE_PIXELS:
            raise ValueError(
                "Image exceeds the 12-megapixel limit."
            )

        reference = self._validate_ground_truth(
            ground_truth,
            (height, width),
        )

        prepared = image.convert("RGB").copy()

        digest = hashlib.sha256()

        digest.update(
            f"{width}x{height}:RGB".encode()
        )

        digest.update(prepared.tobytes())

        self._image = prepared
        self._image_id = digest.hexdigest()
        self._ground_truth = reference

        self._results.clear()
        self._revision += 1

        return self._image_id

    @staticmethod
    def _validate_ground_truth(mask, expected_shape):
        """Validate a binary reference mask without resizing."""

        if mask is None:
            return None

        if not isinstance(mask, np.ndarray):
            raise TypeError(
                "Ground truth must be a NumPy array."
            )

        if mask.ndim != 2:
            raise ValueError(
                "Ground truth must be a two-dimensional mask."
            )

        if mask.shape != expected_shape:
            raise ValueError(
                "Ground-truth dimensions must match the image."
            )

        if mask.dtype.kind not in ("b", "u", "i"):
            raise TypeError(
                "Ground truth must contain boolean or integer values."
            )

        binary_01 = np.all(
            (mask == 0) | (mask == 1)
        )

        binary_255 = np.all(
            (mask == 0) | (mask == 255)
        )

        if not (binary_01 or binary_255):
            raise ValueError(
                "Ground truth must use binary 0/1 or 0/255 values."
            )

        return mask.astype(np.bool_, copy=True)

    @property
    def has_ground_truth(self):

        return self._ground_truth is not None

    def get_ground_truth(self):
        """Return an independent copy of the reference mask."""

        if self._ground_truth is None:
            return None

        return self._ground_truth.copy()

    def get_image(self):
        """Return a copy of the current image."""

        if not self.has_image:
            raise RuntimeError("No image has been uploaded.")

        return self._image.copy()

    def set_config(self, config):
        """Apply validated settings and invalidate stale results."""

        if not isinstance(config, SegmentationConfig):
            raise TypeError(
                "Expected a SegmentationConfig object."
            )

        if config != self._config:

            self._config = config

            self._results.clear()

            self._revision += 1

        return self._config

    def update_config(self, **changes):

        config = self._config.updated(**changes)

        return self.set_config(config)

    def get_analysis_inputs(self):
        """Capture inputs and revision for an analysis operation."""

        return (
            self._revision,
            self.get_image(),
            self._config,
        )

    def store_results(self, results, *, revision):
        """Register results belonging to the current analysis."""

        if not self.has_image:
            raise RuntimeError("No image has been uploaded.")

        if revision != self._revision:
            raise ValueError(
                "Cannot store outdated segmentation results."
            )

        expected = {
            self._MODEL_NAMES[model]
            for model in self._config.models
        }

        expected_shape = (
            self._image.height,
            self._image.width,
        )

        validated = {}

        for result in results:

            if not isinstance(result, SegmentationResult):
                raise TypeError(
                    "Invalid segmentation result."
                )

            if result.algorithm not in expected:
                raise ValueError(
                    f"Unexpected algorithm: {result.algorithm}"
                )

            if result.algorithm in validated:
                raise ValueError(
                    f"Duplicate result: {result.algorithm}"
                )

            if result.mask.shape != expected_shape:
                raise ValueError(
                    "Segmentation dimensions mismatch."
                )

            if result.mask.dtype != np.bool_:
                raise TypeError(
                    "Expected a boolean segmentation mask."
                )

            validated[result.algorithm] = result

        if set(validated) != expected:
            raise ValueError(
                "Incomplete segmentation results."
            )

        self._results = validated

        # Every successfully completed analysis has a
        # distinct revision, even with identical settings.
        self._revision += 1

    def get_result(self, algorithm):
        """Retrieve an existing result by algorithm name."""

        if algorithm not in self._results:
            raise ValueError(
                f"No result available for {algorithm}."
            )

        return self._results[algorithm]

    def get_results(self):
        """Return all current segmentation results."""

        return tuple(self._results.values())

    @classmethod
    def _json_value(cls, value):
        """Convert numerical values to JSON-compatible objects."""

        if isinstance(value, np.generic):
            return cls._json_value(value.item())

        if value is None or isinstance(
            value, (str, bool, int)
        ):
            return value

        if isinstance(value, float):

            if not math.isfinite(value):
                raise ValueError(
                    "Non-finite value in analysis context."
                )

            return value

        if isinstance(value, dict):

            return {
                str(key): cls._json_value(item)
                for key, item in value.items()
            }

        if isinstance(value, (list, tuple)):

            return [
                cls._json_value(item)
                for item in value
            ]

        raise TypeError(
            f"Unsupported context value: {type(value).__name__}"
        )

    def _summarize_result(self, result):
        """Extract facts without including raw image arrays."""

        foreground_pixels = int(
            np.count_nonzero(result.mask)
        )

        total_pixels = int(result.mask.size)

        if foreground_pixels == 0:
            mask_status = "empty"

        elif foreground_pixels == total_pixels:
            mask_status = "full"

        else:
            mask_status = "partial"

        return self._json_value({
            "algorithm": result.algorithm,

            "predicted_lesion_pixels": foreground_pixels,

            "total_image_pixels": total_pixels,

            "predicted_lesion_area_percent": (
                foreground_pixels / total_pixels * 100
            ),

            "mask_status": mask_status,

            "inference_ms": result.inference_ms,

            "measurements": (
                result.measurements
                if self._config.include_morphology
                else None
            ),

            "ground_truth_metrics": result.metrics,

            "ground_truth_available": (
                result.metrics is not None
            ),

            "probability_map_available": (
                result.prob_map is not None
            ),

            "uncertainty_map_available": (
                result.uncertainty_map is not None
            ),

            "border_uncertainty": (
                result.border_uncertainty
                if self._config.include_uncertainty
                else None
            ),

            "is_medical_diagnosis": False,

            "is_calibrated_confidence": False,
        })

    def get_context(self):
        """Build structured context for the conversational assistant."""

        image_information = None

        if self.has_image:

            image_information = {
                "width": self._image.width,
                "height": self._image.height,
                "total_pixels": (
                    self._image.width
                    * self._image.height
                ),
                "mode": self._image.mode,
            }

        return {
            "image_uploaded": self.has_image,

            "ground_truth_available": self.has_ground_truth,

            "analysis_completed": self.has_results,

            "revision": self._revision,

            "image": image_information,

            "configuration": self._config.to_dict(),

            "results": [
                self._summarize_result(result)
                for result in self._results.values()
            ],

            "limitations": [
                "Predicted lesion area is not a diagnosis.",
                "Image-specific accuracy requires ground truth.",
                "Predictive entropy is not calibrated confidence.",
                "Pixel measurements do not represent physical size.",
            ],
        }

    def clear_results(self):

        self._results.clear()
        self._revision += 1

    def reset(self):
        """Remove image data and restore default settings."""

        self._image = None
        self._image_id = None
        self._ground_truth = None

        self._config = SegmentationConfig()

        self._results.clear()

        self._revision += 1
