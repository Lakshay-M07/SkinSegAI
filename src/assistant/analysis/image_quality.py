
"""Local image quality assessment for SkinSeg Assistant."""

import cv2
import numpy as np

from PIL import Image


class ImageQualityAnalyzer:
    """Identify potential image quality problems using image statistics."""

    MAX_ANALYSIS_SIZE = 512
    MAX_IMAGE_PIXELS = 12_000_000

    MIN_RECOMMENDED_RESOLUTION = 256

    BLUR_THRESHOLD = 50.0
    LOW_CONTRAST_THRESHOLD = 30.0

    DARK_PIXEL_THRESHOLD = 20
    BRIGHT_PIXEL_THRESHOLD = 235

    UNDEREXPOSURE_LIMIT = 0.40
    OVEREXPOSURE_LIMIT = 0.40

    LOW_BRIGHTNESS_THRESHOLD = 50.0
    HIGH_BRIGHTNESS_THRESHOLD = 205.0

    GLARE_PIXEL_LIMIT = 0.02

    def _prepare_image(self, image):

        if not isinstance(image, Image.Image):
            raise TypeError("Expected a PIL image.")

        width, height = image.size

        if width <= 0 or height <= 0:
            raise ValueError("Invalid image dimensions.")

        if width * height > self.MAX_IMAGE_PIXELS:
            raise ValueError(
                "Image exceeds the 12-megapixel limit."
            )

        prepared = image.convert("RGB").copy()

        prepared.thumbnail(
            (
                self.MAX_ANALYSIS_SIZE,
                self.MAX_ANALYSIS_SIZE,
            ),
            Image.Resampling.LANCZOS,
        )

        return np.asarray(
            prepared,
            dtype=np.uint8,
        )

    @staticmethod
    def _add_issue(
        issues,
        code,
        severity,
        explanation,
        recommendation,
    ):

        issues.append({
            "code": code,
            "severity": severity,
            "explanation": explanation,
            "recommendation": recommendation,
        })

    def analyze(self, image):

        rgb = self._prepare_image(image)

        width, height = image.size

        gray = cv2.cvtColor(
            rgb,
            cv2.COLOR_RGB2GRAY,
        )

        hsv = cv2.cvtColor(
            rgb,
            cv2.COLOR_RGB2HSV,
        )

        # Brightness statistics.

        mean_brightness = float(
            gray.mean()
        )

        dark_fraction = float(
            (gray <= self.DARK_PIXEL_THRESHOLD).mean()
        )

        bright_fraction = float(
            (gray >= self.BRIGHT_PIXEL_THRESHOLD).mean()
        )

        # Contrast statistics.

        lower, upper = np.percentile(
            gray,
            [5, 95],
        )

        contrast = float(upper - lower)

        brightness_std = float(gray.std())

        # Sharpness estimation.

        laplacian_variance = float(
            cv2.Laplacian(
                gray,
                cv2.CV_64F,
            ).var()
        )

        # Possible bright reflection regions.

        saturation = hsv[:, :, 1]
        value = hsv[:, :, 2]

        glare_candidates = (
            (value >= 245)
            & (saturation <= 30)
        )

        glare_fraction = float(
            glare_candidates.mean()
        )

        issues = []

        # Resolution checks.

        if min(width, height) < (
            self.MIN_RECOMMENDED_RESOLUTION
        ):

            self._add_issue(
                issues,
                "low_resolution",
                "warning",
                "The image has limited spatial resolution.",
                "Consider using a higher-resolution image "
                "with the lesion clearly visible.",
            )

        # Exposure checks.

        underexposed = (
            mean_brightness < self.LOW_BRIGHTNESS_THRESHOLD
            or dark_fraction > self.UNDEREXPOSURE_LIMIT
        )

        overexposed = (
            mean_brightness > self.HIGH_BRIGHTNESS_THRESHOLD
            or bright_fraction > self.OVEREXPOSURE_LIMIT
        )

        if underexposed:

            self._add_issue(
                issues,
                "possible_underexposure",
                "warning",
                "Large regions of the image may be too dark.",
                "Consider using more even lighting without "
                "strong shadows.",
            )

        if overexposed:

            self._add_issue(
                issues,
                "possible_overexposure",
                "warning",
                "Large regions of the image may be too bright.",
                "Avoid excessive exposure or direct flash.",
            )

        # Contrast checks.

        if contrast < self.LOW_CONTRAST_THRESHOLD:

            self._add_issue(
                issues,
                "low_contrast",
                "warning",
                "The image has limited intensity variation, "
                "which may make boundaries difficult to identify.",
                "Consider using a clearer image with "
                "more visible boundary detail.",
            )

        # Blur checks.
        # Low-detail images can also produce small Laplacian values.

        if laplacian_variance < self.BLUR_THRESHOLD:

            self._add_issue(
                issues,
                "possible_blur_or_low_detail",
                "warning",
                "The image contains limited high-frequency detail.",
                "Check whether the image is in focus "
                "and the lesion boundary is clearly visible.",
            )

        # Possible glare checks.

        if (
            glare_fraction > self.GLARE_PIXEL_LIMIT
            and not overexposed
        ):

            self._add_issue(
                issues,
                "possible_glare",
                "notice",
                "Bright, low-saturation regions may "
                "represent reflections or naturally light areas.",
                "Inspect these regions for reflected light.",
            )

        if issues:

            status = "review_suggested"

        else:

            status = "no_obvious_issues_detected"

        return {
            "available": True,

            "status": status,

            "image_dimensions": {
                "width": width,
                "height": height,
                "megapixels": width * height / 1_000_000,
            },

            "analysis_dimensions": {
                "width": int(gray.shape[1]),
                "height": int(gray.shape[0]),
            },

            "metrics": {
                "mean_brightness": mean_brightness,
                "brightness_std": brightness_std,
                "contrast_percentile_range": contrast,
                "laplacian_variance": laplacian_variance,
                "dark_pixel_percent": dark_fraction * 100,
                "bright_pixel_percent": bright_fraction * 100,
                "possible_glare_pixel_percent": glare_fraction * 100,
            },

            "issues": issues,

            "issue_count": len(issues),

            "is_heuristic_assessment": True,

            "is_accuracy_estimate": False,

            "is_clinical_assessment": False,

            "interpretation": (
                "Image quality checks identify potential "
                "technical problems. They cannot establish "
                "whether an image is medically suitable or "
                "whether segmentation will be accurate."
            ),
        }
