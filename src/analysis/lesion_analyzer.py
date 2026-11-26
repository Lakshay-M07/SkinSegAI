
import math

import cv2
import numpy as np
from PIL import Image


class LesionAnalyzer:

    @staticmethod
    def _validate_mask(mask):

        if not isinstance(mask, np.ndarray):
            raise TypeError("Mask must be a NumPy array.")

        if mask.ndim != 2 or mask.size == 0:
            raise ValueError("Invalid mask dimensions.")

        if not np.all((mask == 0) | (mask == 1)):
            raise ValueError("Expected a binary mask.")

        return mask.astype(bool)

    @staticmethod
    def _calculate_perimeter(mask):

        padded = np.pad(
            mask, 1, constant_values=False
        )

        horizontal = np.count_nonzero(
            padded[:, 1:] != padded[:, :-1]
        )

        vertical = np.count_nonzero(
            padded[1:, :] != padded[:-1, :]
        )

        return float(horizontal + vertical)

    @staticmethod
    def _calculate_asymmetry(mask):

        area = np.count_nonzero(mask)

        if area == 0:
            return None

        horizontal = np.fliplr(mask)
        vertical = np.flipud(mask)

        horizontal_overlap = np.count_nonzero(
            mask & horizontal
        )

        vertical_overlap = np.count_nonzero(
            mask & vertical
        )

        horizontal_score = (
            1 - horizontal_overlap / area
        )

        vertical_score = (
            1 - vertical_overlap / area
        )

        return float(
            (horizontal_score + vertical_score) / 2
        )

    def analyze(self, mask):

        mask = self._validate_mask(mask)

        area = int(np.count_nonzero(mask))

        height, width = mask.shape

        area_percent = (
            area / (height * width)
        ) * 100

        if area == 0:

            return {
                "area_pixels": 0,
                "area_percent": 0.0,
                "perimeter_pixels": 0.0,
                "circularity": None,
                "asymmetry": None,
                "bounding_box": None,
                "centroid": None
            }

        perimeter = self._calculate_perimeter(mask)

        circularity = (
            4 * math.pi * area / perimeter**2
            if perimeter > 0 else None
        )

        ys, xs = np.where(mask)

        x_min = int(xs.min())
        x_max = int(xs.max()) + 1

        y_min = int(ys.min())
        y_max = int(ys.max()) + 1

        cropped_mask = mask[
            y_min:y_max,
            x_min:x_max
        ]

        asymmetry = self._calculate_asymmetry(
            cropped_mask
        )

        return {
            "area_pixels": area,

            "area_percent": float(area_percent),

            "perimeter_pixels": perimeter,

            "circularity": circularity,

            "asymmetry": asymmetry,

            "bounding_box": {
                "x": x_min,
                "y": y_min,
                "width": x_max - x_min,
                "height": y_max - y_min
            },

            "centroid": {
                "x": float(xs.mean()),
                "y": float(ys.mean())
            }
        }

    def create_overlay(
        self,
        image,
        mask,
        alpha=0.35,
        color=(0, 180, 255)
    ):

        mask = self._validate_mask(mask)

        if not 0 <= alpha <= 1:
            raise ValueError("Invalid transparency.")

        original = np.array(
            image.convert("RGB"),
            dtype=np.uint8
        )

        if original.shape[:2] != mask.shape:
            raise ValueError("Image-mask size mismatch.")

        overlay = original.copy()

        overlay[mask] = (
            (1 - alpha) * original[mask]
            + alpha * np.array(color)
        ).astype(np.uint8)

        return Image.fromarray(overlay)

    def draw_boundary(self, image, mask):

        mask = self._validate_mask(mask)

        output = np.array(
            image.convert("RGB"),
            dtype=np.uint8
        )

        if output.shape[:2] != mask.shape:
            raise ValueError("Image-mask size mismatch.")

        binary = mask.astype(np.uint8) * 255

        contours, _ = cv2.findContours(
            binary,
            cv2.RETR_LIST,
            cv2.CHAIN_APPROX_SIMPLE
        )

        cv2.drawContours(
            output,
            contours,
            -1,
            (0, 255, 0),
            max(1, round(min(mask.shape) / 256))
        )

        return Image.fromarray(output)

    def compute_border_uncertainty(
        self,
        mask,
        uncertainty_map,
        kernel_size=9
    ):
        """Compute mean uncertainty within a dilated ring around the mask boundary."""
        mask = self._validate_mask(mask)

        if uncertainty_map.shape != mask.shape:
            raise ValueError("Uncertainty map size mismatch.")

        mask_u8 = mask.astype(np.uint8)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (kernel_size, kernel_size)
        )
        dilated = cv2.dilate(mask_u8, kernel, iterations=1)
        eroded = cv2.erode(mask_u8, kernel, iterations=1)
        ring = (dilated > 0) & (eroded == 0)

        if np.any(ring):
            return float(np.mean(uncertainty_map[ring]))

        return float(np.mean(uncertainty_map))

    def create_uncertainty_heatmap(
        self,
        image,
        uncertainty_map,
        alpha=0.55,
        colormap=cv2.COLORMAP_INFERNO
    ):
        """Return (standalone_heatmap_pil, overlaid_heatmap_pil) using inferno colormap."""
        if not 0 <= alpha <= 1:
            raise ValueError("Invalid transparency.")

        original = np.array(
            image.convert("RGB"),
            dtype=np.uint8
        )

        if original.shape[:2] != uncertainty_map.shape:
            raise ValueError("Uncertainty map size mismatch.")

        norm = np.clip(uncertainty_map, 0.0, 1.0)
        heat_u8 = np.round(norm * 255.0).astype(np.uint8)
        heat_bgr = cv2.applyColorMap(heat_u8, colormap)
        heat_rgb = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)

        blended = (
            (1.0 - alpha) * original.astype(np.float32)
            + alpha * heat_rgb.astype(np.float32)
        ).clip(0, 255).astype(np.uint8)

        return Image.fromarray(heat_rgb), Image.fromarray(blended)
