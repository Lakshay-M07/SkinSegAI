
from dataclasses import dataclass

import numpy as np
from PIL import Image


@dataclass
class SegmentationResult:

    algorithm: str

    mask: np.ndarray

    overlay: Image.Image

    boundary: Image.Image

    measurements: dict

    inference_ms: float

    metrics: dict | None = None

    prob_map: np.ndarray | None = None

    uncertainty_map: np.ndarray | None = None

    border_uncertainty: float | None = None

    @property
    def binary_mask(self):
        return self.mask

    @property
    def boundary_overlay(self):
        return self.boundary

    def to_dict(self):

        return {
            "algorithm": self.algorithm,
            "inference_ms": self.inference_ms,
            "measurements": self.measurements,
            "metrics": self.metrics,
            "border_uncertainty": self.border_uncertainty,
        }
