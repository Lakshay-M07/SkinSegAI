
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

    def to_dict(self):

        return {
            "algorithm": self.algorithm,
            "inference_ms": self.inference_ms,
            "measurements": self.measurements,
            "metrics": self.metrics
        }
