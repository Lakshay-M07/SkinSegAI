
from abc import ABC, abstractmethod

import numpy as np
from PIL import Image


class BaseSegmenter(ABC):

    def __init__(self, image_size=(256, 256)):
        self._image_size = image_size

    @property
    @abstractmethod
    def name(self):
        pass

    @abstractmethod
    def segment(self, image):
        pass

    def _prepare_image(self, image):

        if not isinstance(image, Image.Image):
            raise TypeError("Expected a PIL image.")

        return image.convert("RGB").resize(
            self._image_size,
            Image.Resampling.BILINEAR
        )

    def _restore_mask(self, mask, original_size):

        mask_image = Image.fromarray(
            mask.astype(np.uint8) * 255
        )

        mask_image = mask_image.resize(
            original_size,
            Image.Resampling.NEAREST
        )

        return np.asarray(mask_image) > 127
