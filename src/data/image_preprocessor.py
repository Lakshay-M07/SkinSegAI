
from pathlib import Path
from PIL import Image
import numpy as np


class ImagePreprocessor:

    def __init__(self, image_size=(256, 256)):
        self._image_size = image_size

    @property
    def image_size(self):
        return self._image_size

    def resize_image(self, image):

        return image.convert("RGB").resize(
            self._image_size,
            Image.Resampling.BILINEAR
        )

    def resize_mask(self, mask):

        return mask.convert("L").resize(
            self._image_size,
            Image.Resampling.NEAREST
        )

    def normalize_image(self, image):

        return np.asarray(
            image, dtype=np.float32
        ) / 255.0

    def normalize_mask(self, mask):

        return (
            np.asarray(mask) > 127
        ).astype(np.float32)

    def process_pair(
        self, image_path, mask_path, output_dir
    ):

        output_dir = Path(output_dir)

        image_dir = output_dir / "images"
        mask_dir = output_dir / "masks"

        image_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)

        with Image.open(image_path) as image:
            processed_image = self.resize_image(image)

        with Image.open(mask_path) as mask:
            processed_mask = self.resize_mask(mask)

        if processed_mask.getextrema()[1] == 0:
            raise ValueError(
                f"Mask became empty: {mask_path.name}"
            )

        output_image = image_dir / (
            image_path.stem + ".png"
        )

        output_mask = mask_dir / mask_path.name

        processed_image.save(output_image)
        processed_mask.save(output_mask)

        return output_image, output_mask
