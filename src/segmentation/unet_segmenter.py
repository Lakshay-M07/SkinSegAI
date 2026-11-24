
from pathlib import Path

import numpy as np
import torch

from src.core.base_segmenter import BaseSegmenter
from src.models.unet import UNet


class UNetSegmenter(BaseSegmenter):

    def __init__(self, checkpoint_path,
                 device=None, threshold=0.5):

        super().__init__()

        if not 0 < threshold < 1:
            raise ValueError("Invalid prediction threshold.")

        self._threshold = threshold

        self._device = torch.device(
            device or (
                "mps"
                if torch.backends.mps.is_available()
                else "cpu"
            )
        )

        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True
        )

        self._image_size = tuple(
            checkpoint.get("image_size", (256, 256))
        )

        self._model = UNet()

        self._model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self._model.to(self._device)
        self._model.eval()

    @property
    def name(self):
        return "U-Net"

    @property
    def threshold(self):
        return self._threshold

    @threshold.setter
    def threshold(self, value):

        if not 0 < value < 1:
            raise ValueError("Invalid prediction threshold.")

        self._threshold = value

    @torch.inference_mode()
    def segment(self, image):

        original_size = image.size

        prepared = self._prepare_image(image)

        array = np.asarray(
            prepared, dtype=np.float32
        ) / 255.0

        tensor = torch.from_numpy(
            np.ascontiguousarray(
                array.transpose(2, 0, 1)
            )
        ).unsqueeze(0)

        tensor = tensor.to(self._device)

        logits = self._model(tensor)

        probabilities = torch.sigmoid(logits)

        mask = (
            probabilities[0, 0] >= self._threshold
        ).cpu().numpy()

        return self._restore_mask(
            mask, original_size
        )
