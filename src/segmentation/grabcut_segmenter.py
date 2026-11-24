
import cv2
import numpy as np

from src.core.base_segmenter import BaseSegmenter


class GrabCutSegmenter(BaseSegmenter):

    def __init__(self, iterations=5, margin=0.05):

        super().__init__()

        if iterations < 1:
            raise ValueError("Invalid iteration count.")

        if not 0 < margin < 0.5:
            raise ValueError("Invalid margin.")

        self._iterations = iterations
        self._margin = margin

    @property
    def name(self):
        return "GrabCut"

    def segment(self, image):

        original_size = image.size

        prepared = self._prepare_image(image)

        array = cv2.cvtColor(
            np.asarray(prepared),
            cv2.COLOR_RGB2BGR
        )

        height, width = array.shape[:2]

        mask = np.zeros(
            (height, width), dtype=np.uint8
        )

        margin_x = max(
            1, int(width * self._margin)
        )

        margin_y = max(
            1, int(height * self._margin)
        )

        rectangle = (
            margin_x,
            margin_y,
            width - 2 * margin_x,
            height - 2 * margin_y
        )

        background_model = np.zeros(
            (1, 65), dtype=np.float64
        )

        foreground_model = np.zeros(
            (1, 65), dtype=np.float64
        )

        cv2.grabCut(
            array,
            mask,
            rectangle,
            background_model,
            foreground_model,
            self._iterations,
            cv2.GC_INIT_WITH_RECT
        )

        binary_mask = (
            (mask == cv2.GC_FGD)
            | (mask == cv2.GC_PR_FGD)
        )

        return self._restore_mask(
            binary_mask, original_size
        )
