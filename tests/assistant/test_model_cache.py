
"""Regression tests for segmentation model caching."""

import time
import unittest

from unittest.mock import patch

from PIL import Image, ImageDraw

from src.assistant.core.agent import SkinSegAgent

from src.assistant.core.configuration import (
    SegmentationConfig,
)

from src.assistant.tools.segmentation_tools import (
    RunSegmentationTool,
)

from src.services.segmentation_service import (
    SegmentationService,
)


class TestModelCache(unittest.TestCase):

    def test_cache_key_behaviour(self):

        tool = RunSegmentationTool()

        config = SegmentationConfig()

        with patch.object(
            SegmentationService,
            "from_architectures",
            side_effect=lambda **kwargs: object(),
        ) as factory:

            first = tool._get_service(config)

            second = tool._get_service(config)

            self.assertIs(first, second)
            self.assertEqual(factory.call_count, 1)

            # Overlay changes do not require model reloading.

            opacity = config.updated(
                overlay_alpha=0.6
            )

            third = tool._get_service(opacity)

            self.assertIs(first, third)
            self.assertEqual(factory.call_count, 1)

            # MC pass changes do not require reloading.

            multiple_passes = config.updated(
                mc_passes=5
            )

            fourth = tool._get_service(multiple_passes)

            self.assertIs(first, fourth)
            self.assertEqual(factory.call_count, 1)

            # Threshold changes replace the cached service.

            stricter = config.updated(
                threshold=0.7
            )

            fifth = tool._get_service(stricter)

            self.assertIsNot(first, fifth)
            self.assertEqual(factory.call_count, 2)

            # Model selection changes replace the service.

            single = stricter.updated(
                models=("custom_unet",)
            )

            sixth = tool._get_service(single)

            self.assertIsNot(fifth, sixth)
            self.assertEqual(factory.call_count, 3)

            # Explicit cache release.

            tool.clear_cache()

            seventh = tool._get_service(single)

            self.assertIsNot(sixth, seventh)
            self.assertEqual(factory.call_count, 4)

    def test_real_model_reuse(self):

        agent = SkinSegAgent()

        image = Image.new(
            "RGB",
            (320, 240),
            (210, 170, 145),
        )

        draw = ImageDraw.Draw(image)

        draw.ellipse(
            (100, 60, 220, 180),
            fill=(80, 45, 30),
        )

        def segment(threshold=0.5):

            agent.begin()

            agent.choose_objective("speed")

            agent.choose_models(
                ["custom_unet"]
            )

            proposal = agent.choose_threshold(
                threshold
            )

            start = time.perf_counter()

            agent.confirm(
                proposal["confirmation_token"],
                approved=True,
            )

            elapsed = (
                time.perf_counter() - start
            )

            return elapsed * 1000

        agent.upload_image(image)

        with patch.object(
            SegmentationService,
            "from_architectures",
            wraps=SegmentationService.from_architectures,
        ) as factory:

            first_ms = segment()

            self.assertEqual(
                factory.call_count,
                1,
            )

            first_mask = (
                agent.get_results()[0].mask.copy()
            )

            second_ms = segment()

            self.assertEqual(
                factory.call_count,
                1,
            )

            second_mask = (
                agent.get_results()[0].mask
            )

            self.assertTrue(
                (first_mask == second_mask).all()
            )

            print(
                f"\nInitial segmentation: "
                f"{first_ms:.2f} ms"
            )

            print(
                f"Cached segmentation: "
                f"{second_ms:.2f} ms"
            )

            # Reuse the model after a new image upload.

            agent.upload_image(
                image.transpose(
                    Image.Transpose.FLIP_LEFT_RIGHT
                )
            )

            segment()

            self.assertEqual(
                factory.call_count,
                1,
            )

            # Changing the threshold reloads the service.

            segment(threshold=0.7)

            self.assertEqual(
                factory.call_count,
                2,
            )

            # Release models without deleting results.

            agent.release_models()

            self.assertTrue(
                agent.get_context()["analysis_completed"]
            )

            segment(threshold=0.7)

            self.assertEqual(
                factory.call_count,
                3,
            )

            # Reset releases cached models and results.

            agent.reset()

            self.assertFalse(
                agent.get_context()["image_uploaded"]
            )

            agent.upload_image(image)

            segment()

            self.assertEqual(
                factory.call_count,
                4,
            )


if __name__ == "__main__":

    unittest.main(verbosity=2)
