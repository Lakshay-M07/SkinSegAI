"""Ground-truth integration regression tests."""

import json
import unittest

from io import BytesIO
from zipfile import ZipFile

import numpy as np

from PIL import Image, ImageDraw

from src.assistant.core.agent import SkinSegAgent
from src.assistant.core.session_context import SessionContext


class TestGroundTruth(unittest.TestCase):

    def setUp(self):

        self.image = Image.new(
            "RGB", (320, 240), (210, 170, 145)
        )

        ImageDraw.Draw(self.image).ellipse(
            (100, 60, 220, 180),
            fill=(80, 45, 30),
        )

        reference = Image.new(
            "L", (320, 240), 0
        )

        ImageDraw.Draw(reference).ellipse(
            (100, 60, 220, 180),
            fill=255,
        )

        self.mask = np.asarray(reference).copy()

        self.session = SessionContext()

    def test_supported_binary_masks(self):

        for mask in (
            self.mask,
            self.mask > 0,
            (self.mask > 0).astype(np.uint8),
        ):

            self.session.set_image(
                self.image,
                ground_truth=mask,
            )

            self.assertTrue(
                self.session.has_ground_truth
            )

            stored = self.session.get_ground_truth()

            self.assertEqual(
                stored.dtype,
                np.bool_,
            )

            np.testing.assert_array_equal(
                stored,
                self.mask > 0,
            )

    def test_invalid_masks_preserve_session(self):

        self.session.set_image(
            self.image,
            ground_truth=self.mask,
        )

        before = self.session.get_context()

        invalid_masks = [
            np.zeros((100, 100), dtype=np.uint8),
            np.zeros((240, 320, 3), dtype=np.uint8),
            np.full((240, 320), 128, dtype=np.uint8),
            np.zeros((240, 320), dtype=np.float32),
            np.full((240, 320), np.nan),
            "invalid mask",
        ]

        for mask in invalid_masks:

            with self.assertRaises(
                (ValueError, TypeError)
            ):

                self.session.set_image(
                    self.image,
                    ground_truth=mask,
                )

            self.assertEqual(
                before,
                self.session.get_context(),
            )

            np.testing.assert_array_equal(
                self.session.get_ground_truth(),
                self.mask > 0,
            )

    def test_reference_mask_copy_isolation(self):

        expected = self.mask.copy() > 0

        self.session.set_image(
            self.image,
            ground_truth=self.mask,
        )

        self.mask[:] = 0

        returned = self.session.get_ground_truth()

        returned[:] = False

        np.testing.assert_array_equal(
            self.session.get_ground_truth(),
            expected,
        )

    def test_new_image_and_reset_remove_ground_truth(self):

        self.session.set_image(
            self.image,
            ground_truth=self.mask,
        )

        self.session.set_image(self.image)

        self.assertFalse(
            self.session.has_ground_truth
        )

        self.assertIsNone(
            self.session.get_ground_truth()
        )

        self.session.set_image(
            self.image,
            ground_truth=self.mask,
        )

        self.session.reset()

        self.assertFalse(
            self.session.has_ground_truth
        )

    def test_reference_change_invalidates_confirmation(self):

        agent = SkinSegAgent()

        agent.upload_image(
            self.image,
            ground_truth=self.mask,
        )

        agent.choose_objective("speed")
        agent.choose_models()

        proposal = agent.choose_threshold(0.5)

        agent.upload_image(
            self.image,
            ground_truth=np.zeros_like(self.mask),
        )

        with self.assertRaises(
            (ValueError, RuntimeError)
        ):

            agent.confirm(
                proposal["confirmation_token"],
                approved=True,
            )

        self.assertFalse(
            agent.get_context()["analysis_completed"]
        )

    def test_real_segmentation_and_report(self):

        agent = SkinSegAgent()

        self.addCleanup(agent.release_models)

        agent.upload_image(
            self.image,
            ground_truth=self.mask,
        )

        self.assertTrue(
            agent.get_context()[
                "ground_truth_available"
            ]
        )

        agent.choose_objective("neural_comparison")
        agent.choose_models()

        proposal = agent.choose_threshold(0.5)

        agent.confirm(
            proposal["confirmation_token"],
            approved=True,
        )

        results = agent.get_results()

        self.assertEqual(len(results), 2)

        for result in results:

            self.assertIsNotNone(result.metrics)

            for key in (
                "dice", "iou", "precision", "recall"
            ):

                self.assertIn(key, result.metrics)

                self.assertGreaterEqual(
                    result.metrics[key], 0
                )

                self.assertLessEqual(
                    result.metrics[key], 1
                )

            print(
                f"\n{result.algorithm}: "
                f"{result.metrics}"
            )

        response = agent.ask("status")

        self.assertIn(
            "Reference-mask evaluation",
            response["message"],
        )

        self.assertIn("Dice", response["message"])

        report = agent.export_report()

        with ZipFile(BytesIO(report)) as archive:

            self.assertIsNone(archive.testzip())

            context = json.loads(
                archive.read(
                    "assistant/session_context.json"
                )
            )

            self.assertTrue(
                context["ground_truth_available"]
            )

            thresholds = json.loads(
                archive.read(
                    "assistant/threshold_sensitivity.json"
                )
            )

            for entry in thresholds["results"]:

                self.assertTrue(
                    entry["ground_truth_available"]
                )

                for row in entry["threshold_results"]:

                    self.assertIsNotNone(
                        row["metrics"]
                    )

        agent.upload_image(self.image)

        self.assertFalse(
            agent.get_context()[
                "ground_truth_available"
            ]
        )

        self.assertFalse(
            agent.get_context()[
                "analysis_completed"
            ]
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
