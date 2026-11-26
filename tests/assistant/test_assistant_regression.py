
"""Regression tests for the SkinSegAI assistant backend."""

import json
import unittest

from io import BytesIO
from unittest.mock import patch
from zipfile import ZipFile

from PIL import Image, ImageDraw

from src.assistant.core.agent import SkinSegAgent

from src.assistant.core.configuration import (
    SegmentationConfig,
)

from src.assistant.providers.base_provider import (
    BaseProvider,
    ProviderResponse,
    ProviderError,
    ToolCall,
)

from src.assistant.providers.ollama_provider import (
    OllamaProvider,
)

from src.assistant.safety.input_validator import (
    InputValidator,
)

from src.assistant.safety.response_validator import (
    ExecutionBudget,
)


def create_test_image():

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

    return image


class BrokenProvider(BaseProvider):

    name = "broken"

    def chat(self, messages, *, tools=()):

        raise ProviderError(
            "Simulated provider failure."
        )


class UnauthorizedProvider(BaseProvider):

    name = "unauthorized"

    def chat(self, messages, *, tools=()):

        return ProviderResponse(
            tool_calls=(
                ToolCall("run_segmentation"),
            ),
        )


class MultiToolProvider(BaseProvider):

    name = "multi_tool"

    def chat(self, messages, *, tools=()):

        return ProviderResponse(
            tool_calls=(
                ToolCall("get_available_models"),
                ToolCall("get_session_context"),
                ToolCall("inspect_image_quality"),
            ),
        )


class ConfigurationProvider(BaseProvider):

    name = "configuration"

    def chat(self, messages, *, tools=()):

        return ProviderResponse(
            tool_calls=(
                ToolCall(
                    "propose_configuration",
                    {
                        "models": [
                            "custom_unet",
                            "resnet34_unet",
                        ],
                        "threshold": 0.7,
                    },
                ),
            ),
        )


class TestConfiguration(unittest.TestCase):

    def test_default_configuration(self):

        config = SegmentationConfig()

        self.assertEqual(config.threshold, 0.5)
        self.assertEqual(len(config.models), 2)

    def test_immutable_updates(self):

        config = SegmentationConfig()

        updated = config.updated(threshold=0.7)

        self.assertEqual(config.threshold, 0.5)
        self.assertEqual(updated.threshold, 0.7)

    def test_invalid_thresholds(self):

        for threshold in (
            -1,
            2,
            True,
            float("nan"),
            float("inf"),
        ):

            with self.subTest(threshold=threshold):

                with self.assertRaises(ValueError):

                    SegmentationConfig(
                        threshold=threshold
                    )

    def test_duplicate_models(self):

        with self.assertRaises(ValueError):

            SegmentationConfig(
                models=(
                    "custom_unet",
                    "custom_unet",
                )
            )


class TestSafety(unittest.TestCase):

    def test_protected_tools(self):

        validator = InputValidator()

        for name in (
            "run_segmentation",
            "execute_confirmed",
            "delete_checkpoint",
            "execute_shell",
        ):

            with self.subTest(tool=name):

                with self.assertRaises(PermissionError):

                    validator.validate_tool_call(
                        ToolCall(name),
                        [],
                    )

    def test_execution_budget(self):

        budget = ExecutionBudget(
            max_provider_requests=2,
            max_tool_executions=2,
        )

        for _ in range(2):

            budget.consume_provider_request()
            budget.consume_tool_execution()

        with self.assertRaises(RuntimeError):

            budget.consume_provider_request()

        with self.assertRaises(RuntimeError):

            budget.consume_tool_execution()


class TestAgent(unittest.TestCase):

    def setUp(self):

        self.agent = SkinSegAgent()

    def test_initialization(self):

        self.assertEqual(
            self.agent.provider_name,
            "offline",
        )

        self.assertFalse(
            self.agent.get_context()["image_uploaded"]
        )

    def test_verified_benchmarks(self):

        response = self.agent.ask("models")

        for value in (
            "86.87%",
            "87.79%",
            "61.46%",
        ):

            self.assertIn(
                value,
                response["message"],
            )

    def test_memory(self):

        self.agent.ask("models")
        self.agent.ask("help")

        self.assertEqual(
            self.agent.get_memory_statistics()[
                "turn_count"
            ],
            2,
        )

        self.agent.upload_image(
            create_test_image()
        )

        self.assertEqual(
            self.agent.get_memory_statistics()[
                "turn_count"
            ],
            0,
        )

    def test_session_isolation(self):

        self.agent.upload_image(
            create_test_image()
        )

        another = SkinSegAgent()

        self.assertFalse(
            another.get_context()["image_uploaded"]
        )

    def test_provider_fallback(self):

        self.agent.set_provider(
            BrokenProvider()
        )

        response = self.agent.ask("models")

        self.assertTrue(
            response["used_fallback"]
        )

        self.assertEqual(
            response["provider"],
            "offline",
        )

        self.assertIn(
            "87.79%",
            response["message"],
        )

    def test_unauthorized_provider(self):

        self.agent.set_provider(
            UnauthorizedProvider()
        )

        response = self.agent.ask("status")

        self.assertTrue(
            response["used_fallback"]
        )

        self.assertFalse(
            self.agent.get_context()[
                "analysis_completed"
            ]
        )

    def test_multiple_tool_requests(self):

        self.agent.upload_image(
            create_test_image()
        )

        self.agent.set_provider(
            MultiToolProvider()
        )

        response = self.agent.ask(
            "Show models, image status and quality."
        )

        statistics = response[
            "execution_statistics"
        ]

        self.assertEqual(
            statistics["provider_requests"],
            1,
        )

        self.assertEqual(
            statistics["tool_executions"],
            3,
        )

        self.assertIn(
            "87.79%",
            response["message"],
        )

    def test_configuration_proposal(self):

        self.agent.upload_image(
            create_test_image()
        )

        self.agent.set_provider(
            ConfigurationProvider()
        )

        response = self.agent.ask(
            "Compare both AI models at threshold 0.7."
        )

        self.assertTrue(
            response["confirmation_required"]
        )

        self.assertEqual(
            response["proposal"][
                "configuration"
            ]["threshold"],
            0.7,
        )

        self.assertEqual(
            self.agent.get_context()[
                "configuration"
            ]["threshold"],
            0.5,
        )

    def test_stale_confirmation(self):

        self.agent.upload_image(
            create_test_image()
        )

        self.agent.choose_objective("accuracy")
        self.agent.choose_models()

        proposal = self.agent.choose_threshold(0.7)

        self.agent.upload_image(
            Image.new(
                "RGB",
                (300, 300),
                "white",
            )
        )

        with self.assertRaises(
            (RuntimeError, ValueError)
        ):

            self.agent.confirm(
                proposal["confirmation_token"],
                approved=True,
            )

        self.assertFalse(
            self.agent.get_context()[
                "analysis_completed"
            ]
        )

    def test_missing_report(self):

        with self.assertRaises(RuntimeError):

            self.agent.export_report()

        self.agent.upload_image(
            create_test_image()
        )

        with self.assertRaises(RuntimeError):

            self.agent.export_report()


class TestOllamaProvider(unittest.TestCase):

    def setUp(self):

        self.provider = OllamaProvider()

        self.messages = [{
            "role": "user",
            "content": "Show the models.",
        }]

    def test_tool_call_parsing(self):

        with patch.object(
            self.provider,
            "_request",
            return_value={
                "done": True,
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{
                        "function": {
                            "name": "get_available_models",
                            "arguments": {},
                        }
                    }],
                },
            },
        ):

            response = self.provider.chat(
                self.messages
            )

        self.assertEqual(
            response.tool_calls[0].name,
            "get_available_models",
        )

    def test_invalid_tool_call_structure(self):

        for invalid in (
            {},
            False,
            0,
            "",
            "invalid",
        ):

            with self.subTest(value=invalid):

                with patch.object(
                    self.provider,
                    "_request",
                    return_value={
                        "done": True,
                        "message": {
                            "role": "assistant",
                            "content": "Test",
                            "tool_calls": invalid,
                        },
                    },
                ):

                    with self.assertRaises(ProviderError):

                        self.provider.chat(
                            self.messages
                        )

    def test_raw_image_rejection(self):

        with self.assertRaises(ValueError):

            self.provider.chat([{
                "role": "user",
                "content": "Inspect this image.",
                "images": ["invalid-image-data"],
            }])


class TestSegmentationIntegration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        cls.agent = SkinSegAgent()

        cls.agent.upload_image(
            create_test_image()
        )

        cls.agent.choose_objective("comparison")
        cls.agent.choose_models()

        proposal = cls.agent.choose_threshold(0.7)

        cls.agent.confirm(
            proposal["confirmation_token"],
            approved=True,
        )

    def test_three_model_segmentation(self):

        self.assertEqual(
            len(self.agent.get_results()),
            3,
        )

        for result in self.agent.get_results():

            self.assertEqual(
                result.mask.shape,
                (240, 320),
            )

    def test_uncertainty(self):

        response = self.agent.ask("uncertainty")

        self.assertIn(
            "Boundary uncertainty",
            response["message"],
        )

        self.assertIn(
            "not calibrated confidence",
            response["message"],
        )

    def test_model_agreement(self):

        response = self.agent.ask("compare")

        self.assertIn(
            "Dice agreement",
            response["message"],
        )

    def test_threshold_analysis(self):

        before = self.agent.get_context()

        response = self.agent.ask("thresholds")

        self.assertIn(
            "Threshold 0.50",
            response["message"],
        )

        self.assertEqual(
            before,
            self.agent.get_context(),
        )

    def test_enhanced_report(self):

        before = self.agent.get_context()

        report = self.agent.export_report()

        self.assertIsInstance(report, bytes)

        with ZipFile(BytesIO(report)) as archive:

            self.assertIsNone(
                archive.testzip()
            )

            names = archive.namelist()

            self.assertEqual(
                len(names),
                len(set(names)),
            )

            for model in (
                "custom_unet",
                "resnet34_unet",
                "grabcut",
            ):

                for filename in (
                    "mask.png",
                    "overlay.png",
                    "boundary.png",
                    "analysis.json",
                ):

                    self.assertIn(
                        f"{model}/{filename}",
                        names,
                    )

            for filename in (
                "summary.txt",
                "session_context.json",
                "uncertainty.json",
                "model_agreement.json",
                "threshold_sensitivity.json",
                "image_quality.json",
            ):

                self.assertIn(
                    f"assistant/{filename}",
                    names,
                )

            self.assertIn(
                "comparison_report.json",
                names,
            )

            self.assertIn(
                "DISCLAIMER.txt",
                names,
            )

            for name in names:

                if name.endswith(".json"):

                    json.loads(
                        archive.read(name)
                    )

                elif name.endswith(".png"):

                    with Image.open(
                        BytesIO(archive.read(name))
                    ) as image:

                        image.verify()

            context = json.loads(
                archive.read(
                    "assistant/session_context.json"
                )
            )

            self.assertEqual(
                context["configuration"]["threshold"],
                0.7,
            )

            self.assertEqual(
                len(context["results"]),
                3,
            )

            uncertainty = json.loads(
                archive.read(
                    "assistant/uncertainty.json"
                )
            )

            self.assertTrue(
                uncertainty["available"]
            )

            self.assertEqual(
                len(uncertainty["results"]),
                3,
            )

            agreement = json.loads(
                archive.read(
                    "assistant/model_agreement.json"
                )
            )

            self.assertEqual(
                len(agreement["comparisons"]),
                3,
            )

            summary = archive.read(
                "assistant/summary.txt"
            ).decode("utf-8")

            self.assertIn(
                "not calibrated confidence",
                summary,
            )

        self.assertEqual(
            before,
            self.agent.get_context(),
        )


class TestGrabCutOnly(unittest.TestCase):

    def test_grabcut_uncertainty_unavailable(self):

        agent = SkinSegAgent()

        agent.upload_image(
            create_test_image()
        )

        agent.choose_objective("traditional")

        proposal = agent.choose_models()

        agent.confirm(
            proposal["confirmation_token"],
            approved=True,
        )

        self.assertEqual(
            len(agent.get_results()),
            1,
        )

        response = agent.ask("uncertainty")

        self.assertIn(
            "None of the selected algorithms",
            response["message"],
        )


if __name__ == "__main__":

    unittest.main(verbosity=2)
