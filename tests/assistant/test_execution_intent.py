"""Regression tests for explicit segmentation requests."""

import unittest

from PIL import Image

from src.assistant.core.agent import SkinSegAgent

from src.assistant.providers.base_provider import (
    BaseProvider,
    ProviderResponse,
    ToolCall,
)


class IncorrectRoutingProvider(BaseProvider):

    name = "test-provider"

    def chat(self, messages, *, tools=()):

        return ProviderResponse(
            tool_calls=(
                ToolCall("get_available_models"),
            ),
        )


class TestExecutionIntent(unittest.TestCase):

    def setUp(self):

        self.agent = SkinSegAgent(
            provider=IncorrectRoutingProvider()
        )

        self.agent.upload_image(
            Image.new("RGB", (256, 256), "white")
        )

        self.addCleanup(self.agent.reset)

    def test_all_three_model_commands(self):

        prompts = [
            "Perform segmentation on the current skin lesion image and compare all 3 models",
            "Perform and compare all 3 models after segmentation",
            "Run segmentation using all three models",
        ]

        for prompt in prompts:

            with self.subTest(prompt=prompt):

                before = self.agent.get_context()

                reply = self.agent.ask(prompt)

                self.assertTrue(
                    reply["confirmation_required"],
                    reply,
                )

                self.assertEqual(
                    reply["routing"],
                    "deterministic",
                )

                proposal = reply["proposal"]

                self.assertEqual(
                    set(
                        proposal["configuration"]["models"]
                    ),
                    {
                        "custom_unet",
                        "resnet34_unet",
                        "grabcut",
                    },
                )

                self.assertEqual(
                    before,
                    self.agent.get_context(),
                )

                self.agent.confirm(
                    proposal["confirmation_token"],
                    approved=False,
                )

    def test_plain_execution_command(self):

        reply = self.agent.ask("Run segmentation")

        self.assertTrue(
            reply["confirmation_required"]
        )

        self.assertFalse(
            self.agent.get_context()["analysis_completed"]
        )

    def test_benchmark_question(self):

        reply = self.agent.ask(
            "Compare the benchmark results of all three models"
        )

        self.assertFalse(
            reply["confirmation_required"]
        )

    def test_negative_instruction(self):

        detected, _ = self.agent._execution_intent(
            "Do not perform segmentation"
        )

        self.assertFalse(detected)
