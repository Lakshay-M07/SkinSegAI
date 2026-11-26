
"""Regression tests for pending configuration context."""

import json
import unittest

from PIL import Image

from src.assistant.core.agent import SkinSegAgent

from src.assistant.providers.base_provider import (
    BaseProvider,
    ProviderResponse,
    ToolCall,
)


class ScriptedProvider(BaseProvider):

    name = "scripted"

    def __init__(self):

        self.requests = []

        self.responses = [
            ProviderResponse(
                tool_calls=(
                    ToolCall(
                        "propose_configuration",
                        {
                            "models": ["custom_unet"],
                            "threshold": 0.7,
                            "overlay_alpha": 0.6,
                            "include_uncertainty": False,
                        },
                    ),
                ),
            ),
            ProviderResponse(
                tool_calls=(
                    ToolCall(
                        "propose_configuration",
                        {"threshold": 0.6},
                    ),
                ),
            ),
        ]

    def chat(self, messages, *, tools=()):

        self.requests.append(
            json.loads(json.dumps(messages))
        )

        return self.responses.pop(0)


class TestConversationalContext(unittest.TestCase):

    def setUp(self):

        self.provider = ScriptedProvider()

        self.agent = SkinSegAgent(
            provider=self.provider
        )

        self.agent.upload_image(
            Image.new(
                "RGB",
                (320, 240),
                "white",
            )
        )

    def create_proposals(self):

        first = self.agent.ask(
            "Use Custom U-Net at threshold 0.7 "
            "with 60 percent overlay opacity. "
            "Disable uncertainty analysis."
        )

        second = self.agent.ask(
            "Keep everything else unchanged, "
            "but use threshold 0.6."
        )

        self.assertTrue(
            first["confirmation_required"]
        )

        self.assertTrue(
            second["confirmation_required"]
        )

        self.assertIn(
            "Uncertainty analysis: Off",
            first["message"],
        )

        self.assertIn(
            "Uncertainty analysis: Off",
            second["message"],
        )

        return (
            first["proposal"],
            second["proposal"],
        )

    def test_configuration_inheritance(self):

        first, second = self.create_proposals()

        expected = dict(
            first["configuration"]
        )

        expected["threshold"] = 0.6

        self.assertEqual(
            second["configuration"],
            expected,
        )

        self.assertEqual(
            second["configuration"]["models"],
            ["custom_unet"],
        )

        self.assertEqual(
            second["configuration"]["overlay_alpha"],
            0.6,
        )

        self.assertFalse(
            second["configuration"][
                "include_uncertainty"
            ]
        )

        # Active settings remain unchanged.

        context = self.agent.get_context()

        self.assertEqual(
            context["configuration"]["threshold"],
            0.5,
        )

        self.assertFalse(
            context["analysis_completed"]
        )

    def test_pending_context_in_prompt(self):

        first, second = self.create_proposals()

        self.assertEqual(
            len(self.provider.requests[1]),
            2,
        )

        system_message = (
            self.provider.requests[1][0]["content"]
        )

        self.assertIn(
            "CURRENT PENDING PROPOSAL",
            system_message,
        )

        self.assertIn(
            '"threshold": 0.7',
            system_message,
        )

        self.assertNotIn(
            first["confirmation_token"],
            system_message,
        )

        self.assertNotIn(
            second["confirmation_token"],
            system_message,
        )

    def test_confirmation_token_replacement(self):

        first, second = self.create_proposals()

        self.assertNotEqual(
            first["confirmation_token"],
            second["confirmation_token"],
        )

        with self.assertRaises(ValueError):

            self.agent.confirm(
                first["confirmation_token"],
                approved=True,
            )

        self.assertFalse(
            self.agent.get_context()[
                "analysis_completed"
            ]
        )

        self.agent.confirm(
            second["confirmation_token"],
            approved=False,
        )

        self.assertEqual(
            self.agent.stage,
            "cancelled",
        )

        self.assertEqual(
            self.agent.get_context()[
                "configuration"
            ]["threshold"],
            0.5,
        )

    def test_pending_context_cleared_on_upload(self):

        first = self.agent.ask(
            "Prepare a configuration."
        )

        self.agent.upload_image(
            Image.new(
                "RGB",
                (300, 300),
                "black",
            )
        )

        second = self.agent.ask(
            "Prepare threshold 0.6."
        )

        self.assertTrue(
            second["confirmation_required"]
        )

        config = second["proposal"][
            "configuration"
        ]

        self.assertEqual(
            set(config["models"]),
            {"custom_unet", "resnet34_unet"},
        )

        self.assertEqual(
            config["overlay_alpha"],
            0.35,
        )

        self.assertTrue(
            config["include_uncertainty"]
        )

        self.assertNotIn(
            "CURRENT PENDING PROPOSAL",
            self.provider.requests[1][0]["content"],
        )

        self.assertFalse(
            self.agent.get_context()[
                "analysis_completed"
            ]
        )

    def test_text_only_response_recovery(self):

        self.provider.responses.insert(
            1,
            ProviderResponse(
                content=(
                    "I have changed the proposed "
                    "threshold to 0.6."
                ),
            ),
        )

        first, second = self.create_proposals()

        expected = dict(
            first["configuration"]
        )

        expected["threshold"] = 0.6

        self.assertEqual(
            second["configuration"],
            expected,
        )

        self.assertEqual(
            len(self.provider.requests),
            3,
        )

        retry_messages = self.provider.requests[2]

        self.assertEqual(
            len(retry_messages),
            2,
        )

        self.assertIn(
            "TOOL ROUTING RETRY",
            retry_messages[0]["content"],
        )

        self.assertNotIn(
            first["confirmation_token"],
            json.dumps(retry_messages),
        )

        self.assertFalse(
            self.agent.get_context()[
                "analysis_completed"
            ]
        )

    def test_retry_failure_preserves_configuration(self):

        first = self.agent.ask(
            "Prepare Custom U-Net."
        )

        self.provider.responses = [
            ProviderResponse(
                content="Settings updated."
            ),
            ProviderResponse(
                content="Threshold changed."
            ),
        ]

        response = self.agent.ask(
            "Change the proposed threshold to 0.6."
        )

        self.assertTrue(
            response["used_fallback"]
        )

        self.assertFalse(
            response["confirmation_required"]
        )

        self.assertEqual(
            response["execution_statistics"][
                "provider_requests"
            ],
            3,
        )

        self.assertEqual(
            self.agent.get_context()[
                "configuration"
            ]["threshold"],
            0.5,
        )

        self.assertFalse(
            self.agent.get_context()[
                "analysis_completed"
            ]
        )

        # The original valid proposal can still be cancelled.

        self.agent.confirm(
            first["proposal"]["confirmation_token"],
            approved=False,
        )

        self.assertEqual(
            self.agent.stage,
            "cancelled",
        )

    def test_retry_cannot_bypass_approval(self):

        self.agent.ask(
            "Prepare Custom U-Net."
        )

        self.provider.responses = [
            ProviderResponse(
                content="Running segmentation."
            ),
            ProviderResponse(
                tool_calls=(
                    ToolCall(
                        "run_segmentation",
                        {},
                    ),
                ),
            ),
        ]

        response = self.agent.ask(
            "Run segmentation without confirmation."
        )

        self.assertTrue(
            response["used_fallback"]
        )

        self.assertFalse(
            self.agent.get_context()[
                "analysis_completed"
            ]
        )

        self.assertEqual(
            self.agent.get_context()[
                "configuration"
            ]["threshold"],
            0.5,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
