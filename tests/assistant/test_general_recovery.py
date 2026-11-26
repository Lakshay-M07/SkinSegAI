"""Regression tests for general analytical tool recovery."""

import unittest

from PIL import Image, ImageDraw

from src.assistant.core.agent import SkinSegAgent

from src.assistant.providers.base_provider import (
    BaseProvider,
    ProviderResponse,
    ToolCall,
)


class RecoveryProvider(BaseProvider):

    name = "recovery_test"

    def __init__(self, responses):

        self.responses = list(responses)
        self.requests = []

    def chat(self, messages, *, tools=()):

        self.requests.append(messages)

        return self.responses.pop(0)


class TestGeneralRecovery(unittest.TestCase):

    def setUp(self):

        self.agent = SkinSegAgent()

        image = Image.new(
            "RGB",
            (320, 240),
            (210, 170, 145),
        )

        ImageDraw.Draw(image).ellipse(
            (100, 60, 220, 180),
            fill=(80, 45, 30),
        )

        self.agent.upload_image(image)

    def test_model_recommendation_recovery(self):

        provider = RecoveryProvider([
            ProviderResponse(
                content="I recommend a lightweight model."
            ),
            ProviderResponse(
                tool_calls=(
                    ToolCall(
                        "get_available_models",
                        {},
                    ),
                ),
            ),
        ])

        self.agent.set_provider(provider)

        response = self.agent.ask(
            "Which model should I choose for speed?"
        )

        self.assertFalse(response["used_fallback"])

        self.assertIn(
            "For speed",
            response["message"],
        )

        self.assertIn(
            "Custom U-Net",
            response["message"],
        )

        self.assertEqual(
            response["execution_statistics"][
                "provider_requests"
            ],
            2,
        )

        self.assertEqual(
            response["execution_statistics"][
                "tool_executions"
            ],
            1,
        )

        self.assertEqual(
            len(provider.requests[1]),
            2,
        )

        self.assertIn(
            "GENERAL TOOL ROUTING RETRY",
            provider.requests[1][0]["content"],
        )

    def test_uncertainty_recovery(self):

        self.agent.begin()

        self.agent.choose_objective("speed")

        self.agent.choose_models()

        proposal = self.agent.choose_threshold(0.5)

        self.agent.confirm(
            proposal["confirmation_token"],
            approved=True,
        )

        before = self.agent.get_context()

        provider = RecoveryProvider([
            ProviderResponse(
                content="Uncertainty describes ambiguity."
            ),
            ProviderResponse(
                tool_calls=(
                    ToolCall(
                        "analyze_uncertainty",
                        {},
                    ),
                ),
            ),
        ])

        self.agent.set_provider(provider)

        response = self.agent.ask(
            "Does lower boundary uncertainty "
            "prove higher accuracy?"
        )

        self.assertFalse(response["used_fallback"])

        self.assertIn(
            "does not prove",
            response["message"],
        )

        self.assertIn(
            "confident but incorrect",
            response["message"],
        )

        self.assertIn(
            "Boundary uncertainty",
            response["message"],
        )

        self.assertEqual(
            before,
            self.agent.get_context(),
        )

        self.agent.release_models()

    def test_general_retry_failure(self):

        provider = RecoveryProvider([
            ProviderResponse(
                content="Unverified response."
            ),
            ProviderResponse(
                content="Another unverified response."
            ),
        ])

        self.agent.set_provider(provider)

        before = self.agent.get_context()

        response = self.agent.ask(
            "Provide an unsupported explanation."
        )

        self.assertTrue(
            response["used_fallback"]
        )

        self.assertEqual(
            response["execution_statistics"][
                "provider_requests"
            ],
            3,
        )

        self.assertEqual(
            response["execution_statistics"][
                "tool_executions"
            ],
            0,
        )

        self.assertEqual(
            before,
            self.agent.get_context(),
        )

    def test_general_retry_cannot_execute_segmentation(self):

        provider = RecoveryProvider([
            ProviderResponse(
                content="Executing segmentation."
            ),
            ProviderResponse(
                tool_calls=(
                    ToolCall(
                        "run_segmentation",
                        {},
                    ),
                ),
            ),
        ])

        self.agent.set_provider(provider)

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

        self.assertFalse(
            response["confirmation_required"]
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
