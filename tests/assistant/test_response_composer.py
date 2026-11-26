
"""Regression tests for grounded assistant explanations."""

import unittest

from src.assistant.analysis.response_composer import (
    GroundedResponseComposer,
)

from src.assistant.core.agent import SkinSegAgent


class TestGroundedResponseComposer(unittest.TestCase):

    def setUp(self):

        self.composer = GroundedResponseComposer()

    def compose(self, tool, result, question):

        return self.composer.compose(
            tool_name=tool,
            tool_response={
                "status": "success",
                "tool": tool,
                "result": result,
            },
            question=question,
            summary="Verified tool summary.",
        )

    def test_dynamic_model_ranking(self):

        result = [
            {
                "benchmark": {
                    "model_id": "custom_unet",
                    "dice": 0.91,
                    "samples": 390,
                },
            },
            {
                "benchmark": {
                    "model_id": "resnet34_unet",
                    "dice": 0.88,
                    "samples": 390,
                },
            },
        ]

        answer = self.compose(
            "get_available_models",
            result,
            "Which model has the best Dice score?",
        )

        self.assertIn(
            "Custom U-Net has the highest",
            answer,
        )

        self.assertIn("91.00%", answer)

        self.assertIn(
            "3.00 percentage points",
            answer,
        )

        self.assertNotIn(
            "87.79%",
            answer,
        )

    def test_uncertainty_limitations(self):

        answer = self.compose(
            "analyze_uncertainty",
            {"available": True},
            "Does low uncertainty mean high accuracy?",
        )

        self.assertIn(
            "does not prove",
            answer,
        )

        self.assertIn(
            "confident but incorrect",
            answer,
        )

    def test_agreement_is_not_accuracy(self):

        answer = self.compose(
            "analyze_disagreement",
            {"available": True},
            "Does high agreement establish accuracy?",
        )

        self.assertIn(
            "does not establish accuracy",
            answer,
        )

        self.assertIn(
            "reference mask",
            answer,
        )

    def test_threshold_explanation(self):

        answer = self.compose(
            "analyze_thresholds",
            {"available": True},
            "What happens if I increase the threshold?",
        )

        self.assertIn(
            "smaller or equally sized",
            answer,
        )

        self.assertIn(
            "without changing active settings",
            answer,
        )

    def test_unavailable_analysis(self):

        answer = self.compose(
            "analyze_uncertainty",
            {"available": False},
            "Explain uncertainty.",
        )

        self.assertEqual(
            answer,
            "Verified tool summary.",
        )

    def test_failed_tool_response(self):

        answer = self.composer.compose(
            tool_name="get_available_models",
            tool_response={
                "status": "error",
                "tool": "get_available_models",
            },
            question="Which model is best?",
            summary="Model information unavailable.",
        )

        self.assertEqual(
            answer,
            "Model information unavailable.",
        )

    def test_real_benchmark_integration(self):

        agent = SkinSegAgent()

        response = agent.ask("models")

        self.assertIn(
            "ResNet-34 U-Net has the highest",
            response["message"],
        )

        self.assertIn(
            "87.79%",
            response["message"],
        )

        self.assertIn(
            "0.92 percentage points",
            response["message"],
        )

        self.assertIn(
            "86.87%",
            response["message"],
        )

        self.assertIn(
            "61.46%",
            response["message"],
        )

        self.assertFalse(
            agent.get_context()["analysis_completed"]
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
