
"""Assistant tools for threshold sensitivity analysis."""

from src.assistant.tools.base_tool import BaseTool

from src.assistant.analysis.threshold_analysis import (
    ThresholdAnalyzer,
)


class AnalyzeThresholdsTool(BaseTool):

    name = "analyze_thresholds"

    description = (
        "Analyze how different prediction thresholds "
        "affect predicted lesion area using existing "
        "probability maps. Does not modify the current "
        "configuration or rerun neural network inference."
    )

    parameters = {
        "type": "object",
        "properties": {
            "thresholds": {
                "type": "array",
                "items": {
                    "type": "number",
                    "minimum": 0.1,
                    "maximum": 0.9,
                },
                "minItems": 1,
                "maxItems": 20,
            },
        },
        "required": [],
        "additionalProperties": False,
    }

    def __init__(self):

        self._analyzer = ThresholdAnalyzer()

    def execute(self, session, arguments):

        unknown = set(arguments) - {"thresholds"}

        if unknown:
            raise ValueError(
                f"Unsupported arguments: {unknown}"
            )

        if not session.has_results:
            raise RuntimeError(
                "Run segmentation before analyzing thresholds."
            )

        results = self._analyzer.analyze_all(
            session.get_results(),
            thresholds=arguments.get("thresholds"),
            current_threshold=session.config.threshold,
            ground_truth=session.get_ground_truth(),
        )

        return {
            "available": any(
                result["available"]
                for result in results
            ),

            "configuration_changed": False,

            "results": results,
        }
