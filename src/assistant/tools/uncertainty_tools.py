
"""Assistant tools for prediction uncertainty."""

from src.assistant.tools.base_tool import BaseTool

from src.assistant.analysis.confidence import (
    ConfidenceAnalyzer,
)


class AnalyzeUncertaintyTool(BaseTool):

    name = "analyze_uncertainty"

    description = (
        "Analyze predictive entropy, foreground probabilities "
        "and boundary uncertainty for existing segmentation "
        "results. Does not estimate medical accuracy."
    )

    def __init__(self):

        self._analyzer = ConfidenceAnalyzer()

    def execute(self, session, arguments):

        self.require_empty_arguments(arguments)

        if not session.has_results:

            raise RuntimeError(
                "Run segmentation before analyzing uncertainty."
            )

        if not session.config.include_uncertainty:

            return {
                "available": False,
                "reason": (
                    "Uncertainty analysis is disabled "
                    "in the current configuration."
                ),
            }

        results = self._analyzer.analyze_all(
            session.get_results()
        )

        available = any(
            result["available"]
            for result in results
        )

        response = {
            "available": available,
            "results": results,
        }

        if not available:

            response["reason"] = (
                "None of the selected algorithms provide "
                "neural predictive uncertainty maps."
            )

        return response
