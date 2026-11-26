
"""Assistant tools for segmentation model comparison."""

from src.assistant.tools.base_tool import BaseTool

from src.assistant.analysis.disagreement import (
    ModelDisagreementAnalyzer,
)


class AnalyzeDisagreementTool(BaseTool):

    name = "analyze_disagreement"

    description = (
        "Compare existing segmentation masks to determine "
        "model agreement, disagreement and differences "
        "in predicted lesion area."
    )

    def __init__(self):

        self._analyzer = ModelDisagreementAnalyzer()

    def execute(self, session, arguments):

        self.require_empty_arguments(arguments)

        if not session.has_results:

            raise RuntimeError(
                "Run segmentation before comparing predictions."
            )

        if not session.config.include_disagreement:

            return {
                "available": False,
                "reason": (
                    "Disagreement analysis is disabled."
                ),
            }

        results = session.get_results()

        if len(results) < 2:

            return {
                "available": False,
                "reason": (
                    "At least two segmentation models "
                    "are required for comparison."
                ),
            }

        return {
            "available": True,
            "comparisons": (
                self._analyzer.compare_all(results)
            ),
        }
