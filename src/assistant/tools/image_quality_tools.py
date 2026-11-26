
"""Assistant tool for uploaded image quality assessment."""

from src.assistant.tools.base_tool import BaseTool

from src.assistant.analysis.image_quality import (
    ImageQualityAnalyzer,
)


class InspectImageQualityTool(BaseTool):
    """Assess image quality before or after segmentation."""

    name = "inspect_image_quality"

    description = (
        "Inspect the uploaded image for potential blur, "
        "poor lighting, low contrast, insufficient resolution "
        "and bright reflections. Does not require segmentation "
        "or estimate medical accuracy."
    )

    def __init__(self):

        self._analyzer = ImageQualityAnalyzer()

    def execute(self, session, arguments):

        self.require_empty_arguments(arguments)

        if not session.has_image:

            raise RuntimeError(
                "Upload an image before assessing quality."
            )

        if not session.config.include_image_quality:

            return {
                "available": False,
                "reason": (
                    "Image quality assessment is disabled."
                ),
            }

        return self._analyzer.analyze(
            session.get_image()
        )
