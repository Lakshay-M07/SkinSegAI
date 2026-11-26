
"""Deterministic offline provider for SkinSeg Assistant."""

import json
import re

from src.assistant.providers.base_provider import (
    BaseProvider,
    ProviderResponse,
    ToolCall,
)


class OfflineProvider(BaseProvider):
    """Provide useful assistance without an external language model."""

    name = "offline"

    COMMANDS = {
        "models": "get_available_models",
        "status": "get_session_context",
        "uncertainty": "analyze_uncertainty",
        "compare": "analyze_disagreement",
        "thresholds": "analyze_thresholds",
        "quality": "inspect_image_quality",
    }

    ALIASES = {
        "what models are available": "models",
        "which model is best": "models",
        "which model is fastest": "models",
        "show model performance": "models",
        "show benchmarks": "models",
        "explain my results": "status",
        "show my results": "status",
        "have i uploaded an image": "status",
        "show uncertainty": "uncertainty",
        "compare my results": "compare",
        "show model agreement": "compare",
        "analyze thresholds": "thresholds",
        "check image quality": "quality",
        "inspect image quality": "quality",
    }

    HELP_MESSAGE = (
        "SkinSeg Assistant is operating in offline mode.\n\n"
        "Available commands:\n"
        "- models: Model descriptions and benchmarks\n"
        "- status: Current image and segmentation results\n"
        "- uncertainty: Predictive uncertainty analysis\n"
        "- compare: Compare existing segmentation masks\n"
        "- thresholds: Explore prediction thresholds\n"
        "- quality: Inspect image quality\n\n"
        "Use the guided segmentation workflow to select "
        "models, configure thresholds and run analysis."
    )

    @staticmethod
    def _normalize(text):

        text = text.strip().lower()

        text = re.sub(
            r"[^a-z0-9\s]",
            " ",
            text,
        )

        return " ".join(text.split())

    @staticmethod
    def _available_tools(tools):

        return {
            schema["function"]["name"]
            for schema in tools
        }

    @staticmethod
    def _percent(value):

        if value is None:
            return "Unavailable"

        return f"{value * 100:.2f}%"

    @staticmethod
    def _number(value, digits=4):

        if value is None:
            return "Unavailable"

        return f"{value:.{digits}f}"

    def chat(self, messages, *, tools=()):

        if not isinstance(messages, (list, tuple)):
            raise TypeError("Messages must be a list or tuple.")

        if not messages:

            return ProviderResponse(
                content=self.HELP_MESSAGE
            )

        latest = messages[-1]

        role = latest.get("role")

        content = latest.get("content", "")

        if not isinstance(content, str):
            raise TypeError("Message content must be a string.")

        if role == "tool":

            return self._handle_tool_result(latest)

        if role != "user":

            return ProviderResponse(
                content=self.HELP_MESSAGE
            )

        normalized = self._normalize(content)

        command = self.ALIASES.get(
            normalized,
            normalized,
        )

        if command in (
            "hi",
            "hello",
            "hey",
            "help",
        ):

            return ProviderResponse(
                content=self.HELP_MESSAGE
            )

        tool_name = self.COMMANDS.get(command)

        if tool_name is None:

            return ProviderResponse(
                content=(
                    "I cannot reliably interpret that request "
                    "in offline mode. Use the guided workflow "
                    "or one of the supported commands.\n\n"
                    + self.HELP_MESSAGE
                )
            )

        if tool_name not in self._available_tools(tools):

            return ProviderResponse(
                content=(
                    "That analysis tool is currently unavailable."
                )
            )

        return ProviderResponse(
            tool_calls=(
                ToolCall(
                    name=tool_name,
                    arguments={},
                ),
            )
        )

    def _handle_tool_result(self, message):

        try:

            payload = json.loads(
                message["content"]
            )

        except (KeyError, TypeError, ValueError):

            return ProviderResponse(
                content="Unable to interpret the tool response."
            )

        if not isinstance(payload, dict):

            return ProviderResponse(
                content="Invalid tool response."
            )

        if payload.get("status") != "success":

            return ProviderResponse(
                content=(
                    "The requested analysis could not be completed. "
                    "Check whether an image and the required "
                    "segmentation results are available."
                )
            )

        name = message.get("tool_name")

        if name != payload.get("tool"):

            return ProviderResponse(
                content="Tool response identity mismatch."
            )

        result = payload.get("result")

        formatters = {
            "get_available_models": self._format_models,
            "get_session_context": self._format_session,
            "analyze_uncertainty": self._format_uncertainty,
            "analyze_disagreement": self._format_comparison,
            "analyze_thresholds": self._format_thresholds,
            "inspect_image_quality": self._format_quality,
        }

        formatter = formatters.get(name)

        if formatter is None:

            return ProviderResponse(
                content="This tool is unsupported in offline mode."
            )

        try:

            content = formatter(result)

        except (KeyError, TypeError, ValueError, AttributeError):

            content = "Unable to interpret the analysis results."

        return ProviderResponse(content=content)

    def _format_models(self, models):

        lines = ["Available segmentation models:"]

        for model in models:

            lines.append(
                f"\n**{model['display_name']}**"
            )

            lines.append(model["description"])

            benchmark = model.get("benchmark")

            if benchmark is not None:

                lines.append(
                    f"- Mean Dice: "
                    f"{self._percent(benchmark['dice'])}"
                )

                lines.append(
                    f"- IoU: "
                    f"{self._percent(benchmark['iou'])}"
                )

                lines.append(
                    f"- Precision: "
                    f"{self._percent(benchmark['precision'])}"
                )

                lines.append(
                    f"- Recall: "
                    f"{self._percent(benchmark['recall'])}"
                )

                lines.append(
                    f"- Test images: {benchmark['samples']}"
                )

            else:

                lines.append(
                    "- Verified benchmark unavailable."
                )

        lines.append(
            "\nBenchmark results do not establish "
            "accuracy on an individual image."
        )

        return "\n".join(lines)

    def _format_session(self, result):

        if not result["image_uploaded"]:

            return "Please upload an image to begin."

        image = result["image"]

        lines = [
            f"Image dimensions: "
            f"{image['width']} × {image['height']} pixels.",

            f"Current threshold: "
            f"{result['configuration']['threshold']:.2f}",
        ]

        if not result["analysis_completed"]:

            lines.append(
                "Segmentation has not been performed yet."
            )

            return "\n".join(lines)

        for entry in result["results"]:

            lines.append(
                f"\n**{entry['algorithm']}**"
            )

            lines.append(
                f"- Predicted lesion area: "
                f"{entry['predicted_lesion_area_percent']:.2f}%"
            )

            lines.append(
                f"- Mask status: {entry['mask_status']}"
            )

            lines.append(
                f"- Inference time: "
                f"{entry['inference_ms']:.2f} ms"
            )

        lines.append(
            "\nPredicted lesion area is not a medical diagnosis."
        )

        return "\n".join(lines)

    def _format_uncertainty(self, result):

        if not result["available"]:

            return result["reason"]

        lines = ["Predictive uncertainty analysis:"]

        for entry in result["results"]:

            lines.append(
                f"\n**{entry['algorithm']}**"
            )

            if not entry["available"]:

                lines.append(entry["reason"])

                continue

            lines.append(
                f"- Mean entropy: "
                f"{self._number(entry['mean_predictive_entropy'])}"
            )

            lines.append(
                f"- Boundary uncertainty: "
                f"{self._number(entry['border_uncertainty'])}"
            )

            lines.append(
                f"- High-entropy pixels: "
                f"{entry['high_entropy_pixel_percent']:.2f}%"
            )

            lines.extend(
                entry.get("warnings", [])
            )

        lines.append(
            "\nPredictive entropy is not calibrated confidence "
            "or an estimate of segmentation accuracy."
        )

        return "\n".join(lines)

    def _format_comparison(self, result):

        if not result["available"]:

            return result["reason"]

        lines = ["Pairwise model agreement:"]

        for entry in result["comparisons"]:

            lines.append(
                f"\n**{entry['first_model']} vs "
                f"{entry['second_model']}**"
            )

            lines.append(
                f"- Dice agreement: "
                f"{self._percent(entry['dice_agreement'])}"
            )

            lines.append(
                f"- Disagreement pixels: "
                f"{entry['disagreement_pixel_percent']:.2f}%"
            )

            lines.append(
                f"- Area difference: "
                f"{entry['area_difference_percentage_points']:.2f} "
                "percentage points"
            )

            lines.extend(
                entry.get("warnings", [])
            )

        lines.append(
            "\nAgreement measures similarity between "
            "predictions, not correctness."
        )

        return "\n".join(lines)

    def _format_thresholds(self, result):

        if not result["available"]:

            return (
                "Threshold analysis is unavailable "
                "for the current segmentation results."
            )

        lines = ["Threshold sensitivity analysis:"]

        for entry in result["results"]:

            lines.append(
                f"\n**{entry['algorithm']}**"
            )

            if not entry["available"]:

                lines.append(entry["reason"])

                continue

            for row in entry["threshold_results"]:

                lines.append(
                    f"- Threshold {row['threshold']:.2f}: "
                    f"{row['predicted_lesion_area_percent']:.2f}% area"
                )

        lines.append(
            "\nThese are predicted areas, not accuracy "
            "estimates. Current settings remain unchanged."
        )

        return "\n".join(lines)

    def _format_quality(self, result):

        if not result["available"]:

            return result["reason"]

        if not result["issues"]:

            return (
                "No obvious technical image quality issues "
                "were detected by the available checks. "
                "This does not guarantee segmentation accuracy."
            )

        lines = [
            "Potential image quality issues detected:"
        ]

        for issue in result["issues"]:

            lines.append(
                f"\n- {issue['explanation']}"
            )

            lines.append(
                f"  Recommendation: {issue['recommendation']}"
            )

        lines.append(
            "\nThese checks are heuristic and do not "
            "establish clinical image suitability."
        )

        return "\n".join(lines)
