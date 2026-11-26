
"""Grounded explanations for verified segmentation results."""

import math


class GroundedResponseComposer:
    """Compose evidence-based, question-specific explanations."""

    MAX_RESPONSE_CHARACTERS = 4000

    MODEL_NAMES = {
        "custom_unet": "Custom U-Net",
        "resnet34_unet": "ResNet-34 U-Net",
        "grabcut": "GrabCut",
    }

    def compose(
        self,
        tool_name,
        tool_response,
        question,
        summary,
    ):
        """Enhance a verified tool response without inventing facts."""

        if not isinstance(tool_response, dict):
            return summary

        if tool_response.get("status") != "success":
            return summary

        result = tool_response.get("result")

        if isinstance(result, dict):

            if result.get("available") is False:
                return summary

        question = question.lower().strip()

        handlers = {
            "get_available_models": self._explain_models,
            "get_session_context": self._explain_session,
            "analyze_uncertainty": self._explain_uncertainty,
            "analyze_disagreement": self._explain_agreement,
            "analyze_thresholds": self._explain_thresholds,
            "inspect_image_quality": self._explain_quality,
        }

        handler = handlers.get(tool_name)

        if handler is None:
            return summary

        explanation = handler(result, question)

        if not explanation:
            return summary

        answer = f"{explanation}\n\n{summary}"

        if len(answer) > self.MAX_RESPONSE_CHARACTERS:
            return summary

        return answer

    @staticmethod
    def _valid_probability(value):

        return (
            type(value) in (int, float)
            and math.isfinite(value)
            and 0 <= value <= 1
        )

    def _extract_benchmarks(self, result):

        if isinstance(result, list):
            contexts = result

        elif isinstance(result, dict):

            if "benchmark" in result:
                contexts = [result]

            else:
                contexts = list(result.values())

        else:
            return []

        benchmarks = []

        for context in contexts:

            if not isinstance(context, dict):
                continue

            benchmark = context.get("benchmark")

            if not isinstance(benchmark, dict):
                continue

            model_id = benchmark.get(
                "model_id",
                context.get("model_id"),
            )

            if model_id not in self.MODEL_NAMES:
                continue

            dice = benchmark.get("dice")

            if not self._valid_probability(dice):
                continue

            benchmarks.append({
                "model_id": model_id,
                "name": self.MODEL_NAMES[model_id],
                "dice": dice,
                "samples": benchmark.get("samples"),
            })

        return benchmarks

    def _explain_models(self, result, question):

        benchmarks = self._extract_benchmarks(result)

        if not benchmarks:
            return ""

        ranked = sorted(
            benchmarks,
            key=lambda item: item["dice"],
            reverse=True,
        )

        best = ranked[0]

        speed_requested = any(
            term in question
            for term in (
                "fastest",
                "speed",
                "latency",
                "lightweight",
                "quick",
                "faster",
            )
        )

        if speed_requested:

            custom_available = any(
                item["model_id"] == "custom_unet"
                for item in ranked
            )

            if custom_available:

                return (
                    "**For speed and lower computational "
                    "requirements, consider Custom U-Net.** "
                    "Its architecture is lighter than the "
                    "ResNet-34-based model.\n\n"
                    "Actual inference speed depends on the "
                    "device, image preprocessing and whether "
                    "the model is already loaded. Timings "
                    "measured on different hardware should "
                    "not be directly compared."
                )

        explanation = (
            f"**{best['name']} has the highest recorded "
            f"mean Dice score among the benchmarked models: "
            f"{best['dice']:.2%}.**"
        )

        if len(ranked) > 1:

            second = ranked[1]

            difference = (
                best["dice"] - second["dice"]
            ) * 100

            explanation += (
                f" Its mean Dice score is "
                f"{difference:.2f} percentage points "
                f"higher than {second['name']}."
            )

        if any(
            term in question
            for term in (
                "why",
                "recommend",
                "choose",
                "which",
                "better",
                "best",
            )
        ):

            explanation += (
                "\n\nIf your objective is the highest "
                "recorded benchmark Dice score, this "
                "model is a reasonable starting point. "
                "However, better average benchmark "
                "performance does not guarantee better "
                "segmentation on every individual image."
            )

        if "why" in question:

            explanation += (
                "\n\nArchitectural differences can affect "
                "feature extraction and boundary detection. "
                "The benchmark alone does not establish "
                "which architectural feature caused the "
                "performance difference."
            )

        return explanation

    @staticmethod
    def _explain_session(result, question):

        if not isinstance(result, dict):
            return ""

        if not result.get("analysis_completed"):
            return ""

        if any(
            term in question
            for term in (
                "area",
                "size",
                "results",
                "summarize",
                "explain",
                "meaning",
            )
        ):

            return (
                "**The segmentation identifies pixels "
                "that the selected algorithms predict "
                "belong to the lesion.**\n\n"
                "The predicted lesion area represents the "
                "fraction of image pixels classified as "
                "foreground. Without physical calibration, "
                "it does not establish the lesion's real-world "
                "size in millimetres or centimetres."
            )

        return ""

    @staticmethod
    def _explain_uncertainty(result, question):

        if not isinstance(result, dict):
            return ""

        return (
            "**Predictive uncertainty describes how "
            "ambiguous the model's predicted probabilities "
            "are.**\n\n"
            "Lower entropy indicates more decisive pixel "
            "predictions. Higher entropy near the lesion "
            "boundary suggests that the model is less "
            "decisive about where the predicted foreground "
            "ends and the background begins.\n\n"
            "A lower boundary uncertainty value does not "
            "prove that the boundary is more accurate. "
            "A model can make confident but incorrect "
            "predictions."
        )

    @staticmethod
    def _explain_agreement(result, question):

        if not isinstance(result, dict):
            return ""

        return (
            "**Model agreement measures how similarly "
            "the selected algorithms segment the image.**"
            "\n\n"
            "A high Dice agreement indicates substantial "
            "overlap between their predicted masks. "
            "Disagreement identifies regions where their "
            "foreground and background classifications "
            "differ.\n\n"
            "Even very high agreement does not establish "
            "accuracy: both algorithms could make the "
            "same mistake. Measuring segmentation accuracy "
            "requires an appropriate reference mask."
        )

    @staticmethod
    def _explain_thresholds(result, question):

        if not isinstance(result, dict):
            return ""

        return (
            "**The prediction threshold controls which "
            "pixels are included in the neural "
            "segmentation mask.**\n\n"
            "For a fixed probability map, increasing the "
            "threshold produces a smaller or equally sized "
            "foreground mask. Decreasing it produces a "
            "larger or equally sized mask.\n\n"
            "A higher threshold may remove ambiguous "
            "boundary pixels, but it can also exclude "
            "genuine lesion pixels. The best threshold "
            "cannot be established from predicted area "
            "alone.\n\n"
            "This analysis previews the effect of different "
            "thresholds without changing active settings "
            "or rerunning the segmentation models."
        )

    @staticmethod
    def _explain_quality(result, question):

        if not isinstance(result, dict):
            return ""

        if any(
            term in question
            for term in (
                "why",
                "improve",
                "affect",
                "quality",
                "reliable",
                "better",
            )
        ):

            return (
                "**Image quality can influence "
                "segmentation reliability.**\n\n"
                "Limited resolution, poor contrast, blur "
                "and uneven illumination can make lesion "
                "boundaries more difficult to distinguish. "
                "The following checks identify potential "
                "technical limitations rather than "
                "diagnosing medical conditions."
            )

        return ""
