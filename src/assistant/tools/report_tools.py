
"""Enhanced report generation for SkinSeg Assistant."""

import json

from datetime import datetime, timezone
from io import BytesIO
from zipfile import ZipFile, ZIP_DEFLATED

from src.services.report_exporter import ReportExporter

from src.assistant.core.session_context import SessionContext

from src.assistant.analysis.confidence import ConfidenceAnalyzer

from src.assistant.analysis.disagreement import (
    ModelDisagreementAnalyzer,
)

from src.assistant.analysis.threshold_analysis import (
    ThresholdAnalyzer,
)

from src.assistant.analysis.image_quality import (
    ImageQualityAnalyzer,
)


class AssistantReportExporter:
    """Extend the existing report with verified assistant analysis."""

    def __init__(self):

        self._exporter = ReportExporter()

        self._confidence = ConfidenceAnalyzer()

        self._disagreement = ModelDisagreementAnalyzer()

        self._threshold = ThresholdAnalyzer()

        self._quality = ImageQualityAnalyzer()

    @staticmethod
    def _encode_json(data):

        return json.dumps(
            data,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")

    @staticmethod
    def _unavailable(reason):

        return {
            "available": False,
            "reason": reason,
        }

    def build_report(self, session):

        if not isinstance(session, SessionContext):
            raise TypeError("Invalid session context.")

        if not session.has_image:
            raise RuntimeError(
                "Upload an image before exporting a report."
            )

        if not session.has_results:
            raise RuntimeError(
                "Complete segmentation before exporting a report."
            )

        context = session.get_context()

        config = session.config

        results = session.get_results()

        # Predictive uncertainty.

        if config.include_uncertainty:

            entries = self._confidence.analyze_all(
                results
            )

            uncertainty = {
                "available": any(
                    entry["available"]
                    for entry in entries
                ),
                "results": entries,
            }

        else:

            uncertainty = self._unavailable(
                "Uncertainty analysis is disabled."
            )

        # Model agreement.

        if not config.include_disagreement:

            disagreement = self._unavailable(
                "Model disagreement analysis is disabled."
            )

        elif len(results) < 2:

            disagreement = self._unavailable(
                "At least two models are required."
            )

        else:

            disagreement = {
                "available": True,
                "comparisons": (
                    self._disagreement.compare_all(results)
                ),
            }

        # Threshold sensitivity.

        threshold_entries = (
            self._threshold.analyze_all(
                results,
                current_threshold=config.threshold,
                ground_truth=session.get_ground_truth(),
            )
        )

        thresholds = {
            "available": any(
                entry["available"]
                for entry in threshold_entries
            ),
            "results": threshold_entries,
            "configuration_changed": False,
        }

        # Image quality.

        if config.include_image_quality:

            quality = self._quality.analyze(
                session.get_image()
            )

        else:

            quality = self._unavailable(
                "Image quality assessment is disabled."
            )

        return {
            "report_version": "1.0",

            "generated_at_utc": (
                datetime.now(timezone.utc).isoformat()
            ),

            "session_context": context,

            "uncertainty": uncertainty,

            "model_agreement": disagreement,

            "threshold_sensitivity": thresholds,

            "image_quality": quality,

            "llm_generated_statistics": False,

            "medical_diagnosis": False,

            "disclaimer": (
                "SkinSegAI is intended for educational "
                "and research purposes. Predictions, "
                "uncertainty and model agreement do not "
                "constitute a medical diagnosis."
            ),
        }

    def _build_summary(self, report):

        context = report["session_context"]

        image = context["image"]

        configuration = context["configuration"]

        lines = [
            "SKINSEGAI ASSISTANT REPORT",
            "=" * 50,

            f"Generated: {report['generated_at_utc']}",

            f"Image dimensions: "
            f"{image['width']} x {image['height']} pixels",

            f"Neural prediction threshold: "
            f"{configuration['threshold']:.2f}",

            "",
            "SEGMENTATION RESULTS",
            "-" * 50,
        ]

        for result in context["results"]:

            lines.extend([
                "",
                result["algorithm"],

                f"Predicted lesion area: "
                f"{result['predicted_lesion_area_percent']:.2f}%",

                f"Predicted lesion pixels: "
                f"{result['predicted_lesion_pixels']}",

                f"Mask status: {result['mask_status']}",

                f"Inference time: "
                f"{result['inference_ms']:.2f} ms",
            ])

            if result["ground_truth_available"]:

                for metric, value in (
                    result["ground_truth_metrics"].items()
                ):

                    lines.append(
                        f"{metric}: {value}"
                    )

            else:

                lines.append(
                    "Image-specific accuracy: "
                    "Unavailable without ground truth."
                )

        lines.extend([
            "",
            "PREDICTIVE UNCERTAINTY",
            "-" * 50,
        ])

        uncertainty = report["uncertainty"]

        for result in uncertainty.get("results", []):

            lines.append(result["algorithm"])

            if not result["available"]:

                lines.append(result["reason"])

                continue

            lines.append(
                f"Mean entropy: "
                f"{result['mean_predictive_entropy']:.4f}"
            )

            border = result["border_uncertainty"]

            lines.append(
                "Boundary uncertainty: "
                + (
                    f"{border:.4f}"
                    if border is not None
                    else "Unavailable"
                )
            )

            lines.append(
                f"High-entropy pixels: "
                f"{result['high_entropy_pixel_percent']:.2f}%"
            )

            lines.extend(result.get("warnings", []))

        if not uncertainty.get("results"):

            lines.append(
                uncertainty.get("reason", "Unavailable")
            )

        lines.extend([
            "",
            "MODEL AGREEMENT",
            "-" * 50,
        ])

        agreement = report["model_agreement"]

        if agreement["available"]:

            for pair in agreement["comparisons"]:

                lines.append(
                    f"{pair['first_model']} vs "
                    f"{pair['second_model']}"
                )

                dice = pair["dice_agreement_percent"]

                lines.append(
                    "Dice agreement: "
                    + (
                        f"{dice:.2f}%"
                        if dice is not None
                        else "Undefined"
                    )
                )

                lines.append(
                    f"Disagreement pixels: "
                    f"{pair['disagreement_pixel_percent']:.2f}%"
                )

        else:

            lines.append(agreement["reason"])

        lines.extend([
            "",
            "IMAGE QUALITY",
            "-" * 50,
        ])

        quality = report["image_quality"]

        if not quality["available"]:

            lines.append(quality["reason"])

        elif not quality["issues"]:

            lines.append(
                "No obvious technical issues detected."
            )

        else:

            for issue in quality["issues"]:

                lines.append(issue["explanation"])

                lines.append(
                    "Recommendation: "
                    + issue["recommendation"]
                )

        lines.extend([
            "",
            "THRESHOLD SENSITIVITY",
            "-" * 50,

            "See assistant/threshold_sensitivity.json "
            "for the full numerical analysis.",

            "",
            "LIMITATIONS",
            "-" * 50,

            "Benchmark performance does not establish "
            "individual image accuracy.",

            "Predictive entropy is not calibrated confidence.",

            "Model agreement does not establish correctness.",

            "Image quality checks are heuristic.",

            "Pixel measurements are not physical dimensions.",

            "GrabCut does not use neural prediction thresholds.",

            "",
            report["disclaimer"],
        ])

        return "\n".join(lines)

    def export_zip(self, session):
        """Generate an enhanced report entirely in memory."""

        report = self.build_report(session)

        original_zip = self._exporter.export_zip(
            session.get_results()
        )

        buffer = BytesIO(original_zip)

        files = {
            "assistant/session_context.json": {
                "report_version": report["report_version"],
                "generated_at_utc": report["generated_at_utc"],
                **report["session_context"],
            },

            "assistant/uncertainty.json": (
                report["uncertainty"]
            ),

            "assistant/model_agreement.json": (
                report["model_agreement"]
            ),

            "assistant/threshold_sensitivity.json": (
                report["threshold_sensitivity"]
            ),

            "assistant/image_quality.json": (
                report["image_quality"]
            ),
        }

        with ZipFile(
            buffer,
            mode="a",
            compression=ZIP_DEFLATED,
        ) as archive:

            existing = set(archive.namelist())

            new_names = set(files) | {
                "assistant/summary.txt"
            }

            if existing & new_names:

                raise ValueError(
                    "Report contains conflicting file names."
                )

            for name, content in files.items():

                archive.writestr(
                    name,
                    self._encode_json(content),
                )

            archive.writestr(
                "assistant/summary.txt",
                self._build_summary(report).encode("utf-8"),
            )

        return buffer.getvalue()
