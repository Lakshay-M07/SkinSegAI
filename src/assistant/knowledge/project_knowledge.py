
"""Project information and metric definitions for SkinSeg Assistant."""

from copy import deepcopy


class ProjectKnowledgeBase:
    """Provide educational information about SkinSegAI."""

    _PROJECT = {
        "name": "SkinSegAI",
        "purpose": (
            "Educational and research application "
            "for skin lesion segmentation."
        ),
        "language": "Python",
        "interface": "Streamlit",
        "deep_learning_framework": "PyTorch",
        "dataset": {
            "name": "ISIC 2018",
            "total_images": 2594,
            "training_images": 1815,
            "validation_images": 389,
            "test_images": 390,
            "missing_patient_ids": 2204,
            "missing_lesion_ids": 1496,
            "known_cross_split_group_overlap": 0,
        },
        "supported_hardware": [
            "CPU",
            "Apple MPS",
            "NVIDIA CUDA",
        ],
        "medical_diagnosis_supported": False,
        "probability_calibration_verified": False,
        "default_prediction_threshold": 0.5,
    }

    _METRICS = {

        "dice": {
            "name": "Dice Similarity Coefficient",
            "explanation": (
                "Measures how closely a predicted lesion "
                "mask overlaps a reference mask."
            ),
            "range": [0, 1],
            "higher_is_better": True,
            "requires_ground_truth": True,
        },

        "iou": {
            "name": "Intersection over Union",
            "explanation": (
                "Measures the shared region between two masks "
                "relative to their combined area."
            ),
            "range": [0, 1],
            "higher_is_better": True,
            "requires_ground_truth": True,
        },

        "precision": {
            "name": "Precision",
            "explanation": (
                "Measures the proportion of predicted lesion "
                "pixels that are actually lesion pixels "
                "according to a reference mask."
            ),
            "range": [0, 1],
            "higher_is_better": True,
            "requires_ground_truth": True,
        },

        "recall": {
            "name": "Recall",
            "explanation": (
                "Measures the proportion of reference lesion "
                "pixels successfully identified by the model."
            ),
            "range": [0, 1],
            "higher_is_better": True,
            "requires_ground_truth": True,
        },

        "predictive_entropy": {
            "name": "Predictive Entropy",
            "explanation": (
                "Measures how ambiguous the model's "
                "foreground probability predictions are."
            ),
            "range": [0, 1],
            "requires_ground_truth": False,
            "is_accuracy": False,
            "is_calibrated_confidence": False,
        },

        "border_uncertainty": {
            "name": "Border Uncertainty",
            "explanation": (
                "Measures predictive ambiguity around "
                "the predicted lesion boundary."
            ),
            "range": [0, 1],
            "requires_ground_truth": False,
            "is_accuracy": False,
            "may_be_unavailable": True,
        },

        "model_agreement": {
            "name": "Model Agreement",
            "explanation": (
                "Measures how closely predictions from "
                "different segmentation models match."
            ),
            "range": [0, 1],
            "requires_ground_truth": False,
            "is_accuracy": False,
        },

        "threshold": {
            "name": "Prediction Threshold",
            "explanation": (
                "Determines the minimum predicted foreground "
                "probability required to classify a pixel "
                "as part of the lesion. Higher thresholds "
                "generally produce smaller or equal masks "
                "for a fixed probability map."
            ),
            "default": 0.5,
            "supported_range": [0.1, 0.9],
            "applies_to_grabcut": False,
        },
    }

    def get_project_context(self):

        return deepcopy(self._PROJECT)

    def explain_metric(self, metric):

        key = metric.strip().lower().replace(" ", "_")

        if key not in self._METRICS:

            raise ValueError(
                f"Unknown metric: {metric}"
            )

        return deepcopy(self._METRICS[key])

    def available_metrics(self):

        return list(self._METRICS)

    def get_all_metric_contexts(self):

        return deepcopy(self._METRICS)

    def get_limitations(self):

        return [
            (
                "Segmentation models identify predicted "
                "lesion regions, not diseases."
            ),
            (
                "Benchmark performance does not guarantee "
                "accuracy on an individual uploaded image."
            ),
            (
                "Image-specific accuracy requires "
                "a reference segmentation mask."
            ),
            (
                "Predictive entropy is not calibrated "
                "clinical confidence."
            ),
            (
                "Model agreement does not prove correctness."
            ),
            (
                "Missing patient and lesion identifiers "
                "prevent guaranteed patient-level independence."
            ),
            (
                "Increasing the threshold cannot guarantee "
                "a specific precision percentage."
            ),
        ]

    def get_disclaimer(self):

        return (
            "SkinSegAI is intended for educational and research "
            "purposes. It does not diagnose skin diseases or "
            "replace professional medical evaluation."
        )
