
"""Segmentation tools available to SkinSeg Assistant."""

from src.assistant.tools.base_tool import BaseTool

from src.assistant.knowledge.model_knowledge import (
    ModelKnowledgeBase,
)

from src.services.segmentation_service import (
    SegmentationService,
)


class GetAvailableModelsTool(BaseTool):
    """Retrieve available models and verified benchmarks."""

    name = "get_available_models"

    description = (
        "Retrieve available segmentation models, "
        "their capabilities and verified benchmark metrics."
    )

    def __init__(self, project_root=None):

        self._knowledge = ModelKnowledgeBase(
            project_root
        )

    def execute(self, session, arguments):

        self.require_empty_arguments(arguments)

        return self._knowledge.get_all_model_contexts()


class GetSessionContextTool(BaseTool):
    """Retrieve information about the current image and results."""

    name = "get_session_context"

    description = (
        "Retrieve the current image information, "
        "segmentation settings and existing analysis results. "
        "Does not perform new segmentation."
    )

    def execute(self, session, arguments):

        self.require_empty_arguments(arguments)

        return session.get_context()


class ProposeConfigurationTool(BaseTool):
    """Propose segmentation settings without applying them."""

    name = "propose_configuration"


    description = (
        "Create or revise a proposed segmentation configuration "
        "according to the user's instructions. "
        "For follow-up requests, provide only the fields that "
        "the user wants to change. Unspecified fields are "
        "preserved from the valid pending configuration, "
        "or from the active configuration if no proposal exists. "
        "Use this tool when the user requests changes to models, "
        "prediction threshold, overlay opacity, inference passes "
        "or analysis options. "
        "Does not apply settings or execute segmentation. "
        "Explicit user confirmation is required."
    )


    parameters = {
        "type": "object",
        "properties": {
            "models": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": [
                        "custom_unet",
                        "resnet34_unet",
                        "grabcut",
                    ],
                },
                "minItems": 1,
                "maxItems": 3,
                "uniqueItems": True,
            },
            "threshold": {
                "type": "number",
                "minimum": 0.1,
                "maximum": 0.9,
            },
            "overlay_alpha": {
                "type": "number",
                "minimum": 0,
                "maximum": 1,
            },
            "mc_passes": {
                "type": "integer",
                "minimum": 1,
                "maximum": 20,
            },
            "include_uncertainty": {
                "type": "boolean",
            },
            "include_morphology": {
                "type": "boolean",
            },
            "include_disagreement": {
                "type": "boolean",
            },
            "include_image_quality": {
                "type": "boolean",
            },
        },
        "required": [],
        "additionalProperties": False,
    }

    def __init__(self, project_root=None):

        self._knowledge = ModelKnowledgeBase(
            project_root
        )

    def execute(self, session, arguments):

        allowed = set(
            self.parameters["properties"]
        )

        unknown = set(arguments) - allowed

        if unknown:
            raise ValueError(
                f"Unsupported configuration fields: {unknown}"
            )

        proposed = session.config.updated(
            **arguments
        )

        available = set(
            self._knowledge.available_models()
        )

        unavailable = set(proposed.models) - available

        if unavailable:
            raise ValueError(
                f"Unavailable models: {unavailable}"
            )

        warnings = []

        if not proposed.supports_uncertainty:

            warnings.append(
                "GrabCut does not provide neural "
                "probability or uncertainty maps."
            )

            warnings.append(
                "Prediction threshold does not apply "
                "to GrabCut."
            )

        if (
            proposed.include_disagreement
            and not proposed.can_compare
        ):
            warnings.append(
                "Model disagreement analysis requires "
                "at least two segmentation models."
            )

        if proposed.threshold != 0.5:

            warnings.append(
                "Published benchmark results use a "
                "threshold of 0.5. Changing the threshold "
                "does not guarantee improved precision "
                "or accuracy."
            )

        if proposed.mc_passes > 1:

            warnings.append(
                "Multiple inference passes may increase "
                "execution time. Stochastic uncertainty "
                "estimation requires supported dropout layers."
            )

        return {
            "current_configuration": (
                session.config.to_dict()
            ),
            "proposed_configuration": (
                proposed.to_dict()
            ),
            "configuration_changed": (
                proposed != session.config
            ),
            "revision": session.revision,
            "warnings": warnings,
        }



class RunSegmentationTool(BaseTool):
    """Execute segmentation with bounded model caching."""

    name = "run_segmentation"

    description = (
        "Run the selected segmentation models on the "
        "currently uploaded image using approved settings. "
        "Requires explicit user confirmation."
    )

    requires_confirmation = True

    def __init__(
        self,
        project_root=None,
        device=None,
    ):

        self._project_root = project_root
        self._device = device

        self._cached_service = None
        self._cache_key = None

    def clear_cache(self):
        """Release references to the cached segmentation service."""

        self._cached_service = None
        self._cache_key = None

    def _get_service(self, config):
        """Reuse a compatible service or construct a replacement."""

        key = (
            config.neural_models,
            config.threshold if config.neural_models else None,
            config.uses_grabcut,
        )

        if (
            self._cached_service is not None
            and self._cache_key == key
        ):

            return self._cached_service

        # Release the previous service before loading another.
        # This prevents accumulating multiple model collections.

        self.clear_cache()

        service = SegmentationService.from_architectures(
            architectures=config.neural_models,
            device=self._device,
            threshold=config.threshold,
            include_grabcut=config.uses_grabcut,
            project_root=self._project_root,
        )

        self._cached_service = service
        self._cache_key = key

        return service

    def execute(self, session, arguments):

        self.require_empty_arguments(arguments)

        if not session.has_image:

            raise RuntimeError(
                "Upload an image before running segmentation."
            )

        revision, image, config = (
            session.get_analysis_inputs()
        )

        service = self._get_service(config)

        try:

            results = service.compare(
                image,
                alpha=config.overlay_alpha,
                ground_truth=session.get_ground_truth(),
                mc_passes=config.mc_passes,
            )

            session.store_results(
                results,
                revision=revision,
            )

        except Exception:

            # Discard potentially inconsistent model state
            # after a failed segmentation operation.

            self.clear_cache()

            raise

        return session.get_context()
