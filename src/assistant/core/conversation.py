
"""Guided conversational workflow for SkinSeg Assistant."""

from enum import Enum
from uuid import uuid4

from src.assistant.core.configuration import SegmentationConfig
from src.assistant.core.session_context import SessionContext

from src.assistant.knowledge.model_knowledge import (
    ModelKnowledgeBase,
)

from src.assistant.tools.tool_registry import ToolRegistry

from src.assistant.tools.segmentation_tools import (
    ProposeConfigurationTool,
    RunSegmentationTool,
)


class ConversationStage(str, Enum):
    """Supported stages of the guided segmentation workflow."""

    WAITING_FOR_IMAGE = "waiting_for_image"
    SELECT_OBJECTIVE = "select_objective"
    SELECT_MODELS = "select_models"
    SELECT_THRESHOLD = "select_threshold"
    AWAITING_CONFIRMATION = "awaiting_confirmation"
    EXECUTING = "executing"
    RESULTS_READY = "results_ready"
    CANCELLED = "cancelled"
    FAILED = "failed"


class ConversationWorkflow:
    """Guide users through validated segmentation operations."""

    _OBJECTIVES = {
        "speed": "Quick segmentation",
        "accuracy": "Highest benchmark Dice score",
        "neural_comparison": "Compare both AI models",
        "comparison": "Compare all three algorithms",
        "traditional": "Traditional computer vision",
    }

    _THRESHOLDS = {
        "inclusive": 0.3,
        "balanced": 0.5,
        "conservative": 0.7,
    }

    def __init__(
        self,
        session,
        project_root=None,
        device=None,
    ):

        if not isinstance(session, SessionContext):
            raise TypeError("Invalid session context.")

        self._session = session

        self._knowledge = ModelKnowledgeBase(
            project_root
        )

        self._registry = ToolRegistry(session)

        self._registry.register(
            ProposeConfigurationTool(project_root)
        )

        self._segmentation_tool = RunSegmentationTool(
            project_root=project_root,
            device=device,
        )

        self._registry.register(
            self._segmentation_tool
        )

        self._stage = (
            ConversationStage.WAITING_FOR_IMAGE
        )

        self._revision = None
        self._objective = None
        self._proposed_config = None
        self._confirmation_token = None

    @property
    def stage(self):
        return self._stage.value


    def get_pending_configuration(self):
        """Return the current valid, unapproved configuration."""

        if (
            self._stage
            != ConversationStage.AWAITING_CONFIRMATION
        ):
            return None

        if self._revision != self._session.revision:
            return None

        if self._confirmation_token is None:
            return None

        return self._proposed_config


    def _require_stage(self, expected):

        if self._stage != expected:
            raise RuntimeError(
                f"Expected stage '{expected.value}', "
                f"current stage is '{self.stage}'."
            )

    def _check_revision(self):

        if self._revision != self._session.revision:

            self._confirmation_token = None
            self._proposed_config = None
            self._stage = ConversationStage.CANCELLED

            raise ValueError(
                "The image or configuration has changed. "
                "Restart the guided workflow."
            )

    def _response(self, message, **extra):

        return {
            "stage": self.stage,
            "message": message,
            **extra,
        }

    def release_models(self):
        """Release cached segmentation models."""
        if self._stage == ConversationStage.EXECUTING:
            raise RuntimeError(
                "Cannot release models during segmentation."
            )

        self._segmentation_tool.clear_cache()

    def begin(self):
        """Start a new guided analysis."""

        if self._stage == ConversationStage.EXECUTING:
            raise RuntimeError(
                "Segmentation is already running."
            )

        self._confirmation_token = None
        self._objective = None
        self._proposed_config = None

        self._revision = self._session.revision

        if not self._session.has_image:

            self._stage = (
                ConversationStage.WAITING_FOR_IMAGE
            )

            return self._response(
                "Please upload an image to begin."
            )

        self._stage = ConversationStage.SELECT_OBJECTIVE

        return self._response(
            "Your image is ready. "
            "What would you like to achieve?",
            choices=[
                {
                    "value": key,
                    "label": label,
                }
                for key, label in self._OBJECTIVES.items()
            ],
        )

    def choose_objective(self, objective):
        """Recommend models according to the user's objective."""

        self._require_stage(
            ConversationStage.SELECT_OBJECTIVE
        )

        self._check_revision()

        if objective not in self._OBJECTIVES:
            raise ValueError(
                "Unsupported segmentation objective."
            )

        recommended = self._knowledge.recommend(
            objective
        )

        if not recommended:
            raise RuntimeError(
                "No suitable models are available."
            )

        self._objective = objective

        self._proposed_config = (
            self._session.config.updated(
                models=recommended
            )
        )

        self._stage = ConversationStage.SELECT_MODELS

        return self._response(
            "I have recommended models based on your "
            "objective. You can accept or customize "
            "the selection.",
            objective=objective,
            recommended_models=recommended,
            available_models=(
                self._knowledge.get_all_model_contexts()
            ),
        )

    def choose_models(self, models=None):
        """Accept recommended models or customize selection."""

        self._require_stage(
            ConversationStage.SELECT_MODELS
        )

        self._check_revision()

        if models is None:
            models = self._proposed_config.models

        proposed = self._proposed_config.updated(
            models=models
        )

        available = set(
            self._knowledge.available_models()
        )

        unavailable = set(proposed.models) - available

        if unavailable:
            raise ValueError(
                f"Unavailable models: {unavailable}"
            )

        self._proposed_config = proposed

        # GrabCut does not support probability thresholds.
        if not proposed.neural_models:

            return self._prepare_confirmation()

        self._stage = (
            ConversationStage.SELECT_THRESHOLD
        )

        return self._response(
            "How conservative should the neural models "
            "be when selecting lesion pixels?",
            choices=[
                {
                    "value": "inclusive",
                    "threshold": 0.3,
                    "label": "Include more possible lesion pixels",
                },
                {
                    "value": "balanced",
                    "threshold": 0.5,
                    "label": "Use the verified benchmark threshold",
                },
                {
                    "value": "conservative",
                    "threshold": 0.7,
                    "label": "Require higher predicted probabilities",
                },
                {
                    "value": "custom",
                    "label": "Choose a custom threshold",
                },
            ],
            explanation=(
                "Higher thresholds generally produce "
                "smaller or equal masks for a fixed "
                "probability map. They may reduce false "
                "positives but can miss lesion pixels. "
                "No threshold guarantees a particular "
                "precision percentage."
            ),
        )

    def choose_threshold(self, selection):
        """Choose a preset or numerical probability threshold."""

        self._require_stage(
            ConversationStage.SELECT_THRESHOLD
        )

        self._check_revision()

        if isinstance(selection, str):

            if selection not in self._THRESHOLDS:
                raise ValueError(
                    "Unknown threshold preset."
                )

            threshold = self._THRESHOLDS[selection]

        else:
            threshold = selection

        self._proposed_config = (
            self._proposed_config.updated(
                threshold=threshold
            )
        )

        return self._prepare_confirmation()

    def _prepare_confirmation(self):
        """Validate and present proposed settings."""

        self._check_revision()

        proposal = self._registry.execute(
            "propose_configuration",
            self._proposed_config.to_dict(),
        )["result"]

        self._proposed_config = (
            SegmentationConfig.from_dict(
                proposal["proposed_configuration"]
            )
        )

        self._confirmation_token = uuid4().hex

        self._stage = (
            ConversationStage.AWAITING_CONFIRMATION
        )

        return self._response(
            "Your segmentation configuration is ready. "
            "Would you like me to proceed?",
            configuration=(
                self._proposed_config.to_dict()
            ),
            warnings=proposal["warnings"],
            confirmation_required=True,
            confirmation_token=self._confirmation_token,
            choices=[
                {
                    "value": "confirm",
                    "label": "Run segmentation",
                },
                {
                    "value": "cancel",
                    "label": "Cancel",
                },
            ],
        )


    def propose_configuration(self, configuration):
        """Prepare a complete configuration for user approval."""

        if self._stage == ConversationStage.EXECUTING:
            raise RuntimeError(
                "Segmentation is already running."
            )

        if not self._session.has_image:
            raise RuntimeError(
                "Upload an image before configuring segmentation."
            )

        if not isinstance(configuration, SegmentationConfig):
            raise TypeError(
                "Expected a SegmentationConfig instance."
            )

        # Validate before replacing any existing proposal.
        proposal = self._registry.execute(
            "propose_configuration",
            configuration.to_dict(),
        )["result"]

        self._proposed_config = (
            SegmentationConfig.from_dict(
                proposal["proposed_configuration"]
            )
        )

        self._revision = self._session.revision
        self._objective = None
        self._confirmation_token = None

        return self._prepare_confirmation()


    def confirm(self, token, *, approved):
        """Execute only after explicit approval from trusted UI code."""

        self._require_stage(
            ConversationStage.AWAITING_CONFIRMATION
        )

        self._check_revision()

        if type(approved) is not bool:
            raise TypeError(
                "Approval must be a boolean."
            )

        if (
            self._confirmation_token is None
            or token != self._confirmation_token
        ):
            raise ValueError(
                "Invalid or expired confirmation token."
            )

        # Consume the token before executing anything.
        self._confirmation_token = None

        if not approved:

            self._proposed_config = None
            self._stage = ConversationStage.CANCELLED

            return self._response(
                "Analysis cancelled. "
                "Your previous settings remain unchanged."
            )

        self._stage = ConversationStage.EXECUTING

        try:

            self._session.set_config(
                self._proposed_config
            )

            request = self._registry.execute(
                "run_segmentation"
            )

            response = self._registry.execute_confirmed(
                "run_segmentation",
                expected_revision=request["revision"],
            )

        except Exception:

            self._stage = ConversationStage.FAILED
            self._proposed_config = None

            raise

        self._revision = self._session.revision

        self._proposed_config = None

        self._stage = ConversationStage.RESULTS_READY

        return self._response(
            "Segmentation is complete. "
            "Would you like me to explain your results?",
            context=response["result"],
            suggested_questions=[
                "Explain my segmentation results.",
                "Which model performed better in testing?",
                "How uncertain are the predictions?",
                "What is the predicted lesion area?",
                "Can we refine the prediction threshold?",
            ],
        )
