
"""Central orchestration for SkinSeg Assistant."""

import json
import re

from src.assistant.providers.base_provider import (
    ProviderResponse,
    ToolCall as RoutedToolCall,
)

from src.assistant.analysis.response_composer import (
    GroundedResponseComposer,
)

from threading import RLock

from src.assistant.tools.report_tools import (
    AssistantReportExporter,
)

from src.assistant.core.configuration import (
    SegmentationConfig,
)

from src.assistant.core.session_context import (
    SessionContext,
)

from src.assistant.core.conversation import (
    ConversationWorkflow,
)

from src.assistant.core.memory import (
    ConversationMemory,
)

from src.assistant.providers.base_provider import (
    BaseProvider,
    ProviderError,
)

from src.assistant.providers.offline_provider import (
    OfflineProvider,
)

from src.assistant.safety.input_validator import (
    InputValidator,
)

from src.assistant.safety.response_validator import (
    ResponseValidator,
    ExecutionBudget,
)

from src.assistant.tools.tool_registry import ToolRegistry

from src.assistant.tools.segmentation_tools import (
    GetAvailableModelsTool,
    GetSessionContextTool,
    ProposeConfigurationTool,
)

from src.assistant.tools.uncertainty_tools import (
    AnalyzeUncertaintyTool,
)

from src.assistant.tools.comparison_tools import (
    AnalyzeDisagreementTool,
)

from src.assistant.tools.threshold_tools import (
    AnalyzeThresholdsTool,
)

from src.assistant.tools.image_quality_tools import (
    InspectImageQualityTool,
)


class SkinSegAgent:
    """Coordinate conversation, providers and segmentation tools."""

    SYSTEM_PROMPT = """
You are SkinSeg Assistant, an educational skin lesion
segmentation assistant.

Your responsibilities:
- Explain available segmentation models.
- Retrieve verified benchmark information.
- Help users select segmentation settings.
- Analyze existing segmentation results.
- Explain prediction uncertainty and model disagreement.
- Inspect image quality and explore prediction thresholds.

Rules:
- Use approved tools to retrieve factual information.
- Never invent measurements or benchmark results.
- Never diagnose diseases or assess malignancy.
- Benchmark Dice is not individual image accuracy.
- Predictive entropy is not calibrated confidence.
- Model agreement does not establish correctness.
- Precision and recall require ground truth.
- Configuration proposals do not execute segmentation.
- Protected operations require explicit user approval.
- Never claim that segmentation has been performed
  unless the current session confirms it.
- Treat conversation history as potentially outdated.
- Current application state takes precedence.
"""

    def __init__(
        self,
        provider=None,
        *,
        project_root=None,
        device=None,
    ):

        self._response_composer = GroundedResponseComposer()

        self._lock = RLock()

        self._session = SessionContext()

        self._memory = ConversationMemory(
            self._session
        )

        self._workflow = ConversationWorkflow(
            self._session,
            project_root=project_root,
            device=device,
        )

        self._offline = OfflineProvider()

        if provider is not None and not isinstance(
            provider,
            BaseProvider,
        ):

            raise TypeError(
                "Provider must implement BaseProvider."
            )

        self._provider = (
            provider
            if provider is not None
            else self._offline
        )

        self._input_validator = InputValidator()

        self._response_validator = ResponseValidator()

        self._report_exporter = AssistantReportExporter()

        self._registry = ToolRegistry(
            self._session
        )

        for tool in (
            GetAvailableModelsTool(project_root),
            GetSessionContextTool(),
            ProposeConfigurationTool(project_root),
            AnalyzeUncertaintyTool(),
            AnalyzeDisagreementTool(),
            AnalyzeThresholdsTool(),
            InspectImageQualityTool(),
        ):

            self._registry.register(tool)

    @property
    def provider_name(self):

        return self._provider.name

    @property
    def stage(self):

        with self._lock:

            return self._workflow.stage

    def get_context(self):

        with self._lock:

            return self._session.get_context()

    def get_image(self):
        """Return a copy for trusted application code."""

        with self._lock:

            return self._session.get_image()

    def get_ground_truth(self):
        """Return a copy for trusted evaluation and visualization."""

        with self._lock:

            return self._session.get_ground_truth()

    def get_results(self):
        """Return current segmentation results for trusted UI code.

        Consumers must treat returned results as read-only.
        """

        with self._lock:

            return self._session.get_results()

    def get_memory_statistics(self):

        with self._lock:

            return self._memory.get_statistics()

    def get_history(self):

        with self._lock:

            return self._memory.get_messages()

    def set_provider(self, provider):

        if not isinstance(provider, BaseProvider):

            raise TypeError(
                "Invalid assistant provider."
            )

        with self._lock:

            self._provider = provider

    def upload_image(self, image, *, ground_truth=None):
        """Register an image and initiate the guided conversation."""

        with self._lock:

            self._session.set_image(
                image,
                ground_truth=ground_truth,
            )

            self._memory.clear()

            response = self._workflow.begin()

            quality = self._registry.execute(
                "inspect_image_quality"
            )["result"]

            return {
                **response,
                "image_quality": quality,
                "context": self._session.get_context(),
            }

    def begin(self):

        with self._lock:

            return self._workflow.begin()

    def choose_objective(self, objective):

        with self._lock:

            return self._workflow.choose_objective(
                objective
            )

    def choose_models(self, models=None):

        with self._lock:

            return self._workflow.choose_models(
                models
            )

    def choose_threshold(self, threshold):

        with self._lock:

            return self._workflow.choose_threshold(
                threshold
            )

    def propose_configuration(self, configuration):
        """Prepare manual settings through the protected workflow."""

        with self._lock:
            return self._workflow.propose_configuration(
                configuration
            )

    def confirm(self, token, *, approved):
        """Trusted application entry point for explicit approval."""

        with self._lock:

            return self._workflow.confirm(
                token,
                approved=approved,
            )

    def get_analysis(self):
        """Return verified structured analysis for trusted UI code."""

        with self._lock:
            return self._report_exporter.build_report(
                self._session
            )

    def export_report(self):
        """Export current results through trusted application code."""
        with self._lock:
            return self._report_exporter.export_zip(
                self._session
            )

    def release_models(self):
        """Release cached models without deleting results."""

        with self._lock:
            self._workflow.release_models()

    def reset(self):
        """Clear session state and release cached models."""

        with self._lock:
            self._workflow.release_models()
            self._session.reset()
            self._memory.clear()
            return self._workflow.begin()

    def _build_messages(self, message):

        context = self._session.get_context()

        # Only necessary application state is included.
        # Detailed numerical information is retrieved by tools.

        state = {
            "image_uploaded": context["image_uploaded"],
            "analysis_completed": context["analysis_completed"],
            "image": context["image"],
            "configuration": context["configuration"],
            "conversation_stage": self._workflow.stage,
        }

        messages = [
            {
                "role": "system",
                "content": (
                    self.SYSTEM_PROMPT
                    + "\nCurrent application state:\n"
                    + json.dumps(
                        state,
                        allow_nan=False,
                    )
                ),
            }
        ]

        messages.extend(
            self._memory.get_messages()
        )

        messages.append({
            "role": "user",
            "content": message,
        })

        return messages

    def _request_provider(
        self,
        provider,
        messages,
        schemas,
        budget,
    ):

        budget.consume_provider_request()

        response = provider.chat(
            messages,
            tools=schemas,
        )

        return self._response_validator.validate(
            response,
            schemas=schemas,
        )


    def _format_tool_result(
        self,
        name,
        result,
    ):
        """Render verified results without consuming LLM budget."""

        messages = [{
            "role": "tool",
            "tool_name": name,
            "content": json.dumps(
                result,
                allow_nan=False,
            ),
        }]

        response = self._offline.chat(
            messages,
            tools=(),
        )

        response = self._response_validator.validate(
            response,
            schemas=(),
        )

        if response.has_tool_calls:
            raise RuntimeError(
                "Unexpected tool request during formatting."
            )

        return response.content


    def _prepare_proposal(self, call):

        result = self._registry.execute(
            call.name,
            call.arguments,
        )["result"]

        if not self._session.has_image:

            return (
                "Please upload an image before "
                "configuring segmentation.",
                None,
            )

        configuration = SegmentationConfig.from_dict(
            result["proposed_configuration"]
        )

        proposal = self._workflow.propose_configuration(
            configuration
        )

        names = {
            "custom_unet": "Custom U-Net",
            "resnet34_unet": "ResNet-34 U-Net",
            "grabcut": "GrabCut",
        }

        selected = ", ".join(
            names[model]
            for model in configuration.models
        )

        lines = [
            "I have prepared the following configuration:",
            f"- Models: {selected}",
        ]

        if configuration.neural_models:

            lines.append(
                f"- Prediction threshold: "
                f"{configuration.threshold:.2f}"
            )

        else:

            lines.append(
                "- Prediction threshold: Not applicable"
            )

        lines.extend([
            f"- Overlay opacity: "
            f"{configuration.overlay_alpha:.0%}",

            f"- Inference passes: "
            f"{configuration.mc_passes}",
        ])

        lines.extend([
            "- Uncertainty analysis: "
            + (
                "On"
                if configuration.include_uncertainty
                else "Off"
            ),

            "- Morphological analysis: "
            + (
                "On"
                if configuration.include_morphology
                else "Off"
            ),

            "- Model disagreement analysis: "
            + (
                "On"
                if configuration.include_disagreement
                else "Off"
            ),

            "- Image quality assessment: "
            + (
                "On"
                if configuration.include_image_quality
                else "Off"
            ),
        ])

        lines.extend(
            proposal.get("warnings", [])
        )

        lines.append(
            "\nNo segmentation has been executed for "
            "this proposal. Would you like to proceed?"
        )

        return "\n".join(lines), proposal

    @staticmethod
    def _execution_intent(message):
        """Recognize explicit segmentation execution requests."""

        text = message.casefold().strip()
        tokens = re.findall("[a-z0-9]+", text)

        if not tokens:
            return False, None

        words = set(tokens)

        if words.intersection({
            "not", "never", "cancel", "stop", "avoid"
        }) or "don't" in text:
            return False, None

        remaining = tokens.copy()

        while remaining:

            if remaining[0] == "please":
                remaining = remaining[1:]

            elif (
                len(remaining) >= 2
                and remaining[0] in {
                    "can", "could", "would", "will"
                }
                and remaining[1] == "you"
            ):
                remaining = remaining[2:]

            else:
                break

        if not remaining:
            return False, None

        if remaining[0] not in {
            "run", "perform", "execute",
            "start", "do", "segment"
        }:
            return False, None

        if not words.intersection({
            "segment", "segmentation"
        }):
            return False, None

        simple_words = {
            "please", "can", "could", "would", "will",
            "you", "run", "perform", "execute", "start",
            "do", "segment", "segmentation", "on", "the",
            "current", "skin", "lesion", "image", "and",
            "compare", "all", "3", "three", "models",
            "algorithms", "after", "using", "with",
            "then", "now", "this", "my", "uploaded",
            "selected", "results", "of", "for", "me",
            "them",
        }

        if not words.issubset(simple_words):
            return True, None

        normalized = " " + " ".join(tokens) + " "

        all_models = (
            " all models ",
            " all 3 models ",
            " all three models ",
            " all algorithms ",
            " all 3 algorithms ",
            " all three algorithms ",
        )

        if any(
            phrase in normalized
            for phrase in all_models
        ):
            return True, {
                "models": [
                    "custom_unet",
                    "resnet34_unet",
                    "grabcut",
                ]
            }

        if words.intersection({
            "models", "algorithms", "compare"
        }):
            return True, None

        return True, {}


    def ask(self, message):
        """Process a user message using validated assistant tools."""

        message = self._input_validator.validate_message(
            message
        )

        with self._lock:
            revision = self._session.revision

            pending = (
                self._workflow.get_pending_configuration()
            )

            messages = self._build_messages(message)

            if pending is not None:

                pending_context = {
                    "status": "awaiting_confirmation",
                    "configuration": pending.to_dict(),
                    "applied": False,
                    "executed": False,
                }

                messages[0]["content"] += (
                    "\n\nCURRENT PENDING PROPOSAL:\n"
                    + json.dumps(
                        pending_context,
                        allow_nan=False,
                    )
                    + "\n"
                    "This proposal has not been applied. "
                    "It is distinct from the active settings.\n"
                    "When the user asks to change, keep, adjust "
                    "or refine proposed settings, call "
                    "propose_configuration exactly once.\n"
                    "Supply only the fields the user explicitly "
                    "wants to change. All other pending fields "
                    "will be preserved automatically.\n"
                    "Do not return a text-only acknowledgement "
                    "for a valid configuration change.\n"
                    "Never claim that segmentation has already "
                    "been executed. Explicit approval is required."
                )


            if pending is None:

                messages[0]["content"] += (
                    "\n\nANALYTICAL TOOL ROUTING:\n"
                    "You select tools; Python produces verified "
                    "answers and explanations.\n"
                    "Do not answer supported analytical "
                    "questions directly from your own knowledge.\n"
                    "For model recommendations, benchmark scores, "
                    "speed, computational requirements or model "
                    "architecture comparisons, call "
                    "get_available_models.\n"
                    "For uploaded image information, lesion area "
                    "or completed segmentation results, call "
                    "get_session_context.\n"
                    "For predictive entropy, boundary uncertainty "
                    "or whether uncertainty establishes accuracy, "
                    "call analyze_uncertainty.\n"
                    "For agreement between segmentation masks, "
                    "call analyze_disagreement.\n"
                    "For threshold effects and sensitivity "
                    "analysis, call analyze_thresholds.\n"
                    "For image resolution, illumination, blur "
                    "or technical quality, call "
                    "inspect_image_quality.\n"
                    "Use propose_configuration only when the "
                    "user requests preparation or modification "
                    "of segmentation settings.\n"
                    "Explanatory questions are not permission "
                    "to change settings or execute segmentation."
                )

            routing_messages = messages

            if pending is not None:

                routing_messages = [
                    {
                        "role": "system",
                        "content": (
                            messages[0]["content"]
                            + "\n\nCONFIGURATION ROUTING:\n"
                            "You are a tool-selection engine.\n"
                            "Do not generate configuration "
                            "summaries yourself.\n"
                            "For configuration modifications, "
                            "call propose_configuration exactly "
                            "once with structured arguments.\n"
                            "When modifying model selection, "
                            "provide the complete desired model "
                            "list, including retained models.\n"
                            "For other settings, provide only "
                            "explicitly requested changes.\n"
                            "The pending configuration is the "
                            "authoritative proposal context.\n"
                            "Python will validate the settings "
                            "and generate the final response."
                        ),
                    },
                    {
                        "role": "user",
                        "content": message,
                    },
                ]

            schemas = (
                self._input_validator.filter_tool_schemas(
                    self._registry.get_schemas()
                )
            )

            execution_requested, direct_arguments = (
                self._execution_intent(message)
            )

            if execution_requested:

                # Execution requests must prepare a proposal,
                # not retrieve unrelated benchmark information.
                schemas = [
                    schema for schema in schemas
                    if schema["function"]["name"]
                    == "propose_configuration"
                ]

                routing_messages[0]["content"] += (
                    "\\n\\nEXECUTION INTENT PRIORITY:\\n"
                    "The user explicitly requests segmentation. "
                    "Prepare the requested configuration using "
                    "propose_configuration. Preserve settings "
                    "the user did not request to change. "
                    "Comparison after segmentation is not a "
                    "request for historical benchmarks. "
                    "Never execute without explicit approval."
                )

            budget = ExecutionBudget()

            provider = self._provider

            used_fallback = False

            proposal = None

            try:

                if direct_arguments is not None:

                    response = self._response_validator.validate(
                        ProviderResponse(
                            tool_calls=(
                                RoutedToolCall(
                                    "propose_configuration",
                                    direct_arguments,
                                ),
                            ),
                        ),
                        schemas=schemas,
                    )

                else:

                    response = self._request_provider(
                        provider,
                        routing_messages,
                        schemas,
                        budget,
                    )

                # Until unrestricted LLM prose has a stronger
                # grounding mechanism, use deterministic replies
                # whenever the primary provider requests no tools.


                if (
                    not isinstance(provider, OfflineProvider)
                    and not response.has_tool_calls
                    and pending is not None
                ):

                    # Retry using compact, structured context.
                    # Exclude previous assistant replies to reduce
                    # imitation of textual configuration summaries.

                    retry_messages = [
                        {
                            "role": "system",
                            "content": (
                                messages[0]["content"]
                                + "\n\nTOOL ROUTING RETRY:\n"
                                "You are a tool-selection engine.\n"
                                "Do not imitate previous assistant "
                                "responses or generate a configuration "
                                "summary yourself.\n"
                                "For a configuration modification, "
                                "call propose_configuration exactly "
                                "once with structured arguments.\n"
                                "When modifying the model selection, "
                                "the models argument must contain "
                                "the complete desired model list, "
                                "including models being retained.\n"
                                "For other settings, supply only "
                                "explicitly requested changes.\n"
                                "Use the pending configuration as "
                                "the authoritative proposal context.\n"
                                "Never invent values or execute "
                                "segmentation.\n"
                                "Python will generate the final "
                                "configuration summary."
                            ),
                        },
                        {
                            "role": "user",
                            "content": message,
                        },
                    ]

                    response = self._request_provider(
                        provider,
                        retry_messages,
                        schemas,
                        budget,
                    )

                if (
                    not isinstance(provider, OfflineProvider)
                    and not response.has_tool_calls
                    and pending is None
                ):

                    retry_messages = [
                        {
                            "role": "system",
                            "content": (
                                messages[0]["content"]
                                + "\n\nGENERAL TOOL ROUTING RETRY:\n"
                                "Select the appropriate available "
                                "tool for the user's request.\n"
                                "Do not generate a textual "
                                "explanation when a tool can "
                                "retrieve the required evidence.\n"
                                "Ignore the style of previous "
                                "assistant responses.\n"
                                "Use verified project data through "
                                "the provided tools.\n"
                                "Do not invent numerical values, "
                                "medical conclusions or results.\n"
                                "Never execute segmentation "
                                "without explicit approval.\n"
                                "If the request is ambiguous, "
                                "do not invent missing details."
                            ),
                        },
                        {
                            "role": "user",
                            "content": message,
                        },
                    ]

                    response = self._request_provider(
                        provider,
                        retry_messages,
                        schemas,
                        budget,
                    )

                # Fall back only if structured tool calling
                # remains unavailable after the retry.

                if (
                    not isinstance(provider, OfflineProvider)
                    and not response.has_tool_calls
                ):

                    provider = self._offline

                    used_fallback = True

                    response = self._request_provider(
                        provider,
                        messages,
                        schemas,
                        budget,
                    )


            except (
                ProviderError,
                ValueError,
                TypeError,
                PermissionError,
                OverflowError,
            ):

                provider = self._offline

                used_fallback = True

                response = self._request_provider(
                    provider,
                    messages,
                    schemas,
                    budget,
                )

            calls = response.tool_calls

            if not calls:

                if execution_requested:

                    if not self._session.has_image:
                        answer = (
                            "Please upload or select an image "
                            "before running segmentation."
                        )
                    else:
                        answer = (
                            "I could not prepare the requested "
                            "configuration. Please specify the "
                            "models and settings, or use the "
                            "Prepare segmentation button. "
                            "No segmentation has been executed."
                        )

                else:
                    answer = response.content

            elif any(
                call.name == "propose_configuration"
                for call in calls
            ):

                # Keep configuration proposals separate from
                # other tool operations.

                if len(calls) != 1:

                    answer = (
                        "Please request configuration changes "
                        "separately from other analyses."
                    )

                else:

                    budget.consume_tool_execution()

                    try:

                        call = calls[0]

                        pending = (
                            self._workflow
                            .get_pending_configuration()
                        )

                        if pending is not None:

                            merged = pending.to_dict()

                            merged.update(
                                call.arguments
                            )

                            from src.assistant.providers.base_provider import (
                                ToolCall,
                            )

                            call = ToolCall(
                                call.name,
                                merged,
                            )

                        answer, proposal = (
                            self._prepare_proposal(
                                call
                            )
                        )


                    except (
                        RuntimeError,
                        ValueError,
                        TypeError,
                        OSError,
                    ):

                        answer = (
                            "The proposed configuration "
                            "could not be prepared. Check "
                            "your image and model settings."
                        )

            else:

                answers = []

                for call in calls:

                    budget.consume_tool_execution()

                    try:

                        result = self._registry.execute(
                            call.name,
                            call.arguments,
                        )

                    except (
                        RuntimeError,
                        ValueError,
                        TypeError,
                        OSError,
                    ):

                        result = {
                            "status": "error",
                            "tool": call.name,
                        }

                    summary = self._format_tool_result(
                        call.name,
                        result,
                    )

                    answer_text = (
                        self._response_composer.compose(
                            tool_name=call.name,
                            tool_response=result,
                            question=message,
                            summary=summary,
                        )
                    )

                    answers.append(answer_text)

                answer = "\n\n".join(answers)

            if len(answer) > (
                self._response_validator.MAX_RESPONSE_CHARACTERS
            ):

                answer = (
                    "The combined analysis is too large "
                    "for one response. Please request "
                    "each analysis individually."
                )

            self._memory.remember_turn(
                message,
                answer,
                expected_revision=revision,
            )

            return {
                "message": answer,

                "provider": provider.name,

                "used_fallback": used_fallback,

                "routing": (
                    "deterministic"
                    if direct_arguments is not None
                    else "provider"
                ),

                "stage": self._workflow.stage,

                "confirmation_required": (
                    proposal is not None
                ),

                "proposal": proposal,

                "execution_statistics": (
                    budget.get_statistics()
                ),
            }
