
"""Validate assistant provider responses and execution limits."""

import json
from dataclasses import dataclass, field

from src.assistant.providers.base_provider import (
    ProviderResponse,
)

from src.assistant.safety.input_validator import (
    InputValidator,
)


class ResponseValidator:
    """Validate provider responses before processing them."""

    MAX_RESPONSE_CHARACTERS = 4000
    MAX_TOOL_CALLS_PER_RESPONSE = 3

    def __init__(self):

        self._input_validator = InputValidator()

    def validate(self, response, *, schemas=()):

        if not isinstance(response, ProviderResponse):
            raise TypeError(
                "Expected a ProviderResponse instance."
            )

        content = response.content.strip()

        if len(content) > self.MAX_RESPONSE_CHARACTERS:
            raise ValueError(
                "Provider response exceeds size limit."
            )

        if "\x00" in content:
            raise ValueError(
                "Provider response contains invalid characters."
            )

        if (
            not content
            and not response.has_tool_calls
        ):
            raise ValueError("Provider returned an empty response.")

        if len(response.tool_calls) > (
            self.MAX_TOOL_CALLS_PER_RESPONSE
        ):
            raise ValueError(
                "Provider requested too many tools."
            )

        validated_calls = []
        fingerprints = set()

        for call in response.tool_calls:

            validated = self._input_validator.validate_tool_call(
                call,
                schemas,
            )

            fingerprint = json.dumps(
                validated.to_dict(),
                sort_keys=True,
                allow_nan=False,
            )

            if fingerprint in fingerprints:
                raise ValueError(
                    "Duplicate tool calls are not permitted."
                )

            fingerprints.add(fingerprint)

            validated_calls.append(validated)

        return ProviderResponse(
            content=content,
            tool_calls=tuple(validated_calls),
        )


@dataclass(slots=True)
class ExecutionBudget:
    """Limit provider and tool executions within one user turn."""

    max_provider_requests: int = 4

    max_tool_executions: int = 6

    _provider_requests: int = field(
        default=0,
        init=False,
    )

    _tool_executions: int = field(
        default=0,
        init=False,
    )

    def __post_init__(self):

        for value in (
            self.max_provider_requests,
            self.max_tool_executions,
        ):

            if type(value) is not int or not 1 <= value <= 20:

                raise ValueError(
                    "Execution limits must be integers "
                    "between 1 and 20."
                )

    def consume_provider_request(self):

        if self._provider_requests >= self.max_provider_requests:

            raise RuntimeError(
                "Provider request budget exhausted."
            )

        self._provider_requests += 1

    def consume_tool_execution(self):

        if self._tool_executions >= self.max_tool_executions:

            raise RuntimeError(
                "Tool execution budget exhausted."
            )

        self._tool_executions += 1

    def get_statistics(self):

        return {
            "provider_requests": self._provider_requests,
            "tool_executions": self._tool_executions,

            "remaining_provider_requests": (
                self.max_provider_requests
                - self._provider_requests
            ),

            "remaining_tool_executions": (
                self.max_tool_executions
                - self._tool_executions
            ),
        }
