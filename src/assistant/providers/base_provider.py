
"""Common interface for SkinSeg Assistant providers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from copy import deepcopy


class ProviderError(RuntimeError):
    """An expected provider execution failure."""


@dataclass(frozen=True, slots=True)
class ToolCall:
    """A provider-generated request to invoke a backend tool."""

    name: str

    arguments: dict = field(default_factory=dict)

    def __post_init__(self):

        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Invalid tool name.")

        if not isinstance(self.arguments, dict):
            raise TypeError("Tool arguments must be a dictionary.")

        object.__setattr__(
            self,
            "arguments",
            deepcopy(self.arguments),
        )

    def to_dict(self):

        return {
            "function": {
                "name": self.name,
                "arguments": deepcopy(self.arguments),
            },
        }


@dataclass(frozen=True, slots=True)
class ProviderResponse:
    """Common response format for all assistant providers."""

    content: str = ""

    tool_calls: tuple[ToolCall, ...] = ()

    def __post_init__(self):

        if not isinstance(self.content, str):
            raise TypeError("Response content must be a string.")

        calls = tuple(self.tool_calls)

        if not all(
            isinstance(call, ToolCall)
            for call in calls
        ):
            raise TypeError("Invalid provider tool calls.")

        object.__setattr__(
            self,
            "tool_calls",
            calls,
        )

    @property
    def has_tool_calls(self):

        return bool(self.tool_calls)

    def to_message(self):
        """Return a provider-compatible assistant message."""

        message = {
            "role": "assistant",
            "content": self.content,
        }

        if self.has_tool_calls:

            message["tool_calls"] = [
                call.to_dict()
                for call in self.tool_calls
            ]

        return message


class BaseProvider(ABC):
    """Abstract interface for interchangeable AI providers."""

    name = "base"

    @abstractmethod
    def chat(self, messages, *, tools=()):
        """Generate a response or request approved tools."""

        raise NotImplementedError
