
"""Abstract interface for assistant tools."""

from abc import ABC, abstractmethod
from copy import deepcopy


class BaseTool(ABC):
    """Base class for all executable assistant tools."""

    name = ""
    description = ""

    requires_confirmation = False

    parameters = {
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False,
    }

    def get_schema(self):
        """Return an LLM-compatible tool definition."""

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": deepcopy(self.parameters),
            },
        }

    @abstractmethod
    def execute(self, session, arguments):
        """Execute the tool against the current session."""

        raise NotImplementedError

    @staticmethod
    def require_empty_arguments(arguments):

        if arguments:
            raise ValueError(
                "This tool does not accept arguments."
            )
