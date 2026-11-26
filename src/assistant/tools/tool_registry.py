
"""Controlled registry for assistant tool execution."""

from src.assistant.tools.base_tool import BaseTool
from src.assistant.core.session_context import SessionContext


class ToolRegistry:
    """Register and execute approved assistant tools."""

    def __init__(self, session):

        if not isinstance(session, SessionContext):
            raise TypeError("Invalid session context.")

        self._session = session
        self._tools = {}

    def register(self, tool):

        if not isinstance(tool, BaseTool):
            raise TypeError("Expected a BaseTool instance.")

        if not isinstance(tool.name, str) or not tool.name:
            raise ValueError("Tool must have a valid name.")

        if tool.name in self._tools:
            raise ValueError(
                f"Duplicate tool: {tool.name}"
            )

        self._tools[tool.name] = tool

    @property
    def available_tools(self):

        return list(self._tools)

    def get_schemas(self):
        """Return schemas for the registered tools."""

        return [
            tool.get_schema()
            for tool in self._tools.values()
        ]

    def _resolve(self, name, arguments):

        if name not in self._tools:
            raise ValueError(
                f"Unknown tool: {name}"
            )

        if arguments is None:
            arguments = {}

        if not isinstance(arguments, dict):
            raise TypeError(
                "Tool arguments must be a dictionary."
            )

        return self._tools[name], arguments

    def execute(self, name, arguments=None):
        """Execute an ordinary tool or request confirmation."""

        tool, arguments = self._resolve(
            name,
            arguments,
        )

        if tool.requires_confirmation:

            return {
                "status": "confirmation_required",
                "tool": name,
                "revision": self._session.revision,
            }

        return {
            "status": "success",
            "tool": name,
            "result": tool.execute(
                self._session,
                arguments,
            ),
        }

    def execute_confirmed(
        self,
        name,
        arguments=None,
        *,
        expected_revision,
    ):
        """Execute a protected tool after user confirmation.

        This method must only be called by trusted application
        code after receiving explicit user approval.
        """

        tool, arguments = self._resolve(
            name,
            arguments,
        )

        if not tool.requires_confirmation:
            raise ValueError(
                "This tool does not require confirmation."
            )

        if expected_revision != self._session.revision:
            raise ValueError(
                "Configuration or image has changed. "
                "Request confirmation again."
            )

        return {
            "status": "success",
            "tool": name,
            "result": tool.execute(
                self._session,
                arguments,
            ),
        }
