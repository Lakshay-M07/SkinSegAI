
"""Validate user input and AI-generated tool arguments."""

import json
import math

from src.assistant.providers.base_provider import ToolCall


class InputValidator:
    """Enforce input limits and approved tool schemas."""

    MAX_MESSAGE_CHARACTERS = 4000
    MAX_ARGUMENT_CHARACTERS = 4096

    ALLOWED_TOOLS = frozenset({
        "get_available_models",
        "get_session_context",
        "propose_configuration",
        "analyze_uncertainty",
        "analyze_disagreement",
        "analyze_thresholds",
        "inspect_image_quality",
    })

    PROTECTED_TOOLS = frozenset({
        "run_segmentation",
        "apply_configuration",
        "adjust_threshold",
        "generate_report",
        "execute_confirmed",
    })

    def validate_message(self, message):

        if not isinstance(message, str):
            raise TypeError("User message must be a string.")

        message = message.strip()

        if not message:
            raise ValueError("User message cannot be empty.")

        if len(message) > self.MAX_MESSAGE_CHARACTERS:
            raise ValueError("User message exceeds size limit.")

        if "\x00" in message:
            raise ValueError("Invalid null character in message.")

        return message

    def filter_tool_schemas(self, schemas):
        """Expose only explicitly permitted tools to the provider."""

        approved = []

        for schema in schemas:

            name = schema["function"]["name"]

            if (
                name in self.ALLOWED_TOOLS
                and name not in self.PROTECTED_TOOLS
            ):

                approved.append(schema)

        return approved

    def validate_tool_call(self, call, schemas):
        """Validate a provider-generated tool call."""

        if not isinstance(call, ToolCall):
            raise TypeError("Expected a ToolCall instance.")

        if call.name in self.PROTECTED_TOOLS:
            raise PermissionError(
                "Protected operations require trusted "
                "application confirmation."
            )

        if call.name not in self.ALLOWED_TOOLS:
            raise PermissionError(
                "Tool is not approved for provider execution."
            )

        approved = {
            schema["function"]["name"]:
            schema["function"]["parameters"]
            for schema in self.filter_tool_schemas(schemas)
        }

        if call.name not in approved:
            raise ValueError(
                "Requested tool is not currently available."
            )

        try:

            encoded = json.dumps(
                call.arguments,
                allow_nan=False,
            )

        except (TypeError, ValueError, RecursionError) as error:

            raise ValueError(
                "Tool arguments must contain valid JSON data."
            ) from error

        if len(encoded) > self.MAX_ARGUMENT_CHARACTERS:
            raise ValueError(
                "Tool arguments exceed size limit."
            )

        self._validate_schema(
            call.arguments,
            approved[call.name],
            path="arguments",
        )

        # Return a fresh copy rather than trusting mutable
        # arguments owned by the provider.

        return ToolCall(
            name=call.name,
            arguments=json.loads(encoded),
        )

    def _validate_schema(self, value, schema, *, path):
        """Validate the JSON Schema subset used by our tools."""

        expected_type = schema.get("type")

        if expected_type == "object":

            if not isinstance(value, dict):
                raise TypeError(f"{path} must be an object.")

            properties = schema.get("properties", {})

            if any(
                not isinstance(key, str)
                for key in value
            ):
                raise TypeError(
                    f"{path} contains invalid keys."
                )

            required = set(
                schema.get("required", [])
            )

            missing = required - set(value)

            if missing:
                raise ValueError(
                    f"{path} is missing required fields."
                )

            unknown = set(value) - set(properties)

            if (
                unknown
                and schema.get("additionalProperties") is False
            ):
                raise ValueError(
                    f"{path} contains unsupported fields."
                )

            for key, item in value.items():

                if key in properties:

                    self._validate_schema(
                        item,
                        properties[key],
                        path=f"{path}.{key}",
                    )

        elif expected_type == "array":

            if not isinstance(value, list):
                raise TypeError(f"{path} must be an array.")

            minimum = schema.get("minItems", 0)
            maximum = schema.get("maxItems", float("inf"))

            if not minimum <= len(value) <= maximum:
                raise ValueError(
                    f"{path} contains an invalid number of items."
                )

            if schema.get("uniqueItems"):

                serialized = [
                    json.dumps(
                        item,
                        sort_keys=True,
                        allow_nan=False,
                    )
                    for item in value
                ]

                if len(serialized) != len(set(serialized)):
                    raise ValueError(
                        f"{path} contains duplicate items."
                    )

            for index, item in enumerate(value):

                self._validate_schema(
                    item,
                    schema["items"],
                    path=f"{path}[{index}]",
                )

        elif expected_type == "string":

            if not isinstance(value, str):
                raise TypeError(f"{path} must be a string.")

        elif expected_type == "boolean":

            if type(value) is not bool:
                raise TypeError(f"{path} must be a boolean.")

        elif expected_type == "integer":

            if type(value) is not int:
                raise TypeError(f"{path} must be an integer.")

        elif expected_type == "number":

            if (
                type(value) not in (int, float)
                or not math.isfinite(value)
            ):
                raise TypeError(
                    f"{path} must be a finite number."
                )

        else:

            raise ValueError(
                f"Unsupported schema type: {expected_type}"
            )

        if "enum" in schema:

            if value not in schema["enum"]:
                raise ValueError(
                    f"{path} contains an unsupported value."
                )

        if expected_type in ("number", "integer"):

            if (
                "minimum" in schema
                and value < schema["minimum"]
            ):
                raise ValueError(
                    f"{path} is below the permitted minimum."
                )

            if (
                "maximum" in schema
                and value > schema["maximum"]
            ):
                raise ValueError(
                    f"{path} exceeds the permitted maximum."
                )
