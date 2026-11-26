
"""Local Ollama provider for SkinSeg Assistant."""

import json
import http.client

from src.assistant.providers.base_provider import (
    BaseProvider,
    ProviderError,
    ProviderResponse,
    ToolCall,
)


class OllamaProvider(BaseProvider):
    """Connect SkinSeg Assistant to a local Ollama model."""

    name = "ollama"

    MODEL = "qwen3.5:9b-q4_K_M"

    HOST = "127.0.0.1"
    PORT = 11434

    MAX_RESPONSE_BYTES = 1_048_576

    def __init__(self, *, timeout=300):

        if type(timeout) is not int or not 1 <= timeout <= 600:
            raise ValueError("Invalid Ollama timeout.")

        self._timeout = timeout

    @property
    def model(self):

        return self.MODEL

    def _request(self, method, endpoint, payload=None):
        """Send a request exclusively to the local Ollama server."""

        connection = http.client.HTTPConnection(
            self.HOST,
            self.PORT,
            timeout=self._timeout,
        )

        try:

            body = None

            if payload is not None:

                body = json.dumps(
                    payload,
                    allow_nan=False,
                ).encode("utf-8")

            connection.request(
                method,
                endpoint,
                body=body,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                },
            )

            response = connection.getresponse()

            if response.status != 200:

                raise ProviderError(
                    f"Ollama returned HTTP {response.status}."
                )

            raw = response.read(
                self.MAX_RESPONSE_BYTES + 1
            )

            if len(raw) > self.MAX_RESPONSE_BYTES:

                raise ProviderError(
                    "Ollama response exceeds the size limit."
                )

            data = json.loads(raw)

            if not isinstance(data, dict):

                raise ProviderError(
                    "Invalid Ollama response format."
                )

            if data.get("error"):

                raise ProviderError(
                    "Ollama reported an execution error."
                )

            return data

        except (
            OSError,
            http.client.HTTPException,
            ValueError,
            TypeError,
        ) as error:

            raise ProviderError(
                "Unable to communicate with Ollama."
            ) from error

        finally:

            connection.close()

    def is_available(self):
        """Check whether the selected model is installed."""

        try:

            response = self._request(
                "GET",
                "/api/tags",
            )

            return any(
                model.get("name") == self.MODEL
                or model.get("model") == self.MODEL
                for model in response.get("models", [])
                if isinstance(model, dict)
            )

        except (ProviderError, TypeError):

            return False

    def chat(self, messages, *, tools=()):
        """Generate text or structured tool requests."""

        if not isinstance(messages, (list, tuple)):
            raise TypeError("Invalid conversation messages.")

        if not messages:
            raise ValueError("Conversation cannot be empty.")

        for message in messages:

            if not isinstance(message, dict):
                raise TypeError("Invalid message structure.")

            if message.get("role") not in (
                "system",
                "user",
                "assistant",
                "tool",
            ):
                raise ValueError("Invalid message role.")

            if not isinstance(message.get("content"), str):
                raise TypeError("Invalid message content.")

            if set(message) - {
                "role",
                "content",
                "tool_calls",
                "tool_name",
            }:
                raise ValueError(
                    "Unsupported message fields."
                )

        payload = {
            "model": self.MODEL,

            "messages": list(messages),

            "stream": False,

            "think": False,

            "keep_alive": "1m",

            "options": {
                "num_ctx": 4096,
                "num_predict": 512,
                "temperature": 0.2,
            },
        }

        if tools:

            payload["tools"] = list(tools)

        response = self._request(
            "POST",
            "/api/chat",
            payload,
        )

        return self._parse_response(response)

    @staticmethod
    def _parse_response(response):

        if response.get("done") is not True:

            raise ProviderError(
                "Ollama returned an incomplete response."
            )

        if response.get("done_reason") == "length":

            raise ProviderError(
                "Ollama reached the generation limit."
            )

        message = response.get("message")

        if not isinstance(message, dict):

            raise ProviderError(
                "Ollama response is missing its message."
            )

        if message.get("role") != "assistant":

            raise ProviderError(
                "Invalid Ollama response role."
            )

        content = message.get("content", "")

        if not isinstance(content, str):

            raise ProviderError(
                "Invalid Ollama response content."
            )

        raw_calls = message.get("tool_calls")
        if raw_calls is None:
            raw_calls = []

        if not isinstance(raw_calls, list):

            raise ProviderError(
                "Invalid Ollama tool call structure."
            )

        calls = []

        for raw_call in raw_calls:

            try:

                function = raw_call["function"]

                calls.append(
                    ToolCall(
                        name=function["name"],
                        arguments=function["arguments"],
                    )
                )

            except (KeyError, TypeError, ValueError) as error:

                raise ProviderError(
                    "Malformed Ollama tool call."
                ) from error

        if not content.strip() and not calls:

            raise ProviderError(
                "Ollama returned an empty response."
            )

        return ProviderResponse(
            content=content,
            tool_calls=tuple(calls),
        )
