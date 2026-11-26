"""Request-size and response-cache protection for the local API."""

from tempfile import SpooledTemporaryFile

from starlette.responses import JSONResponse


MIB = 1024 * 1024

MAX_UPLOAD_REQUEST_BYTES = 41 * MIB
MAX_JSON_REQUEST_BYTES = 64 * 1024


class APIProtectionMiddleware:
    """Enforce bounded request bodies and prevent API response caching."""

    def __init__(
        self,
        app,
        *,
        upload_limit=MAX_UPLOAD_REQUEST_BYTES,
        json_limit=MAX_JSON_REQUEST_BYTES,
    ):

        self.app = app
        self.upload_limit = upload_limit
        self.json_limit = json_limit

    async def __call__(self, scope, receive, send):

        if scope["type"] != "http":

            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")

        if not (
            path == "/api"
            or path.startswith("/api/")
        ):

            await self.app(scope, receive, send)
            return

        async def protected_send(message):

            if message["type"] == "http.response.start":

                headers = [
                    (key, value)
                    for key, value in message.get(
                        "headers", []
                    )
                    if key.lower() not in {
                        b"cache-control",
                        b"pragma",
                        b"expires",
                    }
                ]

                headers.extend([
                    (
                        b"cache-control",
                        b"no-store, max-age=0",
                    ),
                    (b"pragma", b"no-cache"),
                    (b"expires", b"0"),
                ])

                message = {
                    **message,
                    "headers": headers,
                }

            await send(message)

        async def reject(status, detail):

            response = JSONResponse(
                status_code=status,
                content={"detail": detail},
            )

            await response(
                scope,
                receive,
                protected_send,
            )

        if scope.get("method") not in {
            "POST", "PUT", "PATCH", "DELETE"
        }:

            await self.app(
                scope,
                receive,
                protected_send,
            )

            return

        upload_paths = {
            "/api/segment",
            "/api/v2/image",
        }

        limit = (
            self.upload_limit
            if path.rstrip("/") in upload_paths
            else self.json_limit
        )

        lengths = [
            value
            for key, value in scope.get("headers", [])
            if key.lower() == b"content-length"
        ]

        if lengths:

            try:

                if (
                    len(lengths) != 1
                    or not lengths[0].isdigit()
                ):
                    raise ValueError

                declared_length = int(lengths[0])

            except ValueError:

                await reject(
                    400,
                    "Invalid Content-Length header.",
                )

                return

            if declared_length > limit:

                await reject(
                    413,
                    "Request body exceeds the permitted size.",
                )

                return

        # Validate actual received bytes, including requests
        # without a Content-Length header.
        #
        # Larger bodies spill to temporary disk storage rather
        # than remaining entirely in application memory.

        with SpooledTemporaryFile(
            max_size=MIB,
            mode="w+b",
        ) as body:

            received = 0

            while True:

                message = await receive()

                if message["type"] == "http.disconnect":
                    return

                if message["type"] != "http.request":
                    continue

                chunk = message.get("body", b"")

                received += len(chunk)

                if received > limit:

                    await reject(
                        413,
                        "Request body exceeds the permitted size.",
                    )

                    return

                body.write(chunk)

                if not message.get("more_body", False):
                    break

            body.seek(0)

            remaining = received
            finished = False

            async def replay_receive():

                nonlocal remaining, finished

                if finished:
                    return await receive()

                chunk = body.read(
                    min(65536, remaining)
                )

                remaining -= len(chunk)

                finished = remaining == 0

                return {
                    "type": "http.request",
                    "body": chunk,
                    "more_body": not finished,
                }

            await self.app(
                scope,
                replay_receive,
                protected_send,
            )
