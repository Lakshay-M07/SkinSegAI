"""Bounded, isolated assistant sessions for the local FastAPI server."""

from __future__ import annotations

import math
import secrets
import time

from contextlib import contextmanager
from dataclasses import dataclass
from threading import Lock

from src.assistant.core.agent import SkinSegAgent


class SessionNotFoundError(KeyError):
    """The requested session does not exist or has expired."""


class SessionCapacityError(RuntimeError):
    """The maximum number of sessions has been reached."""


@dataclass(slots=True)
class SessionRecord:

    agent: SkinSegAgent
    created_at: float
    last_used_at: float


class SessionManager:
    """Own assistant sessions and coordinate local compute resources."""

    def __init__(
        self,
        *,
        max_sessions=2,
        ttl_seconds=1800,
        agent_factory=SkinSegAgent,
        clock=time.monotonic,
    ):

        if type(max_sessions) is not int or max_sessions < 1:
            raise ValueError(
                "max_sessions must be a positive integer."
            )

        if (
            type(ttl_seconds) not in (int, float)
            or not math.isfinite(ttl_seconds)
            or ttl_seconds <= 0
        ):
            raise ValueError(
                "ttl_seconds must be positive and finite."
            )

        if not callable(agent_factory):
            raise TypeError("agent_factory must be callable.")

        if not callable(clock):
            raise TypeError("clock must be callable.")

        self._max_sessions = max_sessions
        self._ttl_seconds = ttl_seconds
        self._agent_factory = agent_factory
        self._clock = clock

        self._sessions = {}
        self._active_session_id = None
        self._closed = False

        # Serialize operations to prevent competing model inference
        # and excessive memory usage on local hardware.
        self._lock = Lock()

    def _ensure_open(self):

        if self._closed:
            raise RuntimeError("Session manager is closed.")

    def _remove_locked(self, session_id):

        record = self._sessions.pop(session_id)

        if self._active_session_id == session_id:
            self._active_session_id = None

        record.agent.reset()

    def _cleanup_expired_locked(self):

        now = self._clock()

        expired = [
            session_id
            for session_id, record in self._sessions.items()
            if now - record.last_used_at >= self._ttl_seconds
        ]

        for session_id in expired:
            self._remove_locked(session_id)

        return len(expired)

    def _require_session_locked(self, session_id):

        if (
            not isinstance(session_id, str)
            or session_id not in self._sessions
        ):
            raise SessionNotFoundError(
                "Session does not exist or has expired."
            )

        return self._sessions[session_id]

    def create(self):

        with self._lock:

            self._ensure_open()
            self._cleanup_expired_locked()

            if len(self._sessions) >= self._max_sessions:
                raise SessionCapacityError(
                    "Maximum active session capacity reached."
                )

            session_id = secrets.token_urlsafe(32)

            while session_id in self._sessions:
                session_id = secrets.token_urlsafe(32)

            agent = self._agent_factory()
            now = self._clock()

            self._sessions[session_id] = SessionRecord(
                agent=agent,
                created_at=now,
                last_used_at=now,
            )

            return session_id

    @contextmanager
    def lease(self, session_id):
        """Provide exclusive access to a session during an operation."""

        with self._lock:

            self._ensure_open()
            self._cleanup_expired_locked()

            record = self._require_session_locked(
                session_id
            )

            previous_id = self._active_session_id

            # Preserve results and memory but release models
            # when switching between browser sessions.
            if (
                previous_id is not None
                and previous_id != session_id
            ):

                previous = self._sessions.get(previous_id)

                if previous is not None:
                    previous.agent.release_models()

            self._active_session_id = session_id

            try:
                yield record.agent

            finally:
                record.last_used_at = self._clock()

    def delete(self, session_id):

        with self._lock:

            self._ensure_open()
            self._cleanup_expired_locked()

            self._require_session_locked(session_id)
            self._remove_locked(session_id)

    def cleanup_expired(self):

        with self._lock:

            self._ensure_open()

            return self._cleanup_expired_locked()

    def get_statistics(self):

        with self._lock:

            self._ensure_open()
            self._cleanup_expired_locked()

            return {
                "active_sessions": len(self._sessions),
                "max_sessions": self._max_sessions,
                "ttl_seconds": self._ttl_seconds,
                "available_slots": (
                    self._max_sessions - len(self._sessions)
                ),
            }

    def close(self):
        """Release all session resources during server shutdown."""

        with self._lock:

            if self._closed:
                return

            for session_id in list(self._sessions):
                self._remove_locked(session_id)

            self._closed = True
