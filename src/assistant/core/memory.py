
"""Bounded, session-specific conversational memory."""

from collections import deque
from dataclasses import dataclass

from src.assistant.core.session_context import SessionContext


@dataclass(frozen=True, slots=True)
class ConversationTurn:
    """An immutable user-assistant conversation turn."""

    user: str
    assistant: str

    @property
    def character_count(self):

        return len(self.user) + len(self.assistant)

    def to_messages(self):

        return [
            {
                "role": "user",
                "content": self.user,
            },
            {
                "role": "assistant",
                "content": self.assistant,
            },
        ]


class ConversationMemory:
    """Manage bounded conversational history for one session."""

    MAX_MESSAGE_CHARACTERS = 4000

    def __init__(
        self,
        session,
        *,
        max_turns=8,
        max_characters=8000,
    ):

        if not isinstance(session, SessionContext):

            raise TypeError(
                "Expected a SessionContext instance."
            )

        if (
            type(max_turns) is not int
            or not 1 <= max_turns <= 50
        ):

            raise ValueError(
                "max_turns must be between 1 and 50."
            )

        if (
            type(max_characters) is not int
            or not 100 <= max_characters <= 50000
        ):

            raise ValueError(
                "Invalid conversation character budget."
            )

        self._session = session

        self._max_turns = max_turns

        self._max_characters = max_characters

        self._turns = deque()

        self._character_count = 0

        self._revision = session.revision

    def _synchronize(self):
        """Discard history when application context changes."""

        if self._revision != self._session.revision:

            self._turns.clear()

            self._character_count = 0

            self._revision = self._session.revision

    @staticmethod
    def _validate_message(message):

        if not isinstance(message, str):

            raise TypeError(
                "Conversation messages must be strings."
            )

        message = message.strip()

        if not message:

            raise ValueError(
                "Conversation messages cannot be empty."
            )

        if len(message) > (
            ConversationMemory.MAX_MESSAGE_CHARACTERS
        ):

            raise ValueError(
                "Conversation message exceeds the size limit."
            )

        return message

    @property
    def turn_count(self):

        self._synchronize()

        return len(self._turns)

    @property
    def character_count(self):

        self._synchronize()

        return self._character_count

    @property
    def revision(self):

        self._synchronize()

        return self._revision

    def _trim(self):

        while (
            len(self._turns) > self._max_turns
            or self._character_count > self._max_characters
        ):

            removed = self._turns.popleft()

            self._character_count -= (
                removed.character_count
            )

    def remember_turn(
        self,
        user,
        assistant,
        *,
        expected_revision,
    ):
        """Store one complete conversation turn."""

        self._synchronize()

        if expected_revision != self._revision:

            raise ValueError(
                "Cannot store conversation generated "
                "for an outdated application context."
            )

        user = self._validate_message(user)

        assistant = self._validate_message(assistant)

        turn = ConversationTurn(
            user=user,
            assistant=assistant,
        )

        if turn.character_count > self._max_characters:

            raise ValueError(
                "Conversation turn exceeds memory budget."
            )

        self._turns.append(turn)

        self._character_count += (
            turn.character_count
        )

        self._trim()

    def get_messages(self):
        """Return a fresh list of provider-compatible messages."""

        self._synchronize()

        messages = []

        for turn in self._turns:

            messages.extend(
                turn.to_messages()
            )

        return messages

    def snapshot(self):
        """Capture conversation context for an LLM request."""

        self._synchronize()

        return {
            "revision": self._revision,
            "messages": self.get_messages(),
        }

    def get_statistics(self):

        self._synchronize()

        return {
            "turn_count": len(self._turns),
            "message_count": len(self._turns) * 2,
            "character_count": self._character_count,
            "max_turns": self._max_turns,
            "max_characters": self._max_characters,
            "revision": self._revision,
            "persistent_storage": False,
        }

    def clear(self):
        """Remove conversational history."""

        self._turns.clear()

        self._character_count = 0

        self._revision = self._session.revision
