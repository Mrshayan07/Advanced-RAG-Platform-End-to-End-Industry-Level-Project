from __future__ import annotations

import json
import uuid
from typing import Any

from langchain_google_genai import ChatGoogleGenerativeAI
from redis import Redis

from app.core.config import get_settings

settings = get_settings()


class ConversationMemory:
    """
    Server-side short-term conversation memory.

    Storage strategy:

        Redis
          ├── conversation:{id}:summary
          └── conversation:{id}:messages

    messages contains only the latest N messages.

    Older messages are compressed into an LLM-generated summary.
    """

    def __init__(
        self,
        message_limit: int = 10,
    ):
        self.message_limit = message_limit

        self.redis = Redis.from_url(
            settings.redis_url,
            decode_responses=True,
        )

        # Initialize the LLM only when summary generation is requested.
        # This allows application imports and CI test collection without API keys.
        self._summary_model: ChatGoogleGenerativeAI | None = None

    # =========================================================
    # KEY HELPERS
    # =========================================================

    def _messages_key(
        self,
        conversation_id: str,
    ) -> str:
        return (
            f"conversation:"
            f"{conversation_id}:messages"
        )

    def _summary_key(
        self,
        conversation_id: str,
    ) -> str:
        return (
            f"conversation:"
            f"{conversation_id}:summary"
        )

    # =========================================================
    # CONVERSATION ID
    # =========================================================

    @staticmethod
    def create_conversation_id() -> str:
        """
        Generate a new conversation ID.
        """

        return str(uuid.uuid4())

    # =========================================================
    # REDIS HEALTH
    # =========================================================

    def ping(self) -> bool:
        """
        Check Redis connectivity.
        """

        return bool(self.redis.ping())

    # =========================================================
    # LOAD MEMORY
    # =========================================================

    def get_summary(
        self,
        conversation_id: str,
    ) -> str:
        """
        Retrieve the long portion of the conversation
        represented by an LLM-generated summary.
        """

        summary = self.redis.get(
            self._summary_key(
                conversation_id
            )
        )

        return summary or ""

    def get_recent_messages(
        self,
        conversation_id: str,
    ) -> list[dict[str, Any]]:
        """
        Retrieve recent messages.

        Redis stores messages in chronological order.
        """

        raw_messages = self.redis.lrange(
            self._messages_key(
                conversation_id
            ),
            0,
            -1,
        )

        messages: list[dict[str, Any]] = []

        for raw_message in raw_messages:
            try:
                message = json.loads(
                    raw_message
                )

                if isinstance(
                    message,
                    dict,
                ):
                    messages.append(
                        message
                    )

            except json.JSONDecodeError:
                continue

        return messages

    # =========================================================
    # BUILD CHAT HISTORY
    # =========================================================

    def build_chat_history(
        self,
        conversation_id: str,
    ) -> str:
        """
        Build the context sent to the existing RAG pipeline.

        Format:

            Conversation Summary:
            ...

            Recent Conversation:
            User: ...
            Assistant: ...
        """

        summary = self.get_summary(
            conversation_id
        )

        messages = self.get_recent_messages(
            conversation_id
        )

        parts: list[str] = []

        if summary:
            parts.append(
                "Conversation Summary:\n"
                f"{summary}"
            )

        if messages:
            recent_lines: list[str] = []

            for message in messages:
                role = message.get(
                    "role",
                    "user",
                )

                content = message.get(
                    "content",
                    "",
                )

                recent_lines.append(
                    f"{role.capitalize()}: {content}"
                )

            parts.append(
                "Recent Conversation:\n"
                + "\n".join(recent_lines)
            )

        return "\n\n".join(parts)

    # =========================================================
    # SAVE MESSAGE
    # =========================================================

    def save_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
    ) -> None:
        """
        Save a conversation message.

        We keep recent messages in a Redis list.
        """

        message = {
            "role": role,
            "content": content,
        }

        self.redis.rpush(
            self._messages_key(
                conversation_id
            ),
            json.dumps(
                message,
                ensure_ascii=False,
            ),
        )

    # =========================================================
    # SUMMARY GENERATION
    # =========================================================

    @property
    def summary_model(self) -> ChatGoogleGenerativeAI:
        """Create and cache the summary LLM on first use."""
        if self._summary_model is None:
            self._summary_model = ChatGoogleGenerativeAI(
                model=settings.gemini_model,
                google_api_key=settings.gemini_api_key,
                temperature=0,
                max_output_tokens=1024,
            )

        return self._summary_model

    def _generate_summary(
        self,
        existing_summary: str,
        messages_to_summarize: list[dict[str, Any]],
    ) -> str:
        """
        Merge older conversation messages into
        the existing conversation summary.
        """

        old_summary = (
            existing_summary
            if existing_summary
            else "No previous summary exists."
        )

        conversation_text = "\n".join(
            f"{message.get('role', 'user').capitalize()}: "
            f"{message.get('content', '')}"
            for message in messages_to_summarize
        )

        prompt = f"""
You are maintaining a compact conversation memory
for a production RAG application.

Your task is to update the conversation summary.

Existing summary:
{old_summary}

Older conversation messages:
{conversation_text}

Create an updated summary that preserves information
which may be important for future questions.

Preserve:
- important facts
- user goals
- decisions
- technical context
- important entities
- unresolved questions
- references needed to understand follow-up questions

Do NOT:
- invent information
- include unnecessary greetings
- include irrelevant wording
- mention that you are creating a summary

Return ONLY the updated summary.
""".strip()

        response = self.summary_model.invoke(
            prompt
        )

        content = response.content

        if isinstance(
            content,
            str,
        ):
            return content.strip()

        if isinstance(
            content,
            list,
        ):
            text_parts: list[str] = []

            for block in content:
                if isinstance(
                    block,
                    dict,
                ):
                    text = block.get(
                        "text",
                        "",
                    )

                    if isinstance(
                        text,
                        str,
                    ):
                        text_parts.append(
                            text
                        )

                else:
                    text = getattr(
                        block,
                        "text",
                        None,
                    )

                    if isinstance(
                        text,
                        str,
                    ):
                        text_parts.append(
                            text
                        )

            return "\n".join(
                text_parts
            ).strip()

        return str(content).strip()

    # =========================================================
    # COMPRESS OLD MESSAGES
    # =========================================================

    def compress_if_needed(
        self,
        conversation_id: str,
    ) -> None:
        """
        Keep only the latest `message_limit` messages.

        Older messages are summarized by the LLM.

        Example:

            limit = 10

            12 messages
                 ↓
            summarize first 2
                 ↓
            keep latest 10
        """

        key = self._messages_key(
            conversation_id
        )

        message_count = self.redis.llen(
            key
        )

        if message_count <= self.message_limit:
            return

        # Number of old messages that need
        # to be moved into the summary.
        remove_count = (
            message_count
            - self.message_limit
        )

        old_raw_messages = self.redis.lrange(
            key,
            0,
            remove_count - 1,
        )

        old_messages: list[
            dict[str, Any]
        ] = []

        for raw_message in old_raw_messages:
            try:
                message = json.loads(
                    raw_message
                )

                if isinstance(
                    message,
                    dict,
                ):
                    old_messages.append(
                        message
                    )

            except json.JSONDecodeError:
                continue

        if not old_messages:
            return

        existing_summary = self.get_summary(
            conversation_id
        )

        # Generate summary BEFORE deleting
        # old messages so a model failure does
        # not destroy conversation history.
        new_summary = self._generate_summary(
            existing_summary=existing_summary,
            messages_to_summarize=old_messages,
        )

        if not new_summary:
            return

        # Save summary.
        self.redis.set(
            self._summary_key(
                conversation_id
            ),
            new_summary,
        )

        # Remove exactly the messages that
        # were successfully summarized.
        self.redis.ltrim(
            key,
            remove_count,
            -1,
        )

    # =========================================================
    # SAVE TURN
    # =========================================================

    def save_turn(
        self,
        conversation_id: str,
        question: str,
        answer: str,
    ) -> None:
        """
        Save one complete RAG turn.
        """

        self.save_message(
            conversation_id=conversation_id,
            role="user",
            content=question,
        )

        self.save_message(
            conversation_id=conversation_id,
            role="assistant",
            content=answer,
        )

        self.compress_if_needed(
            conversation_id
        )

    # =========================================================
    # CLEAR CONVERSATION
    # =========================================================

    def clear(
        self,
        conversation_id: str,
    ) -> None:
        """
        Delete conversation memory.
        """

        self.redis.delete(
            self._messages_key(
                conversation_id
            ),
            self._summary_key(
                conversation_id
            ),
        )