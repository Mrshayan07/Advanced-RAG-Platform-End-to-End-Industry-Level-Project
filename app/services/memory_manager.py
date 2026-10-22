
from __future__ import annotations

from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.services.memory_extractor import MemoryExtractor
from app.services.user_memory import UserMemoryService


class MemoryManager:
    """
    Coordinates long-term memory extraction and persistence.

    Responsibilities:
    1. Extract durable facts from the user's message.
    2. Validate/normalize extracted candidates.
    3. Persist valid memories through UserMemoryService.

    This class intentionally does not contain database queries itself.
    """

    # Memory keys that are currently allowed to be persisted.
    #
    # Keeping an explicit allow-list is safer than allowing the LLM
    # to create arbitrary database keys.
    ALLOWED_KEYS = frozenset({
        "name",
        "university",
        "degree",
        "job_title",
        "skill",
        "preference",
    })

    def __init__(
        self,
        db: AsyncSession,
    ) -> None:
        self.extractor = MemoryExtractor()
        self.memory_service = UserMemoryService(db)

    @staticmethod
    def _normalize_value(value: str) -> str:
        """Normalize extracted memory values."""

        return " ".join(value.strip().split())

    def _is_valid_candidate(
        self,
        key: str,
        value: str,
    ) -> bool:
        """
        Validate a memory candidate before persistence.
        """

        if key not in self.ALLOWED_KEYS:
            return False

        if not value.strip():
            return False

        # Prevent accidentally storing very large values.
        if len(value) > 500:
            return False

        return True

    async def process_user_message(
        self,
        *,
        tenant_id: UUID,
        user_id: UUID,
        user_message: str,
    ) -> list[dict[str, str]]:
        """
        Extract and persist durable memories from a user message.

        Returns a list describing memories that were successfully
        persisted.
        """

        if not user_message.strip():
            return []

        extraction_result = await self.extractor.extract(
            user_message
        )

        persisted_memories: list[dict[str, str]] = []

        for candidate in extraction_result.memories:
            key = candidate.key.strip().lower()
            value = self._normalize_value(candidate.value)

            if not self._is_valid_candidate(key, value):
                continue

            memory = await self.memory_service.set_memory(
                tenant_id=tenant_id,
                user_id=user_id,
                memory_key=key,
                memory_value=value,
                memory_type=candidate.memory_type,
            )

            persisted_memories.append(
                {
                    "key": memory.memory_key,
                    "value": memory.memory_value,
                    "memory_type": memory.memory_type,
                }
            )

        return persisted_memories
