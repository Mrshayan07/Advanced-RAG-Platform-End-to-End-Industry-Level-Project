
from __future__ import annotations

from uuid import UUID

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import UserMemory


class UserMemoryService:
    """
    Durable long-term memory service.

    PostgreSQL stores persistent user facts such as:
    - name
    - university
    - degree
    - skills
    - preferences

    Memory is isolated by both tenant_id and user_id.
    """

    def __init__(self, db: AsyncSession) -> None:
        self.db = db

    async def set_memory(
        self,
        *,
        tenant_id: UUID,
        user_id: UUID,
        memory_key: str,
        memory_value: str,
        memory_type: str = "profile",
    ) -> UserMemory:
        """
        Create or update a durable user memory.

        Example:
            name -> Shayan Javed
            university -> Sindh Madressatul Islam University
        """

        result = await self.db.execute(
            select(UserMemory).where(
                UserMemory.tenant_id == tenant_id,
                UserMemory.user_id == user_id,
                UserMemory.memory_key == memory_key,
            )
        )

        memory = result.scalar_one_or_none()

        if memory is None:
            memory = UserMemory(
                tenant_id=tenant_id,
                user_id=user_id,
                memory_key=memory_key,
                memory_value=memory_value,
                memory_type=memory_type,
            )

            self.db.add(memory)
        else:
            memory.memory_value = memory_value
            memory.memory_type = memory_type

        await self.db.commit()
        await self.db.refresh(memory)

        return memory

    async def get_memory(
        self,
        *,
        tenant_id: UUID,
        user_id: UUID,
        memory_key: str,
    ) -> UserMemory | None:
        """Get one memory belonging to the authenticated user."""

        result = await self.db.execute(
            select(UserMemory).where(
                UserMemory.tenant_id == tenant_id,
                UserMemory.user_id == user_id,
                UserMemory.memory_key == memory_key,
            )
        )

        return result.scalar_one_or_none()

    async def get_memories(
        self,
        *,
        tenant_id: UUID,
        user_id: UUID,
        memory_type: str | None = None,
    ) -> list[UserMemory]:
        """
        Get all durable memories for a user.

        Optionally filter by memory type.
        """

        query = select(UserMemory).where(
            UserMemory.tenant_id == tenant_id,
            UserMemory.user_id == user_id,
        )

        if memory_type is not None:
            query = query.where(
                UserMemory.memory_type == memory_type
            )

        query = query.order_by(UserMemory.created_at.asc())

        result = await self.db.execute(query)

        return list(result.scalars().all())

    async def delete_memory(
        self,
        *,
        tenant_id: UUID,
        user_id: UUID,
        memory_key: str,
    ) -> bool:
        """
        Delete one user memory.

        Returns:
            True  -> memory existed and was deleted
            False -> memory did not exist
        """

        result = await self.db.execute(
            select(UserMemory).where(
                UserMemory.tenant_id == tenant_id,
                UserMemory.user_id == user_id,
                UserMemory.memory_key == memory_key,
            )
        )

        memory = result.scalar_one_or_none()

        if memory is None:
            return False

        await self.db.delete(memory)
        await self.db.commit()

        return True

    async def clear_memories(
        self,
        *,
        tenant_id: UUID,
        user_id: UUID,
    ) -> int:
        """
        Delete all durable memories for a user.

        Returns:
            Number of deleted rows.
        """

        result = await self.db.execute(
            delete(UserMemory).where(
                UserMemory.tenant_id == tenant_id,
                UserMemory.user_id == user_id,
            )
        )

        await self.db.commit()

        return result.rowcount or 0
