from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user, get_db
from app.db.models import Conversation, Message
from app.schemas.conversations import (
    ConversationDetailResponse,
    ConversationResponse,
)

router = APIRouter(
    prefix="/conversations",
    tags=["Conversations"],
)


# ---------------------------------------------------------
# GET /conversations
# ---------------------------------------------------------
# Returns only conversations belonging to the authenticated
# user inside the authenticated tenant.
# ---------------------------------------------------------

@router.get(
    "",
    response_model=list[ConversationResponse],
)
async def list_conversations(
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    user_id: UUID = current_user["user_id"]
    tenant_id: UUID = current_user["tenant_id"]

    result = await db.execute(
        select(Conversation)
        .where(
            Conversation.user_id == user_id,
            Conversation.tenant_id == tenant_id,
        )
        .order_by(
            Conversation.created_at.desc(),
        )
    )

    conversations = result.scalars().all()

    return conversations


# ---------------------------------------------------------
# GET /conversations/{conversation_id}
# ---------------------------------------------------------
# Returns one conversation together with its messages.
# ---------------------------------------------------------

@router.get(
    "/{conversation_id}",
    response_model=ConversationDetailResponse,
)
async def get_conversation(
    conversation_id: UUID,
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    user_id: UUID = current_user["user_id"]
    tenant_id: UUID = current_user["tenant_id"]

    # -----------------------------------------------------
    # Ownership check
    # -----------------------------------------------------

    conversation_result = await db.execute(
        select(Conversation)
        .where(
            Conversation.id == conversation_id,
            Conversation.user_id == user_id,
            Conversation.tenant_id == tenant_id,
        )
    )

    conversation = conversation_result.scalar_one_or_none()

    if conversation is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found",
        )

    # -----------------------------------------------------
    # Load messages
    # -----------------------------------------------------

    messages_result = await db.execute(
        select(Message)
        .where(
            Message.conversation_id == conversation_id,
        )
        .order_by(
            Message.sequence_number.asc(),
        )
    )

    messages = messages_result.scalars().all()

    return ConversationDetailResponse(
        id=conversation.id,
        title=conversation.title,
        created_at=conversation.created_at,
        messages=messages,
    )


# ---------------------------------------------------------
# DELETE /conversations/{conversation_id}
# ---------------------------------------------------------
# Deletes the conversation and its messages.
#
# Message does not currently have SQLAlchemy cascade
# configured in models.py, therefore messages are deleted
# explicitly before deleting the conversation.
# ---------------------------------------------------------

@router.delete(
    "/{conversation_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_conversation(
    conversation_id: UUID,
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    user_id: UUID = current_user["user_id"]
    tenant_id: UUID = current_user["tenant_id"]

    # -----------------------------------------------------
    # Ownership check
    # -----------------------------------------------------

    conversation_result = await db.execute(
        select(Conversation)
        .where(
            Conversation.id == conversation_id,
            Conversation.user_id == user_id,
            Conversation.tenant_id == tenant_id,
        )
    )

    conversation = conversation_result.scalar_one_or_none()

    if conversation is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found",
        )

    # -----------------------------------------------------
    # Delete messages first
    # -----------------------------------------------------

    await db.execute(
        delete(Message).where(
            Message.conversation_id == conversation_id,
        )
    )

    # -----------------------------------------------------
    # Delete conversation
    # -----------------------------------------------------

    await db.delete(conversation)

    await db.commit()

    return None