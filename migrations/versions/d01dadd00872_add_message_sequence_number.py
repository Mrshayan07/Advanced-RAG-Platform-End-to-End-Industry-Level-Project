
"""add message sequence number

Revision ID: d01dadd00872
Revises: 50489294872d
Create Date: 2026-10-08 08:21:59.983282

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "d01dadd00872"
down_revision: Union[str, Sequence[str], None] = "50489294872d"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ---------------------------------------------------------
    # 1. Add sequence_number temporarily as nullable.
    # ---------------------------------------------------------
    op.add_column(
        "messages",
        sa.Column(
            "sequence_number",
            sa.Integer(),
            nullable=True,
        ),
    )

    # ---------------------------------------------------------
    # 2. Backfill existing messages.
    #
    # Each conversation gets its own sequence:
    #
    # 1 -> user
    # 2 -> assistant
    # 3 -> user
    # 4 -> assistant
    #
    # created_at gives chronological ordering.
    # id provides a deterministic tie-breaker when two
    # messages have exactly the same created_at timestamp.
    # ---------------------------------------------------------
    op.execute(
        """
        WITH numbered_messages AS (
            SELECT
                id,
                ROW_NUMBER() OVER (
                    PARTITION BY conversation_id
                    ORDER BY created_at ASC, id ASC
                ) AS sequence_number
            FROM messages
        )
        UPDATE messages AS m
        SET sequence_number = nm.sequence_number
        FROM numbered_messages AS nm
        WHERE m.id = nm.id
        """
    )

    # ---------------------------------------------------------
    # 3. sequence_number must exist for every message.
    # ---------------------------------------------------------
    op.alter_column(
        "messages",
        "sequence_number",
        existing_type=sa.Integer(),
        nullable=False,
    )

    # ---------------------------------------------------------
    # 4. Prevent duplicate sequence numbers inside the same
    #    conversation.
    # ---------------------------------------------------------
    op.create_unique_constraint(
        "uq_messages_conversation_sequence",
        "messages",
        ["conversation_id", "sequence_number"],
    )


def downgrade() -> None:
    op.drop_constraint(
        "uq_messages_conversation_sequence",
        "messages",
        type_="unique",
    )

    op.drop_column(
        "messages",
        "sequence_number",
    )
