"""add citations to messages

Revision ID: 50489294872d
Revises: c47e8021a1f3
Create Date: 2026-10-05 07:54:04.705436

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = "50489294872d"
down_revision: Union[str, Sequence[str], None] = "c47e8021a1f3"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""

    # Add the column temporarily as nullable so existing
    # message rows can be populated safely.
    op.add_column(
        "messages",
        sa.Column(
            "citations",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=True,
        ),
    )

    # Existing messages did not have citations.
    # Use an empty JSON array for those historical rows.
    op.execute(
        sa.text(
            """
            UPDATE messages
            SET citations = '[]'::jsonb
            WHERE citations IS NULL
            """
        )
    )

    # All existing rows now have a valid citations value,
    # so enforce the application-level NOT NULL constraint.
    op.alter_column(
        "messages",
        "citations",
        existing_type=postgresql.JSONB(
            astext_type=sa.Text()
        ),
        nullable=False,
    )


def downgrade() -> None:
    """Downgrade schema."""

    op.drop_column(
        "messages",
        "citations",
    )