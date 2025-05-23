"""Add file_name to VectorStore

Revision ID: 4b39d0683da3
Revises: e378c71a60f2
Create Date: 2025-05-15 20:37:16.153919

"""
from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "4b39d0683da3"
down_revision: Union[str, None] = "e378c71a60f2"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column(
        "vectorstores", sa.Column("file_name", sa.String(length=255), nullable=False)
    )
    op.drop_constraint("uix_user_vectorstore_name", "vectorstores", type_="unique")
    op.create_unique_constraint(
        "uix_user_vectorstore_file_name", "vectorstores", ["user_id", "file_name"]
    )
    op.drop_column("vectorstores", "name")
    # ### end Alembic commands ###


def downgrade() -> None:
    """Downgrade schema."""
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column(
        "vectorstores",
        sa.Column("name", sa.VARCHAR(length=255), autoincrement=False, nullable=False),
    )
    op.drop_constraint("uix_user_vectorstore_file_name", "vectorstores", type_="unique")
    op.create_unique_constraint(
        "uix_user_vectorstore_name", "vectorstores", ["user_id", "name"]
    )
    op.drop_column("vectorstores", "file_name")
    # ### end Alembic commands ###
