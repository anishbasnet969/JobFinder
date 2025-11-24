"""Initial migration with Job model

Revision ID: 7c26a668f8ab
Revises:
Create Date: 2025-11-24 16:37:01.664137

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "7c26a668f8ab"
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "jobs",
        sa.Column("id", sa.BigInteger(), nullable=False),
        sa.Column("experience", sa.String(), nullable=True),
        sa.Column("qualifications", sa.String(), nullable=True),
        sa.Column("salary_range", sa.String(), nullable=True),
        sa.Column("location", sa.String(), nullable=True),
        sa.Column("country", sa.String(), nullable=True),
        sa.Column("latitude", sa.Float(), nullable=True),
        sa.Column("longitude", sa.Float(), nullable=True),
        sa.Column("work_type", sa.String(), nullable=True),
        sa.Column("company_size", sa.Integer(), nullable=True),
        sa.Column("job_posting_date", sa.Date(), nullable=True),
        sa.Column("preference", sa.String(), nullable=True),
        sa.Column("contact_person", sa.String(), nullable=True),
        sa.Column("contact", sa.String(), nullable=True),
        sa.Column("job_title", sa.String(), nullable=True),
        sa.Column("role", sa.String(), nullable=True),
        sa.Column("job_portal", sa.String(), nullable=True),
        sa.Column("job_description", sa.Text(), nullable=True),
        sa.Column("benefits", sa.ARRAY(sa.String()), nullable=True),
        sa.Column("skills", sa.Text(), nullable=True),
        sa.Column("responsibilities", sa.Text(), nullable=True),
        sa.Column("company", sa.String(), nullable=True),
        sa.Column("company_profile", sa.JSON(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_jobs_id"), "jobs", ["id"], unique=False)


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(op.f("ix_jobs_id"), table_name="jobs")
    op.drop_table("jobs")
