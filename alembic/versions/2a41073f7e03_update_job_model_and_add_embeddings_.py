"""update job model and add embeddings column

Revision ID: 2a41073f7e03
Revises: 7c26a668f8ab
Create Date: 2025-11-30 08:30:06.536942

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from pgvector.sqlalchemy import Vector

# revision identifiers, used by Alembic.
revision: str = "2a41073f7e03"
down_revision: Union[str, Sequence[str], None] = "7c26a668f8ab"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Enable pgvector extension
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # Drop the old jobs table if it exists
    try:
        op.drop_index(op.f("ix_jobs_id"), table_name="jobs")
    except Exception:
        # index may not exist yet
        pass
    op.drop_table("jobs")

    # Create new jobs table with updated schema
    op.create_table(
        "jobs",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column(
            "job_id",
            postgresql.UUID(),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("title", sa.String(), nullable=True),
        sa.Column("company", sa.String(), nullable=True),
        sa.Column("type", sa.String(), nullable=True),
        sa.Column("date", sa.String(), nullable=True),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("location", sa.JSON(), nullable=True),
        sa.Column("remote", sa.String(), nullable=True),
        sa.Column("salary", sa.String(), nullable=True),
        sa.Column("experience", sa.String(), nullable=True),
        sa.Column("responsibilities", sa.ARRAY(sa.Text()), nullable=True),
        sa.Column("qualifications", sa.ARRAY(sa.Text()), nullable=True),
        sa.Column("education", sa.JSON(), nullable=True),
        sa.Column("skills", sa.JSON(), nullable=True),
        sa.Column("embedding", Vector(3072), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_jobs_id"), "jobs", ["id"], unique=False)
    op.create_index(op.f("ix_jobs_job_id"), "jobs", ["job_id"], unique=True)
    op.create_index(op.f("ix_jobs_title"), "jobs", ["title"], unique=False)
    op.create_index(op.f("ix_jobs_company"), "jobs", ["company"], unique=False)


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(op.f("ix_jobs_company"), table_name="jobs")
    op.drop_index(op.f("ix_jobs_title"), table_name="jobs")
    op.drop_index(op.f("ix_jobs_job_id"), table_name="jobs")
    op.drop_index(op.f("ix_jobs_id"), table_name="jobs")
    op.drop_table("jobs")

    # Recreate old jobs table
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

    # Disable pgvector extension (only if no other tables use it)
    # op.execute("DROP EXTENSION IF EXISTS vector")
