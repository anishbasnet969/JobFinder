from sqlalchemy import Column, Integer, String, Text, JSON, ARRAY, BigInteger, text
from sqlalchemy.dialects.postgresql import UUID
from pgvector.sqlalchemy import Vector
from app.database import Base
import uuid


class Job(Base):
    """
    Job model following JSON Resume Job Description Schema.
    """

    __tablename__ = "jobs"

    id = Column(BigInteger, primary_key=True, index=True, autoincrement=True)
    job_id = Column(
        UUID(as_uuid=True),
        unique=True,
        index=True,
        nullable=False,
        default=uuid.uuid4,
        server_default=text("gen_random_uuid()"),
    )

    # Basic information
    title = Column(String, nullable=True, index=True)
    company = Column(String, nullable=True, index=True)
    employment_type = Column(
        String, nullable=True
    )  # Full-time, Part-time, Contract, etc.
    date = Column(String, nullable=True)  # Posting date YYYY-MM
    description = Column(Text, nullable=True)
    location = Column(JSON, nullable=True)

    # Work arrangement and compensation
    remote = Column(String, nullable=True)  # Remote, Hybrid, On-site
    salary = Column(String, nullable=True)

    # Detailed requirements - Experience
    experience = Column(String, nullable=True)  # Raw text: "5 to 15 Years"
    min_experience_years = Column(Integer, nullable=True)  # e.g., 5
    max_experience_years = Column(Integer, nullable=True)  # e.g., 15
    experience_level = Column(
        String, nullable=True
    )  # Entry-level, Mid-level, Senior, Lead, Principal

    # Other requirements
    responsibilities = Column(ARRAY(Text), nullable=True)
    qualifications = Column(ARRAY(Text), nullable=True)
    education = Column(JSON, nullable=True)  # List of education requirements
    skills = Column(JSON, nullable=True)

    # Vector embedding for semantic search
    embedding = Column(Vector(3072), nullable=True)
