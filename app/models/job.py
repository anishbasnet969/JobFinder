from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    Date,
    Text,
    BigInteger,
    JSON,
    ARRAY,
)
from app.database import Base


class Job(Base):
    __tablename__ = "jobs"

    id = Column(BigInteger, primary_key=True, index=True)
    experience = Column(String)
    qualifications = Column(String)
    salary_range = Column(String)
    location = Column(String)
    country = Column(String)
    latitude = Column(Float)
    longitude = Column(Float)
    work_type = Column(String)
    company_size = Column(Integer)
    job_posting_date = Column(Date)
    preference = Column(String)
    contact_person = Column(String)
    contact = Column(String)
    job_title = Column(String)
    role = Column(String)
    job_portal = Column(String)
    job_description = Column(Text)
    benefits = Column(ARRAY(String))
    skills = Column(Text)
    responsibilities = Column(Text)
    company = Column(String)
    company_profile = Column(JSON)
