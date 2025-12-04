"""
This module defines Pydantic models for parsing and validating resume data
according to the JSON Resume standard (https://jsonresume.org).
"""

from typing import List, Optional
from pydantic import AnyUrl, EmailStr, Field
from app.schemas.base import CustomBaseModel
from app.schemas.common import Iso8601, Location


# ============================================================================
# Basic Information
# ============================================================================


class Profile(CustomBaseModel):
    network: Optional[str] = Field(None, description="e.g. Facebook or Twitter")
    username: Optional[str] = Field(None, description="e.g. neutralthoughts")
    url: Optional[AnyUrl] = Field(
        None, description="e.g. http://twitter.example.com/neutralthoughts"
    )


class Basics(CustomBaseModel):
    name: Optional[str] = None
    label: Optional[str] = Field(None, description="e.g. Web Developer")
    image: Optional[AnyUrl] = Field(
        None, description="URL (as per RFC 3986) to a image in JPEG or PNG format"
    )
    email: Optional[EmailStr] = Field(None, description="e.g. thomas@gmail.com")
    phone: Optional[str] = Field(
        None,
        description="Phone numbers are stored as strings so use any format you like, e.g. 712-117-2923",
    )
    url: Optional[AnyUrl] = Field(
        None,
        description="URL (as per RFC 3986) to your website, e.g. personal homepage",
    )
    summary: Optional[str] = Field(
        None, description="Write a short 2-3 sentence biography about yourself"
    )
    location: Optional[Location] = None
    profiles: Optional[List[Profile]] = Field(
        None,
        description="Specify any number of social networks that you participate in",
    )


# ============================================================================
# Work history, volunteer experience, and educational background.
# ============================================================================


class WorkItem(CustomBaseModel):
    name: Optional[str] = Field(None, description="e.g. Facebook")
    location: Optional[str] = Field(None, description="e.g. Menlo Park, CA")
    description: Optional[str] = Field(None, description="e.g. Social Media Company")
    position: Optional[str] = Field(None, description="e.g. Software Engineer")
    url: Optional[AnyUrl] = Field(None, description="e.g. http://facebook.example.com")
    start_date: Optional[Iso8601] = None
    end_date: Optional[Iso8601] = None
    summary: Optional[str] = Field(
        None, description="Give an overview of your responsibilities at the company"
    )
    highlights: Optional[List[str]] = Field(
        None, description="Specify multiple accomplishments"
    )


class VolunteerItem(CustomBaseModel):
    organization: Optional[str] = Field(None, description="e.g. Facebook")
    position: Optional[str] = Field(None, description="e.g. Software Engineer")
    url: Optional[AnyUrl] = Field(None, description="e.g. http://facebook.example.com")
    start_date: Optional[Iso8601] = None
    end_date: Optional[Iso8601] = None
    summary: Optional[str] = Field(
        None, description="Give an overview of your responsibilities at the company"
    )
    highlights: Optional[List[str]] = Field(
        None, description="Specify accomplishments and achievements"
    )


class EducationItem(CustomBaseModel):
    institution: Optional[str] = Field(
        None, description="e.g. Massachusetts Institute of Technology"
    )
    url: Optional[AnyUrl] = Field(None, description="e.g. http://facebook.example.com")
    area: Optional[str] = Field(None, description="e.g. Arts")
    study_type: Optional[str] = Field(None, description="e.g. Bachelor")
    start_date: Optional[Iso8601] = None
    end_date: Optional[Iso8601] = None
    score: Optional[str] = Field(None, description="grade point average, e.g. 3.67/4.0")
    courses: Optional[List[str]] = Field(
        None, description="List notable courses/subjects"
    )


# ============================================================================
# Skills & Interests
# ============================================================================


class Skill(CustomBaseModel):
    name: Optional[str] = Field(None, description="e.g. Web Development")
    level: Optional[str] = Field(None, description="e.g. Master")
    keywords: Optional[List[str]] = Field(
        None, description="List some keywords pertaining to this skill"
    )


class Language(CustomBaseModel):
    language: Optional[str] = Field(None, description="e.g. English, Spanish")
    fluency: Optional[str] = Field(None, description="e.g. Fluent, Beginner")


class Interest(CustomBaseModel):
    name: Optional[str] = Field(None, description="e.g. Philosophy")
    keywords: Optional[List[str]] = None


# ============================================================================
# Achievements
# ============================================================================


class Award(CustomBaseModel):
    title: Optional[str] = Field(
        None, description="e.g. One of the 100 greatest minds of the century"
    )
    date: Optional[Iso8601] = None
    awarder: Optional[str] = Field(None, description="e.g. Time Magazine")
    summary: Optional[str] = Field(
        None, description="e.g. Received for my work with Quantum Physics"
    )


class Certificate(CustomBaseModel):
    name: Optional[str] = Field(
        None, description="e.g. Certified Kubernetes Administrator"
    )
    date: Optional[Iso8601] = None
    url: Optional[AnyUrl] = Field(None, description="e.g. http://example.com")
    issuer: Optional[str] = Field(None, description="e.g. CNCF")


class Publication(CustomBaseModel):
    name: Optional[str] = Field(None, description="e.g. The World Wide Web")
    publisher: Optional[str] = Field(None, description="e.g. IEEE, Computer Magazine")
    release_date: Optional[Iso8601] = None
    url: Optional[AnyUrl] = Field(
        None,
        description="e.g. http://www.computer.org.example.com/csdl/mags/co/1996/10/rx069-abs.html",
    )
    summary: Optional[str] = Field(
        None,
        description="Short summary of publication. e.g. Discussion of the World Wide Web, HTTP, HTML.",
    )


# ============================================================================
# Additional Information
# ============================================================================


class Project(CustomBaseModel):
    name: Optional[str] = Field(None, description="e.g. The World Wide Web")
    description: Optional[str] = Field(
        None, description="Short summary of project. e.g. Collated works of 2017."
    )
    highlights: Optional[List[str]] = Field(
        None, description="Specify multiple features"
    )
    keywords: Optional[List[str]] = Field(
        None, description="Specify special elements involved"
    )
    start_date: Optional[Iso8601] = None
    end_date: Optional[Iso8601] = None
    url: Optional[AnyUrl] = Field(
        None,
        description="e.g. http://www.computer.org/csdl/mags/co/1996/10/rx069-abs.html",
    )
    roles: Optional[List[str]] = Field(
        None, description="Specify your role on this project or in company"
    )
    entity: Optional[str] = Field(
        None,
        description="Specify the relevant company/entity affiliations e.g. 'greenpeace', 'corporationXYZ'",
    )
    type: Optional[str] = Field(
        None,
        description=" e.g. 'volunteering', 'presentation', 'talk', 'application', 'conference'",
    )


class Reference(CustomBaseModel):
    name: Optional[str] = Field(None, description="e.g. Timothy Cook")
    reference: Optional[str] = Field(
        None,
        description="e.g. Joe blogs was a great employee, who turned up to work at least once a week. He exceeded my expectations when it came to doing nothing.",
    )


# ============================================================================
# Resume Model
# ============================================================================


class Resume(CustomBaseModel):
    basics: Optional[Basics] = None
    work: Optional[List[WorkItem]] = None
    volunteer: Optional[List[VolunteerItem]] = None
    education: Optional[List[EducationItem]] = None
    awards: Optional[List[Award]] = Field(
        None,
        description="Specify any awards you have received throughout your professional career",
    )
    certificates: Optional[List[Certificate]] = Field(
        None,
        description="Specify any certificates you have received throughout your professional career",
    )
    publications: Optional[List[Publication]] = Field(
        None, description="Specify your publications through your career"
    )
    skills: Optional[List[Skill]] = Field(
        None, description="List out your professional skill-set"
    )
    languages: Optional[List[Language]] = Field(
        None, description="List any other languages you speak"
    )
    interests: Optional[List[Interest]] = None

    projects: Optional[List[Project]] = Field(
        None, description="Specify career projects"
    )
    references: Optional[List[Reference]] = Field(
        None, description="List references you have received"
    )
