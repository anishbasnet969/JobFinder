"""
Common schema types shared across different schemas.
"""

from typing import Annotated, Optional
from pydantic import Field, StringConstraints
from app.schemas.base import CustomBaseModel


# ISO 8601 date type: YYYY, YYYY-MM, or YYYY-MM-DD format
Iso8601 = Annotated[
    str,
    StringConstraints(
        pattern=r"^([1-2][0-9]{3}-[0-1][0-9]-[0-3][0-9]|[1-2][0-9]{3}-[0-1][0-9]|[1-2][0-9]{3})$"
    ),
    Field(
        description="Similar to the standard date type, but each section after the year is optional. e.g. 2014-06-29 or 2023-04"
    ),
]


class Location(CustomBaseModel):
    address: Optional[str] = Field(
        None,
        description="To add multiple address lines, use \n. For example, 1234 Glücklichkeit Straße\nHinterhaus 5. Etage li.",
    )
    postal_code: Optional[str] = None
    city: Optional[str] = None
    country_code: Optional[str] = Field(
        None, description="code as per ISO-3166-1 ALPHA-2, e.g. US, AU, IN"
    )
    region: Optional[str] = Field(
        None,
        description="The general region where you live. Can be a US state, or a province, for instance.",
    )
