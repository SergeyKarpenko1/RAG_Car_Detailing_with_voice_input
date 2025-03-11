from pydantic import BaseModel


class Venue(BaseModel):
    """
    Represents the data structure of a Blogs.
    """

    name: str
    content_text: str
    links: str