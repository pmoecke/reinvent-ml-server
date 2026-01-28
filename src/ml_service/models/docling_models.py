from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    # The standard Pydantic V2 way to define a required field
    query: str = Field(...)
