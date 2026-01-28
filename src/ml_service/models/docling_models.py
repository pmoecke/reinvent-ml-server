from typing import Annotated

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: Annotated[str, Field(...)]
