from pydantic import (
    BaseModel,
    Field,
)


class WorkspaceCreate(
    BaseModel
):
    name: str = Field(
        default="Untitled workspace",
        min_length=1,
        max_length=100,
    )
