from __future__ import annotations

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)


# ============================================================
# VERSION
# ============================================================

ANALYTICAL_DATASET_CONTEXT_VERSION = (
    "analytical_dataset_context_v1.0"
)


# ============================================================
# DATASET COLUMN
# ============================================================

class DatasetColumnSpec(
    BaseModel
):
    """
    Minimal production representation of a dataset column
    exposed to structural analytical planning.

    analytical_type remains a string because the upstream
    profiling layer owns analytical type inference and the
    planning stack consumes the resulting trusted value.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    name: str = Field(
        min_length=1,
    )

    analytical_type: str = Field(
        min_length=1,
    )

    semantic_role: (
        str
        | None
    ) = None


# ============================================================
# DATASET CONTEXT
# ============================================================

class DatasetContext(
    BaseModel
):
    """
    Trusted structural description of one dataset.

    No raw dataset values are carried here.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    dataset_id: str = Field(
        min_length=1,
    )

    filename: str = Field(
        min_length=1,
    )

    grain: str = Field(
        min_length=1,
    )

    entity_columns: list[
        str
    ] = Field(
        default_factory=list,
    )

    columns: list[
        DatasetColumnSpec
    ] = Field(
        min_length=1,
    )


    @model_validator(
        mode="after",
    )
    def validate_entity_columns(
        self,
    ) -> "DatasetContext":

        column_names = {
            column.name

            for column
            in self.columns
        }


        unknown_entity_columns = [
            entity_column

            for entity_column
            in self.entity_columns

            if (
                entity_column
                not in column_names
            )
        ]


        if unknown_entity_columns:

            raise ValueError(
                "Dataset entity_columns must reference "
                "existing dataset columns. "
                f"dataset_id={self.dataset_id}, "
                "unknown="
                f"{unknown_entity_columns}"
            )


        return self
