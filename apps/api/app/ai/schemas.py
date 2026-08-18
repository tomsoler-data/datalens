from typing import Literal

from pydantic import (
    BaseModel,
    Field,
)


EvidenceSourceType = Literal[
    "cleaning_operation",
    "review_item",
    "profile_metric",
    "statistical_result",
    "relationship_result",
]


ClaimField = Literal[
    # Cleaning
    "operation",
    "column",
    "affected_rows",
    "affected_values",
    "affected_columns",
    "from_dtype",
    "to_dtype",
    "automatic",
    "reversible",

    # Statistics
    "x_column",
    "y_column",
    "n_total",
    "n_valid",
    "n_excluded",
    "test",
    "relationship_type",
    "coefficient_name",
    "coefficient",
    "p_value",
    "alternative",
    "n",
    "alpha",
    "statistically_significant",
]


class EvidenceReference(
    BaseModel
):
    source_type: EvidenceSourceType

    reference: str = Field(
        min_length=1,
        description=(
            "Exact evidence ID."
        ),
    )


class EvidenceClaim(
    BaseModel
):
    reference: str = Field(
        min_length=1,
        description=(
            "Evidence ID containing "
            "the claimed field."
        ),
    )

    field: ClaimField

    value: str = Field(
        min_length=1,
        description=(
            "Exact textual representation "
            "of the deterministic value."
        ),
    )


class AIFinding(
    BaseModel
):
    statement: str = Field(
        min_length=1,
        description=(
            "Concise human-readable explanation "
            "of deterministic evidence."
        ),
    )

    evidence: list[
        EvidenceReference
    ] = Field(
        min_length=1,
    )

    claims: list[
        EvidenceClaim
    ] = Field(
        min_length=1,
    )


class AIWarning(
    BaseModel
):
    message: str = Field(
        min_length=1,
    )

    severity: Literal[
        "info",
        "low",
        "medium",
        "high",
    ] = "info"

    evidence: list[
        EvidenceReference
    ] = Field(
        default_factory=list,
    )


class DatasetAIExplanation(
    BaseModel
):
    dataset: str = Field(
        min_length=1,
    )

    summary: str = Field(
        min_length=1,
        description=(
            "Short summary based only "
            "on validated findings."
        ),
    )

    findings: list[
        AIFinding
    ] = Field(
        default_factory=list,
    )

    warnings: list[
        AIWarning
    ] = Field(
        default_factory=list,
    )