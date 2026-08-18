from typing import Literal

from pydantic import (
    BaseModel,
    Field,
)


ColumnAnalysisKind = Literal[
    "quantitative",
    "temporal",
    "categorical",
    "boolean",
    "unknown",
]


CorrelationCompatibilityStatus = Literal[
    "ready",
    "not_available",
]


class DatasetColumnManifest(
    BaseModel
):
    name: str

    dtype: str

    missing_count: int

    missing_ratio: float

    unique_count: int

    unique_ratio: float

    unique_candidate: bool

    analysis_kind: ColumnAnalysisKind

    correlation_eligible: bool

    analysis_note: str


class CorrelationCompatibility(
    BaseModel
):
    status: CorrelationCompatibilityStatus

    candidate_columns: list[
        str
    ] = Field(
        default_factory=list
    )

    default_x_column: str | None = None

    default_y_column: str | None = None

    reasons: list[
        str
    ] = Field(
        default_factory=list
    )


class DatasetManifest(
    BaseModel
):
    dataset_id: str

    filename: str

    extension: str

    row_count: int

    column_count: int

    memory_bytes: int

    columns: list[
        DatasetColumnManifest
    ]

    correlation_compatibility: (
        CorrelationCompatibility
    )

    warnings: list[
        str
    ] = Field(
        default_factory=list
    )


class MultiDatasetIngestion(
    BaseModel
):
    status: Literal[
        "ready"
    ] = "ready"

    dataset_count: int

    total_rows: int

    datasets: list[
        DatasetManifest
    ]

    warnings: list[
        str
    ] = Field(
        default_factory=list
    )

    ingestion_rule_version: str = (
        "dataset_ingestion_v0.2"
    )