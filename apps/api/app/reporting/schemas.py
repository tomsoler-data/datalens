from typing import Literal

from pydantic import (
    BaseModel,
    Field,
)


ReportStatus = Literal[
    "ready",
]


# ============================================================
# DATASETS
# ============================================================

class ReportDataset(
    BaseModel
):
    dataset_id: str

    filename: str

    row_count: int = Field(
        ge=0,
    )

    column_count: int = Field(
        ge=0,
    )

    memory_bytes: int = Field(
        ge=0,
    )


# ============================================================
# VARIABLES
# ============================================================

class ReportVariable(
    BaseModel
):
    column: str

    role: str

    analysis_kind: str


# ============================================================
# ANALYSES
# ============================================================

class ReportAnalysis(
    BaseModel
):
    analysis_id: str

    dataset_id: str

    dataset_filename: str

    title: str

    family: str

    priority_score: int = Field(
        ge=0,
        le=100,
    )

    readiness: str

    chart_type: str

    statistical_strategy: (
        str
        | None
    ) = None

    variables: list[
        ReportVariable
    ] = Field(
        default_factory=list,
    )

    reasons: list[
        str
    ] = Field(
        default_factory=list,
    )

    limitations: list[
        str
    ] = Field(
        default_factory=list,
    )


# ============================================================
# RELATIONSHIPS
# ============================================================

class ReportRelationship(
    BaseModel
):
    opportunity_id: str

    dataset_filenames: list[
        str
    ] = Field(
        default_factory=list,
    )

    shared_columns: list[
        str
    ] = Field(
        default_factory=list,
    )

    reason: str

    requires_relationship_validation: bool


# ============================================================
# ADDITIONAL DATA SUGGESTIONS
# ============================================================

class ReportDataSuggestion(
    BaseModel
):
    suggestion_id: str

    title: str

    priority: str

    rationale: str

    example_fields: list[
        str
    ] = Field(
        default_factory=list,
    )

    required_for_current_analysis: bool


# ============================================================
# COMPLETE ANALYSIS REPORT
# ============================================================

class AnalysisReport(
    BaseModel
):
    status: ReportStatus = (
        "ready"
    )

    title: str

    objective: (
        str
        | None
    ) = None

    dataset_count: int = Field(
        ge=0,
    )

    total_rows: int = Field(
        ge=0,
    )

    datasets: list[
        ReportDataset
    ] = Field(
        default_factory=list,
    )

    executive_summary: list[
        str
    ] = Field(
        default_factory=list,
    )

    analyses: list[
        ReportAnalysis
    ] = Field(
        default_factory=list,
    )

    relationships: list[
        ReportRelationship
    ] = Field(
        default_factory=list,
    )

    additional_data_suggestions: list[
        ReportDataSuggestion
    ] = Field(
        default_factory=list,
    )

    methodology_notes: list[
        str
    ] = Field(
        default_factory=list,
    )

    report_rule_version: str = (
        "analysis_report_v0.1"
    )