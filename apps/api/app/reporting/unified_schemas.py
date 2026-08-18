from __future__ import annotations


from typing import (
    Any,
    Literal,
)

from pydantic import (
    BaseModel,
    Field,
)


# ============================================================
# TYPES
# ============================================================

ReportFindingRole = Literal[
    "main_finding",
    "additional_finding",
    "diagnostic",
    "quality",
    "context",
    "blocked",
]


# ============================================================
# DATASET SUMMARY
# ============================================================

class ReportDatasetSummary(
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

    columns: list[
        str
    ] = Field(
        default_factory=list,
    )


# ============================================================
# REQUESTED FINDING
# ============================================================

class ReportRequestedFinding(
    BaseModel
):
    """
    Product-facing analytical result produced from
    an explicit verified documentary request.

    Requested findings are kept outside the
    exploratory ranking.

    Their documentary provenance comes from the
    verified Request Planner evidence, while their
    numerical result comes from the deterministic
    Requested Analysis Executor.
    """

    request_id: str

    analysis_id: str

    title: str

    origin: Literal[
        "requested"
    ] = "requested"

    kind: str

    scope: str

    family: str

    execution_status: str

    inferential_status: (
        str
        | None
    ) = None

    analysis_mode: (
        str
        | None
    ) = None

    dataset_id: (
        str
        | None
    ) = None

    datasets: list[
        str
    ] = Field(
        default_factory=list,
    )

    analytical_grain: (
        str
        | None
    ) = None

    variables: dict[
        str,
        str,
    ] = Field(
        default_factory=dict,
    )

    sample_size: int = Field(
        default=0,
        ge=0,
    )

    summary: list[
        str
    ] = Field(
        default_factory=list,
    )

    reasons: list[
        str
    ] = Field(
        default_factory=list,
    )

    caveats: list[
        str
    ] = Field(
        default_factory=list,
    )

    chart_type: (
        str
        | None
    ) = None

    chart_data: list[
        dict[
            str,
            Any,
        ]
    ] = Field(
        default_factory=list,
    )

    metrics: dict[
        str,
        Any,
    ] = Field(
        default_factory=dict,
    )

    # ========================================================
    # VERIFIED DOCUMENTARY PROVENANCE
    # ========================================================

    source_filename: str

    source_locator: str

    page_number: (
        int
        | None
    ) = None

    source_chunk_id: (
        str
        | None
    ) = None

    evidence_unit_id: (
        int
        | None
    ) = None

    evidence_quote: str

    adapter_rule_version: str = (
        "requested_report_adapter_v0.2"
    )


# ============================================================
# REPORT FINDING
# ============================================================

class ReportFinding(
    BaseModel
):
    analysis_id: str

    title: str

    role: ReportFindingRole

    scope: str

    family: str

    tier: str

    execution_status: str

    interestingness_score: float = Field(
        ge=0.0,
        le=100.0,
    )

    signal_score: float = Field(
        ge=0.0,
        le=100.0,
    )

    coverage_score: float = Field(
        ge=0.0,
        le=100.0,
    )

    consistency_score: float = Field(
        ge=0.0,
        le=100.0,
    )

    direction: str = "unknown"

    strength: str = "unknown"

    sample_size: int = Field(
        default=0,
        ge=0,
    )

    period_count: int = Field(
        default=0,
        ge=0,
    )

    datasets: list[
        str
    ] = Field(
        default_factory=list,
    )

    summary: list[
        str
    ] = Field(
        default_factory=list,
    )

    reasons: list[
        str
    ] = Field(
        default_factory=list,
    )

    caveats: list[
        str
    ] = Field(
        default_factory=list,
    )

    chart_type: (
        str
        | None
    ) = None

    chart_data: list[
        dict[
            str,
            Any,
        ]
    ] = Field(
        default_factory=list,
    )

    metrics: dict[
        str,
        Any,
    ] = Field(
        default_factory=dict,
    )


# ============================================================
# DATA QUALITY ITEM
# ============================================================

class ReportQualityItem(
    BaseModel
):
    analysis_id: str

    dataset: str

    row_count: int = Field(
        ge=0,
    )

    column_count: int = Field(
        ge=0,
    )

    missing_cells: int = Field(
        ge=0,
    )

    missing_ratio: float = Field(
        ge=0.0,
    )

    duplicate_rows: int = Field(
        ge=0,
    )

    duplicate_ratio: float = Field(
        ge=0.0,
    )

    completely_missing_columns: list[
        str
    ] = Field(
        default_factory=list,
    )

    constant_columns: list[
        str
    ] = Field(
        default_factory=list,
    )

    summary: list[
        str
    ] = Field(
        default_factory=list,
    )


# ============================================================
# BLOCKED ANALYSIS
# ============================================================

class ReportBlockedAnalysis(
    BaseModel
):
    analysis_id: str

    title: str

    family: str

    datasets: list[
        str
    ] = Field(
        default_factory=list,
    )

    reason: str

    caveats: list[
        str
    ] = Field(
        default_factory=list,
    )

    discovery_priority_score: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
    )


# ============================================================
# REPORT INVENTORY
# ============================================================

class ReportInventory(
    BaseModel
):
    dataset_count: int = Field(
        ge=0,
    )

    discovered_analysis_count: int = Field(
        ge=0,
    )

    executed_analysis_count: int = Field(
        ge=0,
    )

    requested_finding_count: int = Field(
        default=0,
        ge=0,
    )

    main_finding_count: int = Field(
        ge=0,
    )

    additional_finding_count: int = Field(
        ge=0,
    )

    diagnostic_count: int = Field(
        ge=0,
    )

    quality_check_count: int = Field(
        ge=0,
    )

    context_analysis_count: int = Field(
        ge=0,
    )

    blocked_analysis_count: int = Field(
        ge=0,
    )


# ============================================================
# COMPLETE UNIFIED REPORT
# ============================================================

class UnifiedAnalysisReport(
    BaseModel
):
    status: Literal[
        "ready"
    ] = "ready"

    title: str

    executive_summary: list[
        str
    ] = Field(
        default_factory=list,
    )

    inventory: ReportInventory

    datasets: list[
        ReportDatasetSummary
    ] = Field(
        default_factory=list,
    )

    requested_findings: list[
        ReportRequestedFinding
    ] = Field(
        default_factory=list,
    )

    main_findings: list[
        ReportFinding
    ] = Field(
        default_factory=list,
    )

    additional_findings: list[
        ReportFinding
    ] = Field(
        default_factory=list,
    )

    diagnostics: list[
        ReportFinding
    ] = Field(
        default_factory=list,
    )

    quality: list[
        ReportQualityItem
    ] = Field(
        default_factory=list,
    )

    context_analyses: list[
        ReportFinding
    ] = Field(
        default_factory=list,
    )

    blocked_analyses: list[
        ReportBlockedAnalysis
    ] = Field(
        default_factory=list,
    )

    methodology_notes: list[
        str
    ] = Field(
        default_factory=list,
    )

    report_rule_version: str = (
        "unified_report_composer_v0.2"
    )