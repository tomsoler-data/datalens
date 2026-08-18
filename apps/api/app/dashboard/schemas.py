from typing import Literal

from pydantic import (
    BaseModel,
    Field,
)


# ============================================================
# DASHBOARD TYPES
# ============================================================

DashboardStatus = Literal[
    "complete",
    "ready_for_execution",
    "needs_information",
    "not_applicable",
]


DashboardKPIKind = Literal[
    "sample_size",
    "excluded_rows",
    "test",
    "coefficient",
    "p_value",
    "significance",
]


DashboardChartPurpose = Literal[
    "relationship",
    "diagnostic",
]


# ============================================================
# KPI
# ============================================================

class DashboardKPI(
    BaseModel
):
    """
    One deterministic KPI displayed in the
    generated dashboard.

    `value` is stored as text so the exact
    canonical deterministic value is preserved.
    """

    key: str = Field(
        min_length=1,
    )

    label: str = Field(
        min_length=1,
    )

    kind: DashboardKPIKind

    value: str = Field(
        min_length=1,
    )

    source_reference: str = Field(
        min_length=1,
    )

    source_field: str = Field(
        min_length=1,
    )


# ============================================================
# CHART SPECIFICATION
# ============================================================

class DashboardChart(
    BaseModel
):
    """
    Rendering specification for the main
    dashboard visualization.
    """

    visualization_reference: str = Field(
        min_length=1,
    )

    chart_type: str = Field(
        min_length=1,
    )

    purpose: DashboardChartPurpose

    x_column: str = Field(
        min_length=1,
    )

    y_column: str = Field(
        min_length=1,
    )

    aggregation: str = Field(
        min_length=1,
    )

    trend: str = Field(
        min_length=1,
    )

    show_raw_points: bool

    show_missing_summary: bool

    reasons: list[
        str
    ] = Field(
        default_factory=list,
    )


# ============================================================
# STATISTICAL RESULT BLOCK
# ============================================================

class DashboardStatisticalResult(
    BaseModel
):
    """
    Compact presentation block for an executed
    statistical test.
    """

    statistic_reference: str = Field(
        min_length=1,
    )

    test: str = Field(
        min_length=1,
    )

    relationship_type: str = Field(
        min_length=1,
    )

    coefficient_name: str = Field(
        min_length=1,
    )

    coefficient: str = Field(
        min_length=1,
    )

    p_value: str = Field(
        min_length=1,
    )

    alpha: str = Field(
        min_length=1,
    )

    statistically_significant: bool

    n: int = Field(
        ge=3,
    )

    inference_method: str = Field(
        min_length=1,
    )

    permutation_mode: (
        str | None
    ) = None


# ============================================================
# DECISION EXPLANATION
# ============================================================

class DashboardDecisionExplanation(
    BaseModel
):
    """
    Deterministic explanation of why a
    statistical test was or was not selected.
    """

    decision_reference: str = Field(
        min_length=1,
    )

    status: str = Field(
        min_length=1,
    )

    analysis_goal: str = Field(
        min_length=1,
    )

    analysis_mode: str = Field(
        min_length=1,
    )

    selected_test: (
        str | None
    ) = None

    selection_is_data_driven: bool

    reasons: list[
        str
    ] = Field(
        default_factory=list,
    )

    missing_information: list[
        str
    ] = Field(
        default_factory=list,
    )


# ============================================================
# EVIDENCE REFERENCES
# ============================================================

class DashboardEvidenceReferences(
    BaseModel
):
    """
    References that will later connect directly
    to the DataLens evidence layer.
    """

    decision: str = Field(
        default="decision:0001",
        min_length=1,
    )

    statistic: (
        str | None
    ) = None

    visualization: str = Field(
        min_length=1,
    )


# ============================================================
# COMPLETE DASHBOARD
# ============================================================

class DashboardSpec(
    BaseModel
):
    """
    Deterministic dashboard specification.

    The frontend will eventually render this
    object using reusable Next.js components.
    """

    dashboard_id: str = Field(
        default="dashboard:0001",
        min_length=1,
    )

    status: DashboardStatus

    title: str = Field(
        min_length=1,
    )

    subtitle: str = Field(
        min_length=1,
    )

    summary: str = Field(
        min_length=1,
    )

    x_column: str = Field(
        min_length=1,
    )

    y_column: str = Field(
        min_length=1,
    )

    kpis: list[
        DashboardKPI
    ] = Field(
        default_factory=list,
    )

    chart: (
        DashboardChart | None
    ) = None

    statistical_result: (
        DashboardStatisticalResult | None
    ) = None

    decision: DashboardDecisionExplanation

    action_required: list[
        str
    ] = Field(
        default_factory=list,
    )

    warnings: list[
        str
    ] = Field(
        default_factory=list,
    )

    evidence: DashboardEvidenceReferences

    dashboard_rule_version: str = (
        "correlation_dashboard_v0.1"
    )