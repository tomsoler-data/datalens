from typing import Literal

from pydantic import (
    BaseModel,
    Field,
)


# ============================================================
# VISUALIZATION TYPES
# ============================================================

ChartType = Literal[
    "scatter",
    "hexbin",
    "ordinal_heatmap",
    "boxplot",
]


TrendType = Literal[
    "none",
    "linear",
]


VisualizationPurpose = Literal[
    "relationship",
    "diagnostic",
]


VisualizationStatus = Literal[
    "selected",
    "not_applicable",
]


AggregationType = Literal[
    "none",
    "count",
]


# ============================================================
# VISUALIZATION DECISION
# ============================================================

class VisualizationDecision(
    BaseModel
):
    """
    Deterministic visualization recommendation.

    The chart is selected by Python rules,
    not by the LLM.
    """

    visualization_id: str = Field(
        default="visualization:0001",
        min_length=1,
    )

    status: VisualizationStatus

    purpose: VisualizationPurpose

    chart_type: (
        ChartType | None
    ) = None

    x_column: str = Field(
        min_length=1,
    )

    y_column: str = Field(
        min_length=1,
    )

    aggregation: AggregationType = (
        "none"
    )

    trend: TrendType = (
        "none"
    )

    show_raw_points: bool = True

    show_missing_summary: bool = True

    selection_is_data_driven: bool = False

    reasons: list[
        str
    ] = Field(
        default_factory=list,
    )

    warnings: list[
        str
    ] = Field(
        default_factory=list,
    )

    compatible_alternatives: list[
        ChartType
    ] = Field(
        default_factory=list,
    )

    visualization_rule_version: str = (
        "correlation_visualization_v0.1"
    )