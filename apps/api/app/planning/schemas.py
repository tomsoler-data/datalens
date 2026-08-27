from __future__ import annotations


from typing import (
    Literal,
)


from pydantic import (
    BaseModel,
    Field,
)


# ============================================================
# EXPLORATORY PLANNER
# ============================================================

AnalysisFamily = Literal[
    "time_series",
    "quantitative_association",
    "group_comparison",
    "categorical_association",
    "distribution",
]


AnalysisReadiness = Literal[
    "executable_now",
    "planned",
]


ChartType = Literal[
    "line",
    "scatter",
    "hexbin",
    "boxplot",
    "grouped_bar",
    "stacked_bar",
    "histogram",
    "heatmap",
]


SuggestionPriority = Literal[
    "high",
    "medium",
    "low",
]


# ============================================================
# REQUEST PLANNER
# ============================================================

RequestPlanningStatus = Literal[
    "ready",
    "blocked",
    "ambiguous",
]


RequestedAnalysisKind = Literal[
    "revenue_moving_average",
    "revenue_by_category",
    "customers_by_period",
    "transaction_count",
    "products_sold_count",
    "top_products",
    "flop_products",
    "product_category_distribution",
    "b2b_revenue_distribution",
    "lorenz_curve",
    "gender_category_association",
    "age_total_amount_association",
    "age_frequency_association",
    "age_average_basket_association",
    "age_category_association",
    "unknown",
]


# ============================================================
# EXPLORATORY PLANNER SCHEMAS
# ============================================================

class PlannedVariable(
    BaseModel
):
    column: str

    role: Literal[
        "x",
        "y",
        "time",
        "value",
        "group",
        "category",
    ]

    analysis_kind: str


class AnalysisCandidate(
    BaseModel
):
    analysis_id: str

    dataset_id: str

    dataset_filename: str

    title: str

    family: AnalysisFamily

    priority_score: int

    readiness: AnalysisReadiness

    variables: list[
        PlannedVariable
    ]

    chart_type: ChartType

    statistical_strategy: (
        str
        | None
    ) = None

    reasons: list[
        str
    ] = Field(
        default_factory=list
    )

    limitations: list[
        str
    ] = Field(
        default_factory=list
    )


class CrossDatasetOpportunity(
    BaseModel
):
    opportunity_id: str

    dataset_ids: list[
        str
    ]

    dataset_filenames: list[
        str
    ]

    shared_columns: list[
        str
    ]

    reason: str

    requires_relationship_validation: bool = True


class AdditionalDataSuggestion(
    BaseModel
):
    suggestion_id: str

    title: str

    priority: SuggestionPriority

    rationale: str

    example_fields: list[
        str
    ] = Field(
        default_factory=list
    )

    required_for_current_analysis: bool = False


# ============================================================
# REQUEST PLANNER SCHEMAS
# ============================================================

class RequestedColumnMatch(
    BaseModel
):
    concept: str

    dataset_id: str

    dataset_filename: str

    column: str

    analysis_kind: str

    match_score: int

    reasons: list[
        str
    ] = Field(
        default_factory=list
    )


class RequestedAnalysisResolution(
    BaseModel
):
    resolution_type: Literal[
        "ranking_metric",
        "time_series_parameters",
    ] = "ranking_metric"

    ranking_metric: (
        Literal[
            "revenue",
            "units",
            "transaction_count",
        ]
        | None
    ) = None

    time_granularity: (
        Literal[
            "day",
            "week",
            "month",
            "quarter",
            "year",
        ]
        | None
    ) = None

    moving_average_window: (
        int
        | None
    ) = Field(
        default=None,
        ge=1,
    )


class RequestedAnalysisPlan(
    BaseModel
):
    request_id: str

    request_text: str

    context_text: (
        str
        | None
    ) = None

    evidence_quote: str

    source_filename: str

    source_locator: str

    page_number: (
        int
        | None
    ) = None

    source_chunk_id: str

    evidence_unit_id: int

    kind: RequestedAnalysisKind

    status: RequestPlanningStatus

    resolution: (
        RequestedAnalysisResolution
        | None
    ) = None

    target_family: (
        str
        | None
    ) = None

    matched_columns: list[
        RequestedColumnMatch
    ] = Field(
        default_factory=list
    )

    required_dataset_ids: list[
        str
    ] = Field(
        default_factory=list
    )

    required_dataset_filenames: list[
        str
    ] = Field(
        default_factory=list
    )

    required_operations: list[
        str
    ] = Field(
        default_factory=list
    )

    reasons: list[
        str
    ] = Field(
        default_factory=list
    )

    blockers: list[
        str
    ] = Field(
        default_factory=list
    )


class RequestedAnalysisPlanReport(
    BaseModel
):
    status: Literal[
        "ready"
    ] = "ready"

    request_count: int

    ready_count: int

    blocked_count: int

    ambiguous_count: int

    requests: list[
        RequestedAnalysisPlan
    ]

    planner_notes: list[
        str
    ] = Field(
        default_factory=list
    )

    planner_rule_version: str = (
        "analytical_request_planner_v0.1"
    )


# ============================================================
# GLOBAL ANALYSIS PLAN
# ============================================================

class AnalysisPlanReport(
    BaseModel
):
    status: Literal[
        "ready"
    ] = "ready"

    objective: (
        str
        | None
    ) = None

    dataset_count: int

    total_rows: int

    requested_analyses: list[
        RequestedAnalysisPlan
    ] = Field(
        default_factory=list
    )

    recommended_analyses: list[
        AnalysisCandidate
    ]

    cross_dataset_opportunities: list[
        CrossDatasetOpportunity
    ]

    additional_data_suggestions: list[
        AdditionalDataSuggestion
    ]

    planner_notes: list[
        str
    ] = Field(
        default_factory=list
    )

    planner_rule_version: str = (
        "analysis_planner_v0.2"
    )