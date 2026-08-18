from __future__ import annotations


from typing import (
    Any,
    Literal,
)


from pydantic import (
    BaseModel,
    Field,
)


from app.execution.schemas import (
    ExecutedAnalysis,
)

from app.planning.schemas import (
    RequestedAnalysisKind,
    RequestPlanningStatus,
)


# ============================================================
# STATUS
# ============================================================

RequestedExecutionStatus = Literal[
    "complete",
    "descriptive_only",
    "needs_information",
    "needs_specialized_method",
    "skipped",
    "failed",
    "not_executed",
    "not_supported_yet",
]


RequestedStatisticalMode = Literal[
    "exploratory",
    "confirmatory",
]


RequestedInferentialStatus = Literal[
    "executed",
    "not_selected",
    "not_applicable",
]


# ============================================================
# SINGLE REQUEST RESULT
# ============================================================

class RequestedAnalysisExecution(
    BaseModel
):
    request_id: str

    request_text: str

    kind: RequestedAnalysisKind

    plan_status: RequestPlanningStatus

    execution_status: RequestedExecutionStatus

    inferential_status: (
        RequestedInferentialStatus
        | None
    ) = None

    source_filename: str

    source_locator: str

    evidence_quote: str

    dataset_id: (
        str
        | None
    ) = None

    dataset_filename: (
        str
        | None
    ) = None

    analytical_grain: (
        str
        | None
    ) = None

    analysis_mode: (
        RequestedStatisticalMode
        | None
    ) = None

    variables: dict[
        str,
        str,
    ] = Field(
        default_factory=dict,
    )

    descriptive_statistics: dict[
        str,
        Any,
    ] = Field(
        default_factory=dict,
    )

    result: (
        ExecutedAnalysis
        | None
    ) = None

    warnings: list[
        str
    ] = Field(
        default_factory=list,
    )

    limitations: list[
        str
    ] = Field(
        default_factory=list,
    )

    executor_rule_version: str = (
        "requested_analysis_executor_v0.2"
    )


# ============================================================
# REPORT
# ============================================================

class RequestedAnalysisExecutionReport(
    BaseModel
):
    status: Literal[
        "ready"
    ] = "ready"

    request_count: int

    attempted_count: int

    complete_count: int

    descriptive_only_count: int

    needs_information_count: int

    needs_specialized_method_count: int

    skipped_count: int

    failed_count: int

    not_executed_count: int

    not_supported_yet_count: int

    inference_executed_count: int

    inference_abstained_count: int

    results: list[
        RequestedAnalysisExecution
    ]

    executor_notes: list[
        str
    ] = Field(
        default_factory=list,
    )

    executor_rule_version: str = (
        "requested_analysis_executor_v0.2"
    )