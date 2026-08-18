from typing import (
    Any,
    Literal,
)

from pydantic import (
    BaseModel,
    Field,
)


ExecutionStatus = Literal[
    "complete",
    "descriptive_only",
    "needs_information",
    "needs_specialized_method",
    "skipped",
    "failed",
]


class ExecutedAnalysis(
    BaseModel
):
    analysis_id: str

    dataset_id: str

    dataset_filename: str

    title: str

    family: str

    planned_readiness: str

    execution_status: ExecutionStatus

    chart_type: str

    summary: list[
        str
    ] = Field(
        default_factory=list
    )

    metrics: dict[
        str,
        Any,
    ] = Field(
        default_factory=dict
    )

    chart_data: list[
        dict[
            str,
            Any,
        ]
    ] = Field(
        default_factory=list
    )

    statistical_decision: (
        dict[
            str,
            Any,
        ]
        | None
    ) = None

    statistical_result: (
        dict[
            str,
            Any,
        ]
        | None
    ) = None

    visualization: (
        dict[
            str,
            Any,
        ]
        | None
    ) = None

    warnings: list[
        str
    ] = Field(
        default_factory=list
    )

    limitations: list[
        str
    ] = Field(
        default_factory=list
    )

    execution_rule_version: str = (
        "analysis_executor_v0.1"
    )


class AnalysisExecutionReport(
    BaseModel
):
    status: Literal[
        "ready"
    ] = "ready"

    candidate_count: int

    complete_count: int

    descriptive_only_count: int

    needs_information_count: int

    needs_specialized_method_count: int

    skipped_count: int

    failed_count: int

    results: list[
        ExecutedAnalysis
    ]

    executor_notes: list[
        str
    ] = Field(
        default_factory=list
    )

    executor_rule_version: str = (
        "analysis_plan_executor_v0.1"
    )