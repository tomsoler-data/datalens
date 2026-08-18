from typing import (
    Any,
    Literal,
)

from pydantic import (
    BaseModel,
    Field,
)


SingleExecutionStatus = Literal[
    "complete",
    "descriptive_only",
    "needs_specialized_method",
    "skipped",
    "failed",
]


class SingleDatasetExecutedAnalysis(
    BaseModel
):
    analysis_id: str

    title: str

    family: str

    dataset_id: str

    dataset: str

    execution_status: SingleExecutionStatus

    variables: list[
        str
    ] = Field(
        default_factory=list,
    )

    valid_observations: int = 0

    summary: list[
        str
    ] = Field(
        default_factory=list,
    )

    metrics: dict[
        str,
        Any,
    ] = Field(
        default_factory=dict,
    )

    chart_type: str | None = None

    chart_data: list[
        dict[
            str,
            Any,
        ]
    ] = Field(
        default_factory=list,
    )

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

    execution_rule_version: str = (
        "single_dataset_executor_v0.1"
    )


class SingleDatasetExecutionReport(
    BaseModel
):
    status: Literal[
        "ready"
    ] = "ready"

    candidate_count: int

    complete_count: int

    descriptive_only_count: int

    needs_specialized_method_count: int

    skipped_count: int

    failed_count: int

    results: list[
        SingleDatasetExecutedAnalysis
    ]

    executor_notes: list[
        str
    ] = Field(
        default_factory=list,
    )

    executor_rule_version: str = (
        "single_dataset_executor_v0.1"
    )