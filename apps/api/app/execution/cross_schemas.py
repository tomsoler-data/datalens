from typing import (
    Any,
    Literal,
)

from pydantic import (
    BaseModel,
    Field,
)


CrossExecutionStatus = Literal[
    "complete",
    "descriptive_only",
    "requires_alignment",
    "needs_specialized_method",
    "skipped",
    "failed",
]


class CrossDatasetExecutedAnalysis(
    BaseModel
):
    analysis_id: str

    title: str

    family: str

    dataset_ids: list[
        str
    ]

    datasets: list[
        str
    ]

    execution_status: CrossExecutionStatus

    relationship_status: (
        str
        | None
    ) = None

    relationship_score: (
        float
        | None
    ) = None

    join_safety: str

    join_keys: dict[
        str,
        list[
            str
        ],
    ] = Field(
        default_factory=dict,
    )

    rows_before: dict[
        str,
        int,
    ] = Field(
        default_factory=dict,
    )

    rows_after_alignment: dict[
        str,
        int,
    ] = Field(
        default_factory=dict,
    )

    joined_rows: int = 0

    matched_key_count: int = 0

    left_key_coverage: float = 0.0

    right_key_coverage: float = 0.0

    alignment_actions: list[
        str
    ] = Field(
        default_factory=list,
    )

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
        "cross_dataset_executor_v0.1"
    )


class CrossDatasetExecutionReport(
    BaseModel
):
    status: Literal[
        "ready"
    ] = "ready"

    candidate_count: int

    complete_count: int

    descriptive_only_count: int

    requires_alignment_count: int

    needs_specialized_method_count: int

    skipped_count: int

    failed_count: int

    results: list[
        CrossDatasetExecutedAnalysis
    ]

    executor_notes: list[
        str
    ] = Field(
        default_factory=list,
    )

    executor_rule_version: str = (
        "cross_dataset_executor_v0.1"
    )