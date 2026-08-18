from typing import (
    Any,
    Literal,
)

from pydantic import (
    BaseModel,
    Field,
)


AnalysisScope = Literal[
    "single_dataset",
    "cross_dataset",
]


DiscoveryReadiness = Literal[
    "executable_now",
    "planned",
    "requires_alignment",
]


RelationshipStatus = Literal[
    "validated",
    "partial",
    "requires_alignment",
]


class DiscoveredVariable(
    BaseModel
):
    dataset_id: str

    dataset_filename: str

    column: str

    role: str

    analysis_kind: str

    semantic_role: str

    concepts: list[
        str
    ] = Field(
        default_factory=list
    )


class DiscoveredAnalysis(
    BaseModel
):
    analysis_id: str

    scope: AnalysisScope

    family: str

    title: str

    priority_score: float = Field(
        ge=0.0,
        le=100.0,
    )

    readiness: DiscoveryReadiness

    datasets: list[
        str
    ]

    dataset_ids: list[
        str
    ]

    variables: list[
        DiscoveredVariable
    ] = Field(
        default_factory=list
    )

    chart_type: str

    execution_strategy: str

    why_interesting: list[
        str
    ] = Field(
        default_factory=list
    )

    limitations: list[
        str
    ] = Field(
        default_factory=list
    )

    relationship_status: (
        RelationshipStatus
        | None
    ) = None

    relationship_score: (
        float
        | None
    ) = None

    join_keys: dict[
        str,
        list[
            str
        ],
    ] = Field(
        default_factory=dict
    )

    observed_signals: dict[
        str,
        Any,
    ] = Field(
        default_factory=dict
    )

    redundancy_key: str


class RelationshipSummary(
    BaseModel
):
    left_dataset: str

    right_dataset: str

    relationship_mode: str

    cardinality: str

    score: float

    left_match_rate: float

    right_match_rate: float

    overlap_rate: float

    left_columns: list[
        str
    ]

    right_columns: list[
        str
    ]

    key_roles: list[
        str
    ]

    usable_for_analysis: bool

    requires_grain_alignment: bool

    warnings: list[
        str
    ] = Field(
        default_factory=list
    )


class AnalysisDiscoveryReport(
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

    candidate_count: int

    single_dataset_candidate_count: int

    cross_dataset_candidate_count: int

    candidates: list[
        DiscoveredAnalysis
    ]

    relationships: list[
        RelationshipSummary
    ]

    discovery_notes: list[
        str
    ] = Field(
        default_factory=list
    )

    discovery_rule_version: str = (
        "analysis_discovery_v0.2"
    )