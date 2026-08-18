from typing import (
    Any,
    Literal,
)

from pydantic import (
    BaseModel,
    Field,
)


FindingTier = Literal[
    "key_finding",
    "supporting_finding",
    "supplementary",
    "blocked",
]


AssociationDirection = Literal[
    "positive",
    "negative",
    "mixed",
    "unknown",
]


AssociationStrength = Literal[
    "negligible",
    "weak",
    "moderate",
    "moderately_strong",
    "strong",
    "unknown",
]


class RankedAnalysis(
    BaseModel
):
    rank: int

    analysis_id: str

    title: str

    scope: str

    family: str

    execution_status: str

    interestingness_score: float = Field(
        ge=0.0,
        le=100.0,
    )

    tier: FindingTier

    direction: AssociationDirection

    association_strength: AssociationStrength

    effect_score: float = Field(
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

    robustness_score: float = Field(
        ge=0.0,
        le=100.0,
    )

    relationship_score: (
        float
        | None
    ) = None

    sample_size: int = 0

    period_count: int = 0

    consistency_ratio: (
        float
        | None
    ) = None

    importance_reasons: list[
        str
    ] = Field(
        default_factory=list,
    )

    caveats: list[
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


class AnalysisRankingReport(
    BaseModel
):
    status: Literal[
        "ready"
    ] = "ready"

    ranked_count: int

    key_finding_count: int

    supporting_finding_count: int

    supplementary_count: int

    blocked_count: int

    findings: list[
        RankedAnalysis
    ]

    ranking_notes: list[
        str
    ] = Field(
        default_factory=list,
    )

    ranking_rule_version: str = (
        "interestingness_ranker_v0.1"
    )