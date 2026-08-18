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

UnifiedFindingTier = Literal[
    "key_finding",
    "supporting_finding",
    "supplementary",
    "blocked",
]


UnifiedSignalType = Literal[
    "association",
    "trend",
    "gap",
    "distribution_anomaly",
    "data_quality",
    "group_difference",
    "geographic_ranking",
    "categorical_association",
    "unknown",
]


# ============================================================
# RANKED ANALYSIS
# ============================================================

class UnifiedRankedAnalysis(
    BaseModel
):
    rank: int = Field(
        ge=0,
    )

    analysis_id: str

    title: str

    scope: str

    family: str

    execution_status: str

    tier: UnifiedFindingTier

    signal_type: UnifiedSignalType

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

    discovery_priority_score: float = Field(
        ge=0.0,
        le=100.0,
    )

    execution_confidence_score: float = Field(
        ge=0.0,
        le=100.0,
    )

    semantic_penalty: float = Field(
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

    dataset_ids: list[
        str
    ] = Field(
        default_factory=list,
    )

    datasets: list[
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

    metrics: dict[
        str,
        Any,
    ] = Field(
        default_factory=dict,
    )


# ============================================================
# UNIFIED RANKING REPORT
# ============================================================

class UnifiedRankingReport(
    BaseModel
):
    status: Literal[
        "ready"
    ] = "ready"

    ranked_count: int = Field(
        ge=0,
    )

    key_finding_count: int = Field(
        ge=0,
    )

    supporting_finding_count: int = Field(
        ge=0,
    )

    supplementary_count: int = Field(
        ge=0,
    )

    blocked_count: int = Field(
        ge=0,
    )

    findings: list[
        UnifiedRankedAnalysis
    ] = Field(
        default_factory=list,
    )

    ranking_notes: list[
        str
    ] = Field(
        default_factory=list,
    )

    ranking_rule_version: str = (
        "unified_ranker_v0.2"
    )
