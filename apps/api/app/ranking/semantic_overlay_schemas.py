from __future__ import annotations

from typing import (
    Literal,
)

from pydantic import (
    BaseModel,
    Field,
)


# ============================================================
# TYPES
# ============================================================

SemanticOverlayDecision = Literal[
    "boost",
    "neutral",
    "penalize",
    "review",
    "none",
]


SemanticOverlayApplicationStatus = Literal[
    "applied",
    "neutral",
    "blocked",
    "no_advice",
]


# ============================================================
# FINDING
# ============================================================

class SemanticOverlayFinding(
    BaseModel
):
    analysis_id: str

    title: str

    family: str

    execution_status: str

    base_tier: str

    base_rank: int | None = None

    base_score: float = Field(
        ge=0.0,
        le=100.0,
    )

    semantic_decision: SemanticOverlayDecision

    semantic_advice_status: str | None = None

    raw_delta: float = Field(
        ge=-25.0,
        le=15.0,
    )

    applied_delta: float = Field(
        ge=-25.0,
        le=15.0,
    )

    semantic_score: float = Field(
        ge=0.0,
        le=100.0,
    )

    semantic_rank: int | None = Field(
        default=None,
        ge=1,
    )

    blocked: bool

    application_status: (
        SemanticOverlayApplicationStatus
    )

    reasons: list[
        str
    ] = Field(
        default_factory=list,
    )


# ============================================================
# REPORT
# ============================================================

class SemanticRankingOverlayReport(
    BaseModel
):
    source_ranking_version: str

    semantic_advisor_version: str

    finding_count: int = Field(
        ge=0,
    )

    non_blocked_count: int = Field(
        ge=0,
    )

    blocked_count: int = Field(
        ge=0,
    )

    changed_count: int = Field(
        ge=0,
    )

    boosted_applied_count: int = Field(
        ge=0,
    )

    penalized_applied_count: int = Field(
        ge=0,
    )

    review_applied_count: int = Field(
        ge=0,
    )

    findings: list[
        SemanticOverlayFinding
    ] = Field(
        default_factory=list,
    )

    semantic_rule_version: str = (
        "semantic_ranking_overlay_v0.1"
    )
