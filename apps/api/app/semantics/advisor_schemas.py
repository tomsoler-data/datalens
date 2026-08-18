from __future__ import annotations

from typing import (
    Literal,
)

from pydantic import (
    BaseModel,
    Field,
)

from app.semantics.schemas import (
    SemanticProfileComparison,
)


# ============================================================
# TYPES
# ============================================================

SemanticAdviceStatus = Literal[
    "annotated",
    "not_applicable",
    "insufficient_semantic_context",
]


SemanticAdviceDecision = Literal[
    "boost",
    "neutral",
    "penalize",
    "review",
]


# ============================================================
# CANDIDATE ADVICE
# ============================================================

class SemanticCandidateAdvice(
    BaseModel
):
    analysis_id: str

    title: str

    family: str

    status: SemanticAdviceStatus

    referenced_dataset_ids: list[
        str
    ] = Field(
        default_factory=list,
    )

    referenced_columns: list[
        str
    ] = Field(
        default_factory=list,
    )

    comparison: (
        SemanticProfileComparison
        |
        None
    ) = None

    semantic_score_delta: float = Field(
        default=0.0,
        ge=-25.0,
        le=15.0,
    )

    decision: SemanticAdviceDecision = (
        "neutral"
    )

    reasons: list[
        str
    ] = Field(
        default_factory=list,
    )

    semantic_rule_version: str = (
        "semantic_discovery_advisor_v0.1"
    )


# ============================================================
# ADVICE REPORT
# ============================================================

class SemanticDiscoveryAdviceReport(
    BaseModel
):
    candidate_count: int = Field(
        ge=0,
    )

    annotated_count: int = Field(
        ge=0,
    )

    boosted_count: int = Field(
        ge=0,
    )

    penalized_count: int = Field(
        ge=0,
    )

    review_count: int = Field(
        ge=0,
    )

    neutral_count: int = Field(
        ge=0,
    )

    not_applicable_count: int = Field(
        ge=0,
    )

    insufficient_context_count: int = Field(
        ge=0,
    )

    advice: list[
        SemanticCandidateAdvice
    ] = Field(
        default_factory=list,
    )

    semantic_rule_version: str = (
        "semantic_discovery_advisor_v0.1"
    )
