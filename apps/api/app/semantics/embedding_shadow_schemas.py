from __future__ import annotations

from typing import (
    Literal,
)

from pydantic import (
    BaseModel,
    Field,
)


# ============================================================
# VERSION
# ============================================================

EMBEDDING_DISCOVERY_SHADOW_AUDIT_VERSION = (
    "embedding_discovery_shadow_audit_v0.1"
)


# ============================================================
# TYPES
# ============================================================

ShadowStage = Literal[
    "unvalidated",
    "validated",
]


ShadowClassification = Literal[
    "semantic_pair_in_scope",
    "generic_pair_observation_only",
    "out_of_scope_technical_pair",
    "out_of_scope_univariate",
    "out_of_scope_dataset_level",
    "out_of_scope_cross_dataset",
    "out_of_scope_contract_mismatch",
    "out_of_scope_unclassified_family",
]


# ============================================================
# DATASET GRAPH SUMMARY
# ============================================================

class EmbeddingShadowDatasetSummary(
    BaseModel
):
    dataset_id: str

    column_count: int = Field(
        ge=0,
    )

    all_possible_pair_count: int = Field(
        ge=0,
    )

    embedding_candidate_pair_count: int = Field(
        ge=0,
    )

    pair_reduction_ratio: float

    top_k: int = Field(
        ge=1,
    )


# ============================================================
# CANDIDATE AUDIT
# ============================================================

class EmbeddingShadowCandidateAudit(
    BaseModel
):
    stage: ShadowStage

    analysis_id: str

    family: str

    scope: str

    classification: ShadowClassification

    dataset_id: str | None = None

    left_column: str | None = None

    right_column: str | None = None

    pair_in_embedding_graph: bool | None = None

    cosine_similarity: float | None = None

    left_to_right_rank: int | None = None

    right_to_left_rank: int | None = None

    mutual_retrieval: bool | None = None


# ============================================================
# STAGE SUMMARY
# ============================================================

class EmbeddingShadowStageSummary(
    BaseModel
):
    stage: ShadowStage

    candidate_count: int = Field(
        ge=0,
    )

    semantic_pair_in_scope_count: int = Field(
        ge=0,
    )

    semantic_pair_covered_count: int = Field(
        ge=0,
    )

    semantic_pair_coverage: float | None = None

    generic_pair_observation_count: int = Field(
        ge=0,
    )

    generic_pair_covered_count: int = Field(
        ge=0,
    )

    generic_pair_coverage: float | None = None

    out_of_scope_count: int = Field(
        ge=0,
    )

    classification_counts: dict[
        str,
        int,
    ] = Field(
        default_factory=dict,
    )


# ============================================================
# COMPLETE AUDIT
# ============================================================

class EmbeddingDiscoveryShadowAuditResult(
    BaseModel
):
    embedding_model: str

    top_k: int = Field(
        ge=1,
    )

    dataset_summaries: list[
        EmbeddingShadowDatasetSummary
    ] = Field(
        default_factory=list,
    )

    unvalidated_summary: EmbeddingShadowStageSummary

    validated_summary: EmbeddingShadowStageSummary

    candidate_audits: list[
        EmbeddingShadowCandidateAudit
    ] = Field(
        default_factory=list,
    )

    semantic_pair_families: list[
        str
    ] = Field(
        default_factory=list,
    )

    generic_pair_observation_families: list[
        str
    ] = Field(
        default_factory=list,
    )

    influence_mode: str = (
        "shadow_no_discovery_output_change"
    )

    safety_role: str = (
        "none_observation_only"
    )

    audit_version: str = (
        EMBEDDING_DISCOVERY_SHADOW_AUDIT_VERSION
    )
