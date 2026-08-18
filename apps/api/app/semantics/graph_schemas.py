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

SemanticGraphConfidence = Literal[
    "low",
    "medium",
    "high",
]


SemanticGraphSource = Literal[
    "llm",
]


# ============================================================
# LLM RELATION DRAFT
#
# IMPORTANT:
#
# This schema intentionally contains NO node attributes such
# as state, measure_kind or unit_kind.
#
# The pair adjudicator is responsible only for RELATIONS.
# ============================================================

class SemanticPairRelationDraft(
    BaseModel
):
    same_domain_family: bool

    same_quantity_family: bool

    confidence: SemanticGraphConfidence

    reason: str = Field(
        min_length=1,
        max_length=500,
    )


# ============================================================
# VALIDATED GRAPH EDGE
# ============================================================

class SemanticRelationEdge(
    BaseModel
):
    dataset_id: str

    filename: str

    left_column: str

    right_column: str

    same_domain_family: bool

    same_quantity_family: bool

    confidence: SemanticGraphConfidence

    reason: str

    source: SemanticGraphSource = (
        "llm"
    )

    relation_rule_version: str = (
        "semantic_pair_adjudicator_v0.1"
    )


# ============================================================
# ABSTENTION
#
# Fail closed:
#
# if the model cannot produce a structurally valid relation,
# DataLens records an abstention instead of inventing an edge.
# ============================================================

class SemanticGraphAbstention(
    BaseModel
):
    dataset_id: str

    filename: str

    left_column: str

    right_column: str

    reason: str


# ============================================================
# GRAPH REPORT
# ============================================================

class SemanticGraphReport(
    BaseModel
):
    dataset_id: str

    filename: str

    candidate_pair_count: int = Field(
        ge=0,
    )

    adjudicated_pair_count: int = Field(
        ge=0,
    )

    abstention_count: int = Field(
        ge=0,
    )

    same_domain_edge_count: int = Field(
        ge=0,
    )

    same_quantity_edge_count: int = Field(
        ge=0,
    )

    edges: list[
        SemanticRelationEdge
    ] = Field(
        default_factory=list,
    )

    abstentions: list[
        SemanticGraphAbstention
    ] = Field(
        default_factory=list,
    )

    graph_rule_version: str = (
        "semantic_graph_v0.1"
    )
