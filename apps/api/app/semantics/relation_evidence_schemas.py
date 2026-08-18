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

METRIC_RELATION_EVIDENCE_VERSION = (
    "metric_relation_evidence_v0.2"
)


# ============================================================
# TYPES
# ============================================================

MetricRelationType = Literal[
    "same_metric_different_state",
    "same_process_different_stage",
    "related_distinct_metric",
    "unrelated",
    "uncertain",
]


RelationConfidence = Literal[
    "low",
    "medium",
    "high",
]


# ============================================================
# PROFILE IDENTITY
# ============================================================

class MetricRelationProfileIdentity(
    BaseModel
):
    dataset_id: str

    column: str

    concept: str

    semantic_group: str

    domain: str

    variant: str

    measure_kind: str

    unit_kind: str

    quantity_dimension: str

    quantity_unit: str

    entity_role: str

    profile_confidence: str

    profile_source: str


# ============================================================
# QUANTITY EVIDENCE
# ============================================================

class MetricRelationQuantityEvidence(
    BaseModel
):
    same_known_quantity_dimension: bool

    compatible_quantity_dimensions: bool

    exact_same_known_unit: bool

    compatible_units: bool

    directly_subtractable: bool

    dimension_conflict: bool

    quantity_field_provenance_available: bool = (
        False
    )


# ============================================================
# COMPARATOR EVIDENCE
# ============================================================

class MetricRelationComparatorEvidence(
    BaseModel
):
    same_concept: bool

    same_concept_family: bool

    same_domain: bool

    distinct_variants: bool

    conceptual_proximity: str

    association_novelty: str

    redundancy_risk: str

    derived_gap_compatible: bool

    reasons: list[
        str
    ] = Field(
        default_factory=list,
    )


# ============================================================
# FAMILY EVIDENCE
# ============================================================

class MetricRelationFamilyEvidence(
    BaseModel
):
    available: bool

    same_quantity_family: bool | None = None

    relation_source: str | None = None

    left_family: str | None = None

    right_family: str | None = None

    left_family_source: str | None = None

    right_family_source: str | None = None

    left_family_confidence: str | None = None

    right_family_confidence: str | None = None

    left_signature: str

    right_signature: str

    signature_same: bool

    left_state: str

    right_state: str

    distinct_known_states: bool

    llm_same_family: bool | None = None

    reasons: list[
        str
    ] = Field(
        default_factory=list,
    )


# ============================================================
# EMBEDDING EVIDENCE
#
# Retrieval evidence only.
# Never analytical authorization.
# ============================================================

class MetricRelationEmbeddingEvidence(
    BaseModel
):
    candidate_retrieved: bool

    cosine_similarity: float | None = None

    left_to_right_rank: int | None = None

    right_to_left_rank: int | None = None

    mutual_retrieval: bool | None = None

    role: str = (
        "candidate_generation_only"
    )


# ============================================================
# INTERPRETATION
# ============================================================

class MetricRelationInterpretation(
    BaseModel
):
    proposed_relation: MetricRelationType

    confidence: RelationConfidence

    evidence_for_same_metric_different_state: list[
        str
    ] = Field(
        default_factory=list,
    )

    evidence_for_same_process_different_stage: list[
        str
    ] = Field(
        default_factory=list,
    )

    evidence_for_related_distinct_metric: list[
        str
    ] = Field(
        default_factory=list,
    )

    contradictory_evidence: list[
        str
    ] = Field(
        default_factory=list,
    )

    limitations: list[
        str
    ] = Field(
        default_factory=list,
    )


# ============================================================
# COMPLETE EVIDENCE OBJECT
# ============================================================

class MetricRelationEvidence(
    BaseModel
):
    left: MetricRelationProfileIdentity

    right: MetricRelationProfileIdentity

    quantity: MetricRelationQuantityEvidence

    comparator: MetricRelationComparatorEvidence

    family: MetricRelationFamilyEvidence

    embedding: MetricRelationEmbeddingEvidence

    interpretation: MetricRelationInterpretation

    analytical_authority: str = (
        "none_semantic_evidence_only"
    )

    relation_evidence_version: str = (
        METRIC_RELATION_EVIDENCE_VERSION
    )
