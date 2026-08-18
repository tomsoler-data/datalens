from __future__ import annotations

from typing import (
    Literal,
)

from pydantic import (
    BaseModel,
    Field,
)

from app.semantics.schemas import (
    SemanticConfidence,
)


# ============================================================
# TYPES
# ============================================================

QuantityFamilyAssignmentSource = Literal[
    "llm",
    "deterministic_fallback",
]


QuantityFamilyRelationSource = Literal[
    "dimension_veto",
    "llm_plus_signature",
    "state_abstracted_signature",
    "llm_cluster",
    "no_positive_evidence",
    "unavailable",
]


# ============================================================
# LLM DRAFT
# ============================================================

class QuantityFamilyAssignmentDraft(
    BaseModel
):
    column: str = Field(
        min_length=1,
    )

    quantity_family: str = Field(
        min_length=1,
        max_length=80,
    )

    confidence: SemanticConfidence

    reason: str = Field(
        min_length=1,
        max_length=300,
    )


class QuantityFamilyClusteringDraft(
    BaseModel
):
    assignments: list[
        QuantityFamilyAssignmentDraft
    ] = Field(
        default_factory=list,
    )


# ============================================================
# VALIDATED ASSIGNMENT
# ============================================================

class QuantityFamilyAssignment(
    BaseModel
):
    dataset_id: str

    filename: str

    column: str

    quantity_family: str

    state: str

    state_abstracted_signature: str

    quantity_dimension: str

    quantity_unit: str

    confidence: SemanticConfidence

    source: QuantityFamilyAssignmentSource

    reason: str

    quantity_family_rule_version: str = (
        "quantity_family_clustering_v0.2"
    )


# ============================================================
# DATASET REPORT
# ============================================================

class DatasetQuantityFamilyReport(
    BaseModel
):
    dataset_id: str

    filename: str

    eligible_column_count: int = Field(
        ge=0,
    )

    assignment_count: int = Field(
        ge=0,
    )

    family_count: int = Field(
        ge=0,
    )

    llm_assignment_count: int = Field(
        ge=0,
    )

    fallback_assignment_count: int = Field(
        ge=0,
    )

    clustering_succeeded: bool

    model: str

    assignments: list[
        QuantityFamilyAssignment
    ] = Field(
        default_factory=list,
    )

    quantity_family_rule_version: str = (
        "quantity_family_clustering_v0.2"
    )


# ============================================================
# HYBRID RELATION DECISION
# ============================================================

class QuantityFamilyRelationDecision(
    BaseModel
):
    left_dataset_id: str

    left_column: str

    right_dataset_id: str

    right_column: str

    same_quantity_family: bool

    source: QuantityFamilyRelationSource

    left_family: str

    right_family: str

    left_signature: str

    right_signature: str

    left_state: str

    right_state: str

    left_quantity_dimension: str

    right_quantity_dimension: str

    llm_same_family: bool

    signature_same: bool

    distinct_known_states: bool

    dimension_conflict: bool

    reasons: list[
        str
    ] = Field(
        default_factory=list,
    )

    reconciliation_rule_version: str = (
        "hybrid_quantity_family_reconciler_v0.1"
    )
