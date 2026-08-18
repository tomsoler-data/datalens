from __future__ import annotations

from pydantic import (
    BaseModel,
    Field,
)

from app.semantics.schemas import (
    DatasetSemanticProfile,
)


# ============================================================
# NORMALIZATION CHANGE
# ============================================================

class SemanticNormalizationChange(
    BaseModel
):
    dataset_id: str

    filename: str

    column: str

    field: str

    before: str

    after: str

    rule: str


# ============================================================
# DATASET AUDIT
# ============================================================

class SemanticDatasetNormalizationAudit(
    BaseModel
):
    dataset_id: str

    filename: str

    column_count: int = Field(
        ge=0,
    )

    changed_column_count: int = Field(
        ge=0,
    )

    change_count: int = Field(
        ge=0,
    )

    normalization_applied: bool

    changes: list[
        SemanticNormalizationChange
    ] = Field(
        default_factory=list,
    )

    normalization_rule_version: str = (
        "deterministic_semantic_normalizer_v0.1"
    )


# ============================================================
# SEMANTIC PIPELINE RESULT
# ============================================================

class SemanticPreparationResult(
    BaseModel
):
    dataset_count: int = Field(
        ge=0,
    )

    column_count: int = Field(
        ge=0,
    )

    normalized_dataset_count: int = Field(
        ge=0,
    )

    normalized_column_count: int = Field(
        ge=0,
    )

    normalization_change_count: int = Field(
        ge=0,
    )

    profiles: list[
        DatasetSemanticProfile
    ] = Field(
        default_factory=list,
    )

    audits: list[
        SemanticDatasetNormalizationAudit
    ] = Field(
        default_factory=list,
    )

    semantic_rule_version: str = (
        "semantic_preparation_pipeline_v0.1"
    )
