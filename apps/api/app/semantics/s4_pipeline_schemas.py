from __future__ import annotations

from pydantic import (
    BaseModel,
    Field,
)

from app.semantics.family_schemas import (
    DatasetQuantityFamilyReport,
)

from app.semantics.pipeline_schemas import (
    SemanticPreparationResult,
)


# ============================================================
# S4.1 PREPARATION RESULT
# ============================================================

class SemanticPreparationS4Result(
    BaseModel
):
    base_preparation: SemanticPreparationResult

    quantity_family_dataset_count: int = Field(
        ge=0,
    )

    quantity_family_eligible_column_count: int = Field(
        ge=0,
    )

    quantity_family_assignment_count: int = Field(
        ge=0,
    )

    quantity_family_llm_assignment_count: int = Field(
        ge=0,
    )

    quantity_family_fallback_assignment_count: int = Field(
        ge=0,
    )

    quantity_family_clustering_failure_count: int = Field(
        ge=0,
    )

    quantity_family_reports: list[
        DatasetQuantityFamilyReport
    ] = Field(
        default_factory=list,
    )

    semantic_rule_version: str = (
        "semantic_preparation_s4_1_v0.1"
    )
