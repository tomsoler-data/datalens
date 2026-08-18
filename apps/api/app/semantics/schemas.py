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

SemanticConfidence = Literal[
    "low",
    "medium",
    "high",
]


SemanticSource = Literal[
    "llm",
    "deterministic",
    "deterministic_fallback",
]


SemanticMeasureKind = Literal[
    "count",
    "rate",
    "percentage",
    "index",
    "currency",
    "duration",
    "category",
    "identifier",
    "datetime",
    "boolean",
    "text",
    "unknown",
]


SemanticUnitKind = Literal[
    "count",
    "percent",
    "proportion",
    "rate",
    "score",
    "currency",
    "duration",
    "year",
    "date",
    "category",
    "identifier",
    "boolean",
    "text",
    "unknown",
]


SemanticEntityRole = Literal[
    "measure",
    "dimension",
    "identifier",
    "time",
    "geography",
    "unknown",
]


# ============================================================
# QUANTITY SEMANTICS
#
# dimension:
#     What physical / mathematical quantity is represented?
#
# unit:
#     In which explicit unit is that quantity expressed?
#
# Example:
#
#     storage (GB)
#         dimension = data_size
#         unit      = gigabyte
#
#     queue time (seconds)
#         dimension = duration
#         unit      = second
# ============================================================

SemanticQuantityDimension = Literal[
    "count",
    "proportion",
    "rate",
    "currency",
    "duration",
    "data_size",
    "mass",
    "distance",
    "energy",
    "unknown",
]


SemanticQuantityUnit = Literal[
    "count",
    "percent",
    "proportion",
    "rate",
    "currency",

    "second",
    "minute",
    "hour",
    "day",
    "week",
    "month",
    "year",

    "byte",
    "kilobyte",
    "megabyte",
    "gigabyte",
    "terabyte",

    "gram",
    "kilogram",
    "tonne",

    "metre",
    "kilometre",

    "watt_hour",
    "kilowatt_hour",
    "megawatt_hour",

    "unknown",
]


ConceptualProximity = Literal[
    "low",
    "medium",
    "high",
]


AssociationNovelty = Literal[
    "low",
    "medium",
    "high",
]


RedundancyRisk = Literal[
    "low",
    "medium",
    "high",
]


# ============================================================
# LLM DRAFT
#
# The LLM remains responsible for semantic interpretation.
# Quantity semantics are intentionally NOT requested from the
# LLM yet. They are inferred deterministically afterwards.
# ============================================================

class ColumnSemanticDraft(
    BaseModel
):
    concept: str = Field(
        min_length=1,
        max_length=80,
    )

    domain: str = Field(
        min_length=1,
        max_length=80,
    )

    semantic_group: str = Field(
        min_length=1,
        max_length=80,
    )

    variant: str = Field(
        min_length=1,
        max_length=80,
    )

    measure_kind: SemanticMeasureKind

    unit_kind: SemanticUnitKind

    entity_role: SemanticEntityRole

    qualifiers: list[
        str
    ] = Field(
        default_factory=list,
        max_length=6,
    )

    confidence: SemanticConfidence


# ============================================================
# VALIDATED COLUMN PROFILE
# ============================================================

class ColumnSemanticProfile(
    BaseModel
):
    dataset_id: str

    filename: str

    column: str

    data_type: str

    concept: str

    domain: str

    semantic_group: str

    variant: str

    measure_kind: SemanticMeasureKind

    unit_kind: SemanticUnitKind

    quantity_dimension: SemanticQuantityDimension = (
        "unknown"
    )

    quantity_unit: SemanticQuantityUnit = (
        "unknown"
    )

    entity_role: SemanticEntityRole

    qualifiers: list[
        str
    ] = Field(
        default_factory=list,
    )

    confidence: SemanticConfidence

    source: SemanticSource

    quantity_rule_version: str = (
        "quantity_semantics_unset"
    )

    semantic_rule_version: str = (
        "semantic_data_profiler_v0.2"
    )


# ============================================================
# DATASET PROFILE
# ============================================================

class DatasetSemanticProfile(
    BaseModel
):
    dataset_id: str

    filename: str

    columns: list[
        ColumnSemanticProfile
    ] = Field(
        default_factory=list,
    )

    semantic_rule_version: str = (
        "semantic_data_profiler_v0.2"
    )


# ============================================================
# PROFILE COMPARISON
# ============================================================

class SemanticProfileComparison(
    BaseModel
):
    left_dataset_id: str

    left_column: str

    left_variant: str

    left_quantity_dimension: SemanticQuantityDimension = (
        "unknown"
    )

    left_quantity_unit: SemanticQuantityUnit = (
        "unknown"
    )

    right_dataset_id: str

    right_column: str

    right_variant: str

    right_quantity_dimension: SemanticQuantityDimension = (
        "unknown"
    )

    right_quantity_unit: SemanticQuantityUnit = (
        "unknown"
    )

    same_concept: bool

    same_concept_family: bool

    same_domain: bool

    compatible_units: bool

    distinct_variants: bool

    conceptual_proximity: ConceptualProximity

    association_novelty: AssociationNovelty

    redundancy_risk: RedundancyRisk

    derived_gap_compatible: bool

    reasons: list[
        str
    ] = Field(
        default_factory=list,
    )

    semantic_rule_version: str = (
        "semantic_profile_comparator_v0.3"
    )
