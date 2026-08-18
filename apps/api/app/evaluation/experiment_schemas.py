from __future__ import annotations

from datetime import (
    datetime,
)

from typing import (
    Literal,
)

from pydantic import (
    BaseModel,
    Field,
)

from app.evaluation.runner_schemas import (
    SemanticGlobalBenchmarkResult,
)


# ============================================================
# TYPES
# ============================================================

ExperimentPhase = Literal[
    "regression_baseline",
    "pre_adaptation_holdout",
    "post_adaptation_evaluation",
]


# ============================================================
# SYSTEM MANIFEST
# ============================================================

class SemanticSystemManifest(
    BaseModel
):
    system_label: str

    model_name: str

    component_versions: dict[
        str,
        str,
    ] = Field(
        default_factory=dict,
    )

    notes: list[
        str
    ] = Field(
        default_factory=list,
    )


# ============================================================
# EXPERIMENT SNAPSHOT
# ============================================================

class SemanticExperimentSnapshot(
    BaseModel
):
    experiment_id: str

    created_at: datetime

    phase: ExperimentPhase

    split: str

    benchmark_ids: list[
        str
    ] = Field(
        default_factory=list,
    )

    benchmark_versions: dict[
        str,
        str,
    ] = Field(
        default_factory=dict,
    )

    system: SemanticSystemManifest

    result: SemanticGlobalBenchmarkResult

    notes: list[
        str
    ] = Field(
        default_factory=list,
    )

    snapshot_rule_version: str = (
        "semantic_experiment_snapshot_v0.1"
    )
