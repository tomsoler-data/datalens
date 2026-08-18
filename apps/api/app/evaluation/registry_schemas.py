from __future__ import annotations

from typing import (
    Literal,
)

from pydantic import (
    BaseModel,
    Field,
)

from app.evaluation.schemas import (
    SemanticColumnBenchmarkCase,
    SemanticPairBenchmarkCase,
)


# ============================================================
# TYPES
# ============================================================

BenchmarkSplit = Literal[
    "development",
    "regression",
    "holdout",
]


# ============================================================
# DATASET SPECIFICATION
# ============================================================

class BenchmarkDatasetSpec(
    BaseModel
):
    dataset_id: str

    filename: str


# ============================================================
# SEMANTIC BENCHMARK SUITE
# ============================================================

class SemanticBenchmarkSuite(
    BaseModel
):
    benchmark_id: str

    name: str

    domain: str

    split: BenchmarkSplit

    description: str

    datasets: list[
        BenchmarkDatasetSpec
    ] = Field(
        min_length=1,
    )

    column_cases: list[
        SemanticColumnBenchmarkCase
    ] = Field(
        default_factory=list,
    )

    pair_cases: list[
        SemanticPairBenchmarkCase
    ] = Field(
        default_factory=list,
    )

    safety_critical_fields: list[
        str
    ] = Field(
        default_factory=list,
    )

    tags: list[
        str
    ] = Field(
        default_factory=list,
    )

    benchmark_version: str


    @property
    def dataset_id(
        self,
    ) -> str:
        if (
            len(
                self.datasets
            )
            !=
            1
        ):
            raise ValueError(
                "dataset_id is only available for "
                "single-dataset benchmark suites."
            )


        return (
            self.datasets[
                0
            ].dataset_id
        )


    @property
    def filename(
        self,
    ) -> str:
        if (
            len(
                self.datasets
            )
            !=
            1
        ):
            raise ValueError(
                "filename is only available for "
                "single-dataset benchmark suites."
            )


        return (
            self.datasets[
                0
            ].filename
        )


# ============================================================
# REGISTRY SNAPSHOT
# ============================================================

class BenchmarkRegistrySnapshot(
    BaseModel
):
    suite_count: int = Field(
        ge=0,
    )

    development_count: int = Field(
        ge=0,
    )

    regression_count: int = Field(
        ge=0,
    )

    holdout_count: int = Field(
        ge=0,
    )

    dataset_count: int = Field(
        ge=0,
    )

    domains: list[
        str
    ] = Field(
        default_factory=list,
    )

    benchmark_ids: list[
        str
    ] = Field(
        default_factory=list,
    )

    registry_rule_version: str = (
        "semantic_benchmark_registry_v0.2"
    )
