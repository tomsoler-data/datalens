from __future__ import annotations

from pydantic import (
    BaseModel,
    Field,
)


# ============================================================
# BINARY CLASSIFICATION METRICS
# ============================================================

class BinaryClassificationMetrics(
    BaseModel
):
    sample_count: int = Field(
        ge=0,
    )

    true_positive: int = Field(
        ge=0,
    )

    false_positive: int = Field(
        ge=0,
    )

    true_negative: int = Field(
        ge=0,
    )

    false_negative: int = Field(
        ge=0,
    )

    accuracy: float = Field(
        ge=0.0,
        le=1.0,
    )

    precision: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    recall: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    specificity: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    f1: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    false_positive_rate: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    false_negative_rate: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )


# ============================================================
# COLUMN EXPECTATION
# ============================================================

class SemanticFieldExpectation(
    BaseModel
):
    field: str

    accepted_values: list[
        str
    ] = Field(
        min_length=1,
    )


class SemanticColumnBenchmarkCase(
    BaseModel
):
    case_id: str

    dataset_id: str

    column: str

    expectations: list[
        SemanticFieldExpectation
    ] = Field(
        min_length=1,
    )


# ============================================================
# PAIR EXPECTATION
# ============================================================

class SemanticPairBenchmarkCase(
    BaseModel
):
    case_id: str

    left_dataset_id: str

    right_dataset_id: str

    left_column: str

    right_column: str

    same_concept: bool | None = None

    same_concept_family: bool | None = None

    same_domain: bool | None = None

    distinct_variants: bool | None = None

    compatible_units: bool | None = None

    derived_gap_compatible: bool | None = None


# ============================================================
# ASSERTION RESULT
# ============================================================

class BenchmarkAssertionResult(
    BaseModel
):
    case_id: str

    dataset_id: str

    subject: str

    field: str

    expected: str

    actual: str

    correct: bool


# ============================================================
# BENCHMARK SUMMARY
# ============================================================

class SemanticBenchmarkSummary(
    BaseModel
):
    benchmark_name: str

    case_count: int = Field(
        ge=0,
    )

    assertion_count: int = Field(
        ge=0,
    )

    correct_count: int = Field(
        ge=0,
    )

    incorrect_count: int = Field(
        ge=0,
    )

    accuracy: float = Field(
        ge=0.0,
        le=1.0,
    )

    field_accuracy: dict[
        str,
        float,
    ] = Field(
        default_factory=dict,
    )

    assertions: list[
        BenchmarkAssertionResult
    ] = Field(
        default_factory=list,
    )

    benchmark_rule_version: str = (
        "semantic_benchmark_v0.2"
    )


# ============================================================
# VERSION COMPARISON
# ============================================================

class BenchmarkVersionComparison(
    BaseModel
):
    baseline_name: str

    candidate_name: str

    baseline_accuracy: float = Field(
        ge=0.0,
        le=1.0,
    )

    candidate_accuracy: float = Field(
        ge=0.0,
        le=1.0,
    )

    absolute_accuracy_delta: float

    improved_assertion_count: int = Field(
        ge=0,
    )

    regressed_assertion_count: int = Field(
        ge=0,
    )

    unchanged_assertion_count: int = Field(
        ge=0,
    )

    regression_free: bool

    comparison_rule_version: str = (
        "benchmark_comparison_v0.1"
    )
