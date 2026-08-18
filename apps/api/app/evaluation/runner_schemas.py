from __future__ import annotations

from pydantic import (
    BaseModel,
    Field,
)

from app.evaluation.schemas import (
    BenchmarkAssertionResult,
    SemanticBenchmarkSummary,
)


# ============================================================
# SAFETY DECISION SUMMARY
#
# Positive class:
#     the analytical operation is allowed / compatible.
#
# Therefore:
#
# FP = invalid operation incorrectly allowed
#      -> analytical safety risk
#
# FN = valid operation incorrectly rejected
#      -> capability / recall loss
# ============================================================

class SafetyDecisionSummary(
    BaseModel
):
    assertion_count: int = Field(
        ge=0,
    )

    true_positive_count: int = Field(
        ge=0,
    )

    false_positive_count: int = Field(
        ge=0,
    )

    true_negative_count: int = Field(
        ge=0,
    )

    false_negative_count: int = Field(
        ge=0,
    )

    unclassified_count: int = Field(
        ge=0,
    )

    accuracy: float | None = Field(
        default=None,
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

    decision_rule_version: str = (
        "analytical_safety_decision_v0.2"
    )


# ============================================================
# SUITE RESULT
# ============================================================

class SemanticBenchmarkSuiteResult(
    BaseModel
):
    benchmark_id: str

    benchmark_name: str

    domain: str

    split: str

    raw_columns: SemanticBenchmarkSummary

    normalized_columns: SemanticBenchmarkSummary

    raw_pairs: SemanticBenchmarkSummary

    normalized_pairs: SemanticBenchmarkSummary

    raw_correct_count: int = Field(
        ge=0,
    )

    raw_assertion_count: int = Field(
        ge=0,
    )

    normalized_correct_count: int = Field(
        ge=0,
    )

    normalized_assertion_count: int = Field(
        ge=0,
    )

    raw_accuracy: float = Field(
        ge=0.0,
        le=1.0,
    )

    normalized_accuracy: float = Field(
        ge=0.0,
        le=1.0,
    )

    # --------------------------------------------------------
    # Legacy safety-error counts
    #
    # Kept for backward compatibility.
    # They count every incorrect assertion on a safety field,
    # regardless of direction.
    # --------------------------------------------------------

    raw_safety_assertion_count: int = Field(
        ge=0,
    )

    raw_safety_failure_count: int = Field(
        ge=0,
    )

    normalized_safety_assertion_count: int = Field(
        ge=0,
    )

    normalized_safety_failure_count: int = Field(
        ge=0,
    )

    raw_safety_failures: list[
        BenchmarkAssertionResult
    ] = Field(
        default_factory=list,
    )

    normalized_safety_failures: list[
        BenchmarkAssertionResult
    ] = Field(
        default_factory=list,
    )

    # --------------------------------------------------------
    # Direction-aware safety evaluation.
    #
    # Optional so historical v0.1 experiment snapshots can
    # still be loaded without being rewritten.
    # --------------------------------------------------------

    raw_safety_decisions: (
        SafetyDecisionSummary
        |
        None
    ) = None

    normalized_safety_decisions: (
        SafetyDecisionSummary
        |
        None
    ) = None

    normalized_failures: list[
        BenchmarkAssertionResult
    ] = Field(
        default_factory=list,
    )


# ============================================================
# GLOBAL RESULT
# ============================================================

class SemanticGlobalBenchmarkResult(
    BaseModel
):
    split: str

    suite_count: int = Field(
        ge=0,
    )

    domain_count: int = Field(
        ge=0,
    )

    domains: list[
        str
    ] = Field(
        default_factory=list,
    )

    raw_correct_count: int = Field(
        ge=0,
    )

    raw_assertion_count: int = Field(
        ge=0,
    )

    normalized_correct_count: int = Field(
        ge=0,
    )

    normalized_assertion_count: int = Field(
        ge=0,
    )

    raw_micro_accuracy: float = Field(
        ge=0.0,
        le=1.0,
    )

    normalized_micro_accuracy: float = Field(
        ge=0.0,
        le=1.0,
    )

    micro_accuracy_delta: float

    raw_macro_accuracy: float = Field(
        ge=0.0,
        le=1.0,
    )

    normalized_macro_accuracy: float = Field(
        ge=0.0,
        le=1.0,
    )

    macro_accuracy_delta: float

    # --------------------------------------------------------
    # Legacy counts
    # --------------------------------------------------------

    raw_safety_assertion_count: int = Field(
        ge=0,
    )

    raw_safety_failure_count: int = Field(
        ge=0,
    )

    normalized_safety_assertion_count: int = Field(
        ge=0,
    )

    normalized_safety_failure_count: int = Field(
        ge=0,
    )

    # --------------------------------------------------------
    # Direction-aware metrics
    # --------------------------------------------------------

    raw_safety_decisions: (
        SafetyDecisionSummary
        |
        None
    ) = None

    normalized_safety_decisions: (
        SafetyDecisionSummary
        |
        None
    ) = None

    normalized_failure_count: int = Field(
        ge=0,
    )

    safety_gate_passed: bool

    regression_gate_passed: bool

    suites: list[
        SemanticBenchmarkSuiteResult
    ] = Field(
        default_factory=list,
    )

    # Historical v0.1 snapshots already contain their own
    # stored runner_rule_version and safety_gate_passed value.
    safety_gate_rule_version: (
        str
        |
        None
    ) = None

    runner_rule_version: str = (
        "semantic_global_benchmark_runner_v0.2"
    )
