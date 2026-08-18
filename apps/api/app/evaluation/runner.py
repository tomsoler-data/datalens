from __future__ import annotations

from collections.abc import (
    Callable,
)

from typing import (
    Any,
)

from app.evaluation.registry import (
    SemanticBenchmarkRegistry,
)

from app.evaluation.registry_schemas import (
    BenchmarkSplit,
    SemanticBenchmarkSuite,
)

from app.evaluation.runner_schemas import (
    SafetyDecisionSummary,
    SemanticBenchmarkSuiteResult,
    SemanticGlobalBenchmarkResult,
)

from app.evaluation.schemas import (
    BenchmarkAssertionResult,
)

from app.evaluation.semantic_benchmark import (
    evaluate_semantic_columns,
    evaluate_semantic_pairs,
)

from app.semantics import (
    normalize_dataset_semantics,
    profile_dataset_semantics,
)


# ============================================================
# TYPES
# ============================================================

BenchmarkDatasetProvider = Callable[
    [
        SemanticBenchmarkSuite,
    ],
    list[
        dict[
            str,
            Any,
        ]
    ],
]


# ============================================================
# SAFE RATIO
# ============================================================

def _safe_ratio(
    numerator: int,
    denominator: int,
) -> float | None:
    if (
        denominator
        ==
        0
    ):
        return None


    return round(
        numerator
        /
        denominator,
        6,
    )


# ============================================================
# BOOLEAN PARSING
# ============================================================

def _parse_boolean_value(
    value: str,
) -> bool | None:
    normalized = (
        str(
            value
        )
        .strip()
        .casefold()
    )


    if (
        normalized
        ==
        "true"
    ):
        return True


    if (
        normalized
        ==
        "false"
    ):
        return False


    return None


# ============================================================
# DIRECTION-AWARE SAFETY SUMMARY
# ============================================================

def build_safety_decision_summary(
    assertions: list[
        BenchmarkAssertionResult
    ],
) -> SafetyDecisionSummary:
    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0
    unclassified = 0


    for assertion in assertions:
        expected = (
            _parse_boolean_value(
                assertion.expected
            )
        )


        actual = (
            _parse_boolean_value(
                assertion.actual
            )
        )


        if (
            expected is None
            or
            actual is None
        ):
            unclassified += 1
            continue


        if (
            expected
            and
            actual
        ):
            true_positive += 1

        elif (
            not expected
            and
            actual
        ):
            false_positive += 1

        elif (
            not expected
            and
            not actual
        ):
            true_negative += 1

        elif (
            expected
            and
            not actual
        ):
            false_negative += 1


    classified_count = (
        true_positive
        +
        false_positive
        +
        true_negative
        +
        false_negative
    )


    correct_count = (
        true_positive
        +
        true_negative
    )


    accuracy = (
        _safe_ratio(
            correct_count,
            classified_count,
        )
    )


    precision = (
        _safe_ratio(
            true_positive,
            (
                true_positive
                +
                false_positive
            ),
        )
    )


    recall = (
        _safe_ratio(
            true_positive,
            (
                true_positive
                +
                false_negative
            ),
        )
    )


    specificity = (
        _safe_ratio(
            true_negative,
            (
                true_negative
                +
                false_positive
            ),
        )
    )


    if (
        precision is None
        or
        recall is None
        or
        (
            precision
            +
            recall
        )
        ==
        0
    ):
        f1 = None

    else:
        f1 = round(
            (
                2
                *
                precision
                *
                recall
            )
            /
            (
                precision
                +
                recall
            ),
            6,
        )


    return SafetyDecisionSummary(
        assertion_count=
            len(
                assertions
            ),

        true_positive_count=
            true_positive,

        false_positive_count=
            false_positive,

        true_negative_count=
            true_negative,

        false_negative_count=
            false_negative,

        unclassified_count=
            unclassified,

        accuracy=
            accuracy,

        precision=
            precision,

        recall=
            recall,

        specificity=
            specificity,

        f1=
            f1,
    )


# ============================================================
# SAFETY SUMMARY AGGREGATION
# ============================================================

def aggregate_safety_decision_summaries(
    summaries: list[
        SafetyDecisionSummary
    ],
) -> SafetyDecisionSummary:
    true_positive = sum(
        summary.true_positive_count
        for summary
        in summaries
    )


    false_positive = sum(
        summary.false_positive_count
        for summary
        in summaries
    )


    true_negative = sum(
        summary.true_negative_count
        for summary
        in summaries
    )


    false_negative = sum(
        summary.false_negative_count
        for summary
        in summaries
    )


    unclassified = sum(
        summary.unclassified_count
        for summary
        in summaries
    )


    assertion_count = sum(
        summary.assertion_count
        for summary
        in summaries
    )


    classified_count = (
        true_positive
        +
        false_positive
        +
        true_negative
        +
        false_negative
    )


    correct_count = (
        true_positive
        +
        true_negative
    )


    accuracy = (
        _safe_ratio(
            correct_count,
            classified_count,
        )
    )


    precision = (
        _safe_ratio(
            true_positive,
            (
                true_positive
                +
                false_positive
            ),
        )
    )


    recall = (
        _safe_ratio(
            true_positive,
            (
                true_positive
                +
                false_negative
            ),
        )
    )


    specificity = (
        _safe_ratio(
            true_negative,
            (
                true_negative
                +
                false_positive
            ),
        )
    )


    if (
        precision is None
        or
        recall is None
        or
        (
            precision
            +
            recall
        )
        ==
        0
    ):
        f1 = None

    else:
        f1 = round(
            (
                2
                *
                precision
                *
                recall
            )
            /
            (
                precision
                +
                recall
            ),
            6,
        )


    return SafetyDecisionSummary(
        assertion_count=
            assertion_count,

        true_positive_count=
            true_positive,

        false_positive_count=
            false_positive,

        true_negative_count=
            true_negative,

        false_negative_count=
            false_negative,

        unclassified_count=
            unclassified,

        accuracy=
            accuracy,

        precision=
            precision,

        recall=
            recall,

        specificity=
            specificity,

        f1=
            f1,
    )


# ============================================================
# RAW PROFILE BUILDER
# ============================================================

def build_raw_semantic_profiles(
    *,
    datasets: list[
        dict[
            str,
            Any,
        ]
    ],
):
    profiles = []


    for dataset in datasets:
        required = {
            "dataset_id",
            "filename",
            "dataframe",
        }


        missing = (
            required
            -
            set(
                dataset
            )
        )


        if missing:
            raise ValueError(
                "Benchmark dataset is missing "
                f"required fields: {sorted(missing)}"
            )


        profile = (
            profile_dataset_semantics(
                dataset_id=
                    str(
                        dataset[
                            "dataset_id"
                        ]
                    ),

                filename=
                    str(
                        dataset[
                            "filename"
                        ]
                    ),

                dataframe=
                    dataset[
                        "dataframe"
                    ],
            )
        )


        profiles.append(
            profile
        )


    return profiles


# ============================================================
# DATASET VALIDATION
# ============================================================

def validate_suite_datasets(
    *,
    suite: SemanticBenchmarkSuite,
    datasets: list[
        dict[
            str,
            Any,
        ]
    ],
) -> None:
    expected = {
        (
            specification.dataset_id,
            specification.filename,
        )

        for specification
        in suite.datasets
    }


    actual = {
        (
            str(
                dataset.get(
                    "dataset_id"
                )
            ),
            str(
                dataset.get(
                    "filename"
                )
            ),
        )

        for dataset
        in datasets
    }


    if (
        expected
        !=
        actual
    ):
        missing = (
            expected
            -
            actual
        )


        unexpected = (
            actual
            -
            expected
        )


        raise ValueError(
            "Dataset provider returned a dataset "
            "set that does not match the benchmark "
            f"suite {suite.benchmark_id}. "
            f"Missing: {sorted(missing)}. "
            f"Unexpected: {sorted(unexpected)}."
        )


# ============================================================
# SUITE RUNNER
# ============================================================

def run_semantic_benchmark_suite(
    *,
    suite: SemanticBenchmarkSuite,
    dataset_provider: BenchmarkDatasetProvider,
) -> SemanticBenchmarkSuiteResult:
    datasets = (
        dataset_provider(
            suite
        )
    )


    validate_suite_datasets(
        suite=
            suite,

        datasets=
            datasets,
    )


    raw_profiles = (
        build_raw_semantic_profiles(
            datasets=
                datasets,
        )
    )


    normalized_profiles = [
        normalize_dataset_semantics(
            profile
        )

        for profile
        in raw_profiles
    ]


    raw_columns = (
        evaluate_semantic_columns(
            dataset_profiles=
                raw_profiles,

            cases=
                suite.column_cases,

            benchmark_name=(
                f"{suite.benchmark_id}"
                ":raw:columns"
            ),
        )
    )


    normalized_columns = (
        evaluate_semantic_columns(
            dataset_profiles=
                normalized_profiles,

            cases=
                suite.column_cases,

            benchmark_name=(
                f"{suite.benchmark_id}"
                ":normalized:columns"
            ),
        )
    )


    raw_pairs = (
        evaluate_semantic_pairs(
            dataset_profiles=
                raw_profiles,

            cases=
                suite.pair_cases,

            benchmark_name=(
                f"{suite.benchmark_id}"
                ":raw:pairs"
            ),
        )
    )


    normalized_pairs = (
        evaluate_semantic_pairs(
            dataset_profiles=
                normalized_profiles,

            cases=
                suite.pair_cases,

            benchmark_name=(
                f"{suite.benchmark_id}"
                ":normalized:pairs"
            ),
        )
    )


    raw_correct_count = (
        raw_columns.correct_count
        +
        raw_pairs.correct_count
    )


    raw_assertion_count = (
        raw_columns.assertion_count
        +
        raw_pairs.assertion_count
    )


    normalized_correct_count = (
        normalized_columns.correct_count
        +
        normalized_pairs.correct_count
    )


    normalized_assertion_count = (
        normalized_columns.assertion_count
        +
        normalized_pairs.assertion_count
    )


    raw_accuracy = (
        raw_correct_count
        /
        raw_assertion_count
        if raw_assertion_count
        else 0.0
    )


    normalized_accuracy = (
        normalized_correct_count
        /
        normalized_assertion_count
        if normalized_assertion_count
        else 0.0
    )


    safety_fields = set(
        suite.safety_critical_fields
    )


    raw_safety_assertions = [
        assertion
        for assertion
        in raw_pairs.assertions
        if (
            assertion.field
            in safety_fields
        )
    ]


    normalized_safety_assertions = [
        assertion
        for assertion
        in normalized_pairs.assertions
        if (
            assertion.field
            in safety_fields
        )
    ]


    raw_safety_failures = [
        assertion
        for assertion
        in raw_safety_assertions
        if not assertion.correct
    ]


    normalized_safety_failures = [
        assertion
        for assertion
        in normalized_safety_assertions
        if not assertion.correct
    ]


    raw_safety_decisions = (
        build_safety_decision_summary(
            raw_safety_assertions
        )
    )


    normalized_safety_decisions = (
        build_safety_decision_summary(
            normalized_safety_assertions
        )
    )


    normalized_failures = [
        assertion
        for assertion
        in (
            normalized_columns.assertions
            +
            normalized_pairs.assertions
        )
        if not assertion.correct
    ]


    return SemanticBenchmarkSuiteResult(
        benchmark_id=
            suite.benchmark_id,

        benchmark_name=
            suite.name,

        domain=
            suite.domain,

        split=
            suite.split,

        raw_columns=
            raw_columns,

        normalized_columns=
            normalized_columns,

        raw_pairs=
            raw_pairs,

        normalized_pairs=
            normalized_pairs,

        raw_correct_count=
            raw_correct_count,

        raw_assertion_count=
            raw_assertion_count,

        normalized_correct_count=
            normalized_correct_count,

        normalized_assertion_count=
            normalized_assertion_count,

        raw_accuracy=
            round(
                raw_accuracy,
                6,
            ),

        normalized_accuracy=
            round(
                normalized_accuracy,
                6,
            ),

        raw_safety_assertion_count=
            len(
                raw_safety_assertions
            ),

        raw_safety_failure_count=
            len(
                raw_safety_failures
            ),

        normalized_safety_assertion_count=
            len(
                normalized_safety_assertions
            ),

        normalized_safety_failure_count=
            len(
                normalized_safety_failures
            ),

        raw_safety_failures=
            raw_safety_failures,

        normalized_safety_failures=
            normalized_safety_failures,

        raw_safety_decisions=
            raw_safety_decisions,

        normalized_safety_decisions=
            normalized_safety_decisions,

        normalized_failures=
            normalized_failures,
    )


# ============================================================
# GLOBAL RUNNER
# ============================================================

def run_semantic_benchmark_registry(
    *,
    registry: SemanticBenchmarkRegistry,
    dataset_provider: BenchmarkDatasetProvider,
    split: BenchmarkSplit = "regression",
) -> SemanticGlobalBenchmarkResult:
    suites = (
        registry.list_suites(
            split=
                split,
        )
    )


    if not suites:
        empty_safety = (
            SafetyDecisionSummary(
                assertion_count=
                    0,

                true_positive_count=
                    0,

                false_positive_count=
                    0,

                true_negative_count=
                    0,

                false_negative_count=
                    0,

                unclassified_count=
                    0,

                accuracy=
                    None,

                precision=
                    None,

                recall=
                    None,

                specificity=
                    None,

                f1=
                    None,
            )
        )


        return SemanticGlobalBenchmarkResult(
            split=
                split,

            suite_count=
                0,

            domain_count=
                0,

            domains=[],

            raw_correct_count=
                0,

            raw_assertion_count=
                0,

            normalized_correct_count=
                0,

            normalized_assertion_count=
                0,

            raw_micro_accuracy=
                0.0,

            normalized_micro_accuracy=
                0.0,

            micro_accuracy_delta=
                0.0,

            raw_macro_accuracy=
                0.0,

            normalized_macro_accuracy=
                0.0,

            macro_accuracy_delta=
                0.0,

            raw_safety_assertion_count=
                0,

            raw_safety_failure_count=
                0,

            normalized_safety_assertion_count=
                0,

            normalized_safety_failure_count=
                0,

            raw_safety_decisions=
                empty_safety,

            normalized_safety_decisions=
                empty_safety,

            normalized_failure_count=
                0,

            safety_gate_passed=
                True,

            regression_gate_passed=
                True,

            suites=[],

            safety_gate_rule_version=
                "analytical_safety_gate_v0.2",
        )


    results = [
        run_semantic_benchmark_suite(
            suite=
                suite,

            dataset_provider=
                dataset_provider,
        )

        for suite
        in suites
    ]


    raw_correct_count = sum(
        result.raw_correct_count
        for result
        in results
    )


    raw_assertion_count = sum(
        result.raw_assertion_count
        for result
        in results
    )


    normalized_correct_count = sum(
        result.normalized_correct_count
        for result
        in results
    )


    normalized_assertion_count = sum(
        result.normalized_assertion_count
        for result
        in results
    )


    raw_micro_accuracy = (
        raw_correct_count
        /
        raw_assertion_count
        if raw_assertion_count
        else 0.0
    )


    normalized_micro_accuracy = (
        normalized_correct_count
        /
        normalized_assertion_count
        if normalized_assertion_count
        else 0.0
    )


    raw_macro_accuracy = (
        sum(
            result.raw_accuracy
            for result
            in results
        )
        /
        len(
            results
        )
    )


    normalized_macro_accuracy = (
        sum(
            result.normalized_accuracy
            for result
            in results
        )
        /
        len(
            results
        )
    )


    raw_safety_assertion_count = sum(
        result.raw_safety_assertion_count
        for result
        in results
    )


    raw_safety_failure_count = sum(
        result.raw_safety_failure_count
        for result
        in results
    )


    normalized_safety_assertion_count = sum(
        result.normalized_safety_assertion_count
        for result
        in results
    )


    normalized_safety_failure_count = sum(
        result.normalized_safety_failure_count
        for result
        in results
    )


    raw_safety_decisions = (
        aggregate_safety_decision_summaries(
            [
                result.raw_safety_decisions

                for result
                in results

                if (
                    result.raw_safety_decisions
                    is not None
                )
            ]
        )
    )


    normalized_safety_decisions = (
        aggregate_safety_decision_summaries(
            [
                result.normalized_safety_decisions

                for result
                in results

                if (
                    result.normalized_safety_decisions
                    is not None
                )
            ]
        )
    )


    normalized_failure_count = sum(
        len(
            result.normalized_failures
        )
        for result
        in results
    )


    # ========================================================
    # ANALYTICAL SAFETY GATE V0.2
    #
    # Fail only when DataLens:
    #
    #   - allows an invalid analytical operation (FP), or
    #   - cannot classify a safety-critical assertion.
    #
    # Valid operations that are conservatively rejected (FN)
    # reduce capability / recall but are not unsafe
    # acceptances.
    # ========================================================

    safety_gate_passed = (
        normalized_safety_decisions
        .false_positive_count
        ==
        0
        and
        normalized_safety_decisions
        .unclassified_count
        ==
        0
    )


    regression_gate_passed = (
        normalized_failure_count
        ==
        0
    )


    domains = sorted(
        {
            result.domain
            for result
            in results
        }
    )


    return SemanticGlobalBenchmarkResult(
        split=
            split,

        suite_count=
            len(
                results
            ),

        domain_count=
            len(
                domains
            ),

        domains=
            domains,

        raw_correct_count=
            raw_correct_count,

        raw_assertion_count=
            raw_assertion_count,

        normalized_correct_count=
            normalized_correct_count,

        normalized_assertion_count=
            normalized_assertion_count,

        raw_micro_accuracy=
            round(
                raw_micro_accuracy,
                6,
            ),

        normalized_micro_accuracy=
            round(
                normalized_micro_accuracy,
                6,
            ),

        micro_accuracy_delta=
            round(
                (
                    normalized_micro_accuracy
                    -
                    raw_micro_accuracy
                ),
                6,
            ),

        raw_macro_accuracy=
            round(
                raw_macro_accuracy,
                6,
            ),

        normalized_macro_accuracy=
            round(
                normalized_macro_accuracy,
                6,
            ),

        macro_accuracy_delta=
            round(
                (
                    normalized_macro_accuracy
                    -
                    raw_macro_accuracy
                ),
                6,
            ),

        raw_safety_assertion_count=
            raw_safety_assertion_count,

        raw_safety_failure_count=
            raw_safety_failure_count,

        normalized_safety_assertion_count=
            normalized_safety_assertion_count,

        normalized_safety_failure_count=
            normalized_safety_failure_count,

        raw_safety_decisions=
            raw_safety_decisions,

        normalized_safety_decisions=
            normalized_safety_decisions,

        normalized_failure_count=
            normalized_failure_count,

        safety_gate_passed=
            safety_gate_passed,

        regression_gate_passed=
            regression_gate_passed,

        suites=
            results,

        safety_gate_rule_version=
            "analytical_safety_gate_v0.2",
    )
