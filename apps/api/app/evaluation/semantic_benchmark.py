from __future__ import annotations

from collections import (
    defaultdict,
)

from typing import (
    Any,
)

from app.evaluation.schemas import (
    BenchmarkAssertionResult,
    BenchmarkVersionComparison,
    SemanticBenchmarkSummary,
    SemanticColumnBenchmarkCase,
    SemanticPairBenchmarkCase,
)

from app.semantics import (
    compare_semantic_profiles,
)


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_expected_value(
    value: Any,
) -> str:
    if isinstance(
        value,
        bool,
    ):
        return (
            "true"
            if value
            else "false"
        )


    if value is None:
        return "none"


    return str(
        value
    ).strip().casefold()


# ============================================================
# PROFILE INDEX
# ============================================================

def build_profile_index(
    dataset_profiles: list[
        Any
    ],
) -> dict[
    tuple[
        str,
        str,
    ],
    Any,
]:
    index = {}


    for dataset_profile in (
        dataset_profiles
    ):
        dataset_id = str(
            dataset_profile.dataset_id
        )


        for profile in (
            dataset_profile.columns
        ):
            index[
                (
                    dataset_id,
                    profile.column,
                )
            ] = profile


    return index


# ============================================================
# SUMMARY BUILDER
# ============================================================

def build_summary(
    *,
    benchmark_name: str,
    case_count: int,
    assertions: list[
        BenchmarkAssertionResult
    ],
) -> SemanticBenchmarkSummary:
    assertion_count = len(
        assertions
    )


    correct_count = sum(
        assertion.correct
        for assertion
        in assertions
    )


    incorrect_count = (
        assertion_count
        -
        correct_count
    )


    accuracy = (
        correct_count
        /
        assertion_count
        if assertion_count
        else 0.0
    )


    by_field: dict[
        str,
        list[
            bool
        ],
    ] = defaultdict(
        list
    )


    for assertion in (
        assertions
    ):
        by_field[
            assertion.field
        ].append(
            assertion.correct
        )


    field_accuracy = {
        field:
            round(
                (
                    sum(
                        values
                    )
                    /
                    len(
                        values
                    )
                ),
                6,
            )

        for field, values
        in sorted(
            by_field.items()
        )
    }


    return SemanticBenchmarkSummary(
        benchmark_name=
            benchmark_name,

        case_count=
            case_count,

        assertion_count=
            assertion_count,

        correct_count=
            correct_count,

        incorrect_count=
            incorrect_count,

        accuracy=
            round(
                accuracy,
                6,
            ),

        field_accuracy=
            field_accuracy,

        assertions=
            assertions,
    )


# ============================================================
# COLUMN BENCHMARK
# ============================================================

def evaluate_semantic_columns(
    *,
    dataset_profiles: list[
        Any
    ],
    cases: list[
        SemanticColumnBenchmarkCase
    ],
    benchmark_name: str,
) -> SemanticBenchmarkSummary:
    profile_index = (
        build_profile_index(
            dataset_profiles
        )
    )


    assertions: list[
        BenchmarkAssertionResult
    ] = []


    for case in cases:
        key = (
            case.dataset_id,
            case.column,
        )


        profile = profile_index.get(
            key
        )


        for expectation in (
            case.expectations
        ):
            expected_values = [
                normalize_expected_value(
                    value
                )

                for value
                in expectation.accepted_values
            ]


            if profile is None:
                actual = "__missing_profile__"

            else:
                actual = (
                    normalize_expected_value(
                        getattr(
                            profile,
                            expectation.field,
                            "__missing_field__",
                        )
                    )
                )


            correct = (
                actual
                in expected_values
            )


            assertions.append(
                BenchmarkAssertionResult(
                    case_id=
                        case.case_id,

                    dataset_id=
                        case.dataset_id,

                    subject=
                        case.column,

                    field=
                        expectation.field,

                    expected=
                        " | ".join(
                            expected_values
                        ),

                    actual=
                        actual,

                    correct=
                        correct,
                )
            )


    return build_summary(
        benchmark_name=
            benchmark_name,

        case_count=
            len(
                cases
            ),

        assertions=
            assertions,
    )


# ============================================================
# PAIR BENCHMARK
# ============================================================

PAIR_FIELDS = (
    "same_concept",
    "same_concept_family",
    "same_domain",
    "distinct_variants",
    "compatible_units",
    "derived_gap_compatible",
)


def evaluate_semantic_pairs(
    *,
    dataset_profiles: list[
        Any
    ],
    cases: list[
        SemanticPairBenchmarkCase
    ],
    benchmark_name: str,
) -> SemanticBenchmarkSummary:
    profile_index = (
        build_profile_index(
            dataset_profiles
        )
    )


    assertions: list[
        BenchmarkAssertionResult
    ] = []


    for case in cases:
        left_profile = profile_index.get(
            (
                case.left_dataset_id,
                case.left_column,
            )
        )


        right_profile = profile_index.get(
            (
                case.right_dataset_id,
                case.right_column,
            )
        )


        comparison = None


        if (
            left_profile is not None
            and
            right_profile is not None
        ):
            comparison = (
                compare_semantic_profiles(
                    left_profile,
                    right_profile,
                )
            )


        dataset_scope = (
            case.left_dataset_id
            if (
                case.left_dataset_id
                ==
                case.right_dataset_id
            )
            else (
                f"{case.left_dataset_id}"
                f"::{case.right_dataset_id}"
            )
        )


        for field in (
            PAIR_FIELDS
        ):
            expected = getattr(
                case,
                field,
            )


            if expected is None:
                continue


            if comparison is None:
                actual = "__missing_comparison__"

            else:
                actual = (
                    normalize_expected_value(
                        getattr(
                            comparison,
                            field,
                            "__missing_field__",
                        )
                    )
                )


            expected_value = (
                normalize_expected_value(
                    expected
                )
            )


            assertions.append(
                BenchmarkAssertionResult(
                    case_id=
                        case.case_id,

                    dataset_id=
                        dataset_scope,

                    subject=(
                        f"{case.left_column} "
                        f"vs {case.right_column}"
                    ),

                    field=
                        field,

                    expected=
                        expected_value,

                    actual=
                        actual,

                    correct=(
                        actual
                        ==
                        expected_value
                    ),
                )
            )


    return build_summary(
        benchmark_name=
            benchmark_name,

        case_count=
            len(
                cases
            ),

        assertions=
            assertions,
    )


# ============================================================
# VERSION COMPARISON
# ============================================================

def compare_benchmark_versions(
    *,
    baseline: SemanticBenchmarkSummary,
    candidate: SemanticBenchmarkSummary,
) -> BenchmarkVersionComparison:
    baseline_lookup = {
        (
            assertion.case_id,
            assertion.dataset_id,
            assertion.subject,
            assertion.field,
        ):
            assertion

        for assertion
        in baseline.assertions
    }


    candidate_lookup = {
        (
            assertion.case_id,
            assertion.dataset_id,
            assertion.subject,
            assertion.field,
        ):
            assertion

        for assertion
        in candidate.assertions
    }


    common_keys = (
        set(
            baseline_lookup
        )
        &
        set(
            candidate_lookup
        )
    )


    improved = 0
    regressed = 0
    unchanged = 0


    for key in (
        common_keys
    ):
        before = (
            baseline_lookup[
                key
            ].correct
        )


        after = (
            candidate_lookup[
                key
            ].correct
        )


        if (
            not before
            and
            after
        ):
            improved += 1

        elif (
            before
            and
            not after
        ):
            regressed += 1

        else:
            unchanged += 1


    return BenchmarkVersionComparison(
        baseline_name=
            baseline.benchmark_name,

        candidate_name=
            candidate.benchmark_name,

        baseline_accuracy=
            baseline.accuracy,

        candidate_accuracy=
            candidate.accuracy,

        absolute_accuracy_delta=
            round(
                (
                    candidate.accuracy
                    -
                    baseline.accuracy
                ),
                6,
            ),

        improved_assertion_count=
            improved,

        regressed_assertion_count=
            regressed,

        unchanged_assertion_count=
            unchanged,

        regression_free=(
            regressed
            ==
            0
        ),
    )
