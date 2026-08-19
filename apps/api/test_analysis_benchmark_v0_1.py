from __future__ import annotations

import pandas as pd

from app.analysis_prioritization import (
    prioritize_analysis_discovery,
)

from app.discovery.schemas import (
    AnalysisDiscoveryReport,
    DiscoveredAnalysis,
    DiscoveredVariable,
)

from app.evals.analysis_benchmark import (
    ANALYSIS_BENCHMARK_RULE_VERSION,
    AnalysisBenchmarkExpectation,
    AnalysisBenchmarkScenario,
    BenchmarkVariableExpectation,
    evaluate_analysis_benchmark,
    run_analysis_benchmark,
)


DATASET_ID = "dataset:ecommerce"


# ============================================================
# FIXTURES
# ============================================================


def variable(
    *,
    column: str,
    role: str,
    analysis_kind: str,
    semantic_role: str,
) -> DiscoveredVariable:
    return (
        DiscoveredVariable(
            dataset_id=
                DATASET_ID,

            dataset_filename=
                "ecommerce.csv",

            column=
                column,

            role=
                role,

            analysis_kind=
                analysis_kind,

            semantic_role=
                semantic_role,

            concepts=[],
        )
    )


def candidate(
    *,
    analysis_id: str,
    family: str,
    title: str,
    score: float,
    readiness: str,
    variables: list[
        DiscoveredVariable
    ],
    redundancy_key: str,
    observed_signals: dict | None = None,
) -> DiscoveredAnalysis:
    return (
        DiscoveredAnalysis(
            analysis_id=
                analysis_id,

            scope=
                "single_dataset",

            family=
                family,

            title=
                title,

            priority_score=
                score,

            readiness=
                readiness,

            datasets=[
                "ecommerce.csv"
            ],

            dataset_ids=[
                DATASET_ID
            ],

            variables=
                variables,

            chart_type=
                "test",

            execution_strategy=
                "test",

            why_interesting=[],

            limitations=[],

            relationship_status=
                None,

            relationship_score=
                None,

            join_keys={},

            observed_signals=
                observed_signals
                or {},

            redundancy_key=
                redundancy_key,
        )
    )


def benchmark_dataframe(
) -> pd.DataFrame:
    first_names = [
        f"Name_{index % 16}"

        for index
        in range(
            40
        )
    ]


    return (
        pd.DataFrame(
            {
                "first_name":
                    first_names,

                "category":
                    [
                        [
                            "Furniture",
                            "Electronics",
                            "Accessories",
                        ][
                            index % 3
                        ]

                        for index
                        in range(
                            40
                        )
                    ],

                "quantity":
                    [
                        index % 5 + 1

                        for index
                        in range(
                            40
                        )
                    ],

                "unit_price":
                    [
                        20.0
                        +
                        (
                            index
                            *
                            1.5
                        )

                        for index
                        in range(
                            40
                        )
                    ],

                "customer_id":
                    [
                        f"C{index:03d}"

                        for index
                        in range(
                            40
                        )
                    ],

                "order_date":
                    pd.date_range(
                        "2026-01-01",
                        periods=40,
                        freq="D",
                    ),
            }
        )
    )


def dataset_records(
) -> tuple[
    dict,
    ...,
]:
    return (
        {
            "dataset_id":
                DATASET_ID,

            "filename":
                "ecommerce.csv",

            "dataframe":
                benchmark_dataframe(),
        },
    )


def controlled_discovery(
) -> AnalysisDiscoveryReport:
    candidates = [
        # Expected selected.
        candidate(
            analysis_id=
                f"{DATASET_ID}:group:category:unit_price",

            family=
                "group_comparison",

            title=
                "unit_price selon category",

            score=
                92.0,

            readiness=
                "executable_now",

            variables=[
                variable(
                    column=
                        "category",

                    role=
                        "group",

                    analysis_kind=
                        "categorical",

                    semantic_role=
                        "category",
                ),

                variable(
                    column=
                        "unit_price",

                    role=
                        "value",

                    analysis_kind=
                        "quantitative",

                    semantic_role=
                        "measure",
                ),
            ],

            redundancy_key=
                "group:category:unit_price",
        ),

        # Expected deferred by Analytical Value Guard.
        candidate(
            analysis_id=
                f"{DATASET_ID}:group:first_name:quantity",

            family=
                "group_comparison",

            title=
                "quantity selon first_name",

            score=
                95.0,

            readiness=
                "executable_now",

            variables=[
                variable(
                    column=
                        "first_name",

                    role=
                        "group",

                    analysis_kind=
                        "categorical",

                    semantic_role=
                        "category",
                ),

                variable(
                    column=
                        "quantity",

                    role=
                        "value",

                    analysis_kind=
                        "quantitative",

                    semantic_role=
                        "measure",
                ),
            ],

            redundancy_key=
                "group:first_name:quantity",
        ),

        # Expected rejected as identifier misuse.
        candidate(
            analysis_id=
                f"{DATASET_ID}:association:customer_id:unit_price",

            family=
                "quantitative_association",

            title=
                "Relation customer_id / unit_price",

            score=
                99.0,

            readiness=
                "executable_now",

            variables=[
                variable(
                    column=
                        "customer_id",

                    role=
                        "x",

                    analysis_kind=
                        "identifier",

                    semantic_role=
                        "identifier",
                ),

                variable(
                    column=
                        "unit_price",

                    role=
                        "y",

                    analysis_kind=
                        "quantitative",

                    semantic_role=
                        "measure",
                ),
            ],

            redundancy_key=
                "assoc:customer_id:unit_price",
        ),

        # Expected deferred because not executable now.
        candidate(
            analysis_id=
                f"{DATASET_ID}:time:order_date:quantity",

            family=
                "time_series",

            title=
                "Évolution de quantity",

            score=
                88.0,

            readiness=
                "planned",

            variables=[
                variable(
                    column=
                        "order_date",

                    role=
                        "time",

                    analysis_kind=
                        "temporal",

                    semantic_role=
                        "time",
                ),

                variable(
                    column=
                        "quantity",

                    role=
                        "value",

                    analysis_kind=
                        "quantitative",

                    semantic_role=
                        "measure",
                ),
            ],

            redundancy_key=
                "time:order_date:quantity",
        ),
    ]


    return (
        AnalysisDiscoveryReport(
            objective=
                None,

            dataset_count=
                1,

            candidate_count=
                len(
                    candidates
                ),

            single_dataset_candidate_count=
                len(
                    candidates
                ),

            cross_dataset_candidate_count=
                0,

            candidates=
                candidates,

            relationships=[],

            discovery_notes=[],
        )
    )


def scenario(
) -> AnalysisBenchmarkScenario:
    return (
        AnalysisBenchmarkScenario(
            scenario_id=
                "ecommerce_reference_v0.1",

            description=(
                "Référence minimale pour vérifier sélection utile, "
                "forte cardinalité, identifiants et readiness."
            ),

            split=
                "test",

            frozen=
                True,

            datasets=
                dataset_records(),

            expectations=(
                AnalysisBenchmarkExpectation(
                    expectation_id=
                        "select-category-price",

                    description=
                        "Une comparaison métier à faible cardinalité "
                        "doit être sélectionnée.",

                    family=
                        "group_comparison",

                    variables=[
                        BenchmarkVariableExpectation(
                            role=
                                "group",

                            column=
                                "category",
                        ),

                        BenchmarkVariableExpectation(
                            role=
                                "value",

                            column=
                                "unit_price",
                        ),
                    ],

                    allowed_decisions=[
                        "selected"
                    ],
                ),

                AnalysisBenchmarkExpectation(
                    expectation_id=
                        "defer-first-name",

                    description=
                        "Un prénom à forte cardinalité ne doit pas "
                        "consommer le budget exploratoire.",

                    family=
                        "group_comparison",

                    variables=[
                        BenchmarkVariableExpectation(
                            role=
                                "group",

                            column=
                                "first_name",
                        ),

                        BenchmarkVariableExpectation(
                            role=
                                "value",

                            column=
                                "quantity",
                        ),
                    ],

                    allowed_decisions=[
                        "deferred"
                    ],

                    allowed_reason_codes=[
                        "record_label_dimension"
                    ],
                ),

                AnalysisBenchmarkExpectation(
                    expectation_id=
                        "reject-identifier-measure",

                    description=
                        "Un identifiant utilisé comme variable "
                        "analytique doit être rejeté.",

                    family=
                        "quantitative_association",

                    variables=[
                        BenchmarkVariableExpectation(
                            role=
                                "x",

                            column=
                                "customer_id",
                        ),

                        BenchmarkVariableExpectation(
                            role=
                                "y",

                            column=
                                "unit_price",
                        ),
                    ],

                    allowed_decisions=[
                        "rejected"
                    ],

                    allowed_reason_codes=[
                        "identifier_misuse"
                    ],
                ),

                AnalysisBenchmarkExpectation(
                    expectation_id=
                        "defer-planned-time-series",

                    description=
                        "Un candidat non exécutable immédiatement "
                        "doit rester différé.",

                    family=
                        "time_series",

                    variables=[
                        BenchmarkVariableExpectation(
                            role=
                                "time",

                            column=
                                "order_date",
                        ),

                        BenchmarkVariableExpectation(
                            role=
                                "value",

                            column=
                                "quantity",
                        ),
                    ],

                    allowed_decisions=[
                        "deferred"
                    ],

                    allowed_reason_codes=[
                        "not_executable_now"
                    ],
                ),
            ),

            min_discovered_count=
                4,

            max_selected_count=
                36,
        )
    )


# ============================================================
# CORE EVALUATION
# ============================================================


def test_reference_expectations_score_perfectly(
) -> None:
    benchmark_scenario = (
        scenario()
    )


    discovery = (
        controlled_discovery()
    )


    prioritization = (
        prioritize_analysis_discovery(
            discovery,

            datasets=
                list(
                    dataset_records()
                ),
        )
    )


    report = (
        evaluate_analysis_benchmark(
            scenario=
                benchmark_scenario,

            discovery=
                discovery,

            prioritization=
                prioritization,
        )
    )


    assert report.passed is True

    assert (
        report.metrics
        .expectation_accuracy
        ==
        1.0
    )


    assert (
        report.metrics
        .discovery_recall
        ==
        1.0
    )


    assert (
        report.metrics
        .selection_recall
        ==
        1.0
    )


    assert (
        report.metrics
        .guardrail_success_rate
        ==
        1.0
    )


    assert (
        report.metrics
        .selected_candidate_count
        ==
        1
    )


    print(
        "Reference benchmark expectations and guardrails score 100%: PASS"
    )


# ============================================================
# REGRESSION DETECTION
# ============================================================


def test_wrong_expected_decision_fails_benchmark(
) -> None:
    bad_scenario = (
        AnalysisBenchmarkScenario(
            scenario_id=
                "intentional-regression",

            description=
                "Fixture volontairement incorrecte.",

            split=
                "validation",

            datasets=
                dataset_records(),

            expectations=(
                AnalysisBenchmarkExpectation(
                    expectation_id=
                        "wrong-first-name-expectation",

                    family=
                        "group_comparison",

                    variables=[
                        BenchmarkVariableExpectation(
                            role=
                                "group",

                            column=
                                "first_name",
                        ),

                        BenchmarkVariableExpectation(
                            role=
                                "value",

                            column=
                                "quantity",
                        ),
                    ],

                    allowed_decisions=[
                        "selected"
                    ],
                ),
            ),
        )
    )


    discovery = (
        controlled_discovery()
    )


    prioritization = (
        prioritize_analysis_discovery(
            discovery,

            datasets=
                list(
                    dataset_records()
                ),
        )
    )


    report = (
        evaluate_analysis_benchmark(
            scenario=
                bad_scenario,

            discovery=
                discovery,

            prioritization=
                prioritization,
        )
    )


    assert report.passed is False

    assert (
        report.outcomes[
            0
        ].failure_code
        ==
        "unexpected_decision"
    )


    print(
        "Benchmark detects an intentional decision regression: PASS"
    )


# ============================================================
# STRUCTURAL MATCHING
# ============================================================


def test_benchmark_does_not_depend_on_exact_analysis_id(
) -> None:
    discovery = (
        controlled_discovery()
    )


    discovery.candidates[
        0
    ].analysis_id = (
        "future:new:id:format"
    )


    prioritization = (
        prioritize_analysis_discovery(
            discovery,

            datasets=
                list(
                    dataset_records()
                ),
        )
    )


    report = (
        evaluate_analysis_benchmark(
            scenario=
                scenario(),

            discovery=
                discovery,

            prioritization=
                prioritization,
        )
    )


    assert report.passed is True


    print(
        "Structural benchmark matching survives analysis_id evolution: PASS"
    )


# ============================================================
# FULL PIPELINE / DETERMINISM
# ============================================================


def full_pipeline_scenario(
) -> AnalysisBenchmarkScenario:
    """
    Loose end-to-end contract.

    Exact selections are deliberately not frozen here because
    Discovery may legitimately evolve. This scenario verifies:
    - the engine produces candidates;
    - the same input is deterministic;
    - the global execution budget remains enforced.
    """

    return (
        AnalysisBenchmarkScenario(
            scenario_id=
                "ecommerce_pipeline_smoke_v0.1",

            description=
                "Smoke benchmark du vrai pipeline Discovery + "
                "Prioritization.",

            split=
                "validation",

            datasets=
                dataset_records(),

            expectations=
                tuple(),

            min_discovered_count=
                1,

            max_selected_count=
                36,
        )
    )


def test_full_pipeline_is_deterministic(
) -> None:
    report = (
        run_analysis_benchmark(
            full_pipeline_scenario(),

            deterministic_runs=
                2,
        )
    )


    assert (
        report.metrics
        .deterministic
        is True
    )


    assert (
        report.metrics
        .discovered_candidate_count
        >=
        1
    )


    assert (
        report.metrics
        .selected_candidate_count
        <=
        36
    )


    assert (
        report.passed
        is True
    )


    print(
        "Real Discovery + Prioritization pipeline is deterministic across repeated runs: PASS"
    )


# ============================================================
# VERSION
# ============================================================


def test_test_split_requires_frozen_scenario(
) -> None:
    try:
        AnalysisBenchmarkScenario(
            scenario_id=
                "invalid-unfrozen-test",

            description=
                "Ce scénario doit être refusé.",

            split=
                "test",

            frozen=
                False,

            datasets=
                dataset_records(),

            expectations=
                tuple(),
        )

    except ValueError as error:
        assert (
            "test benchmark scenarios must be frozen"
            in str(
                error
            )
        )

    else:
        raise AssertionError(
            "An unfrozen test scenario should be rejected."
        )


    print(
        "Discovery/Prioritization evals reuse frozen test-set discipline: PASS"
    )


def test_benchmark_version(
) -> None:
    assert (
        ANALYSIS_BENCHMARK_RULE_VERSION
        ==
        "analysis_benchmark_v0.1"
    )


    print(
        "Analysis Benchmark rule version: PASS"
    )


# ============================================================
# MAIN
# ============================================================


def main() -> None:
    print(
        "=== DATALENS ANALYSIS BENCHMARK v0.1 ==="
    )

    print()


    test_reference_expectations_score_perfectly()

    test_wrong_expected_decision_fails_benchmark()

    test_benchmark_does_not_depend_on_exact_analysis_id()

    test_full_pipeline_is_deterministic()

    test_test_split_requires_frozen_scenario()

    test_benchmark_version()


    print()

    print(
        "Analysis Benchmark v0.1: PASS"
    )


if __name__ == "__main__":
    main()
