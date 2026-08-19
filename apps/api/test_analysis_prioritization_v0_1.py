from __future__ import annotations

from app.analysis_prioritization import (
    ANALYSIS_PRIORITIZATION_RULE_VERSION,
    FAMILY_CAPS,
    MAX_SELECTED_ANALYSES,
    build_prioritized_execution_discovery,
    prioritize_analysis_discovery,
)

from app.discovery.schemas import (
    AnalysisDiscoveryReport,
    DiscoveredAnalysis,
    DiscoveredVariable,
)


def variable(
    *,
    column: str,
    role: str,
    semantic_role: str,
    analysis_kind: str = "quantitative",
) -> DiscoveredVariable:
    return DiscoveredVariable(
        dataset_id=
            "dataset:test",

        dataset_filename=
            "test.csv",

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


def candidate(
    *,
    analysis_id: str,
    family: str,
    score: float,
    variables: list[DiscoveredVariable] | None = None,
    readiness: str = "executable_now",
    scope: str = "single_dataset",
) -> DiscoveredAnalysis:
    return DiscoveredAnalysis(
        analysis_id=
            analysis_id,

        scope=
            scope,

        family=
            family,

        title=
            analysis_id,

        priority_score=
            score,

        readiness=
            readiness,

        datasets=[
            "test.csv"
        ],

        dataset_ids=[
            "dataset:test"
        ],

        variables=
            variables
            or [],

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

        observed_signals={},

        redundancy_key=
            analysis_id,
    )


def discovery(
    candidates: list[DiscoveredAnalysis],
) -> AnalysisDiscoveryReport:
    return AnalysisDiscoveryReport(
        objective=
            None,

        dataset_count=
            1,

        candidate_count=
            len(
                candidates
            ),

        single_dataset_candidate_count=
            sum(
                1
                for item
                in candidates
                if (
                    item.scope
                    ==
                    "single_dataset"
                )
            ),

        cross_dataset_candidate_count=
            sum(
                1
                for item
                in candidates
                if (
                    item.scope
                    ==
                    "cross_dataset"
                )
            ),

        candidates=
            candidates,

        relationships=[],

        discovery_notes=[],
    )


def test_identifier_measure_is_rejected() -> None:
    report = prioritize_analysis_discovery(
        discovery(
            [
                candidate(
                    analysis_id=
                        "id-vs-price",

                    family=
                        "quantitative_association",

                    score=
                        99,

                    variables=[
                        variable(
                            column=
                                "order_id",

                            role=
                                "x",

                            semantic_role=
                                "identifier",
                        ),

                        variable(
                            column=
                                "price",

                            role=
                                "y",

                            semantic_role=
                                "measure",
                        ),
                    ],
                )
            ]
        )
    )

    assert report.rejected_count == 1

    assert (
        report.decisions[
            0
        ].reason_code
        ==
        "identifier_misuse"
    )

    print(
        "Identifier used as analytical measure rejected: PASS"
    )


def test_identifier_entity_dimension_is_allowed() -> None:
    report = prioritize_analysis_discovery(
        discovery(
            [
                candidate(
                    analysis_id=
                        "customer-ranking",

                    family=
                        "entity_ranking",

                    score=
                        75,

                    variables=[
                        variable(
                            column=
                                "customer_id",

                            role=
                                "entity",

                            semantic_role=
                                "identifier",

                            analysis_kind=
                                "identifier",
                        ),

                        variable(
                            column=
                                "revenue",

                            role=
                                "value",

                            semantic_role=
                                "measure",
                        ),
                    ],
                )
            ]
        )
    )

    assert report.selected_count == 1

    print(
        "Identifier allowed as entity dimension: PASS"
    )


def test_non_executable_candidate_is_deferred() -> None:
    report = prioritize_analysis_discovery(
        discovery(
            [
                candidate(
                    analysis_id=
                        "planned",

                    family=
                        "distribution",

                    score=
                        90,

                    readiness=
                        "planned",
                )
            ]
        )
    )

    assert report.deferred_count == 1

    assert (
        report.decisions[
            0
        ].reason_code
        ==
        "not_executable_now"
    )

    print(
        "Non-executable discovery candidate deferred: PASS"
    )


def test_quality_is_preserved() -> None:
    report = prioritize_analysis_discovery(
        discovery(
            [
                candidate(
                    analysis_id=
                        "quality",

                    family=
                        "data_quality",

                    score=
                        5,
                )
            ]
        )
    )

    assert report.selected_count == 1

    assert (
        report.decisions[
            0
        ].reason_code
        ==
        "quality_guard"
    )

    print(
        "Data-quality evidence preserved despite low score: PASS"
    )


def test_family_budget_limits_pairwise_explosion() -> None:
    candidates = []

    for index in range(
        30
    ):
        candidates.append(
            candidate(
                analysis_id=
                    f"assoc-{index:02d}",

                family=
                    "quantitative_association",

                score=
                    99
                    -
                    index
                    * 0.5,

                variables=[
                    variable(
                        column=
                            f"x_{index}",

                        role=
                            "x",

                        semantic_role=
                            "measure",
                    ),

                    variable(
                        column=
                            f"y_{index}",

                        role=
                            "y",

                        semantic_role=
                            "measure",
                    ),
                ],
            )
        )

    report = prioritize_analysis_discovery(
        discovery(
            candidates
        )
    )

    assert (
        report
        .family_selected_counts
        .get(
            "quantitative_association"
        )
        ==
        FAMILY_CAPS[
            "quantitative_association"
        ]
    )

    assert any(
        decision.reason_code
        ==
        "family_budget_exhausted"
        for decision
        in report.decisions
    )

    print(
        "Quantitative pairwise explosion limited by family budget: PASS"
    )


def test_variable_budget_limits_redundancy() -> None:
    """
    Use quantitative_association deliberately.

    Its family cap is 12, while MAX_VARIABLE_OCCURRENCES is 8.
    That guarantees the repeated-variable guard is reached
    before the family budget and isolates the rule under test.
    """

    candidates = []

    for index in range(
        20
    ):
        candidates.append(
            candidate(
                analysis_id=
                    f"shared-{index:02d}",

                family=
                    "quantitative_association",

                score=
                    90
                    -
                    index
                    * 0.5,

                variables=[
                    variable(
                        column=
                            "revenue",

                        role=
                            "x",

                        semantic_role=
                            "measure",
                    ),

                    variable(
                        column=
                            f"metric_{index}",

                        role=
                            "y",

                        semantic_role=
                            "measure",
                    ),
                ],
            )
        )

    report = prioritize_analysis_discovery(
        discovery(
            candidates
        )
    )

    variable_budget_decisions = [
        decision
        for decision
        in report.decisions
        if (
            decision.reason_code
            ==
            "variable_budget_exhausted"
        )
    ]

    assert variable_budget_decisions

    assert any(
        "dataset:test:revenue"
        in
        decision.reasons[
            1
        ]
        for decision
        in variable_budget_decisions
        if (
            len(
                decision.reasons
            )
            >
            1
        )
    )

    assert (
        report
        .family_selected_counts
        .get(
            "quantitative_association"
        )
        ==
        8
    )

    print(
        "Repeated-variable redundancy limited: PASS"
    )


def test_global_budget_is_enforced() -> None:
    families = [
        "distribution",
        "quantitative_association",
        "group_comparison",
        "categorical_association",
        "time_series",
        "derived_gap",
        "entity_ranking",
        "geographic_comparison",
        "other_family",
    ]

    candidates = []

    for index in range(
        90
    ):
        family = families[
            index
            %
            len(
                families
            )
        ]

        candidates.append(
            candidate(
                analysis_id=
                    f"global-{index:03d}",

                family=
                    family,

                score=
                    99
                    -
                    index
                    * 0.1,

                variables=[
                    variable(
                        column=
                            f"value_{index}",

                        role=
                            "value",

                        semantic_role=
                            "measure",
                    )
                ],
            )
        )

    report = prioritize_analysis_discovery(
        discovery(
            candidates
        )
    )

    assert (
        report.selected_count
        <=
        MAX_SELECTED_ANALYSES
    )

    assert report.discovered_count == 90

    print(
        "Global execution budget enforced while discovery count is preserved: PASS"
    )


def test_source_discovery_is_not_mutated() -> None:
    source = discovery(
        [
            candidate(
                analysis_id=
                    "a",

                family=
                    "distribution",

                score=
                    80,
            ),

            candidate(
                analysis_id=
                    "b",

                family=
                    "distribution",

                score=
                    30,
            ),
        ]
    )

    original_count = (
        source.candidate_count
    )

    original_ids = [
        item.analysis_id
        for item
        in source.candidates
    ]

    prioritization = (
        prioritize_analysis_discovery(
            source
        )
    )

    execution_view = (
        build_prioritized_execution_discovery(
            source_discovery=
                source,

            prioritization=
                prioritization,
        )
    )

    assert (
        source.candidate_count
        ==
        original_count
    )

    assert (
        [
            item.analysis_id
            for item
            in source.candidates
        ]
        ==
        original_ids
    )

    assert (
        execution_view.candidate_count
        ==
        prioritization.selected_count
    )

    print(
        "Execution shortlist does not mutate broad discovery: PASS"
    )


def test_rule_version() -> None:
    assert (
        ANALYSIS_PRIORITIZATION_RULE_VERSION
        ==
        "analysis_prioritization_v0.1"
    )

    print(
        "Prioritization rule version: PASS"
    )


def main() -> None:
    print(
        "=== DATALENS ANALYSIS PRIORITIZATION v0.1 ==="
    )

    print()

    test_identifier_measure_is_rejected()
    test_identifier_entity_dimension_is_allowed()
    test_non_executable_candidate_is_deferred()
    test_quality_is_preserved()
    test_family_budget_limits_pairwise_explosion()
    test_variable_budget_limits_redundancy()
    test_global_budget_is_enforced()
    test_source_discovery_is_not_mutated()
    test_rule_version()

    print()

    print(
        (
            "Analysis Prioritization version: "
            f"{ANALYSIS_PRIORITIZATION_RULE_VERSION}"
        )
    )

    print(
        "Analysis Prioritization v0.1: PASS"
    )


if __name__ == "__main__":
    main()
