from __future__ import annotations


from app.planning.ai_analytical_planner import (
    AI_ANALYTICAL_PLANNER_RULE_VERSION,
    AIPlannerProposal,
    PlannerCatalog,
    PlannerColumnProfile,
    PlannerDatasetProfile,
    SYSTEM_PROMPT,
    build_benchmark,
    canonicalize_aggregation_ranking_intent,
    decision_coverage_errors,
    explicit_benchmark_operator_from_objective,
    explicit_ranking_limit_from_objective,
    explicit_ranking_order_from_objective,
)


# ============================================================
# FIXTURES
# ============================================================

def catalog() -> PlannerCatalog:
    return PlannerCatalog(
        datasets=[
            PlannerDatasetProfile(
                dataset_id=(
                    "dataset:sales"
                ),
                filename=(
                    "sales.csv"
                ),
                row_count=1000,
                column_count=2,
                columns=[
                    PlannerColumnProfile(
                        name="country",
                        dtype="object",
                        analysis_kind=(
                            "categorical"
                        ),
                        missing_ratio=0.0,
                        unique_count=12,
                    ),
                    PlannerColumnProfile(
                        name="revenue",
                        dtype="float64",
                        analysis_kind=(
                            "quantitative"
                        ),
                        missing_ratio=0.0,
                        unique_count=850,
                    ),
                ],
            )
        ]
    )


def proposal(
    *,
    family="group_comparison",
    group_column="country",
    dimension_column=None,
    ranking_order="none",
    ranking_limit=None,
    benchmark_reference=None,
    benchmark_operator=None,
    benchmark_selection=None,
) -> AIPlannerProposal:
    return AIPlannerProposal(
        decision="propose",
        title="Decision Coverage test",
        family=family,
        dataset_id="dataset:sales",
        analytical_grain=None,
        x_column=None,
        y_column=None,
        group_column=(
            group_column
        ),
        value_column="revenue",
        time_column=None,
        dimension_column=(
            dimension_column
        ),
        entity_column=None,
        aggregation_function="mean",
        ranking_order=(
            ranking_order
        ),
        ranking_limit=(
            ranking_limit
        ),
        window_operation="none",
        window_size=None,
        benchmark_reference=(
            benchmark_reference
        ),
        benchmark_operator=(
            benchmark_operator
        ),
        benchmark_selection=(
            benchmark_selection
        ),
        blockers=[],
        reasons=[
            "decision coverage test",
        ],
        confidence=1.0,
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print(
        "=== DATALENS DECISION COVERAGE v0.1 ==="
    )


    # --------------------------------------------------------
    # VERSION
    # --------------------------------------------------------

    assert (
        AI_ANALYTICAL_PLANNER_RULE_VERSION
        ==
        "ai_analytical_planner_v0.34"
    )


    print(
        "[PASS] planner version v0.34"
    )


    # ========================================================
    # 1. PLUS PERFORMANTS -> RANKING
    # ========================================================

    ranking_objective = (
        "Classe les country les plus performants "
        "selon le revenue moyen."
    )


    assert (
        explicit_ranking_order_from_objective(
            ranking_objective
        )
        ==
        "descending"
    )


    assert (
        explicit_ranking_limit_from_objective(
            ranking_objective,
            ranking_order="descending",
        )
        ==
        10
    )


    assert (
        explicit_benchmark_operator_from_objective(
            ranking_objective
        )
        is None
    )


    (
        normalized_ranking,
        ranking_normalizations,
    ) = (
        canonicalize_aggregation_ranking_intent(
            objective=ranking_objective,
            proposal=proposal(),
            catalog=catalog(),
        )
    )


    assert (
        normalized_ranking.family
        ==
        "ranking"
    )

    assert (
        normalized_ranking.dimension_column
        ==
        "country"
    )

    assert (
        normalized_ranking.group_column
        is None
    )

    assert (
        normalized_ranking.value_column
        ==
        "revenue"
    )

    assert (
        normalized_ranking.aggregation_function
        ==
        "mean"
    )

    assert (
        normalized_ranking.ranking_order
        ==
        "descending"
    )

    assert (
        normalized_ranking.ranking_limit
        ==
        10
    )

    assert (
        normalized_ranking.benchmark_reference
        is None
    )

    assert (
        normalized_ranking.benchmark_operator
        is None
    )

    assert (
        normalized_ranking.benchmark_selection
        is None
    )

    assert (
        ranking_normalizations
    )


    assert (
        decision_coverage_errors(
            objective=ranking_objective,
            proposal=normalized_ranking,
        )
        ==
        []
    )


    print(
        '[PASS] "plus performants" -> ranking'
    )


    # ========================================================
    # 2. EXPLICIT NUMERIC LIMIT
    # ========================================================

    ranking_three_objective = (
        "Donne les 3 country les plus performants "
        "selon le revenue moyen."
    )


    ranking_three_order = (
        explicit_ranking_order_from_objective(
            ranking_three_objective
        )
    )


    assert (
        ranking_three_order
        ==
        "descending"
    )


    assert (
        explicit_ranking_limit_from_objective(
            ranking_three_objective,
            ranking_order=ranking_three_order,
        )
        ==
        3
    )


    print(
        "[PASS] explicit ranking limit preserved"
    )


    # ========================================================
    # 3. VAGUE PERFORMANCE
    # ========================================================

    vague_objective = (
        "Analyse la performance selon country."
    )


    assert (
        explicit_ranking_order_from_objective(
            vague_objective
        )
        ==
        "none"
    )


    assert (
        explicit_benchmark_operator_from_objective(
            vague_objective
        )
        is None
    )


    print(
        "[PASS] vague performance remains non-ranking"
    )


    # ========================================================
    # 4. SUPERIEUR A LA MOYENNE -> BENCHMARK
    # ========================================================

    benchmark_objective = (
        "Quels country ont un revenue moyen "
        "superieur a la moyenne globale ?"
    )


    assert (
        explicit_benchmark_operator_from_objective(
            benchmark_objective
        )
        ==
        "gt"
    )


    assert (
        explicit_ranking_order_from_objective(
            benchmark_objective
        )
        ==
        "none"
    )


    malformed_ranking = (
        proposal(
            family="ranking",
            group_column=None,
            dimension_column="country",
            ranking_order="descending",
            ranking_limit=10,
        )
    )


    (
        normalized_benchmark,
        benchmark_normalizations,
    ) = (
        canonicalize_aggregation_ranking_intent(
            objective=benchmark_objective,
            proposal=malformed_ranking,
            catalog=catalog(),
        )
    )


    assert (
        normalized_benchmark.family
        ==
        "aggregation"
    )

    assert (
        normalized_benchmark.group_column
        ==
        "country"
    )

    assert (
        normalized_benchmark.dimension_column
        is None
    )

    assert (
        normalized_benchmark.value_column
        ==
        "revenue"
    )

    assert (
        normalized_benchmark.aggregation_function
        ==
        "mean"
    )

    assert (
        normalized_benchmark.ranking_order
        ==
        "none"
    )

    assert (
        normalized_benchmark.ranking_limit
        is None
    )

    assert (
        normalized_benchmark.benchmark_reference
        ==
        "overall_aggregate"
    )

    assert (
        normalized_benchmark.benchmark_operator
        ==
        "gt"
    )

    assert (
        normalized_benchmark.benchmark_selection
        ==
        "matching_only"
    )

    assert (
        benchmark_normalizations
    )


    assert (
        decision_coverage_errors(
            objective=benchmark_objective,
            proposal=normalized_benchmark,
        )
        ==
        []
    )


    benchmark_spec = (
        build_benchmark(
            normalized_benchmark
        )
    )


    assert (
        benchmark_spec
        is not None
    )

    assert (
        benchmark_spec.reference
        ==
        "overall_aggregate"
    )

    assert (
        benchmark_spec.operator
        ==
        "gt"
    )

    assert (
        benchmark_spec.selection
        ==
        "matching_only"
    )


    print(
        '[PASS] "superieur a la moyenne" -> benchmark'
    )


    # ========================================================
    # 5. MISSING BENCHMARK FAIL CLOSED
    #
    # Do not depend on the accented diagnostic prefix.
    # C7 owns encoding polish.
    # ========================================================

    missing_benchmark = (
        proposal()
    )


    missing_errors = (
        decision_coverage_errors(
            objective=benchmark_objective,
            proposal=missing_benchmark,
        )
    )


    assert (
        missing_errors
    )


    assert any(
        "explicit benchmark language requires"
        in error

        for error
        in missing_errors
    )


    assert any(
        "benchmark_reference=overall_aggregate"
        in error

        for error
        in missing_errors
    )


    assert any(
        "benchmark operator"
        in error.lower()

        for error
        in missing_errors
    )


    print(
        "[PASS] missing requested benchmark fails closed"
    )


    # ========================================================
    # 6. WRONG OPERATOR FAIL CLOSED
    # ========================================================

    wrong_operator = (
        proposal(
            family="aggregation",
            benchmark_reference=(
                "overall_aggregate"
            ),
            benchmark_operator="lt",
            benchmark_selection=(
                "matching_only"
            ),
        )
    )


    wrong_operator_errors = (
        decision_coverage_errors(
            objective=benchmark_objective,
            proposal=wrong_operator,
        )
    )


    assert (
        wrong_operator_errors
    )


    assert any(
        "benchmark operator"
        in error.lower()

        for error
        in wrong_operator_errors
    )


    print(
        "[PASS] wrong benchmark operator fails closed"
    )


    # ========================================================
    # 7. UNSOLICITED BENCHMARK FAIL CLOSED
    # ========================================================

    ordinary_objective = (
        "Calcule le revenue moyen par country."
    )


    unsolicited = (
        proposal(
            family="aggregation",
            benchmark_reference=(
                "overall_aggregate"
            ),
            benchmark_operator="gt",
            benchmark_selection=(
                "matching_only"
            ),
        )
    )


    unsolicited_errors = (
        decision_coverage_errors(
            objective=ordinary_objective,
            proposal=unsolicited,
        )
    )


    assert (
        unsolicited_errors
    )


    assert any(
        "no supported explicit benchmark"
        in error

        for error
        in unsolicited_errors
    )


    print(
        "[PASS] unsolicited benchmark fails closed"
    )


    # ========================================================
    # 8. EXPLICIT RANKING CLEARS WRONG BENCHMARK
    # ========================================================

    malformed_benchmark_for_ranking = (
        proposal(
            family="aggregation",
            benchmark_reference=(
                "overall_aggregate"
            ),
            benchmark_operator="gt",
            benchmark_selection=(
                "matching_only"
            ),
        )
    )


    (
        recovered_ranking,
        _,
    ) = (
        canonicalize_aggregation_ranking_intent(
            objective=ranking_objective,
            proposal=malformed_benchmark_for_ranking,
            catalog=catalog(),
        )
    )


    assert (
        recovered_ranking.family
        ==
        "ranking"
    )

    assert (
        recovered_ranking.benchmark_reference
        is None
    )

    assert (
        recovered_ranking.benchmark_operator
        is None
    )

    assert (
        recovered_ranking.benchmark_selection
        is None
    )


    assert (
        decision_coverage_errors(
            objective=ranking_objective,
            proposal=recovered_ranking,
        )
        ==
        []
    )


    print(
        "[PASS] explicit ranking clears contradictory benchmark"
    )


    # ========================================================
    # 9. PROMPT COVERAGE
    # ========================================================

    assert (
        "plus performants"
        in
        SYSTEM_PROMPT
    )

    assert (
        "family = ranking"
        in
        SYSTEM_PROMPT
    )

    assert (
        "overall_aggregate"
        in
        SYSTEM_PROMPT
    )


    print(
        "[PASS] Gemma decision prompt coverage"
    )


    print()
    print(
        "PASS - Decision Coverage v0.1"
    )


if __name__ == "__main__":
    main()
