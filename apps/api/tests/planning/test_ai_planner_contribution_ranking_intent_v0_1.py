from __future__ import annotations


from app.planning.ai_analytical_planner import (
    AI_ANALYTICAL_PLANNER_RULE_VERSION,
    explicit_aggregation_from_objective,
    explicit_benchmark_operator_from_objective,
    explicit_ranking_limit_from_objective,
    explicit_ranking_order_from_objective,
)


# ============================================================
# TARGET
# ============================================================


TARGET_OBJECTIVE = (
    "Quelle catégorie contribue le plus au chiffre d'affaires "
    "et quelle est sa part approximative du chiffre d'affaires total ?"
)


# ============================================================
# EXISTING AUTHORITIES
# ============================================================


EXISTING_RANKING_OBJECTIVE = (
    "Quels produits sont les plus performants ?"
)


VAGUE_PERFORMANCE_OBJECTIVE = (
    "Analyse la performance selon la catégorie."
)


BENCHMARK_OBJECTIVE = (
    "Quelles catégories ont un chiffre d'affaires "
    "supérieur à la moyenne ?"
)


NEUTRAL_CONTRIBUTION_OBJECTIVE = (
    "Quelle est la contribution de chaque catégorie "
    "au chiffre d'affaires ?"
)


# ============================================================
# MAIN
# ============================================================


def main() -> None:

    print(
        "=== DATALENS CONTRIBUTION RANKING INTENT v0.1 ==="
    )

    print()


    # ========================================================
    # 1. PRESERVE EXISTING DECISION COVERAGE
    # ========================================================

    existing_ranking = (
        explicit_ranking_order_from_objective(
            EXISTING_RANKING_OBJECTIVE
        )
    )


    assert (
        existing_ranking
        ==
        "descending"
    )


    print(
        "[PASS] existing 'plus performants' ranking preserved"
    )


    vague_ranking = (
        explicit_ranking_order_from_objective(
            VAGUE_PERFORMANCE_OBJECTIVE
        )
    )


    assert (
        vague_ranking
        ==
        "none"
    )


    print(
        "[PASS] vague performance remains non-ranking"
    )


    benchmark_ranking = (
        explicit_ranking_order_from_objective(
            BENCHMARK_OBJECTIVE
        )
    )


    benchmark_operator = (
        explicit_benchmark_operator_from_objective(
            BENCHMARK_OBJECTIVE
        )
    )


    assert (
        benchmark_ranking
        ==
        "none"
    )


    assert (
        benchmark_operator
        ==
        "gt"
    )


    print(
        "[PASS] benchmark keeps semantic priority over ranking"
    )


    neutral_contribution_ranking = (
        explicit_ranking_order_from_objective(
            NEUTRAL_CONTRIBUTION_OBJECTIVE
        )
    )


    assert (
        neutral_contribution_ranking
        ==
        "none"
    ), (
        "A neutral contribution request must not imply "
        "a winner or ranking."
    )


    print(
        "[PASS] neutral contribution remains non-ranking"
    )


    # ========================================================
    # 2. TARGET OBJECTIVE SEMANTICS
    # ========================================================

    target_aggregation = (
        explicit_aggregation_from_objective(
            TARGET_OBJECTIVE
        )
    )


    target_benchmark = (
        explicit_benchmark_operator_from_objective(
            TARGET_OBJECTIVE
        )
    )


    target_ranking = (
        explicit_ranking_order_from_objective(
            TARGET_OBJECTIVE
        )
    )


    # Prove independently that the existing limit authority is
    # already sufficient once descending ranking is recognized.
    forced_limit = (
        explicit_ranking_limit_from_objective(
            TARGET_OBJECTIVE,
            ranking_order=
                "descending",
        )
    )


    current_limit = (
        explicit_ranking_limit_from_objective(
            TARGET_OBJECTIVE,
            ranking_order=
                target_ranking,
        )
    )


    print()
    print(
        "Planner rule                            "
        f"{AI_ANALYTICAL_PLANNER_RULE_VERSION}"
    )

    print(
        "Target aggregation                      "
        f"{target_aggregation}"
    )

    print(
        "Target benchmark                        "
        f"{target_benchmark}"
    )

    print(
        "Target ranking                          "
        f"{target_ranking}"
    )

    print(
        "Forced descending limit                 "
        f"{forced_limit}"
    )

    print(
        "Current inferred limit                  "
        f"{current_limit}"
    )


    # The objective contains "total", therefore SUM authority
    # is already correctly detected.
    assert (
        target_aggregation
        ==
        "sum"
    )


    # "part approximative" is not a same-population threshold
    # comparison and must not become a BenchmarkSpec.
    assert (
        target_benchmark
        is None
    )


    # Once ranking exists, "Quelle catégorie..." is already
    # deterministically Top-1.
    assert (
        forced_limit
        ==
        1
    ), (
        "Top-1 authority is unexpectedly missing independently "
        "of the ranking-order gap."
    )


    gaps: list[
        str
    ] = []


    if (
        target_ranking
        !=
        "descending"
    ):

        gaps.append(
            "contribution_ranking_order_gap"
        )


    if (
        target_ranking
        ==
        "descending"

        and

        current_limit
        !=
        1
    ):

        gaps.append(
            "contribution_top1_limit_gap"
        )


    print()
    print(
        "Observed gaps                           "
        f"{gaps}"
    )


    if gaps:

        raise AssertionError(
            (
                "RED_EXPECTED: "
                +
                ",".join(
                    gaps
                )
            )
        )


    print()
    print(
        "PASS - contribution ranking intent v0.1"
    )


if __name__ == "__main__":

    main()