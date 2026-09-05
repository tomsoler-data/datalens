from __future__ import annotations


import pandas as pd


from app.planning.planner_catalog import (
    planner_catalog_from_dataset_records,
)


from app.planning.objective_coverage import (
    extract_objective_requirements,
)


from app.planning.ai_analytical_planner import (
    AIPlannerProposal,
    canonicalize_analytical_view_intent,
    explicit_aggregation_from_objective,
    explicit_benchmark_operator_from_objective,
    explicit_ranking_order_from_objective,
)


# ============================================================
# OBJECTIVES
# ============================================================


EXPLICIT_TOTAL_OBJECTIVE = (
    "Quel est le chiffre d'affaires total "
    "sur toute la période analysée ?"
)


EXPLICIT_RANKING_OBJECTIVE = (
    "Quels sont les segments les plus performants "
    "par chiffre d'affaires ?"
)


EXPLICIT_BENCHMARK_OBJECTIVE = (
    "Quels segments ont un chiffre d'affaires "
    "supérieur à la moyenne ?"
)


# ============================================================
# SERVER-OWNED AUTHORITY
# ============================================================


TRUSTED_MONETARY_EVENT_SEMANTICS = (
    "The unit monetary measure was propagated from a validated "
    "dimension to fact grain. No explicit quantity measure was "
    "detected, so one fact row is conservatively treated as one "
    "monetary event."
)


SOURCE_DATASET_ID = (
    "dataset_source"
)


SECOND_SOURCE_DATASET_ID = (
    "dataset_source_2"
)


SCALAR_DATASET_ID = (
    "derived:dataset_source:scalar:price"
)


SECOND_SCALAR_DATASET_ID = (
    "derived:dataset_source_2:scalar:price"
)


TARGET_MEASURE = (
    "sum_price"
)


# ============================================================
# RECORD BUILDERS
# ============================================================


def source_record(
    *,
    dataset_id: str = SOURCE_DATASET_ID,
) -> dict:

    return {
        "dataset_id":
            dataset_id,

        "filename":
            f"{dataset_id}.csv",

        "dataframe":
            pd.DataFrame(
                {
                    "event_id": [
                        "e1",
                        "e2",
                        "e3",
                    ],

                    "segment": [
                        "A",
                        "A",
                        "B",
                    ],

                    "price": [
                        10.0,
                        20.0,
                        30.0,
                    ],

                    "other_metric": [
                        1.0,
                        2.0,
                        3.0,
                    ],
                }
            ),
    }


def scalar_record(
    *,
    dataset_id: str = SCALAR_DATASET_ID,
    fact_dataset_id: str = SOURCE_DATASET_ID,
) -> dict:

    return {
        "dataset_id":
            dataset_id,

        "filename":
            (
                f"{fact_dataset_id}"
                "__overall_price.derived"
            ),

        "dataframe":
            pd.DataFrame(
                {
                    "sum_price": [
                        60.0,
                    ],

                    "event_count": [
                        3,
                    ],
                }
            ),

        "is_derived":
            True,

        "derivation_type":
            "scalar_additive_measure",

        "source_dataset_ids": [
            fact_dataset_id,
        ],

        "provenance": {
            "fact_dataset_id":
                fact_dataset_id,

            "operation":
                "scalar_sum",

            "group_column":
                None,

            "source_measure_column":
                "price",

            "target_measure_column":
                "sum_price",

            "aggregation":
                "sum",

            "grain":
                "overall",

            "metric_semantics":
                TRUSTED_MONETARY_EVENT_SEMANTICS,

            "population_semantics":
                (
                    "Complete validated fact-grain analytical "
                    "population. No temporal, categorical, "
                    "entity or session grouping is applied."
                ),
        },
    }


def single_authority_catalog():

    return (
        planner_catalog_from_dataset_records(
            [
                source_record(),
                scalar_record(),
            ]
        )
    )


def multiple_authority_catalog():

    return (
        planner_catalog_from_dataset_records(
            [
                source_record(),
                scalar_record(),
                source_record(
                    dataset_id=
                        SECOND_SOURCE_DATASET_ID,
                ),
                scalar_record(
                    dataset_id=
                        SECOND_SCALAR_DATASET_ID,

                    fact_dataset_id=
                        SECOND_SOURCE_DATASET_ID,
                ),
            ]
        )
    )


# ============================================================
# CAPTURED-R11-LIKE PROPOSAL
# ============================================================


def noisy_abstention(
    *,
    decision: str = "ambiguous",
    aggregation_function: str = "sum",
    x_column: str | None = None,
    y_column: str | None = "sum_price",
    group_column: str | None = None,
    blockers: list[str] | None = None,
) -> AIPlannerProposal:

    return (
        AIPlannerProposal(
            decision=
                decision,

            title=
                "Revenue Total",

            family=
                "aggregation",

            dataset_id=
                SOURCE_DATASET_ID,

            analytical_grain=
                "entity",

            x_column=
                x_column,

            y_column=
                y_column,

            group_column=
                group_column,

            value_column=
                None,

            time_column=
                None,

            dimension_column=
                None,

            entity_column=
                None,

            aggregation_function=
                aggregation_function,

            ranking_order=
                "descending",

            ranking_limit=
                None,

            window_operation=
                "none",

            window_size=
                None,

            benchmark_reference=
                "overall_aggregate",

            benchmark_operator=
                "gt",

            benchmark_selection=
                "matching_only",

            blockers=
                (
                    blockers
                    if blockers is not None
                    else []
                ),

            reasons=[
                (
                    "Model-authored reasons are intentionally "
                    "ignored by deterministic recovery."
                )
            ],

            confidence=
                0.95,
        )
    )


# ============================================================
# PRECONDITION AUTHORITIES
# ============================================================


def assert_objective_authorities() -> None:

    catalog = (
        single_authority_catalog()
    )


    requirements = (
        extract_objective_requirements(
            objective=
                EXPLICIT_TOTAL_OBJECTIVE,

            catalog=
                catalog,
        )
    )


    assert (
        len(
            requirements
        )
        ==
        1
    )


    requirement = (
        requirements[
            0
        ]
    )


    assert (
        requirement.concept
        ==
        "revenue_total"
    )


    assert (
        requirement.required_aggregation
        ==
        "sum"
    )


    assert (
        requirement.candidate_columns
        ==
        [
            TARGET_MEASURE,
        ]
    )


    assert (
        explicit_aggregation_from_objective(
            EXPLICIT_TOTAL_OBJECTIVE
        )
        ==
        "sum"
    )


    assert (
        explicit_ranking_order_from_objective(
            EXPLICIT_TOTAL_OBJECTIVE
        )
        ==
        "none"
    )


    assert (
        explicit_benchmark_operator_from_objective(
            EXPLICIT_TOTAL_OBJECTIVE
        )
        is None
    )


    ranking = (
        explicit_ranking_order_from_objective(
            EXPLICIT_RANKING_OBJECTIVE
        )
    )


    assert (
        ranking
        !=
        "none"
    ), (
        "Ranking negative guard objective is not "
        "recognized by Decision Coverage."
    )


    benchmark = (
        explicit_benchmark_operator_from_objective(
            EXPLICIT_BENCHMARK_OBJECTIVE
        )
    )


    assert (
        benchmark
        is not None
    ), (
        "Benchmark negative guard objective is not "
        "recognized by Decision Coverage."
    )


# ============================================================
# CURRENT PUBLIC BOUNDARY
# ============================================================


def canonicalize(
    *,
    objective: str,
    proposal: AIPlannerProposal,
    catalog=None,
):

    if catalog is None:
        catalog = single_authority_catalog()


    return (
        canonicalize_analytical_view_intent(
            objective=
                objective,

            proposal=
                proposal,

            catalog=
                catalog,
        )
    )


# ============================================================
# NEGATIVE GUARD 1
#
# A genuinely second metric is not harmless slot noise.
# ============================================================


def assert_second_metric_fails_closed() -> None:

    original = (
        noisy_abstention(
            x_column=
                "other_metric",

            y_column=
                "sum_price",
        )
    )


    canonical, notes = (
        canonicalize(
            objective=
                EXPLICIT_TOTAL_OBJECTIVE,

            proposal=
                original,
        )
    )


    assert (
        canonical.decision
        ==
        "ambiguous"
    ), (
        "An abstention containing a second metric must "
        "remain ambiguous."
    )


    assert (
        notes
        ==
        []
    )


# ============================================================
# NEGATIVE GUARD 2
#
# A real grouping slot changes analytical topology.
# ============================================================


def assert_grouping_fails_closed() -> None:

    original = (
        noisy_abstention(
            group_column=
                "segment",
        )
    )


    canonical, notes = (
        canonicalize(
            objective=
                EXPLICIT_TOTAL_OBJECTIVE,

            proposal=
                original,
        )
    )


    assert (
        canonical.decision
        ==
        "ambiguous"
    ), (
        "An abstention carrying an active grouping role "
        "must remain ambiguous."
    )


    assert (
        notes
        ==
        []
    )


# ============================================================
# NEGATIVE GUARD 3
#
# Explicit ranking in the objective makes ranking semantic,
# not removable model noise.
# ============================================================


def assert_requested_ranking_fails_closed() -> None:

    original = (
        noisy_abstention()
    )


    canonical, notes = (
        canonicalize(
            objective=
                EXPLICIT_RANKING_OBJECTIVE,

            proposal=
                original,
        )
    )


    assert (
        canonical.decision
        ==
        "ambiguous"
    ), (
        "A ranking explicitly requested by the objective "
        "must never be stripped by scalar abstention recovery."
    )


    assert (
        notes
        ==
        []
    )


# ============================================================
# NEGATIVE GUARD 4
#
# Explicit benchmark semantics must not be stripped.
# ============================================================


def assert_requested_benchmark_fails_closed() -> None:

    original = (
        noisy_abstention()
    )


    canonical, notes = (
        canonicalize(
            objective=
                EXPLICIT_BENCHMARK_OBJECTIVE,

            proposal=
                original,
        )
    )


    assert (
        canonical.decision
        ==
        "ambiguous"
    ), (
        "A benchmark explicitly requested by the objective "
        "must never be stripped by scalar abstention recovery."
    )


    assert (
        notes
        ==
        []
    )


# ============================================================
# NEGATIVE GUARD 5
#
# Model blocker is stronger than wire-noise cleanup.
# ============================================================


def assert_blocker_fails_closed() -> None:

    original = (
        noisy_abstention(
            blockers=[
                "A real blocker remains unresolved.",
            ]
        )
    )


    canonical, notes = (
        canonicalize(
            objective=
                EXPLICIT_TOTAL_OBJECTIVE,

            proposal=
                original,
        )
    )


    assert (
        canonical.decision
        ==
        "ambiguous"
    ), (
        "A model-authored blocker must prevent scalar "
        "abstention recovery."
    )


    assert (
        notes
        ==
        []
    )


# ============================================================
# NEGATIVE GUARD 6
#
# Multiple server-owned scalar authorities remain ambiguous.
# ============================================================


def assert_multiple_authorities_fail_closed() -> None:

    original = (
        noisy_abstention()
    )


    canonical, notes = (
        canonicalize(
            objective=
                EXPLICIT_TOTAL_OBJECTIVE,

            proposal=
                original,

            catalog=
                multiple_authority_catalog(),
        )
    )


    assert (
        canonical.decision
        ==
        "ambiguous"
    ), (
        "Multiple compatible scalar authorities must "
        "remain ambiguous."
    )


    assert (
        notes
        ==
        []
    )


# ============================================================
# NEGATIVE GUARD 7
#
# blocked is never promoted.
# ============================================================


def assert_blocked_fails_closed() -> None:

    original = (
        noisy_abstention(
            decision=
                "blocked",
        )
    )


    canonical, notes = (
        canonicalize(
            objective=
                EXPLICIT_TOTAL_OBJECTIVE,

            proposal=
                original,
        )
    )


    assert (
        canonical.decision
        ==
        "blocked"
    ), (
        "A blocked decision must never be promoted."
    )


    assert (
        notes
        ==
        []
    )


# ============================================================
# NEGATIVE GUARD 8
#
# Non-SUM remains fail-closed.
# ============================================================


def assert_non_sum_fails_closed() -> None:

    original = (
        noisy_abstention(
            aggregation_function=
                "mean",
        )
    )


    canonical, notes = (
        canonicalize(
            objective=
                EXPLICIT_TOTAL_OBJECTIVE,

            proposal=
                original,
        )
    )


    assert (
        canonical.decision
        ==
        "ambiguous"
    ), (
        "A non-SUM abstention must not be rewritten "
        "to the scalar SUM authority."
    )


    assert (
        canonical.aggregation_function
        ==
        "mean"
    )


    assert (
        notes
        ==
        []
    )


# ============================================================
# POSITIVE R11 WIRE-NOISE RECOVERY
# ============================================================


def assert_r11_wire_noise_recovery() -> list[str]:

    original = (
        noisy_abstention()
    )


    canonical, notes = (
        canonicalize(
            objective=
                EXPLICIT_TOTAL_OBJECTIVE,

            proposal=
                original,
        )
    )


    gaps: list[
        str
    ] = []


    if (
        canonical.decision
        !=
        "propose"
    ):
        gaps.append(
            "decision_recovery_gap"
        )


    if (
        canonical.dataset_id
        !=
        SCALAR_DATASET_ID
    ):
        gaps.append(
            "scalar_dataset_recovery_gap"
        )


    if (
        canonical.analytical_grain
        !=
        "overall"
    ):
        gaps.append(
            "scalar_grain_recovery_gap"
        )


    if (
        canonical.value_column
        !=
        TARGET_MEASURE
    ):
        gaps.append(
            "scalar_value_recovery_gap"
        )


    if (
        canonical.aggregation_function
        !=
        "sum"
    ):
        gaps.append(
            "scalar_aggregation_recovery_gap"
        )


    # --------------------------------------------------------
    # The only non-null role in captured R11 was y=sum_price.
    #
    # It is the exact Objective Coverage target in the wrong
    # role. After deterministic recovery the target belongs
    # only in value_column.
    # --------------------------------------------------------

    if any(
        value
        is not None

        for value
        in [
            canonical.x_column,
            canonical.y_column,
            canonical.group_column,
            canonical.time_column,
            canonical.dimension_column,
            canonical.entity_column,
        ]
    ):
        gaps.append(
            "role_noise_cleanup_gap"
        )


    if (
        canonical.ranking_order
        !=
        "none"

        or

        canonical.ranking_limit
        is not None
    ):
        gaps.append(
            "ranking_noise_cleanup_gap"
        )


    if any(
        value
        is not None

        for value
        in [
            canonical.benchmark_reference,
            canonical.benchmark_operator,
            canonical.benchmark_selection,
        ]
    ):
        gaps.append(
            "benchmark_noise_cleanup_gap"
        )


    if (
        canonical.window_operation
        !=
        "none"

        or

        canonical.window_size
        is not None
    ):
        gaps.append(
            "window_state_cleanup_gap"
        )


    if (
        canonical.blockers
        !=
        []
    ):
        gaps.append(
            "blocker_cleanup_gap"
        )


    if not notes:
        gaps.append(
            "wire_noise_recovery_note_gap"
        )


    print(
        "Positive original decision               "
        f"{original.decision}"
    )

    print(
        "Positive original y                      "
        f"{original.y_column}"
    )

    print(
        "Positive original ranking                "
        f"{original.ranking_order}"
    )

    print(
        "Positive original benchmark              "
        f"{(
            original.benchmark_reference,
            original.benchmark_operator,
            original.benchmark_selection,
        )}"
    )

    print(
        "Positive canonical decision              "
        f"{canonical.decision}"
    )

    print(
        "Positive canonical dataset               "
        f"{canonical.dataset_id}"
    )

    print(
        "Positive canonical grain                 "
        f"{canonical.analytical_grain}"
    )

    print(
        "Positive canonical value                 "
        f"{canonical.value_column}"
    )

    print(
        "Positive canonical y                     "
        f"{canonical.y_column}"
    )

    print(
        "Positive canonical ranking               "
        f"{canonical.ranking_order}"
    )

    print(
        "Positive canonical benchmark             "
        f"{(
            canonical.benchmark_reference,
            canonical.benchmark_operator,
            canonical.benchmark_selection,
        )}"
    )

    print(
        "Positive notes                           "
        f"{notes}"
    )


    return gaps


# ============================================================
# MAIN
# ============================================================


def main() -> None:

    print(
        "=== DATALENS EXPLICIT SCALAR ABSTENTION "
        "WIRE-NOISE RECOVERY v0.1 ==="
    )

    print()


    assert_objective_authorities()

    print(
        "[PASS] objective authorities are deterministic"
    )


    # Run fail-closed cases BEFORE the expected RED.
    assert_second_metric_fails_closed()

    print(
        "[PASS] second metric remains ambiguous"
    )


    assert_grouping_fails_closed()

    print(
        "[PASS] active grouping remains ambiguous"
    )


    assert_requested_ranking_fails_closed()

    print(
        "[PASS] explicitly requested ranking remains fail-closed"
    )


    assert_requested_benchmark_fails_closed()

    print(
        "[PASS] explicitly requested benchmark remains fail-closed"
    )


    assert_blocker_fails_closed()

    print(
        "[PASS] model blocker remains fail-closed"
    )


    assert_multiple_authorities_fail_closed()

    print(
        "[PASS] multiple scalar authorities remain ambiguous"
    )


    assert_blocked_fails_closed()

    print(
        "[PASS] blocked decision remains blocked"
    )


    assert_non_sum_fails_closed()

    print(
        "[PASS] non-SUM abstention remains fail-closed"
    )


    print()


    gaps = (
        assert_r11_wire_noise_recovery()
    )


    print(
        "Observed gaps                            "
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
        "PASS - explicit scalar abstention "
        "wire-noise recovery v0.1"
    )


if __name__ == "__main__":

    main()