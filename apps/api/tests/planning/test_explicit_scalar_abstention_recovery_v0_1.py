from __future__ import annotations


import pandas as pd


from app.planning.planner_catalog import (
    planner_catalog_from_dataset_records,
)


from app.planning.ai_analytical_planner import (
    AIPlannerProposal,
    canonicalize_analytical_view_intent,
)


# ============================================================
# OBJECTIVES
# ============================================================


EXPLICIT_TOTAL_OBJECTIVE = (
    "Quel est le chiffre d'affaires total "
    "sur toute la période analysée ?"
)


VAGUE_OBJECTIVE = (
    "Analyse les performances commerciales."
)


# ============================================================
# SERVER-OWNED SEMANTIC AUTHORITY
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


SCALAR_DATASET_ID = (
    "derived:dataset_source:scalar:price"
)


SECOND_SCALAR_DATASET_ID = (
    "derived:dataset_source_2:scalar:price"
)


# ============================================================
# DATASET RECORDS
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

                    "price": [
                        10.0,
                        20.0,
                        30.0,
                    ],
                }
            ),
    }


def scalar_record(
    *,
    dataset_id: str,
    fact_dataset_id: str,
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
                        60.0
                    ],

                    "event_count": [
                        3
                    ],
                }
            ),

        "is_derived":
            True,

        "derivation_type":
            "scalar_additive_measure",

        "source_dataset_ids": [
            fact_dataset_id
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


# ============================================================
# PROPOSALS
# ============================================================


def abstention_proposal(
    *,
    decision: str = "ambiguous",
    aggregation_function: str = "sum",
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
                None,

            y_column=
                None,

            group_column=
                None,

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
                "none",

            ranking_limit=
                None,

            window_operation=
                "none",

            window_size=
                None,

            benchmark_reference=
                None,

            benchmark_operator=
                None,

            benchmark_selection=
                None,

            blockers=
                [],

            reasons=[
                (
                    "Model-authored reasons are deliberately "
                    "not evidence for deterministic recovery."
                )
            ],

            confidence=
                0.95,
        )
    )


# ============================================================
# POSITIVE CASE
# ============================================================


def assert_explicit_unique_scalar_recovery() -> list[
    str
]:

    catalog = (
        planner_catalog_from_dataset_records(
            [
                source_record(),
                scalar_record(
                    dataset_id=
                        SCALAR_DATASET_ID,

                    fact_dataset_id=
                        SOURCE_DATASET_ID,
                ),
            ]
        )
    )


    proposal = (
        abstention_proposal()
    )


    (
        canonical,
        notes,
    ) = (
        canonicalize_analytical_view_intent(
            objective=
                EXPLICIT_TOTAL_OBJECTIVE,

            proposal=
                proposal,

            catalog=
                catalog,
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
        "sum_price"
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


    if (
        canonical.group_column
        is not None

        or

        canonical.dimension_column
        is not None

        or

        canonical.time_column
        is not None

        or

        canonical.entity_column
        is not None

        or

        canonical.x_column
        is not None

        or

        canonical.y_column
        is not None
    ):

        gaps.append(
            "scalar_role_cleanup_gap"
        )


    if (
        canonical.ranking_order
        !=
        "none"

        or

        canonical.ranking_limit
        is not None

        or

        canonical.window_operation
        !=
        "none"

        or

        canonical.window_size
        is not None

        or

        canonical.benchmark_reference
        is not None

        or

        canonical.benchmark_operator
        is not None

        or

        canonical.benchmark_selection
        is not None
    ):

        gaps.append(
            "scalar_auxiliary_state_gap"
        )


    if (
        canonical.blockers
        !=
        []
    ):

        gaps.append(
            "scalar_blocker_cleanup_gap"
        )


    if not notes:

        gaps.append(
            "scalar_recovery_note_gap"
        )


    print(
        "Positive original decision               "
        f"{proposal.decision}"
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
        "Positive canonical aggregation           "
        f"{canonical.aggregation_function}"
    )

    print(
        "Positive notes                           "
        f"{notes}"
    )


    return gaps


# ============================================================
# NEGATIVE GUARD 1:
# TWO SERVER-OWNED SCALAR AUTHORITIES => REMAIN AMBIGUOUS
# ============================================================


def assert_multiple_scalar_authorities_fail_closed() -> None:

    second_source_id = (
        "dataset_source_2"
    )


    catalog = (
        planner_catalog_from_dataset_records(
            [
                source_record(),
                scalar_record(
                    dataset_id=
                        SCALAR_DATASET_ID,

                    fact_dataset_id=
                        SOURCE_DATASET_ID,
                ),
                source_record(
                    dataset_id=
                        second_source_id,
                ),
                scalar_record(
                    dataset_id=
                        SECOND_SCALAR_DATASET_ID,

                    fact_dataset_id=
                        second_source_id,
                ),
            ]
        )
    )


    original = (
        abstention_proposal()
    )


    (
        canonical,
        notes,
    ) = (
        canonicalize_analytical_view_intent(
            objective=
                EXPLICIT_TOTAL_OBJECTIVE,

            proposal=
                original,

            catalog=
                catalog,
        )
    )


    assert (
        canonical.decision
        ==
        "ambiguous"
    ), (
        "Several compatible scalar authorities must remain "
        "ambiguous."
    )


    assert (
        canonical.value_column
        is None
    )


    assert (
        notes
        ==
        []
    )


# ============================================================
# NEGATIVE GUARD 2:
# VAGUE OBJECTIVE => REMAIN AMBIGUOUS
# ============================================================


def assert_vague_objective_fails_closed() -> None:

    catalog = (
        planner_catalog_from_dataset_records(
            [
                source_record(),
                scalar_record(
                    dataset_id=
                        SCALAR_DATASET_ID,

                    fact_dataset_id=
                        SOURCE_DATASET_ID,
                ),
            ]
        )
    )


    original = (
        abstention_proposal()
    )


    (
        canonical,
        notes,
    ) = (
        canonicalize_analytical_view_intent(
            objective=
                VAGUE_OBJECTIVE,

            proposal=
                original,

            catalog=
                catalog,
        )
    )


    assert (
        canonical.decision
        ==
        "ambiguous"
    ), (
        "A vague business objective must not be converted "
        "into an executable scalar aggregation."
    )


    assert (
        canonical.value_column
        is None
    )


    assert (
        notes
        ==
        []
    )


# ============================================================
# NEGATIVE GUARD 3:
# BLOCKED IS STRONGER THAN AMBIGUOUS
# ============================================================


def assert_blocked_never_recovered() -> None:

    catalog = (
        planner_catalog_from_dataset_records(
            [
                source_record(),
                scalar_record(
                    dataset_id=
                        SCALAR_DATASET_ID,

                    fact_dataset_id=
                        SOURCE_DATASET_ID,
                ),
            ]
        )
    )


    original = (
        abstention_proposal(
            decision=
                "blocked",
        )
    )


    (
        canonical,
        notes,
    ) = (
        canonicalize_analytical_view_intent(
            objective=
                EXPLICIT_TOTAL_OBJECTIVE,

            proposal=
                original,

            catalog=
                catalog,
        )
    )


    assert (
        canonical.decision
        ==
        "blocked"
    ), (
        "A blocked planner decision must never be promoted "
        "by scalar abstention recovery."
    )


    assert (
        notes
        ==
        []
    )


# ============================================================
# NEGATIVE GUARD 4:
# NON-SUM ABSTENTION DOES NOT BECOME TOTAL SUM
# ============================================================


def assert_non_sum_abstention_fails_closed() -> None:

    catalog = (
        planner_catalog_from_dataset_records(
            [
                source_record(),
                scalar_record(
                    dataset_id=
                        SCALAR_DATASET_ID,

                    fact_dataset_id=
                        SOURCE_DATASET_ID,
                ),
            ]
        )
    )


    original = (
        abstention_proposal(
            aggregation_function=
                "mean",
        )
    )


    (
        canonical,
        notes,
    ) = (
        canonicalize_analytical_view_intent(
            objective=
                EXPLICIT_TOTAL_OBJECTIVE,

            proposal=
                original,

            catalog=
                catalog,
        )
    )


    assert (
        canonical.decision
        ==
        "ambiguous"
    ), (
        "Scalar abstention recovery must not silently replace "
        "a non-SUM model decision."
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
# MAIN
# ============================================================


def main() -> None:

    print(
        "=== DATALENS EXPLICIT SCALAR "
        "ABSTENTION RECOVERY v0.1 ==="
    )

    print()


    # Run fail-closed guards FIRST so the expected RED proves
    # only the missing positive recovery boundary.
    assert_multiple_scalar_authorities_fail_closed()

    print(
        "[PASS] multiple scalar authorities remain ambiguous"
    )


    assert_vague_objective_fails_closed()

    print(
        "[PASS] vague objective remains ambiguous"
    )


    assert_blocked_never_recovered()

    print(
        "[PASS] blocked decision remains blocked"
    )


    assert_non_sum_abstention_fails_closed()

    print(
        "[PASS] non-SUM abstention remains fail-closed"
    )


    print()


    gaps = (
        assert_explicit_unique_scalar_recovery()
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
        "PASS - explicit scalar abstention recovery v0.1"
    )


if __name__ == "__main__":

    main()