from __future__ import annotations


from app.planning.ai_analytical_planner import (
    AI_ANALYTICAL_PLANNER_RULE_VERSION,
    AIPlannerProposal,
    PlannerCatalog,
    PlannerColumnProfile,
    PlannerDatasetProfile,
    validate_ai_proposal,
)


EXPECTED_VERSION = (
    "ai_analytical_planner_v0.35"
)


OBJECTIVE = (
    "Quelle est la relation entre l'âge du client "
    "au premier achat et son montant total d'achats ?"
)


CUSTOMER_DATASET_ID = (
    "derived:events:customer:customer_id:amount"
)


EXPECTED_RED_GAP = (
    "quantitative_association_"
    "unsolicited_decision_wire_recovery_gap"
)


# ============================================================
# PROFILE HELPERS
# ============================================================


def column(
    name: str,
    analysis_kind: str,
    *,
    dtype: str,
    unique_count: int,
) -> PlannerColumnProfile:

    return PlannerColumnProfile(
        name=name,
        dtype=dtype,
        analysis_kind=analysis_kind,
        missing_ratio=0.0,
        unique_count=unique_count,
        unique_candidate=False,
    )


def customer_dataset() -> PlannerDatasetProfile:

    return PlannerDatasetProfile(
        dataset_id=
            CUSTOMER_DATASET_ID,

        filename=(
            "events__customers_amount.derived"
        ),

        row_count=
            100,

        column_count=
            3,

        columns=[
            column(
                "customer_id",
                "identifier",
                dtype="object",
                unique_count=100,
            ),
            column(
                "age_at_first_purchase",
                "quantitative",
                dtype="float64",
                unique_count=70,
            ),
            column(
                "total_spend",
                "quantitative",
                dtype="float64",
                unique_count=98,
            ),
        ],

        is_derived=
            True,

        derivation_type=(
            "entity_additive_measure"
        ),

        analytical_grain=(
            "customer_id"
        ),

        operation=(
            "customer_behavior_materialization"
        ),

        entity_column=(
            "customer_id"
        ),

        source_measure_column=(
            "amount"
        ),

        target_measure_column=(
            "total_spend"
        ),
    )


def catalog() -> PlannerCatalog:

    return PlannerCatalog(
        datasets=[
            customer_dataset(),
        ]
    )


# ============================================================
# PROPOSAL HELPERS
# ============================================================


def clean_association_proposal() -> AIPlannerProposal:

    return AIPlannerProposal(
        decision=
            "propose",

        title=(
            "Quantitative association"
        ),

        family=(
            "quantitative_association"
        ),

        dataset_id=
            CUSTOMER_DATASET_ID,

        analytical_grain=(
            "customer_id"
        ),

        x_column=(
            "age_at_first_purchase"
        ),

        y_column=(
            "total_spend"
        ),

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
            "none",

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

        blockers=[],

        reasons=[
            (
                "Study the relationship between "
                "two customer-level quantitative "
                "variables."
            )
        ],

        confidence=
            0.95,
    )


def noisy_live_shape_proposal() -> AIPlannerProposal:
    """
    Generic reconstruction of the P5 live failure shape.

    The analytical family, dataset and x/y pair are correct.

    The LLM nevertheless emitted:
    - SUM aggregation;
    - descending ranking;
    - active overall benchmark.

    The user objective asks only for a relationship.
    """

    return AIPlannerProposal(
        decision=
            "propose",

        title=(
            "Quantitative association"
        ),

        family=(
            "quantitative_association"
        ),

        dataset_id=
            CUSTOMER_DATASET_ID,

        analytical_grain=(
            "customer_id"
        ),

        x_column=(
            "age_at_first_purchase"
        ),

        y_column=(
            "total_spend"
        ),

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
            "sum",

        ranking_order=
            "descending",

        ranking_limit=
            None,

        window_operation=
            "none",

        window_size=
            None,

        benchmark_reference=(
            "overall_aggregate"
        ),

        benchmark_operator=(
            "gt"
        ),

        benchmark_selection=(
            "annotate_all"
        ),

        blockers=[],

        reasons=[
            (
                "The requested analysis is a "
                "quantitative relationship."
            )
        ],

        confidence=
            0.95,
    )


def explicit_benchmark_proposal() -> AIPlannerProposal:

    proposal = (
        noisy_live_shape_proposal()
    )

    proposal.aggregation_function = (
        "none"
    )

    proposal.ranking_order = (
        "none"
    )

    return proposal


def explicit_ranking_proposal() -> AIPlannerProposal:

    proposal = (
        noisy_live_shape_proposal()
    )

    proposal.aggregation_function = (
        "none"
    )

    proposal.benchmark_reference = (
        None
    )

    proposal.benchmark_operator = (
        None
    )

    proposal.benchmark_selection = (
        None
    )

    return proposal


# ============================================================
# BASELINE
# ============================================================


def baseline_clean_association_is_valid() -> None:

    item = validate_ai_proposal(
        objective=
            OBJECTIVE,

        proposal=(
            clean_association_proposal()
        ),

        proposal_index=
            1,

        catalog=(
            catalog()
        ),
    )


    assert (
        item.validation_status
        ==
        "validated"
    ), item.model_dump()


    assert (
        item.contract
        is not None
    ), item.model_dump()


    assert (
        item.proposal.family
        ==
        "quantitative_association"
    )


    assert (
        item.proposal.dataset_id
        ==
        CUSTOMER_DATASET_ID
    )


    assert (
        item.proposal.x_column
        ==
        "age_at_first_purchase"
    )


    assert (
        item.proposal.y_column
        ==
        "total_spend"
    )


    assert (
        item.proposal.aggregation_function
        ==
        "none"
    )


    assert (
        item.proposal.ranking_order
        ==
        "none"
    )


    assert (
        item.proposal.benchmark_reference
        is None
    )


    print(
        "[PASS] clean quantitative association baseline"
    )


# ============================================================
# CURRENT RED / FUTURE GREEN
# ============================================================


def evaluate_unsolicited_wire_recovery() -> tuple[
    bool,
    object,
]:

    item = validate_ai_proposal(
        objective=
            OBJECTIVE,

        proposal=(
            noisy_live_shape_proposal()
        ),

        proposal_index=
            1,

        catalog=(
            catalog()
        ),
    )


    aggregation_clean = bool(
        item.proposal.aggregation_function
        ==
        "none"
    )


    ranking_clean = bool(
        item.proposal.ranking_order
        ==
        "none"
        and
        item.proposal.ranking_limit
        is None
    )


    benchmark_clean = bool(
        item.proposal.benchmark_reference
        is None
        and
        item.proposal.benchmark_operator
        is None
        and
        item.proposal.benchmark_selection
        is None
    )


    ranking_note = any(
        (
            "ranking"
            in
            note.casefold()
            or
            "classement"
            in
            note.casefold()
        )

        for note
        in item.normalizations
    )


    benchmark_note = any(
        (
            "benchmark"
            in
            note.casefold()
            or
            "référence"
            in
            note.casefold()
            or
            "reference"
            in
            note.casefold()
        )

        for note
        in item.normalizations
    )


    recovered = bool(
        item.validation_status
        ==
        "validated"
        and
        item.contract
        is not None
        and
        item.proposal.family
        ==
        "quantitative_association"
        and
        item.proposal.dataset_id
        ==
        CUSTOMER_DATASET_ID
        and
        item.proposal.x_column
        ==
        "age_at_first_purchase"
        and
        item.proposal.y_column
        ==
        "total_spend"
        and
        aggregation_clean
        and
        ranking_clean
        and
        benchmark_clean
        and
        ranking_note
        and
        benchmark_note
    )


    print(
        "Unsolicited live-shape recovery          "
        f"{recovered}"
    )

    print(
        "  validation status                     "
        f"{item.validation_status}"
    )

    print(
        "  aggregation clean                     "
        f"{aggregation_clean}"
    )

    print(
        "  ranking clean                         "
        f"{ranking_clean}"
    )

    print(
        "  benchmark clean                       "
        f"{benchmark_clean}"
    )

    print(
        "  ranking normalization noted           "
        f"{ranking_note}"
    )

    print(
        "  benchmark normalization noted         "
        f"{benchmark_note}"
    )

    print(
        "  normalizations                        "
        f"{item.normalizations}"
    )

    print(
        "  errors                                "
        f"{item.errors}"
    )


    return (
        recovered,
        item,
    )


# ============================================================
# NEGATIVE GUARD — EXPLICIT BENCHMARK
# ============================================================


def explicit_benchmark_is_not_silently_erased() -> None:

    objective = (
        OBJECTIVE
        +
        " Compare aussi le montant total "
        "d'achats à la moyenne générale."
    )


    item = validate_ai_proposal(
        objective=
            objective,

        proposal=(
            explicit_benchmark_proposal()
        ),

        proposal_index=
            1,

        catalog=(
            catalog()
        ),
    )


    silently_erased_into_valid_association = bool(
        item.validation_status
        ==
        "validated"
        and
        item.proposal.benchmark_reference
        is None
        and
        item.proposal.benchmark_operator
        is None
        and
        item.proposal.benchmark_selection
        is None
    )


    assert (
        not silently_erased_into_valid_association
    ), item.model_dump()


    print(
        "[PASS] explicit benchmark intent is not silently erased"
    )


# ============================================================
# NEGATIVE GUARD — EXPLICIT RANKING
# ============================================================


def explicit_ranking_is_not_silently_erased() -> None:

    objective = (
        OBJECTIVE
        +
        " Classe aussi les clients du montant "
        "total d'achats le plus élevé au plus faible."
    )


    item = validate_ai_proposal(
        objective=
            objective,

        proposal=(
            explicit_ranking_proposal()
        ),

        proposal_index=
            1,

        catalog=(
            catalog()
        ),
    )


    silently_erased_into_valid_association = bool(
        item.validation_status
        ==
        "validated"
        and
        item.proposal.ranking_order
        ==
        "none"
        and
        item.proposal.ranking_limit
        is None
    )


    assert (
        not silently_erased_into_valid_association
    ), item.model_dump()


    print(
        "[PASS] explicit ranking intent is not silently erased"
    )


# ============================================================
# MAIN
# ============================================================


def main() -> None:

    print(
        "=== DATALENS QUANTITATIVE ASSOCIATION "
        "UNSOLICITED DECISION WIRE RECOVERY v0.1 ==="
    )

    print()


    assert (
        AI_ANALYTICAL_PLANNER_RULE_VERSION
        ==
        EXPECTED_VERSION
    ), AI_ANALYTICAL_PLANNER_RULE_VERSION


    print(
        "[PASS] planner version v0.35"
    )


    baseline_clean_association_is_valid()


    (
        recovered,
        item,
    ) = (
        evaluate_unsolicited_wire_recovery()
    )


    explicit_benchmark_is_not_silently_erased()

    explicit_ranking_is_not_silently_erased()


    gaps: list[
        str
    ] = []


    if not recovered:

        gaps.append(
            EXPECTED_RED_GAP
        )


    print()
    print("=" * 80)
    print(
        "P5-R1 UNSOLICITED DECISION WIRE MATRIX"
    )
    print("=" * 80)
    print()


    print(
        "Clean association baseline               True"
    )

    print(
        "Unsolicited aggregation removed          "
        f"{item.proposal.aggregation_function == 'none'}"
    )

    print(
        "Unsolicited ranking removed              "
        f"{item.proposal.ranking_order == 'none'}"
    )

    print(
        "Unsolicited benchmark removed            "
        f"{item.proposal.benchmark_reference is None}"
    )

    print(
        "Noisy association validates              "
        f"{recovered}"
    )

    print(
        "Explicit benchmark guard                 True"
    )

    print(
        "Explicit ranking guard                   True"
    )

    print()

    print(
        "Observed gaps                            "
        f"{gaps}"
    )


    # ========================================================
    # RED NOW, GREEN AFTER PRODUCTION FIX
    # ========================================================

    if gaps:

        if gaps != [
            EXPECTED_RED_GAP
        ]:

            raise AssertionError(
                (
                    "Unexpected P5-R1 gap set: "
                    f"{gaps!r}"
                )
            )


        raise AssertionError(
            (
                "RED_EXPECTED: "
                +
                EXPECTED_RED_GAP
            )
        )


    print()

    print(
        "PASS - quantitative association "
        "unsolicited decision wire recovery v0.1"
    )

    print(
        "P5-R1 REGRESSION: GREEN"
    )


if __name__ == "__main__":

    main()