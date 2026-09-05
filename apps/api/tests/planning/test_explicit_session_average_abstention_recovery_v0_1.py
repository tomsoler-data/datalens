from __future__ import annotations


from app.planning.ai_analytical_planner import (
    AI_ANALYTICAL_PLANNER_RULE_VERSION,
    AIPlannerProposal,
    PlannerCatalog,
    PlannerColumnProfile,
    PlannerDatasetProfile,
    explicit_aggregation_from_objective,
    validate_ai_proposal,
)


# ============================================================
# GENERIC AUTHORITIES
# ============================================================


OBJECTIVE = (
    "Quel est le panier moyen par session d'achat ?"
)


SESSION_DATASET_ID = (
    "derived:dataset_orders:session:session_id:amount"
)


SESSION_FILENAME = (
    "orders__sessions_amount.derived"
)


SESSION_GRAIN = (
    "session_id"
)


SESSION_MEASURE = (
    "basket_amount"
)


# ============================================================
# CATALOG BUILDERS
# ============================================================


def column(
    *,
    name: str,
    kind: str,
    dtype: str,
    unique_count: int,
    unique_candidate: bool = False,
) -> PlannerColumnProfile:

    return (
        PlannerColumnProfile(
            name=
                name,

            dtype=
                dtype,

            analysis_kind=
                kind,

            missing_ratio=
                0.0,

            unique_count=
                unique_count,

            unique_candidate=
                unique_candidate,
        )
    )


def session_dataset(
    *,
    include_session_id: bool = True,
    target_measure: str = SESSION_MEASURE,
) -> PlannerDatasetProfile:

    columns = []


    if include_session_id:

        columns.append(
            column(
                name=
                    SESSION_GRAIN,

                kind=
                    "identifier",

                dtype=
                    "str",

                unique_count=
                    100,

                unique_candidate=
                    True,
            )
        )


    columns.extend(
        [
            column(
                name=
                    target_measure,

                kind=
                    "quantitative",

                dtype=
                    "float64",

                unique_count=
                    90,
            ),

            column(
                name=
                    "item_count",

                kind=
                    "quantitative",

                dtype=
                    "int64",

                unique_count=
                    8,
            ),
        ]
    )


    return (
        PlannerDatasetProfile(
            dataset_id=
                SESSION_DATASET_ID,

            filename=
                SESSION_FILENAME,

            row_count=
                100,

            column_count=
                len(
                    columns
                ),

            columns=
                columns,

            is_derived=
                True,

            derivation_type=
                "entity_additive_measure",

            analytical_grain=
                SESSION_GRAIN,

            operation=
                "session_materialization",

            aggregation=
                "sum",

            group_column=
                None,

            entity_column=
                SESSION_GRAIN,

            source_time_column=
                None,

            target_time_column=
                None,

            source_measure_column=
                "amount",

            target_measure_column=
                target_measure,

            source_measure_formula=
                None,

            metric_semantics=
                (
                    "One deterministic row represents one "
                    "validated purchase session."
                ),

            measure_semantic_aliases=[
                "amount",
                target_measure,
            ],
        )
    )


def catalog(
    *,
    include_session_id: bool = True,
) -> PlannerCatalog:

    return (
        PlannerCatalog(
            datasets=[
                session_dataset(
                    include_session_id=
                        include_session_id,
                )
            ]
        )
    )


# ============================================================
# PROPOSAL BUILDERS
# ============================================================


def blocked_session_average_proposal(
    *,
    objective_value_column: str = SESSION_MEASURE,
    analytical_grain: str = SESSION_GRAIN,
    blocker: str = "No session_id column in the dataset",
    aggregation_function: str = "sum",
    ranking_order: str = "descending",
) -> AIPlannerProposal:

    return (
        AIPlannerProposal(
            decision=
                "blocked",

            title=
                "Session metric unavailable",

            family=
                "aggregation",

            # Deliberately the exact catalog filename rather
            # than the canonical server-owned dataset_id.
            dataset_id=
                SESSION_FILENAME,

            analytical_grain=
                analytical_grain,

            x_column=
                None,

            y_column=
                None,

            group_column=
                None,

            value_column=
                objective_value_column,

            time_column=
                None,

            dimension_column=
                None,

            entity_column=
                None,

            aggregation_function=
                aggregation_function,  # type: ignore[arg-type]

            ranking_order=
                ranking_order,  # type: ignore[arg-type]

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

            blockers=[
                blocker,
            ],

            reasons=[
                "Model abstention.",
            ],

            confidence=
                0.9,
        )
    )


# ============================================================
# VALIDATION HELPER
# ============================================================


def validate(
    *,
    objective: str = OBJECTIVE,
    proposal: AIPlannerProposal,
    planner_catalog: PlannerCatalog | None = None,
):

    return (
        validate_ai_proposal(
            objective=
                objective,

            proposal=
                proposal,

            proposal_index=
                1,

            catalog=(
                planner_catalog
                or
                catalog()
            ),
        )
    )


# ============================================================
# MAIN
# ============================================================


def main() -> None:

    print(
        "=== DATALENS EXPLICIT SESSION AVERAGE "
        "ABSTENTION RECOVERY v0.1 ==="
    )

    print()


    # ========================================================
    # 1. EXISTING OBJECTIVE AUTHORITY
    #
    # "moyen" is already deterministically recognized as mean.
    # This RED must not introduce a new natural-language
    # aggregation detector.
    # ========================================================

    explicit_aggregation = (
        explicit_aggregation_from_objective(
            OBJECTIVE
        )
    )


    assert (
        explicit_aggregation
        ==
        "mean"
    ), (
        "Existing explicit aggregation authority no longer "
        "recognizes `moyen` as mean."
    )


    print(
        "[PASS] explicit objective aggregation = mean"
    )


    # ========================================================
    # 2. POSITIVE RECOVERY TARGET
    #
    # Server-owned catalog proves all of the following:
    #
    # - exact filename -> canonical dataset_id;
    # - session_materialization;
    # - analytical grain=session_id;
    # - session_id physically exists;
    # - target_measure=basket_amount;
    # - raw proposal already selected basket_amount;
    # - objective explicitly requires mean.
    #
    # The blocker is therefore contradicted by deterministic
    # catalog authority.
    # ========================================================

    positive = (
        validate(
            proposal=
                blocked_session_average_proposal(),
        )
    )


    positive_contract = (
        positive.contract
    )


    positive_validated = (
        positive.validation_status
        ==
        "validated"
    )


    canonical_dataset = (
        positive_contract
        is not None
        and
        positive_contract.required_dataset_ids
        ==
        [
            SESSION_DATASET_ID,
        ]
    )


    positive_aggregation = (
        positive_contract.aggregation
        if positive_contract
        is not None
        else None
    )


    aggregation_is_mean = (
        positive_aggregation
        is not None
        and
        positive_aggregation.function
        ==
        "mean"
    )


    scalar_session_mean = (
        positive_aggregation
        is not None
        and
        positive_aggregation.group_by_roles
        ==
        []
    )


    value_bindings = (
        [
            binding

            for binding
            in positive_contract.bindings

            if (
                binding.role
                ==
                "value"
            )
        ]

        if positive_contract
        is not None
        else []
    )


    measure_is_preserved = (
        len(
            value_bindings
        )
        ==
        1
        and
        value_bindings[
            0
        ].column
        ==
        SESSION_MEASURE
    )


    ranking_cleared = (
        positive_contract
        is not None
        and
        positive_contract.ranking
        is None
    )


    benchmark_absent = (
        positive_contract
        is not None
        and
        positive_contract.benchmark
        is None
    )


    share_absent = (
        positive_contract
        is not None
        and
        positive_contract.share_of_total
        is None
    )


    print()
    print(
        "Planner rule                            "
        f"{AI_ANALYTICAL_PLANNER_RULE_VERSION}"
    )

    print(
        "Positive raw decision                   blocked"
    )

    print(
        "Positive raw dataset_id                 "
        f"{SESSION_FILENAME}"
    )

    print(
        "Positive expected dataset_id            "
        f"{SESSION_DATASET_ID}"
    )

    print(
        "Positive raw measure                    "
        f"{SESSION_MEASURE}"
    )

    print(
        "Positive objective aggregation          "
        f"{explicit_aggregation}"
    )

    print(
        "Positive validation status              "
        f"{positive.validation_status}"
    )

    print(
        "Positive canonical dataset              "
        f"{canonical_dataset}"
    )

    print(
        "Positive aggregation mean               "
        f"{aggregation_is_mean}"
    )

    print(
        "Positive scalar session mean            "
        f"{scalar_session_mean}"
    )

    print(
        "Positive measure preserved              "
        f"{measure_is_preserved}"
    )

    print(
        "Positive ranking cleared                "
        f"{ranking_cleared}"
    )

    print(
        "Positive benchmark absent               "
        f"{benchmark_absent}"
    )

    print(
        "Positive share absent                   "
        f"{share_absent}"
    )


    # ========================================================
    # 3. TRUE BLOCKER — SESSION COLUMN REALLY ABSENT
    #
    # A recovery MUST NOT invent session_id.
    # ========================================================

    true_missing_session = (
        validate(
            proposal=
                blocked_session_average_proposal(),

            planner_catalog=
                catalog(
                    include_session_id=
                        False,
                ),
        )
    )


    assert (
        true_missing_session.validation_status
        ==
        "blocked"
    ), (
        "A genuinely absent session grain must remain blocked."
    )


    print(
        "[PASS] genuinely absent session_id remains blocked"
    )


    # ========================================================
    # 4. WRONG MEASURE — FAIL CLOSED
    #
    # The raw proposal must already identify the exact
    # server-owned target measure. Python must not guess
    # basket_amount from some unrelated quantitative field.
    # ========================================================

    wrong_measure = (
        validate(
            proposal=
                blocked_session_average_proposal(
                    objective_value_column=
                        "item_count",
                ),
        )
    )


    assert (
        wrong_measure.validation_status
        ==
        "blocked"
    ), (
        "Wrong session measure must remain blocked."
    )


    print(
        "[PASS] wrong session measure remains blocked"
    )


    # ========================================================
    # 5. WRONG GRAIN — FAIL CLOSED
    # ========================================================

    wrong_grain = (
        validate(
            proposal=
                blocked_session_average_proposal(
                    analytical_grain=
                        "customer_id",
                ),
        )
    )


    assert (
        wrong_grain.validation_status
        ==
        "blocked"
    ), (
        "Wrong analytical grain must remain blocked."
    )


    print(
        "[PASS] wrong analytical grain remains blocked"
    )


    # ========================================================
    # 6. UNRELATED BLOCKER — FAIL CLOSED
    #
    # Only a blocker contradicted by the exact server-owned
    # session authority may be recovered.
    # ========================================================

    unrelated_blocker = (
        validate(
            proposal=
                blocked_session_average_proposal(
                    blocker=
                        "Requested currency conversion is unavailable",
                ),
        )
    )


    assert (
        unrelated_blocker.validation_status
        ==
        "blocked"
    ), (
        "An unrelated blocker must remain blocked."
    )


    print(
        "[PASS] unrelated blocker remains blocked"
    )


    # ========================================================
    # 7. NO EXPLICIT AVERAGE — FAIL CLOSED
    #
    # We do not silently turn an arbitrary session request into
    # mean(basket_amount).
    # ========================================================

    vague_objective = (
        validate(
            objective=
                "Analyse les sessions d'achat.",

            proposal=
                blocked_session_average_proposal(),
        )
    )


    assert (
        vague_objective.validation_status
        ==
        "blocked"
    ), (
        "A vague session request must remain blocked."
    )


    print(
        "[PASS] vague session objective remains blocked"
    )


    # ========================================================
    # 8. AMBIGUOUS DECISION — FAIL CLOSED
    # ========================================================

    ambiguous_proposal = (
        blocked_session_average_proposal()
        .model_copy(
            update={
                "decision":
                    "ambiguous",

                "blockers":
                    [
                        "Several session measures may be intended.",
                    ],
            }
        )
    )


    ambiguous = (
        validate(
            proposal=
                ambiguous_proposal,
        )
    )


    assert (
        ambiguous.validation_status
        ==
        "ambiguous"
    ), (
        "An ambiguous proposal must not be promoted."
    )


    print(
        "[PASS] ambiguous proposal remains ambiguous"
    )


    # ========================================================
    # 9. RED CLASSIFICATION
    # ========================================================

    gaps: list[
        str
    ] = []


    if not positive_validated:

        gaps.append(
            "session_false_abstention_recovery_gap"
        )


    if not canonical_dataset:

        gaps.append(
            "session_filename_dataset_canonicalization_gap"
        )


    if not aggregation_is_mean:

        gaps.append(
            "session_explicit_mean_canonicalization_gap"
        )


    if not scalar_session_mean:

        gaps.append(
            "session_scalar_aggregation_grain_gap"
        )


    if not measure_is_preserved:

        gaps.append(
            "session_target_measure_preservation_gap"
        )


    if not ranking_cleared:

        gaps.append(
            "session_unsolicited_ranking_cleanup_gap"
        )


    if not benchmark_absent:

        gaps.append(
            "session_unsolicited_benchmark_gap"
        )


    if not share_absent:

        gaps.append(
            "session_unsolicited_share_gap"
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
        "PASS - explicit session average abstention recovery v0.1"
    )


if __name__ == "__main__":

    main()