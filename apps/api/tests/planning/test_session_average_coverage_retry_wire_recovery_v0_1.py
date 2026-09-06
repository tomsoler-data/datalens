from __future__ import annotations


from app.planning.ai_analytical_planner import (
    AI_ANALYTICAL_PLANNER_RULE_VERSION,
    AIPlannerProposal,
    PlannerCatalog,
    PlannerColumnProfile,
    PlannerDatasetProfile,
    validate_ai_proposal,
)


from app.planning.objective_coverage import (
    OBJECTIVE_COVERAGE_RULE_VERSION,
    extract_objective_requirements,
)


# ============================================================
# GENERIC AUTHORITIES
# ============================================================


OBJECTIVE = (
    "What is the average basket per purchase session?"
)


SESSION_DATASET_ID = (
    "derived:dataset_orders:session:session_id:amount"
)


SESSION_FILENAME = (
    "orders__sessions_amount.derived"
)


SESSION_FILENAME_STEM = (
    "orders__sessions_amount"
)


SESSION_GRAIN = (
    "session_id"
)


TARGET_MEASURE = (
    "basket_amount"
)


REQUIREMENT_ID = (
    "metric:average_basket"
)


# ============================================================
# CATALOG
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
    dataset_id: str = SESSION_DATASET_ID,
    filename: str = SESSION_FILENAME,
    target_measure: str = TARGET_MEASURE,
) -> PlannerDatasetProfile:

    columns = [
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
        ),

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


    return (
        PlannerDatasetProfile(
            dataset_id=
                dataset_id,

            filename=
                filename,

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
                    "One deterministic row represents "
                    "one validated purchase session."
                ),

            measure_semantic_aliases=[
                "amount",
                target_measure,
            ],
        )
    )


def catalog(
    *,
    duplicate_session_authority: bool = False,
) -> PlannerCatalog:

    datasets = [
        session_dataset()
    ]


    if duplicate_session_authority:

        datasets.append(
            session_dataset(
                dataset_id=
                    "derived:dataset_orders_2:session:session_id:amount",

                filename=
                    "orders__sessions_amount_copy.derived",
            )
        )


    return (
        PlannerCatalog(
            datasets=
                datasets
        )
    )


# ============================================================
# EXACT LIVE-STYLE COVERAGE RETRY ABSTENTION
# ============================================================


def live_style_blocked_proposal(
    *,
    dataset_id: str = SESSION_FILENAME_STEM,
    y_column: str = TARGET_MEASURE,
    entity_column: str = SESSION_GRAIN,
    blocker_mode: str = "coverage",
) -> AIPlannerProposal:

    if (
        blocker_mode
        ==
        "coverage"
    ):

        blockers = [
            (
                "The objective contains no supported explicit "
                "benchmark comparison."
            ),
            (
                "The validated analytical plan does not preserve "
                "every deterministic requirement extracted from "
                "the user request."
            ),
            (
                "Missing required concept=average_basket; "
                "type=metric; compatible catalog "
                "column(s)=basket_amount; allowed role(s)=value; "
                "required aggregation=mean."
            ),
        ]


        reasons = [
            (
                "The validated analytical plan does not preserve "
                "every deterministic requirement extracted from "
                "the user request."
            ),
            (
                "Missing required concept: average_basket; "
                "type=metric; compatible catalog "
                "column(s)=basket_amount; allowed role(s)=value; "
                "required aggregation=mean."
            ),
        ]


    elif (
        blocker_mode
        ==
        "unrelated"
    ):

        blockers = [
            (
                "Requested currency conversion is unavailable."
            )
        ]


        reasons = [
            "Model abstention."
        ]


    else:

        raise ValueError(
            blocker_mode
        )


    return (
        AIPlannerProposal(
            decision=
                "blocked",

            title=
                "Missing deterministic requirement",

            family=
                "aggregation",

            # Exact live pattern:
            # server filename stem without `.derived`.
            dataset_id=
                dataset_id,

            analytical_grain=
                SESSION_GRAIN,

            x_column=
                None,

            # Exact live pattern:
            # correct physical candidate in y instead of value.
            y_column=
                y_column,

            group_column=
                None,

            value_column=
                None,

            time_column=
                None,

            dimension_column=
                None,

            # Exact live pattern:
            # session grain also emitted in entity.
            entity_column=
                entity_column,

            # Exact Objective Coverage requirement.
            aggregation_function=
                "mean",

            # Unsolicited ranking wire noise.
            ranking_order=
                "descending",

            ranking_limit=
                None,

            window_operation=
                "none",

            window_size=
                None,

            # Exact live partial benchmark noise.
            benchmark_reference=
                "overall_aggregate",

            benchmark_operator=
                "gt",

            benchmark_selection=
                None,

            blockers=
                blockers,

            reasons=
                reasons,

            confidence=
                0.8,
        )
    )


# ============================================================
# VALIDATION
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
        "=== DATALENS SESSION AVERAGE COVERAGE-RETRY "
        "WIRE RECOVERY v0.1 ==="
    )

    print()


    planner_catalog = (
        catalog()
    )


    # ========================================================
    # 1. OBJECTIVE COVERAGE IS NOW DETERMINISTIC AUTHORITY
    # ========================================================

    requirements = (
        extract_objective_requirements(
            objective=
                OBJECTIVE,

            catalog=
                planner_catalog,
        )
    )


    average_requirements = [
        requirement

        for requirement
        in requirements

        if (
            requirement.requirement_id
            ==
            REQUIREMENT_ID
        )
    ]


    assert (
        len(
            average_requirements
        )
        ==
        1
    )


    requirement = (
        average_requirements[
            0
        ]
    )


    assert (
        requirement.candidate_columns
        ==
        [
            TARGET_MEASURE
        ]
    )


    assert (
        requirement.allowed_roles
        ==
        [
            "value"
        ]
    )


    assert (
        requirement.required_aggregation
        ==
        "mean"
    )


    print(
        "[PASS] Objective Coverage authority = "
        "average_basket / basket_amount / value / mean"
    )


    # ========================================================
    # 2. POSITIVE LIVE-STYLE RECOVERY TARGET
    # ========================================================

    positive = (
        validate(
            proposal=
                live_style_blocked_proposal(),
        )
    )


    positive_proposal = (
        positive.proposal
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
        positive_proposal.dataset_id
        ==
        SESSION_DATASET_ID
    )


    canonical_value_wire = (
        positive_proposal.value_column
        ==
        TARGET_MEASURE
        and
        positive_proposal.y_column
        is None
        and
        positive_proposal.entity_column
        is None
    )


    ranking_wire_cleared = (
        positive_proposal.ranking_order
        ==
        "none"
        and
        positive_proposal.ranking_limit
        is None
    )


    partial_benchmark_cleared = all(
        value is None

        for value
        in (
            positive_proposal.benchmark_reference,
            positive_proposal.benchmark_operator,
            positive_proposal.benchmark_selection,
        )
    )


    mean_contract = (
        positive_contract
        is not None
        and
        positive_contract.aggregation
        is not None
        and
        positive_contract.aggregation.function
        ==
        "mean"
        and
        positive_contract.aggregation.source_role
        ==
        "value"
        and
        positive_contract.aggregation.group_by_roles
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


    target_measure_preserved = (
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
        TARGET_MEASURE
    )


    ranking_contract_absent = (
        positive_contract
        is not None
        and
        positive_contract.ranking
        is None
    )


    benchmark_contract_absent = (
        positive_contract
        is not None
        and
        positive_contract.benchmark
        is None
    )


    share_contract_absent = (
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
        "Objective Coverage rule                 "
        f"{OBJECTIVE_COVERAGE_RULE_VERSION}"
    )

    print(
        "Raw decision                            blocked"
    )

    print(
        "Raw dataset stem                        "
        f"{SESSION_FILENAME_STEM}"
    )

    print(
        "Raw y column                            "
        f"{TARGET_MEASURE}"
    )

    print(
        "Raw aggregation                         mean"
    )

    print(
        "Raw ranking                             descending"
    )

    print(
        "Raw partial benchmark                   "
        "('overall_aggregate', 'gt', None)"
    )

    print()
    print(
        "Positive validation status              "
        f"{positive.validation_status}"
    )

    print(
        "Canonical dataset                       "
        f"{canonical_dataset}"
    )

    print(
        "Canonical value wire                    "
        f"{canonical_value_wire}"
    )

    print(
        "Ranking wire cleared                    "
        f"{ranking_wire_cleared}"
    )

    print(
        "Partial benchmark cleared               "
        f"{partial_benchmark_cleared}"
    )

    print(
        "Mean scalar contract                    "
        f"{mean_contract}"
    )

    print(
        "Target measure preserved                "
        f"{target_measure_preserved}"
    )

    print(
        "Ranking contract absent                 "
        f"{ranking_contract_absent}"
    )

    print(
        "Benchmark contract absent               "
        f"{benchmark_contract_absent}"
    )

    print(
        "Share contract absent                   "
        f"{share_contract_absent}"
    )


    # ========================================================
    # 3. WRONG PHYSICAL CANDIDATE — FAIL CLOSED
    # ========================================================

    wrong_candidate = (
        validate(
            proposal=
                live_style_blocked_proposal(
                    y_column=
                        "item_count",
                ),
        )
    )


    assert (
        wrong_candidate.validation_status
        ==
        "blocked"
    ), (
        "Coverage-retry recovery must not replace an unrelated "
        "quantitative candidate with basket_amount."
    )


    print(
        "[PASS] wrong physical candidate remains blocked"
    )


    # ========================================================
    # 4. UNRELATED BLOCKER — FAIL CLOSED
    # ========================================================

    unrelated_blocker = (
        validate(
            proposal=
                live_style_blocked_proposal(
                    blocker_mode=
                        "unrelated",
                ),
        )
    )


    assert (
        unrelated_blocker.validation_status
        ==
        "blocked"
    ), (
        "An unrelated model blocker must remain blocked."
    )


    print(
        "[PASS] unrelated blocker remains blocked"
    )


    # ========================================================
    # 5. UNRELATED DATASET STEM — FAIL CLOSED
    # ========================================================

    wrong_dataset = (
        validate(
            proposal=
                live_style_blocked_proposal(
                    dataset_id=
                        "totally_unrelated_dataset",
                ),
        )
    )


    assert (
        wrong_dataset.validation_status
        ==
        "blocked"
    ), (
        "An unrelated dataset reference must remain blocked."
    )


    print(
        "[PASS] unrelated dataset stem remains blocked"
    )


    # ========================================================
    # 6. VAGUE OBJECTIVE — FAIL CLOSED
    # ========================================================

    vague = (
        validate(
            objective=
                "Analyse les sessions d'achat.",

            proposal=
                live_style_blocked_proposal(),
        )
    )


    assert (
        vague.validation_status
        ==
        "blocked"
    ), (
        "No Objective Coverage average-basket authority means "
        "the abstention must remain blocked."
    )


    print(
        "[PASS] vague session objective remains blocked"
    )


    # ========================================================
    # 7. TOTAL BASKET — FAIL CLOSED
    # ========================================================

    total_basket = (
        validate(
            objective=
                "Quel est le panier total par session ?",

            proposal=
                live_style_blocked_proposal(),
        )
    )


    assert (
        total_basket.validation_status
        ==
        "blocked"
    ), (
        "A total-basket request must not be rewritten to mean."
    )


    print(
        "[PASS] total-basket objective remains blocked"
    )


    # ========================================================
    # 8. AMBIGUOUS DECISION — FAIL CLOSED
    # ========================================================

    ambiguous_proposal = (
        live_style_blocked_proposal()
        .model_copy(
            update={
                "decision":
                    "ambiguous",

                "blockers":
                    [
                        (
                            "Several analytical session views "
                            "may satisfy the request."
                        )
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
        "Ambiguous proposals must never be promoted by this "
        "recovery."
    )


    print(
        "[PASS] ambiguous proposal remains ambiguous"
    )


    # ========================================================
    # 9. EXPLICIT BENCHMARK — FAIL CLOSED
    # ========================================================

    benchmark_objective = (
        validate(
            objective=(
                "Quel panier moyen par session est supérieur "
                "à la moyenne ?"
            ),

            proposal=
                live_style_blocked_proposal(),
        )
    )


    assert (
        benchmark_objective.validation_status
        ==
        "blocked"
    ), (
        "An explicitly requested benchmark must not be treated "
        "as removable benchmark wire noise."
    )


    print(
        "[PASS] explicit benchmark objective remains blocked"
    )


    # ========================================================
    # 10. EXPLICIT RANKING — FAIL CLOSED
    # ========================================================

    ranking_objective = (
        validate(
            objective=(
                "Classe les paniers moyens par session du plus "
                "élevé au plus faible."
            ),

            proposal=
                live_style_blocked_proposal(),
        )
    )


    assert (
        ranking_objective.validation_status
        ==
        "blocked"
    ), (
        "An explicit ranking decision must not be silently "
        "converted to a scalar mean."
    )


    print(
        "[PASS] explicit ranking objective remains blocked"
    )


    # ========================================================
    # 11. RED CLASSIFICATION
    # ========================================================

    gaps: list[
        str
    ] = []


    if not positive_validated:

        gaps.append(
            "session_coverage_retry_false_abstention_recovery_gap"
        )


    if not canonical_dataset:

        gaps.append(
            "session_filename_stem_dataset_canonicalization_gap"
        )


    if not canonical_value_wire:

        gaps.append(
            "session_coverage_candidate_role_canonicalization_gap"
        )


    if not ranking_wire_cleared:

        gaps.append(
            "session_retry_ranking_noise_cleanup_gap"
        )


    if not partial_benchmark_cleared:

        gaps.append(
            "session_retry_partial_benchmark_cleanup_gap"
        )


    if not mean_contract:

        gaps.append(
            "session_mean_contract_materialization_gap"
        )


    if not target_measure_preserved:

        gaps.append(
            "session_retry_target_measure_preservation_gap"
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
        "PASS - session average coverage-retry "
        "wire recovery v0.1"
    )


if __name__ == "__main__":

    main()