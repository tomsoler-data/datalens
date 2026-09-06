from __future__ import annotations


from app.planning.ai_analytical_planner import (
    AI_ANALYTICAL_PLANNER_RULE_VERSION,
    AIPlannerProposal,
    PlannerCatalog,
    PlannerColumnProfile,
    PlannerDatasetProfile,
    normalize_identifier_for_match,
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


SOURCE_DATASET_ID = (
    "dataset:orders"
)


RAW_SOURCE_REFERENCE = (
    "dataset_orders"
)


SESSION_DATASET_ID = (
    "derived:orders:session:session_id:amount"
)


SESSION_FILENAME = (
    "orders__sessions_amount.derived"
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
# PROFILE HELPERS
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


def source_profile(
    *,
    dataset_id: str = SOURCE_DATASET_ID,
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
        ),

        column(
            name=
                "amount",

            kind=
                "quantitative",

            dtype=
                "float64",

            unique_count=
                95,
        ),
    ]


    return (
        PlannerDatasetProfile(
            dataset_id=
                dataset_id,

            filename=
                "orders.csv",

            row_count=
                200,

            column_count=
                len(
                    columns
                ),

            columns=
                columns,

            is_derived=
                False,

            derivation_type=
                None,

            analytical_grain=
                None,

            operation=
                None,

            aggregation=
                None,

            group_column=
                None,

            entity_column=
                None,

            source_time_column=
                None,

            target_time_column=
                None,

            source_measure_column=
                None,

            target_measure_column=
                None,

            source_measure_formula=
                None,

            metric_semantics=
                None,

            measure_semantic_aliases=[],

            fact_dataset_id=
                None,

            source_dataset_ids=[],
        )
    )


def session_profile(
    *,
    dataset_id: str = SESSION_DATASET_ID,
    fact_dataset_id: str | None = SOURCE_DATASET_ID,
    source_dataset_ids: list[str] | None = None,
    target_measure: str = TARGET_MEASURE,
) -> PlannerDatasetProfile:

    if source_dataset_ids is None:

        source_dataset_ids = [
            SOURCE_DATASET_ID
        ]


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
                7,
        ),
    ]


    return (
        PlannerDatasetProfile(
            dataset_id=
                dataset_id,

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
                    "One server-owned row represents "
                    "one validated purchase session."
                ),

            measure_semantic_aliases=[
                "amount",
                target_measure,
            ],

            fact_dataset_id=
                fact_dataset_id,

            source_dataset_ids=
                source_dataset_ids,
        )
    )


def catalog(
    *,
    duplicate_source_match: bool = False,
    duplicate_session_lineage: bool = False,
    omit_fact_lineage: bool = False,
    inconsistent_source_lineage: bool = False,
) -> PlannerCatalog:

    datasets = [
        source_profile()
    ]


    if duplicate_source_match:

        datasets.append(
            source_profile(
                dataset_id=
                    "dataset_orders",
            )
        )


    session_sources = (
        [
            "dataset:other"
        ]
        if inconsistent_source_lineage
        else [
            SOURCE_DATASET_ID
        ]
    )


    datasets.append(
        session_profile(
            fact_dataset_id=(
                None
                if omit_fact_lineage
                else SOURCE_DATASET_ID
            ),

            source_dataset_ids=
                session_sources,
        )
    )


    if duplicate_session_lineage:

        datasets.append(
            session_profile(
                dataset_id=(
                    "derived:orders:session:"
                    "session_id:amount:copy"
                ),

                fact_dataset_id=
                    SOURCE_DATASET_ID,

                source_dataset_ids=[
                    SOURCE_DATASET_ID
                ],
            )
        )


    return (
        PlannerCatalog(
            datasets=
                datasets
        )
    )


# ============================================================
# LIVE-STYLE SOURCE-REFERENCE ABSTENTION
# ============================================================


def blocked_source_proposal(
    *,
    dataset_id: str = RAW_SOURCE_REFERENCE,
    candidate: str = TARGET_MEASURE,
    grain: str = SESSION_GRAIN,
    blocker_mode: str = "source_measure",
) -> AIPlannerProposal:

    if (
        blocker_mode
        ==
        "source_measure"
    ):

        blockers = [
            (
                "The selected dataset does not contain "
                f"the requested measure {candidate}."
            )
        ]


        reasons = [
            (
                "The requested session-level metric is not "
                "available in the selected dataset."
            )
        ]


    elif (
        blocker_mode
        ==
        "unrelated"
    ):

        blockers = [
            (
                "Requested currency conversion metadata "
                "is unavailable."
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
                "Requested metric unavailable",

            family=
                "aggregation",

            # Model points to the SOURCE dataset rather than
            # the server-owned derived session view.
            dataset_id=
                dataset_id,

            analytical_grain=
                grain,

            x_column=
                None,

            # Exact physical Objective Coverage candidate,
            # but in the noisy y role.
            y_column=
                candidate,

            group_column=
                None,

            value_column=
                None,

            time_column=
                None,

            dimension_column=
                None,

            entity_column=
                grain,

            aggregation_function=
                "mean",

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
                blockers,

            reasons=
                reasons,

            confidence=
                0.8,
        )
    )


def validate(
    *,
    proposal: AIPlannerProposal,
    planner_catalog: PlannerCatalog | None = None,
    objective: str = OBJECTIVE,
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
        "=== DATALENS SESSION AVERAGE SOURCE-LINEAGE "
        "RECOVERY v0.1 ==="
    )

    print()


    planner_catalog = (
        catalog()
    )


    # ========================================================
    # 1. OBJECTIVE COVERAGE AUTHORITY
    # ========================================================

    requirements = (
        extract_objective_requirements(
            objective=
                OBJECTIVE,

            catalog=
                planner_catalog,
        )
    )


    target_requirements = [
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
            target_requirements
        )
        ==
        1
    )


    requirement = (
        target_requirements[
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
        "candidate/value/mean"
    )


    # ========================================================
    # 2. SOURCE REFERENCE RESOLUTION IS DETERMINISTIC
    # ========================================================

    raw_normalized = (
        normalize_identifier_for_match(
            RAW_SOURCE_REFERENCE
        )
    )


    source_matches = [
        profile

        for profile
        in planner_catalog.datasets

        if (
            not profile.is_derived
            and
            normalize_identifier_for_match(
                profile.dataset_id
            )
            ==
            raw_normalized
        )
    ]


    assert (
        len(
            source_matches
        )
        ==
        1
    )


    resolved_source = (
        source_matches[
            0
        ]
    )


    assert (
        resolved_source.dataset_id
        ==
        SOURCE_DATASET_ID
    )


    print(
        "[PASS] raw source reference resolves uniquely"
    )


    # ========================================================
    # 3. DERIVED LINEAGE AUTHORITY IS EXPLICIT
    # ========================================================

    lineage_matches = [
        profile

        for profile
        in planner_catalog.datasets

        if (
            profile.is_derived
            and
            profile.operation
            ==
            "session_materialization"
            and
            profile.fact_dataset_id
            ==
            resolved_source.dataset_id
            and
            resolved_source.dataset_id
            in
            profile.source_dataset_ids
            and
            profile.target_measure_column
            ==
            TARGET_MEASURE
            and
            profile.analytical_grain
            ==
            SESSION_GRAIN
            and
            profile.entity_column
            ==
            SESSION_GRAIN
        )
    ]


    assert (
        len(
            lineage_matches
        )
        ==
        1
    )


    expected_session = (
        lineage_matches[
            0
        ]
    )


    print(
        "[PASS] explicit source -> unique session lineage"
    )


    # ========================================================
    # 4. POSITIVE — CURRENTLY RED
    # ========================================================

    positive = (
        validate(
            proposal=
                blocked_source_proposal(),
        )
    )


    positive_proposal = (
        positive.proposal
    )


    positive_contract = (
        positive.contract
    )


    validated = (
        positive.validation_status
        ==
        "validated"
    )


    canonical_session_dataset = (
        positive_proposal.dataset_id
        ==
        expected_session.dataset_id
    )


    canonical_value_role = (
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


    mean_scalar_contract = (
        positive_contract
        is not None
        and
        positive_contract.family
        ==
        "aggregation"
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


    decision_noise_absent = (
        positive_contract
        is not None
        and
        positive_contract.ranking
        is None
        and
        positive_contract.benchmark
        is None
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
        "Raw source reference                    "
        f"{RAW_SOURCE_REFERENCE}"
    )

    print(
        "Raw source normalized                   "
        f"{raw_normalized}"
    )

    print(
        "Resolved source dataset                 "
        f"{resolved_source.dataset_id}"
    )

    print(
        "Expected session dataset                "
        f"{expected_session.dataset_id}"
    )

    print(
        "Lineage fact_dataset_id                 "
        f"{expected_session.fact_dataset_id}"
    )

    print(
        "Lineage source_dataset_ids              "
        f"{expected_session.source_dataset_ids}"
    )

    print()
    print(
        "Positive validation status              "
        f"{positive.validation_status}"
    )

    print(
        "Canonical session dataset               "
        f"{canonical_session_dataset}"
    )

    print(
        "Canonical value role                    "
        f"{canonical_value_role}"
    )

    print(
        "Mean scalar contract                    "
        f"{mean_scalar_contract}"
    )

    print(
        "Target measure preserved                "
        f"{target_measure_preserved}"
    )

    print(
        "Decision noise absent                   "
        f"{decision_noise_absent}"
    )


    # ========================================================
    # 5. WRONG CANDIDATE — FAIL CLOSED
    # ========================================================

    wrong_candidate = (
        validate(
            proposal=
                blocked_source_proposal(
                    candidate=
                        "item_count",
                )
        )
    )


    assert (
        wrong_candidate.validation_status
        ==
        "blocked"
    )


    print(
        "[PASS] wrong candidate remains blocked"
    )


    # ========================================================
    # 6. WRONG GRAIN — FAIL CLOSED
    # ========================================================

    wrong_grain = (
        validate(
            proposal=
                blocked_source_proposal(
                    grain=
                        "customer_id",
                )
        )
    )


    assert (
        wrong_grain.validation_status
        ==
        "blocked"
    )


    print(
        "[PASS] wrong grain remains blocked"
    )


    # ========================================================
    # 7. UNRELATED BLOCKER — FAIL CLOSED
    # ========================================================

    unrelated_blocker = (
        validate(
            proposal=
                blocked_source_proposal(
                    blocker_mode=
                        "unrelated",
                )
        )
    )


    assert (
        unrelated_blocker.validation_status
        ==
        "blocked"
    )


    print(
        "[PASS] unrelated blocker remains blocked"
    )


    # ========================================================
    # 8. UNKNOWN SOURCE REFERENCE — FAIL CLOSED
    # ========================================================

    unknown_source = (
        validate(
            proposal=
                blocked_source_proposal(
                    dataset_id=
                        "dataset_unknown",
                )
        )
    )


    assert (
        unknown_source.validation_status
        ==
        "blocked"
    )


    print(
        "[PASS] unknown source reference remains blocked"
    )


    # ========================================================
    # 9. AMBIGUOUS SOURCE RESOLUTION — FAIL CLOSED
    # ========================================================

    ambiguous_source = (
        validate(
            proposal=
                blocked_source_proposal(),

            planner_catalog=
                catalog(
                    duplicate_source_match=
                        True,
                ),
        )
    )


    assert (
        ambiguous_source.validation_status
        ==
        "blocked"
    )


    print(
        "[PASS] ambiguous source resolution remains blocked"
    )


    # ========================================================
    # 10. MULTIPLE SESSION LINEAGE TARGETS — FAIL CLOSED
    # ========================================================

    ambiguous_session = (
        validate(
            proposal=
                blocked_source_proposal(),

            planner_catalog=
                catalog(
                    duplicate_session_lineage=
                        True,
                ),
        )
    )


    assert (
        ambiguous_session.validation_status
        ==
        "blocked"
    )


    print(
        "[PASS] multiple session lineage targets remain blocked"
    )


    # ========================================================
    # 11. MISSING fact_dataset_id — FAIL CLOSED
    # ========================================================

    missing_fact_lineage = (
        validate(
            proposal=
                blocked_source_proposal(),

            planner_catalog=
                catalog(
                    omit_fact_lineage=
                        True,
                ),
        )
    )


    assert (
        missing_fact_lineage.validation_status
        ==
        "blocked"
    )


    print(
        "[PASS] missing fact_dataset_id remains blocked"
    )


    # ========================================================
    # 12. INCONSISTENT source_dataset_ids — FAIL CLOSED
    # ========================================================

    inconsistent_lineage = (
        validate(
            proposal=
                blocked_source_proposal(),

            planner_catalog=
                catalog(
                    inconsistent_source_lineage=
                        True,
                ),
        )
    )


    assert (
        inconsistent_lineage.validation_status
        ==
        "blocked"
    )


    print(
        "[PASS] inconsistent source lineage remains blocked"
    )


    # ========================================================
    # 13. VAGUE OBJECTIVE — FAIL CLOSED
    # ========================================================

    vague = (
        validate(
            objective=
                "Analyse les sessions d'achat.",

            proposal=
                blocked_source_proposal(),
        )
    )


    assert (
        vague.validation_status
        ==
        "blocked"
    )


    print(
        "[PASS] vague objective remains blocked"
    )


    # ========================================================
    # 14. TOTAL BASKET — FAIL CLOSED
    # ========================================================

    total_basket = (
        validate(
            objective=
                "Quel est le panier total par session ?",

            proposal=
                blocked_source_proposal(),
        )
    )


    assert (
        total_basket.validation_status
        ==
        "blocked"
    )


    print(
        "[PASS] total-basket objective remains blocked"
    )


    # ========================================================
    # 15. AMBIGUOUS MODEL DECISION — FAIL CLOSED
    # ========================================================

    ambiguous_proposal = (
        blocked_source_proposal()
        .model_copy(
            update={
                "decision":
                    "ambiguous",

                "blockers":
                    [
                        (
                            "Several analytical sources may "
                            "satisfy the request."
                        )
                    ],
            }
        )
    )


    ambiguous_decision = (
        validate(
            proposal=
                ambiguous_proposal,
        )
    )


    assert (
        ambiguous_decision.validation_status
        ==
        "ambiguous"
    )


    print(
        "[PASS] ambiguous model decision remains ambiguous"
    )


    # ========================================================
    # 16. EXPLICIT BENCHMARK — FAIL CLOSED
    # ========================================================

    explicit_benchmark = (
        validate(
            objective=(
                "Quel panier moyen par session est supérieur "
                "à la moyenne ?"
            ),

            proposal=
                blocked_source_proposal(),
        )
    )


    assert (
        explicit_benchmark.validation_status
        ==
        "blocked"
    )


    print(
        "[PASS] explicit benchmark remains blocked"
    )


    # ========================================================
    # 17. EXPLICIT RANKING — FAIL CLOSED
    # ========================================================

    explicit_ranking = (
        validate(
            objective=(
                "Classe les paniers moyens par session du "
                "plus élevé au plus faible."
            ),

            proposal=
                blocked_source_proposal(),
        )
    )


    assert (
        explicit_ranking.validation_status
        ==
        "blocked"
    )


    print(
        "[PASS] explicit ranking remains blocked"
    )


    # ========================================================
    # 18. RED GAPS
    # ========================================================

    gaps: list[
        str
    ] = []


    if not validated:

        gaps.append(
            "source_lineage_false_abstention_recovery_gap"
        )


    if not canonical_session_dataset:

        gaps.append(
            "source_reference_to_session_dataset_gap"
        )


    if not canonical_value_role:

        gaps.append(
            "source_lineage_value_role_canonicalization_gap"
        )


    if not mean_scalar_contract:

        gaps.append(
            "source_lineage_mean_contract_gap"
        )


    if not target_measure_preserved:

        gaps.append(
            "source_lineage_target_measure_preservation_gap"
        )


    if not decision_noise_absent:

        gaps.append(
            "source_lineage_decision_noise_gap"
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
        "PASS - session average source-lineage recovery v0.1"
    )


if __name__ == "__main__":

    main()