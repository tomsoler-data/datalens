from __future__ import annotations


from app.planning.ai_analytical_planner import (
    AI_ANALYTICAL_PLANNER_RULE_VERSION,
    explicit_aggregation_from_objective,
)


from app.planning.objective_coverage import (
    OBJECTIVE_COVERAGE_RULE_VERSION,
    extract_objective_requirements,
)


from tests.planning import (
    test_session_average_source_lineage_recovery_v0_1
    as fixture,
)


# ============================================================
# LIVE-OBSERVED GENERIC SHAPE
# ============================================================


RAW_MATERIALIZATION_AGGREGATION = (
    "sum"
)


REQUESTED_ANALYTICAL_AGGREGATION = (
    "mean"
)


EXPECTED_GAP = (
    "session_source_lineage_materialization_sum_to_requested_mean_gap"
)


def live_style_sum_proposal():
    """
    Generic reproduction of the live shape:

        source dataset reference
        + exact session grain
        + exact derived target measure in y
        + SUM copied from session materialization
        + requested analytical aggregation is MEAN
        + exact missing-measure blocker
    """

    return (
        fixture.blocked_source_proposal()
        .model_copy(
            update={
                "aggregation_function":
                    RAW_MATERIALIZATION_AGGREGATION,

                "ranking_order":
                    "none",

                "ranking_limit":
                    None,

                "benchmark_reference":
                    None,

                "benchmark_operator":
                    None,

                "benchmark_selection":
                    None,

                "blockers":
                    [
                        (
                            "The objective contains a reference "
                            f"to {fixture.TARGET_MEASURE}, "
                            "which is not present in the catalog."
                        )
                    ],

                "reasons":
                    [
                        (
                            "The objective contains a reference "
                            f"to {fixture.TARGET_MEASURE}, "
                            "which is not present in the catalog."
                        )
                    ],
            }
        )
    )


def main() -> None:

    print(
        "=== DATALENS SESSION SOURCE-LINEAGE "
        "MATERIALIZATION AGGREGATION CONFUSION v0.1 ==="
    )

    print()


    planner_catalog = (
        fixture.catalog()
    )


    # ========================================================
    # 1. OBJECTIVE AUTHORITY = MEAN
    # ========================================================

    objective_aggregation = (
        explicit_aggregation_from_objective(
            fixture.OBJECTIVE
        )
    )


    assert (
        objective_aggregation
        ==
        REQUESTED_ANALYTICAL_AGGREGATION
    )


    requirements = (
        extract_objective_requirements(
            objective=
                fixture.OBJECTIVE,

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
            fixture.REQUIREMENT_ID
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
            fixture.TARGET_MEASURE
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
        REQUESTED_ANALYTICAL_AGGREGATION
    )


    print(
        "[PASS] objective explicitly requires analytical mean"
    )

    print(
        "[PASS] Objective Coverage requires candidate/value/mean"
    )


    # ========================================================
    # 2. UNIQUE SOURCE / SESSION LINEAGE AUTHORITY
    # ========================================================

    source_profiles = [
        dataset

        for dataset
        in planner_catalog.datasets

        if (
            dataset.dataset_id
            ==
            fixture.SOURCE_DATASET_ID
        )
    ]


    assert (
        len(
            source_profiles
        )
        ==
        1
    )


    session_profiles = [
        dataset

        for dataset
        in planner_catalog.datasets

        if (
            dataset.is_derived
            and
            dataset.operation
            ==
            "session_materialization"
            and
            dataset.fact_dataset_id
            ==
            fixture.SOURCE_DATASET_ID
            and
            fixture.SOURCE_DATASET_ID
            in
            dataset.source_dataset_ids
            and
            dataset.target_measure_column
            ==
            fixture.TARGET_MEASURE
            and
            dataset.analytical_grain
            ==
            fixture.SESSION_GRAIN
        )
    ]


    assert (
        len(
            session_profiles
        )
        ==
        1
    )


    session_profile = (
        session_profiles[
            0
        ]
    )


    assert (
        session_profile.aggregation
        ==
        RAW_MATERIALIZATION_AGGREGATION
    )


    print(
        "[PASS] exact source dataset authority"
    )

    print(
        "[PASS] unique derived session lineage authority"
    )

    print(
        "[PASS] session materialization itself uses sum"
    )


    # ========================================================
    # 3. BASELINE P3-R10 MEAN FORM REMAINS GREEN
    # ========================================================

    baseline = (
        fixture.validate(
            proposal=
                fixture.blocked_source_proposal(),
        )
    )


    assert (
        baseline.validation_status
        ==
        "validated"
    )


    assert (
        baseline.proposal.dataset_id
        ==
        fixture.SESSION_DATASET_ID
    )


    assert (
        baseline.proposal.value_column
        ==
        fixture.TARGET_MEASURE
    )


    assert (
        baseline.proposal.aggregation_function
        ==
        REQUESTED_ANALYTICAL_AGGREGATION
    )


    assert (
        baseline.contract
        is not None
    )


    assert (
        baseline.contract.aggregation
        is not None
    )


    assert (
        baseline.contract.aggregation.function
        ==
        REQUESTED_ANALYTICAL_AGGREGATION
    )


    print(
        "[PASS] existing P3-R10 raw-mean recovery remains GREEN"
    )


    # ========================================================
    # 4. LIVE SUM-CONFUSION FORM — CURRENT RED
    # ========================================================

    raw_proposal = (
        live_style_sum_proposal()
    )


    assert (
        raw_proposal.dataset_id
        ==
        fixture.RAW_SOURCE_REFERENCE
    )


    assert (
        raw_proposal.analytical_grain
        ==
        fixture.SESSION_GRAIN
    )


    assert (
        raw_proposal.y_column
        ==
        fixture.TARGET_MEASURE
    )


    assert (
        raw_proposal.value_column
        is None
    )


    assert (
        raw_proposal.aggregation_function
        ==
        RAW_MATERIALIZATION_AGGREGATION
    )


    result = (
        fixture.validate(
            proposal=
                raw_proposal,
        )
    )


    proposal = (
        result.proposal
    )


    contract = (
        result.contract
    )


    recovered = (
        result.validation_status
        ==
        "validated"
    )


    canonical_session_dataset = (
        proposal.dataset_id
        ==
        fixture.SESSION_DATASET_ID
    )


    canonical_value_role = (
        proposal.value_column
        ==
        fixture.TARGET_MEASURE
        and
        proposal.y_column
        is None
        and
        proposal.entity_column
        is None
    )


    canonical_requested_mean = (
        proposal.aggregation_function
        ==
        REQUESTED_ANALYTICAL_AGGREGATION
    )


    mean_contract = (
        contract
        is not None
        and
        contract.family
        ==
        "aggregation"
        and
        contract.aggregation
        is not None
        and
        contract.aggregation.function
        ==
        REQUESTED_ANALYTICAL_AGGREGATION
        and
        contract.aggregation.source_role
        ==
        "value"
        and
        contract.aggregation.group_by_roles
        ==
        []
    )


    value_bindings = (
        [
            binding

            for binding
            in contract.bindings

            if (
                binding.role
                ==
                "value"
            )
        ]

        if contract
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
        fixture.TARGET_MEASURE
    )


    no_decision_noise = (
        contract
        is not None
        and
        contract.ranking
        is None
        and
        contract.benchmark
        is None
        and
        contract.share_of_total
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
        "Raw dataset reference                   "
        f"{raw_proposal.dataset_id}"
    )

    print(
        "Raw analytical grain                    "
        f"{raw_proposal.analytical_grain}"
    )

    print(
        "Raw candidate                           "
        f"{raw_proposal.y_column}"
    )

    print(
        "Raw aggregation                         "
        f"{raw_proposal.aggregation_function}"
    )

    print(
        "Session materialization aggregation     "
        f"{session_profile.aggregation}"
    )

    print(
        "Objective requested aggregation         "
        f"{objective_aggregation}"
    )

    print()
    print(
        "Positive validation status              "
        f"{result.validation_status}"
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
        "Canonical requested mean                "
        f"{canonical_requested_mean}"
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
        "Decision noise absent                   "
        f"{no_decision_noise}"
    )


    # ========================================================
    # 5. MEDIAN IS NOT MATERIALIZATION CONFUSION
    # ========================================================

    median_wire = (
        live_style_sum_proposal()
        .model_copy(
            update={
                "aggregation_function":
                    "median",
            }
        )
    )


    median_result = (
        fixture.validate(
            proposal=
                median_wire,
        )
    )


    assert (
        median_result.validation_status
        ==
        "blocked"
    )


    print(
        "[PASS] unrelated median aggregation remains blocked"
    )


    # ========================================================
    # 6. SUM WITHOUT SUM-MATERIALIZATION AUTHORITY
    # ========================================================

    modified_datasets = []


    for dataset in (
        planner_catalog.datasets
    ):

        if (
            dataset.dataset_id
            ==
            fixture.SESSION_DATASET_ID
        ):

            modified_datasets.append(
                dataset.model_copy(
                    update={
                        "aggregation":
                            "mean",
                    }
                )
            )

        else:

            modified_datasets.append(
                dataset
            )


    no_sum_authority_catalog = (
        planner_catalog.model_copy(
            update={
                "datasets":
                    modified_datasets,
            }
        )
    )


    no_sum_authority_result = (
        fixture.validate(
            proposal=
                live_style_sum_proposal(),

            planner_catalog=
                no_sum_authority_catalog,
        )
    )


    assert (
        no_sum_authority_result.validation_status
        ==
        "blocked"
    )


    print(
        "[PASS] SUM without SUM materialization authority remains blocked"
    )


    # ========================================================
    # 7. WRONG CANDIDATE
    # ========================================================

    wrong_candidate = (
        live_style_sum_proposal()
        .model_copy(
            update={
                "y_column":
                    "item_count",

                "blockers":
                    [
                        (
                            "The objective contains a reference "
                            "to item_count, which is not present "
                            "in the catalog."
                        )
                    ],
            }
        )
    )


    wrong_candidate_result = (
        fixture.validate(
            proposal=
                wrong_candidate,
        )
    )


    assert (
        wrong_candidate_result.validation_status
        ==
        "blocked"
    )


    print(
        "[PASS] wrong candidate remains blocked"
    )


    # ========================================================
    # 8. WRONG GRAIN
    # ========================================================

    wrong_grain_result = (
        fixture.validate(
            proposal=(
                live_style_sum_proposal()
                .model_copy(
                    update={
                        "analytical_grain":
                            "customer_id",

                        "entity_column":
                            "customer_id",
                    }
                )
            ),
        )
    )


    assert (
        wrong_grain_result.validation_status
        ==
        "blocked"
    )


    print(
        "[PASS] wrong grain remains blocked"
    )


    # ========================================================
    # 9. UNRELATED BLOCKER
    # ========================================================

    unrelated_blocker_result = (
        fixture.validate(
            proposal=(
                live_style_sum_proposal()
                .model_copy(
                    update={
                        "blockers":
                            [
                                (
                                    "Requested currency metadata "
                                    "is unavailable."
                                )
                            ],
                    }
                )
            ),
        )
    )


    assert (
        unrelated_blocker_result.validation_status
        ==
        "blocked"
    )


    print(
        "[PASS] unrelated blocker remains blocked"
    )


    # ========================================================
    # 10. MISSING LINEAGE
    # ========================================================

    missing_lineage_result = (
        fixture.validate(
            proposal=
                live_style_sum_proposal(),

            planner_catalog=
                fixture.catalog(
                    omit_fact_lineage=
                        True,
                ),
        )
    )


    assert (
        missing_lineage_result.validation_status
        ==
        "blocked"
    )


    print(
        "[PASS] missing fact lineage remains blocked"
    )


    # ========================================================
    # 11. AMBIGUOUS LINEAGE
    # ========================================================

    ambiguous_lineage_result = (
        fixture.validate(
            proposal=
                live_style_sum_proposal(),

            planner_catalog=
                fixture.catalog(
                    duplicate_session_lineage=
                        True,
                ),
        )
    )


    assert (
        ambiguous_lineage_result.validation_status
        ==
        "blocked"
    )


    print(
        "[PASS] ambiguous session lineage remains blocked"
    )


    # ========================================================
    # 12. EXPLICIT BENCHMARK
    # ========================================================

    explicit_benchmark_result = (
        fixture.validate(
            objective=(
                "Quel panier moyen par session est supérieur "
                "à la moyenne ?"
            ),

            proposal=
                live_style_sum_proposal(),
        )
    )


    assert (
        explicit_benchmark_result.validation_status
        ==
        "blocked"
    )


    print(
        "[PASS] explicit benchmark remains blocked"
    )


    # ========================================================
    # 13. EXPLICIT RANKING
    # ========================================================

    explicit_ranking_result = (
        fixture.validate(
            objective=(
                "Classe les paniers moyens par session du "
                "plus élevé au plus faible."
            ),

            proposal=
                live_style_sum_proposal(),
        )
    )


    assert (
        explicit_ranking_result.validation_status
        ==
        "blocked"
    )


    print(
        "[PASS] explicit ranking remains blocked"
    )


    # ========================================================
    # 14. AMBIGUOUS MODEL DECISION
    # ========================================================

    ambiguous_decision_result = (
        fixture.validate(
            proposal=(
                live_style_sum_proposal()
                .model_copy(
                    update={
                        "decision":
                            "ambiguous",
                    }
                )
            ),
        )
    )


    assert (
        ambiguous_decision_result.validation_status
        ==
        "ambiguous"
    )


    print(
        "[PASS] ambiguous model decision remains ambiguous"
    )


    # ========================================================
    # 15. RED — ONE ROOT CAUSE ONLY
    # ========================================================

    full_recovery = (
        recovered
        and
        canonical_session_dataset
        and
        canonical_value_role
        and
        canonical_requested_mean
        and
        mean_contract
        and
        target_measure_preserved
        and
        no_decision_noise
    )


    gaps: list[
        str
    ] = []


    if not (
        full_recovery
    ):

        gaps.append(
            EXPECTED_GAP
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
        "PASS - session source-lineage materialization "
        "aggregation confusion recovery v0.1"
    )


if __name__ == "__main__":

    main()