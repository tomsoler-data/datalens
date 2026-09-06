from __future__ import annotations


import inspect


import app.planning.ai_analytical_planner as planner_module


from app.planning.ai_analytical_planner import (
    AI_ANALYTICAL_PLANNER_RULE_VERSION,
    AIPlannerProposal,
)


from app.planning.analytical_contract import (
    AggregationSpec,
    AnalyticalContract,
    RankingSpec,
    ShareOfTotalSpec,
    VariableBinding,
)


from app.planning.objective_coverage import (
    explicit_share_of_total_request,
)


# ============================================================
# CANONICAL AUTHORITY
# ============================================================


TARGET_OBJECTIVE = (
    "Quelle catégorie contribue le plus au chiffre d'affaires "
    "et quelle est sa part approximative du chiffre d'affaires total ?"
)


FRENCH_PERCENTAGE_OBJECTIVE = (
    "Quelle catégorie génère le plus de chiffre d'affaires "
    "et quel pourcentage du chiffre d'affaires total représente-t-elle ?"
)


ENGLISH_SHARE_OBJECTIVE = (
    "Which category contributes the most revenue and what share "
    "of total revenue does it represent?"
)


RANKING_ONLY_OBJECTIVE = (
    "Quelle catégorie contribue le plus au chiffre d'affaires ?"
)


TOTAL_ONLY_OBJECTIVE = (
    "Quel est le chiffre d'affaires total ?"
)


UNDERSPECIFIED_PART_OBJECTIVE = (
    "Quelle est la part de chaque catégorie ?"
)


SHARE_REFERENCE = (
    "sum_of_group_values"
)


# ============================================================
# GENERIC CONTRACT BUILDERS
# ============================================================


def grouped_ranking_contract() -> AnalyticalContract:

    return (
        AnalyticalContract(
            contract_id=
                "contract:share-planner:ranking:01",

            origin=
                "ai_planner",

            status=
                "validated",

            title=
                "Category revenue leader",

            request_text=
                TARGET_OBJECTIVE,

            family=
                "ranking",

            required_dataset_ids=[
                "dataset:category_revenue",
            ],

            required_dataset_filenames=[
                "category_revenue.derived",
            ],

            analytical_grain=
                "category",

            bindings=[
                VariableBinding(
                    role=
                        "value",

                    column=
                        "sum_value",

                    dataset_id=
                        "dataset:category_revenue",

                    dataset_filename=
                        "category_revenue.derived",

                    semantic_concept=
                        "revenue",

                    analysis_kind=
                        "quantitative",
                ),

                VariableBinding(
                    role=
                        "dimension",

                    column=
                        "category",

                    dataset_id=
                        "dataset:category_revenue",

                    dataset_filename=
                        "category_revenue.derived",

                    semantic_concept=
                        None,

                    analysis_kind=
                        "categorical",
                ),
            ],

            aggregation=
                AggregationSpec(
                    function=
                        "sum",

                    source_role=
                        "value",

                    group_by_roles=[
                        "dimension",
                    ],

                    output_name=
                        "planned_metric",
                ),

            ranking=
                RankingSpec(
                    order=
                        "descending",

                    limit=
                        1,
                ),

            benchmark=
                None,

            share_of_total=
                None,

            window=
                None,

            filters=[],
            joins=[],
            derived_variables=[],

            required_operations=[
                "Execute deterministic ranking.",
            ],

            reasons=[
                "Generic deterministic planner-share test.",
            ],

            blockers=[],

            planner_confidence=
                1.0,
        )
    )


def grouped_aggregation_contract() -> AnalyticalContract:

    base = (
        grouped_ranking_contract()
    )


    payload = (
        base.model_dump()
    )


    payload[
        "contract_id"
    ] = (
        "contract:share-planner:aggregation:01"
    )


    payload[
        "family"
    ] = (
        "aggregation"
    )


    payload[
        "ranking"
    ] = None


    return (
        AnalyticalContract.model_validate(
            payload
        )
    )


def non_sum_ranking_contract() -> AnalyticalContract:

    payload = (
        grouped_ranking_contract()
        .model_dump()
    )


    payload[
        "contract_id"
    ] = (
        "contract:share-planner:mean:01"
    )


    payload[
        "aggregation"
    ][
        "function"
    ] = (
        "mean"
    )


    return (
        AnalyticalContract.model_validate(
            payload
        )
    )


def ungrouped_aggregation_contract() -> AnalyticalContract:

    payload = (
        grouped_aggregation_contract()
        .model_dump()
    )


    payload[
        "contract_id"
    ] = (
        "contract:share-planner:ungrouped:01"
    )


    payload[
        "analytical_grain"
    ] = (
        "overall"
    )


    payload[
        "bindings"
    ] = [
        binding

        for binding
        in payload[
            "bindings"
        ]

        if (
            binding[
                "role"
            ]
            ==
            "value"
        )
    ]


    payload[
        "aggregation"
    ][
        "group_by_roles"
    ] = []


    return (
        AnalyticalContract.model_validate(
            payload
        )
    )


def blocked_ranking_contract() -> AnalyticalContract:

    payload = (
        grouped_ranking_contract()
        .model_dump()
    )


    payload[
        "contract_id"
    ] = (
        "contract:share-planner:blocked:01"
    )


    payload[
        "status"
    ] = (
        "blocked"
    )


    payload[
        "blockers"
    ] = [
        "diagnostic blocker",
    ]


    return (
        AnalyticalContract.model_validate(
            payload
        )
    )


# ============================================================
# EXPECTED FUTURE PLANNER HELPER
# ============================================================


HELPER_NAME = (
    "canonicalize_explicit_share_of_total_contract"
)


def call_future_helper(
    *,
    objective: str,
    contract: AnalyticalContract,
):

    helper = getattr(
        planner_module,
        HELPER_NAME,
        None,
    )


    if (
        helper
        is None
    ):

        return None


    return helper(
        objective=
            objective,

        contract=
            contract,
    )


# ============================================================
# MAIN
# ============================================================


def main() -> None:

    print(
        "=== DATALENS AI PLANNER SHARE-OF-TOTAL WIRING v0.1 ==="
    )

    print()


    # ========================================================
    # 1. P2-R4 OBJECTIVE AUTHORITY PRESERVED
    # ========================================================

    for objective in [
        TARGET_OBJECTIVE,
        FRENCH_PERCENTAGE_OBJECTIVE,
        ENGLISH_SHARE_OBJECTIVE,
    ]:

        assert (
            explicit_share_of_total_request(
                objective
            )
            is True
        )


    print(
        "[PASS] explicit share-of-total Objective Coverage authority preserved"
    )


    for objective in [
        RANKING_ONLY_OBJECTIVE,
        TOTAL_ONLY_OBJECTIVE,
        UNDERSPECIFIED_PART_OBJECTIVE,
    ]:

        assert (
            explicit_share_of_total_request(
                objective
            )
            is False
        )


    print(
        "[PASS] non-share objectives remain fail-closed"
    )


    # ========================================================
    # 2. P2-R6 CANONICAL SPEC PRESERVED
    # ========================================================

    canonical_spec = (
        ShareOfTotalSpec(
            reference=
                SHARE_REFERENCE
        )
    )


    assert (
        canonical_spec.reference
        ==
        SHARE_REFERENCE
    )


    print(
        "[PASS] canonical ShareOfTotalSpec preserved"
    )


    # ========================================================
    # 3. LLM WIRE MUST REMAIN COMPACT
    #
    # Gemma must NOT be asked to choose this deterministic
    # denominator semantic.
    # ========================================================

    assert (
        "share_of_total"
        not in
        AIPlannerProposal.model_fields
    ), (
        "share_of_total must remain outside the compact LLM wire."
    )


    assert (
        "share_reference"
        not in
        AIPlannerProposal.model_fields
    )


    print(
        "[PASS] LLM wire remains share-blind"
    )


    # ========================================================
    # 4. CURRENT PLANNER WIRING SURFACE
    # ========================================================

    planner_source = (
        inspect.getsource(
            planner_module
        )
    )


    helper = getattr(
        planner_module,
        HELPER_NAME,
        None,
    )


    spec_import_present = (
        "ShareOfTotalSpec"
        in
        planner_source
    )


    objective_authority_present = (
        "explicit_share_of_total_request"
        in
        planner_source
    )


    helper_present = (
        helper
        is not None
    )


    # One occurrence is the definition itself.
    # A second occurrence proves that the helper is actually
    # invoked somewhere in the planner's contract path.
    integration_occurrences = (
        planner_source.count(
            HELPER_NAME
            +
            "("
        )
    )


    integration_present = (
        integration_occurrences
        >=
        2
    )


    print()
    print(
        "Planner rule                            "
        f"{AI_ANALYTICAL_PLANNER_RULE_VERSION}"
    )

    print(
        "ShareOfTotalSpec planner import          "
        f"{spec_import_present}"
    )

    print(
        "Objective share authority in planner     "
        f"{objective_authority_present}"
    )

    print(
        "Deterministic share helper               "
        f"{helper_present}"
    )

    print(
        "Helper definition/call occurrences       "
        f"{integration_occurrences}"
    )

    print(
        "Planner integration                      "
        f"{integration_present}"
    )


    # ========================================================
    # 5. POSITIVE BEHAVIOR
    # ========================================================

    positive_ranking = (
        call_future_helper(
            objective=
                TARGET_OBJECTIVE,

            contract=
                grouped_ranking_contract(),
        )
    )


    positive_aggregation = (
        call_future_helper(
            objective=
                TARGET_OBJECTIVE,

            contract=
                grouped_aggregation_contract(),
        )
    )


    ranking_wired = (
        positive_ranking
        is not None

        and

        positive_ranking.share_of_total
        is not None

        and

        positive_ranking
        .share_of_total
        .reference
        ==
        SHARE_REFERENCE
    )


    aggregation_wired = (
        positive_aggregation
        is not None

        and

        positive_aggregation.share_of_total
        is not None

        and

        positive_aggregation
        .share_of_total
        .reference
        ==
        SHARE_REFERENCE
    )


    print()
    print(
        "Positive ranking share wiring            "
        f"{ranking_wired}"
    )

    print(
        "Positive aggregation share wiring        "
        f"{aggregation_wired}"
    )


    # ========================================================
    # 6. FUTURE NEGATIVE GUARDS
    #
    # Run once the helper exists.
    # ========================================================

    negative_guards_pass = False


    if (
        helper_present
    ):

        neutral = (
            call_future_helper(
                objective=
                    RANKING_ONLY_OBJECTIVE,

                contract=
                    grouped_ranking_contract(),
            )
        )


        total_only = (
            call_future_helper(
                objective=
                    TOTAL_ONLY_OBJECTIVE,

                contract=
                    grouped_ranking_contract(),
            )
        )


        non_sum = (
            call_future_helper(
                objective=
                    TARGET_OBJECTIVE,

                contract=
                    non_sum_ranking_contract(),
            )
        )


        ungrouped = (
            call_future_helper(
                objective=
                    TARGET_OBJECTIVE,

                contract=
                    ungrouped_aggregation_contract(),
            )
        )


        blocked = (
            call_future_helper(
                objective=
                    TARGET_OBJECTIVE,

                contract=
                    blocked_ranking_contract(),
            )
        )


        assert (
            neutral.share_of_total
            is None
        ), (
            "Ranking alone must not invent share_of_total."
        )


        assert (
            total_only.share_of_total
            is None
        ), (
            "A total-only objective must not invent share_of_total."
        )


        assert (
            non_sum.share_of_total
            is None
        ), (
            "Non-SUM contracts must remain share-free."
        )


        assert (
            ungrouped.share_of_total
            is None
        ), (
            "Ungrouped contracts must remain share-free."
        )


        assert (
            blocked.share_of_total
            is None
        ), (
            "Blocked contracts must remain share-free."
        )


        negative_guards_pass = True


        print()
        print(
            "[PASS] ranking-only objective remains share-free"
        )

        print(
            "[PASS] total-only objective remains share-free"
        )

        print(
            "[PASS] non-SUM contract remains share-free"
        )

        print(
            "[PASS] ungrouped contract remains share-free"
        )

        print(
            "[PASS] blocked contract remains share-free"
        )


    else:

        print()
        print(
            "Future negative guards                  DEFERRED UNTIL HELPER EXISTS"
        )


    # ========================================================
    # 7. RED CLASSIFICATION
    # ========================================================

    gaps: list[
        str
    ] = []


    if not (
        spec_import_present
    ):

        gaps.append(
            "planner_share_spec_import_gap"
        )


    if not (
        objective_authority_present
    ):

        gaps.append(
            "planner_share_objective_authority_gap"
        )


    if not (
        helper_present
    ):

        gaps.append(
            "planner_share_helper_gap"
        )


    if not (
        integration_present
    ):

        gaps.append(
            "planner_share_integration_gap"
        )


    if not (
        ranking_wired
    ):

        gaps.append(
            "planner_ranking_share_contract_wiring_gap"
        )


    if not (
        aggregation_wired
    ):

        gaps.append(
            "planner_aggregation_share_contract_wiring_gap"
        )


    print()
    print(
        "Negative guards                         "
        f"{(
            'PASS'
            if negative_guards_pass
            else 'DEFERRED'
        )}"
    )

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
        "PASS - AI Planner share-of-total wiring v0.1"
    )


if __name__ == "__main__":

    main()