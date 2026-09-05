from __future__ import annotations


from types import (
    SimpleNamespace,
)


from app.planning.analytical_contract import (
    AggregationSpec,
    AnalyticalContract,
    VariableBinding,
)


from app.planning.objective_coverage import (
    OBJECTIVE_COVERAGE_RULE_VERSION,
    ObjectiveCoverageRequirement,
    build_objective_coverage,
    contract_covers_requirement,
    extract_objective_requirements,
)


# ============================================================
# GENERIC SESSION AUTHORITY
# ============================================================


SESSION_DATASET_ID = (
    "derived:orders:session:session_id:amount"
)


SESSION_FILENAME = (
    "orders__sessions_amount.derived"
)


SESSION_GRAIN = (
    "session_id"
)


BASKET_COLUMN = (
    "basket_amount"
)


FRENCH_OBJECTIVE = (
    "Quel est le panier moyen par session d'achat ?"
)


ENGLISH_OBJECTIVE = (
    "What is the average basket per purchase session?"
)


AVERAGE_BASKET_REQUIREMENT_ID = (
    "metric:average_basket"
)


# ============================================================
# MINIMAL CATALOG
#
# Objective Coverage intentionally accepts a generic catalog
# interface. We therefore use simple immutable-like namespaces
# instead of coupling this RED to the AI planner wire models.
# ============================================================


def catalog(
    *,
    include_basket: bool = True,
):

    columns = [
        SimpleNamespace(
            name=
                SESSION_GRAIN
        ),
    ]


    if include_basket:

        columns.append(
            SimpleNamespace(
                name=
                    BASKET_COLUMN
            )
        )


    columns.append(
        SimpleNamespace(
            name=
                "item_count"
        )
    )


    dataset = (
        SimpleNamespace(
            dataset_id=
                SESSION_DATASET_ID,

            filename=
                SESSION_FILENAME,

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

            entity_column=
                SESSION_GRAIN,

            source_measure_column=
                "amount",

            target_measure_column=
                (
                    BASKET_COLUMN
                    if include_basket
                    else None
                ),

            measure_semantic_aliases=[
                "amount",
                BASKET_COLUMN,
            ],

            columns=
                columns,
        )
    )


    return (
        SimpleNamespace(
            datasets=[
                dataset
            ]
        )
    )


# ============================================================
# CONTRACT BUILDERS
# ============================================================


def mean_basket_contract(
    *,
    status: str = "validated",
    function: str = "mean",
    column: str = BASKET_COLUMN,
) -> AnalyticalContract:

    if (
        status
        in {
            "blocked",
            "ambiguous",
        }
    ):

        return (
            AnalyticalContract(
                contract_id=
                    f"contract:{status}",

                origin=
                    "ai_planner",

                status=
                    status,  # type: ignore[arg-type]

                title=
                    "Session basket unavailable",

                request_text=
                    FRENCH_OBJECTIVE,

                family=
                    "aggregation",

                required_dataset_ids=[
                    SESSION_DATASET_ID
                ],

                analytical_grain=
                    SESSION_GRAIN,

                bindings=[],

                aggregation=
                    None,

                blockers=[
                    "Execution intentionally blocked."
                ],
            )
        )


    return (
        AnalyticalContract(
            contract_id=
                (
                    "contract:"
                    f"{function}:"
                    f"{column}"
                ),

            origin=
                "ai_planner",

            status=
                "validated",

            title=
                "Average basket per session",

            request_text=
                FRENCH_OBJECTIVE,

            family=
                "aggregation",

            required_dataset_ids=[
                SESSION_DATASET_ID
            ],

            required_dataset_filenames=[
                SESSION_FILENAME
            ],

            analytical_grain=
                SESSION_GRAIN,

            bindings=[
                VariableBinding(
                    role=
                        "value",

                    column=
                        column,

                    dataset_id=
                        SESSION_DATASET_ID,

                    dataset_filename=
                        SESSION_FILENAME,

                    semantic_concept=
                        (
                            "average_basket"
                            if (
                                column
                                ==
                                BASKET_COLUMN
                            )
                            else None
                        ),

                    analysis_kind=
                        "quantitative",
                )
            ],

            aggregation=
                AggregationSpec(
                    function=
                        function,  # type: ignore[arg-type]

                    source_role=
                        "value",

                    group_by_roles=[],

                    output_name=
                        (
                            "mean_basket_amount"
                            if function
                            ==
                            "mean"
                            else None
                        ),
                ),

            blockers=[],
        )
    )


# ============================================================
# EXPECTED CANONICAL REQUIREMENT
#
# This object proves that the existing generic
# contract_covers_requirement() machinery already knows how to
# enforce the intended contract once extraction creates the
# semantic requirement.
# ============================================================


def canonical_average_basket_requirement(
) -> ObjectiveCoverageRequirement:

    return (
        ObjectiveCoverageRequirement(
            requirement_id=
                AVERAGE_BASKET_REQUIREMENT_ID,

            concept=
                "average_basket",

            requirement_type=
                "metric",

            requested_phrases=[
                "panier moyen"
            ],

            candidate_columns=[
                BASKET_COLUMN
            ],

            allowed_roles=[
                "value"
            ],

            required_aggregation=
                "mean",

            covered=
                False,

            covered_by_contract_ids=[],

            notes=[],
        )
    )


# ============================================================
# HELPERS
# ============================================================


def requirement_by_id(
    requirements,
    requirement_id: str,
):

    matches = [
        requirement

        for requirement
        in requirements

        if (
            requirement.requirement_id
            ==
            requirement_id
        )
    ]


    if (
        len(
            matches
        )
        !=
        1
    ):

        return None


    return matches[
        0
    ]


# ============================================================
# MAIN
# ============================================================


def main() -> None:

    print(
        "=== DATALENS OBJECTIVE COVERAGE "
        "AVERAGE BASKET v0.1 ==="
    )

    print()


    planner_catalog = (
        catalog()
    )


    # ========================================================
    # 1. GENERIC CONTRACT COVERAGE ALREADY WORKS
    #
    # This is important:
    # P3-R4 should only need to add semantic extraction.
    # We do not want a bespoke contract-coverage path if the
    # generic metric+role+aggregation machinery already works.
    # ========================================================

    canonical_requirement = (
        canonical_average_basket_requirement()
    )


    correct_contract = (
        mean_basket_contract()
    )


    direct_correct_coverage = (
        contract_covers_requirement(
            contract=
                correct_contract,

            requirement=
                canonical_requirement,
        )
    )


    assert (
        direct_correct_coverage
        is True
    ), (
        "Existing generic Objective Coverage machinery should "
        "already cover mean(basket_amount)."
    )


    wrong_sum_contract = (
        mean_basket_contract(
            function=
                "sum",
        )
    )


    assert (
        contract_covers_requirement(
            contract=
                wrong_sum_contract,

            requirement=
                canonical_requirement,
        )
        is False
    ), (
        "sum(basket_amount) must not satisfy average_basket."
    )


    wrong_measure_contract = (
        mean_basket_contract(
            column=
                "item_count",
        )
    )


    assert (
        contract_covers_requirement(
            contract=
                wrong_measure_contract,

            requirement=
                canonical_requirement,
        )
        is False
    ), (
        "mean(item_count) must not satisfy average_basket."
    )


    blocked_contract = (
        mean_basket_contract(
            status=
                "blocked",
        )
    )


    assert (
        contract_covers_requirement(
            contract=
                blocked_contract,

            requirement=
                canonical_requirement,
        )
        is False
    ), (
        "Blocked contracts must not satisfy average_basket."
    )


    print(
        "[PASS] generic mean(basket_amount) coverage already works"
    )

    print(
        "[PASS] sum(basket_amount) does not cover average_basket"
    )

    print(
        "[PASS] wrong measure does not cover average_basket"
    )

    print(
        "[PASS] blocked contract does not cover average_basket"
    )


    # ========================================================
    # 2. FRENCH REQUIREMENT EXTRACTION
    # ========================================================

    french_requirements = (
        extract_objective_requirements(
            objective=
                FRENCH_OBJECTIVE,

            catalog=
                planner_catalog,
        )
    )


    french_requirement = (
        requirement_by_id(
            french_requirements,
            AVERAGE_BASKET_REQUIREMENT_ID,
        )
    )


    french_extracted = (
        french_requirement
        is not None
    )


    french_candidate_ok = (
        french_requirement
        is not None
        and
        french_requirement.candidate_columns
        ==
        [
            BASKET_COLUMN
        ]
    )


    french_role_ok = (
        french_requirement
        is not None
        and
        french_requirement.allowed_roles
        ==
        [
            "value"
        ]
    )


    french_mean_ok = (
        french_requirement
        is not None
        and
        french_requirement.required_aggregation
        ==
        "mean"
    )


    french_type_ok = (
        french_requirement
        is not None
        and
        french_requirement.requirement_type
        ==
        "metric"
    )


    # ========================================================
    # 3. ENGLISH REQUIREMENT EXTRACTION
    # ========================================================

    english_requirements = (
        extract_objective_requirements(
            objective=
                ENGLISH_OBJECTIVE,

            catalog=
                planner_catalog,
        )
    )


    english_requirement = (
        requirement_by_id(
            english_requirements,
            AVERAGE_BASKET_REQUIREMENT_ID,
        )
    )


    english_extracted = (
        english_requirement
        is not None
    )


    english_mean_ok = (
        english_requirement
        is not None
        and
        english_requirement.required_aggregation
        ==
        "mean"
    )


    # ========================================================
    # 4. POSITIVE COMPLETE REPORT
    # ========================================================

    positive_report = (
        build_objective_coverage(
            objective=
                FRENCH_OBJECTIVE,

            catalog=
                planner_catalog,

            contracts=[
                correct_contract
            ],
        )
    )


    positive_report_complete = (
        positive_report.status
        ==
        "complete"
        and
        positive_report.requirement_count
        ==
        1
        and
        positive_report.covered_count
        ==
        1
        and
        positive_report.missing_count
        ==
        0
    )


    # ========================================================
    # 5. WRONG SUM MUST BE INCOMPLETE
    # ========================================================

    wrong_sum_report = (
        build_objective_coverage(
            objective=
                FRENCH_OBJECTIVE,

            catalog=
                planner_catalog,

            contracts=[
                wrong_sum_contract
            ],
        )
    )


    wrong_sum_fail_closed = (
        wrong_sum_report.status
        ==
        "incomplete"
        and
        wrong_sum_report.requirement_count
        ==
        1
        and
        wrong_sum_report.covered_count
        ==
        0
        and
        wrong_sum_report.missing_count
        ==
        1
    )


    # ========================================================
    # 6. NO CONTRACT MUST BE INCOMPLETE
    # ========================================================

    no_contract_report = (
        build_objective_coverage(
            objective=
                FRENCH_OBJECTIVE,

            catalog=
                planner_catalog,

            contracts=[],
        )
    )


    no_contract_fail_closed = (
        no_contract_report.status
        ==
        "incomplete"
        and
        no_contract_report.requirement_count
        ==
        1
        and
        no_contract_report.missing_count
        ==
        1
    )


    # ========================================================
    # 7. CONCEPT STILL REQUIRED IF PHYSICAL COLUMN IS ABSENT
    #
    # Semantic requirement extraction must not disappear merely
    # because the current catalog lacks basket_amount.
    #
    # It must remain visible with candidates=[] so DataLens can
    # report an unmet objective rather than `not_applicable`.
    # ========================================================

    absent_catalog = (
        catalog(
            include_basket=
                False
        )
    )


    absent_requirements = (
        extract_objective_requirements(
            objective=
                FRENCH_OBJECTIVE,

            catalog=
                absent_catalog,
        )
    )


    absent_requirement = (
        requirement_by_id(
            absent_requirements,
            AVERAGE_BASKET_REQUIREMENT_ID,
        )
    )


    absent_requirement_visible = (
        absent_requirement
        is not None
        and
        absent_requirement.candidate_columns
        ==
        []
    )


    absent_report = (
        build_objective_coverage(
            objective=
                FRENCH_OBJECTIVE,

            catalog=
                absent_catalog,

            contracts=[],
        )
    )


    absent_report_fail_closed = (
        absent_report.status
        ==
        "incomplete"
        and
        absent_report.requirement_count
        ==
        1
        and
        absent_report.covered_count
        ==
        0
        and
        absent_report.missing_count
        ==
        1
    )


    # ========================================================
    # 8. NEGATIVE SEMANTIC GUARDS
    # ========================================================

    vague_requirements = (
        extract_objective_requirements(
            objective=
                "Analyse les sessions d'achat.",

            catalog=
                planner_catalog,
        )
    )


    vague_average_basket = (
        requirement_by_id(
            vague_requirements,
            AVERAGE_BASKET_REQUIREMENT_ID,
        )
    )


    assert (
        vague_average_basket
        is None
    ), (
        "A vague session request must not invent average_basket."
    )


    total_basket_requirements = (
        extract_objective_requirements(
            objective=
                "Quel est le panier total par session ?",

            catalog=
                planner_catalog,
        )
    )


    total_average_basket = (
        requirement_by_id(
            total_basket_requirements,
            AVERAGE_BASKET_REQUIREMENT_ID,
        )
    )


    assert (
        total_average_basket
        is None
    ), (
        "`panier total` must not be silently rewritten to "
        "`panier moyen`."
    )


    print(
        "[PASS] vague session request does not invent average_basket"
    )

    print(
        "[PASS] total basket request does not invent average_basket"
    )


    # ========================================================
    # 9. OBSERVATION
    # ========================================================

    print()
    print(
        "Objective Coverage rule                 "
        f"{OBJECTIVE_COVERAGE_RULE_VERSION}"
    )

    print(
        "French requirement count                "
        f"{len(french_requirements)}"
    )

    print(
        "French average-basket requirement       "
        f"{french_requirement.model_dump() if french_requirement else None}"
    )

    print(
        "English requirement count               "
        f"{len(english_requirements)}"
    )

    print(
        "English average-basket requirement      "
        f"{english_requirement.model_dump() if english_requirement else None}"
    )

    print()
    print(
        "French requirement extracted            "
        f"{french_extracted}"
    )

    print(
        "French candidate basket_amount          "
        f"{french_candidate_ok}"
    )

    print(
        "French role=value                       "
        f"{french_role_ok}"
    )

    print(
        "French aggregation=mean                 "
        f"{french_mean_ok}"
    )

    print(
        "French requirement type=metric          "
        f"{french_type_ok}"
    )

    print(
        "English requirement extracted           "
        f"{english_extracted}"
    )

    print(
        "English aggregation=mean                "
        f"{english_mean_ok}"
    )

    print()
    print(
        "Positive report status                  "
        f"{positive_report.status}"
    )

    print(
        "Positive report counts                  "
        f"{positive_report.requirement_count}/"
        f"{positive_report.covered_count}/"
        f"{positive_report.missing_count}"
    )

    print(
        "Wrong-SUM report status                 "
        f"{wrong_sum_report.status}"
    )

    print(
        "No-contract report status               "
        f"{no_contract_report.status}"
    )

    print(
        "Absent-column requirement visible       "
        f"{absent_requirement_visible}"
    )

    print(
        "Absent-column report status             "
        f"{absent_report.status}"
    )


    # ========================================================
    # 10. RED GAPS
    # ========================================================

    gaps: list[
        str
    ] = []


    if not french_extracted:

        gaps.append(
            "average_basket_french_requirement_extraction_gap"
        )


    if not french_candidate_ok:

        gaps.append(
            "average_basket_candidate_resolution_gap"
        )


    if not french_role_ok:

        gaps.append(
            "average_basket_value_role_gap"
        )


    if not french_mean_ok:

        gaps.append(
            "average_basket_required_mean_gap"
        )


    if not french_type_ok:

        gaps.append(
            "average_basket_metric_type_gap"
        )


    if not english_extracted:

        gaps.append(
            "average_basket_english_requirement_extraction_gap"
        )


    if not english_mean_ok:

        gaps.append(
            "average_basket_english_required_mean_gap"
        )


    if not positive_report_complete:

        gaps.append(
            "average_basket_positive_report_completion_gap"
        )


    if not wrong_sum_fail_closed:

        gaps.append(
            "average_basket_wrong_sum_report_fail_closed_gap"
        )


    if not no_contract_fail_closed:

        gaps.append(
            "average_basket_missing_contract_report_gap"
        )


    if not absent_requirement_visible:

        gaps.append(
            "average_basket_missing_physical_candidate_visibility_gap"
        )


    if not absent_report_fail_closed:

        gaps.append(
            "average_basket_missing_physical_candidate_report_gap"
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
        "PASS - Objective Coverage average basket v0.1"
    )


if __name__ == "__main__":

    main()