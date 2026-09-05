from __future__ import annotations


import pandas as pd


from app.planning.analytical_contract import (
    AggregationSpec,
    AnalyticalContract,
    RankingSpec,
    ShareOfTotalSpec,
    VariableBinding,
)


from app.planning.objective_coverage import (
    OBJECTIVE_COVERAGE_RULE_VERSION,
    build_objective_coverage,
    contract_covers_requirement,
    extract_objective_requirements,
)


from app.planning.planner_catalog import (
    planner_catalog_from_dataset_records,
)


# ============================================================
# OBJECTIVE
# ============================================================


OBJECTIVE = (
    "Quelle catégorie contribue le plus au chiffre d'affaires "
    "et quelle est sa part approximative du chiffre d'affaires total ?"
)


SOURCE_DATASET_ID = (
    "dataset_source"
)


CATEGORY_DATASET_ID = (
    "derived:dataset_source:category:category:price"
)


SHARE_REQUIREMENT_ID = (
    "derived:share_of_total"
)


SHARE_REFERENCE = (
    "sum_of_group_values"
)


TRUSTED_MONETARY_EVENT_SEMANTICS = (
    "The unit monetary measure was propagated from a validated "
    "dimension to fact grain. No explicit quantity measure was "
    "detected, so one fact row is conservatively treated as one "
    "monetary event."
)


# ============================================================
# GENERIC SERVER-OWNED CATALOG
# ============================================================


def build_catalog():

    source = {
        "dataset_id":
            SOURCE_DATASET_ID,

        "filename":
            "source.csv",

        "dataframe":
            pd.DataFrame(
                {
                    "event_id": [
                        "e1",
                        "e2",
                        "e3",
                    ],

                    "category": [
                        "A",
                        "B",
                        "C",
                    ],

                    "price": [
                        40.0,
                        35.0,
                        25.0,
                    ],
                }
            ),
    }


    category = {
        "dataset_id":
            CATEGORY_DATASET_ID,

        "filename":
            "source__by_category_price.derived",

        "dataframe":
            pd.DataFrame(
                {
                    "category": [
                        "A",
                        "B",
                        "C",
                    ],

                    "sum_price": [
                        40.0,
                        35.0,
                        25.0,
                    ],

                    "event_count": [
                        1,
                        1,
                        1,
                    ],
                }
            ),

        "is_derived":
            True,

        "derivation_type":
            "categorical_additive_measure",

        "source_dataset_ids": [
            SOURCE_DATASET_ID,
        ],

        "provenance": {
            "fact_dataset_id":
                SOURCE_DATASET_ID,

            "operation":
                "groupby_sum",

            "group_column":
                "category",

            "source_measure_column":
                "price",

            "target_measure_column":
                "sum_price",

            "aggregation":
                "sum",

            "grain":
                "category",

            "metric_semantics":
                TRUSTED_MONETARY_EVENT_SEMANTICS,
        },
    }


    return (
        planner_catalog_from_dataset_records(
            [
                source,
                category,
            ]
        )
    )


# ============================================================
# CONTRACT
# ============================================================


def build_contract(
    *,
    include_share: bool,
    status: str = "validated",
) -> AnalyticalContract:

    return (
        AnalyticalContract(
            contract_id=(
                "contract:share-satisfaction:"
                +
                (
                    "with-share"
                    if include_share
                    else
                    "without-share"
                )
                +
                ":"
                +
                status
            ),

            origin=
                "ai_planner",

            status=status,  # type: ignore[arg-type]

            title=
                "Category revenue leader",

            request_text=
                OBJECTIVE,

            family=
                "ranking",

            required_dataset_ids=[
                CATEGORY_DATASET_ID,
            ],

            required_dataset_filenames=[
                "source__by_category_price.derived",
            ],

            analytical_grain=
                "category",

            bindings=[
                VariableBinding(
                    role=
                        "value",

                    column=
                        "sum_price",

                    dataset_id=
                        CATEGORY_DATASET_ID,

                    dataset_filename=
                        "source__by_category_price.derived",

                    semantic_concept=
                        "price",

                    analysis_kind=
                        "quantitative",
                ),

                VariableBinding(
                    role=
                        "dimension",

                    column=
                        "category",

                    dataset_id=
                        CATEGORY_DATASET_ID,

                    dataset_filename=
                        "source__by_category_price.derived",

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

            share_of_total=(
                ShareOfTotalSpec(
                    reference=
                        SHARE_REFERENCE
                )
                if include_share
                else None
            ),

            window=
                None,

            filters=[],
            joins=[],
            derived_variables=[],

            required_operations=[
                "Execute deterministic ranking.",
            ],

            reasons=[
                "Generic Objective Coverage share satisfaction test.",
            ],

            blockers=[],

            planner_confidence=
                1.0,
        )
    )


# ============================================================
# MAIN
# ============================================================


def main() -> None:

    print(
        "=== DATALENS OBJECTIVE COVERAGE "
        "SHARE CONTRACT SATISFACTION v0.1 ==="
    )

    print()


    catalog = (
        build_catalog()
    )


    requirements = (
        extract_objective_requirements(
            objective=
                OBJECTIVE,

            catalog=
                catalog,
        )
    )


    share_requirements = [
        requirement

        for requirement
        in requirements

        if (
            requirement.requirement_id
            ==
            SHARE_REQUIREMENT_ID
        )
    ]


    revenue_requirements = [
        requirement

        for requirement
        in requirements

        if (
            requirement.concept
            ==
            "revenue_total"
        )
    ]


    assert (
        len(
            share_requirements
        )
        ==
        1
    )


    assert (
        len(
            revenue_requirements
        )
        ==
        1
    )


    share_requirement = (
        share_requirements[
            0
        ]
    )


    revenue_requirement = (
        revenue_requirements[
            0
        ]
    )


    assert (
        share_requirement.requirement_type
        ==
        "derived_metric"
    )


    assert (
        share_requirement.candidate_columns
        ==
        []
    )


    print(
        "[PASS] share_of_total requirement extracted"
    )

    print(
        "[PASS] share_of_total has no physical candidate column"
    )


    # ========================================================
    # CANONICAL CONTRACT WITH SHARE
    # ========================================================

    with_share = (
        build_contract(
            include_share=
                True,
        )
    )


    assert (
        with_share.share_of_total
        is not None
    )


    assert (
        with_share
        .share_of_total
        .reference
        ==
        SHARE_REFERENCE
    )


    print(
        "[PASS] canonical ShareOfTotalSpec present on contract"
    )


    # Existing revenue coverage must remain intact.
    revenue_covered = (
        contract_covers_requirement(
            contract=
                with_share,

            requirement=
                revenue_requirement,
        )
    )


    assert (
        revenue_covered
        is True
    )


    print(
        "[PASS] existing revenue requirement remains covered"
    )


    # ========================================================
    # NEGATIVE CONTROLS
    # ========================================================

    without_share = (
        build_contract(
            include_share=
                False,
        )
    )


    without_share_covered = (
        contract_covers_requirement(
            contract=
                without_share,

            requirement=
                share_requirement,
        )
    )


    assert (
        without_share_covered
        is False
    ), (
        "A contract without ShareOfTotalSpec must not cover "
        "derived:share_of_total."
    )


    print(
        "[PASS] contract without ShareOfTotalSpec remains incomplete"
    )


    proposed_with_share = (
        build_contract(
            include_share=
                True,

            status=
                "proposed",
        )
    )


    proposed_covered = (
        contract_covers_requirement(
            contract=
                proposed_with_share,

            requirement=
                share_requirement,
        )
    )


    assert (
        proposed_covered
        is False
    ), (
        "Only validated contracts may satisfy Objective Coverage."
    )


    print(
        "[PASS] proposed contract remains non-covering"
    )


    # ========================================================
    # POSITIVE TARGET
    # ========================================================

    positive_direct_coverage = (
        contract_covers_requirement(
            contract=
                with_share,

            requirement=
                share_requirement,
        )
    )


    positive_report = (
        build_objective_coverage(
            objective=
                OBJECTIVE,

            catalog=
                catalog,

            contracts=[
                with_share,
            ],
        )
    )


    positive_share_reports = [
        requirement

        for requirement
        in positive_report.requirements

        if (
            requirement.requirement_id
            ==
            SHARE_REQUIREMENT_ID
        )
    ]


    assert (
        len(
            positive_share_reports
        )
        ==
        1
    )


    positive_share_report = (
        positive_share_reports[
            0
        ]
    )


    negative_report = (
        build_objective_coverage(
            objective=
                OBJECTIVE,

            catalog=
                catalog,

            contracts=[
                without_share,
            ],
        )
    )


    assert (
        negative_report.status
        ==
        "incomplete"
    )


    print()
    print(
        "Objective Coverage rule                 "
        f"{OBJECTIVE_COVERAGE_RULE_VERSION}"
    )

    print(
        "Positive contract status                "
        f"{with_share.status}"
    )

    print(
        "Positive contract share spec            "
        f"{with_share.share_of_total.model_dump()}"
    )

    print(
        "Direct share coverage                   "
        f"{positive_direct_coverage}"
    )

    print(
        "Share requirement covered               "
        f"{positive_share_report.covered}"
    )

    print(
        "Share covered by contracts              "
        f"{positive_share_report.covered_by_contract_ids}"
    )

    print(
        "Positive report status                  "
        f"{positive_report.status}"
    )

    print(
        "Positive report covered                 "
        f"{positive_report.covered_count}"
    )

    print(
        "Positive report missing                 "
        f"{positive_report.missing_count}"
    )

    print(
        "Without-share report                    "
        f"{negative_report.status}"
    )


    # ========================================================
    # EXPECTED CURRENT GAPS
    # ========================================================

    gaps: list[
        str
    ] = []


    if not (
        positive_direct_coverage
    ):

        gaps.append(
            "share_of_total_contract_coverage_gap"
        )


    if not (
        positive_share_report.covered
    ):

        gaps.append(
            "share_of_total_requirement_satisfaction_gap"
        )


    if (
        positive_report.status
        !=
        "complete"
    ):

        gaps.append(
            "share_of_total_report_completion_gap"
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
        "PASS - Objective Coverage share contract satisfaction v0.1"
    )


if __name__ == "__main__":

    main()