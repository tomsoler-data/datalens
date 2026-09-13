from __future__ import annotations


from app.planning.ai_analytical_planner import (
    PlannerCatalog,
    PlannerColumnProfile,
    PlannerDatasetProfile,
)

from app.planning.analytical_contract import (
    AggregationSpec,
    AnalyticalContract,
    RankingSpec,
    VariableBinding,
)

from app.planning.objective_coverage import (
    build_objective_coverage,
)


OBJECTIVE = (
    "Analyse les ventes : indique le chiffre d'affaires total, "
    "montre son évolution mensuelle, compare le chiffre d'affaires "
    "par catégorie de produits et identifie les 10 clients qui "
    "génèrent le plus de chiffre d'affaires."
)


SCALAR_DATASET_ID = (
    "derived:demo:scalar:price"
)

MONTHLY_DATASET_ID = (
    "derived:demo:monthly:date:price"
)

CATEGORY_DATASET_ID = (
    "derived:demo:category:categ:price"
)

CUSTOMER_DATASET_ID = (
    "derived:demo:customer:client_id:price"
)


def column(
    name: str,
    *,
    dtype: str,
    analysis_kind: str,
    unique_count: int,
) -> PlannerColumnProfile:
    return PlannerColumnProfile(
        name=name,
        dtype=dtype,
        analysis_kind=analysis_kind,
        missing_ratio=0.0,
        unique_count=unique_count,
    )


def build_catalog() -> PlannerCatalog:
    revenue_aliases = [
        "revenue",
        "turnover",
        "chiffre_affaires",
        "ca",
    ]

    return PlannerCatalog(
        datasets=[
            PlannerDatasetProfile(
                dataset_id=SCALAR_DATASET_ID,
                filename="sales__overall_price.derived",
                row_count=1,
                column_count=2,
                columns=[
                    column(
                        "sum_price",
                        dtype="float64",
                        analysis_kind="quantitative",
                        unique_count=1,
                    ),
                    column(
                        "event_count",
                        dtype="int64",
                        analysis_kind="quantitative",
                        unique_count=1,
                    ),
                ],
                is_derived=True,
                derivation_type="scalar_additive_measure",
                analytical_grain="overall",
                operation="scalar_sum",
                aggregation="sum",
                source_measure_column="price",
                target_measure_column="sum_price",
                measure_semantic_aliases=revenue_aliases,
            ),

            PlannerDatasetProfile(
                dataset_id=MONTHLY_DATASET_ID,
                filename="sales__monthly_price.derived",
                row_count=24,
                column_count=3,
                columns=[
                    column(
                        "month",
                        dtype="datetime64[us]",
                        analysis_kind="temporal",
                        unique_count=24,
                    ),
                    column(
                        "sum_price",
                        dtype="float64",
                        analysis_kind="quantitative",
                        unique_count=24,
                    ),
                    column(
                        "event_count",
                        dtype="int64",
                        analysis_kind="quantitative",
                        unique_count=24,
                    ),
                ],
                is_derived=True,
                derivation_type="monthly_additive_measure",
                analytical_grain="month",
                operation="groupby_sum",
                aggregation="sum",
                source_time_column="date",
                target_time_column="month",
                source_measure_column="price",
                target_measure_column="sum_price",
                measure_semantic_aliases=revenue_aliases,
            ),

            PlannerDatasetProfile(
                dataset_id=CATEGORY_DATASET_ID,
                filename="sales__by_categ_price.derived",
                row_count=3,
                column_count=3,
                columns=[
                    column(
                        "categ",
                        dtype="int64",
                        analysis_kind="categorical",
                        unique_count=3,
                    ),
                    column(
                        "sum_price",
                        dtype="float64",
                        analysis_kind="quantitative",
                        unique_count=3,
                    ),
                    column(
                        "event_count",
                        dtype="int64",
                        analysis_kind="quantitative",
                        unique_count=3,
                    ),
                ],
                is_derived=True,
                derivation_type="categorical_additive_measure",
                analytical_grain="categ",
                operation="groupby_sum",
                aggregation="sum",
                group_column="categ",
                source_measure_column="price",
                target_measure_column="sum_price",
                measure_semantic_aliases=revenue_aliases,
            ),

            PlannerDatasetProfile(
                dataset_id=CUSTOMER_DATASET_ID,
                filename="sales__customers_price.derived",
                row_count=8600,
                column_count=2,
                columns=[
                    column(
                        "client_id",
                        dtype="str",
                        analysis_kind="identifier",
                        unique_count=8600,
                    ),
                    column(
                        "total_spend",
                        dtype="float64",
                        analysis_kind="quantitative",
                        unique_count=8500,
                    ),
                ],
                is_derived=True,
                derivation_type="customer_additive_measure",
                analytical_grain="client_id",
                operation="groupby_sum",
                aggregation="sum",
                group_column="client_id",
                entity_column="client_id",
                source_measure_column="price",
                target_measure_column="total_spend",
                measure_semantic_aliases=revenue_aliases,
            ),
        ]
    )


def scalar_revenue_contract() -> AnalyticalContract:
    return AnalyticalContract(
        contract_id="contract:r10:scalar-revenue",
        origin="ai_planner",
        status="validated",
        title="Chiffre d'affaires total",
        request_text=OBJECTIVE,
        family="aggregation",
        bindings=[
            VariableBinding(
                role="value",
                column="sum_price",
                dataset_id=SCALAR_DATASET_ID,
                dataset_filename="sales__overall_price.derived",
            )
        ],
        aggregation=AggregationSpec(
            function="sum",
            source_role="value",
            group_by_roles=[],
        ),
    )


def monthly_revenue_contract() -> AnalyticalContract:
    return AnalyticalContract(
        contract_id="contract:r10:monthly-revenue",
        origin="ai_planner",
        status="validated",
        title="Évolution mensuelle du chiffre d'affaires",
        request_text=OBJECTIVE,
        family="time_series",
        analytical_grain="month",
        bindings=[
            VariableBinding(
                role="time",
                column="month",
                dataset_id=MONTHLY_DATASET_ID,
                dataset_filename="sales__monthly_price.derived",
            ),
            VariableBinding(
                role="value",
                column="sum_price",
                dataset_id=MONTHLY_DATASET_ID,
                dataset_filename="sales__monthly_price.derived",
            ),
        ],
        aggregation=AggregationSpec(
            function="sum",
            source_role="value",
            group_by_roles=[
                "time"
            ],
        ),
    )


def category_revenue_contract() -> AnalyticalContract:
    return AnalyticalContract(
        contract_id="contract:r10:category-revenue",
        origin="ai_planner",
        status="validated",
        title="Chiffre d'affaires par catégorie",
        request_text=OBJECTIVE,
        family="aggregation",
        analytical_grain="categ",
        bindings=[
            VariableBinding(
                role="group",
                column="categ",
                dataset_id=CATEGORY_DATASET_ID,
                dataset_filename="sales__by_categ_price.derived",
            ),
            VariableBinding(
                role="value",
                column="sum_price",
                dataset_id=CATEGORY_DATASET_ID,
                dataset_filename="sales__by_categ_price.derived",
            ),
        ],
        aggregation=AggregationSpec(
            function="sum",
            source_role="value",
            group_by_roles=[
                "group"
            ],
        ),
    )


def customer_ranking_contract() -> AnalyticalContract:
    return AnalyticalContract(
        contract_id="contract:r10:customer-ranking",
        origin="ai_planner",
        status="validated",
        title="Top 10 clients par chiffre d'affaires",
        request_text=OBJECTIVE,
        family="ranking",
        analytical_grain="client_id",
        bindings=[
            VariableBinding(
                role="dimension",
                column="client_id",
                dataset_id=CUSTOMER_DATASET_ID,
                dataset_filename="sales__customers_price.derived",
            ),
            VariableBinding(
                role="value",
                column="total_spend",
                dataset_id=CUSTOMER_DATASET_ID,
                dataset_filename="sales__customers_price.derived",
            ),
        ],
        aggregation=AggregationSpec(
            function="sum",
            source_role="value",
            group_by_roles=[
                "dimension"
            ],
        ),
        ranking=RankingSpec(
            order="descending",
            limit=10,
        ),
    )


def main() -> None:
    print()
    print("=" * 80)
    print(
        "DATALENS OBJECTIVE COVERAGE "
        "MULTI-INTENT SAME-METRIC v0.1"
    )
    print("=" * 80)
    print()

    report = build_objective_coverage(
        objective=OBJECTIVE,
        catalog=build_catalog(),
        contracts=[
            scalar_revenue_contract()
        ],
    )

    print(
        "Coverage status :",
        report.status,
    )

    print(
        "Requirements    :",
        report.requirement_count,
    )

    print(
        "Missing         :",
        report.missing_count,
    )

    print(
        "Topologies      :",
        report.topology_requirement_count,
    )

    print()

    for requirement in report.requirements:
        print(
            "REQ",
            requirement.concept,
            "covered=",
            requirement.covered,
        )

    for topology in report.topology_requirements:
        print(
            "TOPOLOGY",
            topology.topology_id,
            "covered=",
            topology.covered,
        )

    print()

    # ========================================================
    # INTENT COVERAGE AUTHORITY
    # ========================================================

    assert (
        report.intent_requirement_count
        ==
        4
    ), (
        "The objective must preserve four independent analytical "
        "intents around the same revenue metric."
    )

    assert (
        report.intent_covered_count
        ==
        1
    ), (
        "The scalar contract must cover only the scalar-total "
        "intent."
    )

    assert (
        report.intent_missing_count
        ==
        3
    ), (
        "Monthly, category and customer-ranking intents must "
        "remain missing when only scalar revenue is present."
    )

    intent_by_kind = {
        intent.intent_kind:
            intent

        for intent
        in report.intent_requirements
    }

    assert (
        set(
            intent_by_kind
        )
        ==
        {
            "scalar_total",
            "monthly_time_series",
            "categorical_breakdown",
            "entity_ranking",
        }
    )

    assert (
        intent_by_kind[
            "scalar_total"
        ].covered
        is True
    )

    assert (
        intent_by_kind[
            "monthly_time_series"
        ].covered
        is False
    )

    assert (
        intent_by_kind[
            "monthly_time_series"
        ].required_family
        ==
        "time_series"
    )

    assert (
        intent_by_kind[
            "monthly_time_series"
        ].required_grain
        ==
        "month"
    )

    assert (
        intent_by_kind[
            "categorical_breakdown"
        ].covered
        is False
    )

    assert (
        intent_by_kind[
            "categorical_breakdown"
        ].required_family
        ==
        "aggregation"
    )

    assert (
        intent_by_kind[
            "entity_ranking"
        ].covered
        is False
    )

    assert (
        intent_by_kind[
            "entity_ranking"
        ].required_family
        ==
        "ranking"
    )

    assert (
        intent_by_kind[
            "entity_ranking"
        ].ranking_order
        ==
        "descending"
    )

    assert (
        intent_by_kind[
            "entity_ranking"
        ].ranking_limit
        ==
        10
    )

    print(
        "[PASS] four same-metric analytical intents are "
        "preserved independently"
    )

    print(
        "[PASS] explicit Top-10 descending ranking semantics "
        "are preserved"
    )

    assert (
        report.status
        ==
        "incomplete"
    ), (
        "A scalar revenue contract must NOT satisfy a request "
        "that also explicitly asks for monthly evolution, "
        "category breakdown and Top-N customer ranking."
    )

    print(
        "[PASS] scalar revenue alone does not satisfy "
        "the multi-intent request"
    )


    # ========================================================
    # POSITIVE AUTHORITY — ALL FOUR INTENTS
    # ========================================================

    complete_report = build_objective_coverage(
        objective=OBJECTIVE,
        catalog=build_catalog(),
        contracts=[
            scalar_revenue_contract(),
            monthly_revenue_contract(),
            category_revenue_contract(),
            customer_ranking_contract(),
        ],
    )

    assert (
        complete_report.status
        ==
        "complete"
    ), (
        "The union of scalar, monthly, category and Top-10 "
        "customer contracts must fully satisfy the objective."
    )

    assert (
        complete_report.intent_requirement_count
        ==
        4
    )

    assert (
        complete_report.intent_covered_count
        ==
        4
    )

    assert (
        complete_report.intent_missing_count
        ==
        0
    )

    complete_by_kind = {
        intent.intent_kind:
            intent

        for intent
        in complete_report.intent_requirements
    }

    assert all(
        intent.covered

        for intent
        in complete_report.intent_requirements
    )

    assert (
        complete_by_kind[
            "scalar_total"
        ].covered_by_contract_ids
        ==
        [
            "contract:r10:scalar-revenue"
        ]
    )

    assert (
        complete_by_kind[
            "monthly_time_series"
        ].covered_by_contract_ids
        ==
        [
            "contract:r10:monthly-revenue"
        ]
    )

    assert (
        complete_by_kind[
            "categorical_breakdown"
        ].covered_by_contract_ids
        ==
        [
            "contract:r10:category-revenue"
        ]
    )

    assert (
        complete_by_kind[
            "entity_ranking"
        ].covered_by_contract_ids
        ==
        [
            "contract:r10:customer-ranking"
        ]
    )

    print(
        "[PASS] four validated contracts cover four "
        "independent same-metric intents"
    )

    print(
        "[PASS] complete intent coverage restores "
        "objective status=complete"
    )

    print()
    print(
        "PASS - objective coverage multi-intent "
        "same-metric v0.1"
    )


if __name__ == "__main__":
    main()
