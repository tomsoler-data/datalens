# DATALENS_OBJECTIVE_COVERAGE_V0_3_REGRESSION_ALIGNMENT
from __future__ import annotations


from app.planning.ai_analytical_planner import (
    PlannerCatalog,
    PlannerColumnProfile,
    PlannerDatasetProfile,
)

from app.planning.analytical_contract import (
    AggregationSpec,
    AnalyticalContract,
    VariableBinding,
)

from app.planning.objective_coverage import (
    OBJECTIVE_COVERAGE_RULE_VERSION,
    build_objective_coverage,
)


OBJECTIVE = (
    "Compare le chiffre d'affaires et le taux de retour "
    "par region et par canal. Identifie les regions et les "
    "canaux les plus performants, ceux qui ont un taux de "
    "retour superieur a la moyenne, et resume les principaux "
    "ecarts observes."
)


def build_catalog() -> PlannerCatalog:
    columns = [
        PlannerColumnProfile(
            name=
                "transaction_id",
            dtype=
                "object",
            analysis_kind=
                "categorical",
            missing_ratio=
                0.0,
            unique_count=
                720,
        ),
        PlannerColumnProfile(
            name=
                "region",
            dtype=
                "object",
            analysis_kind=
                "categorical",
            missing_ratio=
                0.0,
            unique_count=
                4,
        ),
        PlannerColumnProfile(
            name=
                "channel",
            dtype=
                "object",
            analysis_kind=
                "categorical",
            missing_ratio=
                0.0,
            unique_count=
                3,
        ),
        PlannerColumnProfile(
            name=
                "product_category",
            dtype=
                "object",
            analysis_kind=
                "categorical",
            missing_ratio=
                0.0,
            unique_count=
                4,
        ),
        PlannerColumnProfile(
            name=
                "revenue",
            dtype=
                "float64",
            analysis_kind=
                "quantitative",
            missing_ratio=
                0.0,
            unique_count=
                650,
        ),
        PlannerColumnProfile(
            name=
                "returned_order",
            dtype=
                "bool",
            analysis_kind=
                "categorical",
            missing_ratio=
                0.0,
            unique_count=
                2,
        ),
    ]

    dataset = (
        PlannerDatasetProfile(
            dataset_id=
                "dataset:demo",
            filename=
                "datalens_demo_sales.csv",
            row_count=
                720,
            column_count=
                len(
                    columns
                ),
            columns=
                columns,
        )
    )

    return (
        PlannerCatalog(
            datasets=[
                dataset
            ]
        )
    )


def binding(
    role: str,
    column: str,
) -> VariableBinding:
    return (
        VariableBinding(
            role=
                role,
            column=
                column,
            dataset_id=
                "dataset:demo",
            dataset_filename=
                "datalens_demo_sales.csv",
        )
    )


def bad_current_contract() -> AnalyticalContract:
    return (
        AnalyticalContract(
            contract_id=
                "contract:bad-current",
            origin=
                "ai_planner",
            status=
                "validated",
            title=
                "Region by product category",
            request_text=
                OBJECTIVE,
            family=
                "categorical_association",
            required_dataset_ids=[
                "dataset:demo"
            ],
            required_dataset_filenames=[
                "datalens_demo_sales.csv"
            ],
            bindings=[
                binding(
                    "x",
                    "region",
                ),
                binding(
                    "y",
                    "product_category",
                ),
            ],
        )
    )



def good_revenue_contract(
    dimension: str,
) -> AnalyticalContract:
    if dimension not in {
        "region",
        "channel",
    }:
        raise ValueError(
            "Unsupported demo dimension."
        )


    return (
        AnalyticalContract(
            contract_id=(
                "contract:revenue:"
                +
                dimension
            ),

            origin=
                "ai_planner",

            status=
                "validated",

            title=(
                "Revenue by "
                +
                dimension
            ),

            request_text=
                OBJECTIVE,

            family=
                "aggregation",

            required_dataset_ids=[
                "dataset:demo"
            ],

            required_dataset_filenames=[
                "datalens_demo_sales.csv"
            ],

            bindings=[
                binding(
                    "value",
                    "revenue",
                ),

                binding(
                    "group",
                    dimension,
                ),
            ],

            aggregation=
                AggregationSpec(
                    function=
                        "sum",

                    source_role=
                        "value",

                    group_by_roles=[
                        "group",
                    ],
                ),
        )
    )


def good_return_rate_contract(
    dimension: str,
) -> AnalyticalContract:
    if dimension not in {
        "region",
        "channel",
    }:
        raise ValueError(
            "Unsupported demo dimension."
        )


    return (
        AnalyticalContract(
            contract_id=(
                "contract:return-rate:"
                +
                dimension
            ),

            origin=
                "ai_planner",

            status=
                "validated",

            title=(
                "Return rate by "
                +
                dimension
            ),

            request_text=
                OBJECTIVE,

            family=
                "aggregation",

            required_dataset_ids=[
                "dataset:demo"
            ],

            required_dataset_filenames=[
                "datalens_demo_sales.csv"
            ],

            bindings=[
                binding(
                    "value",
                    "returned_order",
                ),

                binding(
                    "group",
                    dimension,
                ),
            ],

            aggregation=
                AggregationSpec(
                    function=
                        "mean",

                    source_role=
                        "value",

                    group_by_roles=[
                        "group",
                    ],
                ),
        )
    )

def main() -> None:
    catalog = (
        build_catalog()
    )


    print()
    print("=" * 80)
    print(
        "DATALENS OBJECTIVE COVERAGE v0.1"
    )
    print("=" * 80)
    print()


    # ========================================================
    # CASE 1 - CURRENT WRONG PLAN
    # ========================================================

    bad_report = (
        build_objective_coverage(
            objective=
                OBJECTIVE,
            catalog=
                catalog,
            contracts=[
                bad_current_contract()
            ],
        )
    )


    assert (
        bad_report.status
        ==
        "incomplete"
    )

    assert (
        bad_report.requirement_count
        ==
        4
    )

    assert (
        bad_report.covered_count
        ==
        1
    )

    assert (
        bad_report.missing_count
        ==
        3
    )


    missing_concepts = {
        requirement.concept

        for requirement
        in bad_report.requirements

        if not requirement.covered
    }


    assert (
        missing_concepts
        ==
        {
            "revenue_total",
            "return_rate",
            "channel",
        }
    )


    print(
        "[PASS] current region/product_category plan is incomplete"
    )


    # ========================================================
    # CASE 2 - UNION OF TWO GOOD CONTRACTS
    # ========================================================

    good_report = (
        build_objective_coverage(
            objective=
                OBJECTIVE,
            catalog=
                catalog,
            contracts=[
                good_revenue_contract(
                    "region"
                ),

                good_revenue_contract(
                    "channel"
                ),

                good_return_rate_contract(
                    "region"
                ),

                good_return_rate_contract(
                    "channel"
                ),
            ],
        )
    )


    assert (
        good_report.status
        ==
        "complete"
    )

    assert (
        good_report.requirement_count
        ==
        4
    )

    assert (
        good_report.covered_count
        ==
        4
    )

    assert (
        good_report.missing_count
        ==
        0
    )


    print(
        "[PASS] four marginal revenue/return-rate contracts are complete"
    )


    # ========================================================
    # CASE 3 - AGGREGATION SEMANTICS MATTER
    # DATALENS_OBJECTIVE_COVERAGE_V0_3_CASE3_ALIGNMENT
    # ========================================================

    wrong_return_contract = (
        good_return_rate_contract(
            "region"
        )
        .model_copy(
            update={
                "contract_id":
                    "contract:return-count",

                "aggregation":
                    AggregationSpec(
                        function=
                            "sum",
                        source_role=
                            "value",
                        group_by_roles=[
                            "group",
                        ],
                    ),
            }
        )
    )


    wrong_metric_report = (
        build_objective_coverage(
            objective=
                OBJECTIVE,
            catalog=
                catalog,
            contracts=[
                good_revenue_contract(
                    "region"
                ),

                good_revenue_contract(
                    "channel"
                ),

                wrong_return_contract,

                good_return_rate_contract(
                    "channel"
                ),
            ],
        )
    )


    assert (
        wrong_metric_report.status
        ==
        "incomplete"
    )

    # The return-rate concept remains covered globally by the
    # valid channel contract. The failure is now correctly
    # localized to the missing return_rate x region marginal.
    assert (
        wrong_metric_report.missing_count
        ==
        0
    )


    assert (
        wrong_metric_report.topology_missing_count
        ==
        1
    )


    missing_topology = [
        topology.topology_id

        for topology
        in wrong_metric_report.topology_requirements

        if not topology.covered
    ]


    assert (
        missing_topology
        ==
        [
            "topology:return_rate:by:region"
        ]
    )


    print(
        "[PASS] returned_order sum does not satisfy return rate"
    )


    # ========================================================
    # CASE 4 - NO CONSERVATIVE REQUIREMENT
    # ========================================================

    neutral_report = (
        build_objective_coverage(
            objective=
                "Detecte les valeurs atypiques.",
            catalog=
                catalog,
            contracts=[],
        )
    )


    assert (
        neutral_report.status
        ==
        "not_applicable"
    )


    print(
        "[PASS] unrelated objective is not over-constrained"
    )


    # ========================================================
    # VERSION
    # ========================================================

    assert (
        OBJECTIVE_COVERAGE_RULE_VERSION
        ==
        "objective_coverage_v0.3"
    )


    print(
        "[PASS] objective coverage rule version"
    )


    print()
    print(
        "PASS - Objective Coverage v0.1"
    )


if __name__ == "__main__":
    main()
