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
    "par region et par canal."
)


def catalog() -> PlannerCatalog:
    columns = [
        PlannerColumnProfile(
            name=
                name,

            dtype=(
                "float64"
                if name == "revenue"
                else
                (
                    "bool"
                    if name == "returned_order"
                    else
                    "object"
                )
            ),

            analysis_kind=(
                "quantitative"
                if name == "revenue"
                else
                "categorical"
            ),

            missing_ratio=
                0.0,

            unique_count=
                10,
        )

        for name in (
            "region",
            "channel",
            "revenue",
            "returned_order",
        )
    ]


    return (
        PlannerCatalog(
            datasets=[
                PlannerDatasetProfile(
                    dataset_id=
                        "dataset:demo",

                    filename=
                        "demo.csv",

                    row_count=
                        720,

                    column_count=
                        len(
                            columns
                        ),

                    columns=
                        columns,
                )
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
                "demo.csv",
        )
    )


def metric_by(
    *,
    contract_id: str,
    metric: str,
    reducer: str,
    dimension: str,
) -> AnalyticalContract:
    return (
        AnalyticalContract(
            contract_id=
                contract_id,

            origin=
                "ai_planner",

            status=
                "validated",

            title=
                contract_id,

            request_text=
                OBJECTIVE,

            family=
                "aggregation",

            bindings=[
                binding(
                    "value",
                    metric,
                ),

                binding(
                    "group",
                    dimension,
                ),
            ],

            aggregation=
                AggregationSpec(
                    function=
                        reducer,

                    source_role=
                        "value",

                    group_by_roles=[
                        "group"
                    ],
                ),
        )
    )


def joint_metric(
    *,
    contract_id: str,
    metric: str,
    reducer: str,
) -> AnalyticalContract:
    return (
        AnalyticalContract(
            contract_id=
                contract_id,

            origin=
                "ai_planner",

            status=
                "validated",

            title=
                contract_id,

            request_text=
                OBJECTIVE,

            family=
                "aggregation",

            bindings=[
                binding(
                    "value",
                    metric,
                ),

                binding(
                    "group",
                    "region",
                ),

                binding(
                    "dimension",
                    "channel",
                ),
            ],

            aggregation=
                AggregationSpec(
                    function=
                        reducer,

                    source_role=
                        "value",

                    group_by_roles=[
                        "group",
                        "dimension",
                    ],
                ),
        )
    )


def main() -> None:
    print()
    print("=" * 80)
    print(
        "DATALENS OBJECTIVE COVERAGE TOPOLOGY v0.3"
    )
    print("=" * 80)
    print()


    # ========================================================
    # FOUR MARGINAL REQUIREMENTS
    # ========================================================

    report = (
        build_objective_coverage(
            objective=
                OBJECTIVE,

            catalog=
                catalog(),

            contracts=[
                metric_by(
                    contract_id=
                        "revenue-region",

                    metric=
                        "revenue",

                    reducer=
                        "sum",

                    dimension=
                        "region",
                ),

                metric_by(
                    contract_id=
                        "revenue-channel",

                    metric=
                        "revenue",

                    reducer=
                        "sum",

                    dimension=
                        "channel",
                ),

                metric_by(
                    contract_id=
                        "return-region",

                    metric=
                        "returned_order",

                    reducer=
                        "mean",

                    dimension=
                        "region",
                ),

                metric_by(
                    contract_id=
                        "return-channel",

                    metric=
                        "returned_order",

                    reducer=
                        "mean",

                    dimension=
                        "channel",
                ),
            ],
        )
    )


    assert (
        report.status
        ==
        "complete"
    )

    assert (
        report.topology_requirement_count
        ==
        4
    )

    assert (
        report.topology_covered_count
        ==
        4
    )

    assert (
        report.topology_missing_count
        ==
        0
    )


    print(
        "[PASS] four metric/dimension marginals satisfy the request"
    )


    # ========================================================
    # ONE MARGINAL MISSING
    # ========================================================

    incomplete = (
        build_objective_coverage(
            objective=
                OBJECTIVE,

            catalog=
                catalog(),

            contracts=[
                metric_by(
                    contract_id=
                        "revenue-region",

                    metric=
                        "revenue",

                    reducer=
                        "sum",

                    dimension=
                        "region",
                ),

                metric_by(
                    contract_id=
                        "revenue-channel",

                    metric=
                        "revenue",

                    reducer=
                        "sum",

                    dimension=
                        "channel",
                ),

                metric_by(
                    contract_id=
                        "return-region",

                    metric=
                        "returned_order",

                    reducer=
                        "mean",

                    dimension=
                        "region",
                ),
            ],
        )
    )


    assert (
        incomplete.status
        ==
        "incomplete"
    )

    assert (
        incomplete.topology_missing_count
        ==
        1
    )


    missing = [
        topology.topology_id

        for topology
        in incomplete.topology_requirements

        if not topology.covered
    ]


    assert (
        missing
        ==
        [
            "topology:return_rate:by:channel"
        ]
    )


    print(
        "[PASS] one missing marginal remains fail-closed"
    )


    # ========================================================
    # JOINT CROSS-TAB IS NOT A MARGINAL SUBSTITUTE
    # ========================================================

    joint = (
        build_objective_coverage(
            objective=
                OBJECTIVE,

            catalog=
                catalog(),

            contracts=[
                joint_metric(
                    contract_id=
                        "revenue-joint",

                    metric=
                        "revenue",

                    reducer=
                        "sum",
                ),

                joint_metric(
                    contract_id=
                        "return-joint",

                    metric=
                        "returned_order",

                    reducer=
                        "mean",
                ),
            ],
        )
    )


    assert (
        joint.status
        ==
        "incomplete"
    )

    assert (
        joint.topology_missing_count
        ==
        4
    )


    print(
        "[PASS] region-channel cross-tabs do not replace marginals"
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
        "[PASS] Objective Coverage rule version v0.3"
    )


    print()
    print(
        "PASS - Objective Coverage topology v0.3"
    )


if __name__ == "__main__":
    main()
