# DATALENS_OBJECTIVE_COVERAGE_V0_3_REGRESSION_ALIGNMENT
from __future__ import annotations


from types import (
    SimpleNamespace,
)


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
    ObjectiveCoverageIncompleteError,
    require_objective_coverage,
    validated_contracts_from_planner_report,
)


OBJECTIVE = (
    "Compare le chiffre d'affaires et le taux de retour "
    "par region et par canal."
)


def build_catalog() -> PlannerCatalog:
    column_names = (
        "region",
        "channel",
        "product_category",
        "revenue",
        "returned_order",
    )

    columns = [
        PlannerColumnProfile(
            name=
                name,

            dtype=(
                "float64"

                if name
                ==
                "revenue"

                else
                (
                    "bool"

                    if name
                    ==
                    "returned_order"

                    else
                    "object"
                )
            ),

            analysis_kind=(
                "quantitative"

                if name
                ==
                "revenue"

                else
                "categorical"
            ),

            missing_ratio=
                0.0,

            unique_count=
                10,
        )

        for name
        in column_names
    ]


    return (
        PlannerCatalog(
            datasets=[
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


def bad_contract() -> AnalyticalContract:
    return (
        AnalyticalContract(
            contract_id=
                "contract:bad",

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



def revenue_contract(
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
                        "group"
                    ],
                ),
        )
    )


def return_rate_contract(
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
                        "group"
                    ],
                ),
        )
    )

def planner_report(
    *contracts: AnalyticalContract,
):
    return (
        SimpleNamespace(
            items=[
                SimpleNamespace(
                    validation_status=
                        "validated",

                    contract=
                        contract,
                )

                for contract
                in contracts
            ]
        )
    )


def main() -> None:
    catalog = (
        build_catalog()
    )


    print()
    print("=" * 80)
    print(
        "DATALENS OBJECTIVE COVERAGE FAIL-CLOSED GUARD v0.1"
    )
    print("=" * 80)
    print()


    # ========================================================
    # BAD PLAN MUST FAIL CLOSED
    # ========================================================

    bad_report = (
        planner_report(
            bad_contract()
        )
    )


    try:
        require_objective_coverage(
            objective=
                OBJECTIVE,

            catalog=
                catalog,

            planner_report=
                bad_report,
        )

    except ObjectiveCoverageIncompleteError as error:
        missing = {
            requirement.concept

            for requirement
            in error.report.requirements

            if not requirement.covered
        }

        assert (
            missing
            ==
            {
                "revenue_total",
                "return_rate",
                "channel",
            }
        )

    else:
        raise AssertionError(
            "Incomplete objective coverage did not fail closed."
        )


    print(
        "[PASS] incomplete validated plan fails closed"
    )


    # ========================================================
    # COMPLETE UNION MAY PASS
    # ========================================================

    complete = (
        require_objective_coverage(
            objective=
                OBJECTIVE,

            catalog=
                catalog,

            planner_report=
                planner_report(
                    revenue_contract(
                        "region"
                    ),

                    revenue_contract(
                        "channel"
                    ),

                    return_rate_contract(
                        "region"
                    ),

                    return_rate_contract(
                        "channel"
                    ),
                ),
        )
    )


    assert (
        complete.status
        ==
        "complete"
    )

    assert (
        complete.covered_count
        ==
        4
    )


    print(
        "[PASS] complete multi-contract plan may continue"
    )


    # ========================================================
    # REJECTED / BLOCKED ITEMS DO NOT COUNT
    # ========================================================

    mixed_report = (
        SimpleNamespace(
            items=[
                SimpleNamespace(
                    validation_status=
                        "rejected",

                    contract=
                        revenue_contract("region"),
                ),

                SimpleNamespace(
                    validation_status=
                        "blocked",

                    contract=
                        return_rate_contract("region"),
                ),

                SimpleNamespace(
                    validation_status=
                        "validated",

                    contract=
                        bad_contract(),
                ),
            ]
        )
    )


    extracted = (
        validated_contracts_from_planner_report(
            mixed_report
        )
    )


    assert (
        len(
            extracted
        )
        ==
        1
    )

    assert (
        extracted[
            0
        ].contract_id
        ==
        "contract:bad"
    )


    print(
        "[PASS] only Python-validated contracts contribute"
    )


    # ========================================================
    # NOT APPLICABLE REMAINS NON-BLOCKING
    # ========================================================

    neutral = (
        require_objective_coverage(
            objective=
                "Detecte les valeurs atypiques.",

            catalog=
                catalog,

            planner_report=
                SimpleNamespace(
                    items=[]
                ),
        )
    )


    assert (
        neutral.status
        ==
        "not_applicable"
    )


    print(
        "[PASS] conservative abstention remains non-blocking"
    )


    print()
    print(
        "PASS - Objective Coverage fail-closed guard v0.1"
    )


if __name__ == "__main__":
    main()
