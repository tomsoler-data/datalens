# DATALENS_OBJECTIVE_COVERAGE_V0_3_REGRESSION_ALIGNMENT
from __future__ import annotations


from app.planning import (
    ai_analytical_planner as planner,
)

from app.planning.ai_analytical_planner import (
    AI_ANALYTICAL_PLANNER_RULE_VERSION,
    AIPlannerProposal,
    AIPlannerReport,
    AIPlannerValidatedItem,
    PlannerCatalog,
    PlannerColumnProfile,
    PlannerDatasetProfile,
)

from app.planning.analytical_contract import (
    AggregationSpec,
    AnalyticalContract,
    VariableBinding,
)


OBJECTIVE = (
    "Compare le chiffre d'affaires et le taux de retour "
    "par region et par canal."
)


def build_catalog() -> PlannerCatalog:
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
        in (
            "region",
            "channel",
            "product_category",
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


def proposal(
    *,
    title: str,
    family: str,
    x_column: str | None = None,
    y_column: str | None = None,
    group_column: str | None = None,
    value_column: str | None = None,
    dimension_column: str | None = None,
    aggregation_function: str = "none",
) -> AIPlannerProposal:
    return (
        AIPlannerProposal(
            decision=
                "propose",

            title=
                title,

            family=
                family,

            dataset_id=
                "dataset:demo",

            analytical_grain=
                None,

            x_column=
                x_column,

            y_column=
                y_column,

            group_column=
                group_column,

            value_column=
                value_column,

            time_column=
                None,

            dimension_column=
                dimension_column,

            entity_column=
                None,

            aggregation_function=
                aggregation_function,

            ranking_order=
                "none",

            ranking_limit=
                None,

            window_operation=
                "none",

            window_size=
                None,

            benchmark_reference=None,
            benchmark_operator=None,
            benchmark_selection=None,
            blockers=[],

            reasons=[],

            confidence=
                1.0,
        )
    )


def bad_item() -> AIPlannerValidatedItem:
    p = (
        proposal(
            title=
                "Region by product category",

            family=
                "categorical_association",

            x_column=
                "region",

            y_column=
                "product_category",
        )
    )


    contract = (
        AnalyticalContract(
            contract_id=
                "contract:bad",

            origin=
                "ai_planner",

            status=
                "validated",

            title=
                p.title,

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


    return (
        AIPlannerValidatedItem(
            proposal_index=
                1,

            validation_status=
                "validated",

            raw_proposal=
                p,

            proposal=
                p,

            contract=
                contract,

            errors=[],

            warnings=[],

            normalizations=[],
        )
    )



def revenue_item(
    dimension: str,
    proposal_index: int,
) -> AIPlannerValidatedItem:
    if dimension not in {
        "region",
        "channel",
    }:
        raise ValueError(
            "Unsupported demo dimension."
        )


    p = (
        proposal(
            title=(
                "Revenue by "
                +
                dimension
            ),

            family=
                "aggregation",

            group_column=
                dimension,

            value_column=
                "revenue",

            aggregation_function=
                "sum",
        )
    )


    contract = (
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

            title=
                p.title,

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


    return (
        AIPlannerValidatedItem(
            proposal_index=
                proposal_index,

            validation_status=
                "validated",

            raw_proposal=
                p,

            proposal=
                p,

            contract=
                contract,

            errors=[],

            warnings=[],

            normalizations=[],
        )
    )


def return_rate_item(
    dimension: str,
    proposal_index: int,
) -> AIPlannerValidatedItem:
    if dimension not in {
        "region",
        "channel",
    }:
        raise ValueError(
            "Unsupported demo dimension."
        )


    p = (
        proposal(
            title=(
                "Return rate by "
                +
                dimension
            ),

            family=
                "aggregation",

            group_column=
                dimension,

            value_column=
                "returned_order",

            aggregation_function=
                "mean",
        )
    )


    contract = (
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

            title=
                p.title,

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


    return (
        AIPlannerValidatedItem(
            proposal_index=
                proposal_index,

            validation_status=
                "validated",

            raw_proposal=
                p,

            proposal=
                p,

            contract=
                contract,

            errors=[],

            warnings=[],

            normalizations=[],
        )
    )

def report(
    *,
    items: list[
        AIPlannerValidatedItem
    ],

    attempt_count: int = 1,

    retry_count: int = 0,

    retry_triggered: bool = False,

    retry_feedback: list[
        str
    ] | None = None,
) -> AIPlannerReport:
    return (
        AIPlannerReport(
            objective=
                OBJECTIVE,

            model=
                "fake:planner",

            proposal_count=
                len(
                    items
                ),

            validated_count=sum(
                1

                for item
                in items

                if (
                    item.validation_status
                    ==
                    "validated"
                )
            ),

            blocked_count=sum(
                1

                for item
                in items

                if (
                    item.validation_status
                    ==
                    "blocked"
                )
            ),

            ambiguous_count=sum(
                1

                for item
                in items

                if (
                    item.validation_status
                    ==
                    "ambiguous"
                )
            ),

            rejected_count=sum(
                1

                for item
                in items

                if (
                    item.validation_status
                    ==
                    "rejected"
                )
            ),

            items=
                items,

            attempt_count=
                attempt_count,

            retry_count=
                retry_count,

            retry_triggered=
                retry_triggered,

            retry_feedback=(
                retry_feedback
                or []
            ),
        )
    )


def run_fake_planner(
    reports: list[
        AIPlannerReport
    ],
):
    generation_feedback: list[
        list[
            str
        ]
    ] = []

    validation_calls = []


    original_generate = (
        planner
        ._generate_raw_ai_plan_with_timing
    )

    original_validate = (
        planner
        .validate_ai_planner_output
    )


    def fake_generate(
        *,
        objective,
        catalog,
        model,
        validation_feedback=None,
    ):
        _ = (
            objective,
            catalog,
            model,
        )

        generation_feedback.append(
            list(
                validation_feedback
                or []
            )
        )

        return (
            object(),
            1.0,
            2.0,
            3.0,
        )


    def fake_validate(
        *,
        objective,
        raw_output,
        catalog,
        model,
        attempt_count,
        retry_count,
        retry_triggered,
        retry_feedback,
    ):
        _ = (
            objective,
            raw_output,
            catalog,
            model,
        )

        index = len(
            validation_calls
        )

        if (
            index
            >=
            len(
                reports
            )
        ):
            raise AssertionError(
                "Unexpected validation attempt."
            )


        validation_calls.append(
            attempt_count
        )


        return (
            reports[
                index
            ]
            .model_copy(
                update={
                    "attempt_count":
                        attempt_count,

                    "retry_count":
                        retry_count,

                    "retry_triggered":
                        retry_triggered,

                    "retry_feedback":
                        list(
                            retry_feedback
                        ),
                }
            )
        )


    planner._generate_raw_ai_plan_with_timing = (
        fake_generate
    )

    planner.validate_ai_planner_output = (
        fake_validate
    )


    try:
        result = (
            planner.plan_analyses_with_ai(
                objective=
                    OBJECTIVE,

                catalog=
                    build_catalog(),

                model=
                    "fake:planner",
            )
        )

    finally:
        planner._generate_raw_ai_plan_with_timing = (
            original_generate
        )

        planner.validate_ai_planner_output = (
            original_validate
        )


    return (
        result,
        generation_feedback,
        validation_calls,
    )


def main() -> None:
    print()
    print("=" * 80)
    print(
        "DATALENS AI PLANNER OBJECTIVE COVERAGE RETRY v0.1"
    )
    print("=" * 80)
    print()


    # ========================================================
    # CASE 1 - VALID BUT INCOMPLETE MUST RETRY
    # ========================================================

    (
        result,
        feedback_calls,
        validation_calls,
    ) = run_fake_planner(
        [
            report(
                items=[
                    bad_item()
                ]
            ),

            report(
                items=[
                    revenue_item(
                        "region",
                        1,
                    ),

                    revenue_item(
                        "channel",
                        2,
                    ),

                    return_rate_item(
                        "region",
                        3,
                    ),

                    return_rate_item(
                        "channel",
                        4,
                    ),
                ]
            ),
        ]
    )


    assert (
        validation_calls
        ==
        [
            1,
            2,
        ]
    )


    assert (
        len(
            feedback_calls
        )
        ==
        2
    )


    assert (
        feedback_calls[
            0
        ]
        ==
        []
    )


    second_feedback = (
        "\n".join(
            feedback_calls[
                1
            ]
        )
    )


    assert (
        "OBJECTIVE COVERAGE INCOMPLETE"
        in
        second_feedback
    )

    assert (
        "revenue_total"
        in
        second_feedback
    )

    assert (
        "return_rate"
        in
        second_feedback
    )

    assert (
        "channel"
        in
        second_feedback
    )


    assert (
        result.attempt_count
        ==
        2
    )

    assert (
        result.retry_count
        ==
        1
    )

    assert (
        result.retry_triggered
        is True
    )


    print(
        "[PASS] valid but incomplete plan triggers attempt 2"
    )

    print(
        "[PASS] missing objective concepts reach retry feedback"
    )


    # ========================================================
    # CASE 2 - COMPLETE FIRST ATTEMPT MUST NOT RETRY
    # ========================================================

    (
        complete_result,
        complete_feedback_calls,
        complete_validation_calls,
    ) = run_fake_planner(
        [
            report(
                items=[
                    revenue_item(
                        "region",
                        1,
                    ),

                    revenue_item(
                        "channel",
                        2,
                    ),

                    return_rate_item(
                        "region",
                        3,
                    ),

                    return_rate_item(
                        "channel",
                        4,
                    ),
                ]
            )
        ]
    )


    assert (
        complete_validation_calls
        ==
        [
            1
        ]
    )

    assert (
        len(
            complete_feedback_calls
        )
        ==
        1
    )

    assert (
        complete_result.attempt_count
        ==
        1
    )

    assert (
        complete_result.retry_count
        ==
        0
    )

    assert (
        complete_result.retry_triggered
        is False
    )


    print(
        "[PASS] complete first attempt does not retry"
    )


    # ========================================================
    # VERSION
    # ========================================================

    assert (
        AI_ANALYTICAL_PLANNER_RULE_VERSION
        ==
        "ai_analytical_planner_v0.34"
    )


    print(
        "[PASS] AI Planner rule version v0.34"
    )


    print()
    print(
        "PASS - AI Planner Objective Coverage retry v0.1"
    )


if __name__ == "__main__":
    main()
