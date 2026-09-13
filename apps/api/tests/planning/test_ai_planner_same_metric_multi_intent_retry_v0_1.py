from __future__ import annotations


from app.planning import (
    ai_analytical_planner as planner,
)

from app.planning.ai_analytical_planner import (
    AIPlannerProposal,
    AIPlannerReport,
    AIPlannerValidatedItem,
    build_planner_objective_coverage,
)

from tests.planning.test_objective_coverage_multi_intent_same_metric_v0_1 import (
    OBJECTIVE,
    build_catalog,
    scalar_revenue_contract,
    monthly_revenue_contract,
    category_revenue_contract,
    customer_ranking_contract,
)


def proposal_for_contract(
    contract,
) -> AIPlannerProposal:
    binding_by_role = {
        binding.role:
            binding

        for binding
        in contract.bindings
    }

    aggregation_function = (
        contract.aggregation.function

        if contract.aggregation
        is not None

        else
        "none"
    )

    return AIPlannerProposal(
        decision="propose",
        title=contract.title,
        family=contract.family,
        dataset_id=(
            contract.bindings[
                0
            ].dataset_id
        ),
        analytical_grain=(
            contract.analytical_grain
        ),

        x_column=(
            binding_by_role[
                "x"
            ].column
            if "x" in binding_by_role
            else None
        ),

        y_column=(
            binding_by_role[
                "y"
            ].column
            if "y" in binding_by_role
            else None
        ),

        group_column=(
            binding_by_role[
                "group"
            ].column
            if "group" in binding_by_role
            else None
        ),

        value_column=(
            binding_by_role[
                "value"
            ].column
            if "value" in binding_by_role
            else None
        ),

        time_column=(
            binding_by_role[
                "time"
            ].column
            if "time" in binding_by_role
            else None
        ),

        dimension_column=(
            binding_by_role[
                "dimension"
            ].column
            if "dimension" in binding_by_role
            else None
        ),

        entity_column=(
            binding_by_role[
                "entity"
            ].column
            if "entity" in binding_by_role
            else None
        ),

        aggregation_function=(
            aggregation_function
        ),

        ranking_order=(
            contract.ranking.order
            if contract.ranking
            is not None
            else "none"
        ),

        ranking_limit=(
            contract.ranking.limit
            if contract.ranking
            is not None
            else None
        ),

        window_operation="none",
        window_size=None,

        benchmark_reference=None,
        benchmark_operator=None,
        benchmark_selection=None,

        blockers=[],
        reasons=[],
        confidence=1.0,
    )


def validated_item(
    contract,
    proposal_index: int,
) -> AIPlannerValidatedItem:
    proposal = (
        proposal_for_contract(
            contract
        )
    )

    return AIPlannerValidatedItem(
        proposal_index=
            proposal_index,

        validation_status=
            "validated",

        raw_proposal=
            proposal,

        proposal=
            proposal,

        contract=
            contract,

        errors=[],
        warnings=[],
        normalizations=[],
    )


def report(
    items: list[
        AIPlannerValidatedItem
    ],
) -> AIPlannerReport:
    return AIPlannerReport(
        objective=
            OBJECTIVE,

        model=
            "fake:r10-multi-intent",

        proposal_count=
            len(
                items
            ),

        validated_count=
            len(
                items
            ),

        blocked_count=0,
        ambiguous_count=0,
        rejected_count=0,

        items=
            items,

        attempt_count=1,
        retry_count=0,
        retry_triggered=False,
        retry_feedback=[],
    )


def main() -> None:
    print()
    print("=" * 80)
    print(
        "DATALENS AI PLANNER SAME-METRIC "
        "MULTI-INTENT RETRY v0.1"
    )
    print("=" * 80)
    print()

    catalog = (
        build_catalog()
    )

    first_report = report(
        [
            validated_item(
                scalar_revenue_contract(),
                1,
            )
        ]
    )

    second_report = report(
        [
            validated_item(
                scalar_revenue_contract(),
                1,
            ),

            validated_item(
                monthly_revenue_contract(),
                2,
            ),

            validated_item(
                category_revenue_contract(),
                3,
            ),

            validated_item(
                customer_ranking_contract(),
                4,
            ),
        ]
    )

    reports = [
        first_report,
        second_report,
    ]

    generation_feedback: list[
        list[str]
    ] = []

    validation_attempts: list[
        int
    ] = []


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
            validation_attempts
        )

        if index >= len(
            reports
        ):
            raise AssertionError(
                "Unexpected validation attempt."
            )

        validation_attempts.append(
            attempt_count
        )

        return reports[
            index
        ].model_copy(
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
                    catalog,

                model=
                    "fake:r10-multi-intent",
            )
        )

    finally:
        planner._generate_raw_ai_plan_with_timing = (
            original_generate
        )

        planner.validate_ai_planner_output = (
            original_validate
        )


    # ========================================================
    # TWO ATTEMPTS
    # ========================================================

    assert (
        validation_attempts
        ==
        [
            1,
            2,
        ]
    )

    assert (
        len(
            generation_feedback
        )
        ==
        2
    )

    assert (
        generation_feedback[
            0
        ]
        ==
        []
    )


    # ========================================================
    # SECOND GENERATION RECEIVES ALL MISSING INTENTS
    # ========================================================

    second_feedback = "\n".join(
        generation_feedback[
            1
        ]
    )

    assert (
        "Missing analytical intent=monthly_time_series"
        in
        second_feedback
    )

    assert (
        "required grain=month"
        in
        second_feedback
    )

    assert (
        "Missing analytical intent=categorical_breakdown"
        in
        second_feedback
    )

    assert (
        "required dimension(s)=category"
        in
        second_feedback
    )

    assert (
        "Missing analytical intent=entity_ranking"
        in
        second_feedback
    )

    assert (
        "required dimension(s)=customer"
        in
        second_feedback
    )

    assert (
        "ranking order=descending"
        in
        second_feedback
    )

    assert (
        "ranking limit=10"
        in
        second_feedback
    )


    # ========================================================
    # RETRY METADATA
    # ========================================================

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

    assert (
        result.validated_count
        ==
        4
    )


    # ========================================================
    # FINAL UNION IS COMPLETE
    # ========================================================

    final_coverage = (
        build_planner_objective_coverage(
            objective=
                OBJECTIVE,

            catalog=
                catalog,

            report=
                result,
        )
    )

    assert (
        final_coverage.status
        ==
        "complete"
    )

    assert (
        final_coverage.intent_requirement_count
        ==
        4
    )

    assert (
        final_coverage.intent_covered_count
        ==
        4
    )

    assert (
        final_coverage.intent_missing_count
        ==
        0
    )


    print(
        "[PASS] incomplete first attempt triggers exactly one retry"
    )

    print(
        "[PASS] second generation receives monthly, category "
        "and Top-10 customer intent feedback"
    )

    print(
        "[PASS] second attempt returns four validated contracts"
    )

    print(
        "[PASS] final same-metric multi-intent coverage is 4/4 complete"
    )

    print()
    print(
        "PASS - AI Planner same-metric "
        "multi-intent retry v0.1"
    )


if __name__ == "__main__":
    main()
