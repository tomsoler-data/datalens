from __future__ import annotations


from app.planning.ai_analytical_planner import (
    objective_coverage_retry_feedback,
)

from app.planning.objective_coverage import (
    build_objective_coverage,
)

from tests.planning.test_objective_coverage_multi_intent_same_metric_v0_1 import (
    OBJECTIVE,
    build_catalog,
    scalar_revenue_contract,
)


def main() -> None:
    print()
    print("=" * 80)
    print(
        "DATALENS AI PLANNER OBJECTIVE INTENT "
        "RETRY FEEDBACK v0.1"
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

    assert (
        report.intent_requirement_count
        ==
        4
    )

    assert (
        report.intent_missing_count
        ==
        3
    )

    feedback = (
        objective_coverage_retry_feedback(
            report
        )
    )

    feedback_text = "\n".join(
        feedback
    )

    print(
        feedback_text
    )

    print()

    # ========================================================
    # MONTHLY TIME SERIES
    # ========================================================

    assert (
        "Missing analytical intent=monthly_time_series"
        in
        feedback_text
    ), (
        "Retry feedback must explicitly tell the planner that "
        "the monthly time-series intent is still missing."
    )

    assert (
        "required family=time_series"
        in
        feedback_text
    )

    assert (
        "required grain=month"
        in
        feedback_text
    )


    # ========================================================
    # CATEGORY BREAKDOWN
    # ========================================================

    assert (
        "Missing analytical intent=categorical_breakdown"
        in
        feedback_text
    ), (
        "Retry feedback must explicitly preserve the requested "
        "category breakdown."
    )

    assert (
        "required dimension(s)=category"
        in
        feedback_text
    )


    # ========================================================
    # ENTITY RANKING
    # ========================================================

    assert (
        "Missing analytical intent=entity_ranking"
        in
        feedback_text
    ), (
        "Retry feedback must explicitly preserve the requested "
        "customer ranking."
    )

    assert (
        "required family=ranking"
        in
        feedback_text
    )

    assert (
        "required dimension(s)=customer"
        in
        feedback_text
    )

    assert (
        "ranking order=descending"
        in
        feedback_text
    )

    assert (
        "ranking limit=10"
        in
        feedback_text
    )


    # ========================================================
    # GENERIC MULTI-INTENT INSTRUCTION
    # ========================================================

    assert (
        "multiple analytical intents"
        in
        feedback_text
    ), (
        "The retry instruction must allow multiple proposals "
        "for multiple analytical intents, even when they share "
        "the same metric."
    )


    print(
        "[PASS] missing monthly intent reaches retry feedback"
    )

    print(
        "[PASS] missing category intent reaches retry feedback"
    )

    print(
        "[PASS] missing Top-10 customer ranking reaches "
        "retry feedback"
    )

    print(
        "[PASS] retry supports multiple same-metric "
        "analytical intents"
    )

    print()
    print(
        "PASS - AI Planner objective intent "
        "retry feedback v0.1"
    )


if __name__ == "__main__":
    main()
