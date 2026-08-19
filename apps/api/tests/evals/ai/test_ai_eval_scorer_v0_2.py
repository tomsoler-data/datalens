from __future__ import annotations

from pathlib import Path

from app.evals.benchmark_loader import (
    load_benchmark,
)

from app.evals.schemas import (
    AnalyticalCandidate,
    ToolCallCandidate,
)

from app.evals.scorer_v0_2 import (
    score_candidate_v0_2,
)


BASE_DIR = Path(
    __file__,
).resolve().parents[3]


BENCHMARK_PATH = (
    BASE_DIR
    / "evals"
    / "analytical_reasoning_hard_v0_4.jsonl"
)


def get_case(
    case_id: str,
):
    cases = load_benchmark(
        BENCHMARK_PATH,
        split="validation",
    )

    return next(
        case
        for case
        in cases
        if case.case_id
        == case_id
    )


def test_symmetric_association() -> None:
    case = get_case(
        "hard_v0_4_002"
    )

    candidate = AnalyticalCandidate(
        intent=(
            "measure_relationship"
        ),

        entity=None,

        current_grain=(
            "call_center_day"
        ),

        target_grain=None,

        relevant_columns=[
            "avg_wait_seconds",
            "abandon_rate",
        ],

        family="association",

        tool_calls=[
            ToolCallCandidate(
                name=(
                    "measure_association"
                ),
                arguments={
                    "target":
                        "abandon_rate",

                    "value":
                        "avg_wait_seconds",
                },
            )
        ],

        assumptions=[],
    )

    score = score_candidate_v0_2(
        case,
        candidate,
    )

    assert (
        score.tool_arguments
        == 1.0
    )

    assert (
        score.overall
        == 1.0
    )

    print(
        "Symmetric association: PASS"
    )


def test_entity_plan_consistency() -> None:
    case = get_case(
        "hard_v0_4_001"
    )

    candidate = AnalyticalCandidate(
        intent=(
            "entity_anomaly_analysis"
        ),

        entity=None,

        current_grain="store_day",

        target_grain=None,

        relevant_columns=[
            "store_id",
            "order_count",
            "revenue",
            "return_rate",
        ],

        family="entity_outlier",

        tool_calls=[
            ToolCallCandidate(
                name=(
                    "build_entity_view"
                ),
                arguments={
                    "entity":
                        "store_id",
                },
            ),

            ToolCallCandidate(
                name=(
                    "detect_entity_outliers"
                ),
                arguments={
                    "entity":
                        "store_id",

                    "metrics": [
                        "order_count",
                        "revenue",
                        "return_rate",
                    ],
                },
            ),
        ],

        assumptions=[],
    )

    score = score_candidate_v0_2(
        case,
        candidate,
    )

    assert (
        score.plan_consistency
        < 1.0
    )

    assert (
        "entity_analysis_without_entity"
        in score.consistency_issues
    )

    assert (
        "entity_analysis_without_target_grain"
        in score.consistency_issues
    )

    print(
        "Entity-plan consistency: PASS"
    )


def test_parsimony() -> None:
    case = get_case(
        "hard_v0_4_003"
    )

    candidate = AnalyticalCandidate(
        intent="compare_groups",

        entity=None,

        current_grain=(
            "campaign_day"
        ),

        target_grain=None,

        relevant_columns=[
            "channel",
            "conversion_rate",
        ],

        family=(
            "group_comparison"
        ),

        tool_calls=[
            ToolCallCandidate(
                name="aggregate",
                arguments={
                    "metrics": [
                        "conversion_rate",
                    ],
                    "group_by": [
                        "channel",
                    ],
                },
            ),

            ToolCallCandidate(
                name="compare_groups",
                arguments={
                    "target":
                        "conversion_rate",

                    "group_by":
                        "channel",
                },
            ),
        ],

        assumptions=[],
    )

    score = score_candidate_v0_2(
        case,
        candidate,
    )

    assert (
        score.tool_selection
        < 1.0
    )

    assert (
        score.parsimony
        == 0.5
    )

    assert (
        "aggregate"
        in score.extra_tool_calls
    )

    print(
        "Plan parsimony: PASS"
    )


def test_causal_guardrail() -> None:
    case = get_case(
        "hard_v0_4_004"
    )

    unsafe_candidate = (
        AnalyticalCandidate(
            intent="compare_groups",

            entity=None,

            current_grain="employee",

            target_grain=None,

            relevant_columns=[
                "training_hours",
                "productivity_score",
            ],

            family=(
                "group_comparison"
            ),

            tool_calls=[
                ToolCallCandidate(
                    name=(
                        "compare_groups"
                    ),
                    arguments={
                        "target":
                            "productivity_score",

                        "group_by":
                            "training_hours",
                    },
                )
            ],

            assumptions=[],
        )
    )

    unsafe_score = (
        score_candidate_v0_2(
            case,
            unsafe_candidate,
        )
    )

    assert (
        unsafe_score.guardrails
        == 0.0
    )

    assert (
        "causality_not_established"
        in unsafe_score.failed_guardrails
    )


    safe_candidate = (
        AnalyticalCandidate(
            intent=(
                "measure_relationship"
            ),

            entity=None,

            current_grain="employee",

            target_grain=None,

            relevant_columns=[
                "training_hours",
                "productivity_score",
            ],

            family="association",

            tool_calls=[
                ToolCallCandidate(
                    name=(
                        "measure_association"
                    ),
                    arguments={
                        "target":
                            "training_hours",

                        "value":
                            "productivity_score",
                    },
                )
            ],

            assumptions=[],
        )
    )

    safe_score = (
        score_candidate_v0_2(
            case,
            safe_candidate,
        )
    )

    assert (
        safe_score.guardrails
        == 1.0
    )

    assert (
        "causality_not_established"
        in safe_score.passed_guardrails
    )

    print(
        "Causal guardrail: PASS"
    )


def test_perfect_candidate() -> None:
    case = get_case(
        "hard_v0_4_006"
    )

    candidate = AnalyticalCandidate(
        intent="compare_groups",

        entity=None,

        current_grain="ticket",

        target_grain=None,

        relevant_columns=[
            "priority",
            "resolution_minutes",
        ],

        family=(
            "group_comparison"
        ),

        tool_calls=[
            ToolCallCandidate(
                name="compare_groups",

                arguments={
                    "target":
                        "resolution_minutes",

                    "group_by":
                        "priority",
                },
            )
        ],

        assumptions=[],
    )

    score = score_candidate_v0_2(
        case,
        candidate,
    )

    assert (
        score.overall
        == 1.0
    )

    assert (
        score.comprehension
        == 1.0
    )

    assert (
        score.planning
        == 1.0
    )

    assert (
        score.reliability
        == 1.0
    )

    print(
        "Perfect candidate: PASS"
    )


def main() -> None:
    print(
        "=== DATALENS AI SCORER v0.2 ==="
    )

    print()

    test_symmetric_association()

    test_entity_plan_consistency()

    test_parsimony()

    test_causal_guardrail()

    test_perfect_candidate()

    print()

    print(
        "AI Eval scorer v0.2: PASS"
    )


if __name__ == "__main__":
    main()