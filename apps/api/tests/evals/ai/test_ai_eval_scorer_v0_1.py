from __future__ import annotations

from pathlib import Path

from app.evals import (
    AnalyticalCandidate,
    ToolCallCandidate,
    load_benchmark,
    score_candidate,
)


BASE_DIR = Path(
    __file__,
).resolve().parents[3]


BENCHMARK_PATH = (
    BASE_DIR
    / "evals"
    / "analytical_reasoning_v1.jsonl"
)


def test_perfect_candidate() -> None:
    case = load_benchmark(
        BENCHMARK_PATH,
        split="validation",
    )[0]

    expected = case.expected

    candidate = AnalyticalCandidate(
        intent=expected.intent,
        entity=expected.entity,
        current_grain=expected.current_grain,
        target_grain=expected.target_grain,
        relevant_columns=expected.relevant_columns,
        family=expected.family,
        tool_calls=[
            ToolCallCandidate(
                name=tool_name,
                arguments=(
                    expected
                    .required_tool_arguments
                    .get(
                        tool_name,
                        {},
                    )
                ),
            )
            for tool_name
            in expected.acceptable_tools
        ],
        assumptions=[],
    )

    score = score_candidate(
        case,
        candidate,
    )

    assert score.intent == 1.0
    assert score.entity == 1.0
    assert score.grain == 1.0
    assert score.relevant_columns == 1.0
    assert score.family == 1.0
    assert score.tool_selection == 1.0
    assert score.tool_arguments == 1.0
    assert score.safety == 1.0
    assert score.overall == 1.0

    print(
        "Perfect candidate:",
        score.overall,
    )


def test_hallucinated_candidate() -> None:
    case = load_benchmark(
        BENCHMARK_PATH,
        split="test",
    )[0]

    candidate = AnalyticalCandidate(
        intent="entity_anomaly_analysis",
        entity="airport_id",
        current_grain="flight",
        target_grain="airport",
        relevant_columns=[
            "airport_id",
            "departure_delay_min",
            "imaginary_risk_score",
        ],
        family="entity_outlier",
        tool_calls=[
            ToolCallCandidate(
                name="invented_magic_tool",
                arguments={
                    "target":
                        "imaginary_risk_score",
                },
            )
        ],
        assumptions=[
            "fraud",
        ],
    )

    score = score_candidate(
        case,
        candidate,
    )

    assert (
        "imaginary_risk_score"
        in score.invented_columns
    )

    assert (
        "invented_magic_tool"
        in score.invented_tools
    )

    assert (
        "fraud"
        in score.forbidden_assumptions_used
    )

    assert score.safety < 1.0
    assert score.overall < 1.0

    print()
    print(
        "Hallucinated candidate:"
    )

    print(
        score.as_dict()
    )


def main() -> None:
    print(
        "=== DATALENS AI SCORER v0.1 ==="
    )

    print()

    test_perfect_candidate()

    test_hallucinated_candidate()

    print()
    print(
        "AI Eval scorer: PASS"
    )


if __name__ == "__main__":
    main()