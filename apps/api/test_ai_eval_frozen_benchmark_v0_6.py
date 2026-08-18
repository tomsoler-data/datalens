from __future__ import annotations

from pathlib import Path

from app.evals.decision_benchmark_v0_6 import (
    DECISION_BENCHMARK_SCHEMA_VERSION,
    load_decision_benchmark,
)


BASE_DIR = Path(
    __file__,
).resolve().parent


BENCHMARK_PATH = (
    BASE_DIR
    / "evals"
    / "analytical_decision_frozen_v0_6.jsonl"
)


def main() -> None:
    print(
        "=== DATALENS FROZEN BENCHMARK CONTRACT v0.6 ==="
    )

    print(
        "Schema:",
        DECISION_BENCHMARK_SCHEMA_VERSION,
    )

    print()


    cases = load_decision_benchmark(
        BENCHMARK_PATH,
    )


    # ========================================================
    # GLOBAL CONTRACT
    # ========================================================

    assert (
        len(
            cases
        )
        == 15
    )


    assert (
        len(
            {
                case.case_id
                for case
                in cases
            }
        )
        == 15
    )


    assert all(
        case.split
        == "test"
        for case
        in cases
    )


    assert all(
        case.frozen
        for case
        in cases
    )


    # ========================================================
    # DECISION DISTRIBUTION
    # ========================================================

    analyze = [
        case
        for case
        in cases
        if (
            case.expected.decision
            == "analyze"
        )
    ]


    clarification = [
        case
        for case
        in cases
        if (
            case.expected.decision
            == "needs_clarification"
        )
    ]


    cannot_answer = [
        case
        for case
        in cases
        if (
            case.expected.decision
            == "cannot_answer"
        )
    ]


    assert len(
        analyze
    ) == 7


    assert len(
        clarification
    ) == 3


    assert len(
        cannot_answer
    ) == 5


    # ========================================================
    # ANALYZE CONTRACT
    # ========================================================

    assert all(
        (
            case
            .expected
            .analytical
            is not None
        )
        for case
        in analyze
    )


    assert all(
        (
            case
            .expected
            .decision_reason
            is None
        )
        for case
        in analyze
    )


    # ========================================================
    # CLARIFICATION CONTRACT
    # ========================================================

    assert all(
        (
            case
            .expected
            .analytical
            is None
        )
        for case
        in clarification
    )


    assert all(
        bool(
            case
            .expected
            .clarification_topics
        )
        for case
        in clarification
    )


    # ========================================================
    # CANNOT ANSWER CONTRACT
    # ========================================================

    assert all(
        (
            case
            .expected
            .analytical
            is None
        )
        for case
        in cannot_answer
    )


    assert all(
        (
            case
            .expected
            .decision_reason
            is not None
        )
        for case
        in cannot_answer
    )


    # ========================================================
    # SPECIFIC SAFETY / ABSTENTION CASES
    # ========================================================

    by_id = {
        case.case_id:
            case
        for case
        in cases
    }


    causal = (
        by_id[
            "frozen_v0_6_008"
        ]
    )


    assert (
        causal.expected.decision
        == "cannot_answer"
    )


    assert (
        causal
        .expected
        .decision_reason
        == "causal_identification_missing"
    )


    ambiguity = (
        by_id[
            "frozen_v0_6_009"
        ]
    )


    assert (
        ambiguity.expected.decision
        == "needs_clarification"
    )


    missing_column = (
        by_id[
            "frozen_v0_6_011"
        ]
    )


    assert (
        missing_column
        .expected
        .decision_reason
        == "missing_column"
    )


    unsupported_forecast = (
        by_id[
            "frozen_v0_6_013"
        ]
    )


    assert (
        unsupported_forecast
        .expected
        .decision_reason
        == "unsupported_analysis"
    )


    multi_dataset = (
        by_id[
            "frozen_v0_6_014"
        ]
    )


    assert (
        len(
            multi_dataset.datasets
        )
        == 2
    )


    assert (
        multi_dataset
        .expected
        .decision
        == "cannot_answer"
    )


    # ========================================================
    # SUMMARY
    # ========================================================

    domains = {
        case.domain
        for case
        in cases
    }


    print(
        "Cases:",
        len(
            cases
        ),
    )

    print(
        "Domains:",
        len(
            domains
        ),
    )

    print()

    print(
        "Analyze:",
        len(
            analyze
        ),
    )

    print(
        "Needs clarification:",
        len(
            clarification
        ),
    )

    print(
        "Cannot answer:",
        len(
            cannot_answer
        ),
    )

    print()

    print(
        "Frozen:",
        all(
            case.frozen
            for case
            in cases
        ),
    )

    print()


    for case in cases:
        print(
            "-",
            case.case_id,
            "|",
            case.domain,
            "|",
            case.expected.decision,
        )


    print()

    print(
        "Frozen benchmark contract: PASS"
    )


if __name__ == "__main__":
    main()