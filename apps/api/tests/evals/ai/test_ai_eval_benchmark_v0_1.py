from __future__ import annotations

from pathlib import Path

from app.evals import (
    load_benchmark,
)


BASE_DIR = Path(
    __file__,
).resolve().parents[3]


BENCHMARK_PATH = (
    BASE_DIR
    / "evals"
    / "analytical_reasoning_v1.jsonl"
)


def main() -> None:
    all_cases = load_benchmark(
        BENCHMARK_PATH,
    )

    train_cases = load_benchmark(
        BENCHMARK_PATH,
        split="train",
    )

    validation_cases = load_benchmark(
        BENCHMARK_PATH,
        split="validation",
    )

    test_cases = load_benchmark(
        BENCHMARK_PATH,
        split="test",
    )


    print(
        "=== DATALENS AI EVAL v0.1 ==="
    )

    print(
        "Total:",
        len(
            all_cases,
        ),
    )

    print(
        "Train:",
        len(
            train_cases,
        ),
    )

    print(
        "Validation:",
        len(
            validation_cases,
        ),
    )

    print(
        "Test:",
        len(
            test_cases,
        ),
    )


    assert len(
        all_cases,
    ) == 6

    assert len(
        train_cases,
    ) == 3

    assert len(
        validation_cases,
    ) == 2

    assert len(
        test_cases,
    ) == 1


    case_ids = {
        case.case_id
        for case in all_cases
    }

    assert len(
        case_ids,
    ) == len(
        all_cases,
    )


    assert all(
        case.frozen
        for case in test_cases
    )


    domains = {
        case.domain
        for case in all_cases
    }

    assert len(
        domains,
    ) == 6


    reasoning_cases = [
        case
        for case in all_cases
        if case.expected.requires_reasoning
    ]

    simple_cases = [
        case
        for case in all_cases
        if not case.expected.requires_reasoning
    ]


    print(
        "Reasoning:",
        len(
            reasoning_cases,
        ),
    )

    print(
        "Simple:",
        len(
            simple_cases,
        ),
    )


    print()
    print(
        "Frozen test set:"
    )

    for case in test_cases:
        print(
            "-",
            case.case_id,
            "|",
            case.domain,
            "|",
            case.user_request,
        )


    print()
    print(
        "AI Eval benchmark: PASS"
    )


if __name__ == "__main__":
    main()