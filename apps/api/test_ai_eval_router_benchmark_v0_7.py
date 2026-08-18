from __future__ import annotations

from pathlib import Path

from app.evals.decision_router_benchmark_v0_7 import (
    DECISION_ROUTER_BENCHMARK_VERSION,
    load_decision_router_benchmark,
)


# ============================================================
# PATH
# ============================================================

BASE_DIR = Path(
    __file__,
).resolve().parent


BENCHMARK_PATH = (
    BASE_DIR
    / "evals"
    / "decision_router_development_v0_7.jsonl"
)


# ============================================================
# HELPERS
# ============================================================

def count_decision(
    cases,
    decision: str,
) -> int:
    return sum(
        1
        for case
        in cases
        if (
            case.expected.decision
            == decision
        )
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print(
        "=== DATALENS DECISION ROUTER BENCHMARK CONTRACT v0.7 ==="
    )

    print(
        "Benchmark:",
        DECISION_ROUTER_BENCHMARK_VERSION,
    )

    print()


    all_cases = (
        load_decision_router_benchmark(
            BENCHMARK_PATH,
        )
    )


    train_cases = (
        load_decision_router_benchmark(
            BENCHMARK_PATH,
            split="train",
        )
    )


    validation_cases = (
        load_decision_router_benchmark(
            BENCHMARK_PATH,
            split="validation",
        )
    )


    # ========================================================
    # GLOBAL
    # ========================================================

    assert (
        len(
            all_cases
        )
        == 18
    )


    assert (
        len(
            train_cases
        )
        == 9
    )


    assert (
        len(
            validation_cases
        )
        == 9
    )


    assert (
        len(
            {
                case.case_id
                for case
                in all_cases
            }
        )
        == 18
    )


    assert all(
        not case.frozen
        for case
        in all_cases
    )


    # ========================================================
    # BALANCE
    # ========================================================

    for cases in (
        train_cases,
        validation_cases,
    ):
        assert (
            count_decision(
                cases,
                "analyze",
            )
            == 3
        )

        assert (
            count_decision(
                cases,
                "needs_clarification",
            )
            == 3
        )

        assert (
            count_decision(
                cases,
                "cannot_answer",
            )
            == 3
        )


    # ========================================================
    # EXPECTATION CONTRACT
    # ========================================================

    for case in all_cases:
        expected = (
            case.expected
        )


        if (
            expected.decision
            == "analyze"
        ):
            assert (
                expected.decision_reason
                is None
            )

            assert (
                expected.clarification_topics
                == []
            )


        elif (
            expected.decision
            == "needs_clarification"
        ):
            assert (
                expected.decision_reason
                in {
                    "ambiguous_request",
                    "insufficient_context",
                }
            )

            assert (
                expected.clarification_topics
            )


        elif (
            expected.decision
            == "cannot_answer"
        ):
            assert (
                expected.decision_reason
                in {
                    "missing_column",
                    "missing_dataset",
                    "unsupported_analysis",
                    "causal_identification_missing",
                }
            )

            assert (
                expected.clarification_topics
                == []
            )


        else:
            raise AssertionError(
                "Unexpected router decision."
            )


    # ========================================================
    # IMPORTANT DEVELOPMENT CASES
    # ========================================================

    by_id = {
        case.case_id:
            case
        for case
        in all_cases
    }


    # Explicit causal request.
    causal = (
        by_id[
            "router_v0_7_validation_008"
        ]
    )


    assert (
        causal.expected.decision
        == "cannot_answer"
    )

    assert (
        causal.expected.decision_reason
        == "causal_identification_missing"
    )


    # Multi-dataset operation requiring unsupported join.
    multi_dataset = (
        by_id[
            "router_v0_7_validation_009"
        ]
    )


    assert (
        len(
            multi_dataset.datasets
        )
        == 2
    )

    assert (
        multi_dataset.expected.decision
        == "cannot_answer"
    )

    assert (
        multi_dataset.expected.decision_reason
        == "unsupported_analysis"
    )


    # Ambiguous performance definition.
    ambiguous = (
        by_id[
            "router_v0_7_validation_004"
        ]
    )


    assert (
        ambiguous.expected.decision
        == "needs_clarification"
    )

    assert (
        "performance_metric"
        in ambiguous
        .expected
        .clarification_topics
    )


    # Missing reference threshold.
    threshold = (
        by_id[
            "router_v0_7_validation_005"
        ]
    )


    assert (
        threshold.expected.decision
        == "needs_clarification"
    )

    assert (
        threshold.expected.decision_reason
        == "insufficient_context"
    )


    # ========================================================
    # DOMAINS
    # ========================================================

    domains = {
        case.domain
        for case
        in all_cases
    }


    assert (
        len(
            domains
        )
        == 18
    )


    # ========================================================
    # DISPLAY
    # ========================================================

    print(
        "Total:",
        len(
            all_cases
        ),
    )

    print(
        "Domains:",
        len(
            domains
        ),
    )

    print(
        "Frozen:",
        False,
    )

    print()


    for (
        split_name,
        cases,
    ) in (
        (
            "TRAIN",
            train_cases,
        ),
        (
            "VALIDATION",
            validation_cases,
        ),
    ):
        print(
            split_name,
        )

        print(
            "  Cases:",
            len(
                cases
            ),
        )

        print(
            "  Analyze:",
            count_decision(
                cases,
                "analyze",
            ),
        )

        print(
            "  Clarification:",
            count_decision(
                cases,
                "needs_clarification",
            ),
        )

        print(
            "  Cannot answer:",
            count_decision(
                cases,
                "cannot_answer",
            ),
        )

        print()


    for case in all_cases:
        print(
            "-",
            case.case_id,
            "|",
            case.split,
            "|",
            case.domain,
            "|",
            case.expected.decision,
        )


    print()

    print(
        "Decision Router benchmark contract v0.7: PASS"
    )


if __name__ == "__main__":
    main()