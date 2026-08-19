from __future__ import annotations

from app.evals.suite_runner import (
    run_core_suite,
    summarize_reports,
)


def test_eval_suite_v0_1_core_contract(
) -> None:
    reports = (
        run_core_suite(
            deterministic_runs=
                2,
        )
    )


    summary = (
        summarize_reports(
            reports
        )
    )


    assert len(
        reports
    ) == 3

    assert (
        summary.scenario_count
        ==
        3
    )

    assert (
        summary.passed_scenario_count
        ==
        3
    )

    assert (
        summary.failed_scenario_count
        ==
        0
    )

    assert (
        summary.expectation_count
        ==
        4
    )

    assert (
        summary.passed_expectation_count
        ==
        4
    )

    assert (
        summary.expectation_accuracy
        ==
        1.0
    )

    assert (
        summary.discovery_recall
        ==
        1.0
    )

    assert (
        summary.selection_recall
        ==
        1.0
    )

    assert (
        summary.guardrail_success_rate
        ==
        1.0
    )

    assert (
        summary.determinism_rate
        ==
        1.0
    )


    print(
        "Original three-scenario Eval Suite v0.1 core remains stable: PASS"
    )


def main() -> None:
    print(
        "=== DATALENS EVAL SUITE CORE REGRESSION v0.1 ==="
    )

    print()


    test_eval_suite_v0_1_core_contract()


    print()

    print(
        "Eval Suite Core Regression v0.1: PASS"
    )


if __name__ == "__main__":
    main()
