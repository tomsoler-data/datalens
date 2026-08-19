from __future__ import annotations

from app.evals.scenarios import (
    EVAL_COVERAGE_RULE_VERSION,
)

from app.evals.suite_runner import (
    EVAL_SUITE_RULE_VERSION,
    PRIORITIZATION_REASON_CODE_TARGETS,
    run_guardrail_coverage_suite,
    run_suite,
)


def test_guardrail_coverage_scenarios_pass(
) -> None:
    reports = (
        run_guardrail_coverage_suite(
            deterministic_runs=
                2,
        )
    )


    assert len(
        reports
    ) == 9

    assert all(
        report.passed

        for report
        in reports
    )

    assert all(
        report.metrics.deterministic

        for report
        in reports
    )


    print(
        "Nine controlled Prioritization guardrail scenarios pass: PASS"
    )


def test_full_suite_covers_all_prioritization_reason_codes(
) -> None:
    suite = (
        run_suite(
            deterministic_runs=
                2,
        )
    )


    assert (
        suite.summary.scenario_count
        ==
        12
    )

    assert (
        suite.summary.passed_scenario_count
        ==
        12
    )

    assert (
        suite.summary.failed_scenario_count
        ==
        0
    )

    assert (
        suite.summary.expectation_count
        ==
        13
    )

    assert (
        suite.summary.passed_expectation_count
        ==
        13
    )

    assert (
        suite.summary.reason_code_coverage
        ==
        1.0
    )

    assert (
        suite.summary.missing_reason_codes
        ==
        tuple()
    )

    assert (
        set(
            suite.summary.covered_reason_codes
        )
        ==
        set(
            PRIORITIZATION_REASON_CODE_TARGETS
        )
    )

    assert (
        suite.summary.expectation_accuracy
        ==
        1.0
    )

    assert (
        suite.summary.discovery_recall
        ==
        1.0
    )

    assert (
        suite.summary.selection_recall
        ==
        1.0
    )

    assert (
        suite.summary.guardrail_success_rate
        ==
        1.0
    )

    assert (
        suite.summary.determinism_rate
        ==
        1.0
    )

    assert (
        suite.summary.passed
        is True
    )


    print(
        "All eleven Prioritization reason codes are covered by the Eval Suite: PASS"
    )


def test_eval_coverage_versions(
) -> None:
    assert (
        EVAL_SUITE_RULE_VERSION
        ==
        "eval_suite_v0.2"
    )

    assert (
        EVAL_COVERAGE_RULE_VERSION
        ==
        "eval_coverage_v0.1"
    )


    print(
        "Eval Suite v0.2 / Eval Coverage v0.1 versions: PASS"
    )


def main() -> None:
    print(
        "=== DATALENS EVAL COVERAGE v0.1 ==="
    )

    print()


    test_guardrail_coverage_scenarios_pass()

    test_full_suite_covers_all_prioritization_reason_codes()

    test_eval_coverage_versions()


    print()

    print(
        "Eval Coverage v0.1: PASS"
    )


if __name__ == "__main__":
    main()
