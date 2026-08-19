from __future__ import annotations

from copy import deepcopy

from app.evals.regression_gate import (
    EVAL_REGRESSION_GATE_RULE_VERSION,
    evaluate_regression_gate,
)


BASELINE = {
    "baseline_id":
        "test_baseline",

    "required_versions": {
        "suite_rule_version":
            "eval_suite_v0.2",

        "coverage_rule_version":
            "eval_coverage_v0.1",
    },

    "minimum_metrics": {
        "expectation_accuracy":
            1.0,

        "discovery_recall":
            1.0,

        "selection_recall":
            1.0,

        "guardrail_success_rate":
            1.0,

        "determinism_rate":
            1.0,

        "reason_code_coverage":
            1.0,
    },

    "minimum_counts": {
        "scenario_count":
            12,

        "expectation_count":
            13,

        "passed_scenario_count":
            12,

        "passed_expectation_count":
            13,
    },

    "maximum_counts": {
        "failed_scenario_count":
            0,

        "failed_expectation_count":
            0,
    },

    "required_reason_codes": [
        "selected_by_priority",
        "quality_guard",
        "not_executable_now",
        "identifier_misuse",
        "record_label_dimension",
        "fragmented_group_dimension",
        "sparse_categorical_structure",
        "priority_below_threshold",
        "family_budget_exhausted",
        "variable_budget_exhausted",
        "global_budget_exhausted",
    ],

    "require_suite_pass":
        True,
}


REPORT = {
    "summary": {
        "scenario_count":
            12,

        "passed_scenario_count":
            12,

        "failed_scenario_count":
            0,

        "expectation_count":
            13,

        "passed_expectation_count":
            13,

        "failed_expectation_count":
            0,

        "expectation_accuracy":
            1.0,

        "discovery_recall":
            1.0,

        "selection_recall":
            1.0,

        "guardrail_success_rate":
            1.0,

        "determinism_rate":
            1.0,

        "reason_code_coverage":
            1.0,

        "covered_reason_codes":
            list(
                BASELINE[
                    "required_reason_codes"
                ]
            ),

        "missing_reason_codes":
            [],

        "passed":
            True,

        "suite_rule_version":
            "eval_suite_v0.2",

        "coverage_rule_version":
            "eval_coverage_v0.1",
    },

    "reports":
        [],
}


def test_current_baseline_passes(
) -> None:
    result = (
        evaluate_regression_gate(
            baseline=
                deepcopy(
                    BASELINE
                ),

            report=
                deepcopy(
                    REPORT
                ),
        )
    )


    assert result.passed is True
    assert result.failed_check_count == 0


    print(
        "Current Eval Suite satisfies the frozen baseline: PASS"
    )


def test_metric_regression_is_blocked(
) -> None:
    report = deepcopy(
        REPORT
    )


    report[
        "summary"
    ][
        "discovery_recall"
    ] = 0.92


    result = (
        evaluate_regression_gate(
            baseline=
                deepcopy(
                    BASELINE
                ),

            report=
                report,
        )
    )


    assert result.passed is False

    assert any(
        (
            check.check_id
            ==
            "metric:discovery_recall"
            and
            not check.passed
        )

        for check
        in result.checks
    )


    print(
        "Metric regression is blocked: PASS"
    )


def test_suite_shrinkage_is_blocked(
) -> None:
    report = deepcopy(
        REPORT
    )


    report[
        "summary"
    ][
        "scenario_count"
    ] = 8


    report[
        "summary"
    ][
        "expectation_count"
    ] = 9


    result = (
        evaluate_regression_gate(
            baseline=
                deepcopy(
                    BASELINE
                ),

            report=
                report,
        )
    )


    assert result.passed is False

    assert any(
        (
            check.check_id
            ==
            "count:scenario_count"
            and
            not check.passed
        )

        for check
        in result.checks
    )


    print(
        "Eval-suite shrinkage cannot preserve a fake 100% score: PASS"
    )


def test_reason_code_coverage_regression_is_blocked(
) -> None:
    report = deepcopy(
        REPORT
    )


    report[
        "summary"
    ][
        "covered_reason_codes"
    ].remove(
        "identifier_misuse"
    )


    result = (
        evaluate_regression_gate(
            baseline=
                deepcopy(
                    BASELINE
                ),

            report=
                report,
        )
    )


    assert result.passed is False

    assert any(
        (
            check.check_id
            ==
            "coverage:required_reason_codes"
            and
            not check.passed
        )

        for check
        in result.checks
    )


    print(
        "Required reason-code coverage regression is blocked: PASS"
    )


def test_eval_contract_version_drift_is_blocked(
) -> None:
    report = deepcopy(
        REPORT
    )


    report[
        "summary"
    ][
        "suite_rule_version"
    ] = "eval_suite_v9.9"


    result = (
        evaluate_regression_gate(
            baseline=
                deepcopy(
                    BASELINE
                ),

            report=
                report,
        )
    )


    assert result.passed is False

    assert any(
        (
            check.check_id
            ==
            "exact:suite_rule_version"
            and
            not check.passed
        )

        for check
        in result.checks
    )


    print(
        "Unexpected eval-contract version drift is blocked: PASS"
    )


def test_regression_gate_version(
) -> None:
    assert (
        EVAL_REGRESSION_GATE_RULE_VERSION
        ==
        "eval_regression_gate_v0.1"
    )


    print(
        "Eval Regression Gate rule version: PASS"
    )


def main() -> None:
    print(
        "=== DATALENS EVAL REGRESSION GATE v0.1 ==="
    )

    print()


    test_current_baseline_passes()

    test_metric_regression_is_blocked()

    test_suite_shrinkage_is_blocked()

    test_reason_code_coverage_regression_is_blocked()

    test_eval_contract_version_drift_is_blocked()

    test_regression_gate_version()


    print()

    print(
        "Eval Regression Gate v0.1: PASS"
    )


if __name__ == "__main__":
    main()
