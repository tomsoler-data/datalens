from __future__ import annotations

from dataclasses import dataclass

import argparse
import json
from pathlib import Path

from app.analysis_prioritization import (
    prioritize_analysis_discovery,
)

from app.evals.analysis_benchmark import (
    AnalysisBenchmarkReport,
    discovery_fingerprint,
    evaluate_analysis_benchmark,
    prioritization_fingerprint,
    run_analysis_benchmark,
)

from app.evals.scenarios import (
    ControlledPrioritizationEval,
    EVAL_COVERAGE_RULE_VERSION,
    build_analysis_eval_scenarios,
    build_prioritization_guardrail_evals,
)


EVAL_SUITE_RULE_VERSION = (
    "eval_suite_v0.2"
)


PRIORITIZATION_REASON_CODE_TARGETS = (
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
)


# ============================================================
# SUMMARY
# ============================================================


@dataclass(frozen=True)
class EvalSuiteSummary:
    scenario_count: int
    passed_scenario_count: int
    failed_scenario_count: int

    expectation_count: int
    passed_expectation_count: int
    failed_expectation_count: int

    required_discovery_count: int
    discovered_required_count: int

    required_selection_count: int
    selected_required_count: int

    guardrail_expectation_count: int
    guardrail_pass_count: int

    deterministic_scenario_count: int

    covered_reason_codes: tuple[
        str,
        ...,
    ]

    missing_reason_codes: tuple[
        str,
        ...,
    ]

    passed: bool


    @staticmethod
    def _rate(
        numerator: int,
        denominator: int,
    ) -> float:
        if denominator <= 0:
            return 1.0

        return (
            numerator
            /
            denominator
        )


    @property
    def expectation_accuracy(
        self,
    ) -> float:
        return self._rate(
            self.passed_expectation_count,
            self.expectation_count,
        )


    @property
    def discovery_recall(
        self,
    ) -> float:
        return self._rate(
            self.discovered_required_count,
            self.required_discovery_count,
        )


    @property
    def selection_recall(
        self,
    ) -> float:
        return self._rate(
            self.selected_required_count,
            self.required_selection_count,
        )


    @property
    def guardrail_success_rate(
        self,
    ) -> float:
        return self._rate(
            self.guardrail_pass_count,
            self.guardrail_expectation_count,
        )


    @property
    def determinism_rate(
        self,
    ) -> float:
        return self._rate(
            self.deterministic_scenario_count,
            self.scenario_count,
        )


    @property
    def reason_code_coverage(
        self,
    ) -> float:
        return self._rate(
            len(
                self.covered_reason_codes
            ),
            len(
                PRIORITIZATION_REASON_CODE_TARGETS
            ),
        )


    def as_dict(
        self,
    ) -> dict:
        return {
            "scenario_count":
                self.scenario_count,

            "passed_scenario_count":
                self.passed_scenario_count,

            "failed_scenario_count":
                self.failed_scenario_count,

            "expectation_count":
                self.expectation_count,

            "passed_expectation_count":
                self.passed_expectation_count,

            "failed_expectation_count":
                self.failed_expectation_count,

            "expectation_accuracy":
                round(
                    self.expectation_accuracy,
                    4,
                ),

            "discovery_recall":
                round(
                    self.discovery_recall,
                    4,
                ),

            "selection_recall":
                round(
                    self.selection_recall,
                    4,
                ),

            "guardrail_success_rate":
                round(
                    self.guardrail_success_rate,
                    4,
                ),

            "determinism_rate":
                round(
                    self.determinism_rate,
                    4,
                ),

            "reason_code_coverage":
                round(
                    self.reason_code_coverage,
                    4,
                ),

            "covered_reason_codes":
                list(
                    self.covered_reason_codes
                ),

            "missing_reason_codes":
                list(
                    self.missing_reason_codes
                ),

            "passed":
                self.passed,

            "suite_rule_version":
                EVAL_SUITE_RULE_VERSION,

            "coverage_rule_version":
                EVAL_COVERAGE_RULE_VERSION,
        }


@dataclass(frozen=True)
class EvalSuiteRun:
    reports: tuple[
        AnalysisBenchmarkReport,
        ...,
    ]

    summary: EvalSuiteSummary


    def as_dict(
        self,
    ) -> dict:
        return {
            "summary":
                self.summary.as_dict(),

            "reports": [
                report.model_dump(
                    mode="json",
                )

                for report
                in self.reports
            ],
        }


# ============================================================
# CONTROLLED PRIORITIZATION EVAL
# ============================================================


def _clone_datasets(
    datasets: tuple[
        dict,
        ...,
    ],
) -> list[
    dict,
]:
    output: list[
        dict,
    ] = []


    for dataset in datasets:
        clone = dict(
            dataset
        )

        dataframe = dataset.get(
            "dataframe"
        )

        if dataframe is not None:
            clone[
                "dataframe"
            ] = dataframe.copy(
                deep=True
            )

        output.append(
            clone
        )


    return output


def run_controlled_prioritization_eval(
    controlled: ControlledPrioritizationEval,
    *,
    deterministic_runs: int = 2,
) -> AnalysisBenchmarkReport:
    if deterministic_runs < 1:
        raise ValueError(
            "deterministic_runs must be at least 1."
        )


    prioritizations = []


    for _ in range(
        deterministic_runs
    ):
        discovery = (
            controlled.discovery
            .model_copy(
                deep=True
            )
        )


        prioritization = (
            prioritize_analysis_discovery(
                discovery,

                datasets=
                    _clone_datasets(
                        controlled
                        .scenario
                        .datasets
                    ),
            )
        )


        prioritizations.append(
            prioritization
        )


    discovery_hash = (
        discovery_fingerprint(
            controlled.discovery
        )
    )


    prioritization_hashes = [
        prioritization_fingerprint(
            report
        )

        for report
        in prioritizations
    ]


    deterministic = (
        len(
            set(
                prioritization_hashes
            )
        )
        ==
        1
    )


    report = (
        evaluate_analysis_benchmark(
            scenario=
                controlled.scenario,

            discovery=
                controlled.discovery
                .model_copy(
                    deep=True
                ),

            prioritization=
                prioritizations[
                    0
                ],

            deterministic=
                deterministic,

            deterministic_run_count=
                deterministic_runs,
        )
    )


    if (
        report.discovery_fingerprint
        !=
        discovery_hash
    ):
        raise AssertionError(
            "Controlled Discovery fingerprint changed during eval."
        )


    return report


# ============================================================
# SUITE PARTS
# ============================================================


def run_core_suite(
    *,
    deterministic_runs: int = 2,
) -> tuple[
    AnalysisBenchmarkReport,
    ...,
]:
    """
    Backward-compatible v0.1 core:
    real Discovery + Prioritization.
    """

    return tuple(
        run_analysis_benchmark(
            scenario,

            deterministic_runs=
                deterministic_runs,
        )

        for scenario
        in build_analysis_eval_scenarios()
    )


def run_guardrail_coverage_suite(
    *,
    deterministic_runs: int = 2,
) -> tuple[
    AnalysisBenchmarkReport,
    ...,
]:
    """
    Controlled Prioritization contracts used to cover rare or
    boundary decisions precisely.
    """

    return tuple(
        run_controlled_prioritization_eval(
            controlled,

            deterministic_runs=
                deterministic_runs,
        )

        for controlled
        in build_prioritization_guardrail_evals()
    )


# ============================================================
# AGGREGATION
# ============================================================


def summarize_reports(
    reports: tuple[
        AnalysisBenchmarkReport,
        ...,
    ],
) -> EvalSuiteSummary:
    scenario_count = len(
        reports
    )


    passed_scenario_count = sum(
        1
        for report
        in reports
        if report.passed
    )


    failed_scenario_count = (
        scenario_count
        -
        passed_scenario_count
    )


    expectation_count = sum(
        report.metrics.expectation_count

        for report
        in reports
    )


    passed_expectation_count = sum(
        report.metrics
        .passed_expectation_count

        for report
        in reports
    )


    failed_expectation_count = sum(
        report.metrics
        .failed_expectation_count

        for report
        in reports
    )


    required_discovery_count = sum(
        report.metrics
        .required_discovery_count

        for report
        in reports
    )


    discovered_required_count = sum(
        report.metrics
        .discovered_required_count

        for report
        in reports
    )


    required_selection_count = sum(
        report.metrics
        .required_selection_count

        for report
        in reports
    )


    selected_required_count = sum(
        report.metrics
        .selected_required_count

        for report
        in reports
    )


    guardrail_expectation_count = sum(
        report.metrics
        .guardrail_expectation_count

        for report
        in reports
    )


    guardrail_pass_count = sum(
        report.metrics
        .guardrail_pass_count

        for report
        in reports
    )


    deterministic_scenario_count = sum(
        1
        for report
        in reports
        if report.metrics.deterministic
    )


    actual_reason_codes = {
        outcome.actual_reason_code

        for report
        in reports

        for outcome
        in report.outcomes

        if (
            outcome.passed
            and
            outcome.actual_reason_code
            is not None
        )
    }


    covered_reason_codes = tuple(
        reason_code

        for reason_code
        in PRIORITIZATION_REASON_CODE_TARGETS

        if reason_code
        in actual_reason_codes
    )


    missing_reason_codes = tuple(
        reason_code

        for reason_code
        in PRIORITIZATION_REASON_CODE_TARGETS

        if reason_code
        not in actual_reason_codes
    )


    passed = (
        scenario_count > 0
        and
        failed_scenario_count == 0
        and
        failed_expectation_count == 0
        and
        deterministic_scenario_count
        ==
        scenario_count
        and
        not missing_reason_codes
    )


    return (
        EvalSuiteSummary(
            scenario_count=
                scenario_count,

            passed_scenario_count=
                passed_scenario_count,

            failed_scenario_count=
                failed_scenario_count,

            expectation_count=
                expectation_count,

            passed_expectation_count=
                passed_expectation_count,

            failed_expectation_count=
                failed_expectation_count,

            required_discovery_count=
                required_discovery_count,

            discovered_required_count=
                discovered_required_count,

            required_selection_count=
                required_selection_count,

            selected_required_count=
                selected_required_count,

            guardrail_expectation_count=
                guardrail_expectation_count,

            guardrail_pass_count=
                guardrail_pass_count,

            deterministic_scenario_count=
                deterministic_scenario_count,

            covered_reason_codes=
                covered_reason_codes,

            missing_reason_codes=
                missing_reason_codes,

            passed=
                passed,
        )
    )


# ============================================================
# COMPLETE SUITE
# ============================================================


def run_suite(
    *,
    deterministic_runs: int = 2,
) -> EvalSuiteRun:
    if deterministic_runs < 1:
        raise ValueError(
            "deterministic_runs must be at least 1."
        )


    reports = (
        run_core_suite(
            deterministic_runs=
                deterministic_runs,
        )
        +
        run_guardrail_coverage_suite(
            deterministic_runs=
                deterministic_runs,
        )
    )


    return (
        EvalSuiteRun(
            reports=
                reports,

            summary=
                summarize_reports(
                    reports
                ),
        )
    )


# ============================================================
# DISPLAY
# ============================================================


def _percent(
    value: float,
) -> str:
    return (
        f"{value * 100:.1f}%"
    )


def print_suite(
    suite: EvalSuiteRun,
) -> None:
    print(
        "=== DATALENS EVAL SUITE v0.2 ==="
    )

    print()


    for report in suite.reports:
        status = (
            "PASS"
            if report.passed
            else
            "FAIL"
        )


        print(
            f"{report.scenario_id} — {status}"
        )

        print(
            "  Expectation accuracy   "
            f"{_percent(report.metrics.expectation_accuracy)}"
        )

        print(
            "  Discovery recall       "
            f"{_percent(report.metrics.discovery_recall)}"
        )

        print(
            "  Selection recall       "
            f"{_percent(report.metrics.selection_recall)}"
        )

        print(
            "  Guardrail success      "
            f"{_percent(report.metrics.guardrail_success_rate)}"
        )

        print(
            "  Determinism            "
            f"{'PASS' if report.metrics.deterministic else 'FAIL'}"
        )


        for outcome in report.outcomes:
            if (
                outcome.passed
                and
                outcome.actual_reason_code
            ):
                print(
                    "  ✓ "
                    f"{outcome.expectation_id}: "
                    f"{outcome.actual_reason_code}"
                )


        for outcome in report.outcomes:
            if not outcome.passed:
                print(
                    "  ! "
                    f"{outcome.expectation_id}: "
                    f"{outcome.failure_code}"
                )

                if outcome.actual_decision:
                    print(
                        "      actual decision: "
                        f"{outcome.actual_decision}"
                    )

                if outcome.actual_reason_code:
                    print(
                        "      actual reason:   "
                        f"{outcome.actual_reason_code}"
                    )


        print()


    summary = suite.summary


    print(
        "----------------------------------------"
    )

    print(
        "Scenarios                  "
        f"{summary.scenario_count}"
    )

    print(
        "Passed                     "
        f"{summary.passed_scenario_count}"
    )

    print(
        "Failed                     "
        f"{summary.failed_scenario_count}"
    )

    print(
        "Expectations                "
        f"{summary.passed_expectation_count}"
        "/"
        f"{summary.expectation_count}"
    )

    print()

    print(
        "Overall expectation accuracy "
        f"{_percent(summary.expectation_accuracy)}"
    )

    print(
        "Overall discovery recall     "
        f"{_percent(summary.discovery_recall)}"
    )

    print(
        "Overall selection recall     "
        f"{_percent(summary.selection_recall)}"
    )

    print(
        "Overall guardrail success    "
        f"{_percent(summary.guardrail_success_rate)}"
    )

    print(
        "Overall determinism          "
        f"{_percent(summary.determinism_rate)}"
    )

    print(
        "Prioritization reason cover  "
        f"{_percent(summary.reason_code_coverage)}"
        " "
        f"({len(summary.covered_reason_codes)}"
        "/"
        f"{len(PRIORITIZATION_REASON_CODE_TARGETS)})"
    )


    if summary.missing_reason_codes:
        print(
            "Missing reason codes         "
            +
            ", ".join(
                summary.missing_reason_codes
            )
        )


    print()

    print(
        "Suite                       "
        f"{'PASS' if summary.passed else 'FAIL'}"
    )

    print(
        "Suite rule                  "
        f"{EVAL_SUITE_RULE_VERSION}"
    )

    print(
        "Coverage rule               "
        f"{EVAL_COVERAGE_RULE_VERSION}"
    )


# ============================================================
# JSON
# ============================================================


def write_json_report(
    suite: EvalSuiteRun,
    output_path: str | Path,
) -> Path:
    path = Path(
        output_path
    )


    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    path.write_text(
        json.dumps(
            suite.as_dict(),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


    return path


# ============================================================
# CLI
# ============================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the DataLens Discovery/Prioritization eval suite "
            "and deterministic guardrail coverage."
        ),
    )


    parser.add_argument(
        "--deterministic-runs",
        type=int,
        default=2,
        help=(
            "Number of identical executions used to verify "
            "determinism."
        ),
    )


    parser.add_argument(
        "--json-output",
        type=str,
        default=None,
        help=(
            "Optional path for the complete JSON evaluation report."
        ),
    )


    args = parser.parse_args()


    suite = (
        run_suite(
            deterministic_runs=
                args.deterministic_runs,
        )
    )


    print_suite(
        suite
    )


    if args.json_output:
        path = (
            write_json_report(
                suite,
                args.json_output,
            )
        )

        print()

        print(
            f"JSON report: {path}"
        )


    if not suite.summary.passed:
        raise SystemExit(
            1
        )


if __name__ == "__main__":
    main()
