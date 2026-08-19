from __future__ import annotations

from dataclasses import (
    dataclass,
)

import argparse
import json
from pathlib import Path

from typing import (
    Any,
)


EVAL_REGRESSION_GATE_RULE_VERSION = (
    "eval_regression_gate_v0.1"
)


# ============================================================
# RESULT CONTRACTS
# ============================================================


@dataclass(
    frozen=True,
)
class RegressionCheck:
    check_id: str
    passed: bool

    actual: Any
    expected: Any

    message: str


    def as_dict(
        self,
    ) -> dict[
        str,
        Any,
    ]:
        return {
            "check_id":
                self.check_id,

            "passed":
                self.passed,

            "actual":
                self.actual,

            "expected":
                self.expected,

            "message":
                self.message,
        }


@dataclass(
    frozen=True,
)
class RegressionGateResult:
    baseline_id: str

    checks: tuple[
        RegressionCheck,
        ...,
    ]

    passed: bool


    @property
    def passed_check_count(
        self,
    ) -> int:
        return sum(
            1
            for check
            in self.checks
            if check.passed
        )


    @property
    def failed_check_count(
        self,
    ) -> int:
        return (
            len(
                self.checks
            )
            -
            self.passed_check_count
        )


    def as_dict(
        self,
    ) -> dict[
        str,
        Any,
    ]:
        return {
            "baseline_id":
                self.baseline_id,

            "passed":
                self.passed,

            "passed_check_count":
                self.passed_check_count,

            "failed_check_count":
                self.failed_check_count,

            "checks": [
                check.as_dict()

                for check
                in self.checks
            ],

            "rule_version":
                EVAL_REGRESSION_GATE_RULE_VERSION,
        }


# ============================================================
# JSON LOADING
# ============================================================


def _load_json_object(
    path: str | Path,
    *,
    label: str,
) -> dict[
    str,
    Any,
]:
    target = Path(
        path
    )


    if (
        not target.exists()
    ):
        raise FileNotFoundError(
            f"{label} introuvable : {target}"
        )


    try:
        payload = json.loads(
            target.read_text(
                encoding="utf-8"
            )
        )

    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{label} JSON invalide : {target} "
            f"({exc.msg})"
        ) from exc


    if (
        not isinstance(
            payload,
            dict,
        )
    ):
        raise ValueError(
            f"{label} doit contenir un objet JSON."
        )


    return payload


def load_baseline(
    path: str | Path,
) -> dict[
    str,
    Any,
]:
    payload = (
        _load_json_object(
            path,
            label="Baseline",
        )
    )


    required_fields = {
        "baseline_id",
        "minimum_metrics",
        "minimum_counts",
        "required_reason_codes",
    }


    missing = (
        required_fields
        -
        set(
            payload
        )
    )


    if missing:
        raise ValueError(
            "Baseline incomplète. Champs manquants : "
            +
            ", ".join(
                sorted(
                    missing
                )
            )
        )


    if (
        not isinstance(
            payload[
                "minimum_metrics"
            ],
            dict,
        )
    ):
        raise ValueError(
            "baseline.minimum_metrics doit être un objet."
        )


    if (
        not isinstance(
            payload[
                "minimum_counts"
            ],
            dict,
        )
    ):
        raise ValueError(
            "baseline.minimum_counts doit être un objet."
        )


    if (
        not isinstance(
            payload[
                "required_reason_codes"
            ],
            list,
        )
    ):
        raise ValueError(
            "baseline.required_reason_codes doit être une liste."
        )


    return payload


def load_eval_report(
    path: str | Path,
) -> dict[
    str,
    Any,
]:
    payload = (
        _load_json_object(
            path,
            label="Rapport d'evals",
        )
    )


    summary = payload.get(
        "summary"
    )


    if (
        not isinstance(
            summary,
            dict,
        )
    ):
        raise ValueError(
            "Le rapport d'evals doit contenir "
            "un objet 'summary'."
        )


    return payload


# ============================================================
# CHECK HELPERS
# ============================================================


def _metric_check(
    *,
    summary: dict[
        str,
        Any,
    ],
    metric_name: str,
    minimum: float,
) -> RegressionCheck:
    actual = summary.get(
        metric_name
    )


    if (
        not isinstance(
            actual,
            (
                int,
                float,
            ),
        )
    ):
        return (
            RegressionCheck(
                check_id=
                    f"metric:{metric_name}",

                passed=
                    False,

                actual=
                    actual,

                expected=
                    {
                        "minimum":
                            minimum
                    },

                message=(
                    f"Métrique absente ou invalide : "
                    f"{metric_name}."
                ),
            )
        )


    actual_value = float(
        actual
    )


    passed = (
        actual_value
        >=
        float(
            minimum
        )
    )


    return (
        RegressionCheck(
            check_id=
                f"metric:{metric_name}",

            passed=
                passed,

            actual=
                actual_value,

            expected=
                {
                    "minimum":
                        float(
                            minimum
                        )
                },

            message=(
                f"{metric_name}: "
                f"{actual_value:.4f} "
                f"{'>=' if passed else '<'} "
                f"{float(minimum):.4f}"
            ),
        )
    )


def _count_check(
    *,
    summary: dict[
        str,
        Any,
    ],
    count_name: str,
    minimum: int,
) -> RegressionCheck:
    actual = summary.get(
        count_name
    )


    if (
        not isinstance(
            actual,
            int,
        )
    ):
        return (
            RegressionCheck(
                check_id=
                    f"count:{count_name}",

                passed=
                    False,

                actual=
                    actual,

                expected=
                    {
                        "minimum":
                            minimum
                    },

                message=(
                    f"Compteur absent ou invalide : "
                    f"{count_name}."
                ),
            )
        )


    passed = (
        actual
        >=
        minimum
    )


    return (
        RegressionCheck(
            check_id=
                f"count:{count_name}",

            passed=
                passed,

            actual=
                actual,

            expected=
                {
                    "minimum":
                        minimum
                },

            message=(
                f"{count_name}: "
                f"{actual} "
                f"{'>=' if passed else '<'} "
                f"{minimum}"
            ),
        )
    )


def _maximum_count_check(
    *,
    summary: dict[
        str,
        Any,
    ],
    count_name: str,
    maximum: int,
) -> RegressionCheck:
    actual = summary.get(
        count_name
    )


    if (
        not isinstance(
            actual,
            int,
        )
    ):
        return (
            RegressionCheck(
                check_id=
                    f"maximum:{count_name}",

                passed=
                    False,

                actual=
                    actual,

                expected=
                    {
                        "maximum":
                            maximum
                    },

                message=(
                    f"Compteur absent ou invalide : "
                    f"{count_name}."
                ),
            )
        )


    passed = (
        actual
        <=
        maximum
    )


    return (
        RegressionCheck(
            check_id=
                f"maximum:{count_name}",

            passed=
                passed,

            actual=
                actual,

            expected=
                {
                    "maximum":
                        maximum
                },

            message=(
                f"{count_name}: "
                f"{actual} "
                f"{'<=' if passed else '>'} "
                f"{maximum}"
            ),
        )
    )


def _exact_check(
    *,
    summary: dict[
        str,
        Any,
    ],
    field_name: str,
    expected: Any,
) -> RegressionCheck:
    actual = summary.get(
        field_name
    )


    passed = (
        actual
        ==
        expected
    )


    return (
        RegressionCheck(
            check_id=
                f"exact:{field_name}",

            passed=
                passed,

            actual=
                actual,

            expected=
                expected,

            message=(
                f"{field_name}: "
                f"{actual!r} "
                f"{'==' if passed else '!='} "
                f"{expected!r}"
            ),
        )
    )


# ============================================================
# MAIN EVALUATOR
# ============================================================


def evaluate_regression_gate(
    *,
    baseline: dict[
        str,
        Any,
    ],
    report: dict[
        str,
        Any,
    ],
) -> RegressionGateResult:
    summary = report.get(
        "summary"
    )


    if (
        not isinstance(
            summary,
            dict,
        )
    ):
        raise ValueError(
            "Le rapport d'evals ne contient pas "
            "de summary valide."
        )


    checks: list[
        RegressionCheck
    ] = []


    # --------------------------------------------------------
    # REQUIRED ENGINE / EVAL CONTRACT VERSIONS
    # --------------------------------------------------------

    required_versions = (
        baseline.get(
            "required_versions",
            {},
        )
    )


    if (
        not isinstance(
            required_versions,
            dict,
        )
    ):
        raise ValueError(
            "baseline.required_versions doit être un objet."
        )


    for (
        field_name,
        expected,
    ) in required_versions.items():
        checks.append(
            _exact_check(
                summary=
                    summary,

                field_name=
                    str(
                        field_name
                    ),

                expected=
                    expected,
            )
        )


    # --------------------------------------------------------
    # METRIC FLOORS
    # --------------------------------------------------------

    minimum_metrics = (
        baseline[
            "minimum_metrics"
        ]
    )


    for (
        metric_name,
        minimum,
    ) in minimum_metrics.items():
        if (
            not isinstance(
                minimum,
                (
                    int,
                    float,
                ),
            )
        ):
            raise ValueError(
                "Toutes les valeurs de minimum_metrics "
                "doivent être numériques."
            )


        checks.append(
            _metric_check(
                summary=
                    summary,

                metric_name=
                    str(
                        metric_name
                    ),

                minimum=
                    float(
                        minimum
                    ),
            )
        )


    # --------------------------------------------------------
    # MINIMUM COVERAGE COUNTS
    # --------------------------------------------------------

    minimum_counts = (
        baseline[
            "minimum_counts"
        ]
    )


    for (
        count_name,
        minimum,
    ) in minimum_counts.items():
        if (
            not isinstance(
                minimum,
                int,
            )
        ):
            raise ValueError(
                "Toutes les valeurs de minimum_counts "
                "doivent être des entiers."
            )


        checks.append(
            _count_check(
                summary=
                    summary,

                count_name=
                    str(
                        count_name
                    ),

                minimum=
                    minimum,
            )
        )


    # --------------------------------------------------------
    # MAXIMUM FAILURE COUNTS
    # --------------------------------------------------------

    maximum_counts = (
        baseline.get(
            "maximum_counts",
            {},
        )
    )


    if (
        not isinstance(
            maximum_counts,
            dict,
        )
    ):
        raise ValueError(
            "baseline.maximum_counts doit être un objet."
        )


    for (
        count_name,
        maximum,
    ) in maximum_counts.items():
        if (
            not isinstance(
                maximum,
                int,
            )
        ):
            raise ValueError(
                "Toutes les valeurs de maximum_counts "
                "doivent être des entiers."
            )


        checks.append(
            _maximum_count_check(
                summary=
                    summary,

                count_name=
                    str(
                        count_name
                    ),

                maximum=
                    maximum,
            )
        )


    # --------------------------------------------------------
    # REQUIRED PRIORITIZATION REASON CODES
    # --------------------------------------------------------

    required_reason_codes = {
        str(
            reason
        )

        for reason
        in baseline[
            "required_reason_codes"
        ]
    }


    actual_reason_codes_raw = (
        summary.get(
            "covered_reason_codes",
            [],
        )
    )


    if (
        not isinstance(
            actual_reason_codes_raw,
            list,
        )
    ):
        actual_reason_codes_raw = []


    actual_reason_codes = {
        str(
            reason
        )

        for reason
        in actual_reason_codes_raw
    }


    missing_reason_codes = tuple(
        sorted(
            required_reason_codes
            -
            actual_reason_codes
        )
    )


    checks.append(
        RegressionCheck(
            check_id=
                "coverage:required_reason_codes",

            passed=
                not missing_reason_codes,

            actual=
                sorted(
                    actual_reason_codes
                ),

            expected=
                sorted(
                    required_reason_codes
                ),

            message=(
                "Tous les reason codes requis sont présents."
                if not missing_reason_codes
                else
                (
                    "Reason codes manquants : "
                    +
                    ", ".join(
                        missing_reason_codes
                    )
                )
            ),
        )
    )


    # --------------------------------------------------------
    # SUITE SELF-REPORTED STATUS
    # --------------------------------------------------------

    require_suite_pass = bool(
        baseline.get(
            "require_suite_pass",
            True,
        )
    )


    if require_suite_pass:
        checks.append(
            _exact_check(
                summary=
                    summary,

                field_name=
                    "passed",

                expected=
                    True,
            )
        )


    # --------------------------------------------------------
    # FINAL
    # --------------------------------------------------------

    passed = all(
        check.passed

        for check
        in checks
    )


    return (
        RegressionGateResult(
            baseline_id=
                str(
                    baseline[
                        "baseline_id"
                    ]
                ),

            checks=
                tuple(
                    checks
                ),

            passed=
                passed,
        )
    )


def evaluate_regression_gate_files(
    *,
    baseline_path: str | Path,
    report_path: str | Path,
) -> RegressionGateResult:
    return (
        evaluate_regression_gate(
            baseline=
                load_baseline(
                    baseline_path
                ),

            report=
                load_eval_report(
                    report_path
                ),
        )
    )


# ============================================================
# DISPLAY
# ============================================================


def _format_value(
    value: Any,
) -> str:
    if (
        isinstance(
            value,
            float,
        )
    ):
        if (
            0.0
            <=
            value
            <=
            1.0
        ):
            return (
                f"{value * 100:.1f}%"
            )


        return (
            f"{value:.4f}"
        )


    if (
        isinstance(
            value,
            list,
        )
    ):
        return (
            f"{len(value)} item(s)"
        )


    return str(
        value
    )


def print_regression_gate(
    result: RegressionGateResult,
) -> None:
    print(
        "=== DATALENS EVAL REGRESSION GATE v0.1 ==="
    )

    print()

    print(
        f"Baseline: {result.baseline_id}"
    )

    print()


    for check in result.checks:
        status = (
            "PASS"
            if check.passed
            else
            "FAIL"
        )


        actual = (
            _format_value(
                check.actual
            )
        )


        print(
            f"{check.check_id:<42} "
            f"{actual:<14} "
            f"{status}"
        )


        if (
            not check.passed
        ):
            print(
                f"  {check.message}"
            )


    print()

    print(
        "----------------------------------------"
    )

    print(
        "Checks                      "
        f"{len(result.checks)}"
    )

    print(
        "Passed                      "
        f"{result.passed_check_count}"
    )

    print(
        "Failed                      "
        f"{result.failed_check_count}"
    )

    print()

    print(
        "Regression gate             "
        f"{'PASS' if result.passed else 'FAIL'}"
    )

    print(
        "Rule                        "
        f"{EVAL_REGRESSION_GATE_RULE_VERSION}"
    )


# ============================================================
# OPTIONAL MACHINE-READABLE GATE REPORT
# ============================================================


def write_gate_report(
    result: RegressionGateResult,
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
            result.as_dict(),
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
            "Compare a DataLens eval report against a frozen "
            "regression baseline."
        ),
    )


    parser.add_argument(
        "--baseline",
        required=True,
        help=
            "Path to the frozen eval regression baseline JSON.",
    )


    parser.add_argument(
        "--report",
        required=True,
        help=
            "Path to the DataLens eval suite JSON report.",
    )


    parser.add_argument(
        "--json-output",
        default=None,
        help=(
            "Optional path for the machine-readable regression "
            "gate result."
        ),
    )


    args = parser.parse_args()


    result = (
        evaluate_regression_gate_files(
            baseline_path=
                args.baseline,

            report_path=
                args.report,
        )
    )


    print_regression_gate(
        result
    )


    if args.json_output:
        output = (
            write_gate_report(
                result,
                args.json_output,
            )
        )


        print()

        print(
            f"Gate report: {output}"
        )


    if (
        not result.passed
    ):
        raise SystemExit(
            1
        )


if __name__ == "__main__":
    main()
