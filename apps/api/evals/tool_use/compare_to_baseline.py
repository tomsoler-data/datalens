from __future__ import annotations


# DataLens baseline comparator v0.2
#
# Important: when a results directory contains both a legacy
# summary.json and latest_run.txt, latest_run.txt wins.


import argparse
import json
import sys
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
DEFAULT_BASELINE = HERE / "baselines" / "tool_use_cross_domain_v0_2.json"


CRITICAL_MINIMUMS = {
    "overall_pass_rate": 1.0,
    "planner_family_accuracy": 1.0,
    "planner_binding_accuracy": 1.0,
    "planner_first_pass_rate_executable": 1.0,
    "native_tool_selection_accuracy": 1.0,
    "native_argument_accuracy": 1.0,
    "native_first_pass_rate": 1.0,
    "execution_success_rate": 1.0,
    "chart_type_accuracy": 1.0,
    "guardrail_accuracy": 1.0,
}

CRITICAL_MAXIMUMS = {
    "planner_execution_retry_rate": 0.0,
    "planner_unresolved_retry_count": 0,
    "native_retry_rate": 0.0,
}

INFORMATIONAL = [
    "planner_first_pass_rate",
    "planner_retry_rate",
    "planner_retry_recovery_rate",
    "planner_guardrail_retry_rate",
    "planner_normalization_rate",
    "latency_seconds_median",
]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def pct(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, (int, float)):
        return f"{value * 100:.1f}%"
    return str(value)


def resolve_summary(path: Path) -> Path:
    path = path.resolve()


    if (
        path.is_file()
    ):
        return path


    # --------------------------------------------------------
    # Versioned-run layout (preferred).
    #
    # `latest_run.txt` is the source of truth when present.
    # A legacy root `summary.json` may still exist from the
    # pre-versioned runner and must not shadow the latest run.
    # --------------------------------------------------------

    latest = (
        path
        /
        "latest_run.txt"
    )


    if (
        latest.exists()
    ):
        run_dir_text = (
            latest.read_text(
                encoding="utf-8"
            )
            .strip()
        )


        if (
            run_dir_text
        ):
            run_dir = Path(
                run_dir_text
            )


            if (
                not run_dir.is_absolute()
            ):
                run_dir = (
                    path
                    /
                    run_dir
                ).resolve()


            candidate = (
                run_dir
                /
                "summary.json"
            )


            if (
                candidate.exists()
            ):
                return candidate


            raise FileNotFoundError(
                (
                    "`latest_run.txt` existe mais le "
                    "summary.json du run référencé est absent : "
                    f"{candidate}"
                )
            )


    # --------------------------------------------------------
    # Legacy fallback.
    # --------------------------------------------------------

    candidate = (
        path
        /
        "summary.json"
    )


    if (
        candidate.exists()
    ):
        return candidate


    raise FileNotFoundError(
        (
            "Impossible de trouver un run versionné "
            "ou summary.json depuis : "
            f"{path}"
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare a DataLens tool-use run against a frozen baseline."
    )
    parser.add_argument(
        "--summary",
        type=Path,
        required=True,
        help="Path to summary.json, a run directory, or the results root containing latest_run.txt.",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=DEFAULT_BASELINE,
    )
    parser.add_argument(
        "--fail-on-regression",
        action="store_true",
    )
    args = parser.parse_args()

    summary_path = resolve_summary(args.summary)
    current = load_json(summary_path)
    baseline = load_json(args.baseline)
    baseline_metrics = baseline["metrics"]

    failures: list[str] = []
    warnings: list[str] = []

    print("=== DataLens baseline comparison ===")
    print(f"Baseline : {baseline['baseline_id']}")
    print(f"Run      : {summary_path}")
    print()

    for metric, minimum in CRITICAL_MINIMUMS.items():
        value = current.get(metric)

        if value is None:
            failures.append(f"{metric}: missing")
            print(f"FAIL  {metric:<34} missing")
            continue

        ok = float(value) >= float(minimum)
        print(
            f"{'PASS' if ok else 'FAIL':<5} "
            f"{metric:<34} "
            f"current={pct(value):>8} "
            f"required>={pct(minimum):>8}"
        )

        if not ok:
            failures.append(
                f"{metric}: {value} < {minimum}"
            )

    for metric, maximum in CRITICAL_MAXIMUMS.items():
        value = current.get(metric)

        if value is None:
            failures.append(f"{metric}: missing")
            print(f"FAIL  {metric:<34} missing")
            continue

        ok = float(value) <= float(maximum)
        current_text = (
            str(value)
            if metric.endswith("_count")
            else pct(value)
        )
        max_text = (
            str(maximum)
            if metric.endswith("_count")
            else pct(maximum)
        )

        print(
            f"{'PASS' if ok else 'FAIL':<5} "
            f"{metric:<34} "
            f"current={current_text:>8} "
            f"required<={max_text:>8}"
        )

        if not ok:
            failures.append(
                f"{metric}: {value} > {maximum}"
            )

    print()
    print("--- Informational metrics ---")

    for metric in INFORMATIONAL:
        current_value = current.get(metric)
        baseline_value = baseline_metrics.get(metric)

        if metric == "latency_seconds_median":
            print(
                f"INFO  {metric:<34} "
                f"current={current_value:.2f}s "
                f"baseline={baseline_value:.2f}s"
            )

            if (
                isinstance(current_value, (int, float))
                and
                isinstance(baseline_value, (int, float))
                and
                current_value > baseline_value * 1.5
            ):
                warnings.append(
                    f"{metric}: > 1.5x baseline"
                )
        else:
            print(
                f"INFO  {metric:<34} "
                f"current={pct(current_value):>8} "
                f"baseline={pct(baseline_value):>8}"
            )

    print()
    print(f"Critical regressions : {len(failures)}")
    print(f"Warnings             : {len(warnings)}")

    if warnings:
        for warning in warnings:
            print(f"WARN  {warning}")

    if failures:
        for failure in failures:
            print(f"FAIL  {failure}")

    if not failures:
        print("Baseline gate        : PASS")
    else:
        print("Baseline gate        : FAIL")

    if args.fail_on_regression and failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
