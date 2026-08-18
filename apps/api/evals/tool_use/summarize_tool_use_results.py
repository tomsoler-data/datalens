from __future__ import annotations

import argparse
import csv
import json
import statistics

from collections import defaultdict
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent


def parse_bool(value: str) -> bool:
    return value.strip().casefold() in {
        "true",
        "1",
        "yes",
    }


def parse_int(value: str) -> int:
    try:
        return int(value)
    except Exception:
        return 0


def parse_float(value: str) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def rate(values: list[bool]) -> float:
    if not values:
        return 0.0

    return sum(values) / len(values)


def percent(value: float | None) -> str:
    if value is None:
        return "n/a"

    return f"{value * 100:.1f}%"


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        return list(
            csv.DictReader(
                handle
            )
        )


def summarize_case(
    case_id: str,
    rows: list[dict[str, str]],
) -> dict[str, Any]:
    planner_retry_rows = [
        row
        for row in rows
        if parse_int(
            row.get(
                "planner_retry_count",
                "0",
            )
        ) > 0
    ]

    tool_retry_rows = [
        row
        for row in rows
        if parse_int(
            row.get(
                "tool_retry_count",
                "0",
            )
        ) > 0
    ]

    latencies = [
        parse_float(
            row.get(
                "latency_seconds",
                "0",
            )
        )
        for row in rows
    ]

    return {
        "case_id": case_id,
        "runs": len(rows),
        "pass_rate": rate(
            [
                parse_bool(
                    row.get(
                        "case_pass",
                        "false",
                    )
                )
                for row in rows
            ]
        ),
        "planner_family_accuracy": rate(
            [
                parse_bool(
                    row.get(
                        "planner_family_correct",
                        "false",
                    )
                )
                for row in rows
            ]
        ),
        "planner_binding_accuracy": rate(
            [
                parse_bool(
                    row.get(
                        "planner_bindings_correct",
                        "false",
                    )
                )
                for row in rows
            ]
        ),
        "planner_first_pass_rate": rate(
            [
                parse_bool(
                    row.get(
                        "planner_first_pass",
                        "false",
                    )
                )
                for row in rows
            ]
        ),
        "planner_retry_rate": (
            len(planner_retry_rows)
            /
            len(rows)
            if rows
            else 0.0
        ),
        "planner_retry_recovery_rate": (
            rate(
                [
                    parse_bool(
                        row.get(
                            "planner_recovered_after_retry",
                            "false",
                        )
                    )
                    for row in planner_retry_rows
                ]
            )
            if planner_retry_rows
            else None
        ),
        "planner_normalization_rate": rate(
            [
                parse_int(
                    row.get(
                        "planner_normalization_count",
                        "0",
                    )
                ) > 0
                for row in rows
            ]
        ),
        "tool_selection_accuracy": rate(
            [
                parse_bool(
                    row.get(
                        "tool_selection_correct",
                        "false",
                    )
                )
                for row in rows
            ]
        ),
        "tool_argument_accuracy": rate(
            [
                parse_bool(
                    row.get(
                        "tool_arguments_correct",
                        "false",
                    )
                )
                for row in rows
            ]
        ),
        "tool_first_pass_rate": rate(
            [
                parse_bool(
                    row.get(
                        "tool_first_pass",
                        "false",
                    )
                )
                for row in rows
            ]
        ),
        "tool_retry_rate": (
            len(tool_retry_rows)
            /
            len(rows)
            if rows
            else 0.0
        ),
        "tool_retry_recovery_rate": (
            rate(
                [
                    parse_bool(
                        row.get(
                            "tool_recovered_after_retry",
                            "false",
                        )
                    )
                    for row in tool_retry_rows
                ]
            )
            if tool_retry_rows
            else None
        ),
        "execution_accuracy": rate(
            [
                parse_bool(
                    row.get(
                        "execution_correct",
                        "false",
                    )
                )
                for row in rows
            ]
        ),
        "guardrail_accuracy": rate(
            [
                parse_bool(
                    row.get(
                        "guardrail_correct",
                        "false",
                    )
                )
                for row in rows
            ]
        ),
        "latency_mean_seconds": (
            statistics.mean(
                latencies
            )
            if latencies
            else 0.0
        ),
        "latency_median_seconds": (
            statistics.median(
                latencies
            )
            if latencies
            else 0.0
        ),
        "planner_statuses": sorted(
            {
                row.get(
                    "planner_status",
                    "",
                )
                for row in rows
            }
        ),
        "requested_tools": sorted(
            {
                (
                    row.get(
                        "requested_tool",
                        "",
                    )
                    or "none"
                )
                for row in rows
            }
        ),
        "failure_count": sum(
            1
            for row in rows
            if not parse_bool(
                row.get(
                    "case_pass",
                    "false",
                )
            )
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--results",
        type=Path,
        default=(
            HERE
            /
            "results"
            /
            "results.csv"
        ),
    )

    args = parser.parse_args()

    rows = load_rows(
        args.results
    )

    if not rows:
        raise SystemExit(
            "results.csv is empty."
        )

    grouped: dict[
        str,
        list[dict[str, str]],
    ] = defaultdict(list)

    for row in rows:
        grouped[
            row[
                "case_id"
            ]
        ].append(
            row
        )

    summaries = [
        summarize_case(
            case_id,
            grouped[
                case_id
            ],
        )
        for case_id in sorted(
            grouped
        )
    ]

    output_csv = (
        args.results.parent
        /
        "summary_by_case.csv"
    )

    output_json = (
        args.results.parent
        /
        "summary_by_case.json"
    )

    with output_csv.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                summaries[
                    0
                ].keys()
            ),
        )
        writer.writeheader()
        writer.writerows(
            summaries
        )

    output_json.write_text(
        json.dumps(
            summaries,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "=== DataLens per-case stability ==="
    )

    for row in summaries:
        print(
            f"{row['case_id']:<30} "
            f"pass={percent(row['pass_rate']):>6} · "
            f"planner first={percent(row['planner_first_pass_rate']):>6} · "
            f"planner retry={percent(row['planner_retry_rate']):>6} · "
            f"tool first={percent(row['tool_first_pass_rate']):>6} · "
            f"median={row['latency_median_seconds']:.2f}s"
        )

    print()
    print(f"CSV  : {output_csv}")
    print(f"JSON : {output_json}")


if __name__ == "__main__":
    main()
