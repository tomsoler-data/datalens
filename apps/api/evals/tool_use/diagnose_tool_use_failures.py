from __future__ import annotations

import argparse
import json

from pathlib import Path
from typing import Any


HERE = Path(
    __file__
).resolve().parent


DEFAULT_RESULTS_ROOT = (
    HERE
    /
    "results"
)


DEFAULT_CASE_IDS = [
    "sales_group_natural_fr",
    "sales_guardrail_missing_margin_en",
]


def resolve_latest_run(
    results_root: Path,
) -> Path:
    results_root = (
        results_root
        .resolve()
    )


    latest_file = (
        results_root
        /
        "latest_run.txt"
    )


    if not latest_file.exists():
        raise FileNotFoundError(
            (
                "latest_run.txt introuvable : "
                f"{latest_file}"
            )
        )


    run_dir_text = (
        latest_file.read_text(
            encoding="utf-8"
        )
        .strip()
    )


    if not run_dir_text:
        raise RuntimeError(
            "latest_run.txt est vide."
        )


    run_dir = Path(
        run_dir_text
    )


    if not run_dir.is_absolute():
        run_dir = (
            results_root
            /
            run_dir
        ).resolve()


    if not run_dir.exists():
        raise FileNotFoundError(
            (
                "Le dossier du dernier run "
                "n'existe pas : "
                f"{run_dir}"
            )
        )


    return run_dir


def load_json(
    path: Path,
) -> dict[
    str,
    Any,
]:
    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


def print_lines(
    title: str,
    values: list[
        Any
    ] | None,
) -> None:
    print(
        f"\n{title}:"
    )


    if not values:
        print(
            "  - none"
        )

        return


    for value in values:
        print(
            f"  - {value}"
        )


def inspect_case(
    *,
    run_dir: Path,
    case_id: str,
) -> None:
    raw_dir = (
        run_dir
        /
        "raw"
    )


    matches = sorted(
        raw_dir.glob(
            (
                f"{case_id}"
                "__r*.json"
            )
        )
    )


    if not matches:
        print()
        print(
            "=" * 72
        )

        print(
            f"CASE : {case_id}"
        )

        print(
            "Aucun JSON brut trouvé."
        )

        return


    for path in matches:
        report = (
            load_json(
                path
            )
        )


        planner = (
            report.get(
                "planner"
            )
            or {}
        )


        planner_items = (
            planner.get(
                "items"
            )
            or []
        )


        print()
        print(
            "=" * 72
        )

        print(
            f"CASE : {case_id}"
        )

        print(
            f"RAW  : {path}"
        )

        print(
            (
                "Planner version : "
                f"{planner.get('planner_rule_version')}"
            )
        )

        print(
            (
                "Attempts        : "
                f"{planner.get('attempt_count')}"
            )
        )

        print(
            (
                "Retries         : "
                f"{planner.get('retry_count')}"
            )
        )

        print(
            (
                "Validated       : "
                f"{planner.get('validated_count')}"
            )
        )

        print(
            (
                "Blocked         : "
                f"{planner.get('blocked_count')}"
            )
        )

        print(
            (
                "Rejected        : "
                f"{planner.get('rejected_count')}"
            )
        )

        print(
            (
                "Executed        : "
                f"{report.get('executed_count')}"
            )
        )


        print_lines(
            "Retry feedback",
            planner.get(
                "retry_feedback"
            ),
        )


        for index, item in enumerate(
            planner_items,
            start=1,
        ):
            raw_proposal = (
                item.get(
                    "raw_proposal"
                )
                or {}
            )

            proposal = (
                item.get(
                    "proposal"
                )
                or {}
            )

            contract = (
                item.get(
                    "contract"
                )
            )


            print()
            print(
                f"--- Planner item {index} ---"
            )

            print(
                (
                    "validation_status : "
                    f"{item.get('validation_status')}"
                )
            )

            print(
                (
                    "raw decision      : "
                    f"{raw_proposal.get('decision')}"
                )
            )

            print(
                (
                    "raw family        : "
                    f"{raw_proposal.get('family')}"
                )
            )

            print(
                (
                    "proposal family   : "
                    f"{proposal.get('family')}"
                )
            )


            fields = [
                "dataset_id",
                "analytical_grain",
                "x_column",
                "y_column",
                "group_column",
                "value_column",
                "time_column",
                "dimension_column",
                "entity_column",
                "aggregation_function",
            ]


            print(
                "\nRaw proposal roles:"
            )

            for field in fields:
                print(
                    (
                        f"  {field:<22}"
                        f"{raw_proposal.get(field)}"
                    )
                )


            print(
                "\nCanonical proposal roles:"
            )

            for field in fields:
                print(
                    (
                        f"  {field:<22}"
                        f"{proposal.get(field)}"
                    )
                )


            print_lines(
                "Item errors",
                item.get(
                    "errors"
                ),
            )

            print_lines(
                "Item warnings",
                item.get(
                    "warnings"
                ),
            )

            print_lines(
                "Item normalizations",
                item.get(
                    "normalizations"
                ),
            )

            print_lines(
                "Proposal blockers",
                proposal.get(
                    "blockers"
                ),
            )

            print_lines(
                "Raw blockers",
                raw_proposal.get(
                    "blockers"
                ),
            )


            if contract:
                print(
                    "\nContract:"
                )

                print(
                    (
                        "  status  : "
                        f"{contract.get('status')}"
                    )
                )

                print(
                    (
                        "  family  : "
                        f"{contract.get('family')}"
                    )
                )

                print(
                    (
                        "  bindings: "
                        f"{contract.get('bindings')}"
                    )
                )

                print_lines(
                    "Contract blockers",
                    contract.get(
                        "blockers"
                    ),
                )


        native_items = (
            report.get(
                "items"
            )
            or []
        )


        print(
            "\nNative pipeline items:"
        )


        if not native_items:
            print(
                "  - none"
            )


        for index, item in enumerate(
            native_items,
            start=1,
        ):
            native = (
                item.get(
                    "native_tool"
                )
                or {}
            )

            print(
                (
                    f"  {index}. "
                    f"family={item.get('family')} "
                    f"status={item.get('status')} "
                    f"tool={native.get('requested_tool')} "
                    f"args={native.get('requested_arguments')}"
                )
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect the raw planner/tool traces "
            "for selected DataLens eval cases."
        )
    )


    parser.add_argument(
        "--results-root",
        type=Path,
        default=(
            DEFAULT_RESULTS_ROOT
        ),
    )


    parser.add_argument(
        "--case",
        action="append",
        dest="case_ids",
        default=[],
    )


    return parser.parse_args()


def main() -> None:
    args = (
        parse_args()
    )


    run_dir = (
        resolve_latest_run(
            args.results_root
        )
    )


    case_ids = (
        args.case_ids
        or
        DEFAULT_CASE_IDS
    )


    print(
        "=== DataLens eval failure diagnostics ==="
    )

    print(
        f"Run : {run_dir}"
    )


    for case_id in (
        case_ids
    ):
        inspect_case(
            run_dir=(
                run_dir
            ),
            case_id=(
                case_id
            ),
        )


if __name__ == "__main__":
    main()
