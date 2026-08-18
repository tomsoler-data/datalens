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
    "hard_exact_net_revenue_distribution",
    "hard_time_fiscal_period_exact",
    "hard_ambiguous_performance_region",
    "hard_multi_dataset_online_explicit",
    "hard_multi_dataset_retail_explicit",
    "hard_multi_dataset_unspecified",
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


def print_list(
    title: str,
    values: Any,
) -> None:
    print(
        f"\n{title}:"
    )


    if not values:
        print(
            "  - none"
        )

        return


    if not isinstance(
        values,
        list,
    ):
        values = [
            values,
        ]


    for value in values:
        print(
            f"  - {value}"
        )


def print_mapping(
    title: str,
    mapping: Any,
) -> None:
    print(
        f"\n{title}:"
    )


    if not mapping:
        print(
            "  - none"
        )

        return


    if not isinstance(
        mapping,
        dict,
    ):
        print(
            f"  {mapping}"
        )

        return


    for key, value in (
        mapping.items()
    ):
        print(
            f"  {key:<24} {value}"
        )


def maybe_print_catalog(
    report: dict[
        str,
        Any,
    ],
) -> None:
    catalog = (
        report.get(
            "catalog"
        )
        or
        report.get(
            "planner_catalog"
        )
        or
        {}
    )


    datasets = (
        catalog.get(
            "datasets"
        )
        if isinstance(
            catalog,
            dict,
        )
        else None
    )


    if not datasets:
        return


    print(
        "\nPlanner catalog datasets:"
    )


    for dataset in (
        datasets
    ):
        if not isinstance(
            dataset,
            dict,
        ):
            print(
                f"  - {dataset}"
            )

            continue


        print(
            (
                "  - "
                f"id={dataset.get('dataset_id')} "
                f"filename={dataset.get('filename')} "
                f"rows={dataset.get('row_count')}"
            )
        )


        columns = (
            dataset.get(
                "columns"
            )
            or []
        )


        names = [
            str(
                column.get(
                    "name"
                )
            )

            for column
            in columns

            if isinstance(
                column,
                dict,
            )
        ]


        if names:
            print(
                (
                    "    columns="
                    f"{', '.join(names)}"
                )
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


    print()
    print(
        "=" * 88
    )

    print(
        f"CASE : {case_id}"
    )


    if not matches:
        print(
            "Aucun JSON brut trouvé."
        )

        return


    for path in (
        matches
    ):
        report = (
            load_json(
                path
            )
        )


        print(
            f"RAW  : {path}"
        )


        print(
            "\nTop-level keys:"
        )

        print(
            "  "
            +
            ", ".join(
                sorted(
                    report.keys()
                )
            )
        )


        maybe_print_catalog(
            report
        )


        planner = (
            report.get(
                "planner"
            )
            or {}
        )


        print(
            "\nPlanner summary:"
        )

        for key in [
            "planner_rule_version",
            "attempt_count",
            "retry_count",
            "retry_triggered",
            "validated_count",
            "blocked_count",
            "ambiguous_count",
            "rejected_count",
            "normalization_count",
            "normalization_applied",
        ]:
            print(
                (
                    f"  {key:<24}"
                    f"{planner.get(key)}"
                )
            )


        print_list(
            "Retry feedback",
            planner.get(
                "retry_feedback"
            ),
        )


        planner_items = (
            planner.get(
                "items"
            )
            or []
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
                or {}
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


            role_fields = [
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
                "ranking_order",
                "ranking_limit",
                "window_operation",
                "window_size",
            ]


            print_mapping(
                "Raw proposal",
                {
                    field:
                        raw_proposal.get(
                            field
                        )

                    for field
                    in role_fields
                },
            )


            print_mapping(
                "Canonical proposal",
                {
                    field:
                        proposal.get(
                            field
                        )

                    for field
                    in role_fields
                },
            )


            print_list(
                "Errors",
                item.get(
                    "errors"
                ),
            )

            print_list(
                "Warnings",
                item.get(
                    "warnings"
                ),
            )

            print_list(
                "Normalizations",
                item.get(
                    "normalizations"
                ),
            )

            print_list(
                "Raw blockers",
                raw_proposal.get(
                    "blockers"
                ),
            )

            print_list(
                "Canonical blockers",
                proposal.get(
                    "blockers"
                ),
            )


            if contract:
                print(
                    "\nContract:"
                )

                for key in [
                    "status",
                    "family",
                    "required_dataset_ids",
                    "required_dataset_filenames",
                    "analytical_grain",
                ]:
                    print(
                        (
                            f"  {key:<28}"
                            f"{contract.get(key)}"
                        )
                    )


                print_list(
                    "Contract bindings",
                    contract.get(
                        "bindings"
                    ),
                )

                print_list(
                    "Contract blockers",
                    contract.get(
                        "blockers"
                    ),
                )

                print_list(
                    "Contract reasons",
                    contract.get(
                        "reasons"
                    ),
                )


        pipeline_items = (
            report.get(
                "items"
            )
            or []
        )


        print(
            "\nNative pipeline items:"
        )


        if not pipeline_items:
            print(
                "  - none"
            )


        for index, item in enumerate(
            pipeline_items,
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
                    f"status={item.get('status')} "
                    f"family={item.get('family')} "
                    f"tool={native.get('requested_tool')} "
                    f"args={native.get('requested_arguments')}"
                )
            )


def parse_args() -> argparse.Namespace:
    parser = (
        argparse.ArgumentParser(
            description=(
                "Inspect raw traces for the failing "
                "DataLens hard semantic v0.4 cases."
            )
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


    return (
        parser.parse_args()
    )


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
        "=== DataLens hard-case diagnostics ==="
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
