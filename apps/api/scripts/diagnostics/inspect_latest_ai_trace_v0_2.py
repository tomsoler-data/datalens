from __future__ import annotations


import argparse
import json

from pathlib import Path
from typing import Any


DEFAULT_TRACE_PATH = (
    Path(__file__)
    .resolve()
    .parent
    /
    "data"
    /
    "observability"
    /
    "ai_traces.jsonl"
)


def load_latest_trace(
    path: Path,
) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Trace file not found: {path}"
        )


    lines = [
        line
        for line
        in path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]


    if not lines:
        raise RuntimeError(
            f"Trace file is empty: {path}"
        )


    return json.loads(
        lines[-1]
    )


def print_list(
    title: str,
    values: list[Any] | None,
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Display the latest DataLens local AI observability trace."
        )
    )


    parser.add_argument(
        "--path",
        type=Path,
        default=DEFAULT_TRACE_PATH,
    )


    args = parser.parse_args()


    trace = load_latest_trace(
        args.path.resolve()
    )


    print(
        "=== DataLens latest AI trace ==="
    )

    print(
        f"Trace ID        : {trace.get('trace_id')}"
    )

    print(
        f"Created UTC     : {trace.get('created_at_utc')}"
    )

    print(
        f"Trace version   : {trace.get('trace_rule_version')}"
    )

    print(
        f"Objective       : {trace.get('objective')}"
    )

    print(
        f"Objective SHA256: {trace.get('objective_sha256')}"
    )


    print(
        "\nDatasets:"
    )


    for dataset in (
        trace.get(
            "datasets",
            []
        )
        or []
    ):
        print(
            (
                "  - "
                f"{dataset.get('dataset_id')} "
                f"| {dataset.get('filename')} "
                f"| rows={dataset.get('row_count')} "
                f"| columns={dataset.get('column_count')}"
            )
        )


    planner = (
        trace.get(
            "planner",
            {}
        )
        or {}
    )


    print(
        "\nPlanner:"
    )

    print(
        f"  status         : {planner.get('status')}"
    )

    print(
        f"  model          : {planner.get('model')}"
    )

    print(
        (
            "  rule version   : "
            f"{planner.get('planner_rule_version')}"
        )
    )

    print(
        f"  attempts       : {planner.get('attempt_count')}"
    )

    print(
        f"  retries        : {planner.get('retry_count')}"
    )

    print(
        (
            "  normalizations : "
            f"{planner.get('normalization_count')}"
        )
    )


    print_list(
        "Planner retry feedback",
        planner.get(
            "retry_feedback",
            []
        ),
    )


    for index, item in enumerate(
        planner.get(
            "items",
            []
        )
        or [],
        start=1,
    ):
        raw = (
            item.get(
                "raw_proposal"
            )
            or {}
        )


        canonical = (
            item.get(
                "canonical_proposal"
            )
            or {}
        )


        contract = (
            item.get(
                "contract"
            )
            or {}
        )


        print(
            f"\nPlanner item {index}:"
        )

        print(
            (
                "  validation     : "
                f"{item.get('validation_status')}"
            )
        )

        print(
            (
                "  raw decision   : "
                f"{raw.get('decision')}"
            )
        )

        print(
            (
                "  raw family     : "
                f"{raw.get('family')}"
            )
        )

        print(
            (
                "  raw dataset    : "
                f"{raw.get('dataset_id')}"
            )
        )

        print(
            (
                "  canonical fam. : "
                f"{canonical.get('family')}"
            )
        )

        print(
            (
                "  canonical data.: "
                f"{canonical.get('dataset_id')}"
            )
        )


        canonical_roles = {
            key:
                canonical.get(
                    key
                )

            for key in [
                "x_column",
                "y_column",
                "group_column",
                "value_column",
                "time_column",
                "dimension_column",
                "entity_column",
                "aggregation_function",
            ]

            if canonical.get(
                key
            )
            not in (
                None,
                "",
                "none",
            )
        }


        print(
            (
                "  canonical roles: "
                f"{canonical_roles}"
            )
        )


        print_list(
            "  Normalizations",
            item.get(
                "normalizations",
                []
            ),
        )


        print_list(
            "  Errors",
            item.get(
                "errors",
                []
            ),
        )


        if contract:
            print(
                (
                    "  contract status: "
                    f"{contract.get('status')}"
                )
            )

            print(
                (
                    "  contract data  : "
                    f"{contract.get('required_dataset_filenames')}"
                )
            )

            print(
                (
                    "  bindings       : "
                    f"{contract.get('bindings')}"
                )
            )


            print_list(
                "  Contract blockers",
                contract.get(
                    "blockers",
                    []
                ),
            )


    native = (
        trace.get(
            "native_pipeline",
            {}
        )
        or {}
    )


    print(
        "\nNative pipeline:"
    )

    print(
        f"  status         : {native.get('status')}"
    )

    print(
        (
            "  pipeline rule  : "
            f"{native.get('pipeline_rule_version')}"
        )
    )

    print(
        f"  planner model  : {native.get('planner_model')}"
    )

    print(
        f"  tool model     : {native.get('tool_model')}"
    )

    print(
        f"  executed count : {native.get('executed_count')}"
    )


    for index, item in enumerate(
        native.get(
            "items",
            []
        )
        or [],
        start=1,
    ):
        tool_call = (
            item.get(
                "tool_call"
            )
            or {}
        )


        execution = (
            item.get(
                "execution"
            )
            or {}
        )


        print(
            f"\nNative item {index}:"
        )

        print(
            (
                "  family          : "
                f"{item.get('family')}"
            )
        )

        print(
            (
                "  pipeline status : "
                f"{item.get('pipeline_status')}"
            )
        )

        print(
            (
                "  expected tool   : "
                f"{tool_call.get('expected_tool')}"
            )
        )

        print(
            (
                "  requested tool  : "
                f"{tool_call.get('requested_tool')}"
            )
        )

        print(
            (
                "  tool validation : "
                f"{tool_call.get('validation_status')}"
            )
        )

        print(
            (
                "  tool attempts   : "
                f"{tool_call.get('attempt_count')}"
            )
        )

        print(
            (
                "  tool retries    : "
                f"{tool_call.get('retry_count')}"
            )
        )

        print(
            (
                "  tool arguments  : "
                f"{tool_call.get('requested_arguments')}"
            )
        )

        print(
            (
                "  execution       : "
                f"{execution.get('execution_status')}"
            )
        )

        print(
            (
                "  dataset         : "
                f"{execution.get('dataset_filename')}"
            )
        )

        print(
            (
                "  chart type      : "
                f"{execution.get('chart_type')}"
            )
        )

        print(
            (
                "  execution rule  : "
                f"{execution.get('execution_rule_version')}"
            )
        )


    timings = (
        trace.get(
            "timings",
            {}
        )
        or {}
    )


    print(
        "\nTimings:"
    )

    print(
        (
            "  ingestion       : "
            f"{timings.get('ingestion_ms')} ms"
        )
    )

    print(
        (
            "  planner         : "
            f"{timings.get('planner_ms')} ms"
        )
    )

    print(
        (
            "  native pipeline : "
            f"{timings.get('native_pipeline_ms')} ms"
        )
    )

    print(
        (
            "  total           : "
            f"{timings.get('total_ms')} ms"
        )
    )


    privacy = (
        trace.get(
            "privacy",
            {}
        )
        or {}
    )


    print(
        "\nPrivacy:"
    )

    print(
        (
            "  raw dataset rows : "
            f"{privacy.get('contains_raw_dataset_rows')}"
        )
    )

    print(
        (
            "  uploaded contents: "
            f"{privacy.get('contains_uploaded_file_contents')}"
        )
    )

    print(
        (
            "  document chunks  : "
            f"{privacy.get('contains_document_chunks')}"
        )
    )


if __name__ == "__main__":
    main()
