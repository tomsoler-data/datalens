from __future__ import annotations


import argparse
import json
import shutil

from datetime import (
    datetime,
    timezone,
)

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


def parse_concatenated_json_objects(
    text: str,
) -> list[
    dict[
        str,
        Any,
    ]
]:
    decoder = (
        json.JSONDecoder()
    )


    objects: list[
        dict[
            str,
            Any,
        ]
    ] = []


    index = 0
    length = len(
        text
    )


    while index < length:
        while (
            index < length
            and
            text[
                index
            ].isspace()
        ):
            index += 1


        if index >= length:
            break


        value, end = (
            decoder.raw_decode(
                text,
                index,
            )
        )


        if not isinstance(
            value,
            dict,
        ):
            raise ValueError(
                (
                    "Top-level JSON value is not "
                    f"an object at character {index}."
                )
            )


        objects.append(
            value
        )


        index = end


    return objects


def is_real_ai_trace(
    value: dict[
        str,
        Any,
    ],
) -> bool:
    trace_id = (
        value.get(
            "trace_id"
        )
    )


    version = (
        value.get(
            "trace_rule_version"
        )
    )


    return bool(
        isinstance(
            trace_id,
            str,
        )
        and
        trace_id.startswith(
            "ai:"
        )
        and
        isinstance(
            version,
            str,
        )
        and
        version.startswith(
            "ai_trace_"
        )
    )


def make_backup_path(
    path: Path,
) -> Path:
    timestamp = (
        datetime.now(
            timezone.utc
        )
        .strftime(
            "%Y%m%dT%H%M%S_%fZ"
        )
    )


    return path.with_name(
        (
            path.stem
            +
            ".backup_"
            +
            timestamp
            +
            path.suffix
        )
    )


def main() -> None:
    parser = (
        argparse.ArgumentParser(
            description=(
                "Repair the local DataLens AI trace file "
                "into strict one-object-per-line JSONL."
            )
        )
    )


    parser.add_argument(
        "--path",
        type=Path,
        default=DEFAULT_TRACE_PATH,
    )


    args = (
        parser.parse_args()
    )


    path = (
        args.path
        .resolve()
    )


    print(
        "=== DataLens AI trace JSONL repair ==="
    )

    print(
        f"Path              : {path}"
    )


    if not path.exists():
        raise FileNotFoundError(
            f"Trace file not found: {path}"
        )


    original = (
        path.read_text(
            encoding="utf-8"
        )
    )


    objects = (
        parse_concatenated_json_objects(
            original
        )
    )


    real_traces = [
        value

        for value
        in objects

        if is_real_ai_trace(
            value
        )
    ]


    excluded = (
        len(
            objects
        )
        -
        len(
            real_traces
        )
    )


    if not real_traces:
        raise RuntimeError(
            (
                "No real DataLens AI trace was found. "
                "The original file was left untouched."
            )
        )


    backup = (
        make_backup_path(
            path
        )
    )


    shutil.copy2(
        path,
        backup,
    )


    repaired = "".join(
        (
            json.dumps(
                trace,
                ensure_ascii=False,
                separators=(
                    ",",
                    ":",
                ),
            )
            +
            "\n"
        )

        for trace
        in real_traces
    )


    path.write_text(
        repaired,
        encoding="utf-8",
        newline="\n",
    )


    # Final strict validation.
    repaired_lines = [
        line

        for line
        in path.read_text(
            encoding="utf-8"
        ).splitlines()

        if line.strip()
    ]


    validated = [
        json.loads(
            line
        )

        for line
        in repaired_lines
    ]


    if len(
        validated
    ) != len(
        real_traces
    ):
        raise RuntimeError(
            "Post-repair validation count mismatch."
        )


    print(
        f"Top-level objects : {len(objects)}"
    )

    print(
        f"Real AI traces    : {len(real_traces)}"
    )

    print(
        f"Excluded objects  : {excluded}"
    )

    print(
        f"Backup            : {backup}"
    )

    print(
        "Strict JSONL      : OK"
    )

    print(
        (
            "Latest trace_id   : "
            f"{real_traces[-1].get('trace_id')}"
        )
    )


if __name__ == "__main__":
    main()
