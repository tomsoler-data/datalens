from __future__ import annotations


import json

from pathlib import Path


TRACE_PATH = (
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


def preview(
    text: str,
    limit: int = 240,
) -> str:
    value = repr(
        text[
            :limit
        ]
    )


    if len(
        text
    ) > limit:
        value += " ..."


    return value


def main() -> None:
    path = (
        TRACE_PATH
        .resolve()
    )


    print(
        "=== DataLens AI trace JSONL diagnostic ==="
    )

    print(
        f"Path          : {path}"
    )

    print(
        f"Exists        : {path.exists()}"
    )


    if not path.exists():
        return


    raw = (
        path.read_bytes()
    )


    print(
        f"Size          : {len(raw)} bytes"
    )

    print(
        (
            "First 32 bytes: "
            f"{raw[:32].hex(' ')}"
        )
    )


    try:
        text = raw.decode(
            "utf-8"
        )

        encoding = (
            "utf-8"
        )

    except UnicodeDecodeError as error:
        print(
            (
                "UTF-8 decode  : FAIL "
                f"({error})"
            )
        )

        return


    print(
        f"UTF-8 decode  : OK ({encoding})"
    )


    physical_lines = (
        text.splitlines()
    )


    non_empty = [
        (
            index,
            line,
        )

        for index, line
        in enumerate(
            physical_lines,
            start=1,
        )

        if line.strip()
    ]


    print(
        (
            "Physical lines: "
            f"{len(physical_lines)}"
        )
    )

    print(
        (
            "Non-empty     : "
            f"{len(non_empty)}"
        )
    )


    if not non_empty:
        print(
            "No non-empty JSONL records."
        )

        return


    print(
        "\nLast non-empty line previews:"
    )


    for (
        line_number,
        line,
    ) in non_empty[
        -3:
    ]:
        print(
            (
                f"  line {line_number}: "
                f"{preview(line)}"
            )
        )


    valid_count = 0
    invalid: list[
        tuple[
            int,
            str,
            str,
        ]
    ] = []


    for (
        line_number,
        line,
    ) in non_empty:
        try:
            value = (
                json.loads(
                    line
                )
            )

            if not isinstance(
                value,
                dict,
            ):
                invalid.append(
                    (
                        line_number,
                        (
                            "JSON value is "
                            f"{type(value).__name__}, "
                            "expected object"
                        ),
                        preview(
                            line
                        ),
                    )
                )

                continue


            valid_count += 1


        except json.JSONDecodeError as error:
            invalid.append(
                (
                    line_number,
                    (
                        f"{error.msg} "
                        f"at column {error.colno}"
                    ),
                    preview(
                        line
                    ),
                )
            )


    print()
    print(
        f"Valid records : {valid_count}"
    )

    print(
        f"Invalid lines : {len(invalid)}"
    )


    if invalid:
        print(
            "\nInvalid JSONL lines:"
        )


        for (
            line_number,
            error,
            line_preview,
        ) in invalid[
            -10:
        ]:
            print(
                f"  line {line_number}: {error}"
            )

            print(
                f"    {line_preview}"
            )


    latest_line_number, latest_line = (
        non_empty[
            -1
        ]
    )


    print()
    print(
        f"Latest record line: {latest_line_number}"
    )


    try:
        latest = (
            json.loads(
                latest_line
            )
        )

        print(
            "Latest JSON      : VALID"
        )

        print(
            (
                "Latest trace_id : "
                f"{latest.get('trace_id')}"
            )
        )

        print(
            (
                "Latest objective: "
                f"{latest.get('objective')}"
            )
        )


    except json.JSONDecodeError as error:
        print(
            (
                "Latest JSON      : INVALID "
                f"({error})"
            )
        )


if __name__ == "__main__":
    main()
