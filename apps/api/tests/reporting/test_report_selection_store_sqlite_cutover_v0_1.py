from __future__ import annotations

import json
import os
import sqlite3
import tempfile

from pathlib import Path


_TEMP = tempfile.TemporaryDirectory(
    prefix=
        "datalens-report-selection-cutover-"
)

_ROOT = Path(
    _TEMP.name
)

_DATABASE = (
    _ROOT
    /
    "datalens.sqlite3"
)

_SELECTION_PATH = (
    _ROOT
    /
    "report_selection.json"
)


os.environ[
    "DATALENS_SQLITE_PATH"
] = str(
    _DATABASE
)

os.environ[
    "DATALENS_REPORT_SELECTION_STORE_PATH"
] = str(
    _SELECTION_PATH
)


from app.reporting.report_selection_sqlite_store import (
    report_selection_store_scope,
)

from app.reporting.report_selection_store import (
    REPORT_SELECTION_STORE_RULE_VERSION,
    _read_payload,
    _write_payload,
)


def initial_payload() -> dict:
    return {
        "rule_version":
            REPORT_SELECTION_STORE_RULE_VERSION,

        "workflows": {
            "prep:test": {
                "revision":
                    4,

                "analyses": [
                    {
                        "analysis_id":
                            "analysis:one",

                        "report_order":
                            1,

                        "added_at_utc":
                            "2026-08-25T10:00:00+00:00",
                    },
                    {
                        "analysis_id":
                            "analysis:two",

                        "report_order":
                            2,

                        "added_at_utc":
                            "2026-08-25T10:01:00+00:00",
                    },
                ],
            },
        },
    }


def sqlite_row_count() -> int:
    connection = sqlite3.connect(
        str(
            _DATABASE
        )
    )

    try:
        return int(
            connection.execute(
                """
                SELECT COUNT(*)
                FROM report_selection_workflows
                WHERE store_root = ?
                """,
                (
                    report_selection_store_scope(
                        _SELECTION_PATH
                    ),
                ),
            ).fetchone()[0]
        )

    finally:
        connection.close()


def test_legacy_import() -> None:
    payload = initial_payload()


    _SELECTION_PATH.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


    restored = (
        _read_payload()
    )


    assert restored == payload

    assert (
        sqlite_row_count()
        ==
        1
    )


    print(
        "[PASS] active store imports legacy JSON into SQLite"
    )


def test_sqlite_becomes_authoritative() -> None:
    _SELECTION_PATH.write_text(
        json.dumps(
            {
                "rule_version":
                    REPORT_SELECTION_STORE_RULE_VERSION,

                "workflows":
                    {},
            }
        ),
        encoding="utf-8",
    )


    restored = (
        _read_payload()
    )


    assert (
        "prep:test"
        in
        restored[
            "workflows"
        ]
    )


    print(
        "[PASS] legacy JSON cannot overwrite initialized SQLite"
    )


def test_write_updates_sqlite_only() -> None:
    before_json = (
        _SELECTION_PATH.read_bytes()
    )


    updated = initial_payload()

    updated[
        "workflows"
    ][
        "prep:test"
    ][
        "revision"
    ] = 5

    updated[
        "workflows"
    ][
        "prep:test"
    ][
        "analyses"
    ] = [
        updated[
            "workflows"
        ][
            "prep:test"
        ][
            "analyses"
        ][
            1
        ]
    ]

    updated[
        "workflows"
    ][
        "prep:test"
    ][
        "analyses"
    ][
        0
    ][
        "report_order"
    ] = 1


    _write_payload(
        updated
    )


    restored = (
        _read_payload()
    )


    assert restored == updated


    assert (
        _SELECTION_PATH.read_bytes()
        ==
        before_json
    )


    print(
        "[PASS] active writes commit SQLite only"
    )

    print(
        "[PASS] legacy report_selection.json remains unchanged"
    )


def test_fresh_read() -> None:
    # _read_payload has no process-owned cache: a second read
    # is therefore a fresh SQLite reconstruction.
    restored = (
        _read_payload()
    )


    assert (
        restored[
            "workflows"
        ][
            "prep:test"
        ][
            "revision"
        ]
        ==
        5
    )


    print(
        "[PASS] fresh read restores SQLite state"
    )


def test_rule_version() -> None:
    assert (
        REPORT_SELECTION_STORE_RULE_VERSION
        ==
        "report_selection_store_v0.1"
    )


    print(
        "[PASS] public ReportSelection rule version preserved"
    )


def main() -> None:
    print()

    print(
        "=== DATALENS REPORT SELECTION SQLITE CUTOVER v0.1 ==="
    )

    print()


    test_legacy_import()

    test_sqlite_becomes_authoritative()

    test_write_updates_sqlite_only()

    test_fresh_read()

    test_rule_version()


    print()

    print(
        "PASS - Report Selection SQLite Cutover v0.1"
    )


if __name__ == "__main__":
    main()
