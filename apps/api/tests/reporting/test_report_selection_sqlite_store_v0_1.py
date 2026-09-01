from __future__ import annotations


import json
import os
import sqlite3
import tempfile

from pathlib import (
    Path,
)


_TEMP_DIRECTORY = (
    tempfile.TemporaryDirectory(
        prefix=
            "datalens-report-selection-sqlite-"
    )
)

_ROOT = Path(
    _TEMP_DIRECTORY.name
)

_DATABASE_PATH = (
    _ROOT
    /
    "datalens.sqlite3"
)

_STORE_A = (
    _ROOT
    /
    "report_selection_a.json"
)

_STORE_B = (
    _ROOT
    /
    "report_selection_b.json"
)


os.environ[
    "DATALENS_SQLITE_PATH"
] = str(
    _DATABASE_PATH
)


from app.persistence.sqlite_database import (
    SQLITE_SCHEMA_VERSION,
    sqlite_schema_version,
)

from app.reporting.report_selection_sqlite_store import (
    REPORT_SELECTION_SQLITE_STORE_VERSION,
    import_legacy_report_selection_if_needed,
    load_report_selection_sqlite_payload,
    replace_report_selection_sqlite_payload,
    report_selection_sqlite_is_initialized,
    report_selection_store_scope,
)


RULE_VERSION = (
    "report_selection_store_v0.1"
)


def payload(
    *,
    workflow_id: str,
    suffix: str,
    revision: int = 2,
) -> dict:
    return {
        "rule_version":
            RULE_VERSION,

        "workflows": {
            workflow_id: {
                "revision":
                    revision,

                "analyses": [
                    {
                        "analysis_id":
                            f"analysis:{suffix}:1",

                        "report_order":
                            1,

                        "added_at_utc":
                            "2026-08-25T10:00:00+00:00",
                    },
                    {
                        "analysis_id":
                            f"analysis:{suffix}:2",

                        "report_order":
                            2,

                        "added_at_utc":
                            "2026-08-25T10:01:00+00:00",
                    },
                ],
            },
        },
    }


def test_schema_v4() -> None:
    runtime_schema_version = (
        sqlite_schema_version()
    )


    assert (
        runtime_schema_version
        ==
        SQLITE_SCHEMA_VERSION
    )


    assert (
        SQLITE_SCHEMA_VERSION
        >=
        4
    )


    connection = sqlite3.connect(
        str(
            _DATABASE_PATH
        )
    )

    try:
        tables = {
            row[
                0
            ]

            for row
            in connection.execute(
                """
                SELECT name
                FROM sqlite_master
                WHERE type = 'table'
                """
            )
        }

    finally:
        connection.close()


    assert (
        "report_selection_workflows"
        in tables
    )

    assert (
        "report_selection_store_state"
        in tables
    )


    print(
        "[PASS] current SQLite schema preserves v4 ReportSelection tables"
    )


def test_legacy_import() -> None:
    _STORE_A.write_text(
        json.dumps(
            payload(
                workflow_id=
                    "prep:legacy",

                suffix=
                    "legacy",
            ),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


    imported = (
        import_legacy_report_selection_if_needed(
            store_path=
                _STORE_A,

            fallback_rule_version=
                RULE_VERSION,
        )
    )


    assert imported is True


    restored = (
        load_report_selection_sqlite_payload(
            store_path=
                _STORE_A,

            fallback_rule_version=
                RULE_VERSION,
        )
    )


    assert (
        restored
        ==
        payload(
            workflow_id=
                "prep:legacy",

            suffix=
                "legacy",
        )
    )


    print(
        "[PASS] legacy report_selection.json -> SQLite"
    )


def test_import_once() -> None:
    _STORE_A.write_text(
        json.dumps(
            payload(
                workflow_id=
                    "prep:changed",

                suffix=
                    "changed",
            )
        ),
        encoding="utf-8",
    )


    imported = (
        import_legacy_report_selection_if_needed(
            store_path=
                _STORE_A,

            fallback_rule_version=
                RULE_VERSION,
        )
    )


    assert imported is False


    restored = (
        load_report_selection_sqlite_payload(
            store_path=
                _STORE_A,

            fallback_rule_version=
                RULE_VERSION,
        )
    )


    assert (
        "prep:legacy"
        in
        restored[
            "workflows"
        ]
    )

    assert (
        "prep:changed"
        not in
        restored[
            "workflows"
        ]
    )


    print(
        "[PASS] legacy selection cannot re-import"
    )


def test_store_path_isolation() -> None:
    replace_report_selection_sqlite_payload(
        store_path=
            _STORE_B,

        payload=
            payload(
                workflow_id=
                    "prep:legacy",

                suffix=
                    "store-b",

                revision=
                    5,
            ),

        fallback_rule_version=
            RULE_VERSION,
    )


    a = (
        load_report_selection_sqlite_payload(
            store_path=
                _STORE_A,

            fallback_rule_version=
                RULE_VERSION,
        )
    )

    b = (
        load_report_selection_sqlite_payload(
            store_path=
                _STORE_B,

            fallback_rule_version=
                RULE_VERSION,
        )
    )


    assert (
        a[
            "workflows"
        ][
            "prep:legacy"
        ][
            "revision"
        ]
        ==
        2
    )


    assert (
        b[
            "workflows"
        ][
            "prep:legacy"
        ][
            "revision"
        ]
        ==
        5
    )


    print(
        "[PASS] report-selection store_path isolation"
    )


def test_transactional_scope_replacement() -> None:
    replacement = {
        "rule_version":
            RULE_VERSION,

        "workflows": {
            "prep:new": {
                "revision":
                    7,

                "analyses": [
                    {
                        "analysis_id":
                            "analysis:new",

                        "report_order":
                            1,

                        "added_at_utc":
                            "2026-08-25T12:00:00+00:00",
                    },
                ],
            },
        },
    }


    replace_report_selection_sqlite_payload(
        store_path=
            _STORE_A,

        payload=
            replacement,

        fallback_rule_version=
            RULE_VERSION,
    )


    restored = (
        load_report_selection_sqlite_payload(
            store_path=
                _STORE_A,

            fallback_rule_version=
                RULE_VERSION,
        )
    )


    assert (
        restored
        ==
        replacement
    )


    print(
        "[PASS] transactional ReportSelection replacement"
    )


def test_sql_row_state() -> None:
    connection = sqlite3.connect(
        str(
            _DATABASE_PATH
        )
    )

    try:
        row = connection.execute(
            """
            SELECT
                workflow_id,
                revision
            FROM report_selection_workflows
            WHERE store_root = ?
            """,
            (
                report_selection_store_scope(
                    _STORE_A
                ),
            ),
        ).fetchone()

    finally:
        connection.close()


    assert row == (
        "prep:new",
        7,
    )


    assert (
        report_selection_sqlite_is_initialized(
            store_path=
                _STORE_A
        )
        is True
    )


    print(
        "[PASS] ReportSelection revision persisted in SQLite"
    )


def test_version() -> None:
    assert (
        REPORT_SELECTION_SQLITE_STORE_VERSION
        ==
        "report_selection_sqlite_store_v0.1"
    )


    print(
        "[PASS] ReportSelection SQLite repository version"
    )


def main() -> None:
    print()

    print(
        "=== DATALENS REPORT SELECTION SQLITE STORE v0.1 ==="
    )

    print()


    test_schema_v4()

    test_legacy_import()

    test_import_once()

    test_store_path_isolation()

    test_transactional_scope_replacement()

    test_sql_row_state()

    test_version()


    print()

    print(
        "PASS - Report Selection SQLite Store v0.1"
    )


if __name__ == "__main__":
    main()
