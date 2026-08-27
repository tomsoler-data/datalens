from __future__ import annotations


import os
import sqlite3
import tempfile

from pathlib import (
    Path,
)


_TEMP_DIRECTORY = (
    tempfile.TemporaryDirectory(
        prefix=
            "datalens-analysis-artifact-index-"
    )
)

_ROOT = Path(
    _TEMP_DIRECTORY.name
)

_DATABASE = (
    _ROOT
    /
    "datalens.sqlite3"
)

_STORE_A = (
    _ROOT
    /
    "analysis_artifacts_a.json"
)

_STORE_B = (
    _ROOT
    /
    "analysis_artifacts_b.json"
)


os.environ[
    "DATALENS_SQLITE_PATH"
] = str(
    _DATABASE
)


from app.persistence.sqlite_database import (
    SQLITE_SCHEMA_VERSION,
    sqlite_schema_version,
)

from app.reporting.analysis_artifact_index import (
    ANALYSIS_ARTIFACT_INDEX_VERSION,
    analysis_artifact_index_is_initialized,
    analysis_artifact_store_scope,
    get_analysis_artifact_index_entry,
    load_analysis_artifact_index_scope,
    load_analysis_artifact_store_state,
    replace_analysis_artifact_index_scope,
)


TRACE_ID = (
    "report:requested:shared-trace"
)


def entry(
    *,
    analysis_id: str,
    workflow_id: str,
    payload_path: str,
    trace_id: str = TRACE_ID,
) -> dict:
    return {
        "analysis_id":
            analysis_id,

        "workflow_id":
            workflow_id,

        "trace_id":
            trace_id,

        "source_type":
            "document_request",

        "objective":
            "Analyse demand?e",

        "executed":
            True,

        "executed_count":
            1,

        "created_at_utc":
            "2026-08-25T10:00:00+00:00",

        "rule_version":
            "analysis_artifact_store_v0.2",

        "payload_path":
            payload_path,

        "payload_json_bytes":
            1000,

        "payload_file_bytes":
            250,

        "payload_sha256":
            (
                "a"
                *
                64
            ),
    }


def test_schema_v5() -> None:
    assert (
        sqlite_schema_version()
        ==
        SQLITE_SCHEMA_VERSION
        ==
        5
    )


    connection = sqlite3.connect(
        str(
            _DATABASE
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


        migrations = list(
            connection.execute(
                """
                SELECT
                    version,
                    name
                FROM schema_migrations
                ORDER BY version
                """
            )
        )

    finally:
        connection.close()


    assert (
        "analysis_artifacts"
        in tables
    )

    assert (
        "analysis_artifact_store_state"
        in tables
    )

    assert (
        (
            5,
            "analysis_artifact_metadata_index",
        )
        in migrations
    )


    print(
        "[PASS] SQLite schema v5 AnalysisArtifact tables"
    )


def test_duplicate_trace_is_allowed() -> None:
    entries = [
        entry(
            analysis_id=
                "analysis:report:one",

            workflow_id=
                "prep:a",

            payload_path=
                "data/one.json.gz",
        ),

        entry(
            analysis_id=
                "analysis:report:two",

            workflow_id=
                "prep:a",

            payload_path=
                "data/two.json.gz",
        ),
    ]


    replace_analysis_artifact_index_scope(
        store_path=
            _STORE_A,

        entries=
            entries,

        legacy_json_imported=
            False,

        legacy_rule_version=
            "analysis_artifact_store_v0.2",
    )


    restored = (
        load_analysis_artifact_index_scope(
            store_path=
                _STORE_A
        )
    )


    assert (
        len(
            restored
        )
        ==
        2
    )


    assert {
        item[
            "trace_id"
        ]

        for item
        in restored
    } == {
        TRACE_ID
    }


    print(
        "[PASS] duplicate trace_id values are allowed"
    )

    print(
        "[PASS] analysis_id remains logical identity"
    )


def test_point_read() -> None:
    restored = (
        get_analysis_artifact_index_entry(
            store_path=
                _STORE_A,

            analysis_id=
                "analysis:report:two",
        )
    )


    assert restored is not None

    assert (
        restored[
            "workflow_id"
        ]
        ==
        "prep:a"
    )

    assert (
        restored[
            "payload_path"
        ]
        ==
        "data/two.json.gz"
    )


    print(
        "[PASS] point metadata read by analysis_id"
    )


def test_store_isolation() -> None:
    replace_analysis_artifact_index_scope(
        store_path=
            _STORE_B,

        entries=[
            entry(
                analysis_id=
                    "analysis:report:one",

                workflow_id=
                    "prep:b",

                payload_path=
                    "data/store-b.json.gz",

                trace_id=
                    "trace:b",
            )
        ],

        legacy_json_imported=
            True,

        legacy_rule_version=
            "analysis_artifact_store_v0.2",
    )


    a = (
        get_analysis_artifact_index_entry(
            store_path=
                _STORE_A,

            analysis_id=
                "analysis:report:one",
        )
    )

    b = (
        get_analysis_artifact_index_entry(
            store_path=
                _STORE_B,

            analysis_id=
                "analysis:report:one",
        )
    )


    assert a is not None

    assert b is not None

    assert (
        a[
            "workflow_id"
        ]
        ==
        "prep:a"
    )

    assert (
        b[
            "workflow_id"
        ]
        ==
        "prep:b"
    )


    print(
        "[PASS] AnalysisArtifact store_path isolation"
    )


def test_transactional_replacement() -> None:
    replacement = [
        entry(
            analysis_id=
                "analysis:report:new",

            workflow_id=
                "prep:new",

            payload_path=
                "data/new.json.gz",

            trace_id=
                "trace:new",
        )
    ]


    replace_analysis_artifact_index_scope(
        store_path=
            _STORE_A,

        entries=
            replacement,

        legacy_json_imported=
            True,

        legacy_rule_version=
            "analysis_artifact_store_v0.2",
    )


    restored = (
        load_analysis_artifact_index_scope(
            store_path=
                _STORE_A
        )
    )


    assert (
        len(
            restored
        )
        ==
        1
    )

    assert (
        restored[
            0
        ][
            "analysis_id"
        ]
        ==
        "analysis:report:new"
    )


    assert (
        analysis_artifact_index_is_initialized(
            store_path=
                _STORE_A
        )
        is True
    )


    state = (
        load_analysis_artifact_store_state(
            store_path=
                _STORE_A
        )
    )


    assert state is not None

    assert (
        state[
            "legacy_json_imported"
        ]
        is True
    )


    print(
        "[PASS] transactional AnalysisArtifact scope replacement"
    )

    print(
        "[PASS] AnalysisArtifact migration state persisted"
    )


def test_sql_identity() -> None:
    connection = sqlite3.connect(
        str(
            _DATABASE
        )
    )

    try:
        scope = (
            analysis_artifact_store_scope(
                _STORE_A
            )
        )


        rows = list(
            connection.execute(
                """
                SELECT
                    analysis_id,
                    workflow_id,
                    trace_id
                FROM analysis_artifacts
                WHERE store_root = ?
                """,
                (
                    scope,
                ),
            )
        )

    finally:
        connection.close()


    assert rows == [
        (
            "analysis:report:new",
            "prep:new",
            "trace:new",
        )
    ]


    print(
        "[PASS] SQLite AnalysisArtifact logical identity"
    )


def test_version() -> None:
    assert (
        ANALYSIS_ARTIFACT_INDEX_VERSION
        ==
        "analysis_artifact_sqlite_index_v0.1"
    )


    print(
        "[PASS] AnalysisArtifact SQLite index version"
    )


def main() -> None:
    print()

    print(
        "=== DATALENS ANALYSIS ARTIFACT SQLITE INDEX v0.1 ==="
    )

    print()


    test_schema_v5()

    test_duplicate_trace_is_allowed()

    test_point_read()

    test_store_isolation()

    test_transactional_replacement()

    test_sql_identity()

    test_version()


    print()

    print(
        "PASS - Analysis Artifact SQLite Index v0.1"
    )


if __name__ == "__main__":
    main()
