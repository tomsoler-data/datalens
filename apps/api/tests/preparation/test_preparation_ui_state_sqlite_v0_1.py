from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import tempfile

from pathlib import (
    Path,
)


# ========================================================
# ISOLATED SQLITE DATABASE
# ========================================================


_TEMP_DIRECTORY = (
    tempfile.TemporaryDirectory(
        prefix=
            "datalens-ui-state-sqlite-"
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


os.environ[
    "DATALENS_SQLITE_PATH"
] = str(
    _DATABASE_PATH
)


from app.persistence.sqlite_database import (
    SQLITE_SCHEMA_VERSION,
    sqlite_schema_version,
)

from app.preparation.preparation_ui_state import (
    PREPARATION_UI_STATE_STORE_VERSION,
    delete_preparation_ui_state,
    get_preparation_ui_state,
    reset_preparation_ui_state_store_for_tests,
    update_preparation_ui_state,
)


# ========================================================
# HELPERS
# ========================================================


def delete_database_files() -> None:
    for suffix in [
        "",
        "-wal",
        "-shm",
    ]:
        Path(
            str(
                _DATABASE_PATH
            )
            +
            suffix
        ).unlink(
            missing_ok=True
        )


def reset_state() -> None:
    reset_preparation_ui_state_store_for_tests()


# ========================================================
# 1. EXISTING v1 DATABASE -> v2
# ========================================================


def test_existing_v1_database_upgrades_to_v2(
) -> None:
    """
    Historical test name retained for compatibility.

    The same v1 fixture must now migrate through:
        v1 -> v2 -> v3
    """

    delete_database_files()


    connection = sqlite3.connect(
        str(
            _DATABASE_PATH
        )
    )

    try:
        connection.execute(
            """
            CREATE TABLE schema_migrations (
                version INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                applied_at TEXT NOT NULL
            )
            """
        )


        connection.execute(
            """
            CREATE TABLE preparation_sessions (
                workflow_id TEXT PRIMARY KEY,
                revision INTEGER NOT NULL
                    CHECK (revision >= 0),
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )


        connection.execute(
            """
            INSERT INTO schema_migrations (
                version,
                name,
                applied_at
            )
            VALUES (
                1,
                'initial_control_plane_preparation_sessions',
                '2026-08-25T00:00:00+00:00'
            )
            """
        )


        connection.commit()

    finally:
        connection.close()


    assert (
        sqlite_schema_version()
        ==
        SQLITE_SCHEMA_VERSION
        ==
        5
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


    required_tables = {
        "preparation_sessions",
        "preparation_ui_state",
        "preparation_artifacts",
        "preparation_artifact_store_state",
        "report_selection_workflows",
        "report_selection_store_state",
        "analysis_artifacts",
        "analysis_artifact_store_state",
    }


    assert (
        required_tables
        <=
        tables
    )


    assert migrations == [
        (
            1,
            (
                "initial_control_plane_"
                "preparation_sessions"
            ),
        ),
        (
            2,
            (
                "preparation_ui_state_"
                "control_plane"
            ),
        ),
        (
            3,
            (
                "preparation_artifact_"
                "sqlite_index"
            ),
        ),
        (
            4,
            (
                "report_selection_"
                "control_plane"
            ),
        ),
        (
            5,
            (
                "analysis_artifact_"
                "metadata_index"
            ),
        ),
    ]


    print(
        "[PASS] existing SQLite v1 -> v5 migration"
    )


# ========================================================
# 2. UPDATE / READ
# ========================================================


def test_update_and_read() -> None:
    delete_database_files()

    reset_state()


    workflow_id = (
        "prep:ui-sqlite-update"
    )


    first = update_preparation_ui_state(
        workflow_id=
            workflow_id,

        quality_report={
            "status":
                "ready",

            "issues": [
                {
                    "issue_id":
                        "quality:1",
                },
            ],
        },
    )


    assert first.revision == 1
    assert first.storage == "sqlite"
    assert first.persistent is True


    second = update_preparation_ui_state(
        workflow_id=
            workflow_id,

        cleaning_plan={
            "actions": [
                {
                    "action_id":
                        "clean:1",
                },
            ],
        },

        semantic_review={
            "decisions": [
                {
                    "issue_id":
                        "semantic:1",
                },
            ],
        },
    )


    assert second.revision == 2
    assert second.quality_report is not None
    assert second.cleaning_plan is not None
    assert second.semantic_review is not None


    restored = get_preparation_ui_state(
        workflow_id
    )


    assert restored.revision == 2
    assert restored.storage == "sqlite"
    assert restored.persistent is True


    print(
        "[PASS] SQLite UI-state update/read"
    )


# ========================================================
# 3. EXPLICIT None INVALIDATION
# ========================================================


def test_explicit_invalidation() -> None:
    delete_database_files()

    reset_state()


    workflow_id = (
        "prep:ui-sqlite-invalidation"
    )


    update_preparation_ui_state(
        workflow_id=
            workflow_id,

        cleaning_plan={
            "actions": [
                {
                    "action_id":
                        "clean:1",
                },
            ],
        },

        semantic_review={
            "decisions": [
                {
                    "issue_id":
                        "semantic:1",
                },
            ],
        },
    )


    updated = update_preparation_ui_state(
        workflow_id=
            workflow_id,

        cleaning_plan=None,
        semantic_review=None,
    )


    assert updated.cleaning_plan is None
    assert updated.semantic_review is None


    print(
        "[PASS] explicit None invalidation preserved"
    )


# ========================================================
# 4. COPY ISOLATION
# ========================================================


def test_copy_isolation() -> None:
    delete_database_files()

    reset_state()


    workflow_id = (
        "prep:ui-sqlite-copy"
    )


    source_payload = {
        "status":
            "ready",

        "issues": [
            {
                "issue_id":
                    "quality:1",
            },
        ],
    }


    update_preparation_ui_state(
        workflow_id=
            workflow_id,

        quality_report=
            source_payload,
    )


    source_payload[
        "issues"
    ][
        0
    ][
        "issue_id"
    ] = "outside-mutation"


    first = get_preparation_ui_state(
        workflow_id
    )


    first.quality_report[
        "issues"
    ][
        0
    ][
        "issue_id"
    ] = "read-mutation"


    second = get_preparation_ui_state(
        workflow_id
    )


    assert (
        second
        .quality_report[
            "issues"
        ][
            0
        ][
            "issue_id"
        ]
        ==
        "quality:1"
    )


    print(
        "[PASS] SQLite UI-state copy isolation"
    )


# ========================================================
# 5. FRESH PYTHON PROCESS
# ========================================================


def test_survives_fresh_python_process(
) -> None:
    delete_database_files()

    reset_state()


    workflow_id = (
        "prep:ui-sqlite-process"
    )


    update_preparation_ui_state(
        workflow_id=
            workflow_id,

        semantic_confirmation={
            "confirmed":
                True,

            "unresolved_issue_ids":
                [],
        },

        confirmed_semantic_issue_ids=[
            "semantic:1",
        ],
    )


    code = (
        "from app.preparation.preparation_ui_state "
        "import get_preparation_ui_state;"
        "state=get_preparation_ui_state("
        f"{workflow_id!r}"
        ");"
        "print(state.model_dump_json())"
    )


    result = subprocess.run(
        [
            sys.executable,
            "-c",
            code,
        ],
        capture_output=True,
        text=True,
        check=True,
        env=os.environ.copy(),
    )


    payload = json.loads(
        result.stdout.strip()
    )


    assert (
        payload[
            "workflow_id"
        ]
        ==
        workflow_id
    )

    assert (
        payload[
            "semantic_confirmation"
        ][
            "confirmed"
        ]
        is True
    )

    assert (
        payload[
            "confirmed_semantic_issue_ids"
        ]
        ==
        [
            "semantic:1",
        ]
    )

    assert (
        payload[
            "storage"
        ]
        ==
        "sqlite"
    )

    assert (
        payload[
            "persistent"
        ]
        is True
    )


    print(
        "[PASS] UI state survives fresh Python process"
    )


# ========================================================
# 6. DURABLE DELETE
# ========================================================


def test_durable_delete() -> None:
    delete_database_files()

    reset_state()


    workflow_id = (
        "prep:ui-sqlite-delete"
    )


    update_preparation_ui_state(
        workflow_id=
            workflow_id,

        quality_report={
            "status":
                "ready",
        },
    )


    delete_preparation_ui_state(
        workflow_id
    )


    restored = get_preparation_ui_state(
        workflow_id
    )


    assert restored.revision == 0
    assert restored.quality_report is None
    assert restored.storage == "sqlite"
    assert restored.persistent is True


    print(
        "[PASS] durable UI-state delete"
    )


# ========================================================
# 7. VERSION
# ========================================================


def test_version() -> None:
    assert (
        PREPARATION_UI_STATE_STORE_VERSION
        ==
        "preparation_ui_state_sqlite_store_v0.1"
    )


    print(
        "[PASS] UI-state SQLite store version"
    )


# ========================================================
# MAIN
# ========================================================


def main() -> None:
    print()
    print(
        "=== DATALENS PREPARATION UI STATE SQLITE v0.1 ==="
    )
    print()

    test_existing_v1_database_upgrades_to_v2()

    test_update_and_read()

    test_explicit_invalidation()

    test_copy_isolation()

    test_survives_fresh_python_process()

    test_durable_delete()

    test_version()

    print()
    print(
        "PASS - Preparation UI State SQLite v0.1"
    )


if __name__ == "__main__":
    main()
