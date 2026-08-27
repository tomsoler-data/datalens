from __future__ import annotations

import json
import os
import sqlite3
import tempfile

from pathlib import (
    Path,
)


# ========================================================
# ISOLATED DATABASE
# ========================================================


_TEMP_DIRECTORY = (
    tempfile.TemporaryDirectory(
        prefix=
            "datalens-preparation-sqlite-"
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

_LEGACY_PATH = (
    _ROOT
    /
    "preparation_sessions.json"
)


os.environ[
    "DATALENS_SQLITE_PATH"
] = str(
    _DATABASE_PATH
)

os.environ[
    "DATALENS_PREPARATION_SESSION_STORE_PATH"
] = str(
    _LEGACY_PATH
)


from fastapi import (
    FastAPI,
)

from fastapi.testclient import (
    TestClient,
)


from app.api.preparation_session import (
    router,
)

from app.persistence.sqlite_database import (
    SQLITE_SCHEMA_VERSION,
    resolve_sqlite_database_path,
    sqlite_schema_version,
)

from app.preparation.preparation_session import (
    PREPARATION_SESSION_STORE_VERSION,
    PreparationSessionRevisionConflictError,
    PreparationSessionStore,
    create_preparation_session,
    get_preparation_session,
    record_analysis_output_selection,
    record_required_stage_signal,
    reset_preparation_session_store_for_tests,
)

from app.preparation.preparation_workflow import (
    PreparationStage,
)


app = FastAPI()

app.include_router(
    router
)

client = TestClient(
    app
)


# ========================================================
# HELPERS
# ========================================================


def reset_state() -> None:
    reset_preparation_session_store_for_tests()

    _LEGACY_PATH.unlink(
        missing_ok=True
    )


# ========================================================
# 1. DATABASE / SCHEMA
# ========================================================


def test_database_schema() -> None:
    reset_state()

    session = create_preparation_session(
        selected_analysis_dataset_ids=[
            "dataset:orders",
        ]
    )


    assert (
        resolve_sqlite_database_path()
        ==
        _DATABASE_PATH.resolve()
    )

    assert (
        _DATABASE_PATH.exists()
    )

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

        assert (
            "schema_migrations"
            in tables
        )

        assert (
            "preparation_sessions"
            in tables
        )


        row = connection.execute(
            """
            SELECT
                workflow_id,
                revision,
                payload_json
            FROM preparation_sessions
            WHERE workflow_id = ?
            """,
            (
                session.workflow_id,
            ),
        ).fetchone()


        assert row is not None

        assert row[
            0
        ] == session.workflow_id

        assert row[
            1
        ] == 0


        payload = json.loads(
            row[
                2
            ]
        )

        assert (
            payload[
                "workflow_id"
            ]
            ==
            session.workflow_id
        )

    finally:
        connection.close()


    print(
        "[PASS] SQLite schema + session row"
    )


# ========================================================
# 2. FRESH STORE INSTANCE
# ========================================================


def test_fresh_store_restores_session() -> None:
    reset_state()

    session = create_preparation_session(
        selected_analysis_dataset_ids=[
            "dataset:orders",
        ]
    )


    updated = record_required_stage_signal(
        workflow_id=
            session.workflow_id,

        stage=
            PreparationStage.IMPORT,

        completed=
            True,

        dataset_ids=[
            "dataset:orders",
        ],

        evidence_refs=[
            "test:import",
        ],

        blocking_reasons=[],
    )


    fresh_store = (
        PreparationSessionStore()
    )


    restored = fresh_store.get(
        session.workflow_id
    )


    assert (
        restored.workflow_id
        ==
        session.workflow_id
    )

    assert (
        restored.revision
        ==
        updated.revision
    )

    assert (
        restored.import_stage.completed
        is True
    )


    public_restored = (
        get_preparation_session(
            session.workflow_id
        )
    )


    assert (
        public_restored.revision
        ==
        updated.revision
    )


    print(
        "[PASS] fresh store restores SQLite session"
    )


# ========================================================
# 3. OPTIMISTIC REVISION GUARD
# ========================================================


def test_revision_conflict() -> None:
    reset_state()

    session = create_preparation_session(
        selected_analysis_dataset_ids=[
            "dataset:orders",
        ]
    )


    stale_revision = (
        session.revision
    )


    record_required_stage_signal(
        workflow_id=
            session.workflow_id,

        stage=
            PreparationStage.IMPORT,

        completed=
            True,

        dataset_ids=[
            "dataset:orders",
        ],

        evidence_refs=[
            "test:import",
        ],

        blocking_reasons=[],
    )


    try:
        record_analysis_output_selection(
            workflow_id=
                session.workflow_id,

            analysis_output_dataset_ids=[
                "dataset:orders",
            ],

            expected_revision=
                stale_revision,
        )

    except PreparationSessionRevisionConflictError:
        pass

    else:
        raise AssertionError(
            (
                "Stale Preparation revision "
                "must fail closed."
            )
        )


    print(
        "[PASS] optimistic revision conflict"
    )


# ========================================================
# 4. LEGACY JSON MIGRATION
# ========================================================


def test_legacy_json_migration() -> None:
    reset_state()


    session = create_preparation_session(
        selected_analysis_dataset_ids=[
            "dataset:legacy",
        ]
    )


    stored_state = (
        PreparationSessionStore()
        .get(
            session.workflow_id
        )
    )


    legacy_payload = {
        "store_version":
            "preparation_session_store_v0.1",

        "sessions": {
            stored_state.workflow_id:
                stored_state.model_dump(
                    mode="json"
                )
        },
    }


    _LEGACY_PATH.write_text(
        json.dumps(
            legacy_payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


    # Empty SQLite but keep legacy JSON.
    reset_preparation_session_store_for_tests()


    fresh_store = (
        PreparationSessionStore()
    )


    restored = fresh_store.get(
        session.workflow_id
    )


    assert (
        restored.workflow_id
        ==
        session.workflow_id
    )

    assert (
        restored.selected_analysis_dataset_ids
        ==
        [
            "dataset:legacy",
        ]
    )


    print(
        "[PASS] legacy JSON -> SQLite migration"
    )


# ========================================================
# 5. API CAPABILITIES
# ========================================================


def test_capabilities() -> None:
    response = client.get(
        "/preparation/sessions/capabilities"
    )

    assert (
        response.status_code
        ==
        200
    )

    body = response.json()

    assert (
        body[
            "storage"
        ]
        ==
        "sqlite"
    )

    assert (
        body[
            "persistent"
        ]
        is True
    )


    print(
        "[PASS] API advertises SQLite persistence"
    )


# ========================================================
# 6. VERSION
# ========================================================


def test_version() -> None:
    assert (
        PREPARATION_SESSION_STORE_VERSION
        ==
        "preparation_session_sqlite_store_v0.1"
    )

    print(
        "[PASS] Preparation SQLite store version"
    )


# ========================================================
# MAIN
# ========================================================


def main() -> None:
    print()
    print(
        "=== DATALENS PREPARATION SESSION SQLITE v0.1 ==="
    )
    print()

    test_database_schema()

    test_fresh_store_restores_session()

    test_revision_conflict()

    test_legacy_json_migration()

    test_capabilities()

    test_version()

    print()
    print(
        "PASS - Preparation Session SQLite v0.1"
    )


if __name__ == "__main__":
    main()
