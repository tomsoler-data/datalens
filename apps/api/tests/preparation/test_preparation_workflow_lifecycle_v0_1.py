from __future__ import annotations


import os
import sqlite3
import tempfile
import time


print(
    "=== DATALENS PREPARATION WORKFLOW METADATA v0.1 ==="
)


with tempfile.TemporaryDirectory(
    prefix=
        "datalens-workflow-metadata-"
) as temporary_directory:

    v6_database = os.path.join(
        temporary_directory,
        "schema-v6.sqlite3",
    )

    fresh_database = os.path.join(
        temporary_directory,
        "fresh.sqlite3",
    )


    os.environ.pop(
        "DATALENS_PREPARATION_SESSION_STORE_PATH",
        None,
    )


    # ========================================================
    # V6 -> V7 MIGRATION
    # ========================================================


    os.environ[
        "DATALENS_SQLITE_PATH"
    ] = v6_database


    from app.persistence.sqlite_database import (
        sqlite_schema_version,
    )


    connection = sqlite3.connect(
        v6_database
    )


    try:
        connection.execute(
            "PRAGMA foreign_keys = ON"
        )


        connection.execute(
            """
            CREATE TABLE schema_migrations (
                version INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                applied_at TEXT NOT NULL
            )
            """
        )


        for version in range(
            1,
            7,
        ):
            connection.execute(
                """
                INSERT INTO schema_migrations (
                    version,
                    name,
                    applied_at
                )
                VALUES (
                    ?,
                    ?,
                    ?
                )
                """,
                (
                    version,
                    f"legacy-v{version}",
                    "2026-01-01T00:00:00+00:00",
                ),
            )


        connection.execute(
            """
            CREATE TABLE preparation_sessions (
                workflow_id TEXT PRIMARY KEY,
                revision INTEGER NOT NULL,
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )


        connection.execute(
            """
            CREATE TABLE preparation_workflow_lifecycle (
                workflow_id TEXT PRIMARY KEY,
                archived_at TEXT,
                updated_at TEXT NOT NULL,

                FOREIGN KEY (
                    workflow_id
                )
                REFERENCES preparation_sessions (
                    workflow_id
                )
                ON DELETE CASCADE
            )
            """
        )


        connection.executemany(
            """
            INSERT INTO preparation_sessions (
                workflow_id,
                revision,
                payload_json,
                created_at,
                updated_at
            )
            VALUES (
                ?,
                ?,
                ?,
                ?,
                ?
            )
            """,
            [
                (
                    "prep:v6-active",
                    4,
                    '{"active":true}',
                    "2026-08-25T16:30:00+00:00",
                    "2026-08-25T16:40:00+00:00",
                ),
                (
                    "prep:v6-archived",
                    7,
                    '{"archived":true}',
                    "2026-08-25T17:35:13+00:00",
                    "2026-08-25T17:38:32+00:00",
                ),
            ],
        )


        connection.execute(
            """
            INSERT INTO preparation_workflow_lifecycle (
                workflow_id,
                archived_at,
                updated_at
            )
            VALUES (
                ?,
                ?,
                ?
            )
            """,
            (
                "prep:v6-archived",
                "2026-08-25T18:00:00+00:00",
                "2026-08-25T18:00:00+00:00",
            ),
        )


        connection.commit()


    finally:
        connection.close()


    assert (
        sqlite_schema_version()
        ==
        7
    )


    connection = sqlite3.connect(
        v6_database
    )

    connection.row_factory = sqlite3.Row


    try:
        metadata = (
            connection.execute(
                """
                SELECT
                    workflow_id,
                    display_name,
                    name_source,
                    archived_at

                FROM preparation_workflow_metadata

                ORDER BY workflow_id
                """
            )
            .fetchall()
        )


        assert len(
            metadata
        ) == 2


        by_id = {
            row[
                "workflow_id"
            ]:
                row

            for row
            in metadata
        }


        assert (
            by_id[
                "prep:v6-active"
            ][
                "archived_at"
            ]
            is None
        )


        assert (
            by_id[
                "prep:v6-archived"
            ][
                "archived_at"
            ]
            ==
            "2026-08-25T18:00:00+00:00"
        )


        assert (
            by_id[
                "prep:v6-archived"
            ][
                "display_name"
            ]
            ==
            "Analyse - 2026-08-25 17:35 UTC"
        )


        lifecycle_exists = (
            connection.execute(
                """
                SELECT 1
                FROM sqlite_master
                WHERE
                    type = 'table'
                    AND
                    name =
                        'preparation_workflow_lifecycle'
                """
            )
            .fetchone()
            is not None
        )


        assert (
            lifecycle_exists
            is False
        )


        trigger_exists = (
            connection.execute(
                """
                SELECT 1
                FROM sqlite_master
                WHERE
                    type = 'trigger'
                    AND
                    name =
                        'trg_preparation_workflow_metadata_after_session_insert'
                """
            )
            .fetchone()
            is not None
        )


        assert (
            trigger_exists
            is True
        )


        preserved = (
            connection.execute(
                """
                SELECT
                    revision,
                    payload_json

                FROM preparation_sessions

                WHERE workflow_id =
                    'prep:v6-archived'
                """
            )
            .fetchone()
        )


        assert (
            preserved[
                "revision"
            ]
            ==
            7
        )


        assert (
            preserved[
                "payload_json"
            ]
            ==
            '{"archived":true}'
        )


    finally:
        connection.close()


    print(
        "[PASS] v6 -> v7 metadata migration"
    )

    print(
        "[PASS] v6 archive state preserved"
    )

    print(
        "[PASS] lifecycle-v6 table retired"
    )

    print(
        "[PASS] future-session metadata trigger installed"
    )

    print(
        "[PASS] analytical PreparationSession preserved"
    )


    # ========================================================
    # FRESH V7 DATABASE
    # ========================================================


    os.environ[
        "DATALENS_SQLITE_PATH"
    ] = fresh_database


    from app.preparation.preparation_session import (
        PreparationSessionNotFoundError,
        archive_preparation_session,
        create_preparation_session,
        list_preparation_sessions,
        rename_preparation_session,
        restore_preparation_session,
    )


    automatic = (
        create_preparation_session(
            selected_analysis_dataset_ids=[
                "dataset:auto",
            ]
        )
    )


    time.sleep(
        0.01
    )


    named = (
        create_preparation_session(
            selected_analysis_dataset_ids=[
                "dataset:named",
            ],

            display_name=
                "Analyse ventes Lapage",
        )
    )


    time.sleep(
        0.01
    )


    blank = (
        create_preparation_session(
            selected_analysis_dataset_ids=[
                "dataset:blank",
            ],

            display_name=
                "   ",
        )
    )


    catalog = (
        list_preparation_sessions()
    )


    items = {
        item.session.workflow_id:
            item

        for item
        in catalog
    }


    assert (
        items[
            automatic.workflow_id
        ]
        .name_source
        ==
        "automatic"
    )


    assert (
        items[
            automatic.workflow_id
        ]
        .display_name
        .startswith(
            "Analyse - "
        )
    )


    assert (
        items[
            named.workflow_id
        ]
        .display_name
        ==
        "Analyse ventes Lapage"
    )


    assert (
        items[
            named.workflow_id
        ]
        .name_source
        ==
        "user"
    )


    assert (
        items[
            blank.workflow_id
        ]
        .name_source
        ==
        "automatic"
    )


    print(
        "[PASS] optional name + automatic fallback"
    )


    # ========================================================
    # RENAME PRESERVES ANALYTICAL STATE
    # ========================================================


    connection = sqlite3.connect(
        fresh_database
    )

    connection.row_factory = sqlite3.Row


    try:
        before = (
            connection.execute(
                """
                SELECT
                    revision,
                    payload_json,
                    created_at,
                    updated_at

                FROM preparation_sessions

                WHERE workflow_id = ?
                """,
                (
                    automatic.workflow_id,
                ),
            )
            .fetchone()
        )


    finally:
        connection.close()


    before_tuple = tuple(
        before[
            key
        ]

        for key in [
            "revision",
            "payload_json",
            "created_at",
            "updated_at",
        ]
    )


    renamed = (
        rename_preparation_session(
            workflow_id=
                automatic.workflow_id,

            display_name=
                "Analyse clients",
        )
    )


    assert (
        renamed.display_name
        ==
        "Analyse clients"
    )


    assert (
        renamed.name_source
        ==
        "user"
    )


    connection = sqlite3.connect(
        fresh_database
    )

    connection.row_factory = sqlite3.Row


    try:
        after = (
            connection.execute(
                """
                SELECT
                    revision,
                    payload_json,
                    created_at,
                    updated_at

                FROM preparation_sessions

                WHERE workflow_id = ?
                """,
                (
                    automatic.workflow_id,
                ),
            )
            .fetchone()
        )


    finally:
        connection.close()


    after_tuple = tuple(
        after[
            key
        ]

        for key in [
            "revision",
            "payload_json",
            "created_at",
            "updated_at",
        ]
    )


    assert (
        after_tuple
        ==
        before_tuple
    )


    print(
        "[PASS] rename changes metadata only"
    )


    # ========================================================
    # ARCHIVE / RESTORE
    # ========================================================


    archived = (
        archive_preparation_session(
            automatic.workflow_id
        )
    )


    assert (
        archived.archived
        is True
    )


    assert (
        archived.display_name
        ==
        "Analyse clients"
    )


    archived_at = (
        archived.archived_at_utc
    )


    time.sleep(
        0.01
    )


    archived_again = (
        archive_preparation_session(
            automatic.workflow_id
        )
    )


    assert (
        archived_again
        .archived_at_utc
        ==
        archived_at
    )


    assert (
        list_preparation_sessions()[
            -1
        ]
        .session
        .workflow_id
        ==
        automatic.workflow_id
    )


    restored = (
        restore_preparation_session(
            automatic.workflow_id
        )
    )


    assert (
        restored.archived
        is False
    )


    assert (
        restored.display_name
        ==
        "Analyse clients"
    )


    restored_again = (
        restore_preparation_session(
            automatic.workflow_id
        )
    )


    assert (
        restored_again.archived
        is False
    )


    connection = sqlite3.connect(
        fresh_database
    )


    try:
        metadata_count = int(
            connection.execute(
                """
                SELECT COUNT(*)
                FROM preparation_workflow_metadata
                WHERE workflow_id = ?
                """,
                (
                    automatic.workflow_id,
                ),
            )
            .fetchone()[0]
        )


    finally:
        connection.close()


    assert (
        metadata_count
        ==
        1
    )


    print(
        "[PASS] archive idempotent"
    )

    print(
        "[PASS] restore keeps name + metadata row"
    )


    # ========================================================
    # MISSING WORKFLOW
    # ========================================================


    failed_closed = False


    try:
        rename_preparation_session(
            workflow_id=
                "prep:missing",

            display_name=
                "Missing",
        )

    except PreparationSessionNotFoundError:
        failed_closed = True


    assert (
        failed_closed
    )


    print(
        "[PASS] metadata operations fail closed"
    )


    # ========================================================
    # API
    # ========================================================


    from fastapi.testclient import (
        TestClient,
    )

    from app.main import app


    client = TestClient(
        app
    )


    created = client.post(
        "/preparation/sessions",
        json={
            "selected_analysis_dataset_ids": [
                "dataset:api"
            ],

            "display_name":
                "Projet API",
        },
    )


    assert (
        created.status_code
        ==
        201
    ), created.text


    workflow_id = (
        created.json()[
            "workflow_id"
        ]
    )


    catalog_response = client.get(
        "/preparation/sessions"
    )


    api_item = next(
        item

        for item
        in catalog_response
        .json()[
            "sessions"
        ]

        if (
            item[
                "session"
            ][
                "workflow_id"
            ]
            ==
            workflow_id
        )
    )


    assert (
        api_item[
            "display_name"
        ]
        ==
        "Projet API"
    )


    renamed_response = client.post(
        (
            "/preparation/sessions/"
            +
            workflow_id
            +
            "/rename"
        ),
        json={
            "display_name":
                "Projet renomme",
        },
    )


    assert (
        renamed_response.status_code
        ==
        200
    ), renamed_response.text


    assert (
        renamed_response.json()[
            "display_name"
        ]
        ==
        "Projet renomme"
    )


    archived_response = client.post(
        (
            "/preparation/sessions/"
            +
            workflow_id
            +
            "/archive"
        )
    )


    assert (
        archived_response.status_code
        ==
        200
    ), archived_response.text


    assert (
        archived_response.json()[
            "archived"
        ]
        is True
    )


    restored_response = client.post(
        (
            "/preparation/sessions/"
            +
            workflow_id
            +
            "/restore"
        )
    )


    assert (
        restored_response.status_code
        ==
        200
    ), restored_response.text


    assert (
        restored_response.json()[
            "archived"
        ]
        is False
    )


    too_long = client.post(
        "/preparation/sessions",
        json={
            "selected_analysis_dataset_ids": [
                "dataset:long"
            ],

            "display_name":
                "x" * 121,
        },
    )


    assert (
        too_long.status_code
        ==
        422
    )


    blank_rename = client.post(
        (
            "/preparation/sessions/"
            +
            workflow_id
            +
            "/rename"
        ),
        json={
            "display_name":
                "   ",
        },
    )


    assert (
        blank_rename.status_code
        ==
        422
    )


    print(
        "[PASS] create / rename / archive / restore API"
    )


print()
print(
    "PASS - preparation workflow metadata v0.1"
)
