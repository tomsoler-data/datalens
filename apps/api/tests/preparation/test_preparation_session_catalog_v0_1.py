from __future__ import annotations


import json
import os
import sqlite3
import tempfile
import time


print(
    "=== DATALENS PREPARATION SESSION CATALOG v0.1 ==="
)


with tempfile.TemporaryDirectory(
    prefix=
        "datalens-preparation-session-catalog-"
) as temporary_directory:

    database_path = os.path.join(
        temporary_directory,
        "datalens.sqlite3",
    )


    os.environ[
        "DATALENS_SQLITE_PATH"
    ] = database_path


    os.environ.pop(
        "DATALENS_PREPARATION_SESSION_STORE_PATH",
        None,
    )


    from fastapi.testclient import (
        TestClient,
    )


    from app.main import app


    from app.preparation.preparation_session import (
        PreparationSessionStoreError,
        create_preparation_session,
        get_preparation_session,
        list_preparation_sessions,
    )


    # ========================================================
    # TWO SERVER-OWNED WORKFLOWS
    # ========================================================


    first = (
        create_preparation_session(
            selected_analysis_dataset_ids=[
                "dataset:0001",
            ]
        )
    )


    # utc_now_iso() has microsecond precision, but a tiny delay
    # keeps ordering intention visually obvious across platforms.
    time.sleep(
        0.01
    )


    second = (
        create_preparation_session(
            selected_analysis_dataset_ids=[
                "dataset:0001",
                "dataset:0002",
            ]
        )
    )


    catalog = (
        list_preparation_sessions()
    )


    assert len(
        catalog
    ) == 2


    assert (
        catalog[
            0
        ]
        .session
        .workflow_id
        ==
        second.workflow_id
    )


    assert (
        catalog[
            1
        ]
        .session
        .workflow_id
        ==
        first.workflow_id
    )


    assert (
        catalog[
            0
        ]
        .updated_at_utc
        >=
        catalog[
            1
        ]
        .updated_at_utc
    )


    assert (
        catalog[
            0
        ]
        .created_at_utc
    )


    assert (
        catalog[
            1
        ]
        .created_at_utc
    )


    assert (
        catalog[
            0
        ]
        .session
        .snapshot
        .ready_for_analysis
        is False
    )


    print(
        "[PASS] catalog lists canonical Preparation sessions"
    )

    print(
        "[PASS] catalog is ordered by most recent activity"
    )


    # ========================================================
    # API
    # ========================================================


    client = TestClient(
        app
    )


    response = client.get(
        "/preparation/sessions"
    )


    assert (
        response.status_code
        ==
        200
    ), response.text


    payload = (
        response.json()
    )


    assert (
        payload[
            "count"
        ]
        ==
        2
    )


    assert (
        len(
            payload[
                "sessions"
            ]
        )
        ==
        2
    )


    assert (
        payload[
            "sessions"
        ][
            0
        ][
            "session"
        ][
            "workflow_id"
        ]
        ==
        second.workflow_id
    )


    assert (
        payload[
            "sessions"
        ][
            1
        ][
            "session"
        ][
            "workflow_id"
        ]
        ==
        first.workflow_id
    )


    print(
        "[PASS] GET /preparation/sessions"
    )


    # Existing route must remain unaffected.
    single_response = client.get(
        (
            "/preparation/sessions/"
            +
            first.workflow_id
        )
    )


    assert (
        single_response.status_code
        ==
        200
    ), single_response.text


    assert (
        single_response
        .json()[
            "workflow_id"
        ]
        ==
        first.workflow_id
    )


    existing = (
        get_preparation_session(
            first.workflow_id
        )
    )


    assert (
        existing.workflow_id
        ==
        first.workflow_id
    )


    print(
        "[PASS] existing single-workflow endpoint remains intact"
    )


    # ========================================================
    # FAIL CLOSED ? CORRUPTED SQLITE PAYLOAD
    # ========================================================


    connection = sqlite3.connect(
        database_path
    )


    try:
        corrupted = {
            "workflow_id":
                "prep:wrong-owner",

            "revision":
                first.revision,

            "selected_analysis_dataset_ids": [
                "dataset:0001"
            ],

            "analysis_output_dataset_ids": [],
        }


        connection.execute(
            """
            UPDATE preparation_sessions
            SET payload_json = ?
            WHERE workflow_id = ?
            """,
            (
                json.dumps(
                    corrupted
                ),
                first.workflow_id,
            ),
        )


        connection.commit()


    finally:
        connection.close()


    failed_closed = False


    try:
        list_preparation_sessions()

    except PreparationSessionStoreError:
        failed_closed = True


    assert (
        failed_closed
    )


    print(
        "[PASS] catalog fails closed on invalid persisted session"
    )


print()
print(
    "PASS - preparation session catalog v0.1"
)
