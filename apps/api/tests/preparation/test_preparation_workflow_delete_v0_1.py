from __future__ import annotations

import json
import os
import sqlite3
import tempfile

from pathlib import Path


print(
    "=== DATALENS PREPARATION WORKFLOW PERMANENT DELETE v0.1 ==="
)


with tempfile.TemporaryDirectory(
    prefix=
        "datalens-workflow-delete-"
) as temporary_directory:
    temporary_root = Path(
        temporary_directory
    )

    database_path = (
        temporary_root
        /
        "datalens.sqlite3"
    )

    quarantine_root = (
        temporary_root
        /
        "quarantine"
    )

    preparation_root = (
        temporary_root
        /
        "preparation"
        /
        "artifacts"
    )

    analysis_logical_store = (
        temporary_root
        /
        "reporting"
        /
        "analysis_artifacts.json"
    )

    analysis_data_root = (
        analysis_logical_store.parent
        /
        analysis_logical_store.stem
    )

    report_store = (
        temporary_root
        /
        "reporting"
        /
        "report_selection.json"
    )

    os.environ[
        "DATALENS_SQLITE_PATH"
    ] = str(
        database_path
    )

    os.environ[
        "DATALENS_WORKFLOW_DELETE_QUARANTINE_ROOT"
    ] = str(
        quarantine_root
    )


    from app.persistence.sqlite_database import (
        sqlite_connection,
        sqlite_schema_version,
    )

    from app.preparation.preparation_session import (
        archive_preparation_session,
        create_preparation_session,
        get_preparation_session,
    )

    from app.preparation.preparation_workflow_delete import (
        PreparationWorkflowDeleteConfirmationError,
        PreparationWorkflowDeleteIntegrityError,
        PreparationWorkflowDeleteNotArchivedError,
        PreparationWorkflowDeleteNotFoundError,
        PreparationWorkflowDeleteRevisionConflictError,
        delete_preparation_workflow,
    )


    assert (
        sqlite_schema_version()
        ==
        7
    )


    def make_workflow(
        *,
        display_name: str,
        archived: bool,
    ):
        session = (
            create_preparation_session(
                selected_analysis_dataset_ids=[
                    "dataset:root",
                ],

                display_name=
                    display_name,
            )
        )

        if archived:
            archive_preparation_session(
                session.workflow_id
            )

        return (
            session
        )


    def add_resources(
        workflow_id: str,
    ):
        preparation_data = (
            preparation_root
            /
            "data"
        )

        preparation_data.mkdir(
            parents=True,
            exist_ok=True,
        )

        analysis_data = (
            analysis_data_root
            /
            "data"
        )

        analysis_data.mkdir(
            parents=True,
            exist_ok=True,
        )

        prep_files = []

        for index in range(
            2
        ):
            path = (
                preparation_data
                /
                (
                    "artifact_prep_"
                    f"{index}.json.gz"
                )
            )

            path.write_bytes(
                (
                    f"prep-{index}"
                ).encode(
                    "utf-8"
                )
            )

            prep_files.append(
                path
            )

        analysis_files = []

        for index in range(
            3
        ):
            path = (
                analysis_data
                /
                (
                    "artifact_analysis_"
                    f"{index}.json.gz"
                )
            )

            payload = (
                f"analysis-{index}"
            ).encode(
                "utf-8"
            )

            path.write_bytes(
                payload
            )

            analysis_files.append(
                path
            )


        with sqlite_connection(
            write=True
        ) as connection:
            for index, path in enumerate(
                prep_files
            ):
                connection.execute(
                    """
                    INSERT INTO preparation_artifacts (
                        store_root,
                        workflow_id,
                        dataset_id,
                        dataset_filename,
                        stage,
                        rows,
                        columns,
                        parent_dataset_ids_json,
                        evidence_refs_json,
                        datetime_dtypes_json,
                        data_path
                    )
                    VALUES (
                        ?, ?, ?, ?, ?, ?,
                        ?, ?, ?, ?, ?
                    )
                    """,
                    (
                        str(
                            preparation_root
                        ),
                        workflow_id,
                        (
                            "dataset:"
                            f"{index}"
                        ),
                        (
                            "dataset_"
                            f"{index}.csv"
                        ),
                        "source",
                        1,
                        1,
                        "[]",
                        "[]",
                        "[]",
                        str(
                            path.relative_to(
                                preparation_root
                            )
                        ).replace(
                            "\\",
                            "/",
                        ),
                    ),
                )

            for index, path in enumerate(
                analysis_files
            ):
                connection.execute(
                    """
                    INSERT INTO analysis_artifacts (
                        store_root,
                        analysis_id,
                        workflow_id,
                        trace_id,
                        source_type,
                        objective,
                        executed,
                        executed_count,
                        created_at_utc,
                        rule_version,
                        payload_path,
                        payload_json_bytes,
                        payload_file_bytes,
                        payload_sha256
                    )
                    VALUES (
                        ?, ?, ?, ?, ?, ?, ?,
                        ?, ?, ?, ?, ?, ?, ?
                    )
                    """,
                    (
                        str(
                            analysis_logical_store
                        ),
                        (
                            "analysis:test:"
                            f"{workflow_id}:"
                            f"{index}"
                        ),
                        workflow_id,
                        (
                            "trace:"
                            f"{workflow_id}:"
                            f"{index}"
                        ),
                        "initial_request",
                        "test",
                        1,
                        1,
                        (
                            "2026-08-25T00:00:00"
                            "+00:00"
                        ),
                        "test",
                        str(
                            path.relative_to(
                                analysis_data_root
                            )
                        ).replace(
                            "\\",
                            "/",
                        ),
                        path.stat().st_size,
                        path.stat().st_size,
                        (
                            "0"
                            *
                            64
                        ),
                    ),
                )

            connection.execute(
                """
                INSERT INTO preparation_ui_state (
                    workflow_id,
                    revision,
                    payload_json,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    workflow_id,
                    0,
                    "{}",
                    (
                        "2026-08-25T00:00:00"
                        "+00:00"
                    ),
                    (
                        "2026-08-25T00:00:00"
                        "+00:00"
                    ),
                ),
            )

            connection.execute(
                """
                INSERT INTO report_selection_workflows (
                    store_root,
                    workflow_id,
                    revision,
                    payload_json,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    str(
                        report_store
                    ),
                    workflow_id,
                    0,
                    json.dumps(
                        {
                            "workflow_id":
                                workflow_id,
                        }
                    ),
                    (
                        "2026-08-25T00:00:00"
                        "+00:00"
                    ),
                ),
            )

        return (
            prep_files,
            analysis_files,
        )


    # ========================================================
    # ARCHIVED-ONLY CONTRACT
    # ========================================================

    active = (
        make_workflow(
            display_name=
                "Active workflow",

            archived=
                False,
        )
    )

    try:
        delete_preparation_workflow(
            workflow_id=
                active.workflow_id,

            confirmation_workflow_id=
                active.workflow_id,

            confirmation_display_name=
                "Active workflow",

            expected_revision=
                active.revision,
        )

    except PreparationWorkflowDeleteNotArchivedError:
        pass

    else:
        raise AssertionError(
            (
                "Active workflow deletion "
                "should have been rejected."
            )
        )

    assert (
        get_preparation_session(
            active.workflow_id
        ).workflow_id
        ==
        active.workflow_id
    )

    print(
        "[PASS] active workflow deletion rejected"
    )


    # ========================================================
    # CONFIRMATION + REVISION
    # ========================================================

    confirmation = (
        make_workflow(
            display_name=
                "Confirmation workflow",

            archived=
                True,
        )
    )

    try:
        delete_preparation_workflow(
            workflow_id=
                confirmation.workflow_id,

            confirmation_workflow_id=
                "prep:wrong",

            confirmation_display_name=
                "Confirmation workflow",

            expected_revision=
                confirmation.revision,
        )

    except PreparationWorkflowDeleteConfirmationError:
        pass

    else:
        raise AssertionError(
            "workflow_id confirmation should fail."
        )


    try:
        delete_preparation_workflow(
            workflow_id=
                confirmation.workflow_id,

            confirmation_workflow_id=
                confirmation.workflow_id,

            confirmation_display_name=
                "Wrong name",

            expected_revision=
                confirmation.revision,
        )

    except PreparationWorkflowDeleteConfirmationError:
        pass

    else:
        raise AssertionError(
            "display-name confirmation should fail."
        )


    try:
        delete_preparation_workflow(
            workflow_id=
                confirmation.workflow_id,

            confirmation_workflow_id=
                confirmation.workflow_id,

            confirmation_display_name=
                "Confirmation workflow",

            expected_revision=
                confirmation.revision
                +
                1,
        )

    except PreparationWorkflowDeleteRevisionConflictError:
        pass

    else:
        raise AssertionError(
            "revision confirmation should fail."
        )

    print(
        "[PASS] identity/name/revision confirmation"
    )


    # ========================================================
    # UNKNOWN ROOT FAILS CLOSED
    # ========================================================

    orphan_workflow = (
        "prep:orphan-without-session"
    )

    orphan_directory = (
        analysis_data_root
        /
        "data"
    )

    orphan_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    orphan_file = (
        orphan_directory
        /
        "orphan.json.gz"
    )

    orphan_file.write_bytes(
        b"orphan"
    )

    with sqlite_connection(
        write=True
    ) as connection:
        connection.execute(
            """
            INSERT INTO analysis_artifacts (
                store_root,
                analysis_id,
                workflow_id,
                trace_id,
                source_type,
                objective,
                executed,
                executed_count,
                created_at_utc,
                rule_version,
                payload_path,
                payload_json_bytes,
                payload_file_bytes,
                payload_sha256
            )
            VALUES (
                ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?, ?
            )
            """,
            (
                str(
                    analysis_logical_store
                ),
                "analysis:orphan",
                orphan_workflow,
                "trace:orphan",
                "initial_request",
                "test",
                1,
                1,
                (
                    "2026-08-25T00:00:00"
                    "+00:00"
                ),
                "test",
                str(
                    orphan_file.relative_to(
                        analysis_data_root
                    )
                ).replace(
                    "\\",
                    "/",
                ),
                6,
                6,
                (
                    "0"
                    *
                    64
                ),
            ),
        )

    try:
        delete_preparation_workflow(
            workflow_id=
                orphan_workflow,

            confirmation_workflow_id=
                orphan_workflow,

            confirmation_display_name=
                "Orphan",

            expected_revision=
                0,
        )

    except PreparationWorkflowDeleteNotFoundError:
        pass

    else:
        raise AssertionError(
            (
                "Unknown root workflow should "
                "fail closed."
            )
        )

    assert (
        orphan_file.exists()
    )

    with sqlite_connection(
        write=False
    ) as connection:
        assert int(
            connection.execute(
                """
                SELECT COUNT(*)
                FROM analysis_artifacts
                WHERE workflow_id = ?
                """,
                (
                    orphan_workflow,
                ),
            ).fetchone()[0]
        ) == 1

    print(
        "[PASS] unknown root preserves subordinate orphan data"
    )


    # ========================================================
    # FUTURE TABLE FAIL-CLOSED GUARD
    # ========================================================

    unexpected = (
        make_workflow(
            display_name=
                "Unexpected ownership",

            archived=
                True,
        )
    )

    with sqlite_connection(
        write=True
    ) as connection:
        connection.execute(
            """
            CREATE TABLE unexpected_workflow_refs (
                id INTEGER PRIMARY KEY,
                workflow_id TEXT NOT NULL
            )
            """
        )

        connection.execute(
            """
            INSERT INTO unexpected_workflow_refs (
                workflow_id
            )
            VALUES (?)
            """,
            (
                unexpected.workflow_id,
            ),
        )

    try:
        delete_preparation_workflow(
            workflow_id=
                unexpected.workflow_id,

            confirmation_workflow_id=
                unexpected.workflow_id,

            confirmation_display_name=
                "Unexpected ownership",

            expected_revision=
                unexpected.revision,
        )

    except PreparationWorkflowDeleteIntegrityError:
        pass

    else:
        raise AssertionError(
            (
                "Unknown workflow-owned table "
                "should block deletion."
            )
        )

    with sqlite_connection(
        write=True
    ) as connection:
        connection.execute(
            """
            DELETE FROM unexpected_workflow_refs
            WHERE workflow_id = ?
            """,
            (
                unexpected.workflow_id,
            ),
        )

        connection.execute(
            """
            DROP TABLE unexpected_workflow_refs
            """
        )

    print(
        "[PASS] future workflow-owned tables fail closed"
    )


    # ========================================================
    # ROLLBACK RESTORES QUARANTINED FILES
    # ========================================================

    rollback = (
        make_workflow(
            display_name=
                "Rollback workflow",

            archived=
                True,
        )
    )

    (
        rollback_prep_files,
        rollback_analysis_files,
    ) = (
        add_resources(
            rollback.workflow_id
        )
    )

    def injected_failure(
        stage: str,
    ) -> None:
        if (
            stage
            ==
            "after_quarantine"
        ):
            raise RuntimeError(
                "injected rollback test"
            )

    try:
        delete_preparation_workflow(
            workflow_id=
                rollback.workflow_id,

            confirmation_workflow_id=
                rollback.workflow_id,

            confirmation_display_name=
                "Rollback workflow",

            expected_revision=
                rollback.revision,

            _failure_hook=
                injected_failure,
        )

    except RuntimeError as error:
        assert (
            "injected rollback test"
            in
            str(
                error
            )
        )

    else:
        raise AssertionError(
            (
                "Injected post-quarantine "
                "failure should propagate."
            )
        )

    assert all(
        path.exists()

        for path
        in (
            rollback_prep_files
            +
            rollback_analysis_files
        )
    )

    assert (
        get_preparation_session(
            rollback.workflow_id
        ).workflow_id
        ==
        rollback.workflow_id
    )

    with sqlite_connection(
        write=False
    ) as connection:
        assert int(
            connection.execute(
                """
                SELECT COUNT(*)
                FROM preparation_artifacts
                WHERE workflow_id = ?
                """,
                (
                    rollback.workflow_id,
                ),
            ).fetchone()[0]
        ) == 2

        assert int(
            connection.execute(
                """
                SELECT COUNT(*)
                FROM analysis_artifacts
                WHERE workflow_id = ?
                """,
                (
                    rollback.workflow_id,
                ),
            ).fetchone()[0]
        ) == 3

    print(
        "[PASS] transaction failure restores quarantined payloads"
    )


    # ========================================================
    # SUCCESSFUL FULL DELETE
    # ========================================================

    target = (
        make_workflow(
            display_name=
                "Delete me",

            archived=
                True,
        )
    )

    (
        prep_files,
        analysis_files,
    ) = (
        add_resources(
            target.workflow_id
        )
    )

    result = (
        delete_preparation_workflow(
            workflow_id=
                target.workflow_id,

            confirmation_workflow_id=
                target.workflow_id,

            confirmation_display_name=
                "Delete me",

            expected_revision=
                target.revision,
        )
    )

    assert (
        result.preparation_artifacts_deleted
        ==
        2
    )

    assert (
        result.analysis_artifacts_deleted
        ==
        3
    )

    assert (
        result.preparation_ui_state_deleted
        ==
        1
    )

    assert (
        result.report_selection_deleted
        ==
        1
    )

    assert (
        result.preparation_session_deleted
        ==
        1
    )

    assert (
        result.workflow_metadata_deleted
        ==
        1
    )

    assert (
        result.payload_files_removed_from_live_store
        ==
        5
    )

    assert (
        result.quarantine_cleanup_pending
        is False
    )

    assert all(
        not path.exists()

        for path
        in (
            prep_files
            +
            analysis_files
        )
    )

    with sqlite_connection(
        write=False
    ) as connection:
        for table in [
            "preparation_sessions",
            "preparation_workflow_metadata",
            "preparation_ui_state",
            "preparation_artifacts",
            "report_selection_workflows",
            "analysis_artifacts",
        ]:
            count = int(
                connection.execute(
                    (
                        "SELECT COUNT(*) "
                        "FROM "
                        +
                        table
                        +
                        " WHERE workflow_id = ?"
                    ),
                    (
                        target.workflow_id,
                    ),
                ).fetchone()[0]
            )

            assert (
                count
                ==
                0
            ), (
                table,
                count,
            )

    print(
        "[PASS] full workflow resources deleted"
    )

    print(
        "[PASS] workflow metadata removed by cascade"
    )

    print(
        "[PASS] live payload files removed"
    )


    # ========================================================
    # FINAL TEMP-DB HEALTH
    # ========================================================

    raw_connection = sqlite3.connect(
        database_path
    )

    try:
        integrity = str(
            raw_connection.execute(
                "PRAGMA integrity_check"
            ).fetchone()[0]
        )

        fk_rows = (
            raw_connection.execute(
                "PRAGMA foreign_key_check"
            )
            .fetchall()
        )

    finally:
        raw_connection.close()

    assert (
        integrity.lower()
        ==
        "ok"
    )

    assert (
        fk_rows
        ==
        []
    )

    print(
        "[PASS] temporary SQLite integrity"
    )

    print(
        "[PASS] temporary SQLite foreign keys"
    )


print()
print(
    "PASS - preparation workflow permanent delete v0.1"
)
