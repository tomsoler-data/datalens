from __future__ import annotations

import os
import tempfile

from pathlib import Path


print(
    "=== DATALENS WORKFLOW DELETE API v0.1 ==="
)


with tempfile.TemporaryDirectory(
    prefix=
        "datalens-delete-api-"
) as temporary_directory:
    root = Path(
        temporary_directory
    )

    database = (
        root
        /
        "datalens.sqlite3"
    )

    quarantine = (
        root
        /
        "workflow_delete_quarantine"
    )

    os.environ[
        "DATALENS_SQLITE_PATH"
    ] = str(
        database
    )

    os.environ[
        "DATALENS_WORKFLOW_DELETE_QUARANTINE_ROOT"
    ] = str(
        quarantine
    )


    import app.main as main_module

    from fastapi.testclient import (
        TestClient,
    )


    recovery_calls = []


    original_recovery = (
        main_module
        .recover_pending_workflow_deletions
    )


    def fake_startup_recovery():
        recovery_calls.append(
            "called"
        )

        return 0


    main_module.recover_pending_workflow_deletions = (
        fake_startup_recovery
    )


    try:
        with TestClient(
            main_module.app
        ) as client:

            # =================================================
            # STARTUP RECOVERY
            # =================================================

            assert (
                recovery_calls
                ==
                [
                    "called",
                ]
            )

            print(
                "[PASS] startup invokes workflow-delete recovery"
            )


            # =================================================
            # CORS DELETE
            # =================================================

            cors = client.options(
                "/preparation/sessions",

                headers={
                    "Origin":
                        "http://localhost:3000",

                    "Access-Control-Request-Method":
                        "DELETE",
                },
            )

            assert (
                cors.status_code
                ==
                200
            ), cors.text

            methods = (
                cors.headers.get(
                    "access-control-allow-methods",
                    ""
                )
            )

            assert (
                "DELETE"
                in
                methods
            ), methods

            print(
                "[PASS] CORS allows DELETE"
            )


            # =================================================
            # CREATE ACTIVE WORKFLOW
            # =================================================

            created_response = client.post(
                "/preparation/sessions",

                json={
                    "selected_analysis_dataset_ids":
                        [
                            "dataset:root",
                        ],

                    "display_name":
                        "Workflow API delete test",
                },
            )

            assert (
                created_response.status_code
                ==
                201
            ), created_response.text

            created = (
                created_response.json()
            )

            workflow_id = str(
                created[
                    "workflow_id"
                ]
            )

            revision = int(
                created[
                    "revision"
                ]
            )


            # =================================================
            # ACTIVE DELETE REFUSED
            # =================================================

            active_delete = client.request(
                "DELETE",

                (
                    "/preparation/sessions/"
                    +
                    workflow_id
                ),

                json={
                    "confirmation_workflow_id":
                        workflow_id,

                    "confirmation_display_name":
                        "Workflow API delete test",

                    "expected_revision":
                        revision,
                },
            )

            assert (
                active_delete.status_code
                ==
                409
            ), active_delete.text

            assert (
                active_delete.json()[
                    "detail"
                ][
                    "error"
                ]
                ==
                "preparation_workflow_delete_requires_archive"
            )

            print(
                "[PASS] API rejects active workflow deletion"
            )


            # =================================================
            # ARCHIVE FIRST
            # =================================================

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

            archived = (
                archived_response.json()
            )

            assert (
                archived[
                    "archived"
                ]
                is True
            )

            assert (
                int(
                    archived[
                        "session"
                    ][
                        "revision"
                    ]
                )
                ==
                revision
            )

            print(
                "[PASS] workflow archived before delete"
            )


            # =================================================
            # WRONG NAME
            # =================================================

            wrong_name = client.request(
                "DELETE",

                (
                    "/preparation/sessions/"
                    +
                    workflow_id
                ),

                json={
                    "confirmation_workflow_id":
                        workflow_id,

                    "confirmation_display_name":
                        "Wrong name",

                    "expected_revision":
                        revision,
                },
            )

            assert (
                wrong_name.status_code
                ==
                409
            ), wrong_name.text

            assert (
                wrong_name.json()[
                    "detail"
                ][
                    "error"
                ]
                ==
                "preparation_workflow_delete_confirmation_conflict"
            )

            print(
                "[PASS] API rejects name confirmation mismatch"
            )


            # =================================================
            # WRONG WORKFLOW ID
            # =================================================

            wrong_id = client.request(
                "DELETE",

                (
                    "/preparation/sessions/"
                    +
                    workflow_id
                ),

                json={
                    "confirmation_workflow_id":
                        "prep:wrong",

                    "confirmation_display_name":
                        "Workflow API delete test",

                    "expected_revision":
                        revision,
                },
            )

            assert (
                wrong_id.status_code
                ==
                409
            ), wrong_id.text

            assert (
                wrong_id.json()[
                    "detail"
                ][
                    "error"
                ]
                ==
                "preparation_workflow_delete_confirmation_conflict"
            )

            print(
                "[PASS] API rejects workflow-id confirmation mismatch"
            )


            # =================================================
            # WRONG REVISION
            # =================================================

            wrong_revision = client.request(
                "DELETE",

                (
                    "/preparation/sessions/"
                    +
                    workflow_id
                ),

                json={
                    "confirmation_workflow_id":
                        workflow_id,

                    "confirmation_display_name":
                        "Workflow API delete test",

                    "expected_revision":
                        revision
                        +
                        1,
                },
            )

            assert (
                wrong_revision.status_code
                ==
                409
            ), wrong_revision.text

            assert (
                wrong_revision.json()[
                    "detail"
                ][
                    "error"
                ]
                ==
                "preparation_workflow_delete_revision_conflict"
            )

            print(
                "[PASS] API rejects revision conflict"
            )


            # =================================================
            # REQUEST MODEL
            # =================================================

            invalid_revision = client.request(
                "DELETE",

                (
                    "/preparation/sessions/"
                    +
                    workflow_id
                ),

                json={
                    "confirmation_workflow_id":
                        workflow_id,

                    "confirmation_display_name":
                        "Workflow API delete test",

                    "expected_revision":
                        -1,
                },
            )

            assert (
                invalid_revision.status_code
                ==
                422
            ), invalid_revision.text

            print(
                "[PASS] API validates destructive request model"
            )


            # =================================================
            # SUCCESS
            # =================================================

            deleted_response = client.request(
                "DELETE",

                (
                    "/preparation/sessions/"
                    +
                    workflow_id
                ),

                json={
                    "confirmation_workflow_id":
                        workflow_id,

                    "confirmation_display_name":
                        "Workflow API delete test",

                    "expected_revision":
                        revision,
                },
            )

            assert (
                deleted_response.status_code
                ==
                200
            ), deleted_response.text

            deleted = (
                deleted_response.json()
            )

            assert (
                deleted[
                    "workflow_id"
                ]
                ==
                workflow_id
            )

            assert (
                deleted[
                    "display_name"
                ]
                ==
                "Workflow API delete test"
            )

            assert (
                deleted[
                    "preparation_session_deleted"
                ]
                ==
                1
            )

            assert (
                deleted[
                    "workflow_metadata_deleted"
                ]
                ==
                1
            )

            assert (
                deleted[
                    "preparation_artifacts_deleted"
                ]
                ==
                0
            )

            assert (
                deleted[
                    "analysis_artifacts_deleted"
                ]
                ==
                0
            )

            assert (
                deleted[
                    "payload_files_removed_from_live_store"
                ]
                ==
                0
            )

            print(
                "[PASS] API permanently deletes archived workflow"
            )


            # =================================================
            # ROOT + CATALOG GONE
            # =================================================

            read_deleted = client.get(
                (
                    "/preparation/sessions/"
                    +
                    workflow_id
                )
            )

            assert (
                read_deleted.status_code
                ==
                404
            ), read_deleted.text


            catalog = client.get(
                "/preparation/sessions"
            )

            assert (
                catalog.status_code
                ==
                200
            ), catalog.text

            ids = [
                item[
                    "session"
                ][
                    "workflow_id"
                ]

                for item
                in catalog.json()[
                    "sessions"
                ]
            ]

            assert (
                workflow_id
                not in
                ids
            )

            print(
                "[PASS] deleted workflow disappears from catalog"
            )


            # =================================================
            # REPEATED DELETE
            # =================================================

            repeated = client.request(
                "DELETE",

                (
                    "/preparation/sessions/"
                    +
                    workflow_id
                ),

                json={
                    "confirmation_workflow_id":
                        workflow_id,

                    "confirmation_display_name":
                        "Workflow API delete test",

                    "expected_revision":
                        revision,
                },
            )

            assert (
                repeated.status_code
                ==
                404
            ), repeated.text

            assert (
                repeated.json()[
                    "detail"
                ][
                    "error"
                ]
                ==
                "preparation_workflow_not_found"
            )

            print(
                "[PASS] repeated delete fails closed with 404"
            )


    finally:
        main_module.recover_pending_workflow_deletions = (
            original_recovery
        )


print()
print(
    "PASS - workflow delete API v0.1"
)
