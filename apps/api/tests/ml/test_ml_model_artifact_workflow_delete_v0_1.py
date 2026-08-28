from __future__ import annotations


import os
import tempfile


from pathlib import (
    Path,
)


print(
    (
        "=== DATALENS ML MODEL ARTIFACT "
        "WORKFLOW DELETE v0.1 ==="
    )
)


with tempfile.TemporaryDirectory(
    prefix="datalens-ml-workflow-delete-"
) as temporary_directory:

    root = Path(
        temporary_directory
    )


    database_path = (
        root
        /
        "datalens.sqlite3"
    )


    quarantine_root = (
        root
        /
        "quarantine"
    )


    model_store = (
        root
        /
        "ml"
        /
        "model_artifacts.json"
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


    from app.ml.contracts import (
        MLTrainingContract,
    )


    from app.ml.model_artifact_data_plane import (
        ml_model_artifact_data_root,
        read_ml_model_binary,
        write_ml_model_binary,
    )


    from app.ml.model_artifact_index import (
        get_ml_model_artifact_index_entry,
        upsert_ml_model_artifact_index_entry,
    )


    from app.ml.model_artifacts import (
        MLModelArtifactRecord,
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
        PreparationWorkflowDeleteIntegrityError,
        delete_preparation_workflow,
    )


    assert (
        sqlite_schema_version()
        ==
        9
    )


    # ========================================================
    # HELPERS
    # ========================================================


    def make_archived_workflow(
        *,
        display_name: str,
    ):
        session = (
            create_preparation_session(
                selected_analysis_dataset_ids=[
                    "dataset:validated",
                ],

                display_name=
                    display_name,
            )
        )


        archive_preparation_session(
            session.workflow_id
        )


        return session


    def create_model_artifact(
        *,
        workflow_id: str,
        model_id: str,
        model_bytes: bytes,
    ):
        binary_info = (
            write_ml_model_binary(
                store_path=
                    model_store,

                model_id=
                    model_id,

                model_bytes=
                    model_bytes,
            )
        )


        contract = (
            MLTrainingContract(
                workflow_id=
                    workflow_id,

                dataset_id=
                    "dataset:validated",

                problem_type=
                    "regression",

                target_column=
                    "revenue",

                feature_columns=[
                    "age",
                    "tenure",
                ],

                estimator_key=
                    "linear_regression",
            )
        )


        record = (
            MLModelArtifactRecord(
                model_id=
                    model_id,

                workflow_id=
                    workflow_id,

                dataset_id=
                    "dataset:validated",

                training_contract=
                    contract,

                metrics={
                    "mae":
                        10.0,

                    "r2":
                        0.80,
                },

                train_rows=
                    80,

                test_rows=
                    20,

                created_at_utc=
                    "2026-08-28T18:00:00+00:00",

                **binary_info,
            )
        )


        upsert_ml_model_artifact_index_entry(
            store_path=
                model_store,

            entry=
                record.model_dump(
                    mode="json"
                ),
        )


        binary_path = (
            ml_model_artifact_data_root(
                model_store
            )
            /
            binary_info[
                "model_path"
            ]
        )


        return (
            binary_info,
            binary_path,
        )


    # ========================================================
    # SUCCESSFUL PERMANENT DELETE
    # ========================================================


    success = (
        make_archived_workflow(
            display_name=
                "ML delete success"
        )
    )


    (
        success_info,
        success_binary,
    ) = (
        create_model_artifact(
            workflow_id=
                success.workflow_id,

            model_id=
                "model:delete-success",

            model_bytes=
                b"DATALENS-ML-DELETE-SUCCESS",
        )
    )


    assert (
        success_binary.is_file()
    )


    assert (
        get_ml_model_artifact_index_entry(
            store_path=
                model_store,

            model_id=
                "model:delete-success",
        )
        is not None
    )


    result = (
        delete_preparation_workflow(
            workflow_id=
                success.workflow_id,

            confirmation_workflow_id=
                success.workflow_id,

            confirmation_display_name=
                "ML delete success",

            expected_revision=
                success.revision,
        )
    )


    assert (
        result.ml_model_artifacts_deleted
        ==
        1
    )


    assert (
        result.payload_files_removed_from_live_store
        ==
        1
    )


    assert (
        not success_binary.exists()
    )


    assert (
        get_ml_model_artifact_index_entry(
            store_path=
                model_store,

            model_id=
                "model:delete-success",
        )
        is None
    )


    with sqlite_connection(
        write=False
    ) as connection:

        assert (
            int(
                connection.execute(
                    """
                    SELECT COUNT(*)
                    FROM preparation_sessions
                    WHERE workflow_id = ?
                    """,
                    (
                        success.workflow_id,
                    ),
                ).fetchone()[0]
            )
            ==
            0
        )


        assert (
            int(
                connection.execute(
                    """
                    SELECT COUNT(*)
                    FROM ml_model_artifacts
                    WHERE workflow_id = ?
                    """,
                    (
                        success.workflow_id,
                    ),
                ).fetchone()[0]
            )
            ==
            0
        )


    print(
        (
            "[PASS] permanent workflow deletion removes "
            "ML metadata and model binary"
        )
    )


    # ========================================================
    # ROLLBACK RESTORES MODEL BINARY + SQLITE ROW
    # ========================================================


    rollback = (
        make_archived_workflow(
            display_name=
                "ML rollback"
        )
    )


    (
        rollback_info,
        rollback_binary,
    ) = (
        create_model_artifact(
            workflow_id=
                rollback.workflow_id,

            model_id=
                "model:rollback",

            model_bytes=
                b"DATALENS-ML-ROLLBACK",
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
                "injected ML rollback test"
            )


    try:
        delete_preparation_workflow(
            workflow_id=
                rollback.workflow_id,

            confirmation_workflow_id=
                rollback.workflow_id,

            confirmation_display_name=
                "ML rollback",

            expected_revision=
                rollback.revision,

            _failure_hook=
                injected_failure,
        )

    except RuntimeError as error:
        assert (
            "injected ML rollback test"
            in
            str(
                error
            )
        )

    else:
        raise AssertionError(
            (
                "Injected ML delete failure "
                "should have propagated."
            )
        )


    assert (
        rollback_binary.is_file()
    )


    restored_entry = (
        get_ml_model_artifact_index_entry(
            store_path=
                model_store,

            model_id=
                "model:rollback",
        )
    )


    assert (
        restored_entry
        is not None
    )


    assert (
        read_ml_model_binary(
            store_path=
                model_store,

            entry=
                restored_entry,
        )
        ==
        b"DATALENS-ML-ROLLBACK"
    )


    assert (
        get_preparation_session(
            rollback.workflow_id
        ).workflow_id
        ==
        rollback.workflow_id
    )


    print(
        (
            "[PASS] rollback restores ML binary "
            "and SQLite metadata"
        )
    )


    # ========================================================
    # MISSING MODEL BINARY FAILS CLOSED
    # ========================================================


    missing = (
        make_archived_workflow(
            display_name=
                "ML missing binary"
        )
    )


    missing_contract = (
        MLTrainingContract(
            workflow_id=
                missing.workflow_id,

            dataset_id=
                "dataset:validated",

            problem_type=
                "regression",

            target_column=
                "revenue",

            feature_columns=[
                "age",
            ],

            estimator_key=
                "linear_regression",
        )
    )


    missing_record = (
        MLModelArtifactRecord(
            model_id=
                "model:missing",

            workflow_id=
                missing.workflow_id,

            dataset_id=
                "dataset:validated",

            training_contract=
                missing_contract,

            metrics={
                "mae":
                    10.0,
            },

            train_rows=
                80,

            test_rows=
                20,

            created_at_utc=
                "2026-08-28T18:00:00+00:00",

            model_path=
                "data/model-missing.joblib",

            model_file_bytes=
                100,

            model_sha256=
                "a"
                *
                64,
        )
    )


    upsert_ml_model_artifact_index_entry(
        store_path=
            model_store,

        entry=
            missing_record.model_dump(
                mode="json"
            ),
    )


    try:
        delete_preparation_workflow(
            workflow_id=
                missing.workflow_id,

            confirmation_workflow_id=
                missing.workflow_id,

            confirmation_display_name=
                "ML missing binary",

            expected_revision=
                missing.revision,
        )

    except PreparationWorkflowDeleteIntegrityError:
        pass

    else:
        raise AssertionError(
            (
                "Missing Model Artifact binary "
                "must block permanent deletion."
            )
        )


    assert (
        get_preparation_session(
            missing.workflow_id
        ).workflow_id
        ==
        missing.workflow_id
    )


    assert (
        get_ml_model_artifact_index_entry(
            store_path=
                model_store,

            model_id=
                "model:missing",
        )
        is not None
    )


    print(
        (
            "[PASS] missing ML binary blocks deletion "
            "without losing metadata"
        )
    )


print()

print(
    "ML Model Artifact Workflow Delete v0.1: PASS"
)