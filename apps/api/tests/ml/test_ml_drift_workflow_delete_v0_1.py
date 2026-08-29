from __future__ import annotations


import os


from contextlib import (
    contextmanager,
)


from pathlib import (
    Path,
)


from app.ml.contracts import (
    MLTrainingContract,
)


from app.ml.drift_evaluation_store import (
    get_ml_drift_evaluation,
    register_ml_drift_evaluation,
)


from app.ml.drift_evaluator import (
    evaluate_ml_drift,
)


from app.ml.model_artifact_store import (
    register_ml_model_artifact,
)


from app.ml.monitoring_profile_builder import (
    build_ml_monitoring_profile,
)


from app.ml.monitoring_profile_store import (
    get_ml_monitoring_profile,
    register_ml_monitoring_profile,
)


from app.persistence.sqlite_database import (
    SQLITE_SCHEMA_VERSION,
    sqlite_connection,
    sqlite_schema_version,
)


from app.preparation.preparation_session import (
    archive_preparation_session,
    create_preparation_session,
)


from app.preparation.preparation_workflow_delete import (
    delete_preparation_workflow,
)


from tests.ml.test_ml_monitoring_profile_store_v0_1 import (
    isolated_environment as monitoring_isolated_environment,
    training_frame,
)


# ============================================================
# ENVIRONMENT
# ============================================================


@contextmanager
def isolated_environment(
):
    previous_quarantine = (
        os.environ.get(
            "DATALENS_WORKFLOW_DELETE_QUARANTINE_ROOT"
        )
    )


    with monitoring_isolated_environment() as root:

        quarantine_root = (
            root
            /
            "workflow-delete-quarantine"
        )


        os.environ[
            "DATALENS_WORKFLOW_DELETE_QUARANTINE_ROOT"
        ] = str(
            quarantine_root
        )


        try:
            yield root

        finally:
            if previous_quarantine is None:
                os.environ.pop(
                    "DATALENS_WORKFLOW_DELETE_QUARANTINE_ROOT",
                    None,
                )

            else:
                os.environ[
                    "DATALENS_WORKFLOW_DELETE_QUARANTINE_ROOT"
                ] = previous_quarantine


# ============================================================
# SEED
# ============================================================


def seed_observability_workflow(
    *,
    root: Path,
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


    preparation_root = (
        root
        /
        "preparation"
        /
        session.workflow_id.replace(
            ":",
            "-",
        )
    )


    relative_data_path = (
        Path(
            "data"
        )
        /
        "validated.parquet"
    )


    absolute_data_path = (
        preparation_root
        /
        relative_data_path
    )


    absolute_data_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    absolute_data_path.write_bytes(
        b"DATALENS-VALIDATED-DATASET"
    )


    with sqlite_connection(
        write=True
    ) as connection:

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
                ?,
                ?,
                ?,
                ?,
                ?,
                ?,
                ?,
                ?,
                ?,
                ?,
                ?
            )
            """,
            (
                str(
                    preparation_root.resolve()
                ),

                session.workflow_id,

                "dataset:validated",

                "validated.parquet",

                "clean",

                6,

                2,

                "[]",

                "[]",

                "{}",

                str(
                    relative_data_path
                ).replace(
                    "\\",
                    "/",
                ),
            ),
        )


    contract = (
        MLTrainingContract(
            workflow_id=
                session.workflow_id,

            dataset_id=
                "dataset:validated",

            problem_type=
                "classification",

            target_column=
                "target",

            feature_columns=[
                "age",
                "segment",
            ],

            categorical_feature_columns=[
                "segment"
            ],

            estimator_key=
                "logistic_regression",
        )
    )


    frame = (
        training_frame()
    )


    artifact = (
        register_ml_model_artifact(
            training_contract=
                contract,

            metrics={
                "accuracy":
                    0.75,

                "balanced_accuracy":
                    0.75,

                "f1_macro":
                    0.75,
            },

            train_rows=
                len(
                    frame
                ),

            test_rows=
                2,

            model_bytes=
                b"DATALENS-DRIFT-WORKFLOW-DELETE",

            preparation_session_revision=
                session.revision,

            created_at_utc=
                "2026-08-29T15:00:00+00:00",
        )
    )


    profile = (
        build_ml_monitoring_profile(
            x_train=
                frame,

            model_artifact=
                artifact,
        )
    )


    register_ml_monitoring_profile(
        profile=
            profile
    )


    drift = (
        evaluate_ml_drift(
            observed_features=
                frame,

            observed_dataset_id=
                "dataset:observed",

            monitoring_profile=
                profile,

            model_artifact=
                artifact,
        )
    )


    register_ml_drift_evaluation(
        evaluation=
            drift
    )


    archive_preparation_session(
        session.workflow_id
    )


    return (
        session,
        artifact,
        profile,
        drift,
        absolute_data_path,
    )


# ============================================================
# SUCCESSFUL CASCADE
# ============================================================


def test_permanent_workflow_delete_cascades_observability(
) -> None:

    with isolated_environment() as root:

        (
            session,
            artifact,
            profile,
            drift,
            preparation_payload,
        ) = (
            seed_observability_workflow(
                root=
                    root,

                display_name=
                    "ML drift delete success",
            )
        )


        assert (
            preparation_payload.is_file()
        )


        result = (
            delete_preparation_workflow(
                workflow_id=
                    session.workflow_id,

                confirmation_workflow_id=
                    session.workflow_id,

                confirmation_display_name=
                    "ML drift delete success",

                expected_revision=
                    session.revision,
            )
        )


        assert (
            result.ml_model_artifacts_deleted
            ==
            1
        )


        assert (
            not preparation_payload.exists()
        )


        with sqlite_connection(
            write=False
        ) as connection:

            counts = {
                table:
                    int(
                        connection.execute(
                            (
                                "SELECT COUNT(*) "
                                f"FROM {table} "
                                "WHERE workflow_id = ?"
                            ),
                            (
                                session.workflow_id,
                            ),
                        )
                        .fetchone()[0]
                    )

                for table
                in [
                    "ml_model_artifacts",
                    "ml_monitoring_profiles",
                    "ml_drift_evaluations",
                ]
            }


        assert counts == {
            "ml_model_artifacts": 0,
            "ml_monitoring_profiles": 0,
            "ml_drift_evaluations": 0,
        }


# ============================================================
# TRANSACTION ROLLBACK
# ============================================================


def test_workflow_delete_rollback_restores_observability(
) -> None:

    with isolated_environment() as root:

        (
            session,
            artifact,
            profile,
            drift,
            preparation_payload,
        ) = (
            seed_observability_workflow(
                root=
                    root,

                display_name=
                    "ML drift delete rollback",
            )
        )


        def injected_failure(
            stage: str,
        ) -> None:

            if (
                stage
                ==
                "before_commit"
            ):
                raise RuntimeError(
                    "injected drift rollback"
                )


        try:
            delete_preparation_workflow(
                workflow_id=
                    session.workflow_id,

                confirmation_workflow_id=
                    session.workflow_id,

                confirmation_display_name=
                    "ML drift delete rollback",

                expected_revision=
                    session.revision,

                _failure_hook=
                    injected_failure,
            )

        except RuntimeError as error:

            assert (
                "injected drift rollback"
                in
                str(
                    error
                )
            )

        else:
            raise AssertionError(
                (
                    "Injected workflow-delete "
                    "failure should propagate."
                )
            )


        # Files moved to quarantine must be restored.
        assert (
            preparation_payload.is_file()
        )


        # SQLite DELETE + cascades occurred inside the same
        # transaction and therefore must all be rolled back.
        restored_profile = (
            get_ml_monitoring_profile(
                model_id=
                    artifact.model_id,

                workflow_id=
                    session.workflow_id,
            )
        )


        restored_drift = (
            get_ml_drift_evaluation(
                evaluation_id=
                    drift.evaluation_id,

                workflow_id=
                    session.workflow_id,
            )
        )


        assert (
            restored_profile
            ==
            profile
        )


        assert (
            restored_drift
            ==
            drift
        )


        with sqlite_connection(
            write=False
        ) as connection:

            counts = {
                table:
                    int(
                        connection.execute(
                            (
                                "SELECT COUNT(*) "
                                f"FROM {table} "
                                "WHERE workflow_id = ?"
                            ),
                            (
                                session.workflow_id,
                            ),
                        )
                        .fetchone()[0]
                    )

                for table
                in [
                    "ml_model_artifacts",
                    "ml_monitoring_profiles",
                    "ml_drift_evaluations",
                ]
            }


        assert counts == {
            "ml_model_artifacts": 1,
            "ml_monitoring_profiles": 1,
            "ml_drift_evaluations": 1,
        }


# ============================================================
# SCHEMA
# ============================================================


def test_current_schema(
) -> None:

    with isolated_environment():

        assert (
            SQLITE_SCHEMA_VERSION
            >=
            11
        )


        assert (
            sqlite_schema_version()
            ==
            SQLITE_SCHEMA_VERSION
        )


# ============================================================
# MAIN
# ============================================================


def main(
) -> None:

    print(
        "=== DATALENS ML DRIFT WORKFLOW DELETE v0.1 ==="
    )

    print()


    test_current_schema()

    print(
        "[PASS] Current SQLite schema supports Drift history"
    )


    test_permanent_workflow_delete_cascades_observability()

    print(
        (
            "[PASS] Permanent workflow deletion cascades "
            "Model / Monitoring / Drift metadata"
        )
    )


    test_workflow_delete_rollback_restores_observability()

    print(
        (
            "[PASS] Workflow deletion rollback restores "
            "Monitoring / Drift history"
        )
    )


    print()

    print(
        "PASS - ML Drift Workflow Delete v0.1"
    )


if __name__ == "__main__":
    main()
