from __future__ import annotations


import app.ml.classical_executor as executor_module


from app.ml.classical_executor import (
    ClassicalMLExecutorError,
    execute_classical_ml,
)


from app.ml.experiment_provenance import (
    ml_training_contract_sha256,
)


from app.ml.model_artifact_store import (
    get_ml_model_artifact,
    list_ml_model_artifacts,
)


from app.persistence.sqlite_database import (
    sqlite_connection,
)


from tests.ml.test_classical_ml_executor_v0_1 import (
    isolated_environment,
    patched_handoff,
    regression_contract,
    regression_dataframe,
    seed_preparation_authority,
)


# ============================================================
# CURRENT EXECUTION PROVENANCE
# ============================================================


def test_classical_ml_persists_experiment_provenance(
) -> None:

    with isolated_environment():

        contract = (
            regression_contract()
        )


        seed_preparation_authority(
            workflow_id=
                contract.workflow_id,

            dataset_id=
                contract.dataset_id,
        )


        with patched_handoff(
            dataframe=
                regression_dataframe(),

            workflow_id=
                contract.workflow_id,

            dataset_id=
                contract.dataset_id,
        ):

            result = (
                execute_classical_ml(
                    training_contract=
                        contract
                )
            )


        provenance = (
            result
            .experiment_provenance
        )


        assert (
            provenance
            ==
            result
            .model_artifact
            .experiment_provenance
        )


        assert (
            provenance.experiment_id
            .startswith(
                "experiment:"
            )
        )


        assert (
            provenance.workflow_id
            ==
            contract.workflow_id
        )


        assert (
            provenance.dataset_id
            ==
            contract.dataset_id
        )


        assert (
            provenance.preparation_session_revision
            ==
            0
        )


        assert (
            provenance.model_id
            ==
            result
            .model_artifact
            .model_id
        )


        assert (
            provenance.training_contract_sha256
            ==
            ml_training_contract_sha256(
                contract
            )
        )


        assert (
            provenance.train_rows
            ==
            result.train_rows
        )


        assert (
            provenance.test_rows
            ==
            result.test_rows
        )


        assert (
            provenance.metrics
            ==
            result.metrics
        )


# ============================================================
# SQLITE RESTORE
# ============================================================


def test_experiment_provenance_survives_artifact_restore(
) -> None:

    with isolated_environment():

        contract = (
            regression_contract()
        )


        seed_preparation_authority(
            workflow_id=
                contract.workflow_id,

            dataset_id=
                contract.dataset_id,
        )


        with patched_handoff(
            dataframe=
                regression_dataframe(),

            workflow_id=
                contract.workflow_id,

            dataset_id=
                contract.dataset_id,
        ):

            result = (
                execute_classical_ml(
                    training_contract=
                        contract
                )
            )


        restored = (
            get_ml_model_artifact(
                workflow_id=
                    contract.workflow_id,

                model_id=
                    result
                    .model_artifact
                    .model_id,
            )
        )


        assert (
            restored
            ==
            result.model_artifact
        )


        assert (
            restored.experiment_provenance
            ==
            result.experiment_provenance
        )


        artifacts = (
            list_ml_model_artifacts(
                workflow_id=
                    contract.workflow_id
            )
        )


        assert (
            len(
                artifacts
            )
            ==
            1
        )


        assert (
            artifacts[
                0
            ]
            .experiment_provenance
            ==
            result.experiment_provenance
        )


# ============================================================
# PREPARATION REVISION RACE
# ============================================================


def test_preparation_revision_change_before_persistence_is_fail_closed(
) -> None:

    with isolated_environment():

        contract = (
            regression_contract()
        )


        seed_preparation_authority(
            workflow_id=
                contract.workflow_id,

            dataset_id=
                contract.dataset_id,
        )


        original_serialize = (
            executor_module
            ._serialize_fitted_estimator
        )


        def serialize_after_revision_change(
            estimator,
        ) -> bytes:

            model_bytes = (
                original_serialize(
                    estimator
                )
            )


            with sqlite_connection(
                write=True
            ) as connection:

                connection.execute(
                    """
                    UPDATE preparation_sessions

                    SET
                        revision =
                            revision + 1

                    WHERE
                        workflow_id = ?
                    """,
                    (
                        contract.workflow_id,
                    ),
                )


            return model_bytes


        executor_module._serialize_fitted_estimator = (
            serialize_after_revision_change
        )


        try:

            with patched_handoff(
                dataframe=
                    regression_dataframe(),

                workflow_id=
                    contract.workflow_id,

                dataset_id=
                    contract.dataset_id,
            ):

                try:
                    execute_classical_ml(
                        training_contract=
                            contract
                    )

                except ClassicalMLExecutorError:
                    pass

                else:
                    raise AssertionError(
                        (
                            "Preparation revision race "
                            "must fail Classical ML closed."
                        )
                    )


        finally:
            executor_module._serialize_fitted_estimator = (
                original_serialize
            )


        assert (
            list_ml_model_artifacts(
                workflow_id=
                    contract.workflow_id
            )
            ==
            []
        )


# ============================================================
# LEGACY MODEL ARTIFACT COMPATIBILITY
# ============================================================


def test_legacy_artifact_without_experiment_provenance_remains_valid(
) -> None:

    from app.ml.model_artifacts import (
        MLModelArtifactRecord,
    )


    contract = (
        regression_contract()
    )


    legacy = (
        MLModelArtifactRecord(
            model_id=
                "model:legacy",

            workflow_id=
                contract.workflow_id,

            dataset_id=
                contract.dataset_id,

            training_contract=
                contract,

            metrics={
                "rmse":
                    10.0
            },

            train_rows=
                80,

            test_rows=
                20,

            created_at_utc=
                "2026-08-28T20:00:00+00:00",

            model_path=
                "data/legacy.joblib",

            model_file_bytes=
                128,

            model_sha256=(
                "a"
                *
                64
            ),
        )
    )


    assert (
        legacy.experiment_provenance
        is None
    )


# ============================================================
# MAIN
# ============================================================


def main(
) -> None:

    print(
        "=== DATALENS ML EXPERIMENT PROVENANCE EXECUTOR v0.1 ==="
    )

    print()


    test_classical_ml_persists_experiment_provenance()

    print(
        "Classical ML persists Experiment Provenance: PASS"
    )


    test_experiment_provenance_survives_artifact_restore()

    print(
        "Experiment Provenance survives SQLite restore: PASS"
    )


    test_preparation_revision_change_before_persistence_is_fail_closed()

    print(
        "Preparation revision race is fail-closed: PASS"
    )


    test_legacy_artifact_without_experiment_provenance_remains_valid()

    print(
        "Legacy v8 Model Artifact compatibility: PASS"
    )


    print()

    print(
        "ML Experiment Provenance Executor v0.1: PASS"
    )


if __name__ == "__main__":
    main()
