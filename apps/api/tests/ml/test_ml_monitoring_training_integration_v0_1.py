from __future__ import annotations


from sklearn.pipeline import (
    Pipeline,
)


import app.ml.classical_executor as executor_module


from app.ml.classical_executor import (
    ML_MONITORING_TRAINING_INTEGRATION_RULE_VERSION,
    ClassicalMLExecutorError,
    execute_classical_ml,
)


from app.ml.model_artifact_data_plane import (
    ml_model_artifact_data_root,
)


from app.ml.model_artifact_store import (
    MLModelArtifactNotFoundError,
    delete_ml_model_artifact,
    get_ml_model_artifact,
    resolve_ml_model_artifact_store_path,
)


from app.ml.monitoring_profile_store import (
    MLMonitoringProfileNotFoundError,
    MLMonitoringProfileStoreError,
    get_ml_monitoring_profile,
)


from app.persistence.sqlite_database import (
    sqlite_connection,
)


from tests.ml.test_classical_ml_executor_v0_1 import (
    classification_contract,
    classification_dataframe,
    isolated_environment,
    patched_handoff,
    seed_preparation_authority,
)


# ============================================================
# HELPERS
# ============================================================


WORKFLOW_ID = (
    "prep:ml-executor"
)


DATASET_ID = (
    "dataset:validated"
)


def expect_error(
    error_type,
    factory,
) -> None:

    try:
        factory()

    except error_type:
        return


    raise AssertionError(
        (
            "Expected "
            f"{error_type.__name__}."
        )
    )


def joblib_files(
) -> list:

    root = (
        ml_model_artifact_data_root(
            resolve_ml_model_artifact_store_path()
        )
    )


    if not root.exists():
        return []


    return list(
        root.rglob(
            "*.joblib"
        )
    )


# ============================================================
# AUTOMATIC PROFILE + SAME x_train + SINGLE FIT
# ============================================================


def test_training_automatically_persists_monitoring_profile(
) -> None:

    with isolated_environment():

        seed_preparation_authority(
            workflow_id=
                WORKFLOW_ID,

            dataset_id=
                DATASET_ID,
        )


        dataframe = (
            classification_dataframe()
        )


        original_builder = (
            executor_module
            .build_ml_monitoring_profile
        )


        original_pipeline_fit = (
            Pipeline.fit
        )


        captured_training_frames = []


        fit_calls = {
            "count":
                0
        }


        def capturing_builder(
            *,
            x_train,
            model_artifact,
        ):

            captured_training_frames.append(
                x_train.copy(
                    deep=True
                )
            )


            return (
                original_builder(
                    x_train=
                        x_train,

                    model_artifact=
                        model_artifact,
                )
            )


        def counting_pipeline_fit(
            self,
            *args,
            **kwargs,
        ):

            fit_calls[
                "count"
            ] += 1


            return (
                original_pipeline_fit(
                    self,
                    *args,
                    **kwargs,
                )
            )


        executor_module.build_ml_monitoring_profile = (
            capturing_builder
        )


        Pipeline.fit = (
            counting_pipeline_fit
        )


        try:

            with patched_handoff(
                dataframe=
                    dataframe,

                workflow_id=
                    WORKFLOW_ID,

                dataset_id=
                    DATASET_ID,
            ):

                result = (
                    execute_classical_ml(
                        training_contract=
                            classification_contract()
                    )
                )


        finally:

            executor_module.build_ml_monitoring_profile = (
                original_builder
            )


            Pipeline.fit = (
                original_pipeline_fit
            )


        assert (
            fit_calls[
                "count"
            ]
            ==
            1
        )


        assert (
            len(
                captured_training_frames
            )
            ==
            1
        )


        captured_x_train = (
            captured_training_frames[
                0
            ]
        )


        assert (
            len(
                captured_x_train
            )
            ==
            result.train_rows
        )


        assert (
            result.train_rows
            <
            len(
                dataframe
            )
        )


        profile = (
            get_ml_monitoring_profile(
                model_id=
                    result
                    .model_artifact
                    .model_id,

                workflow_id=
                    WORKFLOW_ID,
            )
        )


        assert (
            profile.model_id
            ==
            result
            .model_artifact
            .model_id
        )


        assert (
            profile.workflow_id
            ==
            result.workflow_id
        )


        assert (
            profile.dataset_id
            ==
            result.dataset_id
        )


        assert (
            profile.experiment_id
            ==
            result
            .experiment_provenance
            .experiment_id
        )


        assert (
            profile.training_contract_sha256
            ==
            result
            .experiment_provenance
            .training_contract_sha256
        )


        assert (
            profile.preparation_session_revision
            ==
            result
            .experiment_provenance
            .preparation_session_revision
        )


        assert (
            profile.reference_row_count
            ==
            result.train_rows
        )


        assert (
            profile.reference_scope
            ==
            "training_split"
        )


        assert (
            profile.privacy_scope
            ==
            "aggregate_only"
        )


        assert (
            [
                feature.feature_name
                for feature
                in profile.feature_profiles
            ]
            ==
            list(
                result
                .model_artifact
                .training_contract
                .feature_columns
            )
        )


        assert (
            len(
                joblib_files()
            )
            ==
            1
        )


# ============================================================
# MONITORING FAILURE COMPENSATION
# ============================================================


def test_monitoring_failure_compensates_model_artifact(
) -> None:

    with isolated_environment():

        seed_preparation_authority(
            workflow_id=
                WORKFLOW_ID,

            dataset_id=
                DATASET_ID,
        )


        dataframe = (
            classification_dataframe()
        )


        original_register = (
            executor_module
            .register_ml_monitoring_profile
        )


        def failing_register(
            *,
            profile,
        ):
            raise (
                MLMonitoringProfileStoreError(
                    (
                        "Forced monitoring "
                        "persistence failure."
                    )
                )
            )


        executor_module.register_ml_monitoring_profile = (
            failing_register
        )


        try:

            with patched_handoff(
                dataframe=
                    dataframe,

                workflow_id=
                    WORKFLOW_ID,

                dataset_id=
                    DATASET_ID,
            ):

                expect_error(
                    ClassicalMLExecutorError,

                    lambda:
                        execute_classical_ml(
                            training_contract=
                                classification_contract()
                        ),
                )


        finally:

            executor_module.register_ml_monitoring_profile = (
                original_register
            )


        with sqlite_connection(
            write=False
        ) as connection:

            model_count = int(
                connection.execute(
                    """
                    SELECT COUNT(*) AS count
                    FROM ml_model_artifacts
                    """
                )
                .fetchone()[
                    "count"
                ]
            )


            monitoring_count = int(
                connection.execute(
                    """
                    SELECT COUNT(*) AS count
                    FROM ml_monitoring_profiles
                    """
                )
                .fetchone()[
                    "count"
                ]
            )


        assert (
            model_count
            ==
            0
        )


        assert (
            monitoring_count
            ==
            0
        )


        assert (
            joblib_files()
            ==
            []
        )


# ============================================================
# PUBLIC COMPENSATION DELETE
# ============================================================


def test_model_artifact_delete_cascades_monitoring_profile(
) -> None:

    with isolated_environment():

        seed_preparation_authority(
            workflow_id=
                WORKFLOW_ID,

            dataset_id=
                DATASET_ID,
        )


        dataframe = (
            classification_dataframe()
        )


        with patched_handoff(
            dataframe=
                dataframe,

            workflow_id=
                WORKFLOW_ID,

            dataset_id=
                DATASET_ID,
        ):

            result = (
                execute_classical_ml(
                    training_contract=
                        classification_contract()
                )
            )


        model_id = (
            result
            .model_artifact
            .model_id
        )


        assert (
            get_ml_model_artifact(
                model_id=
                    model_id,

                workflow_id=
                    WORKFLOW_ID,
            )
            .model_id
            ==
            model_id
        )


        assert (
            get_ml_monitoring_profile(
                model_id=
                    model_id,

                workflow_id=
                    WORKFLOW_ID,
            )
            .model_id
            ==
            model_id
        )


        assert (
            len(
                joblib_files()
            )
            ==
            1
        )


        delete_ml_model_artifact(
            model_id=
                model_id,

            workflow_id=
                WORKFLOW_ID,
        )


        expect_error(
            MLModelArtifactNotFoundError,

            lambda:
                get_ml_model_artifact(
                    model_id=
                        model_id,

                    workflow_id=
                        WORKFLOW_ID,
                ),
        )


        expect_error(
            MLMonitoringProfileNotFoundError,

            lambda:
                get_ml_monitoring_profile(
                    model_id=
                        model_id,

                    workflow_id=
                        WORKFLOW_ID,
                ),
        )


        assert (
            joblib_files()
            ==
            []
        )


# ============================================================
# VERSION
# ============================================================


def test_monitoring_training_integration_rule_version(
) -> None:

    assert (
        ML_MONITORING_TRAINING_INTEGRATION_RULE_VERSION
        ==
        "ml_monitoring_training_integration_v0.1"
    )


# ============================================================
# MAIN
# ============================================================


def main(
) -> None:

    print(
        "=== DATALENS ML MONITORING TRAINING INTEGRATION v0.1 ==="
    )


    tests = [
        (
            "Automatic Monitoring Profile / same x_train / one fit",
            test_training_automatically_persists_monitoring_profile,
        ),
        (
            "Monitoring failure compensates Model Artifact",
            test_monitoring_failure_compensates_model_artifact,
        ),
        (
            "Model Artifact deletion cascades monitoring",
            test_model_artifact_delete_cascades_monitoring_profile,
        ),
        (
            "Monitoring training integration rule version",
            test_monitoring_training_integration_rule_version,
        ),
    ]


    for (
        label,
        test,
    ) in tests:

        test()

        print(
            f"[PASS] {label}"
        )


    print()

    print(
        "PASS - ML Monitoring Training Integration v0.1"
    )


if __name__ == "__main__":
    main()
