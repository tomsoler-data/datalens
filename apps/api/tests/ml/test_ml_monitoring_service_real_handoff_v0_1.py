from __future__ import annotations


import os


from contextlib import (
    contextmanager,
)


from pathlib import (
    Path,
)


from tempfile import (
    TemporaryDirectory,
)


import pandas as pd


from app.ml.contracts import (
    MLTrainingContract,
)


from app.ml.drift_evaluation_store import (
    list_ml_drift_evaluations_for_model,
)


from app.ml.model_artifact_store import (
    ML_MODEL_ARTIFACT_STORE_PATH_ENV,
    register_ml_model_artifact,
)


from app.ml.monitoring_profile_builder import (
    build_ml_monitoring_profile,
)


from app.ml.monitoring_profile_store import (
    register_ml_monitoring_profile,
)


from app.ml.monitoring_service import (
    run_ml_monitoring,
)


from app.persistence.sqlite_database import (
    DATALENS_SQLITE_PATH_ENV,
)


from app.preparation.analysis_input_handoff import (
    load_validated_analysis_input,
)


from app.preparation.preparation_artifact_store import (
    PREPARATION_ARTIFACT_STORE_ENV,
    put_preparation_artifact,
)


from app.preparation.preparation_session import (
    create_preparation_session,
    record_analysis_output_selection,
    record_required_stage_signal,
    record_validation_stage_signal,
)


from app.preparation.preparation_workflow import (
    PreparationStage,
)


# ============================================================
# ISOLATION
# ============================================================


@contextmanager
def isolated_real_handoff_environment(
):

    previous_sqlite = os.environ.get(
        DATALENS_SQLITE_PATH_ENV
    )

    previous_model_store = os.environ.get(
        ML_MODEL_ARTIFACT_STORE_PATH_ENV
    )

    previous_preparation_store = os.environ.get(
        PREPARATION_ARTIFACT_STORE_ENV
    )


    with TemporaryDirectory(
        prefix=
            "datalens-monitoring-real-handoff-"
    ) as directory:

        root = Path(
            directory
        )


        os.environ[
            DATALENS_SQLITE_PATH_ENV
        ] = str(
            root
            /
            "datalens.sqlite3"
        )


        os.environ[
            ML_MODEL_ARTIFACT_STORE_PATH_ENV
        ] = str(
            root
            /
            "ml"
            /
            "model_artifacts.json"
        )


        os.environ[
            PREPARATION_ARTIFACT_STORE_ENV
        ] = str(
            root
            /
            "preparation"
            /
            "artifacts"
        )


        try:
            yield root

        finally:

            if previous_sqlite is None:
                os.environ.pop(
                    DATALENS_SQLITE_PATH_ENV,
                    None,
                )

            else:
                os.environ[
                    DATALENS_SQLITE_PATH_ENV
                ] = previous_sqlite


            if previous_model_store is None:
                os.environ.pop(
                    ML_MODEL_ARTIFACT_STORE_PATH_ENV,
                    None,
                )

            else:
                os.environ[
                    ML_MODEL_ARTIFACT_STORE_PATH_ENV
                ] = previous_model_store


            if previous_preparation_store is None:
                os.environ.pop(
                    PREPARATION_ARTIFACT_STORE_ENV,
                    None,
                )

            else:
                os.environ[
                    PREPARATION_ARTIFACT_STORE_ENV
                ] = previous_preparation_store


# ============================================================
# DATA
# ============================================================


def validated_dataframe(
) -> pd.DataFrame:

    return (
        pd.DataFrame(
            {
                "age": [
                    20.0,
                    25.0,
                    30.0,
                    35.0,
                    40.0,
                    45.0,
                ],

                "segment": [
                    "standard",
                    "premium",
                    "standard",
                    "premium",
                    "standard",
                    "premium",
                ],

                "target": [
                    0,
                    1,
                    0,
                    1,
                    0,
                    1,
                ],

                "business_note": [
                    "not-monitored"
                    for _
                    in range(
                        6
                    )
                ],
            }
        )
    )


# ============================================================
# REAL PREPARATION WORKFLOW
# ============================================================


def build_ready_preparation_workflow(
):
    dataset_id = (
        "dataset:validated-monitoring"
    )


    dataframe = (
        validated_dataframe()
    )


    session = (
        create_preparation_session(
            selected_analysis_dataset_ids=[
                dataset_id,
            ],

            display_name=
                "Real monitoring handoff",
        )
    )


    put_preparation_artifact(
        workflow_id=
            session.workflow_id,

        dataset_id=
            dataset_id,

        dataset_filename=
            "validated_monitoring.csv",

        stage=
            "source",

        dataframe=
            dataframe,
    )


    current = session


    for stage in [
        PreparationStage.IMPORT,
        PreparationStage.UNDERSTAND,
        PreparationStage.QUALITY,
    ]:

        current = (
            record_required_stage_signal(
                workflow_id=
                    session.workflow_id,

                stage=
                    stage,

                completed=
                    True,

                dataset_ids=[
                    dataset_id,
                ],

                evidence_refs=[
                    (
                        "test:"
                        +
                        stage.value
                    )
                ],

                blocking_reasons=[],
            )
        )


    current = (
        record_analysis_output_selection(
            workflow_id=
                session.workflow_id,

            analysis_output_dataset_ids=[
                dataset_id,
            ],

            expected_revision=
                current.revision,
        )
    )


    current = (
        record_validation_stage_signal(
            workflow_id=
                session.workflow_id,

            completed=
                True,

            passed=
                True,

            dataset_ids=[
                dataset_id,
            ],

            evidence_refs=[
                "test:validated"
            ],

            blocking_reasons=[],

            expected_revision=
                current.revision,
        )
    )


    assert (
        current
        .snapshot
        .ready_for_analysis
        is True
    )


    assert (
        current
        .analysis_output_dataset_ids
        ==
        [
            dataset_id
        ]
    )


    return (
        current,
        dataset_id,
        dataframe,
    )


# ============================================================
# MODEL + MONITORING PROFILE
# ============================================================


def persist_model_and_profile(
    *,
    workflow_id: str,
    dataset_id: str,
    preparation_revision: int,
    dataframe: pd.DataFrame,
):

    contract = (
        MLTrainingContract(
            workflow_id=
                workflow_id,

            dataset_id=
                dataset_id,

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


    x_train = (
        dataframe[
            [
                "age",
                "segment",
            ]
        ]
        .copy(
            deep=True
        )
    )


    artifact = (
        register_ml_model_artifact(
            training_contract=
                contract,

            metrics={
                "accuracy":
                    0.80,

                "balanced_accuracy":
                    0.80,

                "f1_macro":
                    0.80,
            },

            train_rows=
                len(
                    x_train
                ),

            test_rows=
                2,

            model_bytes=
                b"REAL-HANDOFF-MODEL",

            preparation_session_revision=
                preparation_revision,

            created_at_utc=
                "2026-08-29T18:00:00+00:00",
        )
    )


    profile = (
        build_ml_monitoring_profile(
            x_train=
                x_train,

            model_artifact=
                artifact,
        )
    )


    register_ml_monitoring_profile(
        profile=
            profile
    )


    return (
        artifact,
        profile,
    )


# ============================================================
# REAL HANDOFF CONTRACT
# ============================================================


def test_real_handoff_is_ready_and_server_owned(
) -> None:

    with isolated_real_handoff_environment():

        (
            session,
            dataset_id,
            dataframe,
        ) = (
            build_ready_preparation_workflow()
        )


        handoff = (
            load_validated_analysis_input(
                workflow_id=
                    session.workflow_id
            )
        )


        assert (
            handoff.workflow_id
            ==
            session.workflow_id
        )


        assert (
            handoff.session_revision
            ==
            session.revision
        )


        assert (
            handoff.dataset_ids
            ==
            (
                dataset_id,
            )
        )


        assert (
            len(
                handoff.dataset_records
            )
            ==
            1
        )


        record = (
            handoff.dataset_records[
                0
            ]
        )


        assert (
            record[
                "dataset_id"
            ]
            ==
            dataset_id
        )


        pd.testing.assert_frame_equal(
            record[
                "dataframe"
            ],
            dataframe,
        )


# ============================================================
# END-TO-END MONITORING
# ============================================================


def test_monitoring_service_uses_real_validated_handoff(
) -> None:

    with isolated_real_handoff_environment():

        (
            session,
            dataset_id,
            dataframe,
        ) = (
            build_ready_preparation_workflow()
        )


        (
            artifact,
            profile,
        ) = (
            persist_model_and_profile(
                workflow_id=
                    session.workflow_id,

                dataset_id=
                    dataset_id,

                preparation_revision=
                    session.revision,

                dataframe=
                    dataframe,
            )
        )


        result = (
            run_ml_monitoring(
                workflow_id=
                    session.workflow_id,

                model_id=
                    artifact.model_id,

                observed_dataset_id=
                    dataset_id,
            )
        )


        assert (
            result.workflow_id
            ==
            session.workflow_id
        )


        assert (
            result.model_id
            ==
            artifact.model_id
        )


        assert (
            result.profile_id
            ==
            profile.profile_id
        )


        assert (
            result.reference_dataset_id
            ==
            dataset_id
        )


        assert (
            result.observed_dataset_id
            ==
            dataset_id
        )


        assert (
            result
            .preparation_session_revision
            ==
            session.revision
        )


        assert (
            result
            .observed_preparation_session_revision
            ==
            session.revision
        )


        assert (
            result.overall_status
            ==
            "ok"
        )


        assert [
            item.feature_name

            for item
            in result.feature_results
        ] == [
            "age",
            "segment",
        ]


        serialized = str(
            result.model_dump(
                mode="json"
            )
        )


        assert (
            "business_note"
            not in
            serialized
        )


        assert (
            "not-monitored"
            not in
            serialized
        )


        history = (
            list_ml_drift_evaluations_for_model(
                model_id=
                    artifact.model_id,

                workflow_id=
                    session.workflow_id,
            )
        )


        assert (
            history
            ==
            [
                result
            ]
        )


# ============================================================
# MAIN
# ============================================================


def main(
) -> None:

    print(
        (
            "=== DATALENS ML MONITORING SERVICE "
            "REAL HANDOFF v0.1 ==="
        )
    )

    print()


    test_real_handoff_is_ready_and_server_owned()

    print(
        (
            "[PASS] Real Preparation READY Handoff "
            "is server-owned"
        )
    )


    test_monitoring_service_uses_real_validated_handoff()

    print(
        (
            "[PASS] Monitoring Service consumes "
            "real validated Handoff end-to-end"
        )
    )


    print()

    print(
        (
            "PASS - ML Monitoring Service "
            "Real Handoff v0.1"
        )
    )


if __name__ == "__main__":
    main()
