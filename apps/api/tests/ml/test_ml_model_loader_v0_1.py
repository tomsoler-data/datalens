from __future__ import annotations

import inspect
import io
import os
import tempfile

from contextlib import contextmanager
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import (
    LinearRegression,
)

from sklearn.pipeline import (
    Pipeline,
)

from sklearn.preprocessing import (
    StandardScaler,
)

import app.ml.model_loader as loader_module

from app.ml.contracts import (
    MLTrainingContract,
)

from app.ml.model_artifact_data_plane import (
    ml_model_artifact_data_root,
)

from app.ml.model_artifact_store import (
    register_ml_model_artifact,
    resolve_ml_model_artifact_store_path,
)

from app.ml.model_loader import (
    ML_MODEL_LOADER_RULE_VERSION,
    MLModelDeserializationError,
    MLModelInterfaceError,
    MLModelLoaderArtifactError,
    MLModelLoaderRaceError,
    load_trusted_ml_model,
)

from app.persistence.sqlite_database import (
    sqlite_connection,
)


# ============================================================
# ISOLATION
# ============================================================


@contextmanager
def isolated_environment():
    previous_sqlite = os.environ.get(
        "DATALENS_SQLITE_PATH"
    )

    previous_store = os.environ.get(
        "DATALENS_ML_MODEL_ARTIFACT_STORE_PATH"
    )

    with tempfile.TemporaryDirectory(
        prefix="datalens-ml-model-loader-"
    ) as temporary_directory:
        root = Path(
            temporary_directory
        )

        database = (
            root
            /
            "datalens.sqlite3"
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
            database
        )

        os.environ[
            "DATALENS_ML_MODEL_ARTIFACT_STORE_PATH"
        ] = str(
            model_store
        )

        try:
            yield (
                root,
                model_store,
            )

        finally:
            if (
                previous_sqlite
                is None
            ):
                os.environ.pop(
                    "DATALENS_SQLITE_PATH",
                    None,
                )

            else:
                os.environ[
                    "DATALENS_SQLITE_PATH"
                ] = previous_sqlite

            if (
                previous_store
                is None
            ):
                os.environ.pop(
                    "DATALENS_ML_MODEL_ARTIFACT_STORE_PATH",
                    None,
                )

            else:
                os.environ[
                    "DATALENS_ML_MODEL_ARTIFACT_STORE_PATH"
                ] = previous_store


# ============================================================
# PREPARATION AUTHORITY
# ============================================================


def seed_preparation_authority(
    *,
    workflow_id: str,
    dataset_id: str,
) -> None:
    with sqlite_connection(
        write=True
    ) as connection:
        connection.execute(
            """
            INSERT INTO preparation_sessions (
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
                "2026-08-28T20:00:00+00:00",
                "2026-08-28T20:00:00+00:00",
            ),
        )

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
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "test-preparation-root",
                workflow_id,
                dataset_id,
                "validated.csv",
                "source",
                100,
                3,
                "[]",
                "[]",
                "[]",
                "data/validated.json.gz",
            ),
        )


# ============================================================
# CONTRACT
# ============================================================


def training_contract(
    *,
    workflow_id: str = "prep:ml-loader",
    dataset_id: str = "dataset:validated",
) -> MLTrainingContract:
    return (
        MLTrainingContract(
            workflow_id=workflow_id,
            dataset_id=dataset_id,
            problem_type="regression",
            target_column="revenue",
            feature_columns=[
                "age",
                "tenure",
            ],
            estimator_key="linear_regression",
        )
    )


# ============================================================
# MODEL FIXTURE
# ============================================================


def trained_pipeline() -> Pipeline:
    features = pd.DataFrame(
        {
            "age": [
                20.0,
                25.0,
                30.0,
                35.0,
                40.0,
                45.0,
                50.0,
                55.0,
            ],
            "tenure": [
                1.0,
                2.0,
                2.0,
                3.0,
                4.0,
                5.0,
                6.0,
                7.0,
            ],
        }
    )

    target = pd.Series(
        [
            125.0,
            150.0,
            165.0,
            190.0,
            220.0,
            250.0,
            280.0,
            310.0,
        ],
        name="revenue",
    )

    pipeline = (
        Pipeline(
            steps=[
                (
                    "scale",
                    StandardScaler(),
                ),
                (
                    "model",
                    LinearRegression(),
                ),
            ]
        )
    )

    pipeline.fit(
        features,
        target,
    )

    return pipeline


def serialize_joblib(
    value: object,
) -> bytes:
    buffer = io.BytesIO()

    joblib.dump(
        value,
        buffer,
        compress=0,
        protocol=5,
    )

    return (
        buffer.getvalue()
    )


def register_pipeline(
    *,
    workflow_id: str = "prep:ml-loader",
    dataset_id: str = "dataset:validated",
):
    seed_preparation_authority(
        workflow_id=workflow_id,
        dataset_id=dataset_id,
    )

    pipeline = (
        trained_pipeline()
    )

    record = (
        register_ml_model_artifact(
            training_contract=(
                training_contract(
                    workflow_id=workflow_id,
                    dataset_id=dataset_id,
                )
            ),
            metrics={
                "mae": 1.0,
                "rmse": 1.2,
                "r2": 0.99,
            },
            train_rows=80,
            test_rows=20,
            model_bytes=(
                serialize_joblib(
                    pipeline
                )
            ),
            created_at_utc=(
                "2026-08-28T20:30:00+00:00"
            ),
        )
    )

    return (
        pipeline,
        record,
    )


# ============================================================
# SERVER-OWNED INPUT SURFACE
# ============================================================


def test_loader_accepts_only_server_owned_identifiers():
    parameters = (
        inspect.signature(
            load_trusted_ml_model
        )
        .parameters
    )

    assert (
        set(
            parameters
        )
        ==
        {
            "workflow_id",
            "model_id",
        }
    )

    assert (
        "model_bytes"
        not in
        parameters
    )

    assert (
        "model_path"
        not in
        parameters
    )


# ============================================================
# ROUND TRIP
# ============================================================


def test_verified_model_reload_and_prediction():
    with isolated_environment():
        original_pipeline, record = (
            register_pipeline()
        )

        loaded = (
            load_trusted_ml_model(
                workflow_id=(
                    "prep:ml-loader"
                ),
                model_id=(
                    record.model_id
                ),
            )
        )

        features = pd.DataFrame(
            {
                "age": [
                    32.0,
                    48.0,
                ],
                "tenure": [
                    3.0,
                    6.0,
                ],
            }
        )

        original_predictions = (
            original_pipeline.predict(
                features
            )
        )

        restored_predictions = (
            loaded.predict(
                features
            )
        )

        assert (
            loaded.artifact
            ==
            record
        )

        assert (
            np.allclose(
                original_predictions,
                restored_predictions,
            )
        )


# ============================================================
# WORKFLOW SCOPE
# ============================================================


def test_cross_workflow_load_is_blocked():
    with isolated_environment():
        _, record = (
            register_pipeline()
        )

        try:
            load_trusted_ml_model(
                workflow_id="prep:other",
                model_id=record.model_id,
            )

        except MLModelLoaderArtifactError:
            return

        raise AssertionError(
            (
                "Cross-workflow trusted model load "
                "should fail closed."
            )
        )


# ============================================================
# UNKNOWN MODEL
# ============================================================


def test_unknown_model_is_blocked():
    with isolated_environment():
        try:
            load_trusted_ml_model(
                workflow_id="prep:ml-loader",
                model_id="model:missing",
            )

        except MLModelLoaderArtifactError:
            return

        raise AssertionError(
            (
                "Unknown Model Artifact should "
                "fail closed."
            )
        )


# ============================================================
# BINARY INTEGRITY
# ============================================================


def test_binary_tampering_is_blocked_before_deserialization():
    with isolated_environment():
        _, record = (
            register_pipeline()
        )

        store_path = (
            resolve_ml_model_artifact_store_path()
        )

        data_root = (
            ml_model_artifact_data_root(
                store_path
            )
        )

        model_file = (
            data_root
            /
            record.model_path
        )

        assert (
            model_file.exists()
        )

        model_file.write_bytes(
            b"DATALENS-TAMPERED-MODEL"
        )

        try:
            load_trusted_ml_model(
                workflow_id=record.workflow_id,
                model_id=record.model_id,
            )

        except MLModelLoaderArtifactError:
            return

        raise AssertionError(
            (
                "Tampered Model Artifact binary "
                "should fail closed before joblib.load()."
            )
        )


# ============================================================
# INVALID JOBLIB
# ============================================================


def test_invalid_trusted_joblib_is_blocked():
    with isolated_environment():
        seed_preparation_authority(
            workflow_id="prep:invalid-joblib",
            dataset_id="dataset:validated",
        )

        record = (
            register_ml_model_artifact(
                training_contract=(
                    training_contract(
                        workflow_id=(
                            "prep:invalid-joblib"
                        ),
                    )
                ),
                metrics={
                    "mae": 1.0,
                },
                train_rows=80,
                test_rows=20,
                model_bytes=(
                    b"not-a-valid-joblib-model"
                ),
                created_at_utc=(
                    "2026-08-28T20:40:00+00:00"
                ),
            )
        )

        try:
            load_trusted_ml_model(
                workflow_id=record.workflow_id,
                model_id=record.model_id,
            )

        except MLModelDeserializationError:
            return

        raise AssertionError(
            (
                "Invalid trusted joblib payload "
                "should fail closed."
            )
        )


# ============================================================
# ESTIMATOR INTERFACE
# ============================================================


def test_non_predictor_joblib_is_blocked():
    with isolated_environment():
        seed_preparation_authority(
            workflow_id="prep:not-predictor",
            dataset_id="dataset:validated",
        )

        record = (
            register_ml_model_artifact(
                training_contract=(
                    training_contract(
                        workflow_id=(
                            "prep:not-predictor"
                        ),
                    )
                ),
                metrics={
                    "mae": 1.0,
                },
                train_rows=80,
                test_rows=20,
                model_bytes=(
                    serialize_joblib(
                        {
                            "message":
                                "valid joblib, not a model"
                        }
                    )
                ),
                created_at_utc=(
                    "2026-08-28T20:45:00+00:00"
                ),
            )
        )

        try:
            load_trusted_ml_model(
                workflow_id=record.workflow_id,
                model_id=record.model_id,
            )

        except MLModelInterfaceError:
            return

        raise AssertionError(
            (
                "Serialized object without predict() "
                "should fail closed."
            )
        )


# ============================================================
# METADATA RACE
# ============================================================


def test_metadata_race_is_blocked():
    with isolated_environment():
        _, record = (
            register_pipeline()
        )

        changed_record = (
            record.model_copy(
                update={
                    "metrics": {
                        "mae": 999.0,
                    }
                }
            )
        )

        real_get = (
            loader_module
            .get_ml_model_artifact
        )

        calls = {
            "count": 0,
        }

        def racing_get(
            *,
            model_id: str,
            workflow_id: str,
        ):
            calls[
                "count"
            ] += 1

            if (
                calls[
                    "count"
                ]
                ==
                1
            ):
                return record

            return changed_record

        loader_module.get_ml_model_artifact = (
            racing_get
        )

        try:
            try:
                load_trusted_ml_model(
                    workflow_id=record.workflow_id,
                    model_id=record.model_id,
                )

            except MLModelLoaderRaceError:
                return

            raise AssertionError(
                (
                    "Model Artifact metadata race "
                    "should fail closed."
                )
            )

        finally:
            loader_module.get_ml_model_artifact = (
                real_get
            )


# ============================================================
# VERSION
# ============================================================


def test_rule_version():
    assert (
        ML_MODEL_LOADER_RULE_VERSION
        ==
        "ml_model_loader_v0.1"
    )


# ============================================================
# MAIN
# ============================================================


def main():
    print(
        "=== DATALENS TRUSTED ML MODEL LOADER v0.1 ==="
    )
    print()

    tests = [
        (
            "Loader accepts server-owned identifiers only",
            test_loader_accepts_only_server_owned_identifiers,
        ),
        (
            "Verified model reload + prediction round-trip",
            test_verified_model_reload_and_prediction,
        ),
        (
            "Cross-workflow Model Artifact load is blocked",
            test_cross_workflow_load_is_blocked,
        ),
        (
            "Unknown Model Artifact is blocked",
            test_unknown_model_is_blocked,
        ),
        (
            "Binary tampering is blocked before deserialization",
            test_binary_tampering_is_blocked_before_deserialization,
        ),
        (
            "Invalid trusted joblib is blocked",
            test_invalid_trusted_joblib_is_blocked,
        ),
        (
            "Non-predictor joblib object is blocked",
            test_non_predictor_joblib_is_blocked,
        ),
        (
            "Model Artifact metadata race is blocked",
            test_metadata_race_is_blocked,
        ),
        (
            "Trusted Model Loader rule version",
            test_rule_version,
        ),
    ]

    for label, test in tests:
        test()

        print(
            f"{label}: PASS"
        )

    print()
    print(
        "Trusted ML Model Loader v0.1: PASS"
    )


if __name__ == "__main__":
    main()