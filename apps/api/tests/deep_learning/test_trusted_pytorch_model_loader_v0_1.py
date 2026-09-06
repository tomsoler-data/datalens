from __future__ import annotations


import os
import tempfile


from contextlib import (
    contextmanager,
)


from pathlib import (
    Path,
)


import numpy as np
import pandas as pd


import app.deep_learning.model_loader as loader_module


from app.deep_learning.model_bundle import (
    serialize_tabular_mlp_bundle,
)


from app.deep_learning.model_loader import (
    DL_TRUSTED_MODEL_LOADER_RULE_VERSION,
    DLTrustedModelArtifactError,
    DLTrustedModelInferenceError,
    DLTrustedModelRaceError,
    LoadedTabularMLPModel,
    load_trusted_tabular_mlp_model,
)


from app.deep_learning.networks import (
    FeedForwardRegressor,
)


from app.deep_learning.runtime import (
    seed_torch,
)


from app.ml.contracts import (
    MLTrainingContract,
)


from app.ml.model_artifact_data_plane import (
    ml_model_artifact_data_root,
)


from app.ml.model_artifact_store import (
    get_ml_model_artifact,
    register_ml_model_artifact,
    resolve_ml_model_artifact_store_path,
)


from app.ml.preprocessing import (
    build_ml_preprocessor,
)


from app.persistence.sqlite_database import (
    sqlite_connection,
)


# ============================================================
# ENVIRONMENT
# ============================================================


@contextmanager
def isolated_environment(
):

    previous_sqlite = os.environ.get(
        "DATALENS_SQLITE_PATH"
    )


    previous_store = os.environ.get(
        "DATALENS_ML_MODEL_ARTIFACT_STORE_PATH"
    )


    with tempfile.TemporaryDirectory(
        prefix=
            "datalens-trusted-dl-loader-"
    ) as root:

        root_path = Path(
            root
        )


        database = (
            root_path
            /
            "datalens.sqlite3"
        )


        store = (
            root_path
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
            store
        )


        try:

            yield (
                root_path,
                store,
            )


        finally:

            if previous_sqlite is None:

                os.environ.pop(
                    "DATALENS_SQLITE_PATH",
                    None,
                )

            else:

                os.environ[
                    "DATALENS_SQLITE_PATH"
                ] = previous_sqlite


            if previous_store is None:

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
                "2026-09-06T13:00:00+00:00",
                "2026-09-06T13:00:00+00:00",
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
# CONTRACT / MODEL
# ============================================================


def build_contract(
    *,
    workflow_id: str = "prep:trusted-dl",
) -> MLTrainingContract:

    return (
        MLTrainingContract(
            workflow_id=
                workflow_id,

            dataset_id=
                "dataset:validated",

            problem_type=
                "regression",

            target_column=
                "target",

            feature_columns=[
                "amount",
                "segment",
            ],

            categorical_feature_columns=[
                "segment",
            ],

            estimator_key=
                "tabular_mlp_regressor",

            estimator_hyperparameters={
                "kind":
                    "tabular_mlp_regressor",

                "hidden_features":
                    10,

                "epochs":
                    20,

                "batch_size":
                    8,

                "learning_rate":
                    0.001,
            },
        )
    )


def build_features(
) -> pd.DataFrame:

    return (
        pd.DataFrame(
            {
                "amount": [
                    10.0,
                    20.0,
                    30.0,
                    40.0,
                    50.0,
                    60.0,
                ],

                "segment": [
                    "A",
                    "B",
                    "A",
                    "C",
                    "B",
                    "C",
                ],
            }
        )
    )


def build_artifact(
    *,
    contract: MLTrainingContract,
):

    features = (
        build_features()
    )


    preprocessor = (
        build_ml_preprocessor(
            contract=contract
        )
    )


    transformed = (
        np.asarray(
            preprocessor.fit_transform(
                features
            ),
            dtype=np.float32,
        )
    )


    seed_torch(
        101
    )


    model = (
        FeedForwardRegressor(
            input_features=
                int(
                    transformed.shape[
                        1
                    ]
                ),

            hidden_features=
                int(
                    contract
                    .effective_estimator_hyperparameters
                    .hidden_features
                ),
        )
    )


    model.eval()


    bundle = (
        serialize_tabular_mlp_bundle(
            model=model,
            preprocessor=preprocessor,
            training_contract=contract,
        )
    )


    artifact = (
        register_ml_model_artifact(
            training_contract=
                contract,

            metrics={
                "rmse":
                    1.0,

                "mae":
                    0.8,

                "r2":
                    0.75,
            },

            train_rows=
                80,

            test_rows=
                20,

            model_bytes=
                bundle,

            serialization_format=
                "pytorch_bundle",

            preparation_session_revision=
                0,

            created_at_utc=
                "2026-09-06T13:01:00+00:00",
        )
    )


    return (
        features,
        model,
        preprocessor,
        artifact,
    )


def expected_predictions(
    *,
    features: pd.DataFrame,
    model: FeedForwardRegressor,
    preprocessor,
) -> np.ndarray:

    transformed = (
        np.asarray(
            preprocessor.transform(
                features
            ),
            dtype=np.float32,
        )
    )


    import torch


    tensor = (
        torch.from_numpy(
            np.ascontiguousarray(
                transformed,
                dtype=np.float32,
            )
        )
    )


    with torch.inference_mode():

        values = (
            model(
                tensor
            )
            .detach()
            .cpu()
            .numpy()
        )


    return (
        np.asarray(
            values,
            dtype=np.float64,
        )
    )


# ============================================================
# TRUSTED ROUND TRIP
# ============================================================


def test_trusted_loader_restores_predictive_pipeline(
) -> None:

    with isolated_environment():

        contract = (
            build_contract()
        )


        seed_preparation_authority(
            workflow_id=
                contract.workflow_id,

            dataset_id=
                contract.dataset_id,
        )


        (
            features,
            original_model,
            original_preprocessor,
            artifact,
        ) = (
            build_artifact(
                contract=contract
            )
        )


        expected = (
            expected_predictions(
                features=features,
                model=original_model,
                preprocessor=original_preprocessor,
            )
        )


        loaded = (
            load_trusted_tabular_mlp_model(
                workflow_id=
                    contract.workflow_id,

                model_id=
                    artifact.model_id,
            )
        )


        assert isinstance(
            loaded,
            LoadedTabularMLPModel,
        )


        assert (
            loaded.artifact
            ==
            artifact
        )


        actual = (
            loaded.predict(
                features
            )
        )


        np.testing.assert_allclose(
            actual,
            expected,
            rtol=0.0,
            atol=0.0,
        )


# ============================================================
# OUTER BINARY INTEGRITY
# ============================================================


def test_outer_artifact_tampering_is_blocked(
) -> None:

    with isolated_environment():

        contract = (
            build_contract()
        )


        seed_preparation_authority(
            workflow_id=
                contract.workflow_id,

            dataset_id=
                contract.dataset_id,
        )


        (
            _,
            _,
            _,
            artifact,
        ) = (
            build_artifact(
                contract=contract
            )
        )


        store_path = (
            resolve_ml_model_artifact_store_path()
        )


        binary_path = (
            ml_model_artifact_data_root(
                store_path
            )
            /
            artifact.model_path
        )


        binary_path.write_bytes(
            binary_path.read_bytes()
            +
            b"TAMPERED"
        )


        try:

            load_trusted_tabular_mlp_model(
                workflow_id=
                    contract.workflow_id,

                model_id=
                    artifact.model_id,
            )

        except DLTrustedModelArtifactError:
            return


    raise AssertionError(
        (
            "Outer Model Artifact binary tampering "
            "must fail before DL deserialization."
        )
    )


# ============================================================
# FORMAT ISOLATION
# ============================================================


def test_dl_loader_refuses_joblib_artifact(
) -> None:

    with isolated_environment():

        contract = (
            MLTrainingContract(
                workflow_id=
                    "prep:trusted-classical",

                dataset_id=
                    "dataset:validated",

                problem_type=
                    "regression",

                target_column=
                    "target",

                feature_columns=[
                    "amount",
                ],

                estimator_key=
                    "linear_regression",
            )
        )


        seed_preparation_authority(
            workflow_id=
                contract.workflow_id,

            dataset_id=
                contract.dataset_id,
        )


        artifact = (
            register_ml_model_artifact(
                training_contract=
                    contract,

                metrics={
                    "rmse":
                        1.0,
                },

                train_rows=
                    80,

                test_rows=
                    20,

                model_bytes=
                    b"joblib-placeholder",
            )
        )


        try:

            load_trusted_tabular_mlp_model(
                workflow_id=
                    contract.workflow_id,

                model_id=
                    artifact.model_id,
            )

        except DLTrustedModelArtifactError as error:

            assert (
                "unsupported serialization format"
                in
                str(
                    error
                )
            )

            return


    raise AssertionError(
        (
            "Trusted DL loader must refuse "
            "joblib artifacts."
        )
    )


# ============================================================
# WORKFLOW SCOPE
# ============================================================


def test_cross_workflow_load_is_blocked(
) -> None:

    with isolated_environment():

        contract = (
            build_contract()
        )


        seed_preparation_authority(
            workflow_id=
                contract.workflow_id,

            dataset_id=
                contract.dataset_id,
        )


        (
            _,
            _,
            _,
            artifact,
        ) = (
            build_artifact(
                contract=contract
            )
        )


        try:

            load_trusted_tabular_mlp_model(
                workflow_id=
                    "prep:wrong-workflow",

                model_id=
                    artifact.model_id,
            )

        except DLTrustedModelArtifactError:
            return


    raise AssertionError(
        (
            "Cross-workflow trusted DL load "
            "must fail closed."
        )
    )


# ============================================================
# METADATA RACE
# ============================================================


def test_metadata_race_is_blocked(
) -> None:

    with isolated_environment():

        contract = (
            build_contract()
        )


        seed_preparation_authority(
            workflow_id=
                contract.workflow_id,

            dataset_id=
                contract.dataset_id,
        )


        (
            _,
            _,
            _,
            artifact,
        ) = (
            build_artifact(
                contract=contract
            )
        )


        original_get = (
            loader_module
            .get_ml_model_artifact
        )


        call_count = 0


        def racing_get(
            *,
            model_id: str,
            workflow_id: str,
        ):

            nonlocal call_count

            call_count += 1


            current = (
                original_get(
                    model_id=
                        model_id,

                    workflow_id=
                        workflow_id,
                )
            )


            if (
                call_count
                ==
                1
            ):
                return current


            payload = (
                current.model_dump(
                    mode="json"
                )
            )


            payload[
                "created_at_utc"
            ] = (
                "2026-09-06T13:02:00+00:00"
            )


            return (
                type(
                    current
                )
                .model_validate(
                    payload
                )
            )


        loader_module.get_ml_model_artifact = (
            racing_get
        )


        try:

            try:

                load_trusted_tabular_mlp_model(
                    workflow_id=
                        contract.workflow_id,

                    model_id=
                        artifact.model_id,
                )

            except DLTrustedModelRaceError:
                pass

            else:

                raise AssertionError(
                    (
                        "Metadata race must "
                        "fail closed."
                    )
                )

        finally:

            loader_module.get_ml_model_artifact = (
                original_get
            )


# ============================================================
# INPUT AUTHORITY
# ============================================================


def test_predict_input_contract_is_enforced(
) -> None:

    with isolated_environment():

        contract = (
            build_contract()
        )


        seed_preparation_authority(
            workflow_id=
                contract.workflow_id,

            dataset_id=
                contract.dataset_id,
        )


        (
            features,
            _,
            _,
            artifact,
        ) = (
            build_artifact(
                contract=contract
            )
        )


        loaded = (
            load_trusted_tabular_mlp_model(
                workflow_id=
                    contract.workflow_id,

                model_id=
                    artifact.model_id,
            )
        )


        invalid = (
            features[
                [
                    "segment",
                    "amount",
                ]
            ]
        )


        try:

            loaded.predict(
                invalid
            )

        except DLTrustedModelInferenceError:
            return


    raise AssertionError(
        (
            "Prediction input column order "
            "must follow the Training Contract."
        )
    )


# ============================================================
# RULE VERSION
# ============================================================


def test_rule_version(
) -> None:

    assert (
        DL_TRUSTED_MODEL_LOADER_RULE_VERSION
        ==
        "dl_trusted_model_loader_v0.1"
    )


# ============================================================
# MAIN
# ============================================================


def main(
) -> None:

    print(
        "=== DATALENS TRUSTED PYTORCH MODEL LOADER v0.1 ==="
    )

    print()


    test_trusted_loader_restores_predictive_pipeline()

    print(
        "Trusted PyTorch prediction round-trip: PASS"
    )


    test_outer_artifact_tampering_is_blocked()

    print(
        "Outer Model Artifact SHA guard: PASS"
    )


    test_dl_loader_refuses_joblib_artifact()

    print(
        "Deep Learning / joblib format isolation: PASS"
    )


    test_cross_workflow_load_is_blocked()

    print(
        "Workflow ownership guard: PASS"
    )


    test_metadata_race_is_blocked()

    print(
        "Model Artifact metadata race guard: PASS"
    )


    test_predict_input_contract_is_enforced()

    print(
        "Prediction input contract authority: PASS"
    )


    test_rule_version()

    print(
        "Trusted DL loader rule version: PASS"
    )


    print()

    print(
        "PASS - DataLens Trusted PyTorch Model Loader v0.1"
    )


if __name__ == "__main__":
    main()
