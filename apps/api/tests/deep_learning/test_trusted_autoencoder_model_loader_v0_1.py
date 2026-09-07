from __future__ import annotations


import inspect


import numpy as np
import pandas as pd
import torch


import app.deep_learning.autoencoder_model_loader as loader_module


from app.deep_learning.autoencoder_bundle import (
    serialize_tabular_autoencoder_bundle,
)


from app.deep_learning.autoencoder_model_loader import (
    DL_TRUSTED_AUTOENCODER_MODEL_LOADER_RULE_VERSION,
    DLTrustedAutoencoderArtifactError,
    DLTrustedAutoencoderInferenceError,
    DLTrustedAutoencoderRaceError,
    LoadedTabularAutoencoderModel,
    load_trusted_tabular_autoencoder_model,
)


from app.deep_learning.autoencoder_network import (
    TabularAutoencoder,
)


from app.deep_learning.autoencoder_threshold import (
    AUTOENCODER_THRESHOLD_COMPARISON,
    AUTOENCODER_THRESHOLD_METHOD,
    ReconstructionErrorThreshold,
    apply_reconstruction_error_threshold,
)


from app.deep_learning.runtime import (
    seed_torch,
)


from app.ml.anomaly_contracts import (
    MLAnomalyTrainingContract,
)


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


from app.ml.preprocessing import (
    build_ml_preprocessor,
)


from tests.ml.test_ml_model_artifact_store_v0_1 import (
    isolated_environment,
    seed_preparation_authority,
)


# ============================================================
# CONTRACT / FEATURES
# ============================================================


def build_contract(
    *,
    workflow_id: str = "prep:trusted-autoencoder",
) -> MLAnomalyTrainingContract:

    return (
        MLAnomalyTrainingContract(
            workflow_id=
                workflow_id,

            dataset_id=
                "dataset:trusted-autoencoder",

            feature_columns=[
                "amount",
                "frequency",
            ],

            estimator_hyperparameters={
                "kind":
                    "tabular_autoencoder",

                "hidden_features":
                    6,

                "latent_features":
                    2,

                "epochs":
                    5,

                "batch_size":
                    4,

                "learning_rate":
                    0.001,
            },

            threshold_quantile=
                0.975,
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

                "frequency": [
                    1.0,
                    2.0,
                    2.0,
                    3.0,
                    4.0,
                    6.0,
                ],
            }
        )
    )


# ============================================================
# REFERENCE INFERENCE
# ============================================================


def reference_inference(
    *,
    features: pd.DataFrame,
    model: TabularAutoencoder,
    preprocessor,
    threshold: ReconstructionErrorThreshold,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:

    transformed = (
        np.asarray(
            preprocessor.transform(
                features
            ),
            dtype=np.float32,
        )
    )


    tensor = (
        torch.from_numpy(
            np.ascontiguousarray(
                transformed,
                dtype=np.float32,
            )
        )
    )


    model.eval()


    with torch.inference_mode():

        reconstruction = (
            model(
                tensor
            )
        )


        errors = (
            torch.mean(
                (
                    reconstruction
                    -
                    tensor
                )
                **
                2,
                dim=1,
            )
            .detach()
            .cpu()
            .to(
                dtype=torch.float32
            )
            .contiguous()
        )


    flags = (
        apply_reconstruction_error_threshold(
            reconstruction_errors=
                errors,
            threshold=
                threshold,
        )
    )


    return (
        errors
        .numpy()
        .astype(
            np.float64,
            copy=True,
        ),

        flags
        .detach()
        .cpu()
        .numpy()
        .astype(
            np.bool_,
            copy=True,
        ),
    )


# ============================================================
# REAL STORE-BACKED ARTIFACT
# ============================================================


def build_artifact(
    *,
    contract: MLAnomalyTrainingContract,
):

    features = (
        build_features()
    )


    preprocessor = (
        build_ml_preprocessor(
            contract=
                contract
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
        211
    )


    model = (
        TabularAutoencoder(
            input_features=
                int(
                    transformed.shape[
                        1
                    ]
                ),

            hidden_features=
                int(
                    contract
                    .estimator_hyperparameters
                    .hidden_features
                ),

            latent_features=
                int(
                    contract
                    .estimator_hyperparameters
                    .latent_features
                ),
        )
    )


    model.eval()


    threshold = (
        ReconstructionErrorThreshold(
            quantile=
                float(
                    contract.threshold_quantile
                ),

            threshold=
                0.42,

            train_rows=
                80,

            method=
                AUTOENCODER_THRESHOLD_METHOD,

            comparison_operator=
                AUTOENCODER_THRESHOLD_COMPARISON,
        )
    )


    bundle = (
        serialize_tabular_autoencoder_bundle(
            model=
                model,

            preprocessor=
                preprocessor,

            training_contract=
                contract,

            threshold=
                threshold,
        )
    )


    artifact = (
        register_ml_model_artifact(
            training_contract=
                contract,

            metrics={
                "train_reconstruction_mse":
                    0.10,

                "test_reconstruction_mse":
                    0.24,

                "anomaly_threshold":
                    0.42,

                "test_anomaly_rate":
                    0.05,
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
                "2026-09-07T14:00:00+00:00",
        )
    )


    return (
        features,
        model,
        preprocessor,
        threshold,
        artifact,
    )


# ============================================================
# IDENTIFIER-ONLY API
# ============================================================


def test_loader_accepts_only_server_owned_identifiers(
) -> None:

    parameters = (
        inspect.signature(
            load_trusted_tabular_autoencoder_model
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


    for forbidden in (
        "path",
        "model_path",
        "bytes",
        "bundle",
        "trusted_bundle_bytes",
    ):

        assert (
            forbidden
            not in
            parameters
        )


# ============================================================
# TRUSTED ROUND TRIP
# ============================================================


def test_store_backed_loader_restores_exact_pipeline(
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
            threshold,
            artifact,
        ) = (
            build_artifact(
                contract=
                    contract
            )
        )


        loaded = (
            load_trusted_tabular_autoencoder_model(
                workflow_id=
                    contract.workflow_id,

                model_id=
                    artifact.model_id,
            )
        )


        assert isinstance(
            loaded,
            LoadedTabularAutoencoderModel,
        )


        assert (
            loaded.artifact
            ==
            artifact
        )


        assert (
            loaded.threshold
            ==
            threshold
        )


        single_row = (
            features
            .iloc[
                [
                    0
                ]
            ]
            .copy()
        )


        (
            expected_errors,
            expected_flags,
        ) = (
            reference_inference(
                features=
                    single_row,

                model=
                    original_model,

                preprocessor=
                    original_preprocessor,

                threshold=
                    threshold,
            )
        )


        actual = (
            loaded.detect(
                single_row
            )
        )


        np.testing.assert_array_equal(
            actual.reconstruction_errors,
            expected_errors,
        )


        np.testing.assert_array_equal(
            actual.anomaly_flags,
            expected_flags,
        )


        assert (
            actual.reconstruction_errors.shape
            ==
            (
                1,
            )
        )


        assert (
            actual.anomaly_flags.shape
            ==
            (
                1,
            )
        )


        assert (
            actual.reconstruction_errors.dtype
            ==
            np.float64
        )


        assert (
            actual.anomaly_flags.dtype
            ==
            np.bool_
        )


        assert (
            actual.reconstruction_errors.flags.writeable
            is False
        )


        assert (
            actual.anomaly_flags.flags.writeable
            is False
        )


# ============================================================
# BATCH INFERENCE
# ============================================================


def test_store_backed_loader_supports_batch_inference(
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
            threshold,
            artifact,
        ) = (
            build_artifact(
                contract=
                    contract
            )
        )


        loaded = (
            load_trusted_tabular_autoencoder_model(
                workflow_id=
                    contract.workflow_id,

                model_id=
                    artifact.model_id,
            )
        )


        batch = (
            features
            .iloc[
                :3
            ]
            .copy()
        )


        (
            expected_errors,
            expected_flags,
        ) = (
            reference_inference(
                features=
                    batch,

                model=
                    original_model,

                preprocessor=
                    original_preprocessor,

                threshold=
                    threshold,
            )
        )


        actual = (
            loaded.detect(
                batch
            )
        )


        np.testing.assert_array_equal(
            actual.reconstruction_errors,
            expected_errors,
        )


        np.testing.assert_array_equal(
            actual.anomaly_flags,
            expected_flags,
        )


# ============================================================
# OUTER BINARY INTEGRITY
# ============================================================


def test_outer_artifact_tampering_is_blocked_before_decoder(
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
            _,
            artifact,
        ) = (
            build_artifact(
                contract=
                    contract
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


        original_decoder = (
            loader_module
            .deserialize_trusted_tabular_autoencoder_bundle
        )


        decoder_called = False


        def forbidden_decoder(
            *args,
            **kwargs,
        ):

            nonlocal decoder_called

            decoder_called = True

            raise AssertionError(
                (
                    "Bundle decoder must not run "
                    "after outer Artifact tampering."
                )
            )


        loader_module.deserialize_trusted_tabular_autoencoder_bundle = (
            forbidden_decoder
        )


        try:

            try:

                load_trusted_tabular_autoencoder_model(
                    workflow_id=
                        contract.workflow_id,

                    model_id=
                        artifact.model_id,
                )

            except DLTrustedAutoencoderArtifactError:
                pass

            else:

                raise AssertionError(
                    (
                        "Outer Artifact tampering "
                        "must fail closed."
                    )
                )

        finally:

            loader_module.deserialize_trusted_tabular_autoencoder_bundle = (
                original_decoder
            )


        assert (
            decoder_called
            is False
        )


# ============================================================
# SERIALIZATION FORMAT
# ============================================================


def test_loader_refuses_joblib_anomaly_artifact(
) -> None:

    with isolated_environment():

        contract = (
            build_contract(
                workflow_id=
                    "prep:autoencoder-joblib"
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
                    "anomaly_threshold":
                        0.42,
                },

                train_rows=
                    80,

                test_rows=
                    20,

                model_bytes=
                    b"joblib-placeholder",

                created_at_utc=
                    "2026-09-07T14:01:00+00:00",
            )
        )


        try:

            load_trusted_tabular_autoencoder_model(
                workflow_id=
                    contract.workflow_id,

                model_id=
                    artifact.model_id,
            )

        except DLTrustedAutoencoderArtifactError as error:

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
            "Trusted Autoencoder loader must "
            "refuse joblib artifacts."
        )
    )


# ============================================================
# MODEL FAMILY
# ============================================================


def test_loader_refuses_supervised_pytorch_artifact(
) -> None:

    with isolated_environment():

        contract = (
            MLTrainingContract(
                workflow_id=
                    "prep:wrong-model-family",

                dataset_id=
                    "dataset:trusted-autoencoder",

                problem_type=
                    "regression",

                target_column=
                    "target",

                feature_columns=[
                    "amount",
                ],

                estimator_key=
                    "tabular_mlp_regressor",
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
                    b"opaque-pytorch-placeholder",

                serialization_format=
                    "pytorch_bundle",

                preparation_session_revision=
                    0,

                created_at_utc=
                    "2026-09-07T14:02:00+00:00",
            )
        )


        try:

            load_trusted_tabular_autoencoder_model(
                workflow_id=
                    contract.workflow_id,

                model_id=
                    artifact.model_id,
            )

        except DLTrustedAutoencoderArtifactError:
            return


    raise AssertionError(
        (
            "Trusted Autoencoder loader must reject "
            "a supervised PyTorch Artifact."
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
            _,
            artifact,
        ) = (
            build_artifact(
                contract=
                    contract
            )
        )


        try:

            load_trusted_tabular_autoencoder_model(
                workflow_id=
                    "prep:wrong-workflow",

                model_id=
                    artifact.model_id,
            )

        except DLTrustedAutoencoderArtifactError:
            return


    raise AssertionError(
        (
            "Cross-workflow trusted Autoencoder "
            "load must fail closed."
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
            _,
            artifact,
        ) = (
            build_artifact(
                contract=
                    contract
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
                "2026-09-07T14:03:00+00:00"
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

                load_trusted_tabular_autoencoder_model(
                    workflow_id=
                        contract.workflow_id,

                    model_id=
                        artifact.model_id,
                )

            except DLTrustedAutoencoderRaceError:
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


def test_detect_input_contract_is_enforced(
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
            _,
            artifact,
        ) = (
            build_artifact(
                contract=
                    contract
            )
        )


        loaded = (
            load_trusted_tabular_autoencoder_model(
                workflow_id=
                    contract.workflow_id,

                model_id=
                    artifact.model_id,
            )
        )


        invalid = (
            features[
                [
                    "frequency",
                    "amount",
                ]
            ]
        )


        try:

            loaded.detect(
                invalid
            )

        except DLTrustedAutoencoderInferenceError:
            return


    raise AssertionError(
        (
            "Trusted Autoencoder inference must "
            "enforce ordered feature authority."
        )
    )


# ============================================================
# RULE VERSION
# ============================================================


def test_rule_version(
) -> None:

    assert (
        DL_TRUSTED_AUTOENCODER_MODEL_LOADER_RULE_VERSION
        ==
        "dl_trusted_autoencoder_model_loader_v0.1"
    )


# ============================================================
# MAIN
# ============================================================


def main(
) -> None:

    print(
        (
            "=== DATALENS TRUSTED AUTOENCODER "
            "MODEL LOADER v0.1 ==="
        )
    )

    print()


    test_loader_accepts_only_server_owned_identifiers()

    print(
        "Identifier-only loader API: PASS"
    )


    test_store_backed_loader_restores_exact_pipeline()

    print(
        "Store-backed single-row exact inference: PASS"
    )


    test_store_backed_loader_supports_batch_inference()

    print(
        "Store-backed batch exact inference: PASS"
    )


    test_outer_artifact_tampering_is_blocked_before_decoder()

    print(
        "Outer binary integrity before decoder: PASS"
    )


    test_loader_refuses_joblib_anomaly_artifact()

    print(
        "Serialization-format isolation: PASS"
    )


    test_loader_refuses_supervised_pytorch_artifact()

    print(
        "Estimator / problem family isolation: PASS"
    )


    test_cross_workflow_load_is_blocked()

    print(
        "Workflow scope guard: PASS"
    )


    test_metadata_race_is_blocked()

    print(
        "Metadata race guard: PASS"
    )


    test_detect_input_contract_is_enforced()

    print(
        "Inference feature authority: PASS"
    )


    test_rule_version()

    print(
        "Trusted Autoencoder loader rule version: PASS"
    )


    print()

    print(
        (
            "PASS - DataLens Trusted Autoencoder "
            "Model Loader v0.1"
        )
    )


if __name__ == "__main__":
    main()
