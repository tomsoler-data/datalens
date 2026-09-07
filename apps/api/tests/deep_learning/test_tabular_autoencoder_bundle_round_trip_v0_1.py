from __future__ import annotations


import io
import json
import zipfile


import numpy as np
import pandas as pd
import torch


import app.deep_learning.autoencoder_bundle as bundle_module


from app.deep_learning.autoencoder_bundle import (
    DL_TABULAR_AUTOENCODER_BUNDLE_CODEC_RULE_VERSION,
    TabularAutoencoderBundleContractError,
    TabularAutoencoderBundleIntegrityError,
    deserialize_trusted_tabular_autoencoder_bundle,
    serialize_tabular_autoencoder_bundle,
)


from app.deep_learning.autoencoder_network import (
    TabularAutoencoder,
)


from app.deep_learning.autoencoder_threshold import (
    AUTOENCODER_THRESHOLD_COMPARISON,
    AUTOENCODER_THRESHOLD_METHOD,
    ReconstructionErrorThreshold,
)


from app.deep_learning.runtime import (
    seed_torch,
)


from app.ml.anomaly_contracts import (
    MLAnomalyTrainingContract,
)


from app.ml.experiment_provenance import (
    ml_model_training_contract_sha256,
)


from app.ml.preprocessing import (
    build_ml_preprocessor,
)


# ============================================================
# FIXTURES
# ============================================================


def build_contract(
    *,
    workflow_id: str = "prep:autoencoder-bundle",
    hidden_features: int = 8,
    latent_features: int = 3,
    threshold_quantile: float = 0.975,
) -> MLAnomalyTrainingContract:

    return (
        MLAnomalyTrainingContract(
            workflow_id=
                workflow_id,

            dataset_id=
                "dataset:autoencoder-bundle",

            feature_columns=[
                "amount",
                "frequency",
                "segment",
            ],

            categorical_feature_columns=[
                "segment",
            ],

            estimator_hyperparameters={
                "kind":
                    "tabular_autoencoder",

                "hidden_features":
                    hidden_features,

                "latent_features":
                    latent_features,

                "epochs":
                    5,

                "batch_size":
                    4,

                "learning_rate":
                    0.001,
            },

            threshold_quantile=
                threshold_quantile,
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
                    70.0,
                    80.0,
                ],

                "frequency": [
                    1.0,
                    2.0,
                    2.0,
                    3.0,
                    4.0,
                    3.0,
                    5.0,
                    6.0,
                ],

                "segment": [
                    "A",
                    "B",
                    "A",
                    "C",
                    "B",
                    "C",
                    "A",
                    "B",
                ],
            }
        )
    )


def build_components(
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
        preprocessor
        .fit_transform(
            features
        )
    )


    transformed = (
        np.asarray(
            transformed,
            dtype=np.float32,
        )
    )


    seed_torch(
        137
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
                8,

            method=
                AUTOENCODER_THRESHOLD_METHOD,

            comparison_operator=
                AUTOENCODER_THRESHOLD_COMPARISON,
        )
    )


    return (
        features,
        transformed,
        model,
        preprocessor,
        threshold,
    )


def reconstruction(
    *,
    model: TabularAutoencoder,
    transformed: np.ndarray,
) -> np.ndarray:

    tensor = (
        torch.from_numpy(
            np.asarray(
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
            dtype=np.float32,
        )
    )


def read_bundle_members(
    bundle: bytes,
) -> dict[
    str,
    bytes,
]:

    with zipfile.ZipFile(
        io.BytesIO(
            bundle
        ),
        mode="r",
    ) as archive:

        return {
            name:
                archive.read(
                    name
                )

            for name
            in archive.namelist()
        }


def rebuild_bundle(
    members: dict[
        str,
        bytes,
    ],
) -> bytes:

    output = (
        io.BytesIO()
    )


    with zipfile.ZipFile(
        output,
        mode="w",
        compression=
            zipfile.ZIP_DEFLATED,
    ) as archive:

        for name in (
            "manifest.json",
            "model_state.pt",
            "preprocessor.joblib",
        ):

            archive.writestr(
                name,
                members[
                    name
                ],
            )


    return (
        output.getvalue()
    )


# ============================================================
# ROUND TRIP
# ============================================================


def test_exact_model_preprocessor_threshold_round_trip(
) -> None:

    contract = (
        build_contract()
    )


    (
        features,
        transformed,
        model,
        preprocessor,
        threshold,
    ) = (
        build_components(
            contract=
                contract
        )
    )


    expected_reconstruction = (
        reconstruction(
            model=
                model,
            transformed=
                transformed,
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


    restored = (
        deserialize_trusted_tabular_autoencoder_bundle(
            trusted_bundle_bytes=
                bundle,

            training_contract=
                contract,
        )
    )


    restored_transformed = (
        np.asarray(
            restored
            .preprocessor
            .transform(
                features
            ),
            dtype=np.float32,
        )
    )


    np.testing.assert_array_equal(
        restored_transformed,
        transformed,
    )


    actual_reconstruction = (
        reconstruction(
            model=
                restored.model,

            transformed=
                restored_transformed,
        )
    )


    np.testing.assert_array_equal(
        actual_reconstruction,
        expected_reconstruction,
    )


    original_state = (
        model.state_dict()
    )


    restored_state = (
        restored
        .model
        .state_dict()
    )


    assert (
        set(
            original_state
        )
        ==
        set(
            restored_state
        )
    )


    for name in original_state:

        assert torch.equal(
            original_state[
                name
            ]
            .detach()
            .cpu(),

            restored_state[
                name
            ]
            .detach()
            .cpu(),
        )


    assert (
        restored.threshold
        ==
        threshold
    )


    assert (
        restored.manifest.training_contract_sha256
        ==
        ml_model_training_contract_sha256(
            contract
        )
    )


    assert (
        restored.model.training
        is False
    )


# ============================================================
# MINIMAL STRUCTURE / MANIFEST
# ============================================================


def test_minimal_bundle_member_and_manifest_surface(
) -> None:

    contract = (
        build_contract()
    )


    (
        _,
        _,
        model,
        preprocessor,
        threshold,
    ) = (
        build_components(
            contract=
                contract
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


    members = (
        read_bundle_members(
            bundle
        )
    )


    assert (
        set(
            members
        )
        ==
        {
            "manifest.json",
            "model_state.pt",
            "preprocessor.joblib",
        }
    )


    manifest = (
        json.loads(
            members[
                "manifest.json"
            ]
            .decode(
                "utf-8"
            )
        )
    )


    assert (
        manifest[
            "estimator_key"
        ]
        ==
        "tabular_autoencoder"
    )


    assert (
        manifest[
            "network_class"
        ]
        ==
        "TabularAutoencoder"
    )


    assert (
        manifest[
            "threshold_quantile"
        ]
        ==
        0.975
    )


    assert (
        manifest[
            "anomaly_threshold"
        ]
        ==
        0.42
    )


    assert (
        manifest[
            "threshold_train_rows"
        ]
        ==
        8
    )


    forbidden = {
        "raw_rows",
        "rows",
        "reconstruction_errors",
        "anomaly_flags",
        "predictions",
        "targets",
        "epoch_losses",
    }


    assert (
        forbidden
        .isdisjoint(
            manifest
        )
    )


# ============================================================
# CONTRACT FINGERPRINT
# ============================================================


def test_contract_fingerprint_mismatch_is_blocked(
) -> None:

    contract = (
        build_contract()
    )


    (
        _,
        _,
        model,
        preprocessor,
        threshold,
    ) = (
        build_components(
            contract=
                contract
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


    different_contract = (
        build_contract(
            workflow_id=
                "prep:different-autoencoder-bundle"
        )
    )


    try:

        deserialize_trusted_tabular_autoencoder_bundle(
            trusted_bundle_bytes=
                bundle,
            training_contract=
                different_contract,
        )

    except TabularAutoencoderBundleIntegrityError as error:

        assert (
            "fingerprint"
            in
            str(
                error
            )
        )

        return


    raise AssertionError(
        (
            "Training Contract fingerprint "
            "mismatch must fail closed."
        )
    )


# ============================================================
# THRESHOLD POLICY BINDING
# ============================================================


def test_threshold_contract_mismatch_is_blocked_on_serialize(
) -> None:

    contract = (
        build_contract(
            threshold_quantile=
                0.975
        )
    )


    (
        _,
        _,
        model,
        preprocessor,
        _,
    ) = (
        build_components(
            contract=
                contract
        )
    )


    wrong_threshold = (
        ReconstructionErrorThreshold(
            quantile=
                0.95,
            threshold=
                0.42,
            train_rows=
                8,
            method=
                AUTOENCODER_THRESHOLD_METHOD,
            comparison_operator=
                AUTOENCODER_THRESHOLD_COMPARISON,
        )
    )


    try:

        serialize_tabular_autoencoder_bundle(
            model=
                model,
            preprocessor=
                preprocessor,
            training_contract=
                contract,
            threshold=
                wrong_threshold,
        )

    except TabularAutoencoderBundleContractError:
        return


    raise AssertionError(
        (
            "Threshold / Training Contract "
            "quantile mismatch must fail closed."
        )
    )


# ============================================================
# ARCHITECTURE BINDING
# ============================================================


def test_model_contract_architecture_mismatch_is_blocked(
) -> None:

    contract = (
        build_contract(
            hidden_features=
                8,
            latent_features=
                3,
        )
    )


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


    wrong_model = (
        TabularAutoencoder(
            input_features=
                int(
                    transformed.shape[
                        1
                    ]
                ),
            hidden_features=
                7,
            latent_features=
                3,
        )
    )


    threshold = (
        ReconstructionErrorThreshold(
            quantile=
                0.975,
            threshold=
                0.42,
            train_rows=
                8,
            method=
                AUTOENCODER_THRESHOLD_METHOD,
            comparison_operator=
                AUTOENCODER_THRESHOLD_COMPARISON,
        )
    )


    try:

        serialize_tabular_autoencoder_bundle(
            model=
                wrong_model,
            preprocessor=
                preprocessor,
            training_contract=
                contract,
            threshold=
                threshold,
        )

    except TabularAutoencoderBundleContractError:
        return


    raise AssertionError(
        (
            "Model / contract architecture "
            "mismatch must fail closed."
        )
    )


# ============================================================
# FITTED PREPROCESSOR AUTHORITY
# ============================================================


def test_unfitted_preprocessor_is_blocked(
) -> None:

    contract = (
        build_contract()
    )


    preprocessor = (
        build_ml_preprocessor(
            contract=
                contract
        )
    )


    model = (
        TabularAutoencoder(
            input_features=
                5,
            hidden_features=
                8,
            latent_features=
                3,
        )
    )


    threshold = (
        ReconstructionErrorThreshold(
            quantile=
                0.975,
            threshold=
                0.42,
            train_rows=
                8,
            method=
                AUTOENCODER_THRESHOLD_METHOD,
            comparison_operator=
                AUTOENCODER_THRESHOLD_COMPARISON,
        )
    )


    try:

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

    except TabularAutoencoderBundleContractError as error:

        assert (
            "fitted"
            in
            str(
                error
            )
        )

        return


    raise AssertionError(
        (
            "Unfitted preprocessor "
            "must fail closed."
        )
    )


# ============================================================
# TAMPER GUARD BEFORE DESERIALIZATION
# ============================================================


def test_state_tampering_is_rejected_before_any_decoder(
) -> None:

    contract = (
        build_contract()
    )


    (
        _,
        _,
        model,
        preprocessor,
        threshold,
    ) = (
        build_components(
            contract=
                contract
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


    members = (
        read_bundle_members(
            bundle
        )
    )


    members[
        "model_state.pt"
    ] = (
        members[
            "model_state.pt"
        ]
        +
        b"TAMPERED"
    )


    tampered = (
        rebuild_bundle(
            members
        )
    )


    original_joblib_load = (
        bundle_module.joblib.load
    )


    original_torch_load = (
        bundle_module.torch.load
    )


    decoder_called = {
        "joblib":
            False,
        "torch":
            False,
    }


    def forbidden_joblib_load(
        *args,
        **kwargs,
    ):

        decoder_called[
            "joblib"
        ] = True

        raise AssertionError(
            (
                "joblib.load must not run before "
                "component integrity validation."
            )
        )


    def forbidden_torch_load(
        *args,
        **kwargs,
    ):

        decoder_called[
            "torch"
        ] = True

        raise AssertionError(
            (
                "torch.load must not run before "
                "component integrity validation."
            )
        )


    bundle_module.joblib.load = (
        forbidden_joblib_load
    )


    bundle_module.torch.load = (
        forbidden_torch_load
    )


    try:

        try:

            deserialize_trusted_tabular_autoencoder_bundle(
                trusted_bundle_bytes=
                    tampered,
                training_contract=
                    contract,
            )

        except TabularAutoencoderBundleIntegrityError as error:

            assert (
                "state_dict"
                in
                str(
                    error
                )
            )

        else:

            raise AssertionError(
                (
                    "Tampered state_dict "
                    "must fail closed."
                )
            )


    finally:

        bundle_module.joblib.load = (
            original_joblib_load
        )

        bundle_module.torch.load = (
            original_torch_load
        )


    assert (
        decoder_called
        ==
        {
            "joblib":
                False,
            "torch":
                False,
        }
    )


def test_preprocessor_tampering_is_rejected_before_any_decoder(
) -> None:

    contract = (
        build_contract()
    )


    (
        _,
        _,
        model,
        preprocessor,
        threshold,
    ) = (
        build_components(
            contract=
                contract
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


    members = (
        read_bundle_members(
            bundle
        )
    )


    members[
        "preprocessor.joblib"
    ] = (
        members[
            "preprocessor.joblib"
        ]
        +
        b"TAMPERED"
    )


    tampered = (
        rebuild_bundle(
            members
        )
    )


    original_joblib_load = (
        bundle_module.joblib.load
    )


    original_torch_load = (
        bundle_module.torch.load
    )


    decoder_called = {
        "joblib":
            False,
        "torch":
            False,
    }


    def forbidden_joblib_load(
        *args,
        **kwargs,
    ):

        decoder_called[
            "joblib"
        ] = True

        raise AssertionError(
            "joblib.load executed too early."
        )


    def forbidden_torch_load(
        *args,
        **kwargs,
    ):

        decoder_called[
            "torch"
        ] = True

        raise AssertionError(
            "torch.load executed too early."
        )


    bundle_module.joblib.load = (
        forbidden_joblib_load
    )


    bundle_module.torch.load = (
        forbidden_torch_load
    )


    try:

        try:

            deserialize_trusted_tabular_autoencoder_bundle(
                trusted_bundle_bytes=
                    tampered,
                training_contract=
                    contract,
            )

        except TabularAutoencoderBundleIntegrityError as error:

            assert (
                "preprocessor"
                in
                str(
                    error
                )
            )

        else:

            raise AssertionError(
                (
                    "Tampered preprocessor "
                    "must fail closed."
                )
            )


    finally:

        bundle_module.joblib.load = (
            original_joblib_load
        )

        bundle_module.torch.load = (
            original_torch_load
        )


    assert (
        decoder_called
        ==
        {
            "joblib":
                False,
            "torch":
                False,
        }
    )


# ============================================================
# UNKNOWN MEMBER SURFACE
# ============================================================


def test_extra_bundle_member_is_blocked(
) -> None:

    contract = (
        build_contract()
    )


    (
        _,
        _,
        model,
        preprocessor,
        threshold,
    ) = (
        build_components(
            contract=
                contract
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


    members = (
        read_bundle_members(
            bundle
        )
    )


    output = (
        io.BytesIO()
    )


    with zipfile.ZipFile(
        output,
        mode="w",
        compression=
            zipfile.ZIP_DEFLATED,
    ) as archive:

        for name, payload in members.items():

            archive.writestr(
                name,
                payload,
            )


        archive.writestr(
            "invented.bin",
            b"NO",
        )


    try:

        deserialize_trusted_tabular_autoencoder_bundle(
            trusted_bundle_bytes=
                output.getvalue(),
            training_contract=
                contract,
        )

    except TabularAutoencoderBundleIntegrityError:
        return


    raise AssertionError(
        (
            "Unexpected bundle member "
            "must fail closed."
        )
    )


# ============================================================
# VERSION
# ============================================================


def test_codec_rule_version(
) -> None:

    assert (
        DL_TABULAR_AUTOENCODER_BUNDLE_CODEC_RULE_VERSION
        ==
        "dl_tabular_autoencoder_bundle_codec_v0.1"
    )


# ============================================================
# MAIN
# ============================================================


def main(
) -> None:

    print(
        (
            "=== DATALENS TABULAR AUTOENCODER "
            "BUNDLE ROUND-TRIP v0.1 ==="
        )
    )

    print()


    test_exact_model_preprocessor_threshold_round_trip()

    print(
        (
            "Exact state_dict / preprocessor / "
            "threshold round-trip: PASS"
        )
    )


    test_minimal_bundle_member_and_manifest_surface()

    print(
        "Minimal bundle / manifest surface: PASS"
    )


    test_contract_fingerprint_mismatch_is_blocked()

    print(
        "Training Contract fingerprint guard: PASS"
    )


    test_threshold_contract_mismatch_is_blocked_on_serialize()

    print(
        "Frozen threshold policy guard: PASS"
    )


    test_model_contract_architecture_mismatch_is_blocked()

    print(
        "Model / contract architecture guard: PASS"
    )


    test_unfitted_preprocessor_is_blocked()

    print(
        "Fitted preprocessing authority: PASS"
    )


    test_state_tampering_is_rejected_before_any_decoder()

    print(
        (
            "State SHA guard before deserialization: "
            "PASS"
        )
    )


    test_preprocessor_tampering_is_rejected_before_any_decoder()

    print(
        (
            "Preprocessor SHA guard before "
            "deserialization: PASS"
        )
    )


    test_extra_bundle_member_is_blocked()

    print(
        "Exact ZIP member surface: PASS"
    )


    test_codec_rule_version()

    print(
        "Autoencoder bundle codec rule version: PASS"
    )


    print()

    print(
        (
            "PASS - DataLens Tabular Autoencoder "
            "Bundle Round-Trip v0.1"
        )
    )


if __name__ == "__main__":
    main()
