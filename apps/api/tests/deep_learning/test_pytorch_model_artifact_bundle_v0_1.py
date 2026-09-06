from __future__ import annotations


import io
import json
import zipfile


import numpy as np
import pandas as pd
import torch


from app.deep_learning.model_bundle import (
    DL_TABULAR_MLP_BUNDLE_RULE_VERSION,
    TabularMLPBundleContractError,
    TabularMLPBundleIntegrityError,
    deserialize_trusted_tabular_mlp_bundle,
    serialize_tabular_mlp_bundle,
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


from app.ml.experiment_provenance import (
    ml_training_contract_sha256,
)


from app.ml.preprocessing import (
    build_ml_preprocessor,
)


# ============================================================
# FIXTURE
# ============================================================


def build_contract(
    *,
    workflow_id: str = "prep:bundle",
    hidden_features: int = 12,
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
                    hidden_features,

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
                    70.0,
                    80.0,
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


def build_fitted_components(
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
        73
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


    return (
        features,
        transformed,
        model,
        preprocessor,
    )


def predictions(
    *,
    model: FeedForwardRegressor,
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
            dtype=np.float64,
        )
    )


# ============================================================
# ROUND TRIP
# ============================================================


def test_state_dict_and_preprocessor_round_trip(
) -> None:

    contract = (
        build_contract()
    )


    (
        features,
        transformed,
        model,
        preprocessor,
    ) = (
        build_fitted_components(
            contract=contract
        )
    )


    expected = (
        predictions(
            model=model,
            transformed=transformed,
        )
    )


    bundle = (
        serialize_tabular_mlp_bundle(
            model=model,
            preprocessor=preprocessor,
            training_contract=contract,
        )
    )


    restored = (
        deserialize_trusted_tabular_mlp_bundle(
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


    actual = (
        predictions(
            model=
                restored.model,

            transformed=
                restored_transformed,
        )
    )


    np.testing.assert_allclose(
        actual,
        expected,
        rtol=0.0,
        atol=0.0,
    )


    assert (
        restored.manifest.training_contract_sha256
        ==
        ml_training_contract_sha256(
            contract
        )
    )


    assert (
        restored.model.input_features
        ==
        model.input_features
    )


    assert (
        restored.model.hidden_features
        ==
        model.hidden_features
    )


# ============================================================
# BUNDLE STRUCTURE
# ============================================================


def test_bundle_contains_only_expected_members(
) -> None:

    contract = (
        build_contract()
    )


    (
        _,
        _,
        model,
        preprocessor,
    ) = (
        build_fitted_components(
            contract=contract
        )
    )


    bundle = (
        serialize_tabular_mlp_bundle(
            model=model,
            preprocessor=preprocessor,
            training_contract=contract,
        )
    )


    with zipfile.ZipFile(
        io.BytesIO(
            bundle
        ),
        mode="r",
    ) as archive:

        assert (
            set(
                archive.namelist()
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
                archive
                .read(
                    "manifest.json"
                )
                .decode(
                    "utf-8"
                )
            )
        )


    assert (
        manifest[
            "network_class"
        ]
        ==
        "FeedForwardRegressor"
    )


    assert (
        manifest[
            "state_dict_format"
        ]
        ==
        "torch_state_dict"
    )


    assert (
        manifest[
            "preprocessor_format"
        ]
        ==
        "joblib"
    )


    forbidden = {
        "rows",
        "predictions",
        "targets",
        "target_values",
        "epoch_losses",
    }


    assert (
        forbidden
        .isdisjoint(
            manifest
        )
    )


# ============================================================
# MODEL / CONTRACT IDENTITY
# ============================================================


def test_hidden_feature_mismatch_is_blocked(
) -> None:

    contract = (
        build_contract(
            hidden_features=
                12
        )
    )


    features = (
        build_features()
    )


    preprocessor = (
        build_ml_preprocessor(
            contract=contract
        )
    )


    transformed = (
        preprocessor
        .fit_transform(
            features
        )
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
                7,
        )
    )


    try:

        serialize_tabular_mlp_bundle(
            model=model,
            preprocessor=preprocessor,
            training_contract=contract,
        )

    except TabularMLPBundleContractError:
        return


    raise AssertionError(
        (
            "Model / Training Contract hidden "
            "feature mismatch must fail closed."
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
    ) = (
        build_fitted_components(
            contract=contract
        )
    )


    bundle = (
        serialize_tabular_mlp_bundle(
            model=model,
            preprocessor=preprocessor,
            training_contract=contract,
        )
    )


    different_contract = (
        build_contract(
            workflow_id=
                "prep:different"
        )
    )


    try:

        deserialize_trusted_tabular_mlp_bundle(
            trusted_bundle_bytes=
                bundle,

            training_contract=
                different_contract,
        )

    except TabularMLPBundleIntegrityError as error:

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
# INTERNAL COMPONENT TAMPERING
# ============================================================


def test_state_dict_component_tampering_is_blocked(
) -> None:

    contract = (
        build_contract()
    )


    (
        _,
        _,
        model,
        preprocessor,
    ) = (
        build_fitted_components(
            contract=contract
        )
    )


    bundle = (
        serialize_tabular_mlp_bundle(
            model=model,
            preprocessor=preprocessor,
            training_contract=contract,
        )
    )


    source = zipfile.ZipFile(
        io.BytesIO(
            bundle
        ),
        mode="r",
    )


    try:

        manifest = source.read(
            "manifest.json"
        )


        state = (
            source.read(
                "model_state.pt"
            )
            +
            b"TAMPERED"
        )


        preprocessor_bytes = (
            source.read(
                "preprocessor.joblib"
            )
        )

    finally:

        source.close()


    output = io.BytesIO()


    with zipfile.ZipFile(
        output,
        mode="w",
        compression=
            zipfile.ZIP_DEFLATED,
    ) as archive:

        archive.writestr(
            "manifest.json",
            manifest,
        )


        archive.writestr(
            "model_state.pt",
            state,
        )


        archive.writestr(
            "preprocessor.joblib",
            preprocessor_bytes,
        )


    tampered = (
        output.getvalue()
    )


    try:

        deserialize_trusted_tabular_mlp_bundle(
            trusted_bundle_bytes=
                tampered,

            training_contract=
                contract,
        )

    except TabularMLPBundleIntegrityError as error:

        assert (
            "state_dict"
            in
            str(
                error
            )
        )

        return


    raise AssertionError(
        (
            "State component tampering "
            "must fail closed."
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
            contract=contract
        )
    )


    model = (
        FeedForwardRegressor(
            input_features=
                4,

            hidden_features=
                12,
        )
    )


    try:

        serialize_tabular_mlp_bundle(
            model=model,
            preprocessor=preprocessor,
            training_contract=contract,
        )

    except TabularMLPBundleContractError as error:

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
            "Unfitted preprocessing state "
            "must fail closed."
        )
    )


# ============================================================
# RULE VERSION
# ============================================================


def test_rule_version(
) -> None:

    assert (
        DL_TABULAR_MLP_BUNDLE_RULE_VERSION
        ==
        "dl_tabular_mlp_bundle_v0.1"
    )


# ============================================================
# MAIN
# ============================================================


def main(
) -> None:

    print(
        "=== DATALENS PYTORCH MODEL ARTIFACT BUNDLE v0.1 ==="
    )

    print()


    test_state_dict_and_preprocessor_round_trip()

    print(
        "state_dict + fitted preprocessor round-trip: PASS"
    )


    test_bundle_contains_only_expected_members()

    print(
        "Minimal bundle member surface: PASS"
    )


    test_hidden_feature_mismatch_is_blocked()

    print(
        "Model / contract architecture guard: PASS"
    )


    test_contract_fingerprint_mismatch_is_blocked()

    print(
        "Training Contract fingerprint guard: PASS"
    )


    test_state_dict_component_tampering_is_blocked()

    print(
        "Internal component SHA guard: PASS"
    )


    test_unfitted_preprocessor_is_blocked()

    print(
        "Fitted preprocessing authority: PASS"
    )


    test_rule_version()

    print(
        "PyTorch bundle rule version: PASS"
    )


    print()

    print(
        "PASS - DataLens PyTorch Model Artifact Bundle v0.1"
    )


if __name__ == "__main__":
    main()
