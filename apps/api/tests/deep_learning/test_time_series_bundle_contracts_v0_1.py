from __future__ import annotations


import ast
from pathlib import Path


from pydantic import (
    ValidationError,
)


from app.deep_learning.time_series_bundle_contracts import (
    DL_TIME_SERIES_NEURAL_BUNDLE_RULE_VERSION,
    DL_TIME_SERIES_STANDARDIZER_SNAPSHOT_RULE_VERSION,
    DLTimeSeriesNeuralBundleManifest,
    DLTimeSeriesStandardizerSnapshot,
)


# ============================================================
# HELPERS
# ============================================================


SHA_A = (
    "a"
    *
    64
)


SHA_B = (
    "b"
    *
    64
)


SHA_C = (
    "c"
    *
    64
)


def require_validation_error(
    callback,
    *,
    label: str,
) -> None:

    try:

        callback()

    except ValidationError:

        return


    raise AssertionError(
        (
            "Expected bundle-contract failure: "
            f"{label}"
        )
    )


# ============================================================
# 1. STANDARDIZER SNAPSHOT
# ============================================================


snapshot = (
    DLTimeSeriesStandardizerSnapshot(
        mean=
            64.5,

        standard_deviation=
            8.65544144839919,

        fitted_observation_count=
            30,

        scaling_rule_version=
            "dl_time_series_scaling_v0.1",
    )
)


assert (
    snapshot.snapshot_version
    ==
    DL_TIME_SERIES_STANDARDIZER_SNAPSHOT_RULE_VERSION
)


assert (
    snapshot.mean
    ==
    64.5
)


assert (
    snapshot.standard_deviation
    >
    0.0
)


assert (
    snapshot.fitted_observation_count
    ==
    30
)


print(
    "[PASS] TRAIN-only standardizer state has explicit JSON contract"
)


# ============================================================
# 2. INVALID STANDARDIZER STATE FAILS CLOSED
# ============================================================


require_validation_error(
    lambda:
        DLTimeSeriesStandardizerSnapshot(
            mean=
                float(
                    "nan"
                ),

            standard_deviation=
                1.0,

            fitted_observation_count=
                30,

            scaling_rule_version=
                "dl_time_series_scaling_v0.1",
        ),
    label=
        "non-finite mean",
)


require_validation_error(
    lambda:
        DLTimeSeriesStandardizerSnapshot(
            mean=
                0.0,

            standard_deviation=
                0.0,

            fitted_observation_count=
                30,

            scaling_rule_version=
                "dl_time_series_scaling_v0.1",
        ),
    label=
        "zero standard deviation",
)


require_validation_error(
    lambda:
        DLTimeSeriesStandardizerSnapshot(
            mean=
                0.0,

            standard_deviation=
                1.0,

            fitted_observation_count=
                1,

            scaling_rule_version=
                "dl_time_series_scaling_v0.1",
        ),
    label=
        "insufficient fitted observations",
)


print(
    "[PASS] invalid persisted scaling state fails closed"
)


# ============================================================
# 3. VALID MLP MANIFEST
# ============================================================


mlp_manifest = (
    DLTimeSeriesNeuralBundleManifest(
        estimator_key=
            "time_series_mlp_regressor",

        network_class=
            "FeedForwardRegressor",

        lookback=
            12,

        hidden_width=
            32,

        training_contract_sha256=
            SHA_A,

        state_dict_sha256=
            SHA_B,

        standardizer_sha256=
            SHA_C,
    )
)


assert (
    mlp_manifest.bundle_version
    ==
    DL_TIME_SERIES_NEURAL_BUNDLE_RULE_VERSION
)


assert (
    mlp_manifest.state_dict_format
    ==
    "torch_state_dict"
)


assert (
    mlp_manifest.standardizer_format
    ==
    "datalens_json"
)


print(
    "[PASS] temporal MLP bundle manifest identity"
)


# ============================================================
# 4. VALID RNN MANIFEST
# ============================================================


rnn_manifest = (
    DLTimeSeriesNeuralBundleManifest(
        estimator_key=
            "time_series_rnn_regressor",

        network_class=
            "TimeSeriesRNNRegressor",

        lookback=
            12,

        hidden_width=
            24,

        training_contract_sha256=
            SHA_A,

        state_dict_sha256=
            SHA_B,

        standardizer_sha256=
            SHA_C,
    )
)


assert (
    rnn_manifest.network_class
    ==
    "TimeSeriesRNNRegressor"
)


print(
    "[PASS] temporal RNN bundle manifest identity"
)


# ============================================================
# 5. VALID LSTM MANIFEST
# ============================================================


lstm_manifest = (
    DLTimeSeriesNeuralBundleManifest(
        estimator_key=
            "time_series_lstm_regressor",

        network_class=
            "TimeSeriesLSTMRegressor",

        lookback=
            12,

        hidden_width=
            24,

        training_contract_sha256=
            SHA_A,

        state_dict_sha256=
            SHA_B,

        standardizer_sha256=
            SHA_C,
    )
)


assert (
    lstm_manifest.network_class
    ==
    "TimeSeriesLSTMRegressor"
)


print(
    "[PASS] temporal LSTM bundle manifest identity"
)


# ============================================================
# 6. ESTIMATOR / NETWORK MISMATCH FAILS CLOSED
# ============================================================


require_validation_error(
    lambda:
        DLTimeSeriesNeuralBundleManifest(
            estimator_key=
                "time_series_rnn_regressor",

            network_class=
                "TimeSeriesLSTMRegressor",

            lookback=
                12,

            hidden_width=
                24,

            training_contract_sha256=
                SHA_A,

            state_dict_sha256=
                SHA_B,

            standardizer_sha256=
                SHA_C,
        ),
    label=
        "RNN estimator with LSTM network",
)


require_validation_error(
    lambda:
        DLTimeSeriesNeuralBundleManifest(
            estimator_key=
                "time_series_mlp_regressor",

            network_class=
                "TimeSeriesRNNRegressor",

            lookback=
                12,

            hidden_width=
                32,

            training_contract_sha256=
                SHA_A,

            state_dict_sha256=
                SHA_B,

            standardizer_sha256=
                SHA_C,
        ),
    label=
        "MLP estimator with RNN network",
)


print(
    "[PASS] estimator/network identity mismatch fails closed"
)


# ============================================================
# 7. UNKNOWN ESTIMATOR FAILS CLOSED
# ============================================================


require_validation_error(
    lambda:
        DLTimeSeriesNeuralBundleManifest(
            estimator_key=
                "time_series_gru_regressor",

            network_class=
                "TimeSeriesRNNRegressor",

            lookback=
                12,

            hidden_width=
                24,

            training_contract_sha256=
                SHA_A,

            state_dict_sha256=
                SHA_B,

            standardizer_sha256=
                SHA_C,
        ),
    label=
        "unsupported GRU estimator",
)


print(
    "[PASS] unsupported temporal estimator fails closed"
)


# ============================================================
# 8. INVALID STRUCTURAL VALUES FAIL CLOSED
# ============================================================


require_validation_error(
    lambda:
        DLTimeSeriesNeuralBundleManifest(
            estimator_key=
                "time_series_mlp_regressor",

            network_class=
                "FeedForwardRegressor",

            lookback=
                0,

            hidden_width=
                32,

            training_contract_sha256=
                SHA_A,

            state_dict_sha256=
                SHA_B,

            standardizer_sha256=
                SHA_C,
        ),
    label=
        "lookback=0",
)


require_validation_error(
    lambda:
        DLTimeSeriesNeuralBundleManifest(
            estimator_key=
                "time_series_mlp_regressor",

            network_class=
                "FeedForwardRegressor",

            lookback=
                12,

            hidden_width=
                0,

            training_contract_sha256=
                SHA_A,

            state_dict_sha256=
                SHA_B,

            standardizer_sha256=
                SHA_C,
        ),
    label=
        "hidden_width=0",
)


require_validation_error(
    lambda:
        DLTimeSeriesNeuralBundleManifest(
            estimator_key=
                "time_series_mlp_regressor",

            network_class=
                "FeedForwardRegressor",

            lookback=
                12,

            hidden_width=
                32,

            training_contract_sha256=
                "not-a-sha256",

            state_dict_sha256=
                SHA_B,

            standardizer_sha256=
                SHA_C,
        ),
    label=
        "invalid training-contract SHA",
)


print(
    "[PASS] invalid temporal bundle structure fails closed"
)


# ============================================================
# 9. EXTRA / LEARNED EVIDENCE FIELDS FAIL CLOSED
# ============================================================


manifest_payload = (
    mlp_manifest.model_dump(
        mode="json"
    )
)


for forbidden_field in (
    "raw_rows",
    "targets",
    "predictions",
    "epoch_losses",
    "optimizer_state",
    "execution_device",
):

    payload = dict(
        manifest_payload
    )

    payload[
        forbidden_field
    ] = "forbidden"


    require_validation_error(
        lambda payload=payload:
            DLTimeSeriesNeuralBundleManifest
            .model_validate(
                payload
            ),
        label=
            forbidden_field,
    )


print(
    "[PASS] manifest excludes raw/evaluation/execution evidence"
)


# ============================================================
# 10. DETERMINISTIC JSON ROUND-TRIP
# ============================================================


for manifest in (
    mlp_manifest,
    rnn_manifest,
    lstm_manifest,
):

    payload = manifest.model_dump(
        mode="json"
    )


    restored = (
        DLTimeSeriesNeuralBundleManifest
        .model_validate(
            payload
        )
    )


    assert (
        restored
        ==
        manifest
    )


snapshot_payload = snapshot.model_dump(
    mode="json"
)


assert (
    DLTimeSeriesStandardizerSnapshot
    .model_validate(
        snapshot_payload
    )
    ==
    snapshot
)


print(
    "[PASS] temporal bundle metadata round-trips deterministically"
)


# ============================================================
# 11. CONTRACT MODULE MUST REMAIN TORCH-FREE
# ============================================================


source_path = Path(
    "app/deep_learning/time_series_bundle_contracts.py"
)


source = source_path.read_text(
    encoding="utf-8"
)


tree = ast.parse(
    source
)


imports = []


for node in ast.walk(
    tree
):

    if isinstance(
        node,
        ast.Import,
    ):

        imports.extend(
            alias.name
            for alias
            in node.names
        )


    elif isinstance(
        node,
        ast.ImportFrom,
    ):

        if node.module:

            imports.append(
                node.module
            )


assert not any(
    name == "torch"
    or name.startswith(
        "torch."
    )
    for name in imports
)


print(
    "[PASS] temporal bundle contract module remains torch-free"
)


print()
print("=" * 80)
print("DL-5-A11-V2-P1 FINAL VERDICT")
print("=" * 80)
print()

print("Shared temporal bundle version              PASS")
print("Standardizer JSON snapshot contract         PASS")
print("MLP manifest identity                      PASS")
print("RNN manifest identity                      PASS")
print("LSTM manifest identity                     PASS")
print("Estimator/network consistency               PASS")
print("Structural checksum contracts               PASS")
print("Privacy-minimal manifest                    PASS")
print("Deterministic metadata round-trip            PASS")
print("Torch-free contract boundary                PASS")

print()
print(
    "DL-5-A11-V2-P1 - TEMPORAL NEURAL BUNDLE CONTRACT: PASS"
)
