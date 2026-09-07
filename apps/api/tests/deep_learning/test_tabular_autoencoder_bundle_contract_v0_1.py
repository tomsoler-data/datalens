from __future__ import annotations


import json
import sys


from pydantic import (
    ValidationError,
)


from app.deep_learning.autoencoder_bundle_contracts import (
    DL_TABULAR_AUTOENCODER_BUNDLE_RULE_VERSION,
    TabularAutoencoderBundleManifest,
)


# ============================================================
# FIXTURE
# ============================================================


def valid_payload(
) -> dict[
    str,
    object,
]:

    return {
        "estimator_key":
            "tabular_autoencoder",

        "network_class":
            "TabularAutoencoder",

        "state_dict_format":
            "torch_state_dict",

        "preprocessor_format":
            "joblib",

        "input_features":
            12,

        "hidden_features":
            8,

        "latent_features":
            3,

        "training_contract_sha256":
            "a"
            *
            64,

        "threshold_rule_version":
            "autoencoder_threshold_v0.1",

        "threshold_quantile":
            0.975,

        "anomaly_threshold":
            0.42,

        "threshold_train_rows":
            80,

        "threshold_method":
            "train_reconstruction_error_quantile",

        "threshold_comparison_operator":
            "greater_than",

        "state_dict_sha256":
            "b"
            *
            64,

        "preprocessor_sha256":
            "c"
            *
            64,
    }


def assert_rejected(
    payload: dict[
        str,
        object,
    ],
) -> None:

    try:

        TabularAutoencoderBundleManifest.model_validate(
            payload
        )

    except ValidationError:
        return


    raise AssertionError(
        "Invalid Autoencoder bundle manifest was accepted."
    )


print(
    "=== DATALENS TABULAR AUTOENCODER BUNDLE CONTRACT v0.1 ==="
)

print()


# ============================================================
# 1. VALID MANIFEST
# ============================================================


manifest = (
    TabularAutoencoderBundleManifest.model_validate(
        valid_payload()
    )
)


assert (
    manifest.bundle_version
    ==
    DL_TABULAR_AUTOENCODER_BUNDLE_RULE_VERSION
)


assert (
    manifest.estimator_key
    ==
    "tabular_autoencoder"
)


assert (
    manifest.network_class
    ==
    "TabularAutoencoder"
)


assert (
    manifest.input_features
    ==
    12
)


assert (
    manifest.hidden_features
    ==
    8
)


assert (
    manifest.latent_features
    ==
    3
)


print(
    "Autoencoder bundle identity: PASS"
)


# ============================================================
# 2. ARCHITECTURE CONTRACT
# ============================================================


bad = valid_payload()
bad[
    "input_features"
] = 0

assert_rejected(
    bad
)


bad = valid_payload()
bad[
    "hidden_features"
] = 0

assert_rejected(
    bad
)


bad = valid_payload()
bad[
    "latent_features"
] = 0

assert_rejected(
    bad
)


bad = valid_payload()
bad[
    "latent_features"
] = 9

assert_rejected(
    bad
)


print(
    "Autoencoder architecture invariants: PASS"
)


# ============================================================
# 3. THRESHOLD CONTRACT
# ============================================================


for invalid_quantile in (
    0.0,
    1.0,
    -0.1,
    1.1,
):

    bad = valid_payload()

    bad[
        "threshold_quantile"
    ] = invalid_quantile

    assert_rejected(
        bad
    )


bad = valid_payload()
bad[
    "anomaly_threshold"
] = -0.01

assert_rejected(
    bad
)


bad = valid_payload()
bad[
    "anomaly_threshold"
] = float(
    "nan"
)

assert_rejected(
    bad
)


bad = valid_payload()
bad[
    "anomaly_threshold"
] = float(
    "inf"
)

assert_rejected(
    bad
)


bad = valid_payload()
bad[
    "threshold_train_rows"
] = 1

assert_rejected(
    bad
)


bad = valid_payload()
bad[
    "threshold_method"
] = "test_quantile"

assert_rejected(
    bad
)


bad = valid_payload()
bad[
    "threshold_comparison_operator"
] = "greater_than_or_equal"

assert_rejected(
    bad
)


print(
    "Frozen TRAIN threshold semantics: PASS"
)


# ============================================================
# 4. HASH CONTRACT
# ============================================================


for field_name in (
    "training_contract_sha256",
    "state_dict_sha256",
    "preprocessor_sha256",
):

    bad = valid_payload()

    bad[
        field_name
    ] = "abc"

    assert_rejected(
        bad
    )


    bad = valid_payload()

    bad[
        field_name
    ] = (
        "G"
        *
        64
    )

    assert_rejected(
        bad
    )


print(
    "Bundle SHA-256 manifest guards: PASS"
)


# ============================================================
# 5. FAMILY FAIL-CLOSED
# ============================================================


bad = valid_payload()
bad[
    "estimator_key"
] = "tabular_mlp_regressor"

assert_rejected(
    bad
)


bad = valid_payload()
bad[
    "network_class"
] = "FeedForwardRegressor"

assert_rejected(
    bad
)


bad = valid_payload()
bad[
    "threshold_rule_version"
] = "autoencoder_threshold_v999"

assert_rejected(
    bad
)


bad = valid_payload()
bad[
    "invented_field"
] = True

assert_rejected(
    bad
)


print(
    "Bundle family fail-closed guards: PASS"
)


# ============================================================
# 6. FROZEN / ROUND TRIP
# ============================================================


payload = (
    manifest.model_dump(
        mode="json"
    )
)


serialized = json.dumps(
    payload,
    ensure_ascii=False,
    sort_keys=True,
    separators=(
        ",",
        ":",
    ),
)


restored = (
    TabularAutoencoderBundleManifest.model_validate(
        json.loads(
            serialized
        )
    )
)


assert (
    restored
    ==
    manifest
)


try:

    manifest.input_features = 99

except ValidationError:
    pass

else:

    raise AssertionError(
        "Bundle manifest must be frozen."
    )


print(
    "Frozen manifest JSON round trip: PASS"
)


# ============================================================
# 7. PRIVACY / RUNTIME BOUNDARY
# ============================================================


field_names = set(
    TabularAutoencoderBundleManifest.model_fields
)


for forbidden in (
    "raw_rows",
    "train_rows_data",
    "test_rows_data",
    "reconstruction_errors",
    "anomaly_flags",
    "predictions",
    "targets",
):

    assert (
        forbidden
        not in
        field_names
    )


assert (
    "torch"
    not in
    sys.modules
)


print(
    "Privacy-minimal / torch-free manifest: PASS"
)


assert (
    DL_TABULAR_AUTOENCODER_BUNDLE_RULE_VERSION
    ==
    "dl_tabular_autoencoder_bundle_v0.1"
)


print(
    "Autoencoder bundle rule version: PASS"
)


print()

print(
    "PASS - DataLens Tabular Autoencoder Bundle Contract v0.1"
)
