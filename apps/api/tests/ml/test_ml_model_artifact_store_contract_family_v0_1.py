from __future__ import annotations


import hashlib
import sys


import app.ml.model_artifact_store as store_module


from app.ml.anomaly_contracts import (
    MLAnomalyTrainingContract,
)


from app.ml.contracts import (
    MLTrainingContract,
)


from app.ml.experiment_provenance import (
    ml_model_training_contract_sha256,
)


from app.ml.model_artifact_index import (
    validate_ml_model_artifact_index_entry,
)


from app.ml.model_artifact_store import (
    register_ml_model_artifact,
)


# ============================================================
# CONTRACTS
# ============================================================


def build_supervised_contract(
) -> MLTrainingContract:

    return (
        MLTrainingContract(
            workflow_id=
                "workflow:store-supervised",
            dataset_id=
                "dataset:store-supervised",
            problem_type=
                "regression",
            target_column=
                "target",
            feature_columns=[
                "x1",
                "x2",
            ],
            estimator_key=
                "tabular_mlp_regressor",
        )
    )


def build_anomaly_contract(
) -> MLAnomalyTrainingContract:

    return (
        MLAnomalyTrainingContract(
            workflow_id=
                "workflow:store-anomaly",
            dataset_id=
                "dataset:store-anomaly",
            feature_columns=[
                "x1",
                "x2",
            ],
            threshold_quantile=
                0.975,
        )
    )


# ============================================================
# CONTROLLED STORE SIDE EFFECTS
# ============================================================


def execute_with_controlled_store(
    contract,
    *,
    model_bytes: bytes,
    metrics: dict[
        str,
        float,
    ],
):

    originals = {
        "authority":
            store_module
            ._assert_preparation_authority,

        "write":
            store_module
            .write_ml_model_binary,

        "upsert":
            store_module
            .upsert_ml_model_artifact_index_entry,

        "delete":
            store_module
            .delete_ml_model_binary,
    }


    captured = {
        "authority_calls":
            0,
        "write_calls":
            0,
        "upsert_calls":
            0,
        "delete_calls":
            0,
        "entry":
            None,
        "revision":
            None,
    }


    def fake_authority(
        *,
        contract,
        preparation_session_revision=None,
    ):

        captured[
            "authority_calls"
        ] += 1

        captured[
            "authority_contract"
        ] = contract

        captured[
            "authority_revision"
        ] = (
            preparation_session_revision
        )


    def fake_write(
        *,
        store_path,
        model_id,
        model_bytes,
        serialization_format,
    ):

        captured[
            "write_calls"
        ] += 1


        assert (
            serialization_format
            ==
            "pytorch_bundle"
        )


        return {
            "model_path":
                (
                    "models/"
                    +
                    model_id
                    +
                    ".ptbundle"
                ),

            "model_file_bytes":
                len(
                    model_bytes
                ),

            "model_sha256":
                hashlib.sha256(
                    model_bytes
                )
                .hexdigest(),
        }


    def fake_upsert(
        *,
        store_path,
        entry,
        expected_preparation_session_revision=None,
    ):

        captured[
            "upsert_calls"
        ] += 1

        captured[
            "entry"
        ] = entry

        captured[
            "revision"
        ] = (
            expected_preparation_session_revision
        )


        return entry


    def forbidden_delete(
        **kwargs,
    ):

        captured[
            "delete_calls"
        ] += 1

        raise AssertionError(
            (
                "Successful controlled Store "
                "execution must not compensate-delete."
            )
        )


    store_module._assert_preparation_authority = (
        fake_authority
    )

    store_module.write_ml_model_binary = (
        fake_write
    )

    store_module.upsert_ml_model_artifact_index_entry = (
        fake_upsert
    )

    store_module.delete_ml_model_binary = (
        forbidden_delete
    )


    try:

        result = (
            register_ml_model_artifact(
                training_contract=
                    contract,
                metrics=
                    metrics,
                train_rows=
                    80,
                test_rows=
                    20,
                model_bytes=
                    model_bytes,
                serialization_format=
                    "pytorch_bundle",
                preparation_session_revision=
                    12,
                created_at_utc=
                    "2026-09-07T00:00:00+00:00",
            )
        )


    finally:

        store_module._assert_preparation_authority = (
            originals[
                "authority"
            ]
        )

        store_module.write_ml_model_binary = (
            originals[
                "write"
            ]
        )

        store_module.upsert_ml_model_artifact_index_entry = (
            originals[
                "upsert"
            ]
        )

        store_module.delete_ml_model_binary = (
            originals[
                "delete"
            ]
        )


    return (
        result,
        captured,
    )


print(
    "=== DATALENS MODEL ARTIFACT STORE CONTRACT FAMILY v0.1 ==="
)

print()


# ============================================================
# SUPERVISED FAMILY MEMBER
# ============================================================


supervised = (
    build_supervised_contract()
)


(
    supervised_artifact,
    supervised_capture,
) = (
    execute_with_controlled_store(
        supervised,
        model_bytes=
            b"supervised-pytorch-bundle",
        metrics={
            "rmse":
                0.25,
        },
    )
)


assert isinstance(
    supervised_artifact.training_contract,
    MLTrainingContract,
)


assert (
    supervised_artifact
    .experiment_provenance
    .training_contract_sha256
    ==
    ml_model_training_contract_sha256(
        supervised
    )
)


assert (
    supervised_capture[
        "authority_calls"
    ]
    ==
    1
)


assert (
    supervised_capture[
        "write_calls"
    ]
    ==
    1
)


assert (
    supervised_capture[
        "upsert_calls"
    ]
    ==
    1
)


print(
    "Supervised Store family member: PASS"
)


# ============================================================
# TARGET-FREE ANOMALY FAMILY MEMBER
# ============================================================


anomaly = (
    build_anomaly_contract()
)


anomaly_metrics = {
    "train_reconstruction_mse":
        0.10,
    "test_reconstruction_mse":
        0.25,
    "anomaly_threshold":
        0.40,
    "test_anomaly_rate":
        0.05,
}


(
    anomaly_artifact,
    anomaly_capture,
) = (
    execute_with_controlled_store(
        anomaly,
        model_bytes=
            b"autoencoder-pytorch-bundle",
        metrics=
            anomaly_metrics,
    )
)


assert isinstance(
    anomaly_artifact.training_contract,
    MLAnomalyTrainingContract,
)


assert (
    anomaly_artifact.training_contract.problem_type
    ==
    "anomaly_detection"
)


assert not hasattr(
    anomaly_artifact.training_contract,
    "target_column"
)


assert (
    anomaly_artifact
    .experiment_provenance
    .training_contract_sha256
    ==
    ml_model_training_contract_sha256(
        anomaly
    )
)


assert (
    anomaly_capture[
        "authority_calls"
    ]
    ==
    1
)


assert (
    anomaly_capture[
        "write_calls"
    ]
    ==
    1
)


assert (
    anomaly_capture[
        "upsert_calls"
    ]
    ==
    1
)


assert (
    anomaly_capture[
        "delete_calls"
    ]
    ==
    0
)


assert (
    anomaly_capture[
        "revision"
    ]
    ==
    12
)


index_entry = (
    validate_ml_model_artifact_index_entry(
        anomaly_capture[
            "entry"
        ]
    )
)


assert (
    index_entry[
        "problem_type"
    ]
    ==
    "anomaly_detection"
)


assert (
    index_entry[
        "target_column"
    ]
    is None
)


assert (
    index_entry[
        "estimator_key"
    ]
    ==
    "tabular_autoencoder"
)


print(
    "Target-free anomaly Store family member: PASS"
)


# ============================================================
# GENERIC PROVENANCE GENERATED BY STORE
# ============================================================


assert (
    anomaly_artifact.experiment_provenance
    is not None
)


assert (
    anomaly_artifact
    .experiment_provenance
    .workflow_id
    ==
    anomaly.workflow_id
)


assert (
    anomaly_artifact
    .experiment_provenance
    .dataset_id
    ==
    anomaly.dataset_id
)


assert (
    anomaly_artifact
    .experiment_provenance
    .preparation_session_revision
    ==
    12
)


assert (
    anomaly_artifact
    .experiment_provenance
    .metrics
    ==
    anomaly_metrics
)


print(
    "Generic Store Experiment Provenance: PASS"
)


# ============================================================
# UNSUPPORTED CONTRACT FAILS BEFORE SIDE EFFECTS
# ============================================================


original_write = (
    store_module
    .write_ml_model_binary
)


write_called = False


def forbidden_write(
    **kwargs,
):

    global write_called

    write_called = True

    raise AssertionError(
        (
            "Unsupported contract must fail "
            "before binary persistence."
        )
    )


store_module.write_ml_model_binary = (
    forbidden_write
)


try:

    try:

        register_ml_model_artifact(
            training_contract={
                "workflow_id":
                    "workflow:unsupported",
                "dataset_id":
                    "dataset:unsupported",
                "problem_type":
                    "clustering",
                "feature_columns":
                    [
                        "x",
                    ],
                "estimator_key":
                    "unknown",
            },
            metrics={
                "score":
                    1.0,
            },
            train_rows=
                8,
            test_rows=
                2,
            model_bytes=
                b"must-not-write",
            serialization_format=
                "pytorch_bundle",
            preparation_session_revision=
                1,
        )

    except (
        ValueError,
        TypeError,
    ):
        pass

    else:

        raise AssertionError(
            (
                "Unsupported Model Lab contract "
                "family was accepted by Store."
            )
        )


finally:

    store_module.write_ml_model_binary = (
        original_write
    )


assert (
    write_called
    is False
)


print(
    "Unsupported Store family fail-closed: PASS"
)


# ============================================================
# SQLITE SCHEMA IS NOT PART OF V3B
# ============================================================


# The successful anomaly test above deliberately replaces only
# the Index persistence call. The record and index payload are
# real; physical SQLite acceptance remains A8-V4.


assert (
    anomaly_capture[
        "upsert_calls"
    ]
    ==
    1
)


print(
    "SQLite persistence intentionally deferred: PASS"
)


# ============================================================
# TORCH-FREE STORE
# ============================================================


assert (
    "torch"
    not in
    sys.modules
)


source = open(
    "app/ml/model_artifact_store.py",
    "r",
    encoding="utf-8",
).read()


for forbidden in (
    "import torch",
    "from torch",
    "torch.",
):

    assert (
        forbidden
        not in
        source
    )


print(
    "Runtime torch-free Artifact Store: PASS"
)


print()

print(
    "PASS - DataLens Model Artifact Store Contract Family v0.1"
)
