from __future__ import annotations


import json
import sys


from app.ml.anomaly_contracts import (
    MLAnomalyTrainingContract,
)


from app.ml.experiment_provenance import (
    ml_model_training_contract_sha256,
)


from app.ml.model_artifact_data_plane import (
    ml_model_artifact_data_root,
)


from app.ml.model_artifact_store import (
    MLModelArtifactWorkflowMismatchError,
    get_ml_model_artifact,
    list_ml_model_artifacts,
    load_ml_model_artifact_binary,
    register_ml_model_artifact,
)


from app.persistence.sqlite_database import (
    SQLITE_SCHEMA_VERSION,
    sqlite_connection,
)


from tests.ml.test_ml_model_artifact_store_v0_1 import (
    isolated_environment,
    seed_preparation_authority,
)


# ============================================================
# FIXTURE
# ============================================================


WORKFLOW_ID = (
    "prep:autoencoder-lifecycle"
)


DATASET_ID = (
    "dataset:autoencoder-validated"
)


PREPARATION_REVISION = (
    0
)


OPAQUE_ARTIFACT_BYTES = (
    b"DATALENS-A8-AUTOENCODER-"
    b"LIFECYCLE-OPAQUE-PAYLOAD-v0.1"
)


def anomaly_contract(
) -> MLAnomalyTrainingContract:

    return (
        MLAnomalyTrainingContract(
            workflow_id=
                WORKFLOW_ID,
            dataset_id=
                DATASET_ID,
            feature_columns=[
                "amount",
                "frequency",
                "recency",
            ],
            threshold_quantile=
                0.975,
        )
    )


ANOMALY_METRICS = {
    "train_reconstruction_mse":
        0.10,

    "test_reconstruction_mse":
        0.24,

    "anomaly_threshold":
        0.42,

    "train_anomaly_rate":
        0.025,

    "test_anomaly_rate":
        0.05,
}


print(
    "=== DATALENS ANOMALY ARTIFACT LIFECYCLE E2E v0.1 ==="
)

print()


# ============================================================
# 1. CONTRACT
# ============================================================


contract = (
    anomaly_contract()
)


assert (
    contract.problem_type
    ==
    "anomaly_detection"
)


assert (
    contract.estimator_key
    ==
    "tabular_autoencoder"
)


assert (
    contract.threshold_quantile
    ==
    0.975
)


assert not hasattr(
    contract,
    "target_column",
)


print(
    "Target-free anomaly contract: PASS"
)


# ============================================================
# 2. REAL SERVER-OWNED ARTIFACT LIFECYCLE
# ============================================================


with isolated_environment() as (
    _,
    model_store,
):

    seed_preparation_authority(
        workflow_id=
            WORKFLOW_ID,
        dataset_id=
            DATASET_ID,
    )


    record = (
        register_ml_model_artifact(
            training_contract=
                contract,
            metrics=
                ANOMALY_METRICS,
            train_rows=
                80,
            test_rows=
                20,
            model_bytes=
                OPAQUE_ARTIFACT_BYTES,
            serialization_format=
                "pytorch_bundle",
            preparation_session_revision=
                PREPARATION_REVISION,
            created_at_utc=
                "2026-09-07T10:00:00+00:00",
        )
    )


    # --------------------------------------------------------
    # SERVER-OWNED IDENTITY / FORMAT
    # --------------------------------------------------------


    assert (
        record.model_id
        .startswith(
            "model:"
        )
    )


    assert (
        record.serialization_format
        ==
        "pytorch_bundle"
    )


    assert (
        record.model_path
        .endswith(
            ".ptbundle"
        )
    )


    assert (
        record.workflow_id
        ==
        WORKFLOW_ID
    )


    assert (
        record.dataset_id
        ==
        DATASET_ID
    )


    print(
        "Server-owned anomaly Artifact identity: PASS"
    )


    # --------------------------------------------------------
    # GENERIC CONTRACT FAMILY RESTORATION
    # --------------------------------------------------------


    assert isinstance(
        record.training_contract,
        MLAnomalyTrainingContract,
    )


    assert (
        record.training_contract.problem_type
        ==
        "anomaly_detection"
    )


    assert not hasattr(
        record.training_contract,
        "target_column",
    )


    assert (
        record.training_contract.threshold_quantile
        ==
        0.975
    )


    print(
        "Anomaly contract family in Artifact: PASS"
    )


    # --------------------------------------------------------
    # GENERIC EXPERIMENT PROVENANCE
    # --------------------------------------------------------


    provenance = (
        record.experiment_provenance
    )


    assert (
        provenance
        is not None
    )


    assert (
        provenance.workflow_id
        ==
        WORKFLOW_ID
    )


    assert (
        provenance.dataset_id
        ==
        DATASET_ID
    )


    assert (
        provenance.preparation_session_revision
        ==
        PREPARATION_REVISION
    )


    assert (
        provenance.training_contract_sha256
        ==
        ml_model_training_contract_sha256(
            contract
        )
    )


    assert (
        provenance.metrics
        ==
        ANOMALY_METRICS
    )


    print(
        "Generic anomaly Experiment Provenance: PASS"
    )


    # --------------------------------------------------------
    # FILESYSTEM DATA PLANE
    # --------------------------------------------------------


    binary_path = (
        ml_model_artifact_data_root(
            model_store
        )
        /
        record.model_path
    )


    assert (
        binary_path.is_file()
    )


    assert (
        binary_path.read_bytes()
        ==
        OPAQUE_ARTIFACT_BYTES
    )


    print(
        "Server-owned binary persistence: PASS"
    )


    # --------------------------------------------------------
    # STORE RESTORE
    # --------------------------------------------------------


    restored = (
        get_ml_model_artifact(
            model_id=
                record.model_id,
            workflow_id=
                WORKFLOW_ID,
        )
    )


    assert (
        restored
        ==
        record
    )


    assert isinstance(
        restored.training_contract,
        MLAnomalyTrainingContract,
    )


    assert not hasattr(
        restored.training_contract,
        "target_column",
    )


    print(
        "Artifact metadata restore: PASS"
    )


    # --------------------------------------------------------
    # VERIFIED BINARY RESTORE
    # --------------------------------------------------------


    restored_binary = (
        load_ml_model_artifact_binary(
            model_id=
                record.model_id,
            workflow_id=
                WORKFLOW_ID,
        )
    )


    assert (
        restored_binary
        ==
        OPAQUE_ARTIFACT_BYTES
    )


    print(
        "Verified Artifact binary restore: PASS"
    )


    # --------------------------------------------------------
    # WORKFLOW LIST
    # --------------------------------------------------------


    listed = (
        list_ml_model_artifacts(
            workflow_id=
                WORKFLOW_ID
        )
    )


    assert (
        len(
            listed
        )
        ==
        1
    )


    assert (
        listed[
            0
        ]
        ==
        record
    )


    print(
        "Workflow Artifact listing: PASS"
    )


    # --------------------------------------------------------
    # CROSS-WORKFLOW READ GUARD
    # --------------------------------------------------------


    try:

        get_ml_model_artifact(
            model_id=
                record.model_id,
            workflow_id=
                "prep:other-workflow",
        )

    except MLModelArtifactWorkflowMismatchError:
        pass

    else:

        raise AssertionError(
            (
                "Cross-workflow anomaly Artifact "
                "read must fail closed."
            )
        )


    print(
        "Cross-workflow Artifact guard: PASS"
    )


    # --------------------------------------------------------
    # REAL SQLITE v15 PHYSICAL ROW
    # --------------------------------------------------------


    assert (
        SQLITE_SCHEMA_VERSION
        ==
        15
    )


    with sqlite_connection(
        write=False
    ) as connection:

        row = (
            connection.execute(
                """
                SELECT
                    problem_type,
                    target_column,
                    estimator_key,
                    serialization_format,
                    training_contract_json,
                    metrics_json,
                    experiment_id,
                    experiment_provenance_json

                FROM ml_model_artifacts

                WHERE
                    model_id = ?
                """,
                (
                    record.model_id,
                ),
            )
            .fetchone()
        )


        assert (
            row
            is not None
        )


        assert (
            row[
                "problem_type"
            ]
            ==
            "anomaly_detection"
        )


        assert (
            row[
                "target_column"
            ]
            is None
        )


        assert (
            row[
                "estimator_key"
            ]
            ==
            "tabular_autoencoder"
        )


        assert (
            row[
                "serialization_format"
            ]
            ==
            "pytorch_bundle"
        )


        persisted_contract = (
            json.loads(
                row[
                    "training_contract_json"
                ]
            )
        )


        assert (
            "target_column"
            not in
            persisted_contract
        )


        assert (
            persisted_contract[
                "threshold_quantile"
            ]
            ==
            0.975
        )


        persisted_metrics = (
            json.loads(
                row[
                    "metrics_json"
                ]
            )
        )


        assert (
            persisted_metrics
            ==
            ANOMALY_METRICS
        )


        assert (
            row[
                "experiment_id"
            ]
            ==
            provenance.experiment_id
        )


        persisted_provenance = (
            json.loads(
                row[
                    "experiment_provenance_json"
                ]
            )
        )


        assert (
            persisted_provenance[
                "training_contract_sha256"
            ]
            ==
            provenance.training_contract_sha256
        )


        assert (
            persisted_provenance[
                "preparation_session_revision"
            ]
            ==
            PREPARATION_REVISION
        )


        assert (
            connection.execute(
                "PRAGMA foreign_key_check"
            )
            .fetchall()
            ==
            []
        )


    print(
        "SQLite v15 anomaly Artifact row: PASS"
    )


# ============================================================
# 3. RUNTIME TORCH-FREE LIFECYCLE BOUNDARY
# ============================================================


assert (
    "torch"
    not in
    sys.modules
)


print(
    "Runtime torch-free Artifact lifecycle: PASS"
)


print()

print(
    "PASS - DataLens Anomaly Artifact Lifecycle E2E v0.1"
)
