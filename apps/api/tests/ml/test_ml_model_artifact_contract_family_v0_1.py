from __future__ import annotations


import sys


from pydantic import (
    ValidationError,
)


from app.ml.anomaly_contracts import (
    MLAnomalyTrainingContract,
)


from app.ml.contracts import (
    MLTrainingContract,
)


from app.ml.experiment_provenance import (
    build_ml_model_experiment_provenance,
    ml_model_training_contract_sha256,
)


from app.ml.model_artifacts import (
    MLModelArtifactRecord,
)


def build_supervised_contract(
) -> MLTrainingContract:

    return (
        MLTrainingContract(
            workflow_id=
                "workflow:artifact-supervised",
            dataset_id=
                "dataset:artifact-supervised",
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
                "workflow:artifact-anomaly",
            dataset_id=
                "dataset:artifact-anomaly",
            feature_columns=[
                "x1",
                "x2",
            ],
            threshold_quantile=
                0.975,
        )
    )


def model_sha(
    character: str,
) -> str:

    return (
        character
        *
        64
    )


print(
    "=== DATALENS MODEL ARTIFACT CONTRACT FAMILY v0.1 ==="
)

print()


# ============================================================
# SUPERVISED ARTIFACT PRESERVED
# ============================================================


supervised = (
    build_supervised_contract()
)


supervised_provenance = (
    build_ml_model_experiment_provenance(
        training_contract=
            supervised,
        preparation_session_revision=
            8,
        model_id=
            "model:"
            +
            "a"
            *
            32,
        train_rows=
            80,
        test_rows=
            20,
        metrics={
            "rmse":
                0.25,
        },
    )
)


supervised_artifact = (
    MLModelArtifactRecord(
        model_id=
            supervised_provenance.model_id,
        workflow_id=
            supervised.workflow_id,
        dataset_id=
            supervised.dataset_id,
        training_contract=
            supervised,
        experiment_provenance=
            supervised_provenance,
        metrics={
            "rmse":
                0.25,
        },
        train_rows=
            80,
        test_rows=
            20,
        created_at_utc=
            "2026-09-07T00:00:00+00:00",
        serialization_format=
            "pytorch_bundle",
        model_path=
            (
                "models/"
                +
                supervised_provenance.model_id
                +
                ".ptbundle"
            ),
        model_file_bytes=
            1234,
        model_sha256=
            model_sha(
                "a"
            ),
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


print(
    "Historical supervised Artifact contract: PASS"
)


# ============================================================
# ANOMALY ARTIFACT
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


anomaly_provenance = (
    build_ml_model_experiment_provenance(
        training_contract=
            anomaly,
        preparation_session_revision=
            11,
        model_id=
            "model:"
            +
            "b"
            *
            32,
        train_rows=
            80,
        test_rows=
            20,
        metrics=
            anomaly_metrics,
    )
)


anomaly_artifact = (
    MLModelArtifactRecord(
        model_id=
            anomaly_provenance.model_id,
        workflow_id=
            anomaly.workflow_id,
        dataset_id=
            anomaly.dataset_id,
        training_contract=
            anomaly,
        experiment_provenance=
            anomaly_provenance,
        metrics=
            anomaly_metrics,
        train_rows=
            80,
        test_rows=
            20,
        created_at_utc=
            "2026-09-07T00:00:00+00:00",
        serialization_format=
            "pytorch_bundle",
        model_path=
            (
                "models/"
                +
                anomaly_provenance.model_id
                +
                ".ptbundle"
            ),
        model_file_bytes=
            4321,
        model_sha256=
            model_sha(
                "b"
            ),
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
    "target_column",
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
    anomaly_artifact.serialization_format
    ==
    "pytorch_bundle"
)


assert (
    anomaly_artifact.model_path.endswith(
        ".ptbundle"
    )
)


print(
    "Target-free anomaly Artifact contract: PASS"
)


# ============================================================
# JSON ROUND TRIP PRESERVES FAMILY MEMBER
# ============================================================


payload = (
    anomaly_artifact.model_dump(
        mode="json"
    )
)


round_trip = (
    MLModelArtifactRecord
    .model_validate(
        payload
    )
)


assert isinstance(
    round_trip.training_contract,
    MLAnomalyTrainingContract,
)


assert (
    round_trip.training_contract
    ==
    anomaly
)


assert (
    round_trip.experiment_provenance
    ==
    anomaly_provenance
)


print(
    "Anomaly Artifact JSON round trip: PASS"
)


# ============================================================
# PROVENANCE CONTRACT HASH MISMATCH FAILS CLOSED
# ============================================================


different_anomaly = (
    MLAnomalyTrainingContract(
        workflow_id=
            anomaly.workflow_id,
        dataset_id=
            anomaly.dataset_id,
        feature_columns=[
            "x1",
            "x2",
        ],
        threshold_quantile=
            0.95,
    )
)


try:

    MLModelArtifactRecord(
        model_id=
            anomaly_provenance.model_id,
        workflow_id=
            anomaly.workflow_id,
        dataset_id=
            anomaly.dataset_id,
        training_contract=
            different_anomaly,
        experiment_provenance=
            anomaly_provenance,
        metrics=
            anomaly_metrics,
        train_rows=
            80,
        test_rows=
            20,
        created_at_utc=
            "2026-09-07T00:00:00+00:00",
        serialization_format=
            "pytorch_bundle",
        model_path=
            (
                "models/"
                +
                anomaly_provenance.model_id
                +
                ".ptbundle"
            ),
        model_file_bytes=
            4321,
        model_sha256=
            model_sha(
                "c"
            ),
    )

except ValidationError:
    pass

else:

    raise AssertionError(
        (
            "Artifact accepted provenance from a "
            "different anomaly threshold policy."
        )
    )


print(
    "Anomaly provenance fingerprint guard: PASS"
)


# ============================================================
# WORKFLOW / DATASET CONSISTENCY PRESERVED
# ============================================================


for field_name, bad_value in (
    (
        "workflow_id",
        "workflow:wrong",
    ),
    (
        "dataset_id",
        "dataset:wrong",
    ),
):

    candidate = {
        "model_id":
            anomaly_provenance.model_id,
        "workflow_id":
            anomaly.workflow_id,
        "dataset_id":
            anomaly.dataset_id,
        "training_contract":
            anomaly,
        "experiment_provenance":
            anomaly_provenance,
        "metrics":
            anomaly_metrics,
        "train_rows":
            80,
        "test_rows":
            20,
        "created_at_utc":
            "2026-09-07T00:00:00+00:00",
        "serialization_format":
            "pytorch_bundle",
        "model_path":
            (
                "models/"
                +
                anomaly_provenance.model_id
                +
                ".ptbundle"
            ),
        "model_file_bytes":
            4321,
        "model_sha256":
            model_sha(
                "d"
            ),
    }


    candidate[
        field_name
    ] = bad_value


    try:

        MLModelArtifactRecord(
            **candidate
        )

    except ValidationError:
        pass

    else:

        raise AssertionError(
            (
                "Artifact accepted mismatched "
                f"{field_name}."
            )
        )


print(
    "Artifact workflow/dataset consistency: PASS"
)


# ============================================================
# UNSUPPORTED CONTRACT FAMILY FAILS CLOSED
# ============================================================


invalid_payload = (
    anomaly_artifact.model_dump(
        mode="json"
    )
)


invalid_payload[
    "training_contract"
][
    "problem_type"
] = "clustering"


try:

    MLModelArtifactRecord.model_validate(
        invalid_payload
    )

except (
    ValidationError,
    ValueError,
):
    pass

else:

    raise AssertionError(
        "Unsupported Artifact contract family was accepted."
    )


print(
    "Unsupported Artifact contract family: PASS"
)


# ============================================================
# PRIVACY / TORCH-FREE METADATA
# ============================================================


assert (
    "torch"
    not in
    sys.modules
)


for forbidden_field in (
    "raw_rows",
    "predictions",
    "reconstruction_errors",
    "anomaly_flags",
    "model_bytes",
):

    assert (
        forbidden_field
        not in
        MLModelArtifactRecord.model_fields
    )


source = open(
    "app/ml/model_artifacts.py",
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
    "Privacy-minimal / torch-free Artifact metadata: PASS"
)


print()

print(
    "PASS - DataLens Model Artifact Contract Family v0.1"
)
