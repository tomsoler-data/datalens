from __future__ import annotations


import json
import sys


from app.ml.anomaly_contracts import (
    MLAnomalyTrainingContract,
)


from app.ml.contracts import (
    MLTrainingContract,
)


from app.ml.experiment_provenance import (
    build_ml_model_experiment_provenance,
)


from app.ml.model_artifact_index import (
    MLModelArtifactIndexError,
    _row_to_entry,
    validate_ml_model_artifact_index_entry,
)


from app.ml.model_artifacts import (
    MLModelArtifactRecord,
)


# ============================================================
# HELPERS
# ============================================================


def build_supervised_contract(
) -> MLTrainingContract:

    return (
        MLTrainingContract(
            workflow_id=
                "workflow:index-supervised",
            dataset_id=
                "dataset:index-supervised",
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
                "workflow:index-anomaly",
            dataset_id=
                "dataset:index-anomaly",
            feature_columns=[
                "x1",
                "x2",
            ],
            threshold_quantile=
                0.975,
        )
    )


def build_artifact(
    contract,
    *,
    model_character: str,
    metrics: dict[
        str,
        float,
    ],
) -> MLModelArtifactRecord:

    model_id = (
        "model:"
        +
        model_character
        *
        32
    )


    provenance = (
        build_ml_model_experiment_provenance(
            training_contract=
                contract,
            preparation_session_revision=
                12,
            model_id=
                model_id,
            train_rows=
                80,
            test_rows=
                20,
            metrics=
                metrics,
        )
    )


    return (
        MLModelArtifactRecord(
            model_id=
                model_id,
            workflow_id=
                contract.workflow_id,
            dataset_id=
                contract.dataset_id,
            training_contract=
                contract,
            experiment_provenance=
                provenance,
            metrics=
                metrics,
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
                    model_id
                    +
                    ".ptbundle"
                ),
            model_file_bytes=
                1234,
            model_sha256=
                model_character
                *
                64,
        )
    )


def row_from_validated_entry(
    entry: dict,
) -> dict:

    return {
        "model_id":
            entry[
                "model_id"
            ],
        "workflow_id":
            entry[
                "workflow_id"
            ],
        "dataset_id":
            entry[
                "dataset_id"
            ],
        "problem_type":
            entry[
                "problem_type"
            ],
        "target_column":
            entry[
                "target_column"
            ],
        "estimator_key":
            entry[
                "estimator_key"
            ],
        "experiment_id":
            entry[
                "experiment_id"
            ],
        "experiment_provenance_json":
            (
                json.dumps(
                    entry[
                        "experiment_provenance"
                    ]
                )
                if entry[
                    "experiment_provenance"
                ]
                is not None
                else None
            ),
        "training_contract_json":
            json.dumps(
                entry[
                    "training_contract"
                ]
            ),
        "metrics_json":
            json.dumps(
                entry[
                    "metrics"
                ]
            ),
        "train_rows":
            entry[
                "train_rows"
            ],
        "test_rows":
            entry[
                "test_rows"
            ],
        "created_at_utc":
            entry[
                "created_at_utc"
            ],
        "serialization_format":
            entry[
                "serialization_format"
            ],
        "rule_version":
            entry[
                "rule_version"
            ],
        "model_path":
            entry[
                "model_path"
            ],
        "model_file_bytes":
            entry[
                "model_file_bytes"
            ],
        "model_sha256":
            entry[
                "model_sha256"
            ],
    }


print(
    "=== DATALENS MODEL ARTIFACT INDEX CONTRACT FAMILY v0.1 ==="
)

print()


# ============================================================
# SUPERVISED TARGET PRESERVED
# ============================================================


supervised = (
    build_supervised_contract()
)


supervised_artifact = (
    build_artifact(
        supervised,
        model_character=
            "a",
        metrics={
            "rmse":
                0.25,
        },
    )
)


supervised_entry = (
    validate_ml_model_artifact_index_entry(
        supervised_artifact.model_dump(
            mode="json"
        )
    )
)


assert (
    supervised_entry[
        "problem_type"
    ]
    ==
    "regression"
)


assert (
    supervised_entry[
        "target_column"
    ]
    ==
    "target"
)


assert (
    supervised_entry[
        "estimator_key"
    ]
    ==
    "tabular_mlp_regressor"
)


print(
    "Supervised target-column index semantics: PASS"
)


# ============================================================
# ANOMALY TARGET-FREE INDEX ENTRY
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


anomaly_artifact = (
    build_artifact(
        anomaly,
        model_character=
            "b",
        metrics=
            anomaly_metrics,
    )
)


anomaly_entry = (
    validate_ml_model_artifact_index_entry(
        anomaly_artifact.model_dump(
            mode="json"
        )
    )
)


assert (
    anomaly_entry[
        "problem_type"
    ]
    ==
    "anomaly_detection"
)


assert (
    anomaly_entry[
        "target_column"
    ]
    is None
)


assert (
    anomaly_entry[
        "estimator_key"
    ]
    ==
    "tabular_autoencoder"
)


assert (
    "target_column"
    not in
    anomaly_entry[
        "training_contract"
    ]
)


print(
    "Target-free anomaly index entry: PASS"
)


# ============================================================
# NULLABLE SQLITE ROW ROUND TRIP
# ============================================================


anomaly_row = (
    row_from_validated_entry(
        anomaly_entry
    )
)


assert (
    anomaly_row[
        "target_column"
    ]
    is None
)


restored_entry = (
    _row_to_entry(
        anomaly_row
    )
)


assert (
    restored_entry[
        "target_column"
    ]
    is None
)


assert (
    restored_entry[
        "training_contract"
    ][
        "problem_type"
    ]
    ==
    "anomaly_detection"
)


assert (
    restored_entry[
        "estimator_key"
    ]
    ==
    "tabular_autoencoder"
)


print(
    "Nullable SQLite target round trip: PASS"
)


# ============================================================
# DENORMALIZED TARGET CONSISTENCY
# ============================================================


tampered_row = dict(
    anomaly_row
)


tampered_row[
    "target_column"
] = "invented-target"


try:

    _row_to_entry(
        tampered_row
    )

except MLModelArtifactIndexError:
    pass

else:

    raise AssertionError(
        (
            "Artifact Index accepted a denormalized "
            "target_column inconsistent with the "
            "target-free anomaly contract."
        )
    )


print(
    "Nullable target consistency guard: PASS"
)


# ============================================================
# JSON FAMILY MEMBER PRESERVED
# ============================================================


restored_artifact = (
    MLModelArtifactRecord.model_validate(
        {
            "model_id":
                anomaly_entry[
                    "model_id"
                ],
            "workflow_id":
                anomaly_entry[
                    "workflow_id"
                ],
            "dataset_id":
                anomaly_entry[
                    "dataset_id"
                ],
            "training_contract":
                anomaly_entry[
                    "training_contract"
                ],
            "experiment_provenance":
                anomaly_entry[
                    "experiment_provenance"
                ],
            "metrics":
                anomaly_entry[
                    "metrics"
                ],
            "train_rows":
                anomaly_entry[
                    "train_rows"
                ],
            "test_rows":
                anomaly_entry[
                    "test_rows"
                ],
            "created_at_utc":
                anomaly_entry[
                    "created_at_utc"
                ],
            "serialization_format":
                anomaly_entry[
                    "serialization_format"
                ],
            "rule_version":
                anomaly_entry[
                    "rule_version"
                ],
            "model_path":
                anomaly_entry[
                    "model_path"
                ],
            "model_file_bytes":
                anomaly_entry[
                    "model_file_bytes"
                ],
            "model_sha256":
                anomaly_entry[
                    "model_sha256"
                ],
        }
    )
)


assert isinstance(
    restored_artifact.training_contract,
    MLAnomalyTrainingContract,
)


print(
    "Artifact family member restoration: PASS"
)


# ============================================================
# TORCH-FREE INDEX
# ============================================================


assert (
    "torch"
    not in
    sys.modules
)


source = open(
    "app/ml/model_artifact_index.py",
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
    "Runtime torch-free Artifact Index: PASS"
)


print()

print(
    "PASS - DataLens Model Artifact Index Contract Family v0.1"
)
