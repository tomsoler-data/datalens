from __future__ import annotations


import json
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
    build_ml_experiment_provenance,
    build_ml_model_experiment_provenance,
    canonical_ml_model_training_contract_json,
    canonical_ml_training_contract_json,
    ml_model_training_contract_sha256,
    ml_training_contract_sha256,
)


from app.ml.model_training_contracts import (
    ML_MODEL_TRAINING_CONTRACT_FAMILY_RULE_VERSION,
    validate_ml_model_training_contract,
)


def build_supervised_contract(
) -> MLTrainingContract:

    return (
        MLTrainingContract(
            workflow_id=
                "workflow:family-supervised",
            dataset_id=
                "dataset:family-supervised",
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
    *,
    threshold_quantile: float = 0.99,
) -> MLAnomalyTrainingContract:

    return (
        MLAnomalyTrainingContract(
            workflow_id=
                "workflow:family-anomaly",
            dataset_id=
                "dataset:family-anomaly",
            feature_columns=[
                "x1",
                "x2",
            ],
            threshold_quantile=
                threshold_quantile,
        )
    )


print(
    "=== DATALENS MODEL TRAINING CONTRACT FAMILY v0.1 ==="
)

print()


# ============================================================
# FAMILY VALIDATION
# ============================================================


supervised = (
    build_supervised_contract()
)

anomaly = (
    build_anomaly_contract()
)


assert (
    validate_ml_model_training_contract(
        supervised
    )
    ==
    supervised
)


assert (
    validate_ml_model_training_contract(
        anomaly
    )
    ==
    anomaly
)


assert isinstance(
    validate_ml_model_training_contract(
        supervised.model_dump(
            mode="json"
        )
    ),
    MLTrainingContract,
)


assert isinstance(
    validate_ml_model_training_contract(
        anomaly.model_dump(
            mode="json"
        )
    ),
    MLAnomalyTrainingContract,
)


print(
    "Supervised + anomaly contract family: PASS"
)


# ============================================================
# UNSUPPORTED FAMILY FAIL-CLOSED
# ============================================================


for invalid in (
    {
        "problem_type":
            "clustering",
    },
    {
        "workflow_id":
            "workflow:missing-problem",
    },
    "not-a-contract",
):

    try:

        validate_ml_model_training_contract(
            invalid
        )

    except (
        ValidationError,
        ValueError,
        TypeError,
    ):
        pass

    else:

        raise AssertionError(
            (
                "Unsupported training-contract "
                "family was accepted."
            )
        )


print(
    "Unsupported family fail-closed guard: PASS"
)


# ============================================================
# GENERIC CANONICAL JSON
# ============================================================


supervised_generic_json = (
    canonical_ml_model_training_contract_json(
        supervised
    )
)


anomaly_generic_json = (
    canonical_ml_model_training_contract_json(
        anomaly
    )
)


assert (
    json.loads(
        supervised_generic_json
    )
    ==
    supervised.model_dump(
        mode="json"
    )
)


assert (
    json.loads(
        anomaly_generic_json
    )
    ==
    anomaly.model_dump(
        mode="json"
    )
)


assert (
    canonical_ml_model_training_contract_json(
        anomaly
    )
    ==
    anomaly_generic_json
)


print(
    "Generic deterministic canonical JSON: PASS"
)


# ============================================================
# SUPERVISED HASH BACKWARD COMPATIBILITY
# ============================================================


historical_json = (
    canonical_ml_training_contract_json(
        supervised
    )
)


assert (
    supervised_generic_json
    ==
    historical_json
)


historical_hash = (
    ml_training_contract_sha256(
        supervised
    )
)


generic_hash = (
    ml_model_training_contract_sha256(
        supervised
    )
)


assert (
    historical_hash
    ==
    generic_hash
)


print(
    "Supervised canonical/hash parity: PASS"
)


# ============================================================
# HISTORICAL API REMAINS SUPERVISED
# ============================================================


for historical_function in (
    canonical_ml_training_contract_json,
    ml_training_contract_sha256,
):

    try:

        historical_function(
            anomaly
        )

    except (
        ValidationError,
        ValueError,
        TypeError,
    ):
        pass

    else:

        raise AssertionError(
            (
                "Historical supervised fingerprint "
                "API unexpectedly accepted anomaly."
            )
        )


try:

    build_ml_experiment_provenance(
        training_contract=
            anomaly,
        preparation_session_revision=
            7,
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
            "test_reconstruction_mse":
                0.25,
        },
    )

except (
    ValidationError,
    ValueError,
    TypeError,
):
    pass

else:

    raise AssertionError(
        (
            "Historical supervised provenance "
            "builder unexpectedly accepted anomaly."
        )
    )


print(
    "Historical supervised API boundary: PASS"
)


# ============================================================
# ANOMALY HASH BINDS THRESHOLD POLICY
# ============================================================


first_anomaly = (
    build_anomaly_contract(
        threshold_quantile=
            0.95,
    )
)


second_anomaly = (
    build_anomaly_contract(
        threshold_quantile=
            0.99,
    )
)


first_hash = (
    ml_model_training_contract_sha256(
        first_anomaly
    )
)


second_hash = (
    ml_model_training_contract_sha256(
        second_anomaly
    )
)


assert (
    first_hash
    !=
    second_hash
)


assert (
    len(
        first_hash
    )
    ==
    64
)


assert (
    len(
        second_hash
    )
    ==
    64
)


print(
    "Anomaly policy fingerprint binding: PASS"
)


# ============================================================
# GENERIC EXPERIMENT PROVENANCE
# ============================================================


provenance = (
    build_ml_model_experiment_provenance(
        training_contract=
            first_anomaly,
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
        metrics={
            "train_reconstruction_mse":
                0.10,
            "test_reconstruction_mse":
                0.25,
            "anomaly_threshold":
                0.40,
            "test_anomaly_rate":
                0.05,
        },
    )
)


assert (
    provenance.workflow_id
    ==
    first_anomaly.workflow_id
)


assert (
    provenance.dataset_id
    ==
    first_anomaly.dataset_id
)


assert (
    provenance.preparation_session_revision
    ==
    11
)


assert (
    provenance.training_contract_sha256
    ==
    first_hash
)


assert (
    provenance.train_rows
    ==
    80
)


assert (
    provenance.test_rows
    ==
    20
)


assert (
    provenance.metrics[
        "test_anomaly_rate"
    ]
    ==
    0.05
)


print(
    "Generic anomaly Experiment Provenance: PASS"
)


# ============================================================
# TORCH-FREE LIFECYCLE AUTHORITY
# ============================================================


assert (
    "torch"
    not in
    sys.modules
)


for source_path in (
    "app/ml/model_training_contracts.py",
    "app/ml/experiment_provenance.py",
):

    source = open(
        source_path,
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
    "Runtime torch-free lifecycle boundary: PASS"
)


# ============================================================
# RULE VERSION
# ============================================================


assert (
    ML_MODEL_TRAINING_CONTRACT_FAMILY_RULE_VERSION
    ==
    "ml_model_training_contract_family_v0.1"
)


print(
    "Contract-family rule version: PASS"
)


print()

print(
    "PASS - DataLens Model Training Contract Family v0.1"
)
