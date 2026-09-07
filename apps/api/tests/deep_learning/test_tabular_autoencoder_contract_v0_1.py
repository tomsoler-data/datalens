from __future__ import annotations


import sys


from pydantic import (
    ValidationError,
)


# Runtime import boundary must remain torch-free.
assert (
    "torch"
    not in
    sys.modules
)


from app.deep_learning.autoencoder_contracts import (
    DL_TABULAR_AUTOENCODER_CONTRACT_RULE_VERSION,
    DLTabularAutoencoderHyperparameters,
)


from app.ml.anomaly_contracts import (
    ML_ANOMALY_TRAINING_CONTRACT_RULE_VERSION,
    MLAnomalyTrainingContract,
)


assert (
    "torch"
    not in
    sys.modules
)


print(
    "=== DATALENS TABULAR AUTOENCODER CONTRACT v0.1 ==="
)
print()


# ============================================================
# DEFAULT CONTRACT
# ============================================================


contract = MLAnomalyTrainingContract(
    workflow_id="workflow:0001",
    dataset_id="dataset:0001",
    feature_columns=[
        "amount",
        "age",
        "segment",
    ],
    categorical_feature_columns=[
        "segment",
    ],
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
    contract.numeric_feature_columns
    ==
    [
        "amount",
        "age",
    ]
)

assert (
    contract.categorical_feature_columns
    ==
    [
        "segment",
    ]
)

assert (
    contract.estimator_hyperparameters.kind
    ==
    "tabular_autoencoder"
)

print(
    "Default anomaly contract: PASS"
)


# ============================================================
# NO TARGET AUTHORITY
# ============================================================


assert (
    "target_column"
    not in
    MLAnomalyTrainingContract.model_fields
)

assert (
    "target_column"
    not in
    contract.model_dump(
        mode="json"
    )
)

print(
    "Target-free unsupervised authority: PASS"
)


# ============================================================
# DEFAULT HYPERPARAMETERS
# ============================================================


hyperparameters = (
    contract
    .estimator_hyperparameters
)


assert (
    hyperparameters.hidden_features
    ==
    64
)

assert (
    hyperparameters.latent_features
    ==
    16
)

assert (
    hyperparameters.epochs
    ==
    100
)

assert (
    hyperparameters.batch_size
    ==
    64
)

assert (
    hyperparameters.learning_rate
    ==
    0.001
)

print(
    "Default autoencoder configuration: PASS"
)


# ============================================================
# EXPLICIT HYPERPARAMETERS
# ============================================================


explicit = MLAnomalyTrainingContract(
    workflow_id="workflow:0002",
    dataset_id="dataset:0002",
    feature_columns=[
        "x1",
        "x2",
    ],
    estimator_hyperparameters={
        "kind":
            "tabular_autoencoder",
        "hidden_features":
            32,
        "latent_features":
            8,
        "epochs":
            25,
        "batch_size":
            16,
        "learning_rate":
            0.005,
    },
)


assert (
    explicit
    .estimator_hyperparameters
    .hidden_features
    ==
    32
)

assert (
    explicit
    .estimator_hyperparameters
    .latent_features
    ==
    8
)

print(
    "Explicit autoencoder configuration: PASS"
)


# ============================================================
# ARCHITECTURE GUARDS
# ============================================================


try:

    DLTabularAutoencoderHyperparameters(
        hidden_features=8,
        latent_features=16,
    )

except ValidationError:
    pass

else:
    raise AssertionError(
        (
            "latent_features > hidden_features "
            "must be rejected."
        )
    )


for kwargs in (
    {
        "hidden_features":
            0,
    },
    {
        "latent_features":
            0,
    },
    {
        "epochs":
            0,
    },
    {
        "batch_size":
            0,
    },
    {
        "learning_rate":
            0.0,
    },
):

    try:

        DLTabularAutoencoderHyperparameters(
            **kwargs
        )

    except ValidationError:
        pass

    else:
        raise AssertionError(
            f"Invalid hyperparameters accepted: {kwargs}"
        )


print(
    "Autoencoder architecture guards: PASS"
)


# ============================================================
# FEATURE AUTHORITY GUARDS
# ============================================================


for kwargs in (
    {
        "workflow_id":
            "workflow:bad1",
        "dataset_id":
            "dataset:bad1",
        "feature_columns":
            [],
    },
    {
        "workflow_id":
            "workflow:bad2",
        "dataset_id":
            "dataset:bad2",
        "feature_columns":
            [
                "x",
                "x",
            ],
    },
    {
        "workflow_id":
            "workflow:bad3",
        "dataset_id":
            "dataset:bad3",
        "feature_columns":
            [
                "x",
            ],
        "categorical_feature_columns":
            [
                "missing",
            ],
    },
):

    try:

        MLAnomalyTrainingContract(
            **kwargs
        )

    except ValidationError:
        pass

    else:
        raise AssertionError(
            f"Invalid feature contract accepted: {kwargs}"
        )


print(
    "Feature-role guards: PASS"
)


# ============================================================
# PROBLEM / ESTIMATOR IDENTITY
# ============================================================


for bad_problem in (
    "regression",
    "classification",
):

    try:

        MLAnomalyTrainingContract(
            workflow_id="workflow:bad-problem",
            dataset_id="dataset:bad-problem",
            problem_type=bad_problem,
            feature_columns=[
                "x",
            ],
        )

    except ValidationError:
        pass

    else:
        raise AssertionError(
            (
                "Supervised problem type entered "
                "the anomaly contract."
            )
        )


try:

    MLAnomalyTrainingContract(
        workflow_id="workflow:bad-estimator",
        dataset_id="dataset:bad-estimator",
        estimator_key="tabular_mlp_regressor",
        feature_columns=[
            "x",
        ],
    )

except ValidationError:
    pass

else:
    raise AssertionError(
        "Wrong estimator identity was accepted."
    )


print(
    "Anomaly problem / estimator identity: PASS"
)


# ============================================================
# SERVER-OWNED CONTROLS
# ============================================================


for forbidden_name, forbidden_value in (
    (
        "random_seed",
        123,
    ),
    (
        "device",
        "cuda",
    ),
    (
        "optimizer",
        "adam",
    ),
    (
        "loss",
        "mse",
    ),
    (
        "activation",
        "relu",
    ),
    (
        "num_workers",
        4,
    ),
):

    payload = {
        "workflow_id":
            "workflow:server-owned",
        "dataset_id":
            "dataset:server-owned",
        "feature_columns":
            [
                "x",
            ],
        forbidden_name:
            forbidden_value,
    }

    try:

        MLAnomalyTrainingContract(
            **payload
        )

    except ValidationError:
        pass

    else:
        raise AssertionError(
            (
                "Server-owned control exposed: "
                f"{forbidden_name}"
            )
        )


print(
    "Server-owned execution controls: PASS"
)


# ============================================================
# UNSUPERVISED SPLIT POLICY
# ============================================================


try:

    MLAnomalyTrainingContract(
        workflow_id=
            "workflow:bad-stratify",
        dataset_id=
            "dataset:bad-stratify",
        feature_columns=[
            "x",
        ],
        split={
            "strategy":
                "holdout",
            "stratify":
                True,
        },
    )

except ValidationError:
    pass

else:
    raise AssertionError(
        (
            "Anomaly Training Contract accepted "
            "target-based stratification."
        )
    )


print(
    "Unsupervised split stratification guard: PASS"
)



# ============================================================
# TRAIN-ONLY THRESHOLD QUANTILE POLICY
# ============================================================


default_threshold_contract = (
    MLAnomalyTrainingContract(
        workflow_id=
            "workflow:threshold-default",
        dataset_id=
            "dataset:threshold-default",
        feature_columns=[
            "x",
        ],
    )
)


assert (
    default_threshold_contract.threshold_quantile
    ==
    0.99
)


explicit_threshold_contract = (
    MLAnomalyTrainingContract(
        workflow_id=
            "workflow:threshold-explicit",
        dataset_id=
            "dataset:threshold-explicit",
        feature_columns=[
            "x",
        ],
        threshold_quantile=
            0.95,
    )
)


assert (
    explicit_threshold_contract.threshold_quantile
    ==
    0.95
)


for invalid_quantile in (
    0.0,
    1.0,
    -0.1,
    1.1,
):

    try:

        MLAnomalyTrainingContract(
            workflow_id=
                "workflow:threshold-invalid",
            dataset_id=
                "dataset:threshold-invalid",
            feature_columns=[
                "x",
            ],
            threshold_quantile=
                invalid_quantile,
        )

    except ValidationError:
        pass

    else:

        raise AssertionError(
            (
                "Invalid anomaly threshold quantile "
                f"was accepted: {invalid_quantile}"
            )
        )


print(
    "TRAIN-only threshold quantile policy: PASS"
)



# ============================================================
# FROZEN CONTRACT
# ============================================================


try:

    contract.workflow_id = (
        "workflow:mutated"
    )

except ValidationError:
    pass

else:
    raise AssertionError(
        "Frozen anomaly contract was mutable."
    )


print(
    "Frozen contract: PASS"
)


# ============================================================
# DETERMINISTIC SERIALIZATION
# ============================================================


first_json = contract.model_dump_json()
second_json = contract.model_dump_json()


assert (
    first_json
    ==
    second_json
)


print(
    "Deterministic serialization: PASS"
)


# ============================================================
# TORCH-FREE CONTRACT SURFACE
# ============================================================


assert (
    "torch"
    not in
    sys.modules
)


print(
    "Runtime torch isolation: PASS"
)


# ============================================================
# RULE VERSIONS
# ============================================================


assert (
    DL_TABULAR_AUTOENCODER_CONTRACT_RULE_VERSION
    ==
    "dl_tabular_autoencoder_contract_v0.1"
)

assert (
    ML_ANOMALY_TRAINING_CONTRACT_RULE_VERSION
    ==
    "ml_anomaly_training_contract_v0.1"
)


print(
    "Contract rule versions: PASS"
)


print()
print(
    "PASS - DataLens Tabular Autoencoder Contract v0.1"
)
