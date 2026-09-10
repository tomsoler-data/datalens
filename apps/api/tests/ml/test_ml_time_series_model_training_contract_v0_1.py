from __future__ import annotations


import ast
from pathlib import Path


from pydantic import (
    ValidationError,
)


from app.deep_learning.time_series_contracts import (
    DLTimeSeriesLSTMRegressorHyperparameters,
    DLTimeSeriesMLPRegressorHyperparameters,
    DLTimeSeriesRNNRegressorHyperparameters,
)


from app.ml.contracts import (
    MLTimeHoldoutSplitContract,
)


from app.ml.experiment_provenance import (
    build_ml_model_experiment_provenance,
    canonical_ml_model_training_contract_json,
    ml_model_training_contract_sha256,
)


from app.ml.model_artifact_index import (
    validate_ml_model_artifact_index_entry,
)


from app.ml.model_artifacts import (
    MLModelArtifactRecord,
)


from app.ml.model_training_contracts import (
    validate_ml_model_training_contract,
)


from app.ml.time_series_contracts import (
    MLTimeSeriesForecastingContract,
)


from app.ml.time_series_model_training_contracts import (
    ML_TIME_SERIES_MODEL_TRAINING_CONTRACT_RULE_VERSION,
    MLTimeSeriesModelTrainingContract,
)


# ============================================================
# HELPERS
# ============================================================


def require_validation_error(
    callback,
    *,
    label: str,
) -> None:

    try:

        callback()

    except (
        ValidationError,
        ValueError,
    ):

        return


    raise AssertionError(
        (
            "Expected training-contract failure: "
            f"{label}"
        )
    )


# ============================================================
# 1. MODEL-NEUTRAL TASK AUTHORITY
# ============================================================


task = (
    MLTimeSeriesForecastingContract(
        workflow_id=
            "workflow:a11-v1",

        dataset_id=
            "dataset:a11-v1",

        target_column=
            "revenue",

        lookback=
            12,

        split=
            MLTimeHoldoutSplitContract(
                time_column=
                    "order_month",

                test_size=
                    0.20,
            ),
    )
)


assert (
    task.task_family
    ==
    "time_series_forecasting"
)


assert (
    task.problem_type
    ==
    "regression"
)


assert not hasattr(
    task,
    "estimator_key",
)


assert not hasattr(
    task,
    "estimator_hyperparameters",
)


print(
    "[PASS] A1 forecasting task remains model-neutral"
)


# ============================================================
# 2. THREE PER-MODEL TRAINING CONTRACTS
# ============================================================


mlp_contract = (
    MLTimeSeriesModelTrainingContract(
        task_contract=
            task,

        estimator_hyperparameters=
            DLTimeSeriesMLPRegressorHyperparameters(
                hidden_features=
                    16,

                epochs=
                    20,

                batch_size=
                    8,

                learning_rate=
                    0.01,
            ),
    )
)


rnn_contract = (
    MLTimeSeriesModelTrainingContract(
        task_contract=
            task,

        estimator_hyperparameters=
            DLTimeSeriesRNNRegressorHyperparameters(
                hidden_size=
                    16,

                epochs=
                    20,

                batch_size=
                    8,

                learning_rate=
                    0.01,
            ),
    )
)


lstm_contract = (
    MLTimeSeriesModelTrainingContract(
        task_contract=
            task,

        estimator_hyperparameters=
            DLTimeSeriesLSTMRegressorHyperparameters(
                hidden_size=
                    16,

                epochs=
                    20,

                batch_size=
                    8,

                learning_rate=
                    0.01,
            ),
    )
)


assert (
    mlp_contract.estimator_key
    ==
    "time_series_mlp_regressor"
)


assert (
    rnn_contract.estimator_key
    ==
    "time_series_rnn_regressor"
)


assert (
    lstm_contract.estimator_key
    ==
    "time_series_lstm_regressor"
)


assert (
    mlp_contract.rule_version
    ==
    ML_TIME_SERIES_MODEL_TRAINING_CONTRACT_RULE_VERSION
)


print(
    "[PASS] task plus estimator produces explicit per-model training identity"
)


# ============================================================
# 3. GENERIC MODEL LAB SURFACE
# ============================================================


for contract in (
    mlp_contract,
    rnn_contract,
    lstm_contract,
):

    assert (
        contract.workflow_id
        ==
        task.workflow_id
    )

    assert (
        contract.dataset_id
        ==
        task.dataset_id
    )

    assert (
        contract.task_family
        ==
        "time_series_forecasting"
    )

    assert (
        contract.problem_type
        ==
        "regression"
    )

    assert (
        contract.target_column
        ==
        "revenue"
    )

    assert (
        contract.time_column
        ==
        "order_month"
    )

    assert (
        contract.lookback
        ==
        12
    )

    assert (
        contract.forecast_horizon
        ==
        1
    )

    assert (
        contract.split
        ==
        task.split
    )

    assert (
        contract.effective_estimator_hyperparameters
        ==
        contract.estimator_hyperparameters
    )


print(
    "[PASS] forecasting training contract exposes generic Model Lab authority"
)


# ============================================================
# 4. DIRECT GENERIC FAMILY VALIDATION
# ============================================================


for contract in (
    mlp_contract,
    rnn_contract,
    lstm_contract,
):

    generic = (
        validate_ml_model_training_contract(
            contract
        )
    )


    assert isinstance(
        generic,
        MLTimeSeriesModelTrainingContract,
    )


    assert (
        generic
        ==
        contract
    )


print(
    "[PASS] generic MLModelTrainingContract family accepts forecasting objects"
)


# ============================================================
# 5. JSON ROUND-TRIP THROUGH GENERIC FAMILY
# ============================================================


for contract in (
    mlp_contract,
    rnn_contract,
    lstm_contract,
):

    payload = (
        contract.model_dump(
            mode="json"
        )
    )


    assert set(
        payload
    ) == {
        "task_contract",
        "estimator_hyperparameters",
        "rule_version",
    }


    generic = (
        validate_ml_model_training_contract(
            payload
        )
    )


    assert isinstance(
        generic,
        MLTimeSeriesModelTrainingContract,
    )


    assert (
        generic
        ==
        contract
    )


print(
    "[PASS] persisted forecasting contracts reconstruct from JSON deterministically"
)


# ============================================================
# 6. PROVENANCE CANONICALIZATION
# ============================================================


mlp_json_1 = (
    canonical_ml_model_training_contract_json(
        mlp_contract
    )
)


mlp_json_2 = (
    canonical_ml_model_training_contract_json(
        mlp_contract.model_dump(
            mode="json"
        )
    )
)


assert (
    mlp_json_1
    ==
    mlp_json_2
)


mlp_sha_1 = (
    ml_model_training_contract_sha256(
        mlp_contract
    )
)


mlp_sha_2 = (
    ml_model_training_contract_sha256(
        mlp_contract.model_dump(
            mode="json"
        )
    )
)


assert (
    mlp_sha_1
    ==
    mlp_sha_2
)


assert (
    len(
        mlp_sha_1
    )
    ==
    64
)


assert (
    len(
        {
            ml_model_training_contract_sha256(
                contract
            )
            for contract in (
                mlp_contract,
                rnn_contract,
                lstm_contract,
            )
        }
    )
    ==
    3
)


print(
    "[PASS] generic provenance fingerprint now binds exact forecasting estimator"
)


# ============================================================
# 7. GENERIC PROVENANCE BUILDER
# ============================================================


metrics = {
    "mae":
        1.25,

    "mse":
        2.25,

    "rmse":
        1.5,

    "mean_error":
        -0.25,
}


provenance = (
    build_ml_model_experiment_provenance(
        training_contract=
            mlp_contract,

        preparation_session_revision=
            7,

        model_id=
            "model:a11-v1",

        train_rows=
            20,

        test_rows=
            5,

        metrics=
            metrics,
    )
)


assert (
    provenance.workflow_id
    ==
    task.workflow_id
)


assert (
    provenance.dataset_id
    ==
    task.dataset_id
)


assert (
    provenance.training_contract_sha256
    ==
    mlp_sha_1
)


assert (
    provenance.metrics
    ==
    metrics
)


print(
    "[PASS] generic Experiment Provenance accepts forecasting contract"
)


# ============================================================
# 8. MODEL ARTIFACT METADATA CONTRACT
# ============================================================


artifact = (
    MLModelArtifactRecord(
        model_id=
            "model:a11-metadata",

        workflow_id=
            task.workflow_id,

        dataset_id=
            task.dataset_id,

        training_contract=
            mlp_contract,

        experiment_provenance=
            None,

        metrics=
            metrics,

        train_rows=
            20,

        test_rows=
            5,

        created_at_utc=
            "2026-09-09T12:00:00+00:00",

        serialization_format=
            "pytorch_bundle",

        model_path=
            "models/model-a11-metadata.ptbundle",

        model_file_bytes=
            123,

        model_sha256=
            (
                "a"
                *
                64
            ),
    )
)


assert isinstance(
    artifact.training_contract,
    MLTimeSeriesModelTrainingContract,
)


assert (
    artifact.training_contract.estimator_key
    ==
    "time_series_mlp_regressor"
)


print(
    "[PASS] generic Model Artifact metadata accepts forecasting training contract"
)


# ============================================================
# 9. ARTIFACT INDEX DERIVED AUTHORITY
# ============================================================


index_entry = (
    validate_ml_model_artifact_index_entry(
        artifact.model_dump(
            mode="json"
        )
    )
)


assert (
    index_entry[
        "problem_type"
    ]
    ==
    "regression"
)


assert (
    index_entry[
        "target_column"
    ]
    ==
    "revenue"
)


assert (
    index_entry[
        "estimator_key"
    ]
    ==
    "time_series_mlp_regressor"
)


assert isinstance(
    index_entry[
        "training_contract"
    ],
    dict,
)


assert (
    index_entry[
        "training_contract"
    ][
        "task_contract"
    ][
        "task_family"
    ]
    ==
    "time_series_forecasting"
)


print(
    "[PASS] existing Artifact index derives forecasting search metadata"
)


# ============================================================
# 10. UNKNOWN ESTIMATOR FAILS CLOSED
# ============================================================


require_validation_error(
    lambda:
        MLTimeSeriesModelTrainingContract(
            task_contract=
                task,

            estimator_hyperparameters={
                "kind":
                    "time_series_gru_regressor",

                "hidden_size":
                    16,

                "epochs":
                    20,

                "batch_size":
                    8,

                "learning_rate":
                    0.01,
            },
        ),
    label=
        "unsupported GRU estimator",
)


print(
    "[PASS] unsupported forecasting estimator family fails closed"
)


# ============================================================
# 11. EXECUTION STATE CANNOT ENTER CONTRACT
# ============================================================


for forbidden_field in (
    "device",
    "random_seed",
    "optimizer",
    "loss",
    "model_state",
    "predictions",
    "raw_rows",
):

    payload = {
        "task_contract":
            task.model_dump(
                mode="json"
            ),

        "estimator_hyperparameters":
            mlp_contract
            .estimator_hyperparameters
            .model_dump(
                mode="json"
            ),

        forbidden_field:
            "forbidden",
    }


    require_validation_error(
        lambda payload=payload:
            MLTimeSeriesModelTrainingContract
            .model_validate(
                payload
            ),
        label=
            forbidden_field,
    )


print(
    "[PASS] execution and learned state remain outside training contract"
)


# ============================================================
# 12. TORCH-FREE STATIC BOUNDARY
# ============================================================


for path in (
    Path(
        "app/ml/"
        "time_series_model_training_contracts.py"
    ),
    Path(
        "app/ml/"
        "model_training_contracts.py"
    ),
):

    source = path.read_text(
        encoding="utf-8"
    )


    tree = ast.parse(
        source
    )


    imported_modules = []


    for node in ast.walk(
        tree
    ):

        if isinstance(
            node,
            ast.Import,
        ):

            imported_modules.extend(
                alias.name
                for alias in node.names
            )


        elif isinstance(
            node,
            ast.ImportFrom,
        ):

            if node.module:

                imported_modules.append(
                    node.module
                )


    assert not any(
        module_name == "torch"
        or module_name.startswith(
            "torch."
        )
        for module_name in imported_modules
    )


print(
    "[PASS] generic forecasting training-contract bridge is torch-free"
)


print()
print("=" * 80)
print("DL-5-A11-V1 FINAL VERDICT")
print("=" * 80)
print()

print("Forecast task remains model-neutral          PASS")
print("Per-model training identity                  PASS")
print("MLP training contract                       PASS")
print("RNN training contract                       PASS")
print("LSTM training contract                      PASS")
print("Generic ModelTrainingContract integration    PASS")
print("Deterministic JSON round-trip                PASS")
print("Generic provenance fingerprint               PASS")
print("Generic Experiment Provenance                PASS")
print("Generic Artifact metadata                    PASS")
print("Artifact index estimator derivation          PASS")
print("Execution-state isolation                    PASS")
print("Torch-free runtime boundary                  PASS")

print()
print(
    "DL-5-A11-V1 - FORECAST MODEL TRAINING CONTRACT BRIDGE: PASS"
)
