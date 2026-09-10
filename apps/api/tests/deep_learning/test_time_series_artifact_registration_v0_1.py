from __future__ import annotations


import ast
from pathlib import Path


import numpy as np
import pandas as pd


from app.deep_learning.time_series_artifact_registration import (
    DL_TIME_SERIES_ARTIFACT_REGISTRATION_RULE_VERSION,
    DLTimeSeriesArtifactRegistrationContractError,
    register_time_series_neural_execution_artifact,
)


from app.deep_learning.time_series_bundle import (
    deserialize_trusted_time_series_neural_bundle,
)


from app.deep_learning.time_series_bundle_inference import (
    predict_trusted_time_series_bundle,
)


from app.deep_learning.time_series_contracts import (
    DLTimeSeriesLSTMRegressorHyperparameters,
    DLTimeSeriesMLPRegressorHyperparameters,
    DLTimeSeriesRNNRegressorHyperparameters,
)


from app.deep_learning.time_series_lstm_executor import (
    execute_time_series_lstm,
)


from app.deep_learning.time_series_mlp_executor import (
    execute_time_series_mlp,
)


from app.deep_learning.time_series_rnn_executor import (
    execute_time_series_rnn,
)


from app.ml.contracts import (
    MLTimeHoldoutSplitContract,
)


from app.ml.experiment_provenance import (
    ml_model_training_contract_sha256,
)


from app.ml.model_artifact_store import (
    list_ml_model_artifacts,
    load_ml_model_artifact_binary,
)


from app.ml.time_series_contracts import (
    MLTimeSeriesForecastingContract,
)


from app.ml.time_series_holdout import (
    resolve_time_series_holdout,
)


from app.ml.time_series_model_training_contracts import (
    MLTimeSeriesModelTrainingContract,
)


from app.ml.time_series_training_input import (
    validate_and_extract_time_series,
)


from app.ml.time_series_windows import (
    build_time_series_supervised_windows,
)


from tests.ml.test_ml_model_artifact_store_v0_1 import (
    isolated_environment,
    seed_preparation_authority,
)


# ============================================================
# HELPERS
# ============================================================


def require_contract_error(
    callback,
    *,
    label: str,
) -> None:

    try:

        callback()

    except DLTimeSeriesArtifactRegistrationContractError:

        return


    raise AssertionError(
        (
            "Expected Artifact registration failure: "
            f"{label}"
        )
    )


# ============================================================
# 1. CONTROLLED FORECASTING POPULATION
# ============================================================


WORKFLOW_ID = (
    "prep:forecast-artifact-lifecycle"
)


DATASET_ID = (
    "dataset:forecast-artifact-validated"
)


PREPARATION_REVISION = (
    0
)


task = (
    MLTimeSeriesForecastingContract(
        workflow_id=
            WORKFLOW_ID,

        dataset_id=
            DATASET_ID,

        target_column=
            "revenue",

        lookback=
            4,

        split=
            MLTimeHoldoutSplitContract(
                time_column=
                    "order_date",

                test_size=
                    0.25,
            ),
    )
)


frame = pd.DataFrame(
    {
        "order_date":
            pd.date_range(
                "2026-01-01",
                periods=40,
                freq="D",
            ),

        "revenue":
            [
                float(
                    50
                    +
                    index
                )
                for index
                in range(
                    40
                )
            ],
    }
)


series = (
    validate_and_extract_time_series(
        dataframe=
            frame,

        contract=
            task,
    )
)


partition = (
    resolve_time_series_holdout(
        series=
            series,

        contract=
            task,
    )
)


windows = (
    build_time_series_supervised_windows(
        series=
            series,

        partition=
            partition,

        contract=
            task,
    )
)


assert (
    windows.train.sample_count
    ==
    26
)


assert (
    windows.test.sample_count
    ==
    10
)


print(
    "[PASS] exact A4 TRAIN/TEST sample populations own Artifact lifecycle"
)


# ============================================================
# 2. TRAIN EACH NEURAL FAMILY ONCE
# ============================================================


mlp_parameters = (
    DLTimeSeriesMLPRegressorHyperparameters(
        hidden_features=
            8,

        epochs=
            2,

        batch_size=
            16,

        learning_rate=
            0.02,
    )
)


rnn_parameters = (
    DLTimeSeriesRNNRegressorHyperparameters(
        hidden_size=
            8,

        epochs=
            2,

        batch_size=
            16,

        learning_rate=
            0.02,
    )
)


lstm_parameters = (
    DLTimeSeriesLSTMRegressorHyperparameters(
        hidden_size=
            8,

        epochs=
            2,

        batch_size=
            16,

        learning_rate=
            0.02,
    )
)


mlp_execution = (
    execute_time_series_mlp(
        windows=
            windows,

        hyperparameters=
            mlp_parameters,

        execution_device=
            "cpu",
    )
)


rnn_execution = (
    execute_time_series_rnn(
        windows=
            windows,

        hyperparameters=
            rnn_parameters,

        execution_device=
            "cpu",
    )
)


lstm_execution = (
    execute_time_series_lstm(
        windows=
            windows,

        hyperparameters=
            lstm_parameters,

        execution_device=
            "cpu",
    )
)


print(
    "[PASS] MLP/RNN/LSTM trained once before Artifact registration"
)


# ============================================================
# 3. PER-MODEL TRAINING CONTRACTS
# ============================================================


mlp_contract = (
    MLTimeSeriesModelTrainingContract(
        task_contract=
            task,

        estimator_hyperparameters=
            mlp_parameters,
    )
)


rnn_contract = (
    MLTimeSeriesModelTrainingContract(
        task_contract=
            task,

        estimator_hyperparameters=
            rnn_parameters,
    )
)


lstm_contract = (
    MLTimeSeriesModelTrainingContract(
        task_contract=
            task,

        estimator_hyperparameters=
            lstm_parameters,
    )
)


cases = (
    (
        mlp_contract,
        mlp_execution,
    ),
    (
        rnn_contract,
        rnn_execution,
    ),
    (
        lstm_contract,
        lstm_execution,
    ),
)


# ============================================================
# 4. REAL GENERIC ARTIFACT STORE LIFECYCLE
# ============================================================


with isolated_environment() as (
    _,
    _,
):

    seed_preparation_authority(
        workflow_id=
            WORKFLOW_ID,

        dataset_id=
            DATASET_ID,
    )


    artifacts = []


    for (
        training_contract,
        execution,
    ) in cases:

        artifact = (
            register_time_series_neural_execution_artifact(
                windows=
                    windows,

                training_contract=
                    training_contract,

                execution=
                    execution,

                preparation_session_revision=
                    PREPARATION_REVISION,
            )
        )


        artifacts.append(
            artifact
        )


        assert (
            artifact.workflow_id
            ==
            WORKFLOW_ID
        )


        assert (
            artifact.dataset_id
            ==
            DATASET_ID
        )


        assert (
            artifact.training_contract
            ==
            training_contract
        )


        assert (
            artifact.training_contract.estimator_key
            ==
            execution.estimator_key
        )


        assert (
            artifact.serialization_format
            ==
            "pytorch_bundle"
        )


        assert (
            artifact.model_path
            .endswith(
                ".ptbundle"
            )
        )


        assert (
            artifact.train_rows
            ==
            windows.train.sample_count
        )


        assert (
            artifact.test_rows
            ==
            windows.test.sample_count
        )


        assert (
            artifact.experiment_provenance
            is not None
        )


        assert (
            artifact.experiment_provenance
            .preparation_session_revision
            ==
            PREPARATION_REVISION
        )


        assert (
            artifact.experiment_provenance
            .training_contract_sha256
            ==
            ml_model_training_contract_sha256(
                training_contract
            )
        )


        assert (
            artifact.metrics
            ==
            {
                "mae":
                    float(
                        execution.metrics.mae
                    ),

                "mse":
                    float(
                        execution.metrics.mse
                    ),

                "rmse":
                    float(
                        execution.metrics.rmse
                    ),

                "mean_error":
                    float(
                        execution.metrics.mean_error
                    ),
            }
        )


    assert (
        len(
            {
                artifact.model_id
                for artifact
                in artifacts
            }
        )
        ==
        3
    )


    print(
        "[PASS] generic Artifact Store persists all three forecasting families"
    )


    # ========================================================
    # 5. REAL STORED BYTES REMAIN EXECUTABLE
    # ========================================================


    for (
        artifact,
        (
            training_contract,
            execution,
        ),
    ) in zip(
        artifacts,
        cases,
    ):

        stored_bytes = (
            load_ml_model_artifact_binary(
                model_id=
                    artifact.model_id,

                workflow_id=
                    WORKFLOW_ID,
            )
        )


        assert isinstance(
            stored_bytes,
            bytes,
        )


        assert (
            len(
                stored_bytes
            )
            ==
            artifact.model_file_bytes
        )


        restored = (
            deserialize_trusted_time_series_neural_bundle(
                trusted_bundle_bytes=
                    stored_bytes,

                training_contract=
                    artifact.training_contract,
            )
        )


        inference = (
            predict_trusted_time_series_bundle(
                components=
                    restored,

                inputs=
                    windows.test.inputs,
            )
        )


        np.testing.assert_array_equal(
            inference.predictions,
            execution.predictions,
        )


    print(
        "[PASS] actual Artifact Store bytes preserve trusted inference parity"
    )


    # ========================================================
    # 6. INDEX / LIST AUTHORITY
    # ========================================================


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
        3
    )


    assert (
        {
            item.training_contract.estimator_key
            for item
            in listed
        }
        ==
        {
            "time_series_mlp_regressor",
            "time_series_rnn_regressor",
            "time_series_lstm_regressor",
        }
    )


    assert all(
        item.serialization_format
        ==
        "pytorch_bundle"
        for item
        in listed
    )


    print(
        "[PASS] generic Artifact index lists forecasting estimator identities"
    )


    # ========================================================
    # 7. TRAIN SAMPLE COUNT IS NOT SCALER OBSERVATION COUNT
    # ========================================================


    assert (
        mlp_execution.standardizer
        .fitted_observation_count
        ==
        30
    )


    assert (
        artifacts[
            0
        ].train_rows
        ==
        26
    )


    assert (
        artifacts[
            0
        ].train_rows
        !=
        mlp_execution.standardizer
        .fitted_observation_count
    )


    print(
        "[PASS] provenance distinguishes training windows from scaler observations"
    )


    # ========================================================
    # 8. CONTRACT / EXECUTION MISMATCH FAILS BEFORE STORAGE
    # ========================================================


    count_before = len(
        list_ml_model_artifacts(
            workflow_id=
                WORKFLOW_ID
        )
    )


    require_contract_error(
        lambda:
            register_time_series_neural_execution_artifact(
                windows=
                    windows,

                training_contract=
                    rnn_contract,

                execution=
                    mlp_execution,

                preparation_session_revision=
                    PREPARATION_REVISION,
            ),
        label=
            "MLP execution with RNN contract",
    )


    count_after = len(
        list_ml_model_artifacts(
            workflow_id=
                WORKFLOW_ID
        )
    )


    assert (
        count_after
        ==
        count_before
    )


    print(
        "[PASS] contract/execution mismatch creates no Artifact"
    )


print(
    "[PASS] isolated Artifact lifecycle leaves no persistent test state"
)


# ============================================================
# 9. STATIC NO-TRAINING BOUNDARY
# ============================================================


source_path = Path(
    "app/deep_learning/"
    "time_series_artifact_registration.py"
)


source_text = source_path.read_text(
    encoding="utf-8"
)


tree = ast.parse(
    source_text
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


for forbidden_module in (
    "app.deep_learning.training",
    "app.deep_learning.time_series_mlp_executor.execute_time_series_mlp",
    "app.deep_learning.time_series_rnn_executor.execute_time_series_rnn",
    "app.deep_learning.time_series_lstm_executor.execute_time_series_lstm",
):

    assert (
        forbidden_module
        not in
        imports
    )


for forbidden_term in (
    "train_regression_epochs",
    "build_sgd_optimizer",
    ".backward(",
    "optimizer.step(",
):

    assert (
        forbidden_term
        not in
        source_text
    )


print(
    "[PASS] Artifact adapter contains no training authority"
)


print()
print("=" * 80)
print("DL-5-A13-P1 FINAL VERDICT")
print("=" * 80)
print()

print("Exact A4 evidence validation                PASS")
print("Per-model Training Contract binding          PASS")
print("MLP Artifact registration                    PASS")
print("RNN Artifact registration                    PASS")
print("LSTM Artifact registration                   PASS")
print("Generic pytorch_bundle storage               PASS")
print("Generic Experiment Provenance                PASS")
print("Training-contract SHA-256                    PASS")
print("Server-owned model IDs                       PASS")
print("Artifact index estimator identities          PASS")
print("Stored-byte trusted inference parity          PASS")
print("Training-window provenance semantics          PASS")
print("No second training                           PASS")

print()
print(
    "DL-5-A13-P1 - FORECAST ARTIFACT REGISTRATION / PROVENANCE: PASS"
)
