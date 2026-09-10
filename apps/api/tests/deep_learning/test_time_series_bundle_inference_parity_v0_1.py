from __future__ import annotations


import ast
import inspect
from pathlib import Path


import numpy as np
import pandas as pd
import torch


from app.deep_learning.time_series_bundle import (
    deserialize_trusted_time_series_neural_bundle,
    serialize_time_series_neural_bundle,
)


from app.deep_learning.time_series_bundle_inference import (
    DL_TIME_SERIES_BUNDLE_INFERENCE_RULE_VERSION,
    DLTimeSeriesBundleInferenceError,
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


# ============================================================
# HELPERS
# ============================================================


def require_inference_error(
    callback,
    *,
    label: str,
) -> None:

    try:

        callback()

    except DLTimeSeriesBundleInferenceError:

        return


    raise AssertionError(
        (
            "Expected Artifact inference failure: "
            f"{label}"
        )
    )


# ============================================================
# 1. EXACT CONTROLLED FORECASTING POPULATION
# ============================================================


task = (
    MLTimeSeriesForecastingContract(
        workflow_id=
            "workflow:a12",

        dataset_id=
            "dataset:a12",

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
    windows.test.sample_count
    ==
    10
)


print(
    "[PASS] A12 owns one exact A4 TEST input population"
)


# ============================================================
# 2. CONTROLLED TRAINING CONFIGURATION
# ============================================================


mlp_parameters = (
    DLTimeSeriesMLPRegressorHyperparameters(
        hidden_features=
            8,

        epochs=
            10,

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
            10,

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
            10,

        batch_size=
            16,

        learning_rate=
            0.02,
    )
)


print(
    "[PASS] exact estimator configuration frozen before training"
)


# ============================================================
# 3. TRAIN ONCE
# ============================================================


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
    "[PASS] MLP/RNN/LSTM are trained exactly once before persistence"
)


# ============================================================
# 4. PER-MODEL TRAINING CONTRACTS
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
        mlp_execution,
        mlp_contract,
    ),
    (
        rnn_execution,
        rnn_contract,
    ),
    (
        lstm_execution,
        lstm_contract,
    ),
)


# ============================================================
# 5. SERIALIZE / TRUSTED RELOAD / INFERENCE PARITY
# ============================================================


for (
    execution,
    training_contract,
) in cases:

    model_state_before = {
        name:
            value.detach().clone()

        for (
            name,
            value,
        ) in execution.model.state_dict().items()
    }


    bundle_bytes = (
        serialize_time_series_neural_bundle(
            model=
                execution.model,

            standardizer=
                execution.standardizer,

            training_contract=
                training_contract,
        )
    )


    restored = (
        deserialize_trusted_time_series_neural_bundle(
            trusted_bundle_bytes=
                bundle_bytes,

            training_contract=
                training_contract,
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


    assert (
        inference.estimator_key
        ==
        execution.estimator_key
    )


    assert (
        inference.sample_count
        ==
        windows.test.sample_count
    )


    assert (
        inference.rule_version
        ==
        DL_TIME_SERIES_BUNDLE_INFERENCE_RULE_VERSION
    )


    np.testing.assert_array_equal(
        inference.predictions,
        execution.predictions,
    )


    assert (
        inference.predictions.flags.writeable
        is False
    )


    assert (
        restored.standardizer
        ==
        execution.standardizer
    )


    assert (
        restored.model.training
        is False
    )


    for parameter in (
        restored.model.parameters()
    ):

        assert (
            parameter.device.type
            ==
            "cpu"
        )


    for (
        name,
        value,
    ) in execution.model.state_dict().items():

        assert torch.equal(
            value,
            model_state_before[
                name
            ],
        )


print(
    "[PASS] MLP/RNN/LSTM predictions survive trusted reload exactly"
)


# ============================================================
# 6. RELOAD REPLAY IS DETERMINISTIC
# ============================================================


for (
    execution,
    training_contract,
) in cases:

    bundle_bytes = (
        serialize_time_series_neural_bundle(
            model=
                execution.model,

            standardizer=
                execution.standardizer,

            training_contract=
                training_contract,
        )
    )


    restored_1 = (
        deserialize_trusted_time_series_neural_bundle(
            trusted_bundle_bytes=
                bundle_bytes,

            training_contract=
                training_contract,
        )
    )


    restored_2 = (
        deserialize_trusted_time_series_neural_bundle(
            trusted_bundle_bytes=
                bundle_bytes,

            training_contract=
                training_contract,
        )
    )


    prediction_1 = (
        predict_trusted_time_series_bundle(
            components=
                restored_1,

            inputs=
                windows.test.inputs,
        )
    )


    prediction_2 = (
        predict_trusted_time_series_bundle(
            components=
                restored_2,

            inputs=
                windows.test.inputs,
        )
    )


    np.testing.assert_array_equal(
        prediction_1.predictions,
        prediction_2.predictions,
    )


print(
    "[PASS] repeated trusted reload produces deterministic inference"
)


# ============================================================
# 7. TARGET-BLIND INFERENCE API
# ============================================================


signature = inspect.signature(
    predict_trusted_time_series_bundle
)


assert set(
    signature.parameters
) == {
    "components",
    "inputs",
}


assert (
    "targets"
    not in
    signature.parameters
)


assert (
    "metrics"
    not in
    signature.parameters
)


print(
    "[PASS] restored Artifact inference API is structurally target-blind"
)


# ============================================================
# 8. TARGET MUTATION CANNOT ALTER RELOADED PREDICTIONS
# ============================================================


mlp_bundle = (
    serialize_time_series_neural_bundle(
        model=
            mlp_execution.model,

        standardizer=
            mlp_execution.standardizer,

        training_contract=
            mlp_contract,
    )
)


restored_mlp = (
    deserialize_trusted_time_series_neural_bundle(
        trusted_bundle_bytes=
            mlp_bundle,

        training_contract=
            mlp_contract,
    )
)


before = (
    predict_trusted_time_series_bundle(
        components=
            restored_mlp,

        inputs=
            windows.test.inputs,
    )
)


mutated_targets = (
    windows.test.targets.copy()
)


mutated_targets += (
    1_000_000.0
)


after = (
    predict_trusted_time_series_bundle(
        components=
            restored_mlp,

        inputs=
            windows.test.inputs,
    )
)


np.testing.assert_array_equal(
    before.predictions,
    after.predictions,
)


assert not np.array_equal(
    mutated_targets,
    windows.test.targets,
)


print(
    "[PASS] TEST targets cannot influence restored Artifact prediction"
)


# ============================================================
# 9. WRONG WINDOW WIDTH FAILS CLOSED
# ============================================================


wrong_width = (
    windows.test.inputs[
        :,
        :-1,
    ]
)


require_inference_error(
    lambda:
        predict_trusted_time_series_bundle(
            components=
                restored_mlp,

            inputs=
                wrong_width,
        ),
    label=
        "wrong lookback width",
)


print(
    "[PASS] restored Artifact requires exact persisted lookback"
)


# ============================================================
# 10. NON-FINITE INFERENCE INPUT FAILS CLOSED
# ============================================================


non_finite_inputs = (
    windows.test.inputs.copy()
)


non_finite_inputs[
    0,
    0,
] = float(
    "nan"
)


require_inference_error(
    lambda:
        predict_trusted_time_series_bundle(
            components=
                restored_mlp,

            inputs=
                non_finite_inputs,
        ),
    label=
        "non-finite input",
)


print(
    "[PASS] non-finite restored Artifact input fails closed"
)


# ============================================================
# 11. NO TRAINING AUTHORITY IN RELOAD INFERENCE MODULE
# ============================================================


source_path = Path(
    "app/deep_learning/time_series_bundle_inference.py"
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


for forbidden_module in (
    "app.deep_learning.training",
    "app.deep_learning.time_series_mlp_executor",
    "app.deep_learning.time_series_rnn_executor",
    "app.deep_learning.time_series_lstm_executor",
):

    assert (
        forbidden_module
        not in
        imports
    )


for forbidden_term in (
    "build_sgd_optimizer",
    "train_regression_epochs",
    ".backward(",
    "optimizer.step(",
):

    assert (
        forbidden_term
        not in
        source
    )


print(
    "[PASS] trusted Artifact inference contains no training authority"
)


print()
print("=" * 80)
print("DL-5-A12-V1 FINAL VERDICT")
print("=" * 80)
print()

print("Train once before persistence                PASS")
print("MLP trusted reload inference parity          PASS")
print("RNN trusted reload inference parity          PASS")
print("LSTM trusted reload inference parity         PASS")
print("Original-unit prediction parity              PASS")
print("Restored TRAIN-only scaler parity            PASS")
print("CPU-owned trusted restored models            PASS")
print("Deterministic reload replay                   PASS")
print("Target-blind inference API                    PASS")
print("Persisted lookback enforcement                PASS")
print("Non-finite input guard                        PASS")
print("No training during Artifact inference         PASS")

print()
print(
    "DL-5-A12-V1 - TRUSTED RELOAD INFERENCE PARITY: PASS"
)
