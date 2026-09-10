from __future__ import annotations


import ast
import math
from pathlib import Path


import numpy as np
import pandas as pd
import torch


from pydantic import (
    ValidationError,
)


from app.deep_learning.time_series_contracts import (
    DL_TIME_SERIES_LSTM_CONTRACT_RULE_VERSION,
    DL_TIME_SERIES_MODEL_RANDOM_SEED,
    DLTimeSeriesLSTMRegressorHyperparameters,
)


from app.deep_learning.time_series_lstm_executor import (
    DL_TIME_SERIES_LSTM_EXECUTOR_RULE_VERSION,
    execute_time_series_lstm,
)


from app.deep_learning.time_series_lstm_network import (
    DL_TIME_SERIES_LSTM_NETWORK_RULE_VERSION,
    TimeSeriesLSTMRegressor,
)


from app.ml.contracts import (
    MLTimeHoldoutSplitContract,
)


from app.ml.time_series_baseline import (
    evaluate_last_value_baseline,
)


from app.ml.time_series_contracts import (
    MLTimeSeriesForecastingContract,
)


from app.ml.time_series_holdout import (
    resolve_time_series_holdout,
)


from app.ml.time_series_training_input import (
    validate_and_extract_time_series,
)


from app.ml.time_series_windows import (
    MLTimeSeriesSupervisedWindows,
    MLTimeSeriesWindowBatch,
    build_time_series_supervised_windows,
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

    except ValidationError:
        return


    raise AssertionError(
        (
            "Expected LSTM contract failure: "
            f"{label}"
        )
    )


# ============================================================
# 1. EXACT COMMON FORECAST POPULATION
# ============================================================


forecast_contract = (
    MLTimeSeriesForecastingContract(
        workflow_id="workflow:lstm",
        dataset_id="dataset:lstm",
        target_column="revenue",
        lookback=4,
        split=
            MLTimeHoldoutSplitContract(
                time_column="order_date",
                test_size=0.25,
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
                for index in range(
                    40
                )
            ],
    }
)


series = validate_and_extract_time_series(
    dataframe=frame,
    contract=forecast_contract,
)


partition = resolve_time_series_holdout(
    series=series,
    contract=forecast_contract,
)


windows = build_time_series_supervised_windows(
    series=series,
    partition=partition,
    contract=forecast_contract,
)


baseline = evaluate_last_value_baseline(
    windows=windows
)


assert (
    windows.test.sample_count
    ==
    baseline.sample_count
)


print(
    "[PASS] LSTM consumes the exact common forecasting population"
)


# ============================================================
# 2. CONTROLLED CONTRACT
# ============================================================


hyperparameters = (
    DLTimeSeriesLSTMRegressorHyperparameters(
        hidden_size=12,
        epochs=80,
        batch_size=16,
        learning_rate=0.02,
    )
)


assert (
    hyperparameters.kind
    ==
    "time_series_lstm_regressor"
)


assert (
    hyperparameters.rule_version
    ==
    DL_TIME_SERIES_LSTM_CONTRACT_RULE_VERSION
)


assert (
    DL_TIME_SERIES_MODEL_RANDOM_SEED
    ==
    42
)


require_validation_error(
    lambda:
        DLTimeSeriesLSTMRegressorHyperparameters(
            hidden_size=0
        ),
    label="hidden_size=0",
)


require_validation_error(
    lambda:
        DLTimeSeriesLSTMRegressorHyperparameters(
            hidden_size=True
        ),
    label="boolean hidden_size",
)


print(
    "[PASS] LSTM capacity is explicit and bounded"
)


# ============================================================
# 3. GENUINE LSTM NETWORK
# ============================================================


network = TimeSeriesLSTMRegressor(
    lookback=4,
    hidden_size=12,
)


assert isinstance(
    network.recurrent,
    torch.nn.LSTM,
)


assert (
    network.recurrent.input_size
    ==
    1
)


assert (
    network.recurrent.hidden_size
    ==
    12
)


assert (
    network.recurrent.num_layers
    ==
    1
)


assert (
    network.recurrent.batch_first
    is True
)


assert (
    network.rule_version
    ==
    DL_TIME_SERIES_LSTM_NETWORK_RULE_VERSION
)


network_input = torch.tensor(
    [
        [
            1.0,
            2.0,
            3.0,
            4.0,
        ],
        [
            5.0,
            6.0,
            7.0,
            8.0,
        ],
    ],
    dtype=torch.float32,
)


sequence = network_input.unsqueeze(
    dim=-1
)


recurrent_output, (
    hidden_state,
    cell_state,
) = network.recurrent(
    sequence
)


assert (
    tuple(
        recurrent_output.shape
    )
    ==
    (
        2,
        4,
        12,
    )
)


assert (
    tuple(
        hidden_state.shape
    )
    ==
    (
        1,
        2,
        12,
    )
)


assert (
    tuple(
        cell_state.shape
    )
    ==
    (
        1,
        2,
        12,
    )
)


network_output = network(
    network_input
)


assert (
    tuple(
        network_output.shape
    )
    ==
    (
        2,
    )
)


print(
    "[PASS] genuine LSTM exposes hidden and cell recurrent states"
)


# ============================================================
# 4. CPU EXECUTION
# ============================================================


result = execute_time_series_lstm(
    windows=windows,
    hyperparameters=hyperparameters,
    execution_device="cpu",
)


assert (
    result.estimator_key
    ==
    "time_series_lstm_regressor"
)


assert (
    result.execution_device
    ==
    "cpu"
)


assert (
    result.random_seed
    ==
    DL_TIME_SERIES_MODEL_RANDOM_SEED
)


assert (
    result.lookback
    ==
    forecast_contract.lookback
)


assert (
    result.hidden_size
    ==
    hyperparameters.hidden_size
)


assert (
    len(
        result.epoch_losses
    )
    ==
    hyperparameters.epochs
)


assert all(
    math.isfinite(
        value
    )
    for value in result.epoch_losses
)


assert math.isfinite(
    result.scaled_test_loss
)


assert (
    result.rule_version
    ==
    DL_TIME_SERIES_LSTM_EXECUTOR_RULE_VERSION
)


assert isinstance(
    result.model.recurrent,
    torch.nn.LSTM,
)


print(
    "[PASS] LSTM trains through shared DL-2 regression authorities"
)


# ============================================================
# 5. EXACT TEST POPULATION / SHARED METRICS
# ============================================================


assert (
    result.test_samples
    ==
    windows.test.sample_count
)


assert (
    result.target_positions
    ==
    windows.test.target_positions
)


np.testing.assert_array_equal(
    result.targets,
    windows.test.targets,
)


assert (
    result.predictions.shape
    ==
    result.targets.shape
)


assert np.isfinite(
    result.predictions
).all()


assert (
    result.predictions.flags.writeable
    is False
)


assert (
    result.targets.flags.writeable
    is False
)


assert (
    result.metrics.sample_count
    ==
    baseline.metrics.sample_count
)


for metric_name in (
    "mae",
    "mse",
    "rmse",
    "mean_error",
):

    assert hasattr(
        result.metrics,
        metric_name,
    )

    assert hasattr(
        baseline.metrics,
        metric_name,
    )

    assert math.isfinite(
        float(
            getattr(
                result.metrics,
                metric_name,
            )
        )
    )


print(
    (
        "[INFO] baseline RMSE="
        f"{baseline.metrics.rmse:.6f} "
        "LSTM RMSE="
        f"{result.metrics.rmse:.6f}"
    )
)


print(
    "[PASS] LSTM uses exact TEST targets and shared metrics"
)


# ============================================================
# 6. TRAIN-ONLY SCALING
# ============================================================


expected_train_values = np.arange(
    50.0,
    80.0,
    dtype=np.float64,
)


assert math.isclose(
    result.standardizer.mean,
    float(
        expected_train_values.mean()
    ),
    rel_tol=0.0,
    abs_tol=1e-12,
)


assert math.isclose(
    result.standardizer.standard_deviation,
    float(
        expected_train_values.std(
            ddof=0
        )
    ),
    rel_tol=0.0,
    abs_tol=1e-12,
)


assert (
    result.standardizer.fitted_observation_count
    ==
    30
)


print(
    "[PASS] LSTM reuses exact TRAIN-only A6 scaling"
)


# ============================================================
# 7. DETERMINISTIC CPU REPLAY
# ============================================================


replay = execute_time_series_lstm(
    windows=windows,
    hyperparameters=hyperparameters,
    execution_device="cpu",
)


np.testing.assert_array_equal(
    replay.predictions,
    result.predictions,
)


assert (
    replay.epoch_losses
    ==
    result.epoch_losses
)


assert (
    replay.metrics
    ==
    result.metrics
)


assert (
    replay.standardizer
    ==
    result.standardizer
)


print(
    "[PASS] fixed server seed reproduces exact CPU LSTM execution"
)


# ============================================================
# 8. TEST TARGET BLINDNESS
# ============================================================


mutated_test = MLTimeSeriesWindowBatch(
    inputs=
        windows.test.inputs.copy(),

    targets=
        (
            windows.test.targets.copy()
            +
            10_000.0
        ),

    context_positions=
        windows.test.context_positions,

    target_positions=
        windows.test.target_positions,

    context_source_row_positions=
        windows.test.context_source_row_positions,

    target_source_row_positions=
        windows.test.target_source_row_positions,

    target_times=
        windows.test.target_times,
)


mutated_windows = MLTimeSeriesSupervisedWindows(
    train=windows.train,
    test=mutated_test,
    lookback=windows.lookback,
    forecast_horizon=windows.forecast_horizon,
    evaluation_protocol=windows.evaluation_protocol,
    rule_version=windows.rule_version,
)


mutated_result = execute_time_series_lstm(
    windows=mutated_windows,
    hyperparameters=hyperparameters,
    execution_device="cpu",
)


np.testing.assert_array_equal(
    mutated_result.predictions,
    result.predictions,
)


assert not np.array_equal(
    mutated_result.targets,
    result.targets,
)


print(
    "[PASS] TEST targets affect LSTM evaluation only, never prediction"
)


# ============================================================
# 9. OPTIONAL CUDA
# ============================================================


if torch.cuda.is_available():

    cuda_parameters = (
        DLTimeSeriesLSTMRegressorHyperparameters(
            hidden_size=8,
            epochs=2,
            batch_size=16,
            learning_rate=0.01,
        )
    )


    cuda_result = execute_time_series_lstm(
        windows=windows,
        hyperparameters=cuda_parameters,
        execution_device="cuda",
    )


    assert (
        cuda_result.execution_device.startswith(
            "cuda"
        )
    )


    assert np.isfinite(
        cuda_result.predictions
    ).all()


    print(
        "[PASS] optional CUDA LSTM execution"
    )

else:

    print(
        "[SKIP] CUDA unavailable; CPU LSTM authority validated"
    )


# ============================================================
# 10. CONTRACT REMAINS TORCH-FREE
# ============================================================


contract_source = Path(
    "app/deep_learning/time_series_contracts.py"
).read_text(
    encoding="utf-8"
)


contract_tree = ast.parse(
    contract_source
)


imports = []


for node in ast.walk(
    contract_tree
):

    if isinstance(
        node,
        ast.Import,
    ):

        imports.extend(
            alias.name
            for alias in node.names
        )


    elif isinstance(
        node,
        ast.ImportFrom,
    ):

        if node.module:

            imports.append(
                node.module
            )


assert not any(
    name == "torch"
    or name.startswith("torch.")
    for name in imports
)


print(
    "[PASS] MLP/RNN/LSTM hyperparameter contracts remain torch-free"
)


print()
print("=" * 80)
print("DL-5-A8-V1 FINAL VERDICT")
print("=" * 80)
print()

print("Genuine torch.nn.LSTM                     PASS")
print("Hidden-state memory                       PASS")
print("Cell-state memory                         PASS")
print("Univariate sequential input               PASS")
print("One recurrent layer                       PASS")
print("Exact A4 TRAIN/TEST windows               PASS")
print("TRAIN-only A6 scaling                     PASS")
print("Shared DL-2 training                      PASS")
print("Predictions in original units             PASS")
print("Shared A5 metrics                         PASS")
print("Deterministic CPU replay                  PASS")
print("TEST-target blindness                     PASS")
print("Controlled CUDA support                   PASS")
print("Torch-free contract boundary              PASS")

print()
print(
    "DL-5-A8-V1 - LSTM CORE: PASS"
)
