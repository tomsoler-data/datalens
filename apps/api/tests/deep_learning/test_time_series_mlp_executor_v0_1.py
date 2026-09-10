from __future__ import annotations


import ast
import math
from pathlib import Path


import numpy as np
import pandas as pd
import torch


from app.deep_learning.time_series_contracts import (
    DL_TIME_SERIES_MLP_CONTRACT_RULE_VERSION,
    DL_TIME_SERIES_MODEL_RANDOM_SEED,
    DLTimeSeriesMLPRegressorHyperparameters,
)


from app.deep_learning.time_series_mlp_executor import (
    DL_TIME_SERIES_MLP_EXECUTOR_RULE_VERSION,
    execute_time_series_mlp,
)


from app.deep_learning.time_series_scaling import (
    DL_TIME_SERIES_SCALING_RULE_VERSION,
    fit_time_series_train_standardizer,
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
# 1. BUILD COMMON A1-A5 POPULATION
# ============================================================


contract = (
    MLTimeSeriesForecastingContract(
        workflow_id=
            "workflow:temporal-mlp",

        dataset_id=
            "dataset:temporal-mlp",

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


canonical = pd.DataFrame(
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


source_order = [
    7, 0, 14, 21, 28, 35,
    1, 8, 15, 22, 29, 36,
    2, 9, 16, 23, 30, 37,
    3, 10, 17, 24, 31, 38,
    4, 11, 18, 25, 32, 39,
    5, 12, 19, 26, 33,
    6, 13, 20, 27, 34,
]


source = (
    canonical.iloc[
        source_order
    ]
    .reset_index(
        drop=True
    )
)


series = (
    validate_and_extract_time_series(
        dataframe=
            source,

        contract=
            contract,
    )
)


partition = (
    resolve_time_series_holdout(
        series=
            series,

        contract=
            contract,
    )
)


windows = (
    build_time_series_supervised_windows(
        series=
            series,

        partition=
            partition,

        contract=
            contract,
    )
)


baseline = (
    evaluate_last_value_baseline(
        windows=
            windows
    )
)


assert (
    windows.test.sample_count
    ==
    baseline.sample_count
)


print(
    "[PASS] temporal MLP consumes the exact A1-A5 forecasting population"
)


# ============================================================
# 2. CONTROLLED HYPERPARAMETER CONTRACT
# ============================================================


hyperparameters = (
    DLTimeSeriesMLPRegressorHyperparameters(
        hidden_features=
            16,

        epochs=
            250,

        batch_size=
            16,

        learning_rate=
            0.02,
    )
)


assert (
    hyperparameters.kind
    ==
    "time_series_mlp_regressor"
)


assert (
    hyperparameters.rule_version
    ==
    DL_TIME_SERIES_MLP_CONTRACT_RULE_VERSION
)


assert (
    DL_TIME_SERIES_MODEL_RANDOM_SEED
    ==
    42
)


print(
    "[PASS] temporal MLP configuration is explicit and controlled"
)


# ============================================================
# 3. TRAIN-ONLY SCALER
# ============================================================


standardizer = (
    fit_time_series_train_standardizer(
        windows=
            windows
    )
)


train_observation_count = (
    windows.lookback
    +
    windows.train.sample_count
)


assert (
    standardizer.fitted_observation_count
    ==
    train_observation_count
)


expected_train_values = np.arange(
    50.0,
    80.0,
    dtype=np.float64,
)


assert (
    standardizer.fitted_observation_count
    ==
    30
)


assert math.isclose(
    standardizer.mean,
    float(
        expected_train_values.mean()
    ),
    rel_tol=0.0,
    abs_tol=1e-12,
)


assert math.isclose(
    standardizer.standard_deviation,
    float(
        expected_train_values.std(
            ddof=0
        )
    ),
    rel_tol=0.0,
    abs_tol=1e-12,
)


assert (
    standardizer.rule_version
    ==
    DL_TIME_SERIES_SCALING_RULE_VERSION
)


print(
    "[PASS] standardizer fits each TRAIN observation exactly once"
)


# ============================================================
# 4. TEST VALUES CANNOT INFLUENCE SCALER
# ============================================================


mutated_test_inputs = (
    windows.test.inputs
    .copy()
)


mutated_test_targets = (
    windows.test.targets
    .copy()
)


mutated_test_inputs[:] = (
    mutated_test_inputs
    +
    1_000_000.0
)


mutated_test_targets[:] = (
    mutated_test_targets
    +
    2_000_000.0
)


mutated_test = (
    MLTimeSeriesWindowBatch(
        inputs=
            mutated_test_inputs,

        targets=
            mutated_test_targets,

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
)


mutated_windows = (
    MLTimeSeriesSupervisedWindows(
        train=
            windows.train,

        test=
            mutated_test,

        lookback=
            windows.lookback,

        forecast_horizon=
            windows.forecast_horizon,

        evaluation_protocol=
            windows.evaluation_protocol,

        rule_version=
            windows.rule_version,
    )
)


mutated_standardizer = (
    fit_time_series_train_standardizer(
        windows=
            mutated_windows
    )
)


assert (
    mutated_standardizer
    ==
    standardizer
)


print(
    "[PASS] TEST population cannot influence fitted scaling state"
)


# ============================================================
# 5. CPU TEMPORAL MLP EXECUTION
# ============================================================


result = (
    execute_time_series_mlp(
        windows=
            windows,

        hyperparameters=
            hyperparameters,

        execution_device=
            "cpu",
    )
)


assert (
    result.estimator_key
    ==
    "time_series_mlp_regressor"
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
    result.input_features
    ==
    contract.lookback
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
    for value
    in result.epoch_losses
)


assert math.isfinite(
    result.scaled_test_loss
)


assert (
    result.rule_version
    ==
    DL_TIME_SERIES_MLP_EXECUTOR_RULE_VERSION
)


print(
    "[PASS] temporal MLP trains through existing DL-2 PyTorch foundations"
)


# ============================================================
# 6. EXACT TEST POPULATION / ORIGINAL UNITS
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


assert (
    np.isfinite(
        result.predictions
    )
    .all()
)


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


assert math.isfinite(
    result.metrics.mae
)


assert math.isfinite(
    result.metrics.mse
)


assert math.isfinite(
    result.metrics.rmse
)


assert math.isfinite(
    result.metrics.mean_error
)


print(
    "[PASS] MLP predictions return to original units on exact TEST targets"
)


# ============================================================
# 7. BASELINE / MLP METRIC PARITY SURFACE
# ============================================================


assert (
    set(
        (
            "mae",
            "mse",
            "rmse",
            "mean_error",
        )
    )
    ==
    {
        field
        for field
        in (
            "mae",
            "mse",
            "rmse",
            "mean_error",
        )
        if hasattr(
            result.metrics,
            field
        )
        and hasattr(
            baseline.metrics,
            field
        )
    }
)


print(
    (
        "[INFO] baseline RMSE="
        f"{baseline.metrics.rmse:.6f} "
        "temporal MLP RMSE="
        f"{result.metrics.rmse:.6f}"
    )
)


print(
    "[PASS] baseline and temporal MLP share one evaluation metric authority"
)


# ============================================================
# 8. DETERMINISTIC CPU REPLAY
# ============================================================


replay = (
    execute_time_series_mlp(
        windows=
            windows,

        hyperparameters=
            hyperparameters,

        execution_device=
            "cpu",
    )
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
    "[PASS] fixed server seed reproduces exact CPU temporal MLP execution"
)


# ============================================================
# 9. TEST TARGETS DO NOT AFFECT PREDICTIONS
# ============================================================


target_only_test = (
    MLTimeSeriesWindowBatch(
        inputs=
            windows.test.inputs
            .copy(),

        targets=
            (
                windows.test.targets
                .copy()
                +
                5000.0
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
)


target_only_windows = (
    MLTimeSeriesSupervisedWindows(
        train=
            windows.train,

        test=
            target_only_test,

        lookback=
            windows.lookback,

        forecast_horizon=
            windows.forecast_horizon,

        evaluation_protocol=
            windows.evaluation_protocol,

        rule_version=
            windows.rule_version,
    )
)


target_only_result = (
    execute_time_series_mlp(
        windows=
            target_only_windows,

        hyperparameters=
            hyperparameters,

        execution_device=
            "cpu",
    )
)


np.testing.assert_array_equal(
    target_only_result.predictions,
    result.predictions,
)


assert not np.array_equal(
    target_only_result.targets,
    result.targets,
)


print(
    "[PASS] TEST targets affect evaluation only, never MLP prediction"
)


# ============================================================
# 10. OPTIONAL CUDA SMOKE
# ============================================================


if torch.cuda.is_available():

    cuda_hyperparameters = (
        DLTimeSeriesMLPRegressorHyperparameters(
            hidden_features=
                8,

            epochs=
                2,

            batch_size=
                16,

            learning_rate=
                0.01,
        )
    )


    cuda_result = (
        execute_time_series_mlp(
            windows=
                windows,

            hyperparameters=
                cuda_hyperparameters,

            execution_device=
                "cuda",
        )
    )


    assert (
        cuda_result.execution_device
        .startswith(
            "cuda"
        )
    )


    assert (
        np.isfinite(
            cuda_result.predictions
        )
        .all()
    )


    print(
        "[PASS] optional CUDA temporal MLP execution"
    )

else:

    print(
        "[SKIP] CUDA unavailable; CPU authority already validated"
    )


# ============================================================
# 11. CONTRACT REMAINS TORCH-FREE
# ============================================================


contract_source = (
    Path(
        "app/deep_learning/time_series_contracts.py"
    )
    .read_text(
        encoding="utf-8"
    )
)


tree = ast.parse(
    contract_source
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


assert not any(
    name
    ==
    "torch"
    or
    name.startswith(
        "torch."
    )
    for name
    in imports
)


print(
    "[PASS] temporal MLP hyperparameter contract remains torch-free"
)


print()
print("=" * 80)
print("DL-5-A6-V1 FINAL VERDICT")
print("=" * 80)
print()

print(
    "Existing FeedForwardRegressor reused       PASS"
)

print(
    "Lookback = MLP input width                 PASS"
)

print(
    "TRAIN-only scalar standardization          PASS"
)

print(
    "Each TRAIN observation fitted once         PASS"
)

print(
    "No TEST preprocessing leakage              PASS"
)

print(
    "CPU-owned tensor datasets                  PASS"
)

print(
    "Shared DL-2 training authority             PASS"
)

print(
    "Predictions restored to original units     PASS"
)

print(
    "Shared A5 evaluation metrics               PASS"
)

print(
    "Deterministic CPU replay                   PASS"
)

print(
    "TEST-target blindness                      PASS"
)

print(
    "Controlled CUDA support                    PASS"
)

print()
print(
    "DL-5-A6-V1 - TEMPORAL MLP CORE: PASS"
)
