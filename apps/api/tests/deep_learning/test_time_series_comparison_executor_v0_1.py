from __future__ import annotations


import ast
import math
from pathlib import Path


import numpy as np
import pandas as pd


from app.deep_learning.time_series_comparison_executor import (
    DL_TIME_SERIES_COMPARISON_EXECUTOR_RULE_VERSION,
    DLTimeSeriesComparisonExecutionResult,
    execute_time_series_model_comparison,
)


from app.deep_learning.time_series_contracts import (
    DLTimeSeriesLSTMRegressorHyperparameters,
    DLTimeSeriesMLPRegressorHyperparameters,
    DLTimeSeriesRNNRegressorHyperparameters,
)


from app.ml.contracts import (
    MLTimeHoldoutSplitContract,
)


from app.ml.time_series_comparison import (
    ML_TIME_SERIES_NAIVE_ESTIMATOR_KEY,
    build_time_series_forecast_candidate,
    compare_time_series_forecast_candidates,
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
    build_time_series_supervised_windows,
)


# ============================================================
# 1. COMMON CONTROLLED FORECASTING PROBLEM
# ============================================================


forecast_contract = (
    MLTimeSeriesForecastingContract(
        workflow_id=
            "workflow:four-candidate-comparison",

        dataset_id=
            "dataset:four-candidate-comparison",

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
            forecast_contract,
    )
)


partition = (
    resolve_time_series_holdout(
        series=
            series,

        contract=
            forecast_contract,
    )
)


windows = (
    build_time_series_supervised_windows(
        series=
            series,

        partition=
            partition,

        contract=
            forecast_contract,
    )
)


train_inputs_before = (
    windows.train.inputs
    .copy()
)


train_targets_before = (
    windows.train.targets
    .copy()
)


test_inputs_before = (
    windows.test.inputs
    .copy()
)


test_targets_before = (
    windows.test.targets
    .copy()
)


print(
    "[PASS] one exact A4 population owns the complete experiment"
)


# ============================================================
# 2. CONTROLLED MODEL CONFIGURATION
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
    "[PASS] MLP/RNN/LSTM configurations are explicit before orchestration"
)


# ============================================================
# 3. COMPLETE FOUR-CANDIDATE EXECUTION
# ============================================================


result = (
    execute_time_series_model_comparison(
        windows=
            windows,

        mlp_hyperparameters=
            mlp_parameters,

        rnn_hyperparameters=
            rnn_parameters,

        lstm_hyperparameters=
            lstm_parameters,

        execution_device=
            "cpu",
    )
)


assert isinstance(
    result,
    DLTimeSeriesComparisonExecutionResult,
)


assert (
    result.rule_version
    ==
    DL_TIME_SERIES_COMPARISON_EXECUTOR_RULE_VERSION
)


print(
    "[PASS] one orchestrator executes the complete candidate family"
)


# ============================================================
# 4. EXACT ESTIMATOR FAMILY
# ============================================================


assert (
    result.mlp.estimator_key
    ==
    "time_series_mlp_regressor"
)


assert (
    result.rnn.estimator_key
    ==
    "time_series_rnn_regressor"
)


assert (
    result.lstm.estimator_key
    ==
    "time_series_lstm_regressor"
)


ranked_keys = (
    result.ranked_estimator_keys
)


assert (
    set(
        ranked_keys
    )
    ==
    {
        ML_TIME_SERIES_NAIVE_ESTIMATOR_KEY,
        "time_series_mlp_regressor",
        "time_series_rnn_regressor",
        "time_series_lstm_regressor",
    }
)


assert (
    len(
        ranked_keys
    )
    ==
    4
)


print(
    "[PASS] exact Naive/MLP/RNN/LSTM family reaches A9 comparison"
)


# ============================================================
# 5. EXACT SAME TEST TARGET POPULATION
# ============================================================


for execution in (
    result.baseline,
    result.mlp,
    result.rnn,
    result.lstm,
):

    assert (
        execution.target_positions
        ==
        windows.test.target_positions
    )


    np.testing.assert_array_equal(
        execution.targets,
        windows.test.targets,
    )


    assert (
        execution.metrics.sample_count
        ==
        windows.test.sample_count
    )


print(
    "[PASS] all four candidates use exact same TEST positions and targets"
)


# ============================================================
# 6. SAME TRAIN-ONLY PREPROCESSING FOR NEURAL FAMILY
# ============================================================


assert (
    result.mlp.standardizer
    ==
    result.rnn.standardizer
)


assert (
    result.rnn.standardizer
    ==
    result.lstm.standardizer
)


assert (
    result.mlp.standardizer.fitted_observation_count
    ==
    30
)


print(
    "[PASS] all neural candidates reuse equivalent TRAIN-only scaling"
)


# ============================================================
# 7. COMMON METRIC AUTHORITY
# ============================================================


for execution in (
    result.baseline,
    result.mlp,
    result.rnn,
    result.lstm,
):

    for metric_name in (
        "mae",
        "mse",
        "rmse",
        "mean_error",
    ):

        assert math.isfinite(
            float(
                getattr(
                    execution.metrics,
                    metric_name,
                )
            )
        )


print(
    "[PASS] every candidate is evaluated through the shared A5 metrics"
)


# ============================================================
# 8. A9 DIRECT RECONSTRUCTION PARITY
# ============================================================


baseline_candidate = (
    build_time_series_forecast_candidate(
        estimator_key=
            ML_TIME_SERIES_NAIVE_ESTIMATOR_KEY,

        predictions=
            result.baseline.predictions,

        targets=
            result.baseline.targets,

        target_positions=
            result.baseline.target_positions,

        metrics=
            result.baseline.metrics,
    )
)


mlp_candidate = (
    build_time_series_forecast_candidate(
        estimator_key=
            result.mlp.estimator_key,

        predictions=
            result.mlp.predictions,

        targets=
            result.mlp.targets,

        target_positions=
            result.mlp.target_positions,

        metrics=
            result.mlp.metrics,
    )
)


rnn_candidate = (
    build_time_series_forecast_candidate(
        estimator_key=
            result.rnn.estimator_key,

        predictions=
            result.rnn.predictions,

        targets=
            result.rnn.targets,

        target_positions=
            result.rnn.target_positions,

        metrics=
            result.rnn.metrics,
    )
)


lstm_candidate = (
    build_time_series_forecast_candidate(
        estimator_key=
            result.lstm.estimator_key,

        predictions=
            result.lstm.predictions,

        targets=
            result.lstm.targets,

        target_positions=
            result.lstm.target_positions,

        metrics=
            result.lstm.metrics,
    )
)


direct_comparison = (
    compare_time_series_forecast_candidates(
        candidates=
            (
                baseline_candidate,
                mlp_candidate,
                rnn_candidate,
                lstm_candidate,
            )
    )
)


assert (
    tuple(
        candidate.estimator_key
        for candidate
        in direct_comparison.ranked_candidates
    )
    ==
    result.ranked_estimator_keys
)


assert (
    direct_comparison.winner.estimator_key
    ==
    result.winner.estimator_key
)


for (
    direct_candidate,
    orchestrated_candidate,
) in zip(
    direct_comparison.ranked_candidates,
    result.comparison.ranked_candidates,
):

    assert (
        direct_candidate.estimator_key
        ==
        orchestrated_candidate.estimator_key
    )


    assert (
        direct_candidate.metrics
        ==
        orchestrated_candidate.metrics
    )


    np.testing.assert_array_equal(
        direct_candidate.predictions,
        orchestrated_candidate.predictions,
    )


print(
    "[PASS] orchestrator delegates ranking exactly to A9 authority"
)


# ============================================================
# 9. WINNER IS EXPERIMENTAL ONLY
# ============================================================


assert (
    result.winner
    ==
    result.comparison.ranked_candidates[
        0
    ]
)


assert (
    result.winner.estimator_key
    in
    ranked_keys
)


print(
    (
        "[INFO] selected experiment winner="
        f"{result.winner.estimator_key} "
        "RMSE="
        f"{result.winner.metrics.rmse:.6f}"
    )
)


print(
    "[PASS] winner is comparison evidence, not automatic promotion"
)


# ============================================================
# 10. TRAINED NEURAL STATE RETAINED
# ============================================================


assert (
    result.mlp.model
    is not None
)


assert (
    result.rnn.model
    is not None
)


assert (
    result.lstm.model
    is not None
)


assert (
    result.mlp.standardizer
    is not None
)


assert (
    result.rnn.standardizer
    is not None
)


assert (
    result.lstm.standardizer
    is not None
)


for candidate in (
    result.comparison.ranked_candidates
):

    assert not hasattr(
        candidate,
        "model",
    )

    assert not hasattr(
        candidate,
        "standardizer",
    )

    assert not hasattr(
        candidate,
        "epoch_losses",
    )


print(
    "[PASS] trained state stays private to execution results, not A9 ranking"
)


# ============================================================
# 11. ORCHESTRATION DOES NOT MUTATE A4 WINDOWS
# ============================================================


np.testing.assert_array_equal(
    windows.train.inputs,
    train_inputs_before,
)


np.testing.assert_array_equal(
    windows.train.targets,
    train_targets_before,
)


np.testing.assert_array_equal(
    windows.test.inputs,
    test_inputs_before,
)


np.testing.assert_array_equal(
    windows.test.targets,
    test_targets_before,
)


print(
    "[PASS] complete experiment leaves shared A4 evidence unchanged"
)


# ============================================================
# 12. DETERMINISTIC CPU REPLAY
# ============================================================


replay = (
    execute_time_series_model_comparison(
        windows=
            windows,

        mlp_hyperparameters=
            mlp_parameters,

        rnn_hyperparameters=
            rnn_parameters,

        lstm_hyperparameters=
            lstm_parameters,

        execution_device=
            "cpu",
    )
)


np.testing.assert_array_equal(
    replay.mlp.predictions,
    result.mlp.predictions,
)


np.testing.assert_array_equal(
    replay.rnn.predictions,
    result.rnn.predictions,
)


np.testing.assert_array_equal(
    replay.lstm.predictions,
    result.lstm.predictions,
)


assert (
    replay.mlp.epoch_losses
    ==
    result.mlp.epoch_losses
)


assert (
    replay.rnn.epoch_losses
    ==
    result.rnn.epoch_losses
)


assert (
    replay.lstm.epoch_losses
    ==
    result.lstm.epoch_losses
)


assert (
    replay.ranked_estimator_keys
    ==
    result.ranked_estimator_keys
)


assert (
    replay.winner.estimator_key
    ==
    result.winner.estimator_key
)


print(
    "[PASS] complete four-candidate CPU experiment replays deterministically"
)


# ============================================================
# 13. NO ARTIFACT / PROMOTION COUPLING YET
# ============================================================


source_path = Path(
    "app/deep_learning/time_series_comparison_executor.py"
)


source_text = source_path.read_text(
    encoding="utf-8"
)


tree = ast.parse(
    source_text
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


for forbidden_prefix in (
    "app.ml.model_artifact",
    "app.ml.model_promotion",
    "app.persistence",
):

    assert not any(
        module_name.startswith(
            forbidden_prefix
        )
        for module_name
        in imported_modules
    )


print(
    "[PASS] orchestration remains separate from Artifact and Promotion lifecycle"
)


print()
print("=" * 80)
print("DL-5-A10-V1 FINAL VERDICT")
print("=" * 80)
print()

print("One exact A4 experiment population          PASS")
print("Naive candidate execution                   PASS")
print("Temporal MLP execution                      PASS")
print("Simple RNN execution                        PASS")
print("LSTM execution                              PASS")
print("Exact shared TEST population                PASS")
print("Equivalent TRAIN-only neural scaling        PASS")
print("Shared A5 metrics                           PASS")
print("Exact A9 ranking delegation                 PASS")
print("Experimental winner                         PASS")
print("Trained neural state retained               PASS")
print("Framework-neutral ranking evidence          PASS")
print("Deterministic complete CPU replay           PASS")
print("Artifact lifecycle remains separate         PASS")

print()
print(
    "DL-5-A10-V1 - FOUR-CANDIDATE ORCHESTRATOR: PASS"
)
