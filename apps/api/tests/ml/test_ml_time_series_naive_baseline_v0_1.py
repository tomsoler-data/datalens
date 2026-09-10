from __future__ import annotations


import ast
import math
from pathlib import Path


import numpy as np
import pandas as pd


from app.ml.contracts import (
    MLTimeHoldoutSplitContract,
)


from app.ml.time_series_baseline import (
    ML_TIME_SERIES_NAIVE_BASELINE_RULE_VERSION,
    MLTimeSeriesNaiveBaselineError,
    evaluate_last_value_baseline,
)


from app.ml.time_series_contracts import (
    MLTimeSeriesForecastingContract,
)


from app.ml.time_series_evaluation import (
    ML_TIME_SERIES_EVALUATION_RULE_VERSION,
    MLTimeSeriesEvaluationError,
    evaluate_forecast_predictions,
)


from app.ml.time_series_holdout import (
    resolve_time_series_holdout,
)


from app.ml.time_series_training_input import (
    validate_and_extract_time_series,
)


from app.ml.time_series_windows import (
    ML_TIME_SERIES_EVALUATION_PROTOCOL,
    build_time_series_supervised_windows,
)


# ============================================================
# HELPERS
# ============================================================


def require_error(
    callback,
    *,
    error_type,
    label: str,
) -> None:

    try:
        callback()

    except error_type:
        return


    raise AssertionError(
        (
            "Expected failure was not raised: "
            f"{label}"
        )
    )


# ============================================================
# 1. BUILD EXACT A1-A4 POPULATION
# ============================================================


contract = (
    MLTimeSeriesForecastingContract(
        workflow_id=
            "workflow:naive-baseline",

        dataset_id=
            "dataset:naive-baseline",

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
                periods=20,
                freq="D",
            ),

        "revenue":
            [
                float(
                    100
                    +
                    index
                )
                for index
                in range(
                    20
                )
            ],
    }
)


source_order = [
    5,
    0,
    10,
    15,
    1,
    6,
    11,
    16,
    2,
    7,
    12,
    17,
    3,
    8,
    13,
    18,
    4,
    9,
    14,
    19,
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


assert (
    windows.evaluation_protocol
    ==
    ML_TIME_SERIES_EVALUATION_PROTOCOL
)


assert (
    windows.test.sample_count
    ==
    5
)


print(
    "[PASS] baseline consumes exact A1-A4 forecasting population"
)


# ============================================================
# 2. EXACT LAST-VALUE PREDICTIONS
# ============================================================


evaluation = (
    evaluate_last_value_baseline(
        windows=
            windows
    )
)


np.testing.assert_array_equal(
    evaluation.predictions,
    np.array(
        [
            114.0,
            115.0,
            116.0,
            117.0,
            118.0,
        ],
        dtype=np.float64,
    ),
)


np.testing.assert_array_equal(
    evaluation.targets,
    np.array(
        [
            115.0,
            116.0,
            117.0,
            118.0,
            119.0,
        ],
        dtype=np.float64,
    ),
)


assert (
    evaluation.target_positions
    ==
    (
        15,
        16,
        17,
        18,
        19,
    )
)


assert (
    evaluation.sample_count
    ==
    5
)


assert (
    evaluation.rule_version
    ==
    ML_TIME_SERIES_NAIVE_BASELINE_RULE_VERSION
)


print(
    "[PASS] naive prediction equals final observed value in each lookback"
)


# ============================================================
# 3. EXACT SHARED METRICS
# ============================================================


metrics = (
    evaluation.metrics
)


assert (
    metrics.sample_count
    ==
    5
)


assert math.isclose(
    metrics.mae,
    1.0,
    rel_tol=0.0,
    abs_tol=1e-12,
)


assert math.isclose(
    metrics.mse,
    1.0,
    rel_tol=0.0,
    abs_tol=1e-12,
)


assert math.isclose(
    metrics.rmse,
    1.0,
    rel_tol=0.0,
    abs_tol=1e-12,
)


assert math.isclose(
    metrics.mean_error,
    -1.0,
    rel_tol=0.0,
    abs_tol=1e-12,
)


assert (
    metrics.rule_version
    ==
    ML_TIME_SERIES_EVALUATION_RULE_VERSION
)


print(
    "[PASS] MAE / MSE / RMSE / mean error are exact"
)


# ============================================================
# 4. BASELINE IS TARGET-BLIND AT PREDICTION TIME
# ============================================================


mutated_targets = (
    windows.test.targets
    .copy()
)


mutated_targets.setflags(
    write=True
)


mutated_targets[:] = (
    np.array(
        [
            9000.0,
            8000.0,
            7000.0,
            6000.0,
            5000.0,
        ],
        dtype=np.float64,
    )
)


from app.ml.time_series_windows import (
    MLTimeSeriesSupervisedWindows,
    MLTimeSeriesWindowBatch,
)


mutated_test = (
    MLTimeSeriesWindowBatch(
        inputs=
            windows.test.inputs
            .copy(),

        targets=
            mutated_targets,

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


mutated_evaluation = (
    evaluate_last_value_baseline(
        windows=
            mutated_windows
    )
)


np.testing.assert_array_equal(
    mutated_evaluation.predictions,
    evaluation.predictions,
)


assert not np.array_equal(
    mutated_evaluation.targets,
    evaluation.targets,
)


print(
    "[PASS] baseline predictions depend only on historical context, not targets"
)


# ============================================================
# 5. GENERIC EVALUATOR SHAPE GUARD
# ============================================================


require_error(
    lambda:
        evaluate_forecast_predictions(
            predictions=
                np.array(
                    [
                        1.0,
                        2.0,
                    ]
                ),

            targets=
                np.array(
                    [
                        1.0,
                    ]
                ),
        ),

    error_type=
        MLTimeSeriesEvaluationError,

    label=
        "prediction/target shape mismatch",
)


print(
    "[PASS] shared evaluator rejects misaligned prediction populations"
)


# ============================================================
# 6. GENERIC EVALUATOR FINITE GUARD
# ============================================================


require_error(
    lambda:
        evaluate_forecast_predictions(
            predictions=
                np.array(
                    [
                        1.0,
                        np.nan,
                    ]
                ),

            targets=
                np.array(
                    [
                        1.0,
                        2.0,
                    ]
                ),
        ),

    error_type=
        MLTimeSeriesEvaluationError,

    label=
        "non-finite predictions",
)


require_error(
    lambda:
        evaluate_forecast_predictions(
            predictions=
                np.array(
                    [
                        1.0,
                        2.0,
                    ]
                ),

            targets=
                np.array(
                    [
                        1.0,
                        np.inf,
                    ]
                ),
        ),

    error_type=
        MLTimeSeriesEvaluationError,

    label=
        "non-finite targets",
)


print(
    "[PASS] shared evaluator fails closed on non-finite values"
)


# ============================================================
# 7. OUTPUT ISOLATION
# ============================================================


assert (
    evaluation.predictions.flags.writeable
    is False
)


assert (
    evaluation.targets.flags.writeable
    is False
)


original_prediction = float(
    evaluation.predictions[
        0
    ]
)


original_target = float(
    evaluation.targets[
        0
    ]
)


windows.test.inputs.setflags(
    write=True
)


windows.test.targets.setflags(
    write=True
)


windows.test.inputs[
    0,
    -1,
] = 999999.0


windows.test.targets[
    0
] = 888888.0


assert (
    evaluation.predictions[
        0
    ]
    ==
    original_prediction
)


assert (
    evaluation.targets[
        0
    ]
    ==
    original_target
)


print(
    "[PASS] baseline result is immutable and isolated from later window mutation"
)


# ============================================================
# 8. BASELINE HAS NO FIT / TRAIN STATE
# ============================================================


baseline_source = (
    Path(
        "app/ml/time_series_baseline.py"
    )
    .read_text(
        encoding="utf-8"
    )
)


for forbidden_token in (
    "fit(",
    ".fit(",
    "optimizer",
    "backward(",
    "state_dict",
    "random_seed",
):

    assert (
        forbidden_token
        not in
        baseline_source
    ), (
        "Unexpected learned/training state in baseline: "
        f"{forbidden_token}"
    )


print(
    "[PASS] naive baseline contains no training or learned state"
)


# ============================================================
# 9. STATIC TORCH-FREE SURFACE
# ============================================================


for source_path in (
    Path(
        "app/ml/time_series_evaluation.py"
    ),
    Path(
        "app/ml/time_series_baseline.py"
    ),
):

    source_text = (
        source_path.read_text(
            encoding="utf-8"
        )
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


    assert not any(
        name.startswith(
            "app.deep_learning"
        )
        for name
        in imports
    )


print(
    "[PASS] baseline and shared evaluation authorities are torch-free"
)


print()
print("=" * 80)
print("DL-5-A5-V1 FINAL VERDICT")
print("=" * 80)
print()

print(
    "Last-observation baseline                  PASS"
)

print(
    "No fitting / learned state                 PASS"
)

print(
    "Exact A4 TEST population                   PASS"
)

print(
    "Target-blind predictions                   PASS"
)

print(
    "Shared MAE metric                          PASS"
)

print(
    "Shared MSE metric                          PASS"
)

print(
    "Shared RMSE metric                         PASS"
)

print(
    "Shared signed-error metric                 PASS"
)

print(
    "Immutable evaluation outputs               PASS"
)

print(
    "Torch-free boundary                        PASS"
)

print()
print(
    "DL-5-A5-V1 - NAIVE FORECAST BASELINE: PASS"
)
