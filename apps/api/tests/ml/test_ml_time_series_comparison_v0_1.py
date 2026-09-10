from __future__ import annotations


import ast
from pathlib import Path


import numpy as np


from app.ml.time_series_comparison import (
    ML_TIME_SERIES_COMPARISON_RULE_VERSION,
    ML_TIME_SERIES_NAIVE_ESTIMATOR_KEY,
    ML_TIME_SERIES_RANKING_KEYS,
    ML_TIME_SERIES_REQUIRED_ESTIMATOR_KEYS,
    MLTimeSeriesComparisonError,
    build_time_series_forecast_candidate,
    compare_time_series_forecast_candidates,
)


from app.ml.time_series_evaluation import (
    evaluate_forecast_predictions,
)


# ============================================================
# HELPERS
# ============================================================


TARGETS = np.array(
    [
        115.0,
        116.0,
        117.0,
        118.0,
        119.0,
    ],
    dtype=np.float64,
)


POSITIONS = (
    15,
    16,
    17,
    18,
    19,
)


def candidate(
    estimator_key: str,
    predictions: list[
        float
    ],
):

    prediction_array = np.array(
        predictions,
        dtype=np.float64,
    )


    metrics = evaluate_forecast_predictions(
        predictions=
            prediction_array,

        targets=
            TARGETS,
    )


    return build_time_series_forecast_candidate(
        estimator_key=
            estimator_key,

        predictions=
            prediction_array,

        targets=
            TARGETS,

        target_positions=
            POSITIONS,

        metrics=
            metrics,
    )


def require_comparison_error(
    callback,
    *,
    label: str,
) -> None:

    try:
        callback()

    except MLTimeSeriesComparisonError:
        return


    raise AssertionError(
        (
            "Expected comparison failure: "
            f"{label}"
        )
    )


# ============================================================
# 1. COMPLETE FOUR-CANDIDATE FAMILY
# ============================================================


baseline = candidate(
    ML_TIME_SERIES_NAIVE_ESTIMATOR_KEY,
    [
        114.0,
        115.0,
        116.0,
        117.0,
        118.0,
    ],
)


mlp = candidate(
    "time_series_mlp_regressor",
    [
        113.0,
        115.0,
        116.0,
        117.0,
        118.0,
    ],
)


rnn = candidate(
    "time_series_rnn_regressor",
    [
        110.0,
        111.0,
        112.0,
        113.0,
        114.0,
    ],
)


lstm = candidate(
    "time_series_lstm_regressor",
    [
        108.0,
        109.0,
        110.0,
        111.0,
        112.0,
    ],
)


assert (
    set(
        candidate.estimator_key
        for candidate
        in (
            baseline,
            mlp,
            rnn,
            lstm,
        )
    )
    ==
    set(
        ML_TIME_SERIES_REQUIRED_ESTIMATOR_KEYS
    )
)


print(
    "[PASS] exact DL-5 four-candidate family projects framework-neutrally"
)


# ============================================================
# 2. EXACT COMMON TEST POPULATION
# ============================================================


for item in (
    baseline,
    mlp,
    rnn,
    lstm,
):

    assert (
        item.target_positions
        ==
        POSITIONS
    )

    np.testing.assert_array_equal(
        item.targets,
        TARGETS,
    )

    assert (
        item.sample_count
        ==
        5
    )


print(
    "[PASS] all candidates expose exact same TEST targets and positions"
)


# ============================================================
# 3. DETERMINISTIC RANKING
# ============================================================


comparison = (
    compare_time_series_forecast_candidates(
        candidates=
            (
                lstm,
                rnn,
                mlp,
                baseline,
            )
    )
)


assert (
    tuple(
        item.estimator_key
        for item
        in comparison.ranked_candidates
    )
    ==
    (
        "naive_last_value",
        "time_series_mlp_regressor",
        "time_series_rnn_regressor",
        "time_series_lstm_regressor",
    )
)


assert (
    comparison.winner.estimator_key
    ==
    "naive_last_value"
)


assert (
    comparison.ranking_keys
    ==
    ML_TIME_SERIES_RANKING_KEYS
)


assert (
    comparison.rule_version
    ==
    ML_TIME_SERIES_COMPARISON_RULE_VERSION
)


print(
    "[PASS] lower forecast error wins regardless of model complexity"
)


# ============================================================
# 4. NEURAL MODEL MAY LEGITIMATELY WIN
# ============================================================


perfect_lstm = candidate(
    "time_series_lstm_regressor",
    TARGETS.tolist(),
)


neural_winner = (
    compare_time_series_forecast_candidates(
        candidates=
            (
                baseline,
                mlp,
                rnn,
                perfect_lstm,
            )
    )
)


assert (
    neural_winner.winner.estimator_key
    ==
    "time_series_lstm_regressor"
)


assert (
    neural_winner.winner.metrics.rmse
    ==
    0.0
)


print(
    "[PASS] ranking contains no baseline or complexity preference"
)


# ============================================================
# 5. MISMATCHED TARGET POSITIONS FAIL CLOSED
# ============================================================


wrong_positions = (
    14,
    16,
    17,
    18,
    19,
)


wrong_position_candidate = (
    build_time_series_forecast_candidate(
        estimator_key=
            "time_series_lstm_regressor",

        predictions=
            lstm.predictions,

        targets=
            TARGETS,

        target_positions=
            wrong_positions,

        metrics=
            lstm.metrics,
    )
)


require_comparison_error(
    lambda:
        compare_time_series_forecast_candidates(
            candidates=
                (
                    baseline,
                    mlp,
                    rnn,
                    wrong_position_candidate,
                )
        ),
    label=
        "mismatched TEST target positions",
)


print(
    "[PASS] mismatched TEST target positions fail closed"
)


# ============================================================
# 6. MISMATCHED TARGET VALUES FAIL CLOSED
# ============================================================


different_targets = (
    TARGETS.copy()
)


different_targets[
    -1
] += 1000.0


different_metrics = evaluate_forecast_predictions(
    predictions=
        lstm.predictions,

    targets=
        different_targets,
)


different_target_candidate = (
    build_time_series_forecast_candidate(
        estimator_key=
            "time_series_lstm_regressor",

        predictions=
            lstm.predictions,

        targets=
            different_targets,

        target_positions=
            POSITIONS,

        metrics=
            different_metrics,
    )
)


require_comparison_error(
    lambda:
        compare_time_series_forecast_candidates(
            candidates=
                (
                    baseline,
                    mlp,
                    rnn,
                    different_target_candidate,
                )
        ),
    label=
        "mismatched TEST target values",
)


print(
    "[PASS] mismatched TEST target values fail closed"
)


# ============================================================
# 7. INCOMPLETE FAMILY FAILS CLOSED
# ============================================================


require_comparison_error(
    lambda:
        compare_time_series_forecast_candidates(
            candidates=
                (
                    baseline,
                    mlp,
                    rnn,
                )
        ),
    label=
        "missing LSTM candidate",
)


print(
    "[PASS] incomplete four-model experiment fails closed"
)


# ============================================================
# 8. DUPLICATE ESTIMATOR FAILS CLOSED
# ============================================================


duplicate_mlp = candidate(
    "time_series_mlp_regressor",
    [
        114.5,
        115.5,
        116.5,
        117.5,
        118.5,
    ],
)


require_comparison_error(
    lambda:
        compare_time_series_forecast_candidates(
            candidates=
                (
                    baseline,
                    mlp,
                    duplicate_mlp,
                    lstm,
                )
        ),
    label=
        "duplicate estimator",
)


print(
    "[PASS] duplicate estimator identities fail closed"
)


# ============================================================
# 9. PROJECTIONS ARE IMMUTABLE / ISOLATED
# ============================================================


assert (
    baseline.predictions.flags.writeable
    is False
)


assert (
    baseline.targets.flags.writeable
    is False
)


original_prediction = float(
    baseline.predictions[
        0
    ]
)


external_predictions = np.array(
    [
        114.0,
        115.0,
        116.0,
        117.0,
        118.0,
    ],
    dtype=np.float64,
)


external_candidate = (
    build_time_series_forecast_candidate(
        estimator_key=
            ML_TIME_SERIES_NAIVE_ESTIMATOR_KEY,

        predictions=
            external_predictions,

        targets=
            TARGETS,

        target_positions=
            POSITIONS,

        metrics=
            baseline.metrics,
    )
)


external_predictions[
    0
] = 999999.0


assert (
    external_candidate.predictions[
        0
    ]
    ==
    original_prediction
)


print(
    "[PASS] comparison projections isolate their numeric evidence"
)


# ============================================================
# 10. TORCH-FREE / FRAMEWORK-NEUTRAL
# ============================================================


source_path = Path(
    "app/ml/time_series_comparison.py"
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


assert not any(
    name == "torch"
    or name.startswith("torch.")
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
    "[PASS] forecasting comparison core is torch-free and framework-neutral"
)


print()
print("=" * 80)
print("DL-5-A9-V1 FINAL VERDICT")
print("=" * 80)
print()

print("Complete four-candidate authority            PASS")
print("Exact shared TEST population                 PASS")
print("Exact shared target positions                PASS")
print("Exact shared target values                   PASS")
print("RMSE-first deterministic ranking             PASS")
print("No model-complexity preference               PASS")
print("Incomplete experiment guard                  PASS")
print("Duplicate estimator guard                    PASS")
print("Immutable evidence projections               PASS")
print("Torch-free comparison core                   PASS")

print()
print(
    "DL-5-A9-V1 - FORECAST COMPARISON CORE: PASS"
)
