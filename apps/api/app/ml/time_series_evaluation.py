from __future__ import annotations


from dataclasses import (
    dataclass,
)


import math
import numpy as np


# ============================================================
# VERSION
# ============================================================


ML_TIME_SERIES_EVALUATION_RULE_VERSION = (
    "ml_time_series_evaluation_v0.1"
)


# ============================================================
# ERRORS
# ============================================================


class MLTimeSeriesEvaluationError(
    RuntimeError
):
    pass


# ============================================================
# METRICS
# ============================================================


@dataclass(
    frozen=True
)
class MLForecastRegressionMetrics:
    """
    Shared deterministic regression metrics for forecasting
    candidates.

    The same authority is intended to evaluate:

    - naive baseline;
    - temporal MLP;
    - RNN;
    - LSTM.

    No model-specific behavior belongs here.
    """

    sample_count: int

    mae: float

    mse: float

    rmse: float

    mean_error: float

    rule_version: str = (
        ML_TIME_SERIES_EVALUATION_RULE_VERSION
    )


# ============================================================
# ARRAY VALIDATION
# ============================================================


def _validated_float_vector(
    *,
    values: object,
    label: str,
) -> np.ndarray:

    try:

        array = np.asarray(
            values,
            dtype=np.float64,
        )

    except Exception as error:

        raise (
            MLTimeSeriesEvaluationError(
                (
                    f"{label} could not be converted "
                    "to a float64 vector."
                )
            )
        ) from error


    if array.ndim != 1:

        raise (
            MLTimeSeriesEvaluationError(
                (
                    f"{label} must be exactly "
                    "one-dimensional."
                )
            )
        )


    if array.size == 0:

        raise (
            MLTimeSeriesEvaluationError(
                (
                    f"{label} cannot be empty."
                )
            )
        )


    if not (
        np.isfinite(
            array
        )
        .all()
    ):

        raise (
            MLTimeSeriesEvaluationError(
                (
                    f"{label} contains non-finite values."
                )
            )
        )


    return (
        array.copy()
    )


# ============================================================
# PUBLIC EVALUATION AUTHORITY
# ============================================================


def evaluate_forecast_predictions(
    *,
    predictions: object,
    targets: object,
) -> MLForecastRegressionMetrics:
    """
    Evaluate one forecast prediction vector against the exact
    aligned target population.

    Metrics:

        MAE
        MSE
        RMSE
        mean signed error

    The function performs no fitting and no mutation.
    """

    prediction_values = (
        _validated_float_vector(
            values=
                predictions,

            label=
                "Forecast predictions",
        )
    )


    target_values = (
        _validated_float_vector(
            values=
                targets,

            label=
                "Forecast targets",
        )
    )


    if (
        prediction_values.shape
        !=
        target_values.shape
    ):

        raise (
            MLTimeSeriesEvaluationError(
                (
                    "Forecast predictions and targets "
                    "must have exactly the same shape. "
                    f"predictions_shape="
                    f"{prediction_values.shape}, "
                    f"targets_shape="
                    f"{target_values.shape}"
                )
            )
        )


    errors = (
        prediction_values
        -
        target_values
    )


    absolute_errors = (
        np.abs(
            errors
        )
    )


    squared_errors = (
        np.square(
            errors
        )
    )


    mae = float(
        np.mean(
            absolute_errors
        )
    )


    mse = float(
        np.mean(
            squared_errors
        )
    )


    rmse = float(
        math.sqrt(
            mse
        )
    )


    mean_error = float(
        np.mean(
            errors
        )
    )


    metrics = (
        mae,
        mse,
        rmse,
        mean_error,
    )


    if not all(
        math.isfinite(
            metric
        )
        for metric
        in metrics
    ):

        raise (
            MLTimeSeriesEvaluationError(
                (
                    "Forecast metric calculation produced "
                    "a non-finite result."
                )
            )
        )


    return (
        MLForecastRegressionMetrics(
            sample_count=
                int(
                    target_values.size
                ),

            mae=
                mae,

            mse=
                mse,

            rmse=
                rmse,

            mean_error=
                mean_error,
        )
    )
