from __future__ import annotations


from dataclasses import (
    dataclass,
)


import numpy as np


from app.ml.time_series_evaluation import (
    MLForecastRegressionMetrics,
    evaluate_forecast_predictions,
)


from app.ml.time_series_windows import (
    MLTimeSeriesSupervisedWindows,
    MLTimeSeriesWindowBatch,
)


# ============================================================
# VERSION
# ============================================================


ML_TIME_SERIES_NAIVE_BASELINE_RULE_VERSION = (
    "ml_time_series_naive_baseline_v0.1"
)


# ============================================================
# ERRORS
# ============================================================


class MLTimeSeriesNaiveBaselineError(
    RuntimeError
):
    pass


# ============================================================
# EVALUATION RESULT
# ============================================================


@dataclass(
    frozen=True
)
class MLTimeSeriesNaiveBaselineEvaluation:
    """
    Deterministic persistence / last-observation baseline.

    For every supervised sample:

        prediction = final value in the lookback context

    No fitting, learned parameter, seed or framework-specific
    runtime is involved.
    """

    predictions: np.ndarray

    targets: np.ndarray

    metrics: MLForecastRegressionMetrics

    target_positions: tuple[
        int,
        ...
    ]

    rule_version: str = (
        ML_TIME_SERIES_NAIVE_BASELINE_RULE_VERSION
    )


    @property
    def sample_count(
        self,
    ) -> int:

        return int(
            self.predictions.size
        )


# ============================================================
# BATCH VALIDATION
# ============================================================


def _validate_test_batch(
    *,
    batch: MLTimeSeriesWindowBatch,
) -> None:

    if not isinstance(
        batch,
        MLTimeSeriesWindowBatch,
    ):

        raise (
            MLTimeSeriesNaiveBaselineError(
                (
                    "Naive forecast baseline requires a "
                    "validated forecasting window batch."
                )
            )
        )


    if batch.inputs.ndim != 2:

        raise (
            MLTimeSeriesNaiveBaselineError(
                (
                    "Forecast baseline inputs must be "
                    "two-dimensional."
                )
            )
        )


    if batch.targets.ndim != 1:

        raise (
            MLTimeSeriesNaiveBaselineError(
                (
                    "Forecast baseline targets must be "
                    "one-dimensional."
                )
            )
        )


    if (
        batch.inputs.shape[
            0
        ]
        !=
        batch.targets.shape[
            0
        ]
    ):

        raise (
            MLTimeSeriesNaiveBaselineError(
                (
                    "Forecast baseline inputs and targets "
                    "are not sample-aligned."
                )
            )
        )


    if (
        batch.inputs.shape[
            0
        ]
        ==
        0
    ):

        raise (
            MLTimeSeriesNaiveBaselineError(
                "Forecast baseline TEST batch cannot be empty."
            )
        )


    if (
        batch.inputs.shape[
            1
        ]
        < 1
    ):

        raise (
            MLTimeSeriesNaiveBaselineError(
                (
                    "Forecast baseline requires at least "
                    "one historical value per sample."
                )
            )
        )


    if not (
        np.isfinite(
            batch.inputs
        )
        .all()
    ):

        raise (
            MLTimeSeriesNaiveBaselineError(
                (
                    "Forecast baseline inputs contain "
                    "non-finite values."
                )
            )
        )


    if not (
        np.isfinite(
            batch.targets
        )
        .all()
    ):

        raise (
            MLTimeSeriesNaiveBaselineError(
                (
                    "Forecast baseline targets contain "
                    "non-finite values."
                )
            )
        )


    if (
        len(
            batch.target_positions
        )
        !=
        batch.inputs.shape[
            0
        ]
    ):

        raise (
            MLTimeSeriesNaiveBaselineError(
                (
                    "Forecast baseline target-position "
                    "provenance is not sample-aligned."
                )
            )
        )


# ============================================================
# PUBLIC BASELINE AUTHORITY
# ============================================================


def evaluate_last_value_baseline(
    *,
    windows: MLTimeSeriesSupervisedWindows,
) -> MLTimeSeriesNaiveBaselineEvaluation:
    """
    Evaluate the deterministic one-step persistence baseline on
    exactly the TEST windows owned by the forecasting population.

    For each sample:

        y_hat(t) = final observed value in X(t)

    Under the v0.1 rolling-origin protocol, this corresponds to
    predicting that the next observation will equal the most
    recently observed one.
    """

    if not isinstance(
        windows,
        MLTimeSeriesSupervisedWindows,
    ):

        raise (
            MLTimeSeriesNaiveBaselineError(
                (
                    "Naive baseline requires validated "
                    "supervised forecasting windows."
                )
            )
        )


    batch = (
        windows.test
    )


    _validate_test_batch(
        batch=
            batch
    )


    predictions = (
        batch.inputs[
            :,
            -1,
        ]
        .astype(
            np.float64,
            copy=True,
        )
    )


    targets = (
        batch.targets
        .astype(
            np.float64,
            copy=True,
        )
    )


    metrics = (
        evaluate_forecast_predictions(
            predictions=
                predictions,

            targets=
                targets,
        )
    )


    predictions.setflags(
        write=False
    )


    targets.setflags(
        write=False
    )


    return (
        MLTimeSeriesNaiveBaselineEvaluation(
            predictions=
                predictions,

            targets=
                targets,

            metrics=
                metrics,

            target_positions=
                tuple(
                    int(
                        position
                    )
                    for position
                    in batch.target_positions
                ),
        )
    )
