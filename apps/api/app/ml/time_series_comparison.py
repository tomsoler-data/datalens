from __future__ import annotations


from dataclasses import (
    dataclass,
)


import math
import numpy as np


from app.ml.time_series_evaluation import (
    MLForecastRegressionMetrics,
)


# ============================================================
# VERSION
# ============================================================


ML_TIME_SERIES_COMPARISON_RULE_VERSION = (
    "ml_time_series_comparison_v0.1"
)


# ============================================================
# REQUIRED CANDIDATES
# ============================================================


ML_TIME_SERIES_NAIVE_ESTIMATOR_KEY = (
    "naive_last_value"
)


ML_TIME_SERIES_REQUIRED_ESTIMATOR_KEYS = (
    ML_TIME_SERIES_NAIVE_ESTIMATOR_KEY,
    "time_series_mlp_regressor",
    "time_series_rnn_regressor",
    "time_series_lstm_regressor",
)


ML_TIME_SERIES_RANKING_KEYS = (
    "rmse:asc",
    "mae:asc",
    "absolute_mean_error:asc",
    "estimator_key:asc",
)


# ============================================================
# ERRORS
# ============================================================


class MLTimeSeriesComparisonError(
    RuntimeError
):
    pass


# ============================================================
# CANDIDATE
# ============================================================


@dataclass(
    frozen=True
)
class MLTimeSeriesForecastCandidate:
    """
    Framework-neutral projection of one already-evaluated
    forecasting candidate.

    PyTorch models, learned state and training objects are
    deliberately absent.

    Comparison owns only:

    - candidate identity;
    - exact TEST population identity;
    - predictions;
    - targets;
    - shared forecasting metrics.
    """

    estimator_key: str

    predictions: np.ndarray

    targets: np.ndarray

    target_positions: tuple[
        int,
        ...
    ]

    metrics: MLForecastRegressionMetrics


    @property
    def sample_count(
        self,
    ) -> int:

        return int(
            self.targets.size
        )


# ============================================================
# COMPARISON RESULT
# ============================================================


@dataclass(
    frozen=True
)
class MLTimeSeriesForecastComparison:
    """
    Deterministically ranked common forecasting experiment.

    Lower RMSE wins.

    Ties are resolved by:

    1. lower MAE;
    2. lower absolute mean signed error;
    3. estimator key lexical order.

    This is ranking, not automatic promotion.
    """

    ranked_candidates: tuple[
        MLTimeSeriesForecastCandidate,
        ...
    ]

    ranking_keys: tuple[
        str,
        ...
    ] = (
        ML_TIME_SERIES_RANKING_KEYS
    )

    rule_version: str = (
        ML_TIME_SERIES_COMPARISON_RULE_VERSION
    )


    @property
    def winner(
        self,
    ) -> MLTimeSeriesForecastCandidate:

        if not self.ranked_candidates:

            raise MLTimeSeriesComparisonError(
                (
                    "Forecast comparison contains "
                    "no ranked candidates."
                )
            )


        return self.ranked_candidates[
            0
        ]


# ============================================================
# INTERNAL VECTOR AUTHORITY
# ============================================================


def _validated_vector(
    values: object,
    *,
    label: str,
) -> np.ndarray:

    try:

        array = np.asarray(
            values,
            dtype=np.float64,
        )

    except Exception as error:

        raise MLTimeSeriesComparisonError(
            (
                f"{label} could not be converted "
                "to float64."
            )
        ) from error


    if array.ndim != 1:

        raise MLTimeSeriesComparisonError(
            f"{label} must be one-dimensional."
        )


    if array.size == 0:

        raise MLTimeSeriesComparisonError(
            f"{label} cannot be empty."
        )


    if not np.isfinite(
        array
    ).all():

        raise MLTimeSeriesComparisonError(
            f"{label} contains non-finite values."
        )


    result = array.copy()

    result.setflags(
        write=False
    )


    return result


# ============================================================
# CANDIDATE PROJECTION
# ============================================================


def build_time_series_forecast_candidate(
    *,
    estimator_key: str,
    predictions: object,
    targets: object,
    target_positions: tuple[
        int,
        ...
    ],
    metrics: MLForecastRegressionMetrics,
) -> MLTimeSeriesForecastCandidate:
    """
    Project one framework-specific execution result into the
    framework-neutral comparison plane.

    Metrics are accepted only when they describe exactly the
    supplied aligned prediction/target population.
    """

    normalized_key = str(
        estimator_key
    ).strip()


    if not normalized_key:

        raise MLTimeSeriesComparisonError(
            "Forecast estimator key cannot be empty."
        )


    prediction_values = _validated_vector(
        predictions,
        label="Forecast candidate predictions",
    )


    target_values = _validated_vector(
        targets,
        label="Forecast candidate targets",
    )


    if (
        prediction_values.shape
        !=
        target_values.shape
    ):

        raise MLTimeSeriesComparisonError(
            (
                "Forecast candidate predictions and targets "
                "must have exactly the same shape."
            )
        )


    if not isinstance(
        metrics,
        MLForecastRegressionMetrics,
    ):

        raise MLTimeSeriesComparisonError(
            (
                "Forecast candidate metrics must use "
                "the shared forecasting metric authority."
            )
        )


    if (
        metrics.sample_count
        !=
        target_values.size
    ):

        raise MLTimeSeriesComparisonError(
            (
                "Forecast candidate metric sample count "
                "does not match TEST population."
            )
        )


    if (
        len(
            target_positions
        )
        !=
        target_values.size
    ):

        raise MLTimeSeriesComparisonError(
            (
                "Forecast candidate target positions "
                "do not match TEST population."
            )
        )


    normalized_positions = tuple(
        int(
            position
        )
        for position
        in target_positions
    )


    if (
        len(
            set(
                normalized_positions
            )
        )
        !=
        len(
            normalized_positions
        )
    ):

        raise MLTimeSeriesComparisonError(
            (
                "Forecast candidate target positions "
                "must be unique."
            )
        )


    if (
        tuple(
            sorted(
                normalized_positions
            )
        )
        !=
        normalized_positions
    ):

        raise MLTimeSeriesComparisonError(
            (
                "Forecast candidate target positions "
                "must be chronologically ordered."
            )
        )


    for metric in (
        metrics.mae,
        metrics.mse,
        metrics.rmse,
        metrics.mean_error,
    ):

        if not math.isfinite(
            float(
                metric
            )
        ):

            raise MLTimeSeriesComparisonError(
                (
                    "Forecast candidate contains "
                    "non-finite metrics."
                )
            )


    return MLTimeSeriesForecastCandidate(
        estimator_key=
            normalized_key,

        predictions=
            prediction_values,

        targets=
            target_values,

        target_positions=
            normalized_positions,

        metrics=
            metrics,
    )


# ============================================================
# RANKING
# ============================================================


def _ranking_key(
    candidate: MLTimeSeriesForecastCandidate,
) -> tuple[
    float,
    float,
    float,
    str,
]:

    return (
        float(
            candidate.metrics.rmse
        ),

        float(
            candidate.metrics.mae
        ),

        abs(
            float(
                candidate.metrics.mean_error
            )
        ),

        candidate.estimator_key,
    )


# ============================================================
# PUBLIC COMPARISON AUTHORITY
# ============================================================


def compare_time_series_forecast_candidates(
    *,
    candidates: tuple[
        MLTimeSeriesForecastCandidate,
        ...
    ],
) -> MLTimeSeriesForecastComparison:
    """
    Validate and rank the complete DL-5 v0.1 candidate family.

    Every candidate must use exactly the same TEST:

    - target positions;
    - target values;
    - sample count.

    Predictions may differ.

    Model complexity never influences ranking.
    """

    if len(
        candidates
    ) != len(
        ML_TIME_SERIES_REQUIRED_ESTIMATOR_KEYS
    ):

        raise MLTimeSeriesComparisonError(
            (
                "DL-5 v0.1 comparison requires exactly "
                "four forecasting candidates."
            )
        )


    estimator_keys = tuple(
        candidate.estimator_key
        for candidate
        in candidates
    )


    if (
        len(
            set(
                estimator_keys
            )
        )
        !=
        len(
            estimator_keys
        )
    ):

        raise MLTimeSeriesComparisonError(
            (
                "Forecast comparison estimator keys "
                "must be unique."
            )
        )


    if (
        set(
            estimator_keys
        )
        !=
        set(
            ML_TIME_SERIES_REQUIRED_ESTIMATOR_KEYS
        )
    ):

        raise MLTimeSeriesComparisonError(
            (
                "Forecast comparison candidate family "
                "does not match DL-5 v0.1 authority."
            )
        )


    reference = candidates[
        0
    ]


    for candidate in candidates[
        1:
    ]:

        if (
            candidate.sample_count
            !=
            reference.sample_count
        ):

            raise MLTimeSeriesComparisonError(
                (
                    "Forecast candidates do not share "
                    "the same TEST sample count."
                )
            )


        if (
            candidate.target_positions
            !=
            reference.target_positions
        ):

            raise MLTimeSeriesComparisonError(
                (
                    "Forecast candidates do not share "
                    "the exact TEST target positions."
                )
            )


        if not np.array_equal(
            candidate.targets,
            reference.targets,
        ):

            raise MLTimeSeriesComparisonError(
                (
                    "Forecast candidates do not share "
                    "the exact TEST targets."
                )
            )


    ranked = tuple(
        sorted(
            candidates,
            key=_ranking_key,
        )
    )


    return MLTimeSeriesForecastComparison(
        ranked_candidates=
            ranked
    )
