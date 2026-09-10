from __future__ import annotations


from dataclasses import (
    dataclass,
)


import math
import numpy as np


from app.ml.time_series_windows import (
    MLTimeSeriesSupervisedWindows,
)


# ============================================================
# VERSION
# ============================================================


DL_TIME_SERIES_SCALING_RULE_VERSION = (
    "dl_time_series_scaling_v0.1"
)


# ============================================================
# ERRORS
# ============================================================


class DLTimeSeriesScalingError(
    RuntimeError
):
    pass


# ============================================================
# STANDARDIZER
# ============================================================


@dataclass(
    frozen=True
)
class DLTimeSeriesStandardizer:
    """
    Scalar standardizer fitted only from the historical TRAIN
    observation population.

    Because v0.1 forecasting is univariate, the same mean and
    standard deviation apply to:

    - context values;
    - regression targets;
    - predictions before inverse transformation.
    """

    mean: float

    standard_deviation: float

    fitted_observation_count: int

    rule_version: str = (
        DL_TIME_SERIES_SCALING_RULE_VERSION
    )


    def transform(
        self,
        values: object,
    ) -> np.ndarray:

        try:

            array = np.asarray(
                values,
                dtype=np.float64,
            )

        except Exception as error:

            raise (
                DLTimeSeriesScalingError(
                    (
                        "Time-series values could not be "
                        "converted to float64."
                    )
                )
            ) from error


        if not (
            np.isfinite(
                array
            )
            .all()
        ):

            raise (
                DLTimeSeriesScalingError(
                    (
                        "Time-series values contain "
                        "non-finite observations."
                    )
                )
            )


        transformed = (
            (
                array
                -
                self.mean
            )
            /
            self.standard_deviation
        )


        if not (
            np.isfinite(
                transformed
            )
            .all()
        ):

            raise (
                DLTimeSeriesScalingError(
                    (
                        "Time-series standardization produced "
                        "non-finite values."
                    )
                )
            )


        return np.array(
            transformed,
            dtype=np.float64,
            copy=True,
        )


    def inverse_transform(
        self,
        values: object,
    ) -> np.ndarray:

        try:

            array = np.asarray(
                values,
                dtype=np.float64,
            )

        except Exception as error:

            raise (
                DLTimeSeriesScalingError(
                    (
                        "Scaled time-series values could not "
                        "be converted to float64."
                    )
                )
            ) from error


        if not (
            np.isfinite(
                array
            )
            .all()
        ):

            raise (
                DLTimeSeriesScalingError(
                    (
                        "Scaled time-series values contain "
                        "non-finite observations."
                    )
                )
            )


        restored = (
            (
                array
                *
                self.standard_deviation
            )
            +
            self.mean
        )


        if not (
            np.isfinite(
                restored
            )
            .all()
        ):

            raise (
                DLTimeSeriesScalingError(
                    (
                        "Inverse time-series transformation "
                        "produced non-finite values."
                    )
                )
            )


        return np.array(
            restored,
            dtype=np.float64,
            copy=True,
        )


# ============================================================
# TRAIN POPULATION RECONSTRUCTION
# ============================================================


def _reconstruct_train_observations(
    *,
    windows: MLTimeSeriesSupervisedWindows,
) -> np.ndarray:

    if not isinstance(
        windows,
        MLTimeSeriesSupervisedWindows,
    ):

        raise (
            DLTimeSeriesScalingError(
                (
                    "Time-series scaling requires validated "
                    "supervised forecasting windows."
                )
            )
        )


    train = (
        windows.train
    )


    lookback = int(
        windows.lookback
    )


    if (
        train.sample_count
        < 1
        or
        lookback
        < 1
    ):

        raise (
            DLTimeSeriesScalingError(
                (
                    "Forecast TRAIN windows cannot be empty."
                )
            )
        )


    if (
        train.inputs.ndim
        !=
        2
        or
        train.inputs.shape[
            1
        ]
        !=
        lookback
    ):

        raise (
            DLTimeSeriesScalingError(
                (
                    "Forecast TRAIN input shape does not "
                    "match the configured lookback."
                )
            )
        )


    if (
        train.targets.ndim
        !=
        1
        or
        train.targets.shape[
            0
        ]
        !=
        train.sample_count
    ):

        raise (
            DLTimeSeriesScalingError(
                (
                    "Forecast TRAIN targets are not "
                    "sample-aligned."
                )
            )
        )


    expected_target_positions = tuple(
        range(
            lookback,
            lookback
            +
            train.sample_count,
        )
    )


    if (
        train.target_positions
        !=
        expected_target_positions
    ):

        raise (
            DLTimeSeriesScalingError(
                (
                    "Forecast TRAIN target positions do not "
                    "form the expected contiguous historical "
                    "population."
                )
            )
        )


    expected_first_context = tuple(
        range(
            lookback
        )
    )


    if (
        train.context_positions[
            0
        ]
        !=
        expected_first_context
    ):

        raise (
            DLTimeSeriesScalingError(
                (
                    "Forecast TRAIN first lookback context "
                    "does not start at the beginning of the "
                    "historical population."
                )
            )
        )


    first_context = np.asarray(
        train.inputs[
            0
        ],
        dtype=np.float64,
    )


    targets = np.asarray(
        train.targets,
        dtype=np.float64,
    )


    observations = np.concatenate(
        (
            first_context,
            targets,
        )
    )


    expected_count = (
        lookback
        +
        train.sample_count
    )


    if (
        observations.ndim
        !=
        1
        or
        observations.size
        !=
        expected_count
    ):

        raise (
            DLTimeSeriesScalingError(
                (
                    "Reconstructed TRAIN chronology has "
                    "an unexpected population size."
                )
            )
        )


    if not (
        np.isfinite(
            observations
        )
        .all()
    ):

        raise (
            DLTimeSeriesScalingError(
                (
                    "Reconstructed TRAIN chronology contains "
                    "non-finite observations."
                )
            )
        )


    return np.array(
        observations,
        dtype=np.float64,
        copy=True,
    )


# ============================================================
# PUBLIC FIT AUTHORITY
# ============================================================


def fit_time_series_train_standardizer(
    *,
    windows: MLTimeSeriesSupervisedWindows,
) -> DLTimeSeriesStandardizer:
    """
    Fit one scalar standardizer from each historical TRAIN
    observation exactly once.

    Overlapping window values are deliberately not counted
    repeatedly.

    No TEST observation participates in fitting.
    """

    observations = (
        _reconstruct_train_observations(
            windows=
                windows
        )
    )


    mean = float(
        np.mean(
            observations
        )
    )


    standard_deviation = float(
        np.std(
            observations,
            ddof=0,
        )
    )


    if not math.isfinite(
        mean
    ):

        raise (
            DLTimeSeriesScalingError(
                "TRAIN mean is non-finite."
            )
        )


    if (
        not math.isfinite(
            standard_deviation
        )
        or
        standard_deviation
        <=
        0.0
    ):

        raise (
            DLTimeSeriesScalingError(
                (
                    "TRAIN standard deviation must be "
                    "finite and greater than zero."
                )
            )
        )


    return (
        DLTimeSeriesStandardizer(
            mean=
                mean,

            standard_deviation=
                standard_deviation,

            fitted_observation_count=
                int(
                    observations.size
                ),
        )
    )
