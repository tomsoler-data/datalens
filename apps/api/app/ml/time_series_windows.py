from __future__ import annotations


from dataclasses import (
    dataclass,
)


import numpy as np
import pandas as pd


from app.ml.time_series_contracts import (
    MLTimeSeriesForecastingContract,
)


from app.ml.time_series_holdout import (
    MLTimeSeriesHoldoutPartition,
)


from app.ml.time_series_training_input import (
    MLValidatedUnivariateTimeSeries,
)


# ============================================================
# VERSION / POLICY
# ============================================================


ML_TIME_SERIES_WINDOWS_RULE_VERSION = (
    "ml_time_series_windows_v0.1"
)


ML_TIME_SERIES_EVALUATION_PROTOCOL = (
    "one_step_rolling_origin_observed_history"
)


# ============================================================
# ERRORS
# ============================================================


class MLTimeSeriesWindowError(
    RuntimeError
):
    pass


# ============================================================
# WINDOW BATCH
# ============================================================


@dataclass(
    frozen=True
)
class MLTimeSeriesWindowBatch:
    """
    One immutable TRAIN or TEST forecasting sample population.

    inputs has shape:

        samples x lookback

    targets has shape:

        samples

    Every context position is strictly earlier than its target
    position.

    source-row metadata maps canonical chronological positions
    back to positional rows of the validated Preparation input.
    """

    inputs: np.ndarray

    targets: np.ndarray

    context_positions: tuple[
        tuple[
            int,
            ...
        ],
        ...
    ]

    target_positions: tuple[
        int,
        ...
    ]

    context_source_row_positions: tuple[
        tuple[
            int,
            ...
        ],
        ...
    ]

    target_source_row_positions: tuple[
        int,
        ...
    ]

    target_times: tuple[
        pd.Timestamp,
        ...
    ]


    @property
    def sample_count(
        self,
    ) -> int:

        return int(
            self.inputs.shape[
                0
            ]
        )


    @property
    def lookback(
        self,
    ) -> int:

        return int(
            self.inputs.shape[
                1
            ]
        )


# ============================================================
# COMPLETE SUPERVISED POPULATION
# ============================================================


@dataclass(
    frozen=True
)
class MLTimeSeriesSupervisedWindows:
    """
    Framework-neutral supervised forecasting population.

    The exact same TRAIN and TEST examples are intended to be
    consumed by:

    - deterministic naive baseline;
    - temporal MLP;
    - RNN;
    - LSTM.

    v0.1 uses one-step rolling-origin evaluation with observed
    history.

    Therefore a TEST prediction may use earlier TEST observations
    as context once those observations are chronologically in the
    past.

    It may never use its own target or any future observation.
    """

    train: MLTimeSeriesWindowBatch

    test: MLTimeSeriesWindowBatch

    lookback: int

    forecast_horizon: int = 1

    evaluation_protocol: str = (
        ML_TIME_SERIES_EVALUATION_PROTOCOL
    )

    rule_version: str = (
        ML_TIME_SERIES_WINDOWS_RULE_VERSION
    )


# ============================================================
# INPUT / PARTITION INTEGRITY
# ============================================================


def _validate_window_inputs(
    *,
    series: MLValidatedUnivariateTimeSeries,
    partition: MLTimeSeriesHoldoutPartition,
    contract: MLTimeSeriesForecastingContract,
) -> None:

    if not isinstance(
        series,
        MLValidatedUnivariateTimeSeries,
    ):

        raise (
            MLTimeSeriesWindowError(
                (
                    "Forecast windows require a validated "
                    "univariate time-series input."
                )
            )
        )


    if not isinstance(
        partition,
        MLTimeSeriesHoldoutPartition,
    ):

        raise (
            MLTimeSeriesWindowError(
                (
                    "Forecast windows require a validated "
                    "time-series holdout partition."
                )
            )
        )


    row_count = int(
        series.row_count
    )


    if (
        partition.source_row_count
        !=
        row_count
    ):

        raise (
            MLTimeSeriesWindowError(
                (
                    "Forecast series and holdout population "
                    "sizes do not match."
                )
            )
        )


    if (
        len(
            series.time_values
        )
        !=
        row_count
        or
        len(
            series.target_values
        )
        !=
        row_count
        or
        len(
            series.source_row_positions
        )
        !=
        row_count
    ):

        raise (
            MLTimeSeriesWindowError(
                (
                    "Forecast series components are "
                    "not row-aligned."
                )
            )
        )


    if not (
        series.time_values
        .is_monotonic_increasing
    ):

        raise (
            MLTimeSeriesWindowError(
                (
                    "Forecast series must remain in "
                    "canonical chronological order."
                )
            )
        )


    if bool(
        series.time_values
        .duplicated(
            keep=False
        )
        .any()
    ):

        raise (
            MLTimeSeriesWindowError(
                (
                    "Forecast series contains duplicate "
                    "timestamps."
                )
            )
        )


    train_positions = (
        partition.train_positions
    )


    test_positions = (
        partition.test_positions
    )


    if (
        not train_positions
        or
        not test_positions
    ):

        raise (
            MLTimeSeriesWindowError(
                (
                    "Forecast holdout must expose non-empty "
                    "TRAIN and TEST populations."
                )
            )
        )


    cut_position = int(
        test_positions[
            0
        ]
    )


    expected_train_positions = tuple(
        range(
            cut_position
        )
    )


    expected_test_positions = tuple(
        range(
            cut_position,
            row_count,
        )
    )


    if (
        train_positions
        !=
        expected_train_positions
        or
        test_positions
        !=
        expected_test_positions
    ):

        raise (
            MLTimeSeriesWindowError(
                (
                    "Forecast holdout must remain an exact "
                    "historical-prefix / future-suffix "
                    "partition."
                )
            )
        )


    minimum_train_rows = (
        int(
            contract.lookback
        )
        +
        1
    )


    if (
        len(
            train_positions
        )
        <
        minimum_train_rows
    ):

        raise (
            MLTimeSeriesWindowError(
                (
                    "Forecast TRAIN population cannot form "
                    "a supervised lookback sample."
                )
            )
        )


    source_positions = np.asarray(
        series.source_row_positions,
        dtype=np.int64,
    )


    expected_train_source = tuple(
        int(
            source_positions[
                position
            ]
        )
        for position
        in train_positions
    )


    expected_test_source = tuple(
        int(
            source_positions[
                position
            ]
        )
        for position
        in test_positions
    )


    if (
        partition.train_source_row_positions
        !=
        expected_train_source
        or
        partition.test_source_row_positions
        !=
        expected_test_source
    ):

        raise (
            MLTimeSeriesWindowError(
                (
                    "Forecast holdout source-row provenance "
                    "does not match the validated series."
                )
            )
        )


    actual_train_end = pd.Timestamp(
        series.time_values.iloc[
            train_positions[
                -1
            ]
        ]
    )


    actual_test_start = pd.Timestamp(
        series.time_values.iloc[
            test_positions[
                0
            ]
        ]
    )


    if (
        actual_train_end
        !=
        partition.train_end_time
        or
        actual_test_start
        !=
        partition.test_start_time
    ):

        raise (
            MLTimeSeriesWindowError(
                (
                    "Forecast holdout temporal-boundary "
                    "metadata does not match the series."
                )
            )
        )


    if not (
        actual_train_end
        <
        actual_test_start
    ):

        raise (
            MLTimeSeriesWindowError(
                (
                    "Forecast TRAIN/TEST chronology is "
                    "not strictly ordered."
                )
            )
        )


# ============================================================
# BATCH CONSTRUCTION
# ============================================================


def _build_window_batch(
    *,
    target_positions: tuple[
        int,
        ...
    ],
    target_values: np.ndarray,
    time_values: pd.Series,
    source_row_positions: np.ndarray,
    lookback: int,
) -> MLTimeSeriesWindowBatch:

    if not target_positions:

        raise (
            MLTimeSeriesWindowError(
                "Forecast window batch cannot be empty."
            )
        )


    inputs = np.empty(
        (
            len(
                target_positions
            ),
            lookback,
        ),
        dtype=np.float64,
    )


    targets = np.empty(
        (
            len(
                target_positions
            ),
        ),
        dtype=np.float64,
    )


    context_positions = []

    context_source_positions = []

    target_source_positions = []

    target_times = []


    for (
        sample_index,
        target_position,
    ) in enumerate(
        target_positions
    ):

        context_start = (
            int(
                target_position
            )
            -
            int(
                lookback
            )
        )


        if context_start < 0:

            raise (
                MLTimeSeriesWindowError(
                    (
                        "Forecast target does not have "
                        "enough historical observations "
                        "for the configured lookback."
                    )
                )
            )


        context = tuple(
            range(
                context_start,
                int(
                    target_position
                ),
            )
        )


        if (
            len(
                context
            )
            !=
            lookback
        ):

            raise (
                MLTimeSeriesWindowError(
                    (
                        "Forecast context length does not "
                        "match configured lookback."
                    )
                )
            )


        if (
            not context
            or
            max(
                context
            )
            >=
            int(
                target_position
            )
        ):

            raise (
                MLTimeSeriesWindowError(
                    (
                        "Forecast context contains the "
                        "current target or a future row."
                    )
                )
            )


        inputs[
            sample_index,
            :,
        ] = target_values[
            list(
                context
            )
        ]


        targets[
            sample_index
        ] = target_values[
            int(
                target_position
            )
        ]


        context_positions.append(
            context
        )


        context_source_positions.append(
            tuple(
                int(
                    source_row_positions[
                        position
                    ]
                )
                for position
                in context
            )
        )


        target_source_positions.append(
            int(
                source_row_positions[
                    int(
                        target_position
                    )
                ]
            )
        )


        target_times.append(
            pd.Timestamp(
                time_values.iloc[
                    int(
                        target_position
                    )
                ]
            )
        )


    inputs.setflags(
        write=False
    )


    targets.setflags(
        write=False
    )


    return (
        MLTimeSeriesWindowBatch(
            inputs=
                inputs,

            targets=
                targets,

            context_positions=
                tuple(
                    context_positions
                ),

            target_positions=
                tuple(
                    int(
                        position
                    )
                    for position
                    in target_positions
                ),

            context_source_row_positions=
                tuple(
                    context_source_positions
                ),

            target_source_row_positions=
                tuple(
                    target_source_positions
                ),

            target_times=
                tuple(
                    target_times
                ),
        )
    )


# ============================================================
# PUBLIC WINDOW AUTHORITY
# ============================================================


def build_time_series_supervised_windows(
    *,
    series: MLValidatedUnivariateTimeSeries,
    partition: MLTimeSeriesHoldoutPartition,
    contract: MLTimeSeriesForecastingContract,
) -> MLTimeSeriesSupervisedWindows:
    """
    Build deterministic supervised forecasting examples.

    TRAIN:
        every target inside the historical TRAIN population that
        has exactly lookback preceding observations.

    TEST:
        every future TEST target.

    TEST evaluation protocol:
        one-step rolling origin with observed history.

    For target t:

        X_t = [y_(t-lookback), ..., y_(t-1)]
        y_t = y_t

    No target or future value can enter X_t.
    """

    contract = (
        MLTimeSeriesForecastingContract
        .model_validate(
            contract
        )
    )


    _validate_window_inputs(
        series=
            series,

        partition=
            partition,

        contract=
            contract,
    )


    lookback = int(
        contract.lookback
    )


    target_values = (
        series.target_values
        .to_numpy(
            dtype=np.float64,
            copy=True,
        )
    )


    if not (
        np.isfinite(
            target_values
        )
        .all()
    ):

        raise (
            MLTimeSeriesWindowError(
                (
                    "Forecast target population contains "
                    "non-finite values."
                )
            )
        )


    source_row_positions = np.asarray(
        series.source_row_positions,
        dtype=np.int64,
    )


    train_target_positions = tuple(
        position
        for position
        in partition.train_positions
        if position >= lookback
    )


    test_target_positions = tuple(
        int(
            position
        )
        for position
        in partition.test_positions
    )


    if not train_target_positions:

        raise (
            MLTimeSeriesWindowError(
                (
                    "Forecast TRAIN population produced "
                    "no supervised samples."
                )
            )
        )


    if not test_target_positions:

        raise (
            MLTimeSeriesWindowError(
                (
                    "Forecast TEST population produced "
                    "no supervised samples."
                )
            )
        )


    train_batch = (
        _build_window_batch(
            target_positions=
                train_target_positions,

            target_values=
                target_values,

            time_values=
                series.time_values,

            source_row_positions=
                source_row_positions,

            lookback=
                lookback,
        )
    )


    test_batch = (
        _build_window_batch(
            target_positions=
                test_target_positions,

            target_values=
                target_values,

            time_values=
                series.time_values,

            source_row_positions=
                source_row_positions,

            lookback=
                lookback,
        )
    )


    return (
        MLTimeSeriesSupervisedWindows(
            train=
                train_batch,

            test=
                test_batch,

            lookback=
                lookback,

            forecast_horizon=
                int(
                    contract.forecast_horizon
                ),
        )
    )
