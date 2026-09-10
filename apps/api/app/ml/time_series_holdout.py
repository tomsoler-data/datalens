from __future__ import annotations


from dataclasses import (
    dataclass,
)


import numpy as np
import pandas as pd


from app.ml.splitting import (
    MLSplitError,
    resolve_ml_feature_holdout_partition,
)


from app.ml.time_series_contracts import (
    MLTimeSeriesForecastingContract,
)


from app.ml.time_series_training_input import (
    MLValidatedUnivariateTimeSeries,
)


# ============================================================
# VERSION
# ============================================================


ML_TIME_SERIES_HOLDOUT_RULE_VERSION = (
    "ml_time_series_holdout_v0.1"
)


# ============================================================
# ERRORS
# ============================================================


class MLTimeSeriesHoldoutError(
    RuntimeError
):
    pass


# ============================================================
# PARTITION
# ============================================================


@dataclass(
    frozen=True
)
class MLTimeSeriesHoldoutPartition:
    """
    Exact forecasting observation positions assigned to the
    chronological outer TRAIN and TEST populations.

    Positions are relative to the canonical chronological series.

    source-row positions preserve the positional identity of the
    validated Preparation dataframe before chronological sorting.

    No forecasting windows are created here.
    """

    source_row_count: int

    train_positions: tuple[
        int,
        ...
    ]

    test_positions: tuple[
        int,
        ...
    ]

    train_source_row_positions: tuple[
        int,
        ...
    ]

    test_source_row_positions: tuple[
        int,
        ...
    ]

    train_end_time: pd.Timestamp

    test_start_time: pd.Timestamp

    rule_version: str = (
        ML_TIME_SERIES_HOLDOUT_RULE_VERSION
    )


    @property
    def train_rows(
        self,
    ) -> int:

        return len(
            self.train_positions
        )


    @property
    def test_rows(
        self,
    ) -> int:

        return len(
            self.test_positions
        )


# ============================================================
# SERIES INTEGRITY
# ============================================================


def _validate_series_integrity(
    *,
    series: MLValidatedUnivariateTimeSeries,
) -> None:

    if not isinstance(
        series,
        MLValidatedUnivariateTimeSeries,
    ):

        raise (
            MLTimeSeriesHoldoutError(
                (
                    "Forecast holdout requires a validated "
                    "univariate time-series input."
                )
            )
        )


    row_count = int(
        series.row_count
    )


    if row_count < 2:

        raise (
            MLTimeSeriesHoldoutError(
                (
                    "Forecast holdout requires at least "
                    "two validated observations."
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
            MLTimeSeriesHoldoutError(
                (
                    "Validated forecasting series components "
                    "are not row-aligned."
                )
            )
        )


    if not (
        pd.api.types
        .is_datetime64_any_dtype(
            series.time_values.dtype
        )
    ):

        raise (
            MLTimeSeriesHoldoutError(
                (
                    "Validated forecasting time values "
                    "must remain pandas datetime data."
                )
            )
        )


    if bool(
        series.time_values
        .isna()
        .any()
    ):

        raise (
            MLTimeSeriesHoldoutError(
                (
                    "Validated forecasting time values "
                    "contain missing observations."
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
            MLTimeSeriesHoldoutError(
                (
                    "Validated forecasting time values "
                    "contain duplicate timestamps."
                )
            )
        )


    if not (
        series.time_values
        .is_monotonic_increasing
    ):

        raise (
            MLTimeSeriesHoldoutError(
                (
                    "Validated forecasting series is not "
                    "in canonical chronological order."
                )
            )
        )


    try:

        numeric_target = (
            series.target_values
            .to_numpy(
                dtype=np.float64,
                copy=True,
            )
        )

    except Exception as error:

        raise (
            MLTimeSeriesHoldoutError(
                (
                    "Validated forecasting target values "
                    "are not numeric."
                )
            )
        ) from error


    if not (
        np.isfinite(
            numeric_target
        )
        .all()
    ):

        raise (
            MLTimeSeriesHoldoutError(
                (
                    "Validated forecasting target values "
                    "contain non-finite observations."
                )
            )
        )


    source_positions = (
        np.asarray(
            series.source_row_positions,
            dtype=np.int64,
        )
    )


    if (
        source_positions.ndim
        !=
        1
        or
        len(
            source_positions
        )
        !=
        row_count
    ):

        raise (
            MLTimeSeriesHoldoutError(
                (
                    "Forecasting source-row position "
                    "provenance is malformed."
                )
            )
        )


    expected_source_positions = (
        np.arange(
            row_count,
            dtype=np.int64,
        )
    )


    if not np.array_equal(
        np.sort(
            source_positions
        ),
        expected_source_positions,
    ):

        raise (
            MLTimeSeriesHoldoutError(
                (
                    "Forecasting source-row positions "
                    "must form an exact permutation of "
                    "the validated source population."
                )
            )
        )


# ============================================================
# HOLDOUT AUTHORITY
# ============================================================


def resolve_time_series_holdout(
    *,
    series: MLValidatedUnivariateTimeSeries,
    contract: MLTimeSeriesForecastingContract,
) -> MLTimeSeriesHoldoutPartition:
    """
    Resolve the forecasting outer holdout through the existing
    shared Model Lab chronological split authority.

    The target values are deliberately not supplied to the split
    resolver.

    Therefore TRAIN / TEST assignment depends only on:

    - the server-validated time axis;
    - the forecasting split contract.

    v0.1 also requires enough TRAIN observations to create at
    least one later lookback window.
    """

    contract = (
        MLTimeSeriesForecastingContract
        .model_validate(
            contract
        )
    )


    _validate_series_integrity(
        series=series,
    )


    row_count = int(
        series.row_count
    )


    canonical_dataframe = pd.DataFrame(
        {
            contract.time_column:
                series.time_values
                .copy(
                    deep=True
                )
                .reset_index(
                    drop=True
                )
        }
    )


    alignment_features = pd.DataFrame(
        index=canonical_dataframe.index
    )


    try:

        shared_partition = (
            resolve_ml_feature_holdout_partition(
                x=
                    alignment_features,

                dataframe=
                    canonical_dataframe,

                contract=
                    contract,
            )
        )

    except MLSplitError as error:

        raise (
            MLTimeSeriesHoldoutError(
                str(
                    error
                )
            )
        ) from error


    if (
        shared_partition.source_row_count
        !=
        row_count
    ):

        raise (
            MLTimeSeriesHoldoutError(
                (
                    "Shared chronological holdout returned "
                    "an unexpected source population size."
                )
            )
        )


    if shared_partition.purged_positions:

        raise (
            MLTimeSeriesHoldoutError(
                (
                    "Univariate forecasting time_holdout "
                    "must not produce purged observations."
                )
            )
        )


    train_positions = tuple(
        int(
            position
        )
        for position
        in shared_partition.train_positions
    )


    test_positions = tuple(
        int(
            position
        )
        for position
        in shared_partition.test_positions
    )


    if (
        tuple(
            sorted(
                train_positions
            )
        )
        !=
        train_positions
        or
        tuple(
            sorted(
                test_positions
            )
        )
        !=
        test_positions
    ):

        raise (
            MLTimeSeriesHoldoutError(
                (
                    "Forecasting chronological positions "
                    "must remain monotonically ordered."
                )
            )
        )


    if (
        set(
            train_positions
        )
        &
        set(
            test_positions
        )
    ):

        raise (
            MLTimeSeriesHoldoutError(
                (
                    "Forecasting TRAIN and TEST positions "
                    "must not overlap."
                )
            )
        )


    covered_positions = tuple(
        sorted(
            (
                *train_positions,
                *test_positions,
            )
        )
    )


    if (
        covered_positions
        !=
        tuple(
            range(
                row_count
            )
        )
    ):

        raise (
            MLTimeSeriesHoldoutError(
                (
                    "Forecasting holdout must cover the "
                    "entire validated observation population."
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
            MLTimeSeriesHoldoutError(
                (
                    "Forecasting TRAIN population is too "
                    "short for the configured lookback. "
                    f"train_rows={len(train_positions)}, "
                    f"lookback={contract.lookback}, "
                    "minimum_train_rows="
                    f"{minimum_train_rows}"
                )
            )
        )


    train_times = (
        series.time_values.iloc[
            list(
                train_positions
            )
        ]
    )


    test_times = (
        series.time_values.iloc[
            list(
                test_positions
            )
        ]
    )


    train_end_time = pd.Timestamp(
        train_times.max()
    )


    test_start_time = pd.Timestamp(
        test_times.min()
    )


    if not (
        train_end_time
        <
        test_start_time
    ):

        raise (
            MLTimeSeriesHoldoutError(
                (
                    "Forecasting holdout violated the "
                    "strict chronological boundary."
                )
            )
        )


    source_positions = (
        np.asarray(
            series.source_row_positions,
            dtype=np.int64,
        )
    )


    train_source_row_positions = tuple(
        int(
            source_positions[
                position
            ]
        )
        for position
        in train_positions
    )


    test_source_row_positions = tuple(
        int(
            source_positions[
                position
            ]
        )
        for position
        in test_positions
    )


    if (
        set(
            train_source_row_positions
        )
        &
        set(
            test_source_row_positions
        )
    ):

        raise (
            MLTimeSeriesHoldoutError(
                (
                    "Forecasting source-row provenance "
                    "overlaps across TRAIN and TEST."
                )
            )
        )


    return (
        MLTimeSeriesHoldoutPartition(
            source_row_count=
                row_count,

            train_positions=
                train_positions,

            test_positions=
                test_positions,

            train_source_row_positions=
                train_source_row_positions,

            test_source_row_positions=
                test_source_row_positions,

            train_end_time=
                train_end_time,

            test_start_time=
                test_start_time,
        )
    )
