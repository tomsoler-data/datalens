from __future__ import annotations


from dataclasses import (
    dataclass,
)


import numpy as np
import pandas as pd


from app.ml.splitting import (
    MLSplitInputError,
    validated_time_values,
)


from app.ml.time_series_contracts import (
    MLTimeSeriesForecastingContract,
)


# ============================================================
# VERSION
# ============================================================


ML_TIME_SERIES_TRAINING_INPUT_RULE_VERSION = (
    "ml_time_series_training_input_v0.1"
)


# ============================================================
# ERRORS
# ============================================================


class MLTimeSeriesTrainingInputError(
    RuntimeError
):
    pass


# ============================================================
# VALIDATED SERIES
# ============================================================


@dataclass(
    frozen=True
)
class MLValidatedUnivariateTimeSeries:
    """
    Internal canonical representation of one validated
    univariate forecasting series.

    time_values and target_values are ordered chronologically.

    source_row_positions records the original positional row
    identity before chronological ordering.

    This is internal execution state, not a public API contract.
    """

    time_values: pd.Series

    target_values: pd.Series

    source_row_positions: np.ndarray

    rule_version: str = (
        ML_TIME_SERIES_TRAINING_INPUT_RULE_VERSION
    )


    @property
    def row_count(
        self,
    ) -> int:

        return int(
            len(
                self.target_values
            )
        )


# ============================================================
# COLUMN AUTHORITY
# ============================================================


def _require_exactly_one_column(
    *,
    dataframe: pd.DataFrame,
    column: str,
) -> None:

    matching_count = sum(
        1
        for existing
        in dataframe.columns
        if str(
            existing
        )
        ==
        column
    )


    if matching_count == 0:

        raise (
            MLTimeSeriesTrainingInputError(
                (
                    "Forecasting input dataset is missing "
                    "a required column. "
                    f"column={column}"
                )
            )
        )


    if matching_count != 1:

        raise (
            MLTimeSeriesTrainingInputError(
                (
                    "Forecasting input requires exactly "
                    "one column for each declared role. "
                    f"column={column}, "
                    f"matching_count={matching_count}"
                )
            )
        )


# ============================================================
# TARGET AUTHORITY
# ============================================================


def _validated_numeric_target(
    *,
    dataframe: pd.DataFrame,
    contract: MLTimeSeriesForecastingContract,
) -> tuple[
    pd.Series,
    np.ndarray,
]:

    target = (
        dataframe.loc[
            :,
            contract.target_column,
        ]
    )


    if not isinstance(
        target,
        pd.Series,
    ):

        raise (
            MLTimeSeriesTrainingInputError(
                (
                    "Forecasting target must resolve "
                    "to exactly one pandas Series."
                )
            )
        )


    target = (
        target.copy(
            deep=True
        )
    )


    if bool(
        target.isna().any()
    ):

        raise (
            MLTimeSeriesTrainingInputError(
                (
                    "Forecasting target contains "
                    "missing values. Target imputation "
                    "is not allowed."
                )
            )
        )


    target_dtype = (
        target.dtype
    )


    if (
        pd.api.types
        .is_bool_dtype(
            target_dtype
        )
        or
        not pd.api.types
        .is_numeric_dtype(
            target_dtype
        )
    ):

        raise (
            MLTimeSeriesTrainingInputError(
                (
                    "Forecasting target must be "
                    "numeric and non-boolean. "
                    f"target={contract.target_column}, "
                    f"dtype={target_dtype}"
                )
            )
        )


    try:

        numeric_target = (
            target.to_numpy(
                dtype=np.float64,
                copy=True,
            )
        )

    except Exception as error:

        raise (
            MLTimeSeriesTrainingInputError(
                (
                    "Forecasting target could not "
                    "be converted to float64."
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
            MLTimeSeriesTrainingInputError(
                (
                    "Forecasting target contains "
                    "non-finite values."
                )
            )
        )


    if (
        int(
            target.nunique(
                dropna=False
            )
        )
        <
        2
    ):

        raise (
            MLTimeSeriesTrainingInputError(
                (
                    "Forecasting target must contain "
                    "at least two distinct values."
                )
            )
        )


    return (
        target,
        numeric_target,
    )


# ============================================================
# PUBLIC INPUT AUTHORITY
# ============================================================


def validate_and_extract_time_series(
    *,
    dataframe: pd.DataFrame,
    contract: MLTimeSeriesForecastingContract,
) -> MLValidatedUnivariateTimeSeries:
    """
    Validate and canonicalize one univariate forecasting series.

    v0.1 policy:

    - source must be a non-empty pandas DataFrame;
    - target and time columns must exist exactly once;
    - time validation is delegated to the shared Model Lab
      temporal authority;
    - timestamps must be unique;
    - no implicit aggregation or resampling is performed;
    - target must be numeric, finite and non-constant;
    - at least lookback + 1 observations must exist;
    - output is deterministically ordered by time.
    """

    if not isinstance(
        dataframe,
        pd.DataFrame,
    ):

        raise (
            MLTimeSeriesTrainingInputError(
                (
                    "Forecasting input must be "
                    "a pandas DataFrame."
                )
            )
        )


    if dataframe.empty:

        raise (
            MLTimeSeriesTrainingInputError(
                "Forecasting input dataset cannot be empty."
            )
        )


    contract = (
        MLTimeSeriesForecastingContract
        .model_validate(
            contract
        )
    )


    _require_exactly_one_column(
        dataframe=dataframe,
        column=contract.target_column,
    )


    _require_exactly_one_column(
        dataframe=dataframe,
        column=contract.time_column,
    )


    (
        target,
        numeric_target,
    ) = (
        _validated_numeric_target(
            dataframe=dataframe,
            contract=contract,
        )
    )


    alignment_frame = pd.DataFrame(
        index=dataframe.index.copy()
    )


    try:

        time_values = (
            validated_time_values(
                dataframe=dataframe,
                x=alignment_frame,
                y=target,
                contract=contract,
            )
        )

    except MLSplitInputError as error:

        raise (
            MLTimeSeriesTrainingInputError(
                str(
                    error
                )
            )
        ) from error


    if bool(
        time_values.duplicated(
            keep=False
        ).any()
    ):

        raise (
            MLTimeSeriesTrainingInputError(
                (
                    "Forecasting v0.1 requires exactly "
                    "one observation per timestamp. "
                    "Repeated timestamps must be explicitly "
                    "aggregated during Preparation."
                )
            )
        )


    row_count = int(
        len(
            target
        )
    )


    minimum_rows = (
        int(
            contract.lookback
        )
        +
        1
    )


    if row_count < minimum_rows:

        raise (
            MLTimeSeriesTrainingInputError(
                (
                    "Forecasting input does not contain "
                    "enough observations for the configured "
                    "lookback. "
                    f"rows={row_count}, "
                    f"lookback={contract.lookback}, "
                    f"minimum_rows={minimum_rows}"
                )
            )
        )


    ordered_positions = (
        np.argsort(
            time_values.to_numpy(),
            kind="stable",
        )
        .astype(
            np.int64,
            copy=True,
        )
    )


    ordered_time = (
        time_values.iloc[
            ordered_positions
        ]
        .reset_index(
            drop=True
        )
        .copy(
            deep=True
        )
    )


    ordered_target = pd.Series(
        numeric_target[
            ordered_positions
        ],
        name=contract.target_column,
        dtype=np.float64,
    )


    if not (
        ordered_time
        .is_monotonic_increasing
    ):

        raise (
            MLTimeSeriesTrainingInputError(
                (
                    "Canonical forecasting time order "
                    "is not monotonically increasing."
                )
            )
        )


    ordered_positions.setflags(
        write=False
    )


    return (
        MLValidatedUnivariateTimeSeries(
            time_values=
                ordered_time,

            target_values=
                ordered_target,

            source_row_positions=
                ordered_positions,
        )
    )
