import math

import numpy as np
import pandas as pd

from pandas.api.types import (
    is_bool_dtype,
    is_numeric_dtype,
)

from scipy.stats import (
    pearsonr,
    spearmanr,
)

from app.statistics.schemas import (
    CorrelationAnalysis,
    CorrelationResult,
)


MINIMUM_SAMPLE_SIZE = 3

SPEARMAN_LARGE_SAMPLE_THRESHOLD = (
    500
)

DEFAULT_ALPHA = 0.05


class StatisticalAnalysisError(
    ValueError
):
    """
    Raised when a deterministic statistical
    analysis cannot safely be performed.
    """

    pass


def validate_alpha(
    alpha: float,
) -> float:
    """
    Validate the significance threshold.
    """

    numeric_alpha = float(
        alpha
    )

    if not (
        0.0
        < numeric_alpha
        < 1.0
    ):
        raise StatisticalAnalysisError(
            (
                "alpha must be strictly "
                "between 0 and 1."
            )
        )

    return numeric_alpha


def validate_column_exists(
    dataframe: pd.DataFrame,
    column: str,
) -> None:
    """
    Ensure that a requested column exists.
    """

    if column not in dataframe.columns:
        raise StatisticalAnalysisError(
            (
                "Column does not exist: "
                f"{column!r}"
            )
        )


def validate_numeric_column(
    dataframe: pd.DataFrame,
    column: str,
) -> None:
    """
    Ensure that a column is genuinely numeric.

    DataLens does not silently coerce text
    columns into numbers inside the statistical
    engine.
    """

    validate_column_exists(
        dataframe=dataframe,
        column=column,
    )

    dtype = dataframe[
        column
    ].dtype

    if is_bool_dtype(
        dtype
    ):
        raise StatisticalAnalysisError(
            (
                f"Column {column!r} is boolean, "
                "not a quantitative numeric "
                "variable."
            )
        )

    if not is_numeric_dtype(
        dtype
    ):
        raise StatisticalAnalysisError(
            (
                f"Column {column!r} must be "
                "numeric before statistical "
                "analysis. "
                f"Current dtype: {dtype}."
            )
        )


def prepare_numeric_pair(
    dataframe: pd.DataFrame,
    x_column: str,
    y_column: str,
) -> tuple[
    np.ndarray,
    np.ndarray,
    int,
    int,
    int,
]:
    """
    Prepare two quantitative variables.

    Missing and non-finite observations are
    removed pairwise.
    """

    if x_column == y_column:
        raise StatisticalAnalysisError(
            (
                "x_column and y_column must "
                "refer to different variables."
            )
        )

    validate_numeric_column(
        dataframe=dataframe,
        column=x_column,
    )

    validate_numeric_column(
        dataframe=dataframe,
        column=y_column,
    )

    n_total = int(
        len(
            dataframe
        )
    )

    x_values = (
        dataframe[
            x_column
        ]
        .to_numpy(
            dtype=float,
            na_value=np.nan,
        )
    )

    y_values = (
        dataframe[
            y_column
        ]
        .to_numpy(
            dtype=float,
            na_value=np.nan,
        )
    )

    valid_mask = (
        np.isfinite(
            x_values
        )
        &
        np.isfinite(
            y_values
        )
    )

    x_valid = x_values[
        valid_mask
    ]

    y_valid = y_values[
        valid_mask
    ]

    n_valid = int(
        len(
            x_valid
        )
    )

    n_excluded = (
        n_total
        - n_valid
    )

    if (
        n_valid
        < MINIMUM_SAMPLE_SIZE
    ):
        raise StatisticalAnalysisError(
            (
                "Not enough valid paired "
                "observations. "
                f"At least "
                f"{MINIMUM_SAMPLE_SIZE} "
                "are required; "
                f"{n_valid} remain."
            )
        )

    return (
        x_valid,
        y_valid,
        n_total,
        n_valid,
        n_excluded,
    )


def validate_variability(
    values: np.ndarray,
    column: str,
) -> int:
    """
    Ensure that a variable contains at least
    two distinct finite values.
    """

    unique_count = int(
        np.unique(
            values
        ).size
    )

    if unique_count < 2:
        raise StatisticalAnalysisError(
            (
                f"Column {column!r} is constant "
                "after pairwise exclusion. "
                "A correlation coefficient "
                "cannot be computed."
            )
        )

    return unique_count


def validate_statistical_value(
    value: float,
    name: str,
) -> float:
    """
    Ensure SciPy returned a finite scalar.
    """

    numeric_value = float(
        value
    )

    if not math.isfinite(
        numeric_value
    ):
        raise StatisticalAnalysisError(
            (
                "Statistical computation "
                "returned a non-finite value "
                f"for {name!r}."
            )
        )

    return numeric_value


def run_pearson(
    x_values: np.ndarray,
    y_values: np.ndarray,
    alpha: float = DEFAULT_ALPHA,
) -> CorrelationResult:
    """
    Compute Pearson's linear correlation.
    """

    validated_alpha = (
        validate_alpha(
            alpha
        )
    )

    result = pearsonr(
        x_values,
        y_values,
        alternative="two-sided",
    )

    coefficient = (
        validate_statistical_value(
            value=result.statistic,
            name="pearson_coefficient",
        )
    )

    p_value = (
        validate_statistical_value(
            value=result.pvalue,
            name="pearson_p_value",
        )
    )

    return CorrelationResult(
        test="pearson",

        relationship_type=
            "linear",

        coefficient_name=
            "r",

        coefficient=
            coefficient,

        p_value=
            p_value,

        alternative=
            "two-sided",

        n=int(
            len(
                x_values
            )
        ),

        alpha=
            validated_alpha,

        statistically_significant=(
            p_value
            < validated_alpha
        ),
    )


def run_spearman(
    x_values: np.ndarray,
    y_values: np.ndarray,
    alpha: float = DEFAULT_ALPHA,
) -> CorrelationResult:
    """
    Compute Spearman's rank correlation.
    """

    validated_alpha = (
        validate_alpha(
            alpha
        )
    )

    result = spearmanr(
        x_values,
        y_values,
        alternative="two-sided",
    )

    coefficient = (
        validate_statistical_value(
            value=result.statistic,
            name="spearman_coefficient",
        )
    )

    p_value = (
        validate_statistical_value(
            value=result.pvalue,
            name="spearman_p_value",
        )
    )

    return CorrelationResult(
        test="spearman",

        relationship_type=
            "monotonic",

        coefficient_name=
            "rho",

        coefficient=
            coefficient,

        p_value=
            p_value,

        alternative=
            "two-sided",

        n=int(
            len(
                x_values
            )
        ),

        alpha=
            validated_alpha,

        statistically_significant=(
            p_value
            < validated_alpha
        ),
    )


def analyze_numeric_relationship(
    dataframe: pd.DataFrame,
    x_column: str,
    y_column: str,
    alpha: float = DEFAULT_ALPHA,
) -> CorrelationAnalysis:
    """
    Analyze the relationship between two
    quantitative variables.

    Python and SciPy are the source of truth.
    The LLM is not involved here.
    """

    validated_alpha = (
        validate_alpha(
            alpha
        )
    )

    (
        x_values,
        y_values,
        n_total,
        n_valid,
        n_excluded,
    ) = prepare_numeric_pair(
        dataframe=dataframe,
        x_column=x_column,
        y_column=y_column,
    )

    x_unique = (
        validate_variability(
            values=x_values,
            column=x_column,
        )
    )

    y_unique = (
        validate_variability(
            values=y_values,
            column=y_column,
        )
    )

    pearson_result = (
        run_pearson(
            x_values=x_values,
            y_values=y_values,
            alpha=validated_alpha,
        )
    )

    spearman_result = (
        run_spearman(
            x_values=x_values,
            y_values=y_values,
            alpha=validated_alpha,
        )
    )

    warnings = []

    if n_excluded > 0:
        warnings.append(
            (
                f"{n_excluded} row(s) were "
                "excluded because at least one "
                "value in the pair was missing "
                "or non-finite."
            )
        )

    if (
        n_valid
        <= SPEARMAN_LARGE_SAMPLE_THRESHOLD
    ):
        warnings.append(
            (
                "The Spearman p-value uses "
                "SciPy's asymptotic calculation. "
                "For smaller samples, a "
                "permutation test should be "
                "considered."
            )
        )

    return CorrelationAnalysis(
        x_column=x_column,

        y_column=y_column,

        n_total=n_total,

        n_valid=n_valid,

        n_excluded=
            n_excluded,

        x_unique=
            x_unique,

        y_unique=
            y_unique,

        pearson=
            pearson_result,

        spearman=
            spearman_result,

        warnings=
            warnings,
    )