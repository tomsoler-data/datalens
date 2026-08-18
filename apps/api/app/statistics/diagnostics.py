import math

import numpy as np
import pandas as pd

from app.statistics.engine import (
    prepare_numeric_pair,
    run_pearson,
    run_spearman,
    validate_variability,
)

from app.statistics.schemas import (
    CorrelationDiagnostics,
    ShapeSignal,
    VariableKind,
)


# ============================================================
# DATALENS EXPLORATORY HEURISTICS
# ============================================================
#
# These thresholds are project decision rules.
#
# They are deliberately exposed here rather
# than being hidden inside the LLM.
#
# They are NOT universal statistical laws.
# ============================================================

MINIMUM_SHAPE_SAMPLE_SIZE = 8

MINIMUM_ASSOCIATION_SIGNAL = 0.20

CORRELATION_GAP_THRESHOLD = 0.10

QUADRATIC_R2_GAIN_THRESHOLD = 0.05

NONMONOTONIC_R2_GAIN_THRESHOLD = 0.10

ZERO_TOLERANCE = 1e-12


# ============================================================
# TIES
# ============================================================

def count_tied_observations(
    values: np.ndarray,
) -> int:
    """
    Count observations belonging to duplicated
    value groups.

    Example:
        [1, 1, 2, 3, 3]

    tied observations:
        4
    """

    (
        _,
        counts,
    ) = np.unique(
        values,
        return_counts=True,
    )

    tied_counts = counts[
        counts > 1
    ]

    return int(
        np.sum(
            tied_counts
        )
    )


# ============================================================
# IQR OUTLIER DIAGNOSTIC
# ============================================================

def count_iqr_outliers(
    values: np.ndarray,
) -> int:
    """
    Flag potential univariate outliers using
    the conventional 1.5 × IQR diagnostic.

    DataLens does not remove these values.

    This is only a diagnostic signal.
    """

    q1 = float(
        np.quantile(
            values,
            0.25,
        )
    )

    q3 = float(
        np.quantile(
            values,
            0.75,
        )
    )

    iqr = (
        q3
        - q1
    )

    if (
        not math.isfinite(
            iqr
        )
        or iqr <= 0
    ):
        return 0

    lower_bound = (
        q1
        - 1.5 * iqr
    )

    upper_bound = (
        q3
        + 1.5 * iqr
    )

    outlier_mask = (
        (values < lower_bound)
        |
        (values > upper_bound)
    )

    return int(
        np.sum(
            outlier_mask
        )
    )


# ============================================================
# R-SQUARED SHAPE DIAGNOSTIC
# ============================================================

def calculate_r_squared(
    x_values: np.ndarray,
    y_values: np.ndarray,
    degree: int,
) -> float | None:
    """
    Fit a simple polynomial model using
    NumPy least squares.

    This is used only as an exploratory
    relationship-shape diagnostic.

    It is not a hypothesis test.
    """

    if degree not in {
        1,
        2,
    }:
        raise ValueError(
            "degree must be 1 or 2."
        )

    unique_x = int(
        np.unique(
            x_values
        ).size
    )

    if (
        unique_x
        < degree + 1
    ):
        return None

    x_mean = float(
        np.mean(
            x_values
        )
    )

    x_std = float(
        np.std(
            x_values
        )
    )

    if (
        not math.isfinite(
            x_std
        )
        or x_std <= 0
    ):
        return None

    x_scaled = (
        x_values
        - x_mean
    ) / x_std

    columns = [
        np.ones_like(
            x_scaled
        )
    ]

    for power in range(
        1,
        degree + 1,
    ):
        columns.append(
            x_scaled
            ** power
        )

    design_matrix = (
        np.column_stack(
            columns
        )
    )

    coefficients = (
        np.linalg.lstsq(
            design_matrix,
            y_values,
            rcond=None,
        )[0]
    )

    predictions = (
        design_matrix
        @ coefficients
    )

    residual_sum_squares = float(
        np.sum(
            (
                y_values
                - predictions
            )
            ** 2
        )
    )

    total_sum_squares = float(
        np.sum(
            (
                y_values
                - np.mean(
                    y_values
                )
            )
            ** 2
        )
    )

    if total_sum_squares <= 0:
        return None

    r_squared = (
        1.0
        - (
            residual_sum_squares
            / total_sum_squares
        )
    )

    return float(
        min(
            1.0,
            max(
                0.0,
                r_squared,
            ),
        )
    )


# ============================================================
# COEFFICIENT SIGN
# ============================================================

def coefficient_sign(
    value: float,
) -> int:
    """
    Convert a coefficient to:

    -1 = negative
     0 = approximately zero
     1 = positive
    """

    if abs(
        value
    ) <= ZERO_TOLERANCE:
        return 0

    if value > 0:
        return 1

    return -1


# ============================================================
# EXPLORATORY SHAPE SIGNAL
# ============================================================

def determine_shape_signal(
    n_valid: int,
    pearson_coefficient: float,
    spearman_coefficient: float,
    linear_r_squared: float,
    quadratic_gain: float | None,
) -> ShapeSignal:
    """
    Produce a conservative exploratory signal
    about relationship shape.

    The output is a DataLens heuristic.

    It is NOT a formal proof that a population
    relationship is linear or monotonic.
    """

    if (
        n_valid
        < MINIMUM_SHAPE_SAMPLE_SIZE
    ):
        return (
            "insufficient_for_shape"
        )

    absolute_pearson = abs(
        pearson_coefficient
    )

    absolute_spearman = abs(
        spearman_coefficient
    )

    pearson_sign = (
        coefficient_sign(
            pearson_coefficient
        )
    )

    spearman_sign = (
        coefficient_sign(
            spearman_coefficient
        )
    )

    # --------------------------------------------------------
    # Meaningful directional conflict
    # --------------------------------------------------------

    if (
        absolute_pearson
        >= MINIMUM_ASSOCIATION_SIGNAL
        and absolute_spearman
        >= MINIMUM_ASSOCIATION_SIGNAL
        and pearson_sign
        != spearman_sign
    ):
        return "conflicting"

    # --------------------------------------------------------
    # No obvious association pattern
    # --------------------------------------------------------

    if (
        max(
            absolute_pearson,
            absolute_spearman,
        )
        < MINIMUM_ASSOCIATION_SIGNAL
        and linear_r_squared < 0.05
    ):
        return "no_clear_pattern"

    coefficient_gap = abs(
        absolute_spearman
        - absolute_pearson
    )

    # --------------------------------------------------------
    # Strong quadratic improvement but little
    # monotonic signal:
    # possible curved / U-shaped relationship
    # --------------------------------------------------------

    if (
        quadratic_gain is not None
        and quadratic_gain
        >= NONMONOTONIC_R2_GAIN_THRESHOLD
        and absolute_spearman
        < MINIMUM_ASSOCIATION_SIGNAL
    ):
        return (
            "nonlinear_nonmonotonic_candidate"
        )

    # --------------------------------------------------------
    # Monotonic but potentially non-linear
    # --------------------------------------------------------

    if (
        pearson_sign
        == spearman_sign
        and absolute_spearman
        >= MINIMUM_ASSOCIATION_SIGNAL
        and (
            (
                absolute_spearman
                - absolute_pearson
            )
            >= CORRELATION_GAP_THRESHOLD
            or (
                quadratic_gain is not None
                and quadratic_gain
                >= QUADRATIC_R2_GAIN_THRESHOLD
            )
        )
    ):
        return (
            "monotonic_non_linear_candidate"
        )

    # --------------------------------------------------------
    # Plausible approximately-linear candidate
    # --------------------------------------------------------

    if (
        absolute_pearson
        >= MINIMUM_ASSOCIATION_SIGNAL
        and coefficient_gap
        < CORRELATION_GAP_THRESHOLD
        and (
            quadratic_gain is None
            or quadratic_gain
            < QUADRATIC_R2_GAIN_THRESHOLD
        )
    ):
        return "linear_candidate"

    return "no_clear_pattern"


# ============================================================
# MAIN DIAGNOSTICS
# ============================================================

def build_correlation_diagnostics(
    dataframe: pd.DataFrame,
    x_column: str,
    y_column: str,
    x_kind: VariableKind = "unknown",
    y_kind: VariableKind = "unknown",
    assess_shape: bool = False,
) -> CorrelationDiagnostics:
    """
    Build deterministic diagnostics for two
    numeric variables.

    `assess_shape=False` is the default.

    Confirmatory decision paths should avoid
    inspecting observed correlation results
    simply to decide which test to report.

    Exploratory general-association analysis
    may explicitly request shape assessment.
    """

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

    excluded_fraction = (
        n_excluded
        / n_total
        if n_total > 0
        else 0.0
    )

    # ========================================================
    # TIES
    # ========================================================

    x_tied_observation_count = (
        count_tied_observations(
            x_values
        )
    )

    y_tied_observation_count = (
        count_tied_observations(
            y_values
        )
    )

    x_tied_observation_fraction = (
        x_tied_observation_count
        / n_valid
    )

    y_tied_observation_fraction = (
        y_tied_observation_count
        / n_valid
    )

    # ========================================================
    # OUTLIERS
    # ========================================================

    x_outlier_count = (
        count_iqr_outliers(
            x_values
        )
    )

    y_outlier_count = (
        count_iqr_outliers(
            y_values
        )
    )

    x_outlier_fraction = (
        x_outlier_count
        / n_valid
    )

    y_outlier_fraction = (
        y_outlier_count
        / n_valid
    )

    reliability = (
        "standard"
        if n_valid
        >= MINIMUM_SHAPE_SAMPLE_SIZE
        else "limited"
    )

    warnings = []

    if n_excluded > 0:
        warnings.append(
            (
                f"{n_excluded} row(s) were "
                "excluded pairwise because at "
                "least one value was missing "
                "or non-finite."
            )
        )

    if (
        x_outlier_count > 0
        or y_outlier_count > 0
    ):
        warnings.append(
            (
                "Potential outliers were detected "
                "with the 1.5 × IQR diagnostic. "
                "They were not removed."
            )
        )

    # ========================================================
    # NO OUTCOME-DRIVEN SHAPE ASSESSMENT
    # ========================================================

    if not assess_shape:
        return CorrelationDiagnostics(
            x_column=x_column,

            y_column=y_column,

            x_kind=x_kind,

            y_kind=y_kind,

            n_total=n_total,

            n_valid=n_valid,

            n_excluded=n_excluded,

            excluded_fraction=
                excluded_fraction,

            x_unique=x_unique,

            y_unique=y_unique,

            x_tied_observation_count=
                x_tied_observation_count,

            y_tied_observation_count=
                y_tied_observation_count,

            x_tied_observation_fraction=
                x_tied_observation_fraction,

            y_tied_observation_fraction=
                y_tied_observation_fraction,

            x_outlier_count=
                x_outlier_count,

            y_outlier_count=
                y_outlier_count,

            x_outlier_fraction=
                x_outlier_fraction,

            y_outlier_fraction=
                y_outlier_fraction,

            pearson_coefficient=None,

            spearman_coefficient=None,

            coefficient_gap=None,

            linear_r_squared=None,

            quadratic_r_squared=None,

            quadratic_r_squared_gain=None,

            same_direction=None,

            shape_signal=
                "not_evaluated",

            reliability=
                reliability,

            data_driven_shape_assessment=
                False,

            warnings=
                warnings,
        )

    # ========================================================
    # EXPLORATORY SHAPE ASSESSMENT
    # ========================================================

    pearson_result = (
        run_pearson(
            x_values=x_values,
            y_values=y_values,
        )
    )

    spearman_result = (
        run_spearman(
            x_values=x_values,
            y_values=y_values,
        )
    )

    pearson_coefficient = float(
        pearson_result.coefficient
    )

    spearman_coefficient = float(
        spearman_result.coefficient
    )

    coefficient_gap = abs(
        abs(
            spearman_coefficient
        )
        - abs(
            pearson_coefficient
        )
    )

    linear_r_squared = (
        calculate_r_squared(
            x_values=x_values,
            y_values=y_values,
            degree=1,
        )
    )

    if linear_r_squared is None:
        linear_r_squared = 0.0

    quadratic_r_squared = (
        calculate_r_squared(
            x_values=x_values,
            y_values=y_values,
            degree=2,
        )
    )

    quadratic_gain = None

    if (
        quadratic_r_squared
        is not None
    ):
        quadratic_gain = float(
            max(
                0.0,
                (
                    quadratic_r_squared
                    - linear_r_squared
                ),
            )
        )

    pearson_sign = (
        coefficient_sign(
            pearson_coefficient
        )
    )

    spearman_sign = (
        coefficient_sign(
            spearman_coefficient
        )
    )

    same_direction = (
        pearson_sign
        == spearman_sign
    )

    shape_signal = (
        determine_shape_signal(
            n_valid=n_valid,

            pearson_coefficient=
                pearson_coefficient,

            spearman_coefficient=
                spearman_coefficient,

            linear_r_squared=
                linear_r_squared,

            quadratic_gain=
                quadratic_gain,
        )
    )

    if (
        reliability
        == "limited"
    ):
        warnings.append(
            (
                "The valid sample is small for "
                "automatic exploratory "
                "relationship-shape assessment."
            )
        )

    if (
        shape_signal
        == "conflicting"
    ):
        warnings.append(
            (
                "Exploratory Pearson and Spearman "
                "signals point in conflicting "
                "directions."
            )
        )

    if (
        shape_signal
        == "nonlinear_nonmonotonic_candidate"
    ):
        warnings.append(
            (
                "The exploratory diagnostics "
                "suggest a non-linear pattern "
                "that is not clearly monotonic."
            )
        )

    if (
        shape_signal
        == "no_clear_pattern"
    ):
        warnings.append(
            (
                "The exploratory diagnostics "
                "do not identify a sufficiently "
                "clear linear or monotonic "
                "relationship shape."
            )
        )

    return CorrelationDiagnostics(
        x_column=x_column,

        y_column=y_column,

        x_kind=x_kind,

        y_kind=y_kind,

        n_total=n_total,

        n_valid=n_valid,

        n_excluded=n_excluded,

        excluded_fraction=
            excluded_fraction,

        x_unique=x_unique,

        y_unique=y_unique,

        x_tied_observation_count=
            x_tied_observation_count,

        y_tied_observation_count=
            y_tied_observation_count,

        x_tied_observation_fraction=
            x_tied_observation_fraction,

        y_tied_observation_fraction=
            y_tied_observation_fraction,

        x_outlier_count=
            x_outlier_count,

        y_outlier_count=
            y_outlier_count,

        x_outlier_fraction=
            x_outlier_fraction,

        y_outlier_fraction=
            y_outlier_fraction,

        pearson_coefficient=
            pearson_coefficient,

        spearman_coefficient=
            spearman_coefficient,

        coefficient_gap=
            coefficient_gap,

        linear_r_squared=
            linear_r_squared,

        quadratic_r_squared=
            quadratic_r_squared,

        quadratic_r_squared_gain=
            quadratic_gain,

        same_direction=
            same_direction,

        shape_signal=
            shape_signal,

        reliability=
            reliability,

        data_driven_shape_assessment=
            True,

        warnings=
            warnings,
    )