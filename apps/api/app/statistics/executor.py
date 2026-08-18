import math

import numpy as np
import pandas as pd

from scipy.stats import (
    pearsonr,
    permutation_test,
    spearmanr,
)

from app.statistics.engine import (
    DEFAULT_ALPHA,
    StatisticalAnalysisError,
    prepare_numeric_pair,
    validate_alpha,
    validate_statistical_value,
    validate_variability,
)

from app.statistics.schemas import (
    CorrelationExecution,
    CorrelationResult,
    CorrelationTestDecision,
)


# ============================================================
# EXECUTION DEFAULTS
# ============================================================

DEFAULT_PERMUTATION_RESAMPLES = 9999

DEFAULT_RANDOM_SEED = 42


# ============================================================
# EXECUTION ERROR
# ============================================================

class StatisticalExecutionError(
    ValueError
):
    """
    Raised when DataLens cannot safely execute
    a statistical decision.
    """

    pass


# ============================================================
# DECISION VALIDATION
# ============================================================

def validate_executable_decision(
    decision: CorrelationTestDecision,
) -> None:
    """
    Ensure that the decision engine actually
    selected a test before execution.

    DataLens must never convert
    needs_information into an implicit test.
    """

    if (
        decision.status
        != "selected"
    ):
        raise StatisticalExecutionError(
            (
                "The statistical decision is not "
                "executable because its status is "
                f"{decision.status!r}."
            )
        )

    if (
        decision.selected_test
        is None
    ):
        raise StatisticalExecutionError(
            (
                "The statistical decision has "
                "status='selected' but does not "
                "contain selected_test."
            )
        )

    if (
        decision.inference_method
        is None
    ):
        raise StatisticalExecutionError(
            (
                "The statistical decision has "
                "status='selected' but does not "
                "contain an inference method."
            )
        )


# ============================================================
# DATA / DECISION CONSISTENCY
# ============================================================

def validate_decision_columns(
    decision: CorrelationTestDecision,
    x_column: str,
    y_column: str,
) -> None:
    """
    Prevent a decision created for one pair of
    variables from being executed on another.
    """

    if (
        decision.x_column
        != x_column
    ):
        raise StatisticalExecutionError(
            (
                "Decision x_column does not match "
                "the requested execution column: "
                f"{decision.x_column!r} != "
                f"{x_column!r}."
            )
        )

    if (
        decision.y_column
        != y_column
    ):
        raise StatisticalExecutionError(
            (
                "Decision y_column does not match "
                "the requested execution column: "
                f"{decision.y_column!r} != "
                f"{y_column!r}."
            )
        )


# ============================================================
# PERMUTATION MODE
# ============================================================

def determine_permutation_mode(
    n_valid: int,
    n_resamples: int,
) -> str:
    """
    Determine whether the requested number of
    pairings is sufficient for an exact
    permutation test.

    With one sample permuted relative to a fixed
    second sample, there are n! index pairings.
    """

    if n_resamples < 1:
        raise StatisticalExecutionError(
            (
                "permutation_resamples must be "
                "at least 1."
            )
        )

    number_of_pairings = (
        math.factorial(
            n_valid
        )
    )

    if (
        n_resamples
        >= number_of_pairings
    ):
        return "exact"

    return "randomized"


# ============================================================
# PEARSON STATISTIC
# ============================================================

def calculate_pearson_statistic(
    x_values: np.ndarray,
    y_values: np.ndarray,
) -> float:
    """
    Calculate Pearson r only.

    This helper is also used as the statistic
    inside the permutation test.
    """

    result = pearsonr(
        x_values,
        y_values,
    )

    return validate_statistical_value(
        result.statistic,
        "pearson_coefficient",
    )


# ============================================================
# SPEARMAN STATISTIC
# ============================================================

def calculate_spearman_statistic(
    x_values: np.ndarray,
    y_values: np.ndarray,
) -> float:
    """
    Calculate Spearman rho only.

    This helper is also used as the statistic
    inside the permutation test.
    """

    result = spearmanr(
        x_values,
        y_values,
    )

    return validate_statistical_value(
        result.statistic,
        "spearman_coefficient",
    )


# ============================================================
# PEARSON STANDARD INFERENCE
# ============================================================

def execute_pearson_standard(
    x_values: np.ndarray,
    y_values: np.ndarray,
    alpha: float,
) -> CorrelationResult:
    """
    Execute Pearson with SciPy's standard
    inference.
    """

    result = pearsonr(
        x_values,
        y_values,
        alternative="two-sided",
    )

    coefficient = (
        validate_statistical_value(
            result.statistic,
            "pearson_coefficient",
        )
    )

    p_value = (
        validate_statistical_value(
            result.pvalue,
            "pearson_p_value",
        )
    )

    return CorrelationResult(
        test=
            "pearson",

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

        n=
            int(
                len(
                    x_values
                )
            ),

        alpha=
            alpha,

        statistically_significant=(
            p_value
            < alpha
        ),
    )


# ============================================================
# SPEARMAN STANDARD INFERENCE
# ============================================================

def execute_spearman_standard(
    x_values: np.ndarray,
    y_values: np.ndarray,
    alpha: float,
) -> CorrelationResult:
    """
    Execute Spearman with SciPy's standard
    asymptotic inference.
    """

    result = spearmanr(
        x_values,
        y_values,
        alternative="two-sided",
    )

    coefficient = (
        validate_statistical_value(
            result.statistic,
            "spearman_coefficient",
        )
    )

    p_value = (
        validate_statistical_value(
            result.pvalue,
            "spearman_p_value",
        )
    )

    return CorrelationResult(
        test=
            "spearman",

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

        n=
            int(
                len(
                    x_values
                )
            ),

        alpha=
            alpha,

        statistically_significant=(
            p_value
            < alpha
        ),
    )


# ============================================================
# GENERIC ASSOCIATION PERMUTATION
# ============================================================

def execute_permutation_inference(
    x_values: np.ndarray,
    y_values: np.ndarray,
    selected_test: str,
    alpha: float,
    n_resamples: int,
    random_seed: int,
) -> CorrelationResult:
    """
    Execute a pairing permutation test for
    Pearson r or Spearman rho.

    Only x is permuted.

    y remains fixed.

    This represents random reassignment of the
    pairing between x and y under the null
    hypothesis of no association.
    """

    if (
        selected_test
        == "pearson"
    ):
        observed_coefficient = (
            calculate_pearson_statistic(
                x_values=
                    x_values,

                y_values=
                    y_values,
            )
        )

        def statistic(
            x_permuted,
        ):
            return (
                calculate_pearson_statistic(
                    x_values=
                        x_permuted,

                    y_values=
                        y_values,
                )
            )

        relationship_type = (
            "linear"
        )

        coefficient_name = (
            "r"
        )

    elif (
        selected_test
        == "spearman"
    ):
        observed_coefficient = (
            calculate_spearman_statistic(
                x_values=
                    x_values,

                y_values=
                    y_values,
            )
        )

        def statistic(
            x_permuted,
        ):
            return (
                calculate_spearman_statistic(
                    x_values=
                        x_permuted,

                    y_values=
                        y_values,
                )
            )

        relationship_type = (
            "monotonic"
        )

        coefficient_name = (
            "rho"
        )

    else:
        raise StatisticalExecutionError(
            (
                "Unsupported correlation test "
                f"for permutation inference: "
                f"{selected_test!r}."
            )
        )

    rng = (
        np.random.default_rng(
            random_seed
        )
    )

    permutation_result = (
        permutation_test(
            data=(
                x_values,
            ),

            statistic=
                statistic,

            permutation_type=
                "pairings",

            vectorized=
                False,

            n_resamples=
                n_resamples,

            alternative=
                "two-sided",

            rng=
                rng,
        )
    )

    p_value = (
        validate_statistical_value(
            permutation_result.pvalue,
            (
                f"{selected_test}_"
                "permutation_p_value"
            ),
        )
    )

    return CorrelationResult(
        test=
            selected_test,

        relationship_type=
            relationship_type,

        coefficient_name=
            coefficient_name,

        coefficient=
            observed_coefficient,

        p_value=
            p_value,

        alternative=
            "two-sided",

        n=
            int(
                len(
                    x_values
                )
            ),

        alpha=
            alpha,

        statistically_significant=(
            p_value
            < alpha
        ),
    )


# ============================================================
# MAIN EXECUTOR
# ============================================================

def execute_correlation_decision(
    dataframe: pd.DataFrame,
    decision: CorrelationTestDecision,
    alpha: float = DEFAULT_ALPHA,
    permutation_resamples: int = (
        DEFAULT_PERMUTATION_RESAMPLES
    ),
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> CorrelationExecution:
    """
    Execute exactly one statistical test.

    The decision engine must already have
    selected the test.

    This function does not choose between
    Pearson and Spearman.
    """

    validate_executable_decision(
        decision
    )

    validated_alpha = (
        validate_alpha(
            alpha
        )
    )

    x_column = (
        decision.x_column
    )

    y_column = (
        decision.y_column
    )

    validate_decision_columns(
        decision=
            decision,

        x_column=
            x_column,

        y_column=
            y_column,
    )

    (
        x_values,
        y_values,
        n_total,
        n_valid,
        n_excluded,
    ) = prepare_numeric_pair(
        dataframe=
            dataframe,

        x_column=
            x_column,

        y_column=
            y_column,
    )

    validate_variability(
        values=
            x_values,

        column=
            x_column,
    )

    validate_variability(
        values=
            y_values,

        column=
            y_column,
    )

    selected_test = (
        decision.selected_test
    )

    if selected_test is None:
        raise StatisticalExecutionError(
            "No statistical test was selected."
        )

    warnings = []

    # ========================================================
    # STANDARD INFERENCE
    # ========================================================

    if (
        decision.inference_method
        == "standard"
    ):
        if (
            selected_test
            == "pearson"
        ):
            result = (
                execute_pearson_standard(
                    x_values=
                        x_values,

                    y_values=
                        y_values,

                    alpha=
                        validated_alpha,
                )
            )

        elif (
            selected_test
            == "spearman"
        ):
            result = (
                execute_spearman_standard(
                    x_values=
                        x_values,

                    y_values=
                        y_values,

                    alpha=
                        validated_alpha,
                )
            )

        else:
            raise StatisticalExecutionError(
                (
                    "Unsupported selected test: "
                    f"{selected_test!r}."
                )
            )

        inference_method_used = (
            "standard"
        )

        permutation_mode = None

        permutation_resamples_used = (
            None
        )

        execution_seed = None

    # ========================================================
    # PERMUTATION INFERENCE
    # ========================================================

    elif (
        decision.inference_method
        == "permutation_recommended"
    ):
        permutation_mode = (
            determine_permutation_mode(
                n_valid=
                    n_valid,

                n_resamples=
                    permutation_resamples,
            )
        )

        result = (
            execute_permutation_inference(
                x_values=
                    x_values,

                y_values=
                    y_values,

                selected_test=
                    selected_test,

                alpha=
                    validated_alpha,

                n_resamples=
                    permutation_resamples,

                random_seed=
                    random_seed,
            )
        )

        inference_method_used = (
            "permutation"
        )

        permutation_resamples_used = (
            permutation_resamples
        )

        execution_seed = (
            random_seed
        )

        if (
            permutation_mode
            == "exact"
        ):
            warnings.append(
                (
                    "The requested permutation "
                    "count was sufficient for an "
                    "exact pairing permutation "
                    "test at this sample size."
                )
            )

        else:
            warnings.append(
                (
                    "A randomized pairing "
                    "permutation test was used. "
                    "The random seed and requested "
                    "number of resamples are "
                    "recorded for reproducibility."
                )
            )

    else:
        raise StatisticalExecutionError(
            (
                "Unsupported inference method: "
                f"{decision.inference_method!r}."
            )
        )

    # ========================================================
    # CONSISTENCY CHECK
    # ========================================================

    if (
        result.test
        != selected_test
    ):
        raise StatisticalExecutionError(
            (
                "Executed test does not match "
                "the deterministic decision."
            )
        )

    return CorrelationExecution(
        x_column=
            x_column,

        y_column=
            y_column,

        selected_test=
            selected_test,

        inference_method_used=
            inference_method_used,

        permutation_mode=
            permutation_mode,

        permutation_resamples_requested=
            permutation_resamples_used,

        random_seed=
            execution_seed,

        n_total=
            n_total,

        n_valid=
            n_valid,

        n_excluded=
            n_excluded,

        result=
            result,

        decision=
            decision,

        warnings=(
            list(
                decision.warnings
            )
            + warnings
        ),
    )