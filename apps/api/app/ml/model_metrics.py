from __future__ import annotations


import math


from typing import (
    Any,
)


import pandas as pd


from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    explained_variance_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    precision_score,
    r2_score,
    recall_score,
)


# ============================================================
# VERSION
# ============================================================


ML_MODEL_METRICS_RULE_VERSION = (
    "ml_model_metrics_v0.1"
)


# ============================================================
# ERRORS
# ============================================================


class MLModelMetricsError(
    RuntimeError
):
    pass


# ============================================================
# REGRESSION
# ============================================================


def compute_ml_regression_metrics(
    *,
    y_true: pd.Series,
    predictions: Any,
) -> dict[
    str,
    float,
]:
    """
    Canonical deterministic regression metric surface.

    This function intentionally preserves the exact metric
    definitions previously implemented inside
    classical_executor.py.
    """

    mae = float(
        mean_absolute_error(
            y_true,
            predictions,
        )
    )


    mse = float(
        mean_squared_error(
            y_true,
            predictions,
        )
    )


    rmse = float(
        math.sqrt(
            mse
        )
    )


    r2 = float(
        r2_score(
            y_true,
            predictions,
        )
    )


    median_ae = float(
        median_absolute_error(
            y_true,
            predictions,
        )
    )


    explained_variance = float(
        explained_variance_score(
            y_true,
            predictions,
        )
    )


    return {
        "mae":
            mae,

        "rmse":
            rmse,

        "r2":
            r2,

        "median_absolute_error":
            median_ae,

        "explained_variance":
            explained_variance,
    }


# ============================================================
# CLASSIFICATION
# ============================================================


def compute_ml_classification_metrics(
    *,
    y_true: pd.Series,
    predictions: Any,
) -> dict[
    str,
    float,
]:
    """
    Canonical deterministic classification metric surface.

    Supports the same binary / multiclass label surface already
    supported by Classical ML.
    """

    return {
        "accuracy":
            float(
                accuracy_score(
                    y_true,
                    predictions,
                )
            ),

        "f1_macro":
            float(
                f1_score(
                    y_true,
                    predictions,
                    average="macro",
                    zero_division=0,
                )
            ),

        "precision_macro":
            float(
                precision_score(
                    y_true,
                    predictions,
                    average="macro",
                    zero_division=0,
                )
            ),

        "recall_macro":
            float(
                recall_score(
                    y_true,
                    predictions,
                    average="macro",
                    zero_division=0,
                )
            ),

        "balanced_accuracy":
            float(
                balanced_accuracy_score(
                    y_true,
                    predictions,
                )
            ),
    }


# ============================================================
# BASELINE v0.1 PROJECTION
# ============================================================


def project_ml_baseline_metrics_v0_1(
    *,
    problem_type: str,
    metrics: dict[
        str,
        float,
    ],
) -> dict[
    str,
    float,
]:
    """
    Project richer canonical ML metrics onto the exact Baseline
    v0.1 metric surface.

    Baseline v0.1 remains intentionally compatible:

    regression
        mae / rmse / r2

    classification
        accuracy / f1_macro
    """

    if (
        problem_type
        ==
        "regression"
    ):
        required_names = (
            "mae",
            "rmse",
            "r2",
        )

    elif (
        problem_type
        ==
        "classification"
    ):
        required_names = (
            "accuracy",
            "f1_macro",
        )

    else:
        raise MLModelMetricsError(
            (
                "Unsupported problem type for "
                "Baseline v0.1 metric projection. "
                f"problem_type={problem_type}"
            )
        )


    missing = [
        metric_name

        for metric_name
        in required_names

        if metric_name
        not in metrics
    ]


    if missing:
        raise MLModelMetricsError(
            (
                "Richer ML metric surface is missing "
                "metrics required by Baseline v0.1. "
                f"missing={missing}"
            )
        )


    return {
        metric_name:
            float(
                metrics[
                    metric_name
                ]
            )

        for metric_name
        in required_names
    }
