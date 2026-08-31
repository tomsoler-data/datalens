from __future__ import annotations


import math


import numpy as np
import pandas as pd


from app.ml.classical_executor import (
    ClassicalMLExecutorError,
    _baseline_metrics_v0_1,
    _classification_metrics,
    _regression_metrics,
)


from app.ml.model_metrics import (
    ML_MODEL_METRICS_RULE_VERSION,
    MLModelMetricsError,
    compute_ml_classification_metrics,
    compute_ml_regression_metrics,
    project_ml_baseline_metrics_v0_1,
)


# ============================================================
# HELPERS
# ============================================================


def expect_error(
    error_type,
    factory,
) -> None:

    try:
        factory()

    except error_type:
        return


    raise AssertionError(
        (
            "Expected "
            f"{error_type.__name__}."
        )
    )


def assert_close(
    actual: float,
    expected: float,
) -> None:

    assert math.isclose(
        actual,
        expected,
        rel_tol=1e-12,
        abs_tol=1e-12,
    )


# ============================================================
# REGRESSION
# ============================================================


def test_regression_metric_surface(
) -> None:

    y_true = pd.Series(
        [
            1.0,
            2.0,
            3.0,
            4.0,
        ]
    )


    predictions = np.array(
        [
            1.0,
            2.0,
            4.0,
            4.0,
        ],
        dtype=float,
    )


    result = (
        compute_ml_regression_metrics(
            y_true=
                y_true,

            predictions=
                predictions,
        )
    )


    assert (
        set(
            result
        )
        ==
        {
            "mae",
            "rmse",
            "r2",
            "median_absolute_error",
            "explained_variance",
        }
    )


    assert_close(
        result[
            "mae"
        ],
        0.25,
    )


    assert_close(
        result[
            "rmse"
        ],
        0.5,
    )


    assert_close(
        result[
            "r2"
        ],
        0.8,
    )


    assert_close(
        result[
            "median_absolute_error"
        ],
        0.0,
    )


    assert_close(
        result[
            "explained_variance"
        ],
        0.85,
    )


# ============================================================
# CLASSIFICATION
# ============================================================


def test_classification_metric_surface(
) -> None:

    y_true = pd.Series(
        [
            0,
            0,
            1,
            1,
        ]
    )


    predictions = np.array(
        [
            0,
            1,
            1,
            1,
        ]
    )


    result = (
        compute_ml_classification_metrics(
            y_true=
                y_true,

            predictions=
                predictions,
        )
    )


    assert (
        set(
            result
        )
        ==
        {
            "accuracy",
            "f1_macro",
            "precision_macro",
            "recall_macro",
            "balanced_accuracy",
        }
    )


    assert_close(
        result[
            "accuracy"
        ],
        0.75,
    )


    assert_close(
        result[
            "f1_macro"
        ],
        (
            (
                2.0
                /
                3.0
            )
            +
            0.8
        )
        /
        2.0,
    )


    assert_close(
        result[
            "precision_macro"
        ],
        (
            1.0
            +
            (
                2.0
                /
                3.0
            )
        )
        /
        2.0,
    )


    assert_close(
        result[
            "recall_macro"
        ],
        0.75,
    )


    assert_close(
        result[
            "balanced_accuracy"
        ],
        0.75,
    )


# ============================================================
# BASELINE PROJECTION
# ============================================================


def test_baseline_projection(
) -> None:

    regression = (
        project_ml_baseline_metrics_v0_1(
            problem_type=
                "regression",

            metrics={
                "mae":
                    1.0,

                "rmse":
                    2.0,

                "r2":
                    0.5,

                "median_absolute_error":
                    0.75,

                "explained_variance":
                    0.6,
            },
        )
    )


    assert regression == {
        "mae":
            1.0,

        "rmse":
            2.0,

        "r2":
            0.5,
    }


    classification = (
        project_ml_baseline_metrics_v0_1(
            problem_type=
                "classification",

            metrics={
                "accuracy":
                    0.8,

                "f1_macro":
                    0.7,

                "precision_macro":
                    0.9,

                "recall_macro":
                    0.6,

                "balanced_accuracy":
                    0.75,
            },
        )
    )


    assert classification == {
        "accuracy":
            0.8,

        "f1_macro":
            0.7,
    }


# ============================================================
# FAIL CLOSED
# ============================================================


def test_invalid_projection_fails_closed(
) -> None:

    expect_error(
        MLModelMetricsError,

        lambda:
            project_ml_baseline_metrics_v0_1(
                problem_type=
                    "unsupported",

                metrics={},
            ),
    )


    expect_error(
        MLModelMetricsError,

        lambda:
            project_ml_baseline_metrics_v0_1(
                problem_type=
                    "classification",

                metrics={
                    "accuracy":
                        0.8
                },
            ),
    )


# ============================================================
# BACKWARD COMPATIBILITY
# ============================================================


def test_classical_executor_private_aliases_preserve_metrics(
) -> None:

    regression_y = pd.Series(
        [
            1.0,
            2.0,
            3.0,
        ]
    )


    regression_predictions = np.array(
        [
            1.0,
            2.5,
            3.0,
        ]
    )


    assert (
        _regression_metrics(
            y_true=
                regression_y,

            predictions=
                regression_predictions,
        )
        ==
        compute_ml_regression_metrics(
            y_true=
                regression_y,

            predictions=
                regression_predictions,
        )
    )


    classification_y = pd.Series(
        [
            0,
            1,
            0,
            1,
        ]
    )


    classification_predictions = np.array(
        [
            0,
            1,
            1,
            1,
        ]
    )


    assert (
        _classification_metrics(
            y_true=
                classification_y,

            predictions=
                classification_predictions,
        )
        ==
        compute_ml_classification_metrics(
            y_true=
                classification_y,

            predictions=
                classification_predictions,
        )
    )


    projected = (
        _baseline_metrics_v0_1(
            problem_type=
                "classification",

            metrics={
                "accuracy":
                    0.75,

                "f1_macro":
                    0.73,
            },
        )
    )


    assert projected == {
        "accuracy":
            0.75,

        "f1_macro":
            0.73,
    }


    expect_error(
        ClassicalMLExecutorError,

        lambda:
            _baseline_metrics_v0_1(
                problem_type=
                    "classification",

                metrics={},
            ),
    )


# ============================================================
# INPUT IMMUTABILITY
# ============================================================


def test_metric_inputs_remain_immutable(
) -> None:

    y_true = pd.Series(
        [
            0,
            1,
            1,
            0,
        ]
    )


    predictions = np.array(
        [
            0,
            1,
            0,
            0,
        ]
    )


    y_before = (
        y_true.copy(
            deep=True
        )
    )


    predictions_before = (
        predictions.copy()
    )


    compute_ml_classification_metrics(
        y_true=
            y_true,

        predictions=
            predictions,
    )


    pd.testing.assert_series_equal(
        y_true,
        y_before,
    )


    assert np.array_equal(
        predictions,
        predictions_before,
    )


# ============================================================
# VERSION
# ============================================================


def test_rule_version(
) -> None:

    assert (
        ML_MODEL_METRICS_RULE_VERSION
        ==
        "ml_model_metrics_v0.1"
    )


# ============================================================
# MAIN
# ============================================================


def main(
) -> None:

    print(
        "=== DATALENS ML MODEL METRICS v0.1 ==="
    )

    print()


    tests = [
        (
            "Regression canonical metric surface",
            test_regression_metric_surface,
        ),
        (
            "Classification canonical metric surface",
            test_classification_metric_surface,
        ),
        (
            "Baseline metric projection",
            test_baseline_projection,
        ),
        (
            "Invalid metric projection fails closed",
            test_invalid_projection_fails_closed,
        ),
        (
            "Classical Executor aliases remain compatible",
            test_classical_executor_private_aliases_preserve_metrics,
        ),
        (
            "Metric inputs remain immutable",
            test_metric_inputs_remain_immutable,
        ),
        (
            "Model Metrics rule version",
            test_rule_version,
        ),
    ]


    for (
        label,
        callback,
    ) in tests:

        callback()

        print(
            f"[PASS] {label}"
        )


    print()

    print(
        "PASS - ML Model Metrics v0.1"
    )


if __name__ == "__main__":
    main()
