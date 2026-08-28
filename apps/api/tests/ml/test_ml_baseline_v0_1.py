from __future__ import annotations


import math


import numpy as np
import pandas as pd


from app.ml.baseline import (
    ML_BASELINE_RULE_VERSION,
    MLBaselineError,
    build_ml_baseline_evaluation,
    build_ml_baseline_predictions,
    compare_model_to_baseline,
)


# ============================================================
# REGRESSION BASELINE
# ============================================================


def test_regression_mean_baseline(
) -> None:

    y_train = pd.Series(
        [
            1.0,
            2.0,
            9.0,
        ]
    )


    result = (
        build_ml_baseline_predictions(
            problem_type=
                "regression",

            y_train=
                y_train,

            test_rows=
                4,
        )
    )


    assert (
        result.strategy
        ==
        "mean_train_target"
    )


    expected = float(
        y_train.mean()
    )


    assert (
        np.allclose(
            result.predictions,
            np.full(
                4,
                expected,
            ),
        )
    )


# ============================================================
# CLASSIFICATION BASELINE
# ============================================================


def test_classification_majority_baseline(
) -> None:

    y_train = pd.Series(
        [
            "yes",
            "no",
            "yes",
            "yes",
            "no",
        ]
    )


    result = (
        build_ml_baseline_predictions(
            problem_type=
                "classification",

            y_train=
                y_train,

            test_rows=
                3,
        )
    )


    assert (
        result.strategy
        ==
        "majority_train_class"
    )


    assert (
        result.predictions.tolist()
        ==
        [
            "yes",
            "yes",
            "yes",
        ]
    )


# ============================================================
# DETERMINISTIC TIE BREAKER
# ============================================================


def test_majority_tie_uses_first_training_appearance(
) -> None:

    y_train = pd.Series(
        [
            "beta",
            "alpha",
            "alpha",
            "beta",
        ]
    )


    result = (
        build_ml_baseline_predictions(
            problem_type=
                "classification",

            y_train=
                y_train,

            test_rows=
                2,
        )
    )


    assert (
        result.predictions.tolist()
        ==
        [
            "beta",
            "beta",
        ]
    )


# ============================================================
# RESULT CONTRACT
# ============================================================


def test_regression_baseline_result(
) -> None:

    result = (
        build_ml_baseline_evaluation(
            problem_type=
                "regression",

            strategy=
                "mean_train_target",

            metrics={
                "mae":
                    8.0,

                "rmse":
                    10.0,

                "r2":
                    -0.2,
            },

            train_rows=
                80,

            test_rows=
                20,
        )
    )


    assert (
        result.primary_metric
        ==
        "rmse"
    )


    assert (
        result.metrics[
            "rmse"
        ]
        ==
        10.0
    )


# ============================================================
# REGRESSION COMPARISON
# ============================================================


def test_regression_model_beats_baseline(
) -> None:

    comparison = (
        compare_model_to_baseline(
            problem_type=
                "regression",

            model_metrics={
                "mae":
                    3.0,

                "rmse":
                    4.0,

                "r2":
                    0.8,
            },

            baseline_metrics={
                "mae":
                    9.0,

                "rmse":
                    10.0,

                "r2":
                    0.0,
            },
        )
    )


    assert (
        comparison.beats_baseline
        is True
    )


    assert (
        comparison.absolute_improvement
        ==
        6.0
    )


    assert math.isclose(
        comparison.relative_improvement_pct,
        60.0,
    )


# ============================================================
# CLASSIFICATION COMPARISON
# ============================================================


def test_classification_model_beats_baseline(
) -> None:

    comparison = (
        compare_model_to_baseline(
            problem_type=
                "classification",

            model_metrics={
                "accuracy":
                    0.90,

                "f1_macro":
                    0.80,
            },

            baseline_metrics={
                "accuracy":
                    0.60,

                "f1_macro":
                    0.50,
            },
        )
    )


    assert (
        comparison.beats_baseline
        is True
    )


    assert math.isclose(
        comparison.absolute_improvement,
        0.30,
    )


    assert math.isclose(
        comparison.relative_improvement_pct,
        60.0,
    )


# ============================================================
# ZERO BASELINE DENOMINATOR
# ============================================================


def test_zero_baseline_metric_has_no_relative_percentage(
) -> None:

    comparison = (
        compare_model_to_baseline(
            problem_type=
                "classification",

            model_metrics={
                "accuracy":
                    0.50,

                "f1_macro":
                    0.20,
            },

            baseline_metrics={
                "accuracy":
                    0.0,

                "f1_macro":
                    0.0,
            },
        )
    )


    assert (
        comparison.relative_improvement_pct
        is None
    )


    assert (
        comparison.beats_baseline
        is True
    )


# ============================================================
# NON-FINITE GUARD
# ============================================================


def test_non_finite_primary_metric_is_blocked(
) -> None:

    try:
        compare_model_to_baseline(
            problem_type=
                "regression",

            model_metrics={
                "rmse":
                    float(
                        "nan"
                    )
            },

            baseline_metrics={
                "rmse":
                    10.0
            },
        )

    except MLBaselineError:
        return


    raise AssertionError(
        (
            "Non-finite primary metric must "
            "be blocked."
        )
    )


# ============================================================
# RULE VERSION
# ============================================================


def test_rule_version(
) -> None:

    assert (
        ML_BASELINE_RULE_VERSION
        ==
        "ml_baseline_v0.1"
    )


# ============================================================
# MAIN
# ============================================================


def main(
) -> None:

    print(
        "=== DATALENS ML BASELINE v0.1 ==="
    )

    print()


    test_regression_mean_baseline()

    print(
        "Regression mean(y_train) baseline: PASS"
    )


    test_classification_majority_baseline()

    print(
        "Classification majority(y_train) baseline: PASS"
    )


    test_majority_tie_uses_first_training_appearance()

    print(
        "Deterministic majority tie-breaker: PASS"
    )


    test_regression_baseline_result()

    print(
        "Typed regression baseline result: PASS"
    )


    test_regression_model_beats_baseline()

    print(
        "Regression model vs baseline: PASS"
    )


    test_classification_model_beats_baseline()

    print(
        "Classification model vs baseline: PASS"
    )


    test_zero_baseline_metric_has_no_relative_percentage()

    print(
        "Zero-denominator relative improvement: PASS"
    )


    test_non_finite_primary_metric_is_blocked()

    print(
        "Non-finite baseline comparison is blocked: PASS"
    )


    test_rule_version()

    print(
        "ML Baseline rule version: PASS"
    )


    print()

    print(
        "ML Baseline v0.1: PASS"
    )


if __name__ == "__main__":
    main()
