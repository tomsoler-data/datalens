from __future__ import annotations


import math


import numpy as np
import pandas as pd


from app.ml.classical_executor import (
    ML_RICHER_METRICS_RULE_VERSION,
    _baseline_metrics_v0_1,
    _classification_metrics,
    _regression_metrics,
    execute_classical_ml,
)


from tests.ml.test_classical_ml_executor_v0_1 import (
    classification_contract,
    classification_dataframe,
    isolated_environment,
    patched_handoff,
    regression_contract,
    regression_dataframe,
    seed_preparation_authority,
)


REGRESSION_METRIC_NAMES = {
    "mae",
    "rmse",
    "r2",
    "median_absolute_error",
    "explained_variance",
}


CLASSIFICATION_METRIC_NAMES = {
    "accuracy",
    "f1_macro",
    "precision_macro",
    "recall_macro",
    "balanced_accuracy",
}


REGRESSION_BASELINE_METRIC_NAMES = {
    "mae",
    "rmse",
    "r2",
}


CLASSIFICATION_BASELINE_METRIC_NAMES = {
    "accuracy",
    "f1_macro",
}


# ============================================================
# REGRESSION DEFINITIONS
# ============================================================


def test_regression_richer_metrics_known_values(
) -> None:

    y_true = pd.Series(
        [
            0.0,
            1.0,
            2.0,
            3.0,
        ]
    )


    predictions = np.asarray(
        [
            0.0,
            1.0,
            4.0,
            2.0,
        ],
        dtype=np.float64,
    )


    metrics = (
        _regression_metrics(
            y_true=
                y_true,

            predictions=
                predictions,
        )
    )


    assert (
        set(
            metrics
        )
        ==
        REGRESSION_METRIC_NAMES
    )


    assert math.isclose(
        metrics[
            "mae"
        ],
        0.75,
        rel_tol=1e-12,
        abs_tol=1e-12,
    )


    assert math.isclose(
        metrics[
            "rmse"
        ],
        math.sqrt(
            1.25
        ),
        rel_tol=1e-12,
        abs_tol=1e-12,
    )


    assert math.isclose(
        metrics[
            "r2"
        ],
        0.0,
        rel_tol=1e-12,
        abs_tol=1e-12,
    )


    assert math.isclose(
        metrics[
            "median_absolute_error"
        ],
        0.5,
        rel_tol=1e-12,
        abs_tol=1e-12,
    )


    assert math.isclose(
        metrics[
            "explained_variance"
        ],
        0.05,
        rel_tol=1e-12,
        abs_tol=1e-12,
    )


# ============================================================
# CLASSIFICATION DEFINITIONS
# ============================================================


def test_classification_richer_metrics_known_values(
) -> None:

    y_true = pd.Series(
        [
            "a",
            "a",
            "a",
            "b",
            "b",
            "c",
        ]
    )


    predictions = np.asarray(
        [
            "a",
            "a",
            "b",
            "b",
            "c",
            "c",
        ],
        dtype=object,
    )


    metrics = (
        _classification_metrics(
            y_true=
                y_true,

            predictions=
                predictions,
        )
    )


    assert (
        set(
            metrics
        )
        ==
        CLASSIFICATION_METRIC_NAMES
    )


    assert math.isclose(
        metrics[
            "accuracy"
        ],
        (
            4.0
            /
            6.0
        ),
        rel_tol=1e-12,
        abs_tol=1e-12,
    )


    assert math.isclose(
        metrics[
            "precision_macro"
        ],
        (
            2.0
            /
            3.0
        ),
        rel_tol=1e-12,
        abs_tol=1e-12,
    )


    assert math.isclose(
        metrics[
            "recall_macro"
        ],
        (
            13.0
            /
            18.0
        ),
        rel_tol=1e-12,
        abs_tol=1e-12,
    )


    assert math.isclose(
        metrics[
            "balanced_accuracy"
        ],
        (
            13.0
            /
            18.0
        ),
        rel_tol=1e-12,
        abs_tol=1e-12,
    )


# ============================================================
# BASELINE COMPATIBILITY
# ============================================================


def test_baseline_projection_preserves_v0_1_contract(
) -> None:

    regression = (
        _baseline_metrics_v0_1(
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


    classification = (
        _baseline_metrics_v0_1(
            problem_type=
                "classification",

            metrics={
                "accuracy":
                    0.8,

                "f1_macro":
                    0.7,

                "precision_macro":
                    0.75,

                "recall_macro":
                    0.72,

                "balanced_accuracy":
                    0.72,
            },
        )
    )


    assert (
        set(
            regression
        )
        ==
        REGRESSION_BASELINE_METRIC_NAMES
    )


    assert (
        set(
            classification
        )
        ==
        CLASSIFICATION_BASELINE_METRIC_NAMES
    )


# ============================================================
# REAL REGRESSION EXECUTION
# ============================================================


def test_regression_executor_persists_richer_metrics(
) -> None:

    with isolated_environment():

        contract = (
            regression_contract()
        )


        seed_preparation_authority(
            workflow_id=
                contract.workflow_id,

            dataset_id=
                contract.dataset_id,
        )


        with patched_handoff(
            dataframe=
                regression_dataframe(),

            workflow_id=
                contract.workflow_id,

            dataset_id=
                contract.dataset_id,
        ):

            result = (
                execute_classical_ml(
                    training_contract=
                        contract
                )
            )


        assert (
            set(
                result.metrics
            )
            ==
            REGRESSION_METRIC_NAMES
        )


        assert (
            set(
                result.baseline.metrics
            )
            ==
            REGRESSION_BASELINE_METRIC_NAMES
        )


        assert (
            result.model_artifact.metrics
            ==
            result.metrics
        )


        assert (
            result
            .experiment_provenance
            .metrics
            ==
            result.metrics
        )


        for value in (
            result.metrics.values()
        ):
            assert math.isfinite(
                float(
                    value
                )
            )


# ============================================================
# REAL CLASSIFICATION EXECUTION
# ============================================================


def test_classification_executor_persists_richer_metrics(
) -> None:

    with isolated_environment():

        contract = (
            classification_contract()
        )


        seed_preparation_authority(
            workflow_id=
                contract.workflow_id,

            dataset_id=
                contract.dataset_id,
        )


        with patched_handoff(
            dataframe=
                classification_dataframe(),

            workflow_id=
                contract.workflow_id,

            dataset_id=
                contract.dataset_id,
        ):

            result = (
                execute_classical_ml(
                    training_contract=
                        contract
                )
            )


        assert (
            set(
                result.metrics
            )
            ==
            CLASSIFICATION_METRIC_NAMES
        )


        assert (
            set(
                result.baseline.metrics
            )
            ==
            CLASSIFICATION_BASELINE_METRIC_NAMES
        )


        assert (
            result.model_artifact.metrics
            ==
            result.metrics
        )


        assert (
            result
            .experiment_provenance
            .metrics
            ==
            result.metrics
        )


        for value in (
            result.metrics.values()
        ):
            assert math.isfinite(
                float(
                    value
                )
            )


# ============================================================
# VERSION
# ============================================================


def test_richer_metrics_rule_version(
) -> None:

    assert (
        ML_RICHER_METRICS_RULE_VERSION
        ==
        "ml_richer_metrics_v0.1"
    )


# ============================================================
# MAIN
# ============================================================


def main(
) -> None:

    print(
        "=== DATALENS RICHER ML METRICS v0.1 ==="
    )

    print()


    test_regression_richer_metrics_known_values()

    print(
        "Regression richer metric definitions: PASS"
    )


    test_classification_richer_metrics_known_values()

    print(
        "Classification richer metric definitions: PASS"
    )


    test_baseline_projection_preserves_v0_1_contract()

    print(
        "Baseline v0.1 metric projection: PASS"
    )


    test_regression_executor_persists_richer_metrics()

    print(
        (
            "Regression metrics -> Model Artifact -> "
            "Experiment Provenance: PASS"
        )
    )


    test_classification_executor_persists_richer_metrics()

    print(
        (
            "Classification metrics -> Model Artifact -> "
            "Experiment Provenance: PASS"
        )
    )


    test_richer_metrics_rule_version()

    print(
        "Richer ML Metrics rule version: PASS"
    )


    print()

    print(
        "Richer ML Metrics v0.1: PASS"
    )


if __name__ == "__main__":
    main()
