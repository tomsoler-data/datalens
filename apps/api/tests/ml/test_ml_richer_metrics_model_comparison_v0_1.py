from __future__ import annotations


import math


from app.ml.model_comparison_contracts import (
    MLModelComparisonContract,
)


from app.ml.model_comparison_executor import (
    _ranking_key,
    execute_ml_model_comparison,
)


from tests.ml.test_classical_ml_executor_v0_1 import (
    classification_dataframe,
    isolated_environment,
    patched_handoff,
    regression_dataframe,
    seed_preparation_authority,
)


from tests.ml.test_ml_model_comparison_executor_v0_1 import (
    DATASET_ID,
    WORKFLOW_ID,
    classification_candidate,
    patched_comparison_readiness,
    regression_candidate,
)


# ============================================================
# EXPECTED METRIC SURFACES
# ============================================================


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
# REGRESSION MODEL COMPARISON
# ============================================================


def test_regression_comparison_propagates_richer_metrics(
) -> None:

    with isolated_environment():

        seed_preparation_authority(
            workflow_id=
                WORKFLOW_ID,

            dataset_id=
                DATASET_ID,
        )


        contract = (
            MLModelComparisonContract(
                candidates=[
                    regression_candidate(
                        estimator_key=
                            "random_forest_regressor",

                        estimator_hyperparameters={
                            "kind":
                                "random_forest_regressor",

                            "n_estimators":
                                64,

                            "max_depth":
                                8,
                        },
                    ),

                    regression_candidate(
                        estimator_key=
                            "linear_regression"
                    ),

                    regression_candidate(
                        estimator_key=
                            "ridge_regression",

                        estimator_hyperparameters={
                            "kind":
                                "ridge_regression",

                            "alpha":
                                2.0,
                        },
                    ),
                ]
            )
        )


        with patched_comparison_readiness():

            with patched_handoff(
                dataframe=
                    regression_dataframe(),

                workflow_id=
                    WORKFLOW_ID,

                dataset_id=
                    DATASET_ID,
            ):

                result = (
                    execute_ml_model_comparison(
                        comparison_contract=
                            contract
                    )
                )


        assert (
            result.primary_metric
            ==
            "rmse"
        )


        assert (
            result.ranking_policy
            ==
            "regression_rmse_v0.1"
        )


        assert (
            set(
                result.baseline.metrics
            )
            ==
            REGRESSION_BASELINE_METRIC_NAMES
        )


        assert (
            len(
                result.candidates
            )
            ==
            3
        )


        for candidate in (
            result.candidates
        ):

            assert (
                set(
                    candidate.metrics
                )
                ==
                REGRESSION_METRIC_NAMES
            )


            assert (
                candidate.model_artifact.metrics
                ==
                candidate.metrics
            )


            assert (
                candidate
                .experiment_provenance
                .metrics
                ==
                candidate.metrics
            )


            assert (
                candidate
                .model_artifact
                .experiment_provenance
                ==
                candidate
                .experiment_provenance
            )


            assert (
                candidate.primary_metric
                ==
                "rmse"
            )


            assert math.isclose(
                candidate.primary_metric_value,
                candidate.metrics[
                    "rmse"
                ],
                rel_tol=1e-12,
                abs_tol=1e-12,
            )


            for value in (
                candidate.metrics.values()
            ):
                assert math.isfinite(
                    float(
                        value
                    )
                )


        expected_winner = min(
            result.candidates,

            key=lambda candidate: (
                _ranking_key(
                    problem_type=
                        "regression",

                    estimator_key=
                        candidate.estimator_key,

                    metrics=
                        candidate.metrics,
                )
            ),
        )


        assert (
            result.selected_estimator_key
            ==
            expected_winner.estimator_key
        )


        assert (
            result.selected_experiment_id
            ==
            expected_winner
            .experiment_provenance
            .experiment_id
        )


        assert (
            result.selected_model_id
            ==
            expected_winner
            .model_artifact
            .model_id
        )


# ============================================================
# CLASSIFICATION MODEL COMPARISON
# ============================================================


def test_classification_comparison_propagates_richer_metrics(
) -> None:

    with isolated_environment():

        seed_preparation_authority(
            workflow_id=
                WORKFLOW_ID,

            dataset_id=
                DATASET_ID,
        )


        contract = (
            MLModelComparisonContract(
                candidates=[
                    classification_candidate(
                        estimator_key=
                            "random_forest_classifier",

                        estimator_hyperparameters={
                            "kind":
                                "random_forest_classifier",

                            "n_estimators":
                                80,

                            "max_depth":
                                6,
                        },
                    ),

                    classification_candidate(
                        estimator_key=
                            "logistic_regression"
                    ),
                ]
            )
        )


        with patched_comparison_readiness():

            with patched_handoff(
                dataframe=
                    classification_dataframe(),

                workflow_id=
                    WORKFLOW_ID,

                dataset_id=
                    DATASET_ID,
            ):

                result = (
                    execute_ml_model_comparison(
                        comparison_contract=
                            contract
                    )
                )


        assert (
            result.primary_metric
            ==
            "f1_macro"
        )


        assert (
            result.ranking_policy
            ==
            "classification_f1_macro_v0.1"
        )


        assert (
            set(
                result.baseline.metrics
            )
            ==
            CLASSIFICATION_BASELINE_METRIC_NAMES
        )


        assert (
            len(
                result.candidates
            )
            ==
            2
        )


        for candidate in (
            result.candidates
        ):

            assert (
                set(
                    candidate.metrics
                )
                ==
                CLASSIFICATION_METRIC_NAMES
            )


            assert (
                candidate.model_artifact.metrics
                ==
                candidate.metrics
            )


            assert (
                candidate
                .experiment_provenance
                .metrics
                ==
                candidate.metrics
            )


            assert (
                candidate
                .model_artifact
                .experiment_provenance
                ==
                candidate
                .experiment_provenance
            )


            assert (
                candidate.primary_metric
                ==
                "f1_macro"
            )


            assert math.isclose(
                candidate.primary_metric_value,
                candidate.metrics[
                    "f1_macro"
                ],
                rel_tol=1e-12,
                abs_tol=1e-12,
            )


            for metric_name in (
                CLASSIFICATION_METRIC_NAMES
            ):

                assert (
                    0.0
                    <=
                    candidate.metrics[
                        metric_name
                    ]
                    <=
                    1.0
                )


        expected_winner = min(
            result.candidates,

            key=lambda candidate: (
                _ranking_key(
                    problem_type=
                        "classification",

                    estimator_key=
                        candidate.estimator_key,

                    metrics=
                        candidate.metrics,
                )
            ),
        )


        assert (
            result.selected_estimator_key
            ==
            expected_winner.estimator_key
        )


        assert (
            result.selected_experiment_id
            ==
            expected_winner
            .experiment_provenance
            .experiment_id
        )


        assert (
            result.selected_model_id
            ==
            expected_winner
            .model_artifact
            .model_id
        )


# ============================================================
# RANKING ISOLATION
# ============================================================


def test_regression_richer_metrics_do_not_change_ranking_key(
) -> None:

    first = (
        _ranking_key(
            problem_type=
                "regression",

            estimator_key=
                "linear_regression",

            metrics={
                "rmse":
                    2.0,

                "mae":
                    1.5,

                "r2":
                    0.8,

                "median_absolute_error":
                    0.1,

                "explained_variance":
                    0.99,
            },
        )
    )


    second = (
        _ranking_key(
            problem_type=
                "regression",

            estimator_key=
                "linear_regression",

            metrics={
                "rmse":
                    2.0,

                "mae":
                    1.5,

                "r2":
                    0.8,

                "median_absolute_error":
                    999999.0,

                "explained_variance":
                    -999999.0,
            },
        )
    )


    assert (
        first
        ==
        second
    )


def test_classification_richer_metrics_do_not_change_ranking_key(
) -> None:

    first = (
        _ranking_key(
            problem_type=
                "classification",

            estimator_key=
                "logistic_regression",

            metrics={
                "f1_macro":
                    0.8,

                "accuracy":
                    0.85,

                "precision_macro":
                    1.0,

                "recall_macro":
                    1.0,

                "balanced_accuracy":
                    1.0,
            },
        )
    )


    second = (
        _ranking_key(
            problem_type=
                "classification",

            estimator_key=
                "logistic_regression",

            metrics={
                "f1_macro":
                    0.8,

                "accuracy":
                    0.85,

                "precision_macro":
                    0.0,

                "recall_macro":
                    0.0,

                "balanced_accuracy":
                    0.0,
            },
        )
    )


    assert (
        first
        ==
        second
    )


# ============================================================
# MAIN
# ============================================================


def main(
) -> None:

    print(
        (
            "=== DATALENS RICHER ML METRICS / "
            "MODEL COMPARISON v0.1 ==="
        )
    )

    print()


    test_regression_comparison_propagates_richer_metrics()

    print(
        (
            "Regression comparison propagates "
            "five richer metrics: PASS"
        )
    )


    test_classification_comparison_propagates_richer_metrics()

    print(
        (
            "Classification comparison propagates "
            "five richer metrics: PASS"
        )
    )


    test_regression_richer_metrics_do_not_change_ranking_key()

    print(
        (
            "Regression richer-only metrics do not "
            "change ranking: PASS"
        )
    )


    test_classification_richer_metrics_do_not_change_ranking_key()

    print(
        (
            "Classification richer-only metrics do not "
            "change ranking: PASS"
        )
    )


    print()

    print(
        (
            "Richer ML Metrics / Model Comparison "
            "v0.1: PASS"
        )
    )


if __name__ == "__main__":
    main()
