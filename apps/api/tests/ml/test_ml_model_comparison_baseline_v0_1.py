from __future__ import annotations


import math


from app.ml.baseline import (
    MLBaselineEvaluationResult,
    compare_model_to_baseline,
)


from app.ml.model_comparison_contracts import (
    MLModelComparisonContract,
)


from app.ml.model_comparison_executor import (
    MLModelComparisonExecutorError,
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
# REGRESSION SHARED BASELINE
# ============================================================


def test_regression_comparison_exposes_shared_baseline(
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
            result.baseline.problem_type
            ==
            "regression"
        )


        assert (
            result.baseline.strategy
            ==
            "mean_train_target"
        )


        assert (
            result.baseline.primary_metric
            ==
            "rmse"
        )


        baseline_rmse = (
            result
            .baseline
            .metrics[
                "rmse"
            ]
        )


        assert math.isfinite(
            baseline_rmse
        )


        for candidate in (
            result.candidates
        ):

            comparison = (
                candidate
                .baseline_comparison
            )


            assert (
                comparison.primary_metric
                ==
                "rmse"
            )


            assert math.isclose(
                (
                    comparison
                    .baseline_primary_metric_value
                ),
                baseline_rmse,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )


            assert math.isclose(
                (
                    comparison
                    .model_primary_metric_value
                ),
                candidate.metrics[
                    "rmse"
                ],
                rel_tol=1e-12,
                abs_tol=1e-12,
            )


            assert (
                comparison.beats_baseline
                ==
                (
                    candidate.metrics[
                        "rmse"
                    ]
                    <
                    baseline_rmse
                )
            )


        winner = (
            result.candidates[
                0
            ]
        )


        assert (
            winner
            .baseline_comparison
            .beats_baseline
            is True
        )


# ============================================================
# CLASSIFICATION SHARED BASELINE
# ============================================================


def test_classification_comparison_exposes_shared_baseline(
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
            result.baseline.problem_type
            ==
            "classification"
        )


        assert (
            result.baseline.strategy
            ==
            "majority_train_class"
        )


        assert (
            result.baseline.primary_metric
            ==
            "f1_macro"
        )


        baseline_f1 = (
            result
            .baseline
            .metrics[
                "f1_macro"
            ]
        )


        for candidate in (
            result.candidates
        ):

            comparison = (
                candidate
                .baseline_comparison
            )


            assert math.isclose(
                (
                    comparison
                    .baseline_primary_metric_value
                ),
                baseline_f1,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )


            assert math.isclose(
                (
                    comparison
                    .model_primary_metric_value
                ),
                candidate.metrics[
                    "f1_macro"
                ],
                rel_tol=1e-12,
                abs_tol=1e-12,
            )


            assert (
                comparison.beats_baseline
                ==
                (
                    candidate.metrics[
                        "f1_macro"
                    ]
                    >
                    baseline_f1
                )
            )


# ============================================================
# DIVERGENT BASELINE MUST FAIL CLOSED
# ============================================================


def test_candidate_baseline_divergence_is_fail_closed(
) -> None:

    import app.ml.model_comparison_executor as comparison_module


    original_execute = (
        comparison_module
        .execute_classical_ml
    )


    execution_count = 0


    def divergent_execute(
        *,
        training_contract,
    ):

        nonlocal execution_count


        execution_count += 1


        result = (
            original_execute(
                training_contract=
                    training_contract
            )
        )


        if (
            execution_count
            !=
            2
        ):
            return result


        changed_metrics = dict(
            result.baseline.metrics
        )


        changed_metrics[
            "rmse"
        ] = (
            float(
                changed_metrics[
                    "rmse"
                ]
            )
            +
            1.0
        )


        changed_baseline = (
            MLBaselineEvaluationResult(
                problem_type=
                    result.baseline.problem_type,

                strategy=
                    result.baseline.strategy,

                primary_metric=
                    result.baseline.primary_metric,

                train_rows=
                    result.baseline.train_rows,

                test_rows=
                    result.baseline.test_rows,

                metrics=
                    changed_metrics,
            )
        )


        changed_comparison = (
            compare_model_to_baseline(
                problem_type=
                    result.problem_type,

                model_metrics=
                    result.metrics,

                baseline_metrics=
                    changed_baseline.metrics,
            )
        )


        return (
            result.model_copy(
                update={
                    "baseline":
                        changed_baseline,

                    "baseline_comparison":
                        changed_comparison,
                }
            )
        )


    comparison_module.execute_classical_ml = (
        divergent_execute
    )


    try:

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
                                "linear_regression"
                        ),

                        regression_candidate(
                            estimator_key=
                                "ridge_regression"
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

                    try:
                        comparison_module.execute_ml_model_comparison(
                            comparison_contract=
                                contract
                        )

                    except MLModelComparisonExecutorError:

                        assert (
                            execution_count
                            ==
                            2
                        )

                        return


        raise AssertionError(
            (
                "Divergent candidate baselines must "
                "fail Model Comparison closed."
            )
        )


    finally:

        comparison_module.execute_classical_ml = (
            original_execute
        )


# ============================================================
# BASELINE IS NOT A MODEL ARTIFACT
# ============================================================


def test_comparison_baseline_is_not_persisted_as_model(
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
                            "linear_regression"
                    ),

                    regression_candidate(
                        estimator_key=
                            "ridge_regression"
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


        assert not hasattr(
            result.baseline,
            "model_id",
        )


        for candidate in (
            result.candidates
        ):

            artifact_payload = (
                candidate
                .model_artifact
                .model_dump()
            )


            assert (
                "baseline"
                not in
                artifact_payload
            )


            assert (
                "baseline_comparison"
                not in
                artifact_payload
            )


# ============================================================
# MAIN
# ============================================================


def main(
) -> None:

    print(
        "=== DATALENS ML MODEL COMPARISON BASELINE v0.1 ==="
    )

    print()


    test_regression_comparison_exposes_shared_baseline()

    print(
        "Regression comparison shared baseline: PASS"
    )


    test_classification_comparison_exposes_shared_baseline()

    print(
        "Classification comparison shared baseline: PASS"
    )


    test_candidate_baseline_divergence_is_fail_closed()

    print(
        "Divergent candidate baseline is fail-closed: PASS"
    )


    test_comparison_baseline_is_not_persisted_as_model()

    print(
        "Comparison baseline is not a Model Artifact: PASS"
    )


    print()

    print(
        "ML Model Comparison Baseline v0.1: PASS"
    )


if __name__ == "__main__":
    main()
