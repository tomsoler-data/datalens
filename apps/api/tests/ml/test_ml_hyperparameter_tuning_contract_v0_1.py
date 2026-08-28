from __future__ import annotations


from pydantic import (
    ValidationError,
)


from app.ml.contracts import (
    MLTrainingContract,
)


from app.ml.experiment_provenance import (
    canonical_ml_training_contract_json,
    ml_training_contract_sha256,
)


from app.ml.hyperparameter_tuning import (
    ML_HYPERPARAMETER_TUNING_RULE_VERSION,
    MLHyperparameterCandidateResult,
    MLHyperparameterMetricSummary,
    MLHyperparameterSearchContract,
    MLHyperparameterSearchResult,
    expected_hyperparameter_metric_names,
    hyperparameter_metric_direction,
    hyperparameter_primary_metric,
    hyperparameter_validation_strategy,
    server_owned_hyperparameter_candidates,
)


# ============================================================
# HELPERS
# ============================================================


def expect_validation_error(
    factory,
) -> None:

    try:
        factory()

    except ValidationError:
        return


    raise AssertionError(
        "Expected ValidationError."
    )


def base_training_contract(
) -> MLTrainingContract:

    return (
        MLTrainingContract(
            workflow_id=
                "prep:tuning-contract",

            dataset_id=
                "dataset:validated",

            problem_type=
                "regression",

            target_column=
                "revenue",

            feature_columns=[
                "age",
                "tenure",
                "segment",
            ],

            categorical_feature_columns=[
                "segment",
            ],

            estimator_key=
                "ridge_regression",
        )
    )


def regression_metric_summary(
    *,
    rmse_mean: float,
    rmse_std: float,
) -> dict[
    str,
    MLHyperparameterMetricSummary,
]:

    return {
        "mae":
            MLHyperparameterMetricSummary(
                mean=
                    rmse_mean
                    *
                    0.8,
                std=
                    rmse_std,
            ),

        "rmse":
            MLHyperparameterMetricSummary(
                mean=
                    rmse_mean,
                std=
                    rmse_std,
            ),

        "r2":
            MLHyperparameterMetricSummary(
                mean=
                    0.9,
                std=
                    0.01,
            ),

        "median_absolute_error":
            MLHyperparameterMetricSummary(
                mean=
                    rmse_mean
                    *
                    0.7,
                std=
                    rmse_std,
            ),

        "explained_variance":
            MLHyperparameterMetricSummary(
                mean=
                    0.91,
                std=
                    0.01,
            ),
    }


def classification_metric_summary(
    *,
    f1_mean: float,
    f1_std: float,
) -> dict[
    str,
    MLHyperparameterMetricSummary,
]:

    return {
        "accuracy":
            MLHyperparameterMetricSummary(
                mean=
                    f1_mean,
                std=
                    f1_std,
            ),

        "f1_macro":
            MLHyperparameterMetricSummary(
                mean=
                    f1_mean,
                std=
                    f1_std,
            ),

        "precision_macro":
            MLHyperparameterMetricSummary(
                mean=
                    f1_mean,
                std=
                    f1_std,
            ),

        "recall_macro":
            MLHyperparameterMetricSummary(
                mean=
                    f1_mean,
                std=
                    f1_std,
            ),

        "balanced_accuracy":
            MLHyperparameterMetricSummary(
                mean=
                    f1_mean,
                std=
                    f1_std,
            ),
    }


def ridge_candidate_results(
) -> list[
    MLHyperparameterCandidateResult
]:

    grid = (
        server_owned_hyperparameter_candidates(
            estimator_key=
                "ridge_regression"
        )
    )


    # Ranking:
    # candidate 2 -> RMSE 1.00
    # candidate 1 -> RMSE 1.20
    # candidate 3 -> RMSE 1.50

    return [
        MLHyperparameterCandidateResult(
            candidate_index=
                2,
            rank=
                1,
            hyperparameters=
                grid[
                    1
                ],
            training_contract_sha256=
                (
                    "2"
                    *
                    64
                ),
            metric_summary=(
                regression_metric_summary(
                    rmse_mean=
                        1.00,
                    rmse_std=
                        0.10,
                )
            ),
        ),

        MLHyperparameterCandidateResult(
            candidate_index=
                1,
            rank=
                2,
            hyperparameters=
                grid[
                    0
                ],
            training_contract_sha256=
                (
                    "1"
                    *
                    64
                ),
            metric_summary=(
                regression_metric_summary(
                    rmse_mean=
                        1.20,
                    rmse_std=
                        0.08,
                )
            ),
        ),

        MLHyperparameterCandidateResult(
            candidate_index=
                3,
            rank=
                3,
            hyperparameters=
                grid[
                    2
                ],
            training_contract_sha256=
                (
                    "3"
                    *
                    64
                ),
            metric_summary=(
                regression_metric_summary(
                    rmse_mean=
                        1.50,
                    rmse_std=
                        0.05,
                )
            ),
        ),
    ]


def regression_search_result(
    **overrides,
) -> MLHyperparameterSearchResult:

    payload = {
        "workflow_id":
            "prep:tuning-contract",

        "dataset_id":
            "dataset:validated",

        "problem_type":
            "regression",

        "estimator_key":
            "ridge_regression",

        "preparation_session_revision":
            7,

        "base_training_contract_sha256":
            (
                "a"
                *
                64
            ),

        "search_strategy":
            "server_owned_grid",

        "validation_strategy":
            "k_fold",

        "primary_metric":
            "rmse",

        "metric_direction":
            "minimize",

        "folds":
            5,

        "shuffle":
            True,

        "random_seed":
            42,

        "outer_train_rows":
            80,

        "holdout_test_rows":
            20,

        "candidate_count":
            3,

        "best_candidate_index":
            2,

        "candidate_results":
            ridge_candidate_results(),
    }


    payload.update(
        overrides
    )


    return (
        MLHyperparameterSearchResult(
            **payload
        )
    )


# ============================================================
# CONFIGURATION
# ============================================================


def test_contract_defaults(
) -> None:

    contract = (
        MLHyperparameterSearchContract()
    )


    assert (
        contract.search_strategy
        ==
        "server_owned_grid"
    )


    assert (
        contract.folds
        ==
        5
    )


    assert (
        contract.shuffle
        is True
    )


    assert (
        contract.random_seed
        ==
        42
    )


    assert (
        contract.rule_version
        ==
        ML_HYPERPARAMETER_TUNING_RULE_VERSION
    )


def test_contract_is_strict_frozen_and_bounded(
) -> None:

    expect_validation_error(
        lambda:
            MLHyperparameterSearchContract(
                arbitrary_grid={
                    "alpha": [
                        1.0,
                        2.0,
                    ]
                }
            )
    )


    expect_validation_error(
        lambda:
            MLHyperparameterSearchContract(
                folds=1
            )
    )


    expect_validation_error(
        lambda:
            MLHyperparameterSearchContract(
                folds=21
            )
    )


    contract = (
        MLHyperparameterSearchContract()
    )


    try:
        contract.folds = 10

    except ValidationError:
        pass

    else:
        raise AssertionError(
            (
                "Hyperparameter Search Contract "
                "must be frozen."
            )
        )


# ============================================================
# SERVER-OWNED POLICY
# ============================================================


def test_server_owned_problem_policy(
) -> None:

    assert (
        hyperparameter_validation_strategy(
            problem_type=
                "regression"
        )
        ==
        "k_fold"
    )


    assert (
        hyperparameter_validation_strategy(
            problem_type=
                "classification"
        )
        ==
        "stratified_k_fold"
    )


    assert (
        hyperparameter_primary_metric(
            problem_type=
                "regression"
        )
        ==
        "rmse"
    )


    assert (
        hyperparameter_primary_metric(
            problem_type=
                "classification"
        )
        ==
        "f1_macro"
    )


    assert (
        hyperparameter_metric_direction(
            problem_type=
                "regression"
        )
        ==
        "minimize"
    )


    assert (
        hyperparameter_metric_direction(
            problem_type=
                "classification"
        )
        ==
        "maximize"
    )


# ============================================================
# SERVER-OWNED GRIDS
# ============================================================


def test_server_owned_candidate_grids(
) -> None:

    expected_sizes = {
        "linear_regression":
            2,

        "ridge_regression":
            3,

        "logistic_regression":
            4,

        "random_forest_regressor":
            4,

        "random_forest_classifier":
            5,
    }


    for (
        estimator_key,
        expected_size,
    ) in expected_sizes.items():

        candidates = (
            server_owned_hyperparameter_candidates(
                estimator_key=
                    estimator_key
            )
        )


        assert (
            len(
                candidates
            )
            ==
            expected_size
        )


        assert (
            all(
                candidate.kind
                ==
                estimator_key

                for candidate
                in candidates
            )
        )


        serialized = [
            candidate.model_dump_json()

            for candidate
            in candidates
        ]


        assert (
            len(
                serialized
            )
            ==
            len(
                set(
                    serialized
                )
            )
        )


def test_unsupported_estimator_grid_fails_closed(
) -> None:

    try:
        server_owned_hyperparameter_candidates(
            estimator_key=
                "unsupported_estimator"
        )

    except ValueError:
        pass

    else:
        raise AssertionError(
            (
                "Unsupported estimator grid "
                "must fail closed."
            )
        )


# ============================================================
# TRAINING CONTRACT FINGERPRINT ISOLATION
# ============================================================


def test_search_contract_does_not_mutate_training_contract_fingerprint(
) -> None:

    contract = (
        base_training_contract()
    )


    canonical_before = (
        canonical_ml_training_contract_json(
            contract
        )
    )


    sha_before = (
        ml_training_contract_sha256(
            contract
        )
    )


    _ = (
        MLHyperparameterSearchContract(
            folds=
                7,

            random_seed=
                73,
        )
    )


    canonical_after = (
        canonical_ml_training_contract_json(
            contract
        )
    )


    sha_after = (
        ml_training_contract_sha256(
            contract
        )
    )


    assert (
        canonical_before
        ==
        canonical_after
    )


    assert (
        sha_before
        ==
        sha_after
    )


    training_payload = (
        contract.model_dump(
            mode="json"
        )
    )


    assert (
        "hyperparameter_search"
        not in
        training_payload
    )


    assert (
        "hyperparameter_tuning"
        not in
        training_payload
    )


# ============================================================
# METRICS
# ============================================================


def test_expected_metric_surfaces(
) -> None:

    assert (
        expected_hyperparameter_metric_names(
            problem_type=
                "regression"
        )
        ==
        (
            "mae",
            "rmse",
            "r2",
            "median_absolute_error",
            "explained_variance",
        )
    )


    assert (
        expected_hyperparameter_metric_names(
            problem_type=
                "classification"
        )
        ==
        (
            "accuracy",
            "f1_macro",
            "precision_macro",
            "recall_macro",
            "balanced_accuracy",
        )
    )


def test_metric_summary_requires_finite_values(
) -> None:

    expect_validation_error(
        lambda:
            MLHyperparameterMetricSummary(
                mean=
                    float(
                        "nan"
                    ),
                std=
                    0.1,
            )
    )


    expect_validation_error(
        lambda:
            MLHyperparameterMetricSummary(
                mean=
                    1.0,
                std=
                    float(
                        "inf"
                    ),
            )
    )


    expect_validation_error(
        lambda:
            MLHyperparameterMetricSummary(
                mean=
                    1.0,
                std=
                    -0.1,
            )
    )


# ============================================================
# REGRESSION RESULT
# ============================================================


def test_regression_result_structure(
) -> None:

    result = (
        regression_search_result()
    )


    assert (
        result.validation_strategy
        ==
        "k_fold"
    )


    assert (
        result.primary_metric
        ==
        "rmse"
    )


    assert (
        result.metric_direction
        ==
        "minimize"
    )


    assert (
        result.candidate_count
        ==
        3
    )


    assert (
        result.best_candidate_index
        ==
        2
    )


    assert (
        [
            candidate.candidate_index

            for candidate
            in result.candidate_results
        ]
        ==
        [
            2,
            1,
            3,
        ]
    )


def test_regression_ranking_must_be_deterministic(
) -> None:

    invalid = (
        ridge_candidate_results()
    )


    invalid[
        0
    ] = (
        invalid[
            0
        ]
        .model_copy(
            update={
                "rank":
                    2,
            }
        )
    )


    invalid[
        1
    ] = (
        invalid[
            1
        ]
        .model_copy(
            update={
                "rank":
                    1,
            }
        )
    )


    expect_validation_error(
        lambda:
            regression_search_result(
                candidate_results=
                    invalid
            )
    )


# ============================================================
# GRID / METRIC FAIL-CLOSED
# ============================================================


def test_candidate_grid_mismatch_fails_closed(
) -> None:

    candidates = (
        ridge_candidate_results()
    )


    wrong_grid = (
        server_owned_hyperparameter_candidates(
            estimator_key=
                "linear_regression"
        )
    )


    candidates[
        0
    ] = (
        candidates[
            0
        ]
        .model_copy(
            update={
                "hyperparameters":
                    wrong_grid[
                        0
                    ],
            }
        )
    )


    expect_validation_error(
        lambda:
            regression_search_result(
                candidate_results=
                    candidates
            )
    )


def test_candidate_metric_surface_mismatch_fails_closed(
) -> None:

    candidates = (
        ridge_candidate_results()
    )


    candidate = (
        candidates[
            0
        ]
    )


    invalid_metrics = dict(
        candidate.metric_summary
    )


    invalid_metrics.pop(
        "explained_variance"
    )


    candidates[
        0
    ] = (
        candidate.model_copy(
            update={
                "metric_summary":
                    invalid_metrics,
            }
        )
    )


    expect_validation_error(
        lambda:
            regression_search_result(
                candidate_results=
                    candidates
            )
    )


# ============================================================
# CLASSIFICATION RANKING
# ============================================================


def test_classification_ranking_maximizes_f1_macro(
) -> None:

    grid = (
        server_owned_hyperparameter_candidates(
            estimator_key=
                "logistic_regression"
        )
    )


    results = [
        MLHyperparameterCandidateResult(
            candidate_index=
                3,
            rank=
                1,
            hyperparameters=
                grid[
                    2
                ],
            training_contract_sha256=
                (
                    "3"
                    *
                    64
                ),
            metric_summary=(
                classification_metric_summary(
                    f1_mean=
                        0.92,
                    f1_std=
                        0.02,
                )
            ),
        ),
        MLHyperparameterCandidateResult(
            candidate_index=
                2,
            rank=
                2,
            hyperparameters=
                grid[
                    1
                ],
            training_contract_sha256=
                (
                    "2"
                    *
                    64
                ),
            metric_summary=(
                classification_metric_summary(
                    f1_mean=
                        0.88,
                    f1_std=
                        0.01,
                )
            ),
        ),
        MLHyperparameterCandidateResult(
            candidate_index=
                4,
            rank=
                3,
            hyperparameters=
                grid[
                    3
                ],
            training_contract_sha256=
                (
                    "4"
                    *
                    64
                ),
            metric_summary=(
                classification_metric_summary(
                    f1_mean=
                        0.84,
                    f1_std=
                        0.03,
                )
            ),
        ),
        MLHyperparameterCandidateResult(
            candidate_index=
                1,
            rank=
                4,
            hyperparameters=
                grid[
                    0
                ],
            training_contract_sha256=
                (
                    "1"
                    *
                    64
                ),
            metric_summary=(
                classification_metric_summary(
                    f1_mean=
                        0.80,
                    f1_std=
                        0.02,
                )
            ),
        ),
    ]


    result = (
        MLHyperparameterSearchResult(
            workflow_id=
                "prep:tuning-classification",

            dataset_id=
                "dataset:classification",

            problem_type=
                "classification",

            estimator_key=
                "logistic_regression",

            preparation_session_revision=
                4,

            base_training_contract_sha256=
                (
                    "b"
                    *
                    64
                ),

            search_strategy=
                "server_owned_grid",

            validation_strategy=
                "stratified_k_fold",

            primary_metric=
                "f1_macro",

            metric_direction=
                "maximize",

            folds=
                5,

            shuffle=
                True,

            random_seed=
                42,

            outer_train_rows=
                80,

            holdout_test_rows=
                20,

            candidate_count=
                4,

            best_candidate_index=
                3,

            candidate_results=
                results,
        )
    )


    assert (
        result.best_candidate_index
        ==
        3
    )


    assert (
        result.candidate_results[
            0
        ].metric_summary[
            "f1_macro"
        ].mean
        ==
        0.92
    )


# ============================================================
# SERVER-OWNED RESULT POLICY
# ============================================================


def test_result_problem_policy_mismatch_fails_closed(
) -> None:

    expect_validation_error(
        lambda:
            regression_search_result(
                primary_metric=
                    "f1_macro"
            )
    )


    expect_validation_error(
        lambda:
            regression_search_result(
                metric_direction=
                    "maximize"
            )
    )


    expect_validation_error(
        lambda:
            regression_search_result(
                validation_strategy=
                    "stratified_k_fold"
            )
    )


# ============================================================
# PRIVACY
# ============================================================


def test_result_is_privacy_minimal(
) -> None:

    result = (
        regression_search_result()
    )


    payload = (
        result.model_dump(
            mode="json"
        )
    )


    forbidden = {
        "raw_rows",
        "predictions",
        "fold_predictions",
        "x_train",
        "x_test",
        "y_train",
        "y_test",
        "holdout_predictions",
        "model_bytes",
        "model_path",
        "estimator",
    }


    assert (
        forbidden
        .isdisjoint(
            payload
        )
    )


# ============================================================
# VERSION
# ============================================================


def test_rule_version(
) -> None:

    assert (
        ML_HYPERPARAMETER_TUNING_RULE_VERSION
        ==
        "ml_hyperparameter_tuning_v0.1"
    )


# ============================================================
# MAIN
# ============================================================


def main(
) -> None:

    print(
        "=== DATALENS ML HYPERPARAMETER TUNING CONTRACT v0.1 ==="
    )


    tests = [
        (
            "Contract defaults",
            test_contract_defaults,
        ),
        (
            "Strict frozen bounded contract",
            test_contract_is_strict_frozen_and_bounded,
        ),
        (
            "Server-owned problem policy",
            test_server_owned_problem_policy,
        ),
        (
            "Server-owned finite candidate grids",
            test_server_owned_candidate_grids,
        ),
        (
            "Unsupported estimator fails closed",
            test_unsupported_estimator_grid_fails_closed,
        ),
        (
            "Training Contract fingerprint isolation",
            test_search_contract_does_not_mutate_training_contract_fingerprint,
        ),
        (
            "Five-metric surfaces",
            test_expected_metric_surfaces,
        ),
        (
            "Finite metric summaries",
            test_metric_summary_requires_finite_values,
        ),
        (
            "Regression result structure",
            test_regression_result_structure,
        ),
        (
            "Deterministic regression ranking",
            test_regression_ranking_must_be_deterministic,
        ),
        (
            "Candidate grid mismatch fail-closed",
            test_candidate_grid_mismatch_fails_closed,
        ),
        (
            "Metric surface mismatch fail-closed",
            test_candidate_metric_surface_mismatch_fails_closed,
        ),
        (
            "Classification maximizes F1 macro",
            test_classification_ranking_maximizes_f1_macro,
        ),
        (
            "Server-owned result policy",
            test_result_problem_policy_mismatch_fails_closed,
        ),
        (
            "Privacy-minimal result",
            test_result_is_privacy_minimal,
        ),
        (
            "Rule version",
            test_rule_version,
        ),
    ]


    for (
        label,
        test,
    ) in tests:

        test()

        print(
            f"[PASS] {label}"
        )


    print()
    print(
        "PASS - ML Hyperparameter Tuning Contract v0.1"
    )


if __name__ == "__main__":
    main()
