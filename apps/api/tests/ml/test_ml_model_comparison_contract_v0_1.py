from __future__ import annotations


from pydantic import (
    ValidationError,
)


from app.ml.contracts import (
    MLPreprocessingContract,
    MLSplitContract,
    MLTrainingContract,
)


from app.ml.model_comparison_contracts import (
    CLASSIFICATION_RANKING_KEYS,
    MLModelComparisonContract,
    ML_MODEL_COMPARISON_CONTRACT_RULE_VERSION,
    REGRESSION_RANKING_KEYS,
)


# ============================================================
# HELPERS
# ============================================================


def expect_validation_error(
    callback,
) -> None:

    try:
        callback()

    except ValidationError:
        return


    raise AssertionError(
        "Expected Pydantic ValidationError."
    )


def regression_candidate(
    *,
    estimator_key: str,
    workflow_id: str = "prep:comparison",
    dataset_id: str = "dataset:validated",
    target_column: str = "revenue",
    feature_columns=None,
    categorical_feature_columns=None,
    preprocessing=None,
    split=None,
    estimator_hyperparameters=None,
) -> MLTrainingContract:

    if (
        feature_columns
        is None
    ):
        feature_columns = [
            "age",
            "tenure",
            "segment",
        ]


    if (
        categorical_feature_columns
        is None
    ):
        categorical_feature_columns = [
            "segment",
        ]


    if (
        preprocessing
        is None
    ):
        preprocessing = (
            MLPreprocessingContract(
                numeric_imputation=
                    "error",

                categorical_imputation=
                    "error",

                categorical_encoding=
                    "one_hot",

                handle_unknown_categories=
                    "ignore",

                scale_numeric=
                    True,
            )
        )


    if (
        split
        is None
    ):
        split = (
            MLSplitContract(
                test_size=
                    0.20,

                random_seed=
                    42,

                shuffle=
                    True,

                stratify=
                    False,
            )
        )


    return (
        MLTrainingContract(
            workflow_id=
                workflow_id,

            dataset_id=
                dataset_id,

            problem_type=
                "regression",

            target_column=
                target_column,

            feature_columns=
                feature_columns,

            categorical_feature_columns=
                categorical_feature_columns,

            estimator_key=
                estimator_key,

            estimator_hyperparameters=
                estimator_hyperparameters,

            preprocessing=
                preprocessing,

            split=
                split,
        )
    )


def classification_candidate(
    *,
    estimator_key: str,
    workflow_id: str = "prep:classification",
    dataset_id: str = "dataset:validated",
    target_column: str = "churned",
    feature_columns=None,
    categorical_feature_columns=None,
    preprocessing=None,
    split=None,
    estimator_hyperparameters=None,
) -> MLTrainingContract:

    if (
        feature_columns
        is None
    ):
        feature_columns = [
            "age",
            "tenure",
            "segment",
        ]


    if (
        categorical_feature_columns
        is None
    ):
        categorical_feature_columns = [
            "segment",
        ]


    if (
        preprocessing
        is None
    ):
        preprocessing = (
            MLPreprocessingContract(
                numeric_imputation=
                    "error",

                categorical_imputation=
                    "error",

                categorical_encoding=
                    "one_hot",

                handle_unknown_categories=
                    "ignore",

                scale_numeric=
                    True,
            )
        )


    if (
        split
        is None
    ):
        split = (
            MLSplitContract(
                test_size=
                    0.25,

                random_seed=
                    42,

                shuffle=
                    True,

                stratify=
                    True,
            )
        )


    return (
        MLTrainingContract(
            workflow_id=
                workflow_id,

            dataset_id=
                dataset_id,

            problem_type=
                "classification",

            target_column=
                target_column,

            feature_columns=
                feature_columns,

            categorical_feature_columns=
                categorical_feature_columns,

            estimator_key=
                estimator_key,

            estimator_hyperparameters=
                estimator_hyperparameters,

            preprocessing=
                preprocessing,

            split=
                split,
        )
    )


# ============================================================
# VALID REGRESSION COMPARISON
# ============================================================


def test_valid_regression_comparison(
) -> None:

    contract = (
        MLModelComparisonContract(
            candidates=[
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

                regression_candidate(
                    estimator_key=
                        "random_forest_regressor",

                    estimator_hyperparameters={
                        "kind":
                            "random_forest_regressor",

                        "n_estimators":
                            100,
                    },
                ),
            ]
        )
    )


    assert (
        contract.problem_type
        ==
        "regression"
    )


    assert (
        contract.workflow_id
        ==
        "prep:comparison"
    )


    assert (
        contract.dataset_id
        ==
        "dataset:validated"
    )


    assert (
        contract.target_column
        ==
        "revenue"
    )


    assert (
        contract.primary_metric
        ==
        "rmse"
    )


    assert (
        contract.ranking_policy
        ==
        "regression_rmse_v0.1"
    )


    assert (
        contract.ranking_keys
        ==
        REGRESSION_RANKING_KEYS
    )


# ============================================================
# VALID CLASSIFICATION COMPARISON
# ============================================================


def test_valid_classification_comparison(
) -> None:

    contract = (
        MLModelComparisonContract(
            candidates=[
                classification_candidate(
                    estimator_key=
                        "logistic_regression"
                ),

                classification_candidate(
                    estimator_key=
                        "random_forest_classifier"
                ),
            ]
        )
    )


    assert (
        contract.problem_type
        ==
        "classification"
    )


    assert (
        contract.primary_metric
        ==
        "f1_macro"
    )


    assert (
        contract.ranking_policy
        ==
        "classification_f1_macro_v0.1"
    )


    assert (
        contract.ranking_keys
        ==
        CLASSIFICATION_RANKING_KEYS
    )


# ============================================================
# MINIMUM CANDIDATES
# ============================================================


def test_at_least_two_candidates_are_required(
) -> None:

    def build(
    ) -> None:

        MLModelComparisonContract(
            candidates=[
                regression_candidate(
                    estimator_key=
                        "linear_regression"
                )
            ]
        )


    expect_validation_error(
        build
    )


# ============================================================
# UNIQUE ESTIMATOR KEYS
# ============================================================


def test_duplicate_estimator_keys_are_blocked(
) -> None:

    def build(
    ) -> None:

        MLModelComparisonContract(
            candidates=[
                regression_candidate(
                    estimator_key=
                        "ridge_regression"
                ),

                regression_candidate(
                    estimator_key=
                        "ridge_regression",

                    estimator_hyperparameters={
                        "kind":
                            "ridge_regression",

                        "alpha":
                            5.0,
                    },
                ),
            ]
        )


    expect_validation_error(
        build
    )


# ============================================================
# UNSUPPORTED ESTIMATOR
# ============================================================


def test_unsupported_estimator_is_blocked(
) -> None:

    def build(
    ) -> None:

        MLModelComparisonContract(
            candidates=[
                regression_candidate(
                    estimator_key=
                        "linear_regression"
                ),

                regression_candidate(
                    estimator_key=
                        "future_unknown_estimator"
                ),
            ]
        )


    expect_validation_error(
        build
    )


# ============================================================
# ESTIMATOR / PROBLEM TYPE
# ============================================================


def test_estimator_problem_type_mismatch_is_blocked(
) -> None:

    invalid_candidate = (
        MLTrainingContract(
            workflow_id=
                "prep:comparison",

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
                "logistic_regression",

            preprocessing=(
                MLPreprocessingContract()
            ),

            split=(
                MLSplitContract(
                    test_size=
                        0.20,

                    random_seed=
                        42,

                    shuffle=
                        True,

                    stratify=
                        False,
                )
            ),
        )
    )


    def build(
    ) -> None:

        MLModelComparisonContract(
            candidates=[
                regression_candidate(
                    estimator_key=
                        "linear_regression",

                    preprocessing=(
                        MLPreprocessingContract()
                    ),
                ),

                invalid_candidate,
            ]
        )


    expect_validation_error(
        build
    )


# ============================================================
# SAME WORKFLOW
# ============================================================


def test_workflow_mismatch_is_blocked(
) -> None:

    def build(
    ) -> None:

        MLModelComparisonContract(
            candidates=[
                regression_candidate(
                    estimator_key=
                        "linear_regression"
                ),

                regression_candidate(
                    estimator_key=
                        "ridge_regression",

                    workflow_id=
                        "prep:other",
                ),
            ]
        )


    expect_validation_error(
        build
    )


# ============================================================
# SAME DATASET
# ============================================================


def test_dataset_mismatch_is_blocked(
) -> None:

    def build(
    ) -> None:

        MLModelComparisonContract(
            candidates=[
                regression_candidate(
                    estimator_key=
                        "linear_regression"
                ),

                regression_candidate(
                    estimator_key=
                        "ridge_regression",

                    dataset_id=
                        "dataset:other",
                ),
            ]
        )


    expect_validation_error(
        build
    )


# ============================================================
# SAME TARGET
# ============================================================


def test_target_mismatch_is_blocked(
) -> None:

    def build(
    ) -> None:

        MLModelComparisonContract(
            candidates=[
                regression_candidate(
                    estimator_key=
                        "linear_regression"
                ),

                regression_candidate(
                    estimator_key=
                        "ridge_regression",

                    target_column=
                        "profit",
                ),
            ]
        )


    expect_validation_error(
        build
    )


# ============================================================
# SAME ORDERED FEATURES
# ============================================================


def test_feature_order_mismatch_is_blocked(
) -> None:

    def build(
    ) -> None:

        MLModelComparisonContract(
            candidates=[
                regression_candidate(
                    estimator_key=
                        "linear_regression"
                ),

                regression_candidate(
                    estimator_key=
                        "ridge_regression",

                    feature_columns=[
                        "tenure",
                        "age",
                        "segment",
                    ],
                ),
            ]
        )


    expect_validation_error(
        build
    )


# ============================================================
# SAME CATEGORICAL ROLES
# ============================================================


def test_categorical_role_mismatch_is_blocked(
) -> None:

    def build(
    ) -> None:

        MLModelComparisonContract(
            candidates=[
                regression_candidate(
                    estimator_key=
                        "linear_regression"
                ),

                regression_candidate(
                    estimator_key=
                        "ridge_regression",

                    categorical_feature_columns=[],
                ),
            ]
        )


    expect_validation_error(
        build
    )


# ============================================================
# SAME PREPROCESSING
# ============================================================


def test_preprocessing_mismatch_is_blocked(
) -> None:

    def build(
    ) -> None:

        MLModelComparisonContract(
            candidates=[
                regression_candidate(
                    estimator_key=
                        "linear_regression"
                ),

                regression_candidate(
                    estimator_key=
                        "ridge_regression",

                    preprocessing=(
                        MLPreprocessingContract(
                            numeric_imputation=
                                "median",

                            categorical_imputation=
                                "most_frequent",

                            scale_numeric=
                                True,
                        )
                    ),
                ),
            ]
        )


    expect_validation_error(
        build
    )


# ============================================================
# SAME SPLIT
# ============================================================


def test_split_mismatch_is_blocked(
) -> None:

    def build(
    ) -> None:

        MLModelComparisonContract(
            candidates=[
                regression_candidate(
                    estimator_key=
                        "linear_regression"
                ),

                regression_candidate(
                    estimator_key=
                        "ridge_regression",

                    split=(
                        MLSplitContract(
                            test_size=
                                0.20,

                            random_seed=
                                99,

                            shuffle=
                                True,

                            stratify=
                                False,
                        )
                    ),
                ),
            ]
        )


    expect_validation_error(
        build
    )


# ============================================================
# HYPERPARAMETER PROVENANCE
# ============================================================


def test_candidate_hyperparameters_are_preserved(
) -> None:

    contract = (
        MLModelComparisonContract(
            candidates=[
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
                            7.5,

                        "fit_intercept":
                            False,
                    },
                ),
            ]
        )
    )


    ridge = (
        contract.candidates[
            1
        ]
    )


    hyperparameters = (
        ridge
        .effective_estimator_hyperparameters
    )


    assert (
        hyperparameters
        is not None
    )


    assert (
        hyperparameters.alpha
        ==
        7.5
    )


    assert (
        hyperparameters.fit_intercept
        is False
    )


# ============================================================
# UNKNOWN COMPARISON FIELDS
# ============================================================


def test_unknown_comparison_fields_are_blocked(
) -> None:

    def build(
    ) -> None:

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
            ],

            custom_metric=
                "something",
        )


    expect_validation_error(
        build
    )


# ============================================================
# DETERMINISTIC SERIALIZATION
# ============================================================


def test_comparison_contract_serialization_is_deterministic(
) -> None:

    first = (
        MLModelComparisonContract(
            candidates=[
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
                            3.0,
                    },
                ),

                regression_candidate(
                    estimator_key=
                        "random_forest_regressor",

                    estimator_hyperparameters={
                        "kind":
                            "random_forest_regressor",

                        "n_estimators":
                            80,
                    },
                ),
            ]
        )
    )


    second = (
        MLModelComparisonContract(
            candidates=[
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
                            3.0,
                    },
                ),

                regression_candidate(
                    estimator_key=
                        "random_forest_regressor",

                    estimator_hyperparameters={
                        "kind":
                            "random_forest_regressor",

                        "n_estimators":
                            80,
                    },
                ),
            ]
        )
    )


    assert (
        first.model_dump(
            mode="json"
        )
        ==
        second.model_dump(
            mode="json"
        )
    )


# ============================================================
# RULE VERSION
# ============================================================


def test_model_comparison_contract_rule_version(
) -> None:

    assert (
        ML_MODEL_COMPARISON_CONTRACT_RULE_VERSION
        ==
        "ml_model_comparison_contract_v0.1"
    )


# ============================================================
# MAIN
# ============================================================


def main(
) -> None:

    print(
        (
            "=== DATALENS ML MODEL "
            "COMPARISON CONTRACT v0.1 ==="
        )
    )

    print()


    test_valid_regression_comparison()

    print(
        "Valid regression comparison contract: PASS"
    )


    test_valid_classification_comparison()

    print(
        "Valid classification comparison contract: PASS"
    )


    test_at_least_two_candidates_are_required()

    print(
        "At least two comparison candidates required: PASS"
    )


    test_duplicate_estimator_keys_are_blocked()

    print(
        "Duplicate estimator keys are blocked: PASS"
    )


    test_unsupported_estimator_is_blocked()

    print(
        "Unsupported estimator is blocked: PASS"
    )


    test_estimator_problem_type_mismatch_is_blocked()

    print(
        "Estimator/problem type mismatch is blocked: PASS"
    )


    test_workflow_mismatch_is_blocked()

    print(
        "Workflow authority mismatch is blocked: PASS"
    )


    test_dataset_mismatch_is_blocked()

    print(
        "Dataset authority mismatch is blocked: PASS"
    )


    test_target_mismatch_is_blocked()

    print(
        "Target mismatch is blocked: PASS"
    )


    test_feature_order_mismatch_is_blocked()

    print(
        "Ordered feature mismatch is blocked: PASS"
    )


    test_categorical_role_mismatch_is_blocked()

    print(
        "Categorical role mismatch is blocked: PASS"
    )


    test_preprocessing_mismatch_is_blocked()

    print(
        "Preprocessing mismatch is blocked: PASS"
    )


    test_split_mismatch_is_blocked()

    print(
        "Split mismatch is blocked: PASS"
    )


    test_candidate_hyperparameters_are_preserved()

    print(
        "Candidate hyperparameter provenance: PASS"
    )


    test_unknown_comparison_fields_are_blocked()

    print(
        "Unknown comparison fields are blocked: PASS"
    )


    test_comparison_contract_serialization_is_deterministic()

    print(
        "Comparison contract serialization is deterministic: PASS"
    )


    test_model_comparison_contract_rule_version()

    print(
        "ML Model Comparison Contract rule version: PASS"
    )


    print()

    print(
        "ML Model Comparison Contract v0.1: PASS"
    )


if __name__ == "__main__":
    main()