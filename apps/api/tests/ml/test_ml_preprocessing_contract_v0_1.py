from __future__ import annotations


from pydantic import (
    ValidationError,
)


from app.ml.contracts import (
    MLPreprocessingContract,
    MLTrainingContract,
    ML_PREPROCESSING_CONTRACT_RULE_VERSION,
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


# ============================================================
# LEGACY NUMERIC DEFAULT
# ============================================================


def test_numeric_only_contract_remains_backward_compatible(
) -> None:
    contract = (
        MLTrainingContract(
            workflow_id=
                "prep:numeric",

            dataset_id=
                "dataset:validated",

            problem_type=
                "regression",

            target_column=
                "revenue",

            feature_columns=[
                "age",
                "tenure",
            ],

            estimator_key=
                "linear_regression",
        )
    )


    assert (
        contract.feature_columns
        ==
        [
            "age",
            "tenure",
        ]
    )


    assert (
        contract.categorical_feature_columns
        ==
        []
    )


    assert (
        contract.numeric_feature_columns
        ==
        [
            "age",
            "tenure",
        ]
    )


    assert (
        contract.preprocessing.numeric_imputation
        ==
        "error"
    )


    assert (
        contract.preprocessing.categorical_imputation
        ==
        "error"
    )


    assert (
        contract.preprocessing.categorical_encoding
        ==
        "one_hot"
    )


    assert (
        contract.preprocessing.handle_unknown_categories
        ==
        "ignore"
    )


    assert (
        contract.preprocessing.scale_numeric
        is True
    )


# ============================================================
# MIXED FEATURE ROLES
# ============================================================


def test_mixed_numeric_and_categorical_feature_roles(
) -> None:
    contract = (
        MLTrainingContract(
            workflow_id=
                "prep:mixed",

            dataset_id=
                "dataset:customers",

            problem_type=
                "classification",

            target_column=
                "churned",

            feature_columns=[
                "age",
                "country",
                "monthly_spend",
                "segment",
            ],

            categorical_feature_columns=[
                "country",
                "segment",
            ],

            estimator_key=
                "logistic_regression",

            preprocessing=(
                MLPreprocessingContract(
                    numeric_imputation=
                        "median",

                    categorical_imputation=
                        "most_frequent",

                    categorical_encoding=
                        "one_hot",

                    handle_unknown_categories=
                        "ignore",

                    scale_numeric=
                        True,
                )
            ),
        )
    )


    assert (
        contract.categorical_feature_columns
        ==
        [
            "country",
            "segment",
        ]
    )


    assert (
        contract.numeric_feature_columns
        ==
        [
            "age",
            "monthly_spend",
        ]
    )


    assert (
        contract.preprocessing.numeric_imputation
        ==
        "median"
    )


    assert (
        contract.preprocessing.categorical_imputation
        ==
        "most_frequent"
    )


# ============================================================
# NORMALIZATION
# ============================================================


def test_categorical_feature_names_are_normalized(
) -> None:
    contract = (
        MLTrainingContract(
            workflow_id=
                "prep:normalize",

            dataset_id=
                "dataset:customers",

            problem_type=
                "classification",

            target_column=
                "churned",

            feature_columns=[
                "age",
                "country",
            ],

            categorical_feature_columns=[
                " country ",
            ],

            estimator_key=
                "logistic_regression",
        )
    )


    assert (
        contract.categorical_feature_columns
        ==
        [
            "country",
        ]
    )


    assert (
        contract.numeric_feature_columns
        ==
        [
            "age",
        ]
    )


# ============================================================
# UNKNOWN CATEGORICAL FEATURE
# ============================================================


def test_categorical_feature_must_belong_to_feature_columns(
) -> None:

    def build(
    ) -> None:
        MLTrainingContract(
            workflow_id=
                "prep:invalid",

            dataset_id=
                "dataset:customers",

            problem_type=
                "classification",

            target_column=
                "churned",

            feature_columns=[
                "age",
            ],

            categorical_feature_columns=[
                "country",
            ],

            estimator_key=
                "logistic_regression",
        )


    expect_validation_error(
        build
    )


# ============================================================
# DUPLICATE CATEGORICAL FEATURE
# ============================================================


def test_duplicate_categorical_features_are_blocked(
) -> None:

    def build(
    ) -> None:
        MLTrainingContract(
            workflow_id=
                "prep:duplicates",

            dataset_id=
                "dataset:customers",

            problem_type=
                "classification",

            target_column=
                "churned",

            feature_columns=[
                "age",
                "country",
            ],

            categorical_feature_columns=[
                "country",
                " country ",
            ],

            estimator_key=
                "logistic_regression",
        )


    expect_validation_error(
        build
    )


# ============================================================
# INVALID PREPROCESSING STRATEGY
# ============================================================


def test_unknown_numeric_imputation_strategy_is_blocked(
) -> None:

    def build(
    ) -> None:
        MLPreprocessingContract(
            numeric_imputation=
                "mean"
        )


    expect_validation_error(
        build
    )


def test_unknown_categorical_imputation_strategy_is_blocked(
) -> None:

    def build(
    ) -> None:
        MLPreprocessingContract(
            categorical_imputation=
                "constant"
        )


    expect_validation_error(
        build
    )


# ============================================================
# EXTRA FIELDS
# ============================================================


def test_unknown_preprocessing_fields_are_blocked(
) -> None:

    def build(
    ) -> None:
        MLPreprocessingContract(
            learned_median=
                123.0
        )


    expect_validation_error(
        build
    )


# ============================================================
# NO LEARNED STATISTICS IN CONTRACT
# ============================================================


def test_preprocessing_contract_contains_policy_only(
) -> None:
    contract = (
        MLPreprocessingContract(
            numeric_imputation=
                "median",

            categorical_imputation=
                "most_frequent",
        )
    )


    payload = (
        contract.model_dump(
            mode="json"
        )
    )


    assert (
        "median_values"
        not in
        payload
    )


    assert (
        "category_vocabulary"
        not in
        payload
    )


    assert (
        "scaler_mean"
        not in
        payload
    )


# ============================================================
# DETERMINISTIC SERIALIZATION
# ============================================================


def test_preprocessing_contract_serialization_is_deterministic(
) -> None:
    first = (
        MLTrainingContract(
            workflow_id=
                "prep:deterministic",

            dataset_id=
                "dataset:customers",

            problem_type=
                "classification",

            target_column=
                "churned",

            feature_columns=[
                "age",
                "country",
            ],

            categorical_feature_columns=[
                "country",
            ],

            estimator_key=
                "logistic_regression",

            preprocessing=(
                MLPreprocessingContract(
                    numeric_imputation=
                        "median",

                    categorical_imputation=
                        "most_frequent",
                )
            ),
        )
    )


    second = (
        MLTrainingContract(
            workflow_id=
                "prep:deterministic",

            dataset_id=
                "dataset:customers",

            problem_type=
                "classification",

            target_column=
                "churned",

            feature_columns=[
                "age",
                "country",
            ],

            categorical_feature_columns=[
                "country",
            ],

            estimator_key=
                "logistic_regression",

            preprocessing=(
                MLPreprocessingContract(
                    numeric_imputation=
                        "median",

                    categorical_imputation=
                        "most_frequent",
                )
            ),
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


def test_preprocessing_contract_rule_version(
) -> None:
    assert (
        ML_PREPROCESSING_CONTRACT_RULE_VERSION
        ==
        "ml_preprocessing_contract_v0.1"
    )


# ============================================================
# MAIN
# ============================================================


def main(
) -> None:
    print(
        "=== DATALENS ML PREPROCESSING CONTRACT v0.1 ==="
    )

    print()


    test_numeric_only_contract_remains_backward_compatible()

    print(
        "Numeric-only Training Contract remains compatible: PASS"
    )


    test_mixed_numeric_and_categorical_feature_roles()

    print(
        "Mixed numeric/categorical feature roles: PASS"
    )


    test_categorical_feature_names_are_normalized()

    print(
        "Categorical feature names are normalized: PASS"
    )


    test_categorical_feature_must_belong_to_feature_columns()

    print(
        "Categorical feature scope is enforced: PASS"
    )


    test_duplicate_categorical_features_are_blocked()

    print(
        "Duplicate categorical feature roles are blocked: PASS"
    )


    test_unknown_numeric_imputation_strategy_is_blocked()

    print(
        "Unsupported numeric imputation strategy is blocked: PASS"
    )


    test_unknown_categorical_imputation_strategy_is_blocked()

    print(
        "Unsupported categorical imputation strategy is blocked: PASS"
    )


    test_unknown_preprocessing_fields_are_blocked()

    print(
        "Unknown preprocessing fields are blocked: PASS"
    )


    test_preprocessing_contract_contains_policy_only()

    print(
        "Learned statistics cannot enter preprocessing contract: PASS"
    )


    test_preprocessing_contract_serialization_is_deterministic()

    print(
        "Preprocessing contract serialization is deterministic: PASS"
    )


    test_preprocessing_contract_rule_version()

    print(
        "ML Preprocessing Contract rule version: PASS"
    )


    print()

    print(
        "ML Preprocessing Contract v0.1: PASS"
    )


if __name__ == "__main__":
    main()