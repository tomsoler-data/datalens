from __future__ import annotations


from pydantic import (
    ValidationError,
)


from app.ml.contracts import (
    MLSplitContract,
    MLTrainingContract,
    ML_TRAINING_CONTRACT_RULE_VERSION,
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
# VALID REGRESSION
# ============================================================


def test_valid_regression_contract(
) -> None:
    contract = MLTrainingContract(
        workflow_id=(
            " prep:ml-regression "
        ),
        dataset_id=(
            " dataset:validated "
        ),
        problem_type="regression",
        target_column=(
            " revenue "
        ),
        feature_columns=[
            "age",
            "tenure",
            "orders",
        ],
        estimator_key=(
            " linear_regression "
        ),
    )


    assert (
        contract.workflow_id
        ==
        "prep:ml-regression"
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
        contract.estimator_key
        ==
        "linear_regression"
    )


    assert (
        contract.problem_type
        ==
        "regression"
    )


    assert (
        contract.feature_columns
        ==
        [
            "age",
            "tenure",
            "orders",
        ]
    )


    assert (
        contract.split.strategy
        ==
        "holdout"
    )


    assert (
        contract.split.test_size
        ==
        0.20
    )


    assert (
        contract.split.random_seed
        ==
        42
    )


    assert (
        contract.split.shuffle
        is True
    )


    assert (
        contract.split.stratify
        is False
    )


    assert (
        contract.rule_version
        ==
        ML_TRAINING_CONTRACT_RULE_VERSION
    )


# ============================================================
# VALID CLASSIFICATION
# ============================================================


def test_valid_classification_contract(
) -> None:
    contract = MLTrainingContract(
        workflow_id="prep:classification",
        dataset_id="dataset:customers",
        problem_type="classification",
        target_column="churned",
        feature_columns=[
            "age",
            "tenure",
            "monthly_spend",
        ],
        estimator_key="logistic_regression",
        split=MLSplitContract(
            test_size=0.25,
            random_seed=123,
            shuffle=True,
            stratify=True,
        ),
    )


    assert (
        contract.problem_type
        ==
        "classification"
    )


    assert (
        contract.split.stratify
        is True
    )


    assert (
        contract.split.test_size
        ==
        0.25
    )


    assert (
        contract.split.random_seed
        ==
        123
    )


# ============================================================
# TARGET LEAKAGE
# ============================================================


def test_target_cannot_be_feature(
) -> None:

    def build(
    ) -> None:
        MLTrainingContract(
            workflow_id="prep:001",
            dataset_id="dataset:001",
            problem_type="regression",
            target_column="revenue",
            feature_columns=[
                "age",
                "revenue",
            ],
            estimator_key="linear_regression",
        )


    expect_validation_error(
        build
    )


# ============================================================
# DUPLICATE FEATURES
# ============================================================


def test_duplicate_features_are_blocked(
) -> None:

    def build(
    ) -> None:
        MLTrainingContract(
            workflow_id="prep:001",
            dataset_id="dataset:001",
            problem_type="regression",
            target_column="revenue",
            feature_columns=[
                "age",
                " age ",
            ],
            estimator_key="linear_regression",
        )


    expect_validation_error(
        build
    )


# ============================================================
# EMPTY FEATURE
# ============================================================


def test_empty_feature_name_is_blocked(
) -> None:

    def build(
    ) -> None:
        MLTrainingContract(
            workflow_id="prep:001",
            dataset_id="dataset:001",
            problem_type="regression",
            target_column="revenue",
            feature_columns=[
                "age",
                "   ",
            ],
            estimator_key="linear_regression",
        )


    expect_validation_error(
        build
    )


# ============================================================
# REGRESSION STRATIFICATION
# ============================================================


def test_regression_stratification_is_blocked(
) -> None:

    def build(
    ) -> None:
        MLTrainingContract(
            workflow_id="prep:001",
            dataset_id="dataset:001",
            problem_type="regression",
            target_column="revenue",
            feature_columns=[
                "age",
            ],
            estimator_key="linear_regression",
            split=MLSplitContract(
                stratify=True,
            ),
        )


    expect_validation_error(
        build
    )


# ============================================================
# EXTRA FIELDS
# ============================================================


def test_unknown_contract_fields_are_blocked(
) -> None:

    def build(
    ) -> None:
        MLTrainingContract(
            workflow_id="prep:001",
            dataset_id="dataset:001",
            problem_type="regression",
            target_column="revenue",
            feature_columns=[
                "age",
            ],
            estimator_key="linear_regression",
            unknown_field="not-allowed",
        )


    expect_validation_error(
        build
    )


# ============================================================
# DETERMINISM
# ============================================================


def test_contract_serialization_is_deterministic(
) -> None:
    first = MLTrainingContract(
        workflow_id="prep:001",
        dataset_id="dataset:001",
        problem_type="classification",
        target_column="churned",
        feature_columns=[
            "age",
            "tenure",
        ],
        estimator_key="logistic_regression",
        split=MLSplitContract(
            test_size=0.20,
            random_seed=42,
            shuffle=True,
            stratify=True,
        ),
    )


    second = MLTrainingContract(
        workflow_id="prep:001",
        dataset_id="dataset:001",
        problem_type="classification",
        target_column="churned",
        feature_columns=[
            "age",
            "tenure",
        ],
        estimator_key="logistic_regression",
        split=MLSplitContract(
            test_size=0.20,
            random_seed=42,
            shuffle=True,
            stratify=True,
        ),
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


def test_ml_contract_rule_version(
) -> None:
    assert (
        ML_TRAINING_CONTRACT_RULE_VERSION
        ==
        "ml_training_contract_v0.1"
    )


# ============================================================
# MAIN
# ============================================================


def main(
) -> None:
    print(
        "=== DATALENS ML TRAINING CONTRACT v0.1 ==="
    )

    print()


    test_valid_regression_contract()

    print(
        "Valid deterministic regression contract: PASS"
    )


    test_valid_classification_contract()

    print(
        "Valid deterministic classification contract: PASS"
    )


    test_target_cannot_be_feature()

    print(
        "Target leakage through feature selection is blocked: PASS"
    )


    test_duplicate_features_are_blocked()

    print(
        "Duplicate feature columns are blocked: PASS"
    )


    test_empty_feature_name_is_blocked()

    print(
        "Empty feature column names are blocked: PASS"
    )


    test_regression_stratification_is_blocked()

    print(
        "Regression stratification misuse is blocked: PASS"
    )


    test_unknown_contract_fields_are_blocked()

    print(
        "Unknown contract fields are blocked: PASS"
    )


    test_contract_serialization_is_deterministic()

    print(
        "Contract serialization is deterministic: PASS"
    )


    test_ml_contract_rule_version()

    print(
        "ML Training Contract rule version: PASS"
    )


    print()

    print(
        "ML Training Contract v0.1: PASS"
    )


if __name__ == "__main__":
    main()