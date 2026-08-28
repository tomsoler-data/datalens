from __future__ import annotations


from pydantic import (
    ValidationError,
)


from app.ml.contracts import (
    MLTrainingContract,
)


from app.ml.estimator_contracts import (
    MLLinearRegressionHyperparameters,
    MLLogisticRegressionHyperparameters,
    MLRandomForestClassifierHyperparameters,
    MLRandomForestRegressorHyperparameters,
    MLRidgeRegressionHyperparameters,
    ML_ESTIMATOR_CONTRACT_RULE_VERSION,
    SUPPORTED_ESTIMATOR_KEYS,
    default_estimator_hyperparameters,
    estimator_problem_type,
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
# SUPPORTED ESTIMATORS
# ============================================================


def test_supported_estimator_registry(
) -> None:
    assert (
        SUPPORTED_ESTIMATOR_KEYS
        ==
        (
            "linear_regression",
            "ridge_regression",
            "logistic_regression",
            "random_forest_regressor",
            "random_forest_classifier",
        )
    )


    assert (
        estimator_problem_type(
            "linear_regression"
        )
        ==
        "regression"
    )


    assert (
        estimator_problem_type(
            "ridge_regression"
        )
        ==
        "regression"
    )


    assert (
        estimator_problem_type(
            "random_forest_regressor"
        )
        ==
        "regression"
    )


    assert (
        estimator_problem_type(
            "logistic_regression"
        )
        ==
        "classification"
    )


    assert (
        estimator_problem_type(
            "random_forest_classifier"
        )
        ==
        "classification"
    )


    assert (
        estimator_problem_type(
            "unknown_estimator"
        )
        is None
    )


# ============================================================
# DEFAULTS
# ============================================================


def test_server_owned_estimator_defaults(
) -> None:
    linear = (
        default_estimator_hyperparameters(
            "linear_regression"
        )
    )


    ridge = (
        default_estimator_hyperparameters(
            "ridge_regression"
        )
    )


    logistic = (
        default_estimator_hyperparameters(
            "logistic_regression"
        )
    )


    forest_regressor = (
        default_estimator_hyperparameters(
            "random_forest_regressor"
        )
    )


    forest_classifier = (
        default_estimator_hyperparameters(
            "random_forest_classifier"
        )
    )


    assert isinstance(
        linear,
        MLLinearRegressionHyperparameters,
    )


    assert isinstance(
        ridge,
        MLRidgeRegressionHyperparameters,
    )


    assert isinstance(
        logistic,
        MLLogisticRegressionHyperparameters,
    )


    assert isinstance(
        forest_regressor,
        MLRandomForestRegressorHyperparameters,
    )


    assert isinstance(
        forest_classifier,
        MLRandomForestClassifierHyperparameters,
    )


    assert (
        ridge.alpha
        ==
        1.0
    )


    assert (
        logistic.inverse_regularization_strength
        ==
        1.0
    )


    assert (
        forest_regressor.n_estimators
        ==
        200
    )


    assert (
        forest_classifier.n_estimators
        ==
        200
    )


# ============================================================
# LEGACY TRAINING CONTRACT
# ============================================================


def test_legacy_training_contract_remains_compatible(
) -> None:
    contract = (
        MLTrainingContract(
            workflow_id=
                "prep:legacy",

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
        contract.estimator_key
        ==
        "linear_regression"
    )


    assert (
        contract.estimator_hyperparameters
        is None
    )


    assert isinstance(
        contract.effective_estimator_hyperparameters,
        MLLinearRegressionHyperparameters,
    )


# ============================================================
# UNKNOWN LEGACY ESTIMATOR
# ============================================================


def test_unknown_estimator_remains_executor_authority(
) -> None:
    contract = (
        MLTrainingContract(
            workflow_id=
                "prep:legacy-unknown",

            dataset_id=
                "dataset:validated",

            problem_type=
                "regression",

            target_column=
                "revenue",

            feature_columns=[
                "age",
            ],

            estimator_key=
                "future_unknown_estimator",
        )
    )


    assert (
        contract.estimator_key
        ==
        "future_unknown_estimator"
    )


    assert (
        contract.effective_estimator_hyperparameters
        is None
    )


# ============================================================
# RIDGE CONTRACT
# ============================================================


def test_explicit_ridge_hyperparameters(
) -> None:
    contract = (
        MLTrainingContract(
            workflow_id=
                "prep:ridge",

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
                "ridge_regression",

            estimator_hyperparameters={
                "kind":
                    "ridge_regression",

                "alpha":
                    2.5,

                "fit_intercept":
                    False,
            },
        )
    )


    hyperparameters = (
        contract
        .effective_estimator_hyperparameters
    )


    assert isinstance(
        hyperparameters,
        MLRidgeRegressionHyperparameters,
    )


    assert (
        hyperparameters.alpha
        ==
        2.5
    )


    assert (
        hyperparameters.fit_intercept
        is False
    )


# ============================================================
# RANDOM FOREST CONTRACT
# ============================================================


def test_explicit_random_forest_hyperparameters(
) -> None:
    contract = (
        MLTrainingContract(
            workflow_id=
                "prep:forest",

            dataset_id=
                "dataset:validated",

            problem_type=
                "classification",

            target_column=
                "churned",

            feature_columns=[
                "age",
                "segment",
            ],

            categorical_feature_columns=[
                "segment",
            ],

            estimator_key=
                "random_forest_classifier",

            estimator_hyperparameters={
                "kind":
                    "random_forest_classifier",

                "n_estimators":
                    300,

                "max_depth":
                    12,

                "min_samples_split":
                    4,

                "min_samples_leaf":
                    2,

                "max_features":
                    "sqrt",

                "bootstrap":
                    True,

                "class_weight":
                    "balanced",
            },
        )
    )


    hyperparameters = (
        contract
        .effective_estimator_hyperparameters
    )


    assert isinstance(
        hyperparameters,
        MLRandomForestClassifierHyperparameters,
    )


    assert (
        hyperparameters.n_estimators
        ==
        300
    )


    assert (
        hyperparameters.max_depth
        ==
        12
    )


    assert (
        hyperparameters.class_weight
        ==
        "balanced"
    )


# ============================================================
# IDENTITY MISMATCH
# ============================================================


def test_hyperparameter_kind_must_match_estimator_key(
) -> None:

    def build(
    ) -> None:
        MLTrainingContract(
            workflow_id=
                "prep:mismatch",

            dataset_id=
                "dataset:validated",

            problem_type=
                "regression",

            target_column=
                "revenue",

            feature_columns=[
                "age",
            ],

            estimator_key=
                "ridge_regression",

            estimator_hyperparameters={
                "kind":
                    "linear_regression",
            },
        )


    expect_validation_error(
        build
    )


# ============================================================
# BOUNDS
# ============================================================


def test_ridge_alpha_bounds_are_enforced(
) -> None:

    def zero_alpha(
    ) -> None:
        MLRidgeRegressionHyperparameters(
            alpha=0.0
        )


    def excessive_alpha(
    ) -> None:
        MLRidgeRegressionHyperparameters(
            alpha=1_000_001.0
        )


    expect_validation_error(
        zero_alpha
    )


    expect_validation_error(
        excessive_alpha
    )


def test_logistic_bounds_are_enforced(
) -> None:

    def invalid_strength(
    ) -> None:
        MLLogisticRegressionHyperparameters(
            inverse_regularization_strength=
                0.0
        )


    def invalid_iterations(
    ) -> None:
        MLLogisticRegressionHyperparameters(
            max_iter=
                50
        )


    expect_validation_error(
        invalid_strength
    )


    expect_validation_error(
        invalid_iterations
    )


def test_random_forest_bounds_are_enforced(
) -> None:

    def too_few_trees(
    ) -> None:
        MLRandomForestRegressorHyperparameters(
            n_estimators=
                9
        )


    def excessive_trees(
    ) -> None:
        MLRandomForestRegressorHyperparameters(
            n_estimators=
                2001
        )


    def invalid_depth(
    ) -> None:
        MLRandomForestRegressorHyperparameters(
            max_depth=
                0
        )


    def invalid_split(
    ) -> None:
        MLRandomForestRegressorHyperparameters(
            min_samples_split=
                1
        )


    def invalid_leaf(
    ) -> None:
        MLRandomForestRegressorHyperparameters(
            min_samples_leaf=
                0
        )


    expect_validation_error(
        too_few_trees
    )


    expect_validation_error(
        excessive_trees
    )


    expect_validation_error(
        invalid_depth
    )


    expect_validation_error(
        invalid_split
    )


    expect_validation_error(
        invalid_leaf
    )


# ============================================================
# SERVER-OWNED EXECUTION CONTROLS
# ============================================================


def test_random_state_cannot_be_client_supplied(
) -> None:

    def build(
    ) -> None:
        MLRandomForestRegressorHyperparameters(
            n_estimators=
                100,

            random_state=
                123,
        )


    expect_validation_error(
        build
    )


def test_n_jobs_cannot_be_client_supplied(
) -> None:

    def build(
    ) -> None:
        MLRandomForestClassifierHyperparameters(
            n_estimators=
                100,

            n_jobs=
                -1,
        )


    expect_validation_error(
        build
    )


# ============================================================
# ARBITRARY SKLEARN KWARGS
# ============================================================


def test_arbitrary_estimator_kwargs_are_blocked(
) -> None:

    def build(
    ) -> None:
        MLRidgeRegressionHyperparameters(
            alpha=
                1.0,

            solver=
                "sag",
        )


    expect_validation_error(
        build
    )


# ============================================================
# DETERMINISM
# ============================================================


def test_estimator_contract_serialization_is_deterministic(
) -> None:
    first = (
        MLTrainingContract(
            workflow_id=
                "prep:deterministic",

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
                "ridge_regression",

            estimator_hyperparameters={
                "kind":
                    "ridge_regression",

                "alpha":
                    3.0,
            },
        )
    )


    second = (
        MLTrainingContract(
            workflow_id=
                "prep:deterministic",

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
                "ridge_regression",

            estimator_hyperparameters={
                "kind":
                    "ridge_regression",

                "alpha":
                    3.0,
            },
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


def test_estimator_contract_rule_version(
) -> None:
    assert (
        ML_ESTIMATOR_CONTRACT_RULE_VERSION
        ==
        "ml_estimator_contract_v0.1"
    )


# ============================================================
# MAIN
# ============================================================


def main(
) -> None:
    print(
        "=== DATALENS ML ESTIMATOR CONTRACT v0.1 ==="
    )

    print()


    test_supported_estimator_registry()

    print(
        "Supported estimator registry: PASS"
    )


    test_server_owned_estimator_defaults()

    print(
        "Server-owned estimator defaults: PASS"
    )


    test_legacy_training_contract_remains_compatible()

    print(
        "Legacy ML Training Contract compatibility: PASS"
    )


    test_unknown_estimator_remains_executor_authority()

    print(
        "Unknown estimator remains executor fail-closed authority: PASS"
    )


    test_explicit_ridge_hyperparameters()

    print(
        "Typed Ridge hyperparameters: PASS"
    )


    test_explicit_random_forest_hyperparameters()

    print(
        "Typed Random Forest hyperparameters: PASS"
    )


    test_hyperparameter_kind_must_match_estimator_key()

    print(
        "Estimator/hyperparameter identity mismatch is blocked: PASS"
    )


    test_ridge_alpha_bounds_are_enforced()

    print(
        "Ridge hyperparameter bounds: PASS"
    )


    test_logistic_bounds_are_enforced()

    print(
        "Logistic Regression hyperparameter bounds: PASS"
    )


    test_random_forest_bounds_are_enforced()

    print(
        "Random Forest hyperparameter bounds: PASS"
    )


    test_random_state_cannot_be_client_supplied()

    print(
        "Client random_state injection is blocked: PASS"
    )


    test_n_jobs_cannot_be_client_supplied()

    print(
        "Client n_jobs injection is blocked: PASS"
    )


    test_arbitrary_estimator_kwargs_are_blocked()

    print(
        "Arbitrary sklearn kwargs are blocked: PASS"
    )


    test_estimator_contract_serialization_is_deterministic()

    print(
        "Estimator contract serialization is deterministic: PASS"
    )


    test_estimator_contract_rule_version()

    print(
        "ML Estimator Contract rule version: PASS"
    )


    print()

    print(
        "ML Estimator Contract v0.1: PASS"
    )


if __name__ == "__main__":
    main()