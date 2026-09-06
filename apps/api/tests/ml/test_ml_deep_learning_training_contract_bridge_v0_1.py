from __future__ import annotations


import sys


from pydantic import (
    ValidationError,
)


from app.deep_learning.tabular_contracts import (
    DLTabularMLPRegressorHyperparameters,
)


from app.ml.contracts import (
    MLTrainingContract,
)


from app.ml.estimator_contracts import (
    MLLinearRegressionHyperparameters,
    SUPPORTED_ESTIMATOR_KEYS,
)


from app.ml.training_estimator_contracts import (
    ML_TRAINING_ESTIMATOR_BRIDGE_RULE_VERSION,
    default_training_estimator_hyperparameters,
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
# DEFAULT MLP BRIDGE
# ============================================================


def test_mlp_default_configuration(
) -> None:

    contract = (
        MLTrainingContract(
            workflow_id="prep:dl-default",
            dataset_id="dataset:validated",
            problem_type="regression",
            target_column="revenue",
            feature_columns=[
                "age",
                "tenure",
            ],
            estimator_key=
                "tabular_mlp_regressor",
        )
    )


    hyperparameters = (
        contract
        .effective_estimator_hyperparameters
    )


    assert isinstance(
        hyperparameters,
        DLTabularMLPRegressorHyperparameters,
    )


    assert (
        hyperparameters.kind
        ==
        "tabular_mlp_regressor"
    )


    assert (
        hyperparameters.hidden_features
        ==
        64
    )


# ============================================================
# EXPLICIT MLP CONFIGURATION
# ============================================================


def test_mlp_explicit_configuration(
) -> None:

    contract = (
        MLTrainingContract(
            workflow_id="prep:dl-explicit",
            dataset_id="dataset:validated",
            problem_type="regression",
            target_column="revenue",
            feature_columns=[
                "age",
                "tenure",
            ],
            estimator_key=
                "tabular_mlp_regressor",
            estimator_hyperparameters={
                "kind":
                    "tabular_mlp_regressor",
                "hidden_features":
                    128,
                "epochs":
                    250,
                "batch_size":
                    32,
                "learning_rate":
                    0.005,
            },
        )
    )


    hyperparameters = (
        contract
        .effective_estimator_hyperparameters
    )


    assert isinstance(
        hyperparameters,
        DLTabularMLPRegressorHyperparameters,
    )


    assert (
        hyperparameters.hidden_features
        ==
        128
    )


    assert (
        hyperparameters.epochs
        ==
        250
    )


# ============================================================
# IDENTITY GUARD
# ============================================================


def test_mlp_hyperparameter_identity_guard(
) -> None:

    def build(
    ) -> None:

        MLTrainingContract(
            workflow_id="prep:dl-mismatch",
            dataset_id="dataset:validated",
            problem_type="regression",
            target_column="revenue",
            feature_columns=[
                "age",
            ],
            estimator_key=
                "tabular_mlp_regressor",
            estimator_hyperparameters={
                "kind":
                    "linear_regression",
            },
        )


    expect_validation_error(
        build
    )


# ============================================================
# PROBLEM-TYPE GUARD
# ============================================================


def test_mlp_requires_regression(
) -> None:

    def build(
    ) -> None:

        MLTrainingContract(
            workflow_id="prep:dl-classification",
            dataset_id="dataset:validated",
            problem_type="classification",
            target_column="label",
            feature_columns=[
                "age",
            ],
            estimator_key=
                "tabular_mlp_regressor",
        )


    expect_validation_error(
        build
    )


# ============================================================
# CLIENT CONTROL GUARDS
# ============================================================


def test_mlp_arbitrary_configuration_is_blocked(
) -> None:

    def build(
    ) -> None:

        MLTrainingContract(
            workflow_id="prep:dl-extra",
            dataset_id="dataset:validated",
            problem_type="regression",
            target_column="revenue",
            feature_columns=[
                "age",
            ],
            estimator_key=
                "tabular_mlp_regressor",
            estimator_hyperparameters={
                "kind":
                    "tabular_mlp_regressor",
                "optimizer":
                    "adam",
            },
        )


    expect_validation_error(
        build
    )


# ============================================================
# CLASSICAL PRESERVATION
# ============================================================


def test_classical_contract_remains_supported(
) -> None:

    contract = (
        MLTrainingContract(
            workflow_id="prep:classical",
            dataset_id="dataset:validated",
            problem_type="regression",
            target_column="revenue",
            feature_columns=[
                "age",
            ],
            estimator_key=
                "linear_regression",
        )
    )


    assert isinstance(
        contract
        .effective_estimator_hyperparameters,
        MLLinearRegressionHyperparameters,
    )


def test_classical_registry_is_not_redefined(
) -> None:

    assert (
        "tabular_mlp_regressor"
        not in
        SUPPORTED_ESTIMATOR_KEYS
    )


# ============================================================
# UNKNOWN-ESTIMATOR PRESERVATION
# ============================================================


def test_unknown_estimator_remains_constructible(
) -> None:

    contract = (
        MLTrainingContract(
            workflow_id="prep:future",
            dataset_id="dataset:validated",
            problem_type="regression",
            target_column="revenue",
            feature_columns=[
                "age",
            ],
            estimator_key=
                "future_unknown_estimator",
        )
    )


    assert (
        contract
        .effective_estimator_hyperparameters
        is None
    )


# ============================================================
# DETERMINISTIC CONTRACT SERIALIZATION
# ============================================================


def test_mlp_training_contract_serialization_is_deterministic(
) -> None:

    first = (
        MLTrainingContract(
            workflow_id="prep:deterministic",
            dataset_id="dataset:validated",
            problem_type="regression",
            target_column="revenue",
            feature_columns=[
                "age",
                "tenure",
            ],
            estimator_key=
                "tabular_mlp_regressor",
        )
    )


    second = (
        MLTrainingContract(
            workflow_id="prep:deterministic",
            dataset_id="dataset:validated",
            problem_type="regression",
            target_column="revenue",
            feature_columns=[
                "age",
                "tenure",
            ],
            estimator_key=
                "tabular_mlp_regressor",
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
# DEFAULT RESOLVER
# ============================================================


def test_training_default_resolver(
) -> None:

    mlp = (
        default_training_estimator_hyperparameters(
            "tabular_mlp_regressor"
        )
    )


    classical = (
        default_training_estimator_hyperparameters(
            "linear_regression"
        )
    )


    unknown = (
        default_training_estimator_hyperparameters(
            "unknown_estimator"
        )
    )


    assert isinstance(
        mlp,
        DLTabularMLPRegressorHyperparameters,
    )


    assert isinstance(
        classical,
        MLLinearRegressionHyperparameters,
    )


    assert (
        unknown
        is None
    )


# ============================================================
# RUNTIME ISOLATION
# ============================================================


def test_runtime_bridge_does_not_load_torch(
) -> None:

    assert (
        "torch"
        not in
        sys.modules
    )


# ============================================================
# RULE VERSION
# ============================================================


def test_rule_version(
) -> None:

    assert (
        ML_TRAINING_ESTIMATOR_BRIDGE_RULE_VERSION
        ==
        "ml_training_estimator_bridge_v0.1"
    )


# ============================================================
# MAIN
# ============================================================


def main(
) -> None:

    print(
        "=== DATALENS ML / DEEP LEARNING "
        "TRAINING CONTRACT BRIDGE v0.1 ==="
    )

    print()


    test_mlp_default_configuration()

    print(
        "MLP default training configuration: PASS"
    )


    test_mlp_explicit_configuration()

    print(
        "MLP explicit training configuration: PASS"
    )


    test_mlp_hyperparameter_identity_guard()

    print(
        "Estimator / MLP identity guard: PASS"
    )


    test_mlp_requires_regression()

    print(
        "MLP regression-only authority: PASS"
    )


    test_mlp_arbitrary_configuration_is_blocked()

    print(
        "MLP arbitrary configuration guard: PASS"
    )


    test_classical_contract_remains_supported()

    print(
        "Classical training contract preservation: PASS"
    )


    test_classical_registry_is_not_redefined()

    print(
        "Classical estimator registry preserved: PASS"
    )


    test_unknown_estimator_remains_constructible()

    print(
        "Unknown-estimator fail-closed path preserved: PASS"
    )


    test_mlp_training_contract_serialization_is_deterministic()

    print(
        "MLP training contract determinism: PASS"
    )


    test_training_default_resolver()

    print(
        "Framework-neutral default resolver: PASS"
    )


    test_runtime_bridge_does_not_load_torch()

    print(
        "Runtime torch isolation: PASS"
    )


    test_rule_version()

    print(
        "Bridge rule version: PASS"
    )


    print()

    print(
        "PASS - DataLens ML / Deep Learning "
        "Training Contract Bridge v0.1"
    )


if __name__ == "__main__":
    main()
