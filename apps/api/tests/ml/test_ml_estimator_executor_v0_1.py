from __future__ import annotations


import math


import numpy as np
import pandas as pd


from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
)


from sklearn.linear_model import (
    LinearRegression,
    LogisticRegression,
    Ridge,
)


import app.ml.classical_executor as executor_module


from app.ml.classical_executor import (
    ClassicalMLEstimatorError,
    execute_classical_ml,
)


from app.ml.contracts import (
    MLSplitContract,
    MLTrainingContract,
)


from app.ml.model_loader import (
    load_trusted_ml_model,
)


from tests.ml.test_classical_ml_executor_v0_1 import (
    classification_dataframe,
    isolated_environment,
    patched_handoff,
    regression_dataframe,
    seed_preparation_authority,
)


# ============================================================
# CONTRACT HELPERS
# ============================================================


def regression_contract(
    *,
    estimator_key: str,
    estimator_hyperparameters=None,
    random_seed: int = 42,
) -> MLTrainingContract:

    return (
        MLTrainingContract(
            workflow_id=
                "prep:ml-executor",

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
                estimator_key,

            estimator_hyperparameters=
                estimator_hyperparameters,

            split=(
                MLSplitContract(
                    test_size=
                        0.20,

                    random_seed=
                        random_seed,

                    shuffle=
                        True,

                    stratify=
                        False,
                )
            ),
        )
    )


def classification_contract(
    *,
    estimator_key: str,
    estimator_hyperparameters=None,
    random_seed: int = 42,
) -> MLTrainingContract:

    return (
        MLTrainingContract(
            workflow_id=
                "prep:ml-executor",

            dataset_id=
                "dataset:validated",

            problem_type=
                "classification",

            target_column=
                "churned",

            feature_columns=[
                "signal",
                "aux",
            ],

            estimator_key=
                estimator_key,

            estimator_hyperparameters=
                estimator_hyperparameters,

            split=(
                MLSplitContract(
                    test_size=
                        0.25,

                    random_seed=
                        random_seed,

                    shuffle=
                        True,

                    stratify=
                        True,
                )
            ),
        )
    )


# ============================================================
# LEGACY LINEAR / LOGISTIC DEFAULTS
# ============================================================


def test_legacy_estimators_use_server_owned_defaults(
) -> None:

    linear_pipeline = (
        executor_module
        ._build_estimator(
            contract=(
                regression_contract(
                    estimator_key=
                        "linear_regression"
                )
            )
        )
    )


    linear = (
        linear_pipeline
        .named_steps[
            "estimator"
        ]
    )


    assert isinstance(
        linear,
        LinearRegression,
    )


    assert (
        linear.fit_intercept
        is True
    )


    logistic_pipeline = (
        executor_module
        ._build_estimator(
            contract=(
                classification_contract(
                    estimator_key=
                        "logistic_regression"
                )
            )
        )
    )


    logistic = (
        logistic_pipeline
        .named_steps[
            "estimator"
        ]
    )


    assert isinstance(
        logistic,
        LogisticRegression,
    )


    assert (
        logistic.C
        ==
        1.0
    )


    assert (
        logistic.max_iter
        ==
        1000
    )


    assert (
        logistic.random_state
        ==
        42
    )


# ============================================================
# EXPLICIT LINEAR / LOGISTIC PARAMETERS
# ============================================================


def test_explicit_linear_and_logistic_parameters_are_applied(
) -> None:

    linear_pipeline = (
        executor_module
        ._build_estimator(
            contract=(
                regression_contract(
                    estimator_key=
                        "linear_regression",

                    estimator_hyperparameters={
                        "kind":
                            "linear_regression",

                        "fit_intercept":
                            False,
                    },
                )
            )
        )
    )


    linear = (
        linear_pipeline
        .named_steps[
            "estimator"
        ]
    )


    assert (
        linear.fit_intercept
        is False
    )


    logistic_pipeline = (
        executor_module
        ._build_estimator(
            contract=(
                classification_contract(
                    estimator_key=
                        "logistic_regression",

                    estimator_hyperparameters={
                        "kind":
                            "logistic_regression",

                        "inverse_regularization_strength":
                            2.5,

                        "fit_intercept":
                            False,

                        "max_iter":
                            1500,

                        "class_weight":
                            "balanced",
                    },
                )
            )
        )
    )


    logistic = (
        logistic_pipeline
        .named_steps[
            "estimator"
        ]
    )


    assert (
        logistic.C
        ==
        2.5
    )


    assert (
        logistic.fit_intercept
        is False
    )


    assert (
        logistic.max_iter
        ==
        1500
    )


    assert (
        logistic.class_weight
        ==
        "balanced"
    )


# ============================================================
# RIDGE EXECUTION + TRUSTED RELOAD
# ============================================================


def test_ridge_execution_and_reload(
) -> None:

    with isolated_environment():

        seed_preparation_authority(
            workflow_id=
                "prep:ml-executor",

            dataset_id=
                "dataset:validated",
        )


        dataframe = (
            regression_dataframe()
        )


        contract = (
            regression_contract(
                estimator_key=
                    "ridge_regression",

                estimator_hyperparameters={
                    "kind":
                        "ridge_regression",

                    "alpha":
                        2.5,

                    "fit_intercept":
                        True,
                },
            )
        )


        with patched_handoff(
            dataframe=
                dataframe,

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
            result.estimator_key
            ==
            "ridge_regression"
        )


        for value in (
            result.metrics.values()
        ):
            assert (
                math.isfinite(
                    float(
                        value
                    )
                )
            )


        loaded = (
            load_trusted_ml_model(
                workflow_id=
                    contract.workflow_id,

                model_id=
                    result
                    .model_artifact
                    .model_id,
            )
        )


        estimator = (
            loaded
            .estimator
            .named_steps[
                "estimator"
            ]
        )


        assert isinstance(
            estimator,
            Ridge,
        )


        assert (
            estimator.alpha
            ==
            2.5
        )


        predictions = (
            loaded.predict(
                pd.DataFrame(
                    {
                        "age": [
                            40.0,
                        ],

                        "tenure": [
                            2.0,
                        ],
                    }
                )
            )
        )


        assert (
            len(
                predictions
            )
            ==
            1
        )


        assert (
            np.isfinite(
                np.asarray(
                    predictions,
                    dtype=np.float64,
                )
            )
            .all()
        )


# ============================================================
# RANDOM FOREST REGRESSOR
# ============================================================


def test_random_forest_regressor_is_deterministic_and_server_owned(
) -> None:

    with isolated_environment():

        seed_preparation_authority(
            workflow_id=
                "prep:ml-executor",

            dataset_id=
                "dataset:validated",
        )


        dataframe = (
            regression_dataframe()
        )


        contract = (
            regression_contract(
                estimator_key=
                    "random_forest_regressor",

                estimator_hyperparameters={
                    "kind":
                        "random_forest_regressor",

                    "n_estimators":
                        64,

                    "max_depth":
                        8,

                    "min_samples_split":
                        2,

                    "min_samples_leaf":
                        1,

                    "max_features":
                        "sqrt",

                    "bootstrap":
                        True,
                },

                random_seed=
                    17,
            )
        )


        with patched_handoff(
            dataframe=
                dataframe,

            workflow_id=
                contract.workflow_id,

            dataset_id=
                contract.dataset_id,
        ):
            first = (
                execute_classical_ml(
                    training_contract=
                        contract
                )
            )


            second = (
                execute_classical_ml(
                    training_contract=
                        contract
                )
            )


        assert (
            first.metrics
            ==
            second.metrics
        )


        loaded = (
            load_trusted_ml_model(
                workflow_id=
                    contract.workflow_id,

                model_id=
                    first
                    .model_artifact
                    .model_id,
            )
        )


        estimator = (
            loaded
            .estimator
            .named_steps[
                "estimator"
            ]
        )


        assert isinstance(
            estimator,
            RandomForestRegressor,
        )


        assert (
            estimator.n_estimators
            ==
            64
        )


        assert (
            estimator.max_depth
            ==
            8
        )


        assert (
            estimator.random_state
            ==
            17
        )


        assert (
            estimator.n_jobs
            ==
            1
        )


        assert (
            first
            .model_artifact
            .training_contract
            ==
            contract
        )


# ============================================================
# RANDOM FOREST CLASSIFIER
# ============================================================


def test_random_forest_classifier_execution_and_reload(
) -> None:

    with isolated_environment():

        seed_preparation_authority(
            workflow_id=
                "prep:ml-executor",

            dataset_id=
                "dataset:validated",
        )


        dataframe = (
            classification_dataframe()
        )


        contract = (
            classification_contract(
                estimator_key=
                    "random_forest_classifier",

                estimator_hyperparameters={
                    "kind":
                        "random_forest_classifier",

                    "n_estimators":
                        80,

                    "max_depth":
                        6,

                    "min_samples_split":
                        2,

                    "min_samples_leaf":
                        1,

                    "max_features":
                        "sqrt",

                    "bootstrap":
                        True,

                    "class_weight":
                        "balanced",
                },

                random_seed=
                    23,
            )
        )


        with patched_handoff(
            dataframe=
                dataframe,

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
            result.estimator_key
            ==
            "random_forest_classifier"
        )


        assert (
            0.0
            <=
            result.metrics[
                "accuracy"
            ]
            <=
            1.0
        )


        assert (
            0.0
            <=
            result.metrics[
                "f1_macro"
            ]
            <=
            1.0
        )


        loaded = (
            load_trusted_ml_model(
                workflow_id=
                    contract.workflow_id,

                model_id=
                    result
                    .model_artifact
                    .model_id,
            )
        )


        estimator = (
            loaded
            .estimator
            .named_steps[
                "estimator"
            ]
        )


        assert isinstance(
            estimator,
            RandomForestClassifier,
        )


        assert (
            estimator.n_estimators
            ==
            80
        )


        assert (
            estimator.max_depth
            ==
            6
        )


        assert (
            estimator.class_weight
            ==
            "balanced"
        )


        assert (
            estimator.random_state
            ==
            23
        )


        assert (
            estimator.n_jobs
            ==
            1
        )


        predictions = (
            loaded.predict(
                pd.DataFrame(
                    {
                        "signal": [
                            -15.0,
                            15.0,
                        ],

                        "aux": [
                            0.0,
                            0.0,
                        ],
                    }
                )
            )
        )


        assert (
            len(
                predictions
            )
            ==
            2
        )


        assert set(
            predictions.tolist()
        ).issubset(
            {
                "no",
                "yes",
            }
        )


# ============================================================
# UNKNOWN ESTIMATOR
# ============================================================


def test_unknown_estimator_remains_fail_closed(
) -> None:

    contract = (
        regression_contract(
            estimator_key=
                "future_unknown_estimator"
        )
    )


    try:
        executor_module._build_estimator(
            contract=
                contract
        )

    except ClassicalMLEstimatorError:
        return


    raise AssertionError(
        (
            "Unknown estimator should remain "
            "fail-closed."
        )
    )


# ============================================================
# PROBLEM TYPE MISMATCH
# ============================================================


def test_estimator_problem_type_mismatch_remains_fail_closed(
) -> None:

    contract = (
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
                "logistic_regression",
        )
    )


    try:
        executor_module._build_estimator(
            contract=
                contract
        )

    except ClassicalMLEstimatorError:
        return


    raise AssertionError(
        (
            "Estimator/problem type mismatch "
            "should remain fail-closed."
        )
    )


# ============================================================
# MAIN
# ============================================================


def main(
) -> None:

    print(
        (
            "=== DATALENS ML ESTIMATOR "
            "EXECUTOR v0.1 ==="
        )
    )

    print()


    test_legacy_estimators_use_server_owned_defaults()

    print(
        "Legacy estimator defaults: PASS"
    )


    test_explicit_linear_and_logistic_parameters_are_applied()

    print(
        "Explicit Linear/Logistic hyperparameters: PASS"
    )


    test_ridge_execution_and_reload()

    print(
        (
            "Ridge -> train -> artifact -> "
            "trusted reload: PASS"
        )
    )


    test_random_forest_regressor_is_deterministic_and_server_owned()

    print(
        (
            "Random Forest Regressor deterministic "
            "+ server-owned controls: PASS"
        )
    )


    test_random_forest_classifier_execution_and_reload()

    print(
        (
            "Random Forest Classifier -> artifact -> "
            "trusted reload: PASS"
        )
    )


    test_unknown_estimator_remains_fail_closed()

    print(
        "Unknown estimator remains fail-closed: PASS"
    )


    test_estimator_problem_type_mismatch_remains_fail_closed()

    print(
        (
            "Estimator/problem type mismatch "
            "remains fail-closed: PASS"
        )
    )


    print()

    print(
        "ML Estimator Executor v0.1: PASS"
    )


if __name__ == "__main__":
    main()