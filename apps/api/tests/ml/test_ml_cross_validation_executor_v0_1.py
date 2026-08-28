from __future__ import annotations


from contextlib import (
    contextmanager,
)


import math


import pandas as pd


from app.ml.contracts import (
    MLTrainingContract,
)


from app.ml.cross_validation import (
    MLCrossValidationContract,
)


import app.ml.cross_validation_executor as cv_executor


from app.ml.cross_validation_executor import (
    MLCrossValidationInputError,
    execute_ml_cross_validation,
)


from app.ml.experiment_provenance import (
    ml_training_contract_sha256,
)


# ============================================================
# CONSTANTS
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


# ============================================================
# DATA
# ============================================================


def regression_dataframe(
    rows: int = 30,
) -> pd.DataFrame:

    records = []


    for index in range(
        rows
    ):
        x1 = float(
            index
        )

        x2 = float(
            (
                index
                %
                5
            )
            *
            2
        )

        target = (
            12.0
            +
            (
                3.5
                *
                x1
            )
            -
            (
                1.25
                *
                x2
            )
        )


        records.append(
            {
                "x1":
                    x1,

                "x2":
                    x2,

                "target":
                    target,
            }
        )


    return pd.DataFrame(
        records
    )


def classification_dataframe(
) -> pd.DataFrame:

    records = []


    for index in range(
        30
    ):
        target = (
            "A"

            if (
                index
                %
                2
                ==
                0
            )

            else
            "B"
        )


        records.append(
            {
                "x1":
                    float(
                        index
                    ),

                "x2":
                    float(
                        (
                            index
                            %
                            3
                        )
                    ),

                "target":
                    target,
            }
        )


    return pd.DataFrame(
        records
    )


# ============================================================
# CONTRACTS
# ============================================================


def regression_contract(
) -> MLTrainingContract:

    return (
        MLTrainingContract(
            workflow_id=
                "prep:cv-regression",

            dataset_id=
                "dataset:cv-regression",

            problem_type=
                "regression",

            target_column=
                "target",

            feature_columns=[
                "x1",
                "x2",
            ],

            estimator_key=
                "linear_regression",
        )
    )


def classification_contract(
) -> MLTrainingContract:

    return (
        MLTrainingContract(
            workflow_id=
                "prep:cv-classification",

            dataset_id=
                "dataset:cv-classification",

            problem_type=
                "classification",

            target_column=
                "target",

            feature_columns=[
                "x1",
                "x2",
            ],

            estimator_key=
                "logistic_regression",
        )
    )


# ============================================================
# SERVER-OWNED DATAFRAME PATCH
# ============================================================


@contextmanager
def authorized_dataframe(
    dataframe: pd.DataFrame,
    *,
    revision: int = 7,
):

    original = (
        cv_executor
        ._load_authorized_dataframe
    )


    def fake_load_authorized_dataframe(
        *,
        contract,
    ):
        return (
            dataframe.copy(
                deep=True
            ),
            revision,
        )


    cv_executor._load_authorized_dataframe = (
        fake_load_authorized_dataframe
    )


    try:
        yield

    finally:
        cv_executor._load_authorized_dataframe = (
            original
        )


# ============================================================
# REGRESSION
# ============================================================


def test_regression_kfold_richer_metrics(
) -> None:

    training = (
        regression_contract()
    )


    cv = (
        MLCrossValidationContract(
            folds=5,
            shuffle=True,
            random_seed=123,
        )
    )


    build_count = 0

    original_builder = (
        cv_executor
        ._build_estimator
    )


    def counting_builder(
        *,
        contract,
    ):
        nonlocal build_count

        build_count += 1

        return (
            original_builder(
                contract=
                    contract
            )
        )


    cv_executor._build_estimator = (
        counting_builder
    )


    try:
        with authorized_dataframe(
            regression_dataframe(),
            revision=11,
        ):
            result = (
                execute_ml_cross_validation(
                    training_contract=
                        training,

                    cross_validation_contract=
                        cv,
                )
            )

    finally:
        cv_executor._build_estimator = (
            original_builder
        )


    assert (
        result.problem_type
        ==
        "regression"
    )


    assert (
        result.strategy
        ==
        "k_fold"
    )


    assert (
        result.folds
        ==
        5
    )


    assert (
        result.shuffle
        is True
    )


    assert (
        result.random_seed
        ==
        123
    )


    assert (
        result.preparation_session_revision
        ==
        11
    )


    assert (
        result.training_contract_sha256
        ==
        ml_training_contract_sha256(
            training
        )
    )


    assert (
        build_count
        ==
        5
    )


    assert (
        len(
            result.fold_results
        )
        ==
        5
    )


    for fold in (
        result.fold_results
    ):
        assert (
            set(
                fold.metrics
            )
            ==
            REGRESSION_METRIC_NAMES
        )


        assert (
            fold.train_rows
            +
            fold.validation_rows
            ==
            30
        )


        for value in (
            fold.metrics.values()
        ):
            assert (
                math.isfinite(
                    float(
                        value
                    )
                )
            )


    assert (
        set(
            result.metric_summary
        )
        ==
        REGRESSION_METRIC_NAMES
    )


    for summary in (
        result
        .metric_summary
        .values()
    ):
        assert (
            math.isfinite(
                summary.mean
            )
        )

        assert (
            math.isfinite(
                summary.std
            )
        )

        assert (
            summary.std
            >=
            0.0
        )


# ============================================================
# CLASSIFICATION
# ============================================================


def test_classification_uses_stratified_kfold(
) -> None:

    training = (
        classification_contract()
    )


    cv = (
        MLCrossValidationContract(
            folds=5,
            shuffle=True,
            random_seed=456,
        )
    )


    with authorized_dataframe(
        classification_dataframe(),
        revision=9,
    ):
        result = (
            execute_ml_cross_validation(
                training_contract=
                    training,

                cross_validation_contract=
                    cv,
            )
        )


    assert (
        result.problem_type
        ==
        "classification"
    )


    assert (
        result.strategy
        ==
        "stratified_k_fold"
    )


    assert (
        result.folds
        ==
        5
    )


    for fold in (
        result.fold_results
    ):
        assert (
            set(
                fold.metrics
            )
            ==
            CLASSIFICATION_METRIC_NAMES
        )


        assert (
            fold.validation_rows
            ==
            6
        )


    assert (
        set(
            result.metric_summary
        )
        ==
        CLASSIFICATION_METRIC_NAMES
    )


# ============================================================
# DETERMINISM
# ============================================================


def test_same_seed_produces_same_result(
) -> None:

    training = (
        regression_contract()
    )


    cv = (
        MLCrossValidationContract(
            folds=5,
            shuffle=True,
            random_seed=987,
        )
    )


    dataframe = (
        regression_dataframe()
    )


    with authorized_dataframe(
        dataframe,
        revision=3,
    ):
        first = (
            execute_ml_cross_validation(
                training_contract=
                    training,

                cross_validation_contract=
                    cv,
            )
        )


    with authorized_dataframe(
        dataframe,
        revision=3,
    ):
        second = (
            execute_ml_cross_validation(
                training_contract=
                    training,

                cross_validation_contract=
                    cv,
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
# FAIL CLOSED
# ============================================================


def test_regression_requires_two_validation_rows_per_fold(
) -> None:

    training = (
        regression_contract()
    )


    cv = (
        MLCrossValidationContract(
            folds=5
        )
    )


    try:
        with authorized_dataframe(
            regression_dataframe(
                rows=9
            )
        ):
            execute_ml_cross_validation(
                training_contract=
                    training,

                cross_validation_contract=
                    cv,
            )

    except MLCrossValidationInputError:
        return


    raise AssertionError(
        (
            "Regression CV must reject folds "
            "that could contain fewer than "
            "two validation observations."
        )
    )


def test_classification_requires_each_class_per_fold(
) -> None:

    training = (
        classification_contract()
    )


    dataframe = pd.DataFrame(
        {
            "x1":
                [
                    float(
                        index
                    )

                    for index
                    in range(
                        10
                    )
                ],

            "x2":
                [
                    0.0
                    for _
                    in range(
                        10
                    )
                ],

            "target":
                (
                    [
                        "A"
                    ]
                    *
                    8
                    +
                    [
                        "B"
                    ]
                    *
                    2
                ),
        }
    )


    cv = (
        MLCrossValidationContract(
            folds=3
        )
    )


    try:
        with authorized_dataframe(
            dataframe
        ):
            execute_ml_cross_validation(
                training_contract=
                    training,

                cross_validation_contract=
                    cv,
            )

    except MLCrossValidationInputError:
        return


    raise AssertionError(
        (
            "Stratified CV must reject a fold "
            "count larger than the smallest "
            "target class."
        )
    )


# ============================================================
# SHUFFLE FALSE
# ============================================================


def test_shuffle_false_is_supported_without_random_state_error(
) -> None:

    training = (
        regression_contract()
    )


    cv = (
        MLCrossValidationContract(
            folds=3,
            shuffle=False,
            random_seed=999,
        )
    )


    with authorized_dataframe(
        regression_dataframe()
    ):
        result = (
            execute_ml_cross_validation(
                training_contract=
                    training,

                cross_validation_contract=
                    cv,
            )
        )


    assert (
        result.shuffle
        is False
    )


    assert (
        result.random_seed
        ==
        999
    )


    assert (
        result.folds
        ==
        3
    )


# ============================================================
# MAIN
# ============================================================


def main(
) -> None:

    print(
        "=== DATALENS ML CROSS-VALIDATION EXECUTOR v0.1 ==="
    )


    tests = [
        (
            "Regression KFold + five richer metrics",
            test_regression_kfold_richer_metrics,
        ),
        (
            "Classification StratifiedKFold + five richer metrics",
            test_classification_uses_stratified_kfold,
        ),
        (
            "Same seed is deterministic",
            test_same_seed_produces_same_result,
        ),
        (
            "Regression minimum fold size fail-closed",
            test_regression_requires_two_validation_rows_per_fold,
        ),
        (
            "Classification minimum class count fail-closed",
            test_classification_requires_each_class_per_fold,
        ),
        (
            "shuffle=False deterministic splitter",
            test_shuffle_false_is_supported_without_random_state_error,
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
        "PASS - ML Cross-Validation Executor v0.1"
    )


if __name__ == "__main__":
    main()
