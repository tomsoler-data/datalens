from __future__ import annotations


import math


from unittest.mock import (
    patch,
)


import numpy as np
import pandas as pd


import app.ml.cross_validation_executor as cross_validation_executor
import app.ml.hyperparameter_tuning_executor as hyperparameter_tuning_executor


from app.ml.classical_executor import (
    _split_dataset,
    _validate_and_extract_xy,
    _validated_group_values,
)


from app.ml.contracts import (
    MLGroupHoldoutSplitContract,
    MLSplitContract,
    MLTrainingContract,
)


from app.ml.cross_validation import (
    MLCrossValidationContract,
    cross_validation_strategy,
)


from app.ml.cross_validation_executor import (
    MLCrossValidationInputError,
    _build_cross_validation_pairs,
    _validate_cross_validation_feasibility,
    execute_ml_cross_validation,
)


from app.ml.hyperparameter_tuning import (
    MLHyperparameterSearchContract,
    hyperparameter_validation_strategy,
)


from app.ml.hyperparameter_tuning_executor import (
    execute_ml_hyperparameter_tuning,
)


WORKFLOW_ID = (
    "prep:group-aware-cv-tuning"
)


CLASSIFICATION_DATASET_ID = (
    "dataset:group-aware-classification"
)


REGRESSION_DATASET_ID = (
    "dataset:group-aware-regression"
)


def classification_dataframe(
) -> pd.DataFrame:

    rows = []


    for group_index in range(
        24
    ):

        for within_group in range(
            4
        ):

            rows.append(
                {
                    "row_id":
                        group_index
                        *
                        4
                        +
                        within_group,

                    "client_id":
                        (
                            f"client_"
                            f"{group_index:02d}"
                        ),

                    "age":
                        (
                            20
                            +
                            group_index
                        ),

                    "categ":
                        (
                            within_group
                            %
                            2
                        ),
                }
            )


    return pd.DataFrame(
        rows
    )


def regression_dataframe(
) -> pd.DataFrame:

    rows = []


    for group_index in range(
        24
    ):

        for within_group in range(
            4
        ):

            age = (
                20
                +
                group_index
            )


            rows.append(
                {
                    "row_id":
                        group_index
                        *
                        4
                        +
                        within_group,

                    "client_id":
                        (
                            f"client_"
                            f"{group_index:02d}"
                        ),

                    "age":
                        age,

                    "target":
                        (
                            50.0
                            +
                            float(age)
                            *
                            2.0
                            +
                            float(within_group)
                            *
                            0.25
                        ),
                }
            )


    return pd.DataFrame(
        rows
    )


def classification_group_contract(
) -> MLTrainingContract:

    return (
        MLTrainingContract(
            workflow_id=
                WORKFLOW_ID,

            dataset_id=
                CLASSIFICATION_DATASET_ID,

            problem_type=
                "classification",

            target_column=
                "categ",

            feature_columns=[
                "age",
            ],

            categorical_feature_columns=[],

            estimator_key=
                "logistic_regression",

            split=
                MLGroupHoldoutSplitContract(
                    group_column=
                        "client_id",

                    test_size=
                        0.2,

                    random_seed=
                        42,
                ),
        )
    )


def regression_group_contract(
) -> MLTrainingContract:

    return (
        MLTrainingContract(
            workflow_id=
                WORKFLOW_ID,

            dataset_id=
                REGRESSION_DATASET_ID,

            problem_type=
                "regression",

            target_column=
                "target",

            feature_columns=[
                "age",
            ],

            categorical_feature_columns=[],

            estimator_key=
                "ridge_regression",

            split=
                MLGroupHoldoutSplitContract(
                    group_column=
                        "client_id",

                    test_size=
                        0.2,

                    random_seed=
                        42,
                ),
        )
    )


def classification_row_contract(
) -> MLTrainingContract:

    return (
        MLTrainingContract(
            workflow_id=
                WORKFLOW_ID,

            dataset_id=
                CLASSIFICATION_DATASET_ID,

            problem_type=
                "classification",

            target_column=
                "categ",

            feature_columns=[
                "age",
            ],

            categorical_feature_columns=[],

            estimator_key=
                "logistic_regression",

            split=
                MLSplitContract(
                    test_size=
                        0.2,

                    random_seed=
                        42,

                    shuffle=
                        True,

                    stratify=
                        True,
                ),
        )
    )


def material(
    *,
    dataframe: pd.DataFrame,
    contract: MLTrainingContract,
):

    (
        x,
        y,
    ) = (
        _validate_and_extract_xy(
            dataframe=
                dataframe,

            contract=
                contract,
        )
    )


    groups = (
        _validated_group_values(
            dataframe=
                dataframe,

            x=
                x,

            y=
                y,

            contract=
                contract,
        )
    )


    return (
        x,
        y,
        groups,
    )


def assert_zero_entity_overlap(
    *,
    pairs,
    groups: pd.Series,
) -> None:

    validation_seen = set()


    for (
        train_indices,
        validation_indices,
    ) in pairs:

        train_groups = set(
            groups.iloc[
                train_indices
            ].tolist()
        )

        validation_groups = set(
            groups.iloc[
                validation_indices
            ].tolist()
        )


        assert train_groups

        assert validation_groups

        assert (
            train_groups
            &
            validation_groups
        ) == set()


        assert (
            validation_seen
            &
            validation_groups
        ) == set()


        validation_seen.update(
            validation_groups
        )


    assert validation_seen == set(
        groups.tolist()
    )


def test_strategy_policy(
) -> None:

    assert (
        cross_validation_strategy(
            problem_type=
                "regression"
        )
        ==
        "k_fold"
    )


    assert (
        cross_validation_strategy(
            problem_type=
                "classification"
        )
        ==
        "stratified_k_fold"
    )


    assert (
        cross_validation_strategy(
            problem_type=
                "regression",

            group_aware=
                True,
        )
        ==
        "group_k_fold"
    )


    assert (
        cross_validation_strategy(
            problem_type=
                "classification",

            group_aware=
                True,
        )
        ==
        "stratified_group_k_fold"
    )


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
        hyperparameter_validation_strategy(
            problem_type=
                "regression",

            group_aware=
                True,
        )
        ==
        "group_k_fold"
    )


    assert (
        hyperparameter_validation_strategy(
            problem_type=
                "classification",

            group_aware=
                True,
        )
        ==
        "stratified_group_k_fold"
    )


def test_group_classification_cv(
) -> None:

    dataframe = (
        classification_dataframe()
    )

    contract = (
        classification_group_contract()
    )

    (
        x,
        y,
        groups,
    ) = material(
        dataframe=
            dataframe,

        contract=
            contract,
    )


    cv_contract = (
        MLCrossValidationContract(
            folds=
                4,

            shuffle=
                True,

            random_seed=
                42,
        )
    )


    first_pairs = (
        _build_cross_validation_pairs(
            x=
                x,

            y=
                y,

            training_contract=
                contract,

            cross_validation_contract=
                cv_contract,

            groups=
                groups,
        )
    )


    second_pairs = (
        _build_cross_validation_pairs(
            x=
                x,

            y=
                y,

            training_contract=
                contract,

            cross_validation_contract=
                cv_contract,

            groups=
                groups,
        )
    )


    assert (
        len(first_pairs)
        ==
        4
    )


    for (
        first_pair,
        second_pair,
    ) in zip(
        first_pairs,
        second_pairs,
    ):

        assert np.array_equal(
            first_pair[0],
            second_pair[0],
        )

        assert np.array_equal(
            first_pair[1],
            second_pair[1],
        )


    assert_zero_entity_overlap(
        pairs=
            first_pairs,

        groups=
            groups,
    )


    with patch.object(
        cross_validation_executor,
        "_load_authorized_dataframe",
        return_value=(
            dataframe.copy(
                deep=True
            ),
            17,
        ),
    ):

        result = (
            execute_ml_cross_validation(
                training_contract=
                    contract,

                cross_validation_contract=
                    cv_contract,
            )
        )


    assert (
        result.strategy
        ==
        "stratified_group_k_fold"
    )


    assert result.folds == 4


    for fold in result.fold_results:

        assert fold.train_rows > 0

        assert fold.validation_rows > 0


        for value in fold.metrics.values():

            assert math.isfinite(
                float(value)
            )


def test_group_regression_cv(
) -> None:

    dataframe = (
        regression_dataframe()
    )

    contract = (
        regression_group_contract()
    )


    (
        x,
        y,
        groups,
    ) = material(
        dataframe=
            dataframe,

        contract=
            contract,
    )


    cv_contract = (
        MLCrossValidationContract(
            folds=
                4,

            shuffle=
                True,

            random_seed=
                42,
        )
    )


    pairs = (
        _build_cross_validation_pairs(
            x=
                x,

            y=
                y,

            training_contract=
                contract,

            cross_validation_contract=
                cv_contract,

            groups=
                groups,
        )
    )


    assert_zero_entity_overlap(
        pairs=
            pairs,

        groups=
            groups,
    )


    with patch.object(
        cross_validation_executor,
        "_load_authorized_dataframe",
        return_value=(
            dataframe.copy(
                deep=True
            ),
            18,
        ),
    ):

        result = (
            execute_ml_cross_validation(
                training_contract=
                    contract,

                cross_validation_contract=
                    cv_contract,
            )
        )


    assert (
        result.strategy
        ==
        "group_k_fold"
    )


def test_group_feasibility_fail_closed(
) -> None:

    contract = (
        classification_group_contract()
    )


    x = pd.DataFrame(
        {
            "age":
                list(
                    range(12)
                ),
        }
    )


    y = pd.Series(
        [
            1,
            1,
            1,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ]
    )


    groups = pd.Series(
        [
            "a",
            "a",
            "b",
            "b",
            "c",
            "c",
            "d",
            "d",
            "e",
            "e",
            "f",
            "f",
        ]
    )


    try:

        _validate_cross_validation_feasibility(
            x=
                x,

            y=
                y,

            training_contract=
                contract,

            cross_validation_contract=
                MLCrossValidationContract(
                    folds=
                        3
                ),

            groups=
                groups,
        )


    except MLCrossValidationInputError as error:

        assert (
            "target class"
            in
            str(error)
        )


    else:

        raise AssertionError(
            (
                "Invalid class/group "
                "coverage was accepted."
            )
        )


def test_group_tuning_inner_cv(
) -> None:

    dataframe = (
        classification_dataframe()
    )

    contract = (
        classification_group_contract()
    )


    (
        x,
        y,
    ) = (
        _validate_and_extract_xy(
            dataframe=
                dataframe,

            contract=
                contract,
        )
    )


    (
        x_outer_train,
        x_holdout_test,
        y_outer_train,
        y_holdout_test,
        outer_train_groups,
        holdout_test_groups,
    ) = (
        _split_dataset(
            x=
                x,

            y=
                y,

            contract=
                contract,

            dataframe=
                dataframe,

            return_group_partitions=
                True,
        )
    )


    assert outer_train_groups is not None

    assert holdout_test_groups is not None


    outer_group_set = set(
        outer_train_groups.tolist()
    )

    holdout_group_set = set(
        holdout_test_groups.tolist()
    )


    assert (
        outer_group_set
        &
        holdout_group_set
    ) == set()


    captured_groups = []


    original_builder = (
        hyperparameter_tuning_executor
        ._build_cross_validation_pairs
    )


    def capture_pairs(
        **kwargs,
    ):

        groups = kwargs[
            "groups"
        ]

        assert groups is not None


        current_group_set = set(
            groups.tolist()
        )


        captured_groups.append(
            current_group_set
        )


        assert (
            current_group_set
            ==
            outer_group_set
        )


        assert (
            current_group_set
            &
            holdout_group_set
        ) == set()


        pairs = original_builder(
            **kwargs
        )


        assert_zero_entity_overlap(
            pairs=
                pairs,

            groups=
                groups,
        )


        return pairs


    with (
        patch.object(
            hyperparameter_tuning_executor,
            "_load_authorized_dataframe",
            return_value=(
                dataframe.copy(
                    deep=True
                ),
                19,
            ),
        ),
        patch.object(
            hyperparameter_tuning_executor,
            "_build_cross_validation_pairs",
            side_effect=
                capture_pairs,
        ),
    ):

        result = (
            execute_ml_hyperparameter_tuning(
                training_contract=
                    contract,

                search_contract=
                    MLHyperparameterSearchContract(
                        folds=
                            4,

                        shuffle=
                            True,

                        random_seed=
                            42,
                    ),
            )
        )


    assert captured_groups


    assert (
        result.validation_strategy
        ==
        "stratified_group_k_fold"
    )


    assert (
        result.outer_train_rows
        ==
        len(x_outer_train)
    )


    assert (
        result.holdout_test_rows
        ==
        len(x_holdout_test)
    )


    assert (
        len(y_outer_train)
        ==
        len(x_outer_train)
    )


    assert (
        len(y_holdout_test)
        ==
        len(x_holdout_test)
    )


def test_group_regression_tuning(
) -> None:

    dataframe = (
        regression_dataframe()
    )

    contract = (
        regression_group_contract()
    )


    with patch.object(
        hyperparameter_tuning_executor,
        "_load_authorized_dataframe",
        return_value=(
            dataframe.copy(
                deep=True
            ),
            20,
        ),
    ):

        result = (
            execute_ml_hyperparameter_tuning(
                training_contract=
                    contract,

                search_contract=
                    MLHyperparameterSearchContract(
                        folds=
                            4,

                        shuffle=
                            True,

                        random_seed=
                            42,
                    ),
            )
        )


    assert (
        result.validation_strategy
        ==
        "group_k_fold"
    )

    assert result.candidate_count > 0


def test_historical_row_behavior(
) -> None:

    dataframe = (
        classification_dataframe()
    )

    contract = (
        classification_row_contract()
    )


    with patch.object(
        cross_validation_executor,
        "_load_authorized_dataframe",
        return_value=(
            dataframe.copy(
                deep=True
            ),
            21,
        ),
    ):

        cv_result = (
            execute_ml_cross_validation(
                training_contract=
                    contract,

                cross_validation_contract=
                    MLCrossValidationContract(
                        folds=
                            4,

                        shuffle=
                            True,

                        random_seed=
                            42,
                    ),
            )
        )


    assert (
        cv_result.strategy
        ==
        "stratified_k_fold"
    )


    with patch.object(
        hyperparameter_tuning_executor,
        "_load_authorized_dataframe",
        return_value=(
            dataframe.copy(
                deep=True
            ),
            22,
        ),
    ):

        tuning_result = (
            execute_ml_hyperparameter_tuning(
                training_contract=
                    contract,

                search_contract=
                    MLHyperparameterSearchContract(
                        folds=
                            4,

                        shuffle=
                            True,

                        random_seed=
                            42,
                    ),
            )
        )


    assert (
        tuning_result.validation_strategy
        ==
        "stratified_k_fold"
    )


def main(
) -> None:

    print(
        "=== DATALENS GROUP-AWARE CV / TUNING v0.1 ==="
    )


    test_strategy_policy()

    print(
        "[PASS] row/group strategy policy"
    )


    test_group_classification_cv()

    print(
        "[PASS] StratifiedGroupKFold classification"
    )


    test_group_regression_cv()

    print(
        "[PASS] GroupKFold regression"
    )


    test_group_feasibility_fail_closed()

    print(
        "[PASS] invalid group fold feasibility fails closed"
    )


    test_group_tuning_inner_cv()

    print(
        "[PASS] tuning INNER CV uses OUTER train groups only"
    )


    test_group_regression_tuning()

    print(
        "[PASS] regression tuning uses group-aware INNER CV"
    )


    test_historical_row_behavior()

    print(
        "[PASS] historical row CV/tuning preserved"
    )


    print()

    print(
        "GROUP-AWARE CV / TUNING v0.1: PASS"
    )


if __name__ == "__main__":

    main()
