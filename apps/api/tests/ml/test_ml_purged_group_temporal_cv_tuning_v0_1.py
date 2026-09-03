from __future__ import annotations


from unittest.mock import (
    patch,
)


import pandas as pd


import app.ml.cross_validation_executor as cross_validation_executor
import app.ml.hyperparameter_tuning_executor as hyperparameter_tuning_executor


from app.ml.classical_executor import (
    _split_dataset,
    _validate_and_extract_xy,
    _validated_group_values,
    _validated_time_values,
)


from app.ml.contracts import (
    MLPurgedGroupTimeHoldoutSplitContract,
    MLTimeHoldoutSplitContract,
    MLTrainingContract,
)


from app.ml.cross_validation import (
    MLCrossValidationContract,
    cross_validation_strategy,
)


from app.ml.cross_validation_executor import (
    MLCrossValidationInputError,
    _build_cross_validation_pairs,
    execute_ml_cross_validation,
)


from app.ml.hyperparameter_tuning import (
    MLHyperparameterSearchContract,
    hyperparameter_validation_strategy,
)


from app.ml.hyperparameter_tuning_executor import (
    _build_inner_cv_pairs,
    execute_ml_hyperparameter_tuning,
)


WORKFLOW_ID = (
    "workflow:purged-group-temporal-cv-tuning"
)


REGRESSION_DATASET_ID = (
    "dataset:purged-group-temporal-regression"
)


CLASSIFICATION_DATASET_ID = (
    "dataset:purged-group-temporal-classification"
)


GROUP_SEQUENCE = (
    "a",
    "b",
    "c",
    "d",
    "a",
    "e",
    "b",
    "f",
    "c",
    "g",
    "d",
    "h",
)


def expect_error(
    callback,
    errors,
) -> None:

    try:

        callback()


    except errors:

        return


    raise AssertionError(
        "Expected error was not raised."
    )


def build_dataframe(
    *,
    classification: bool = False,
) -> pd.DataFrame:

    rows = []


    for (
        timestamp_index,
        group_name,
    ) in enumerate(
        GROUP_SEQUENCE
    ):

        timestamp = (
            pd.Timestamp(
                "2026-03-01"
            )
            +
            pd.Timedelta(
                days=
                    timestamp_index
            )
        )


        for duplicate_index in range(
            2
        ):

            feature = (
                float(
                    timestamp_index
                )
                +
                float(
                    duplicate_index
                )
                *
                0.25
            )


            target = (
                (
                    "A"
                    if duplicate_index == 0
                    else
                    "B"
                )

                if classification

                else
                (
                    100.0
                    +
                    feature
                    *
                    4.0
                )
            )


            rows.append(
                {
                    "event_time":
                        timestamp,

                    "client_id":
                        (
                            "client_"
                            +
                            group_name
                        ),

                    "feature":
                        feature,

                    "target":
                        target,
                }
            )


    return (
        pd.DataFrame(
            rows
        )
        .iloc[
            ::-1
        ]
        .reset_index(
            drop=True
        )
    )


def combined_contract(
    *,
    classification: bool = False,
) -> MLTrainingContract:

    return (
        MLTrainingContract(
            workflow_id=
                WORKFLOW_ID,

            dataset_id=(
                CLASSIFICATION_DATASET_ID

                if classification

                else
                REGRESSION_DATASET_ID
            ),

            problem_type=(
                "classification"

                if classification

                else
                "regression"
            ),

            target_column=
                "target",

            feature_columns=[
                "feature",
            ],

            categorical_feature_columns=[],

            estimator_key=(
                "logistic_regression"

                if classification

                else
                "linear_regression"
            ),

            split=
                MLPurgedGroupTimeHoldoutSplitContract(
                    group_column=
                        "client_id",

                    time_column=
                        "event_time",

                    test_size=
                        0.25,

                    random_seed=
                        42,

                    shuffle=
                        False,

                    stratify=
                        False,
                ),
        )
    )


def temporal_contract(
    *,
    classification: bool = False,
) -> MLTrainingContract:

    return (
        MLTrainingContract(
            workflow_id=
                WORKFLOW_ID,

            dataset_id=(
                CLASSIFICATION_DATASET_ID

                if classification

                else
                REGRESSION_DATASET_ID
            ),

            problem_type=(
                "classification"

                if classification

                else
                "regression"
            ),

            target_column=
                "target",

            feature_columns=[
                "feature",
            ],

            categorical_feature_columns=[],

            estimator_key=(
                "logistic_regression"

                if classification

                else
                "linear_regression"
            ),

            split=
                MLTimeHoldoutSplitContract(
                    time_column=
                        "event_time",

                    test_size=
                        0.25,

                    random_seed=
                        42,

                    shuffle=
                        False,

                    stratify=
                        False,
                ),
        )
    )


def combined_material(
    *,
    classification: bool = False,
):

    dataframe = (
        build_dataframe(
            classification=
                classification
        )
    )


    contract = (
        combined_contract(
            classification=
                classification
        )
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


    times = (
        _validated_time_values(
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
        dataframe,
        x,
        y,
        groups,
        times,
        contract,
    )


def test_combined_strategy_policy() -> None:

    for problem_type in (
        "regression",
        "classification",
    ):

        assert (
            cross_validation_strategy(
                problem_type=
                    problem_type,

                group_aware=
                    True,

                temporal_aware=
                    True,
            )
            ==
            "purged_group_time_series_split"
        )


        assert (
            hyperparameter_validation_strategy(
                problem_type=
                    problem_type,

                group_aware=
                    True,

                temporal_aware=
                    True,
            )
            ==
            "purged_group_time_series_split"
        )


def test_validation_future_unchanged_and_train_purged() -> None:

    (
        dataframe,
        x,
        y,
        groups,
        times,
        combined,
    ) = combined_material()


    pure_temporal = (
        temporal_contract()
    )


    pure_times = (
        _validated_time_values(
            dataframe=
                dataframe,

            x=
                x,

            y=
                y,

            contract=
                pure_temporal,
        )
    )


    cv = (
        MLCrossValidationContract(
            folds=
                3,

            shuffle=
                False,

            random_seed=
                42,
        )
    )


    temporal_pairs = (
        _build_cross_validation_pairs(
            x=
                x,

            y=
                y,

            training_contract=
                pure_temporal,

            cross_validation_contract=
                cv,

            times=
                pure_times,
        )
    )


    combined_pairs = (
        _build_cross_validation_pairs(
            x=
                x,

            y=
                y,

            training_contract=
                combined,

            cross_validation_contract=
                cv,

            groups=
                groups,

            times=
                times,
        )
    )


    assert (
        len(
            combined_pairs
        )
        ==
        3
    )


    assert (
        len(
            temporal_pairs
        )
        ==
        3
    )


    purge_observed = False


    for (
        (
            temporal_train,
            temporal_validation,
        ),
        (
            combined_train,
            combined_validation,
        ),
    ) in zip(
        temporal_pairs,
        combined_pairs,
    ):

        assert (
            combined_validation.tolist()
            ==
            temporal_validation.tolist()
        )


        assert (
            set(
                combined_train.tolist()
            )
            <=
            set(
                temporal_train.tolist()
            )
        )


        if (
            len(
                combined_train
            )
            <
            len(
                temporal_train
            )
        ):

            purge_observed = True


        train_groups = set(
            groups.iloc[
                combined_train
            ].tolist()
        )


        validation_groups = set(
            groups.iloc[
                combined_validation
            ].tolist()
        )


        assert train_groups

        assert validation_groups


        assert not (
            train_groups
            &
            validation_groups
        )


        train_times = (
            times.iloc[
                combined_train
            ]
        )


        validation_times = (
            times.iloc[
                combined_validation
            ]
        )


        assert (
            train_times.max()
            <
            validation_times.min()
        )


        assert not (
            set(
                train_times.tolist()
            )
            &
            set(
                validation_times.tolist()
            )
        )


    assert purge_observed


def test_combined_classification_fold_classes() -> None:

    (
        _,
        x,
        y,
        groups,
        times,
        contract,
    ) = combined_material(
        classification=True
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
                MLCrossValidationContract(
                    folds=
                        3,

                    shuffle=
                        False,
                ),

            groups=
                groups,

            times=
                times,
        )
    )


    expected_classes = {
        "A",
        "B",
    }


    for (
        train_indices,
        validation_indices,
    ) in pairs:

        assert (
            set(
                y.iloc[
                    train_indices
                ].tolist()
            )
            ==
            expected_classes
        )


        assert (
            set(
                y.iloc[
                    validation_indices
                ].tolist()
            )
            ==
            expected_classes
        )


def test_combined_shuffle_fails_closed() -> None:

    (
        _,
        x,
        y,
        groups,
        times,
        contract,
    ) = combined_material()


    expect_error(
        lambda:
            _build_cross_validation_pairs(
                x=
                    x,

                y=
                    y,

                training_contract=
                    contract,

                cross_validation_contract=
                    MLCrossValidationContract(
                        folds=
                            3,

                        shuffle=
                            True,
                    ),

                groups=
                    groups,

                times=
                    times,
            ),

        (
            MLCrossValidationInputError,
        ),
    )


def test_combined_cross_validation_deterministic() -> None:

    (
        _,
        x,
        y,
        groups,
        times,
        contract,
    ) = combined_material()


    cv = (
        MLCrossValidationContract(
            folds=
                3,

            shuffle=
                False,

            random_seed=
                42,
        )
    )


    first = (
        _build_cross_validation_pairs(
            x=
                x,

            y=
                y,

            training_contract=
                contract,

            cross_validation_contract=
                cv,

            groups=
                groups,

            times=
                times,
        )
    )


    second = (
        _build_cross_validation_pairs(
            x=
                x,

            y=
                y,

            training_contract=
                contract,

            cross_validation_contract=
                cv,

            groups=
                groups,

            times=
                times,
        )
    )


    assert [
        (
            train.tolist(),
            validation.tolist(),
        )

        for (
            train,
            validation,
        )
        in first
    ] == [
        (
            train.tolist(),
            validation.tolist(),
        )

        for (
            train,
            validation,
        )
        in second
    ]


def test_full_combined_cross_validation_executor() -> None:

    dataframe = (
        build_dataframe()
    )


    contract = (
        combined_contract()
    )


    cv = (
        MLCrossValidationContract(
            folds=
                3,

            shuffle=
                False,

            random_seed=
                42,
        )
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

        result = (
            execute_ml_cross_validation(
                training_contract=
                    contract,

                cross_validation_contract=
                    cv,
            )
        )


    assert (
        result.strategy
        ==
        "purged_group_time_series_split"
    )


    assert (
        result.shuffle
        is False
    )


    assert (
        result.folds
        ==
        3
    )


    assert (
        len(
            result.fold_results
        )
        ==
        3
    )


    for fold in (
        result.fold_results
    ):

        assert (
            fold.train_rows
            >=
            2
        )

        assert (
            fold.validation_rows
            >=
            2
        )


def test_combined_outer_holdout_and_inner_cv() -> None:

    (
        dataframe,
        x,
        y,
        _,
        _,
        contract,
    ) = combined_material()


    (
        x_outer_train,
        x_holdout_test,
        y_outer_train,
        y_holdout_test,
        outer_train_groups,
        holdout_test_groups,
        outer_train_times,
        holdout_test_times,
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

            return_time_partitions=
                True,
        )
    )


    assert (
        len(
            x_outer_train
        )
        +
        len(
            x_holdout_test
        )
        <
        len(
            x
        )
    )


    assert not (
        set(
            outer_train_groups.tolist()
        )
        &
        set(
            holdout_test_groups.tolist()
        )
    )


    assert (
        outer_train_times.max()
        <
        holdout_test_times.min()
    )


    inner_pairs = (
        _build_inner_cv_pairs(
            x_train=
                x_outer_train,

            y_train=
                y_outer_train,

            training_contract=
                contract,

            search_contract=
                MLHyperparameterSearchContract(
                    folds=
                        3,

                    shuffle=
                        False,

                    random_seed=
                        42,
                ),

            groups_train=
                outer_train_groups,

            times_train=
                outer_train_times,
        )
    )


    holdout_indices = set(
        x_holdout_test.index.tolist()
    )


    holdout_timestamps = set(
        holdout_test_times.tolist()
    )


    for (
        inner_train_indices,
        inner_validation_indices,
    ) in inner_pairs:

        train_groups = set(
            outer_train_groups.iloc[
                inner_train_indices
            ].tolist()
        )


        validation_groups = set(
            outer_train_groups.iloc[
                inner_validation_indices
            ].tolist()
        )


        assert not (
            train_groups
            &
            validation_groups
        )


        train_times = (
            outer_train_times.iloc[
                inner_train_indices
            ]
        )


        validation_times = (
            outer_train_times.iloc[
                inner_validation_indices
            ]
        )


        assert (
            train_times.max()
            <
            validation_times.min()
        )


        assert not (
            set(
                train_times.tolist()
            )
            &
            holdout_timestamps
        )


        assert not (
            set(
                validation_times.tolist()
            )
            &
            holdout_timestamps
        )


        inner_source_indices = (
            set(
                x_outer_train.iloc[
                    inner_train_indices
                ].index.tolist()
            )
            |
            set(
                x_outer_train.iloc[
                    inner_validation_indices
                ].index.tolist()
            )
        )


        assert not (
            inner_source_indices
            &
            holdout_indices
        )


    assert (
        len(
            y_holdout_test
        )
        ==
        len(
            x_holdout_test
        )
    )


def test_full_combined_hyperparameter_tuning_executor() -> None:

    dataframe = (
        build_dataframe()
    )


    contract = (
        combined_contract()
    )


    search = (
        MLHyperparameterSearchContract(
            folds=
                3,

            shuffle=
                False,

            random_seed=
                42,
        )
    )


    with patch.object(
        hyperparameter_tuning_executor,
        "_load_authorized_dataframe",
        return_value=(
            dataframe.copy(
                deep=True
            ),
            34,
        ),
    ):

        result = (
            execute_ml_hyperparameter_tuning(
                training_contract=
                    contract,

                search_contract=
                    search,
            )
        )


    assert (
        result.validation_strategy
        ==
        "purged_group_time_series_split"
    )


    assert (
        result.shuffle
        is False
    )


    assert (
        result.outer_train_rows
        +
        result.holdout_test_rows
        <
        len(
            dataframe
        )
    )


    assert (
        result.candidate_count
        ==
        2
    )


    assert (
        len(
            result.candidate_results
        )
        ==
        2
    )


def test_historical_strategy_policy_preserved() -> None:

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
        cross_validation_strategy(
            problem_type=
                "regression",

            temporal_aware=
                True,
        )
        ==
        "time_series_split"
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


    assert (
        hyperparameter_validation_strategy(
            problem_type=
                "regression",

            temporal_aware=
                True,
        )
        ==
        "time_series_split"
    )


def main() -> None:

    print(
        "=== DATALENS E15b PURGED GROUP + TEMPORAL CV / TUNING v0.1 ==="
    )


    test_combined_strategy_policy()

    print(
        "[PASS] combined server-owned strategy policy"
    )


    test_validation_future_unchanged_and_train_purged()

    print(
        "[PASS] future validation unchanged"
    )

    print(
        "[PASS] validation groups purged from historical train"
    )

    print(
        "[PASS] train/validation entity overlap = zero"
    )

    print(
        "[PASS] strict train_time < validation_time"
    )


    test_combined_classification_fold_classes()

    print(
        "[PASS] combined classification class feasibility"
    )


    test_combined_shuffle_fails_closed()

    print(
        "[PASS] combined shuffle=True fails closed"
    )


    test_combined_cross_validation_deterministic()

    print(
        "[PASS] deterministic combined CV folds"
    )


    test_full_combined_cross_validation_executor()

    print(
        "[PASS] full combined Cross-Validation executor"
    )


    test_combined_outer_holdout_and_inner_cv()

    print(
        "[PASS] tuning INNER CV receives OUTER TRAIN only"
    )

    print(
        "[PASS] inner fold entity leakage = zero"
    )

    print(
        "[PASS] outer holdout remains untouched by tuning"
    )


    test_full_combined_hyperparameter_tuning_executor()

    print(
        "[PASS] full combined Hyperparameter Tuning executor"
    )


    test_historical_strategy_policy_preserved()

    print(
        "[PASS] historical row/group/temporal strategies preserved"
    )


    print()

    print(
        "E15b PURGED GROUP + TEMPORAL CV / TUNING v0.1: PASS"
    )


if __name__ == "__main__":

    main()
