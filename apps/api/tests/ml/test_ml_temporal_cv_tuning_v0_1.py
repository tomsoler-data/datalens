from __future__ import annotations


from unittest.mock import (
    patch,
)


import pandas as pd


import app.ml.cross_validation_executor as cross_validation_executor
import app.ml.hyperparameter_tuning_executor as hyperparameter_tuning_executor


from app.ml.classical_executor import (
    ClassicalMLInputError,
    _split_dataset,
    _validate_and_extract_xy,
    _validated_time_values,
)


from app.ml.contracts import (
    MLGroupHoldoutSplitContract,
    MLSplitContract,
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
    "workflow:temporal-cv-tuning"
)


REGRESSION_DATASET_ID = (
    "dataset:temporal-regression"
)


CLASSIFICATION_DATASET_ID = (
    "dataset:temporal-classification"
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


def regression_dataframe(
) -> pd.DataFrame:

    rows = []


    for timestamp_index in range(
        12
    ):

        timestamp = pd.Timestamp(
            "2026-01-01"
        ) + pd.Timedelta(
            days=
                timestamp_index
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
                0.1
            )


            rows.append(
                {
                    "event_time":
                        timestamp,

                    "feature":
                        feature,

                    "target":
                        (
                            10.0
                            +
                            feature
                            *
                            3.0
                        ),
                }
            )


    dataframe = pd.DataFrame(
        rows
    )


    # Deliberately reverse physical row order.
    #
    # Temporal CV must use the authoritative timestamps,
    # not source-row order.
    return (
        dataframe.iloc[
            ::-1
        ]
        .reset_index(
            drop=True
        )
    )


def classification_dataframe(
) -> pd.DataFrame:

    rows = []


    for timestamp_index in range(
        12
    ):

        timestamp = pd.Timestamp(
            "2026-02-01"
        ) + pd.Timedelta(
            days=
                timestamp_index
        )


        rows.append(
            {
                "event_time":
                    timestamp,

                "feature":
                    float(
                        timestamp_index
                    ),

                "target":
                    "A",
            }
        )


        rows.append(
            {
                "event_time":
                    timestamp,

                "feature":
                    float(
                        timestamp_index
                    )
                    +
                    0.5,

                "target":
                    "B",
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


def regression_contract(
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
                "feature",
            ],

            categorical_feature_columns=[],

            estimator_key=
                "linear_regression",

            split=
                MLTimeHoldoutSplitContract(
                    time_column=
                        "event_time",

                    test_size=
                        0.25,

                    random_seed=
                        42,
                ),
        )
    )


def classification_contract(
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
                "target",

            feature_columns=[
                "feature",
            ],

            categorical_feature_columns=[],

            estimator_key=
                "logistic_regression",

            split=
                MLTimeHoldoutSplitContract(
                    time_column=
                        "event_time",

                    test_size=
                        0.25,

                    random_seed=
                        42,
                ),
        )
    )


def temporal_material(
    *,
    classification: bool = False,
):

    dataframe = (
        classification_dataframe()

        if classification

        else
        regression_dataframe()
    )


    contract = (
        classification_contract()

        if classification

        else
        regression_contract()
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
        times,
        contract,
    )


def test_temporal_strategy_policy():

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
        cross_validation_strategy(
            problem_type=
                "classification",

            temporal_aware=
                True,
        )
        ==
        "time_series_split"
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


    assert (
        hyperparameter_validation_strategy(
            problem_type=
                "classification",

            temporal_aware=
                True,
        )
        ==
        "time_series_split"
    )


    expect_error(
        lambda:
            cross_validation_strategy(
                problem_type=
                    "regression",

                group_aware=
                    True,

                temporal_aware=
                    True,
            ),
        (
            ValueError,
        ),
    )


    expect_error(
        lambda:
            hyperparameter_validation_strategy(
                problem_type=
                    "regression",

                group_aware=
                    True,

                temporal_aware=
                    True,
            ),
        (
            ValueError,
        ),
    )


def test_temporal_regression_expanding_folds():

    (
        dataframe,
        x,
        y,
        times,
        contract,
    ) = temporal_material()


    cv_contract = (
        MLCrossValidationContract(
            folds=
                3,

            shuffle=
                False,
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

            times=
                times,
        )
    )


    assert len(
        pairs
    ) == 3


    previous_train_timestamps = None

    previous_validation_max = None

    validation_seen = set()


    for (
        train_indices,
        validation_indices,
    ) in pairs:

        train_times = (
            times.iloc[
                train_indices
            ]
        )


        validation_times = (
            times.iloc[
                validation_indices
            ]
        )


        train_values = set(
            train_times.tolist()
        )


        validation_values = set(
            validation_times.tolist()
        )


        assert (
            train_times.max()
            <
            validation_times.min()
        )


        assert not (
            train_values
            &
            validation_values
        )


        assert not (
            validation_seen
            &
            validation_values
        )


        validation_seen.update(
            validation_values
        )


        if (
            previous_train_timestamps
            is not None
        ):

            assert (
                previous_train_timestamps
                <
                train_values
            )


        if (
            previous_validation_max
            is not None
        ):

            assert (
                previous_validation_max
                <
                validation_times.min()
            )


        previous_train_timestamps = (
            train_values
        )


        previous_validation_max = (
            validation_times.max()
        )


        # Two rows exist for every timestamp.
        assert len(
            validation_indices
        ) >= 2


def test_temporal_classification_no_stratification():

    (
        dataframe,
        x,
        y,
        times,
        contract,
    ) = temporal_material(
        classification=True
    )


    cv_contract = (
        MLCrossValidationContract(
            folds=
                3,

            shuffle=
                False,
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

            times=
                times,
        )
    )


    assert len(
        pairs
    ) == 3


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


        assert (
            times.iloc[
                train_indices
            ].max()
            <
            times.iloc[
                validation_indices
            ].min()
        )


def test_temporal_shuffle_fails_closed():

    (
        dataframe,
        x,
        y,
        times,
        contract,
    ) = temporal_material()


    cv_contract = (
        MLCrossValidationContract(
            folds=
                3,

            shuffle=
                True,
        )
    )


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
                    cv_contract,

                times=
                    times,
            ),
        (
            MLCrossValidationInputError,
        ),
    )


def test_temporal_outer_metadata_handoff():

    (
        dataframe,
        x,
        y,
        times,
        contract,
    ) = temporal_material()


    (
        x_train,
        x_test,
        y_train,
        y_test,
        train_times,
        test_times,
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

            return_time_partitions=
                True,
        )
    )


    assert train_times is not None

    assert test_times is not None


    assert (
        len(
            train_times
        )
        ==
        len(
            x_train
        )
    )


    assert (
        len(
            test_times
        )
        ==
        len(
            x_test
        )
    )


    assert (
        train_times.index.equals(
            x_train.index
        )
    )


    assert (
        test_times.index.equals(
            x_test.index
        )
    )


    assert (
        train_times.max()
        <
        test_times.min()
    )


    assert not (
        set(
            train_times.tolist()
        )
        &
        set(
            test_times.tolist()
        )
    )


    expect_error(
        lambda:
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
            ),
        (
            ClassicalMLInputError,
        ),
    )


def test_temporal_inner_tuning_sees_outer_train_only():

    (
        dataframe,
        x,
        y,
        times,
        contract,
    ) = temporal_material()


    (
        x_outer_train,
        x_holdout_test,
        y_outer_train,
        y_holdout_test,
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

            return_time_partitions=
                True,
        )
    )


    assert outer_train_times is not None

    assert holdout_test_times is not None


    search_contract = (
        MLHyperparameterSearchContract(
            folds=
                2,

            shuffle=
                False,
        )
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
                search_contract,

            times_train=
                outer_train_times,
        )
    )


    holdout_values = set(
        holdout_test_times.tolist()
    )


    for (
        inner_train_indices,
        inner_validation_indices,
    ) in inner_pairs:

        inner_train_times = (
            outer_train_times.iloc[
                inner_train_indices
            ]
        )


        inner_validation_times = (
            outer_train_times.iloc[
                inner_validation_indices
            ]
        )


        assert (
            inner_train_times.max()
            <
            inner_validation_times.min()
        )


        assert not (
            set(
                inner_train_times.tolist()
            )
            &
            holdout_values
        )


        assert not (
            set(
                inner_validation_times.tolist()
            )
            &
            holdout_values
        )


def test_full_temporal_cross_validation_executor():

    dataframe = (
        regression_dataframe()
    )


    contract = (
        regression_contract()
    )


    cv_contract = (
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
            7,
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
        "time_series_split"
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


    train_sizes = [
        fold.train_rows

        for fold
        in result.fold_results
    ]


    assert (
        train_sizes
        ==
        sorted(
            train_sizes
        )
    )


    assert (
        len(
            set(
                train_sizes
            )
        )
        ==
        len(
            train_sizes
        )
    )


def test_full_temporal_hyperparameter_tuning_executor():

    dataframe = (
        regression_dataframe()
    )


    contract = (
        regression_contract()
    )


    search_contract = (
        MLHyperparameterSearchContract(
            folds=
                2,

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
            11,
        ),
    ):

        result = (
            execute_ml_hyperparameter_tuning(
                training_contract=
                    contract,

                search_contract=
                    search_contract,
            )
        )


    assert (
        result.validation_strategy
        ==
        "time_series_split"
    )


    assert (
        result.shuffle
        is False
    )


    assert (
        result.outer_train_rows
        <
        len(
            dataframe
        )
    )


    assert (
        result.holdout_test_rows
        >
        0
    )


    assert (
        result.outer_train_rows
        +
        result.holdout_test_rows
        ==
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


def test_historical_row_group_strategy_policy_preserved():

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


def main():

    print(
        "=== DATALENS TEMPORAL CV / TUNING v0.1 ==="
    )


    test_temporal_strategy_policy()

    print(
        "[PASS] temporal server-owned strategy policy"
    )


    test_temporal_regression_expanding_folds()

    print(
        "[PASS] expanding regression TimeSeriesSplit folds"
    )


    test_temporal_classification_no_stratification()

    print(
        "[PASS] temporal classification without stratification"
    )


    test_temporal_shuffle_fails_closed()

    print(
        "[PASS] temporal shuffle=True fails closed"
    )


    test_temporal_outer_metadata_handoff()

    print(
        "[PASS] temporal OUTER split metadata handoff"
    )


    test_temporal_inner_tuning_sees_outer_train_only()

    print(
        "[PASS] tuning receives OUTER TRAIN timestamps only"
    )


    test_full_temporal_cross_validation_executor()

    print(
        "[PASS] full temporal Cross-Validation executor"
    )


    test_full_temporal_hyperparameter_tuning_executor()

    print(
        "[PASS] full temporal Hyperparameter Tuning executor"
    )


    test_historical_row_group_strategy_policy_preserved()

    print(
        "[PASS] historical row/group CV and tuning policy"
    )


    print()

    print(
        "TEMPORAL CROSS-VALIDATION / TUNING v0.1: PASS"
    )


if __name__ == "__main__":

    main()
