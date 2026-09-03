from __future__ import annotations


from pathlib import Path


import pandas as pd


from pydantic import (
    ValidationError,
)


from app.api.model_training_service import (
    _column_ml_readiness,
)


from app.ml.classical_executor import (
    ClassicalMLInputError,
    _split_dataset,
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
)


from app.ml.cross_validation_executor import (
    MLCrossValidationInputError,
    _build_cross_validation_pairs,
    _validate_cross_validation_feasibility,
)


from app.ml.hyperparameter_tuning import (
    MLHyperparameterSearchContract,
)


from app.ml.hyperparameter_tuning_executor import (
    MLHyperparameterTuningInputError,
    _build_inner_cv_pairs,
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


def material():
    dataframe = pd.DataFrame(
        {
            "event_time":
                pd.to_datetime(
                    [
                        "2026-01-01",
                        "2026-01-02",
                        "2026-01-03",
                        "2026-01-04",
                        "2026-01-05",
                        "2026-01-06",
                        "2026-01-07",
                        "2026-01-08",
                        "2026-01-08",
                        "2026-01-08",
                    ]
                ),

            "feature":
                [
                    1.0,
                    2.0,
                    3.0,
                    4.0,
                    5.0,
                    6.0,
                    7.0,
                    8.0,
                    9.0,
                    10.0,
                ],

            "target":
                [
                    10.0,
                    12.0,
                    14.0,
                    16.0,
                    18.0,
                    20.0,
                    22.0,
                    24.0,
                    26.0,
                    28.0,
                ],
        }
    )


    x = (
        dataframe[
            [
                "feature"
            ]
        ]
        .copy(
            deep=True
        )
    )


    y = (
        dataframe[
            "target"
        ]
        .copy(
            deep=True
        )
    )


    contract = (
        MLTrainingContract(
            workflow_id=
                "workflow:temporal-test",

            dataset_id=
                "dataset:temporal-test",

            problem_type=
                "regression",

            target_column=
                "target",

            feature_columns=
                [
                    "feature"
                ],

            categorical_feature_columns=
                [],

            estimator_key=
                "linear_regression",

            split=
                MLTimeHoldoutSplitContract(
                    time_column=
                        "event_time",

                    test_size=
                        0.20,
                ),
        )
    )


    return (
        dataframe,
        x,
        y,
        contract,
    )


def test_temporal_contract_and_role_guards():
    split = (
        MLTimeHoldoutSplitContract(
            time_column=
                " event_time "
        )
    )


    assert (
        split.strategy
        ==
        "time_holdout"
    )


    assert (
        split.time_column
        ==
        "event_time"
    )


    assert (
        split.shuffle
        is False
    )


    assert (
        split.stratify
        is False
    )


    expect_error(
        lambda:
            MLTrainingContract(
                workflow_id=
                    "workflow:test",

                dataset_id=
                    "dataset:test",

                problem_type=
                    "regression",

                target_column=
                    "event_time",

                feature_columns=
                    [
                        "feature"
                    ],

                estimator_key=
                    "linear_regression",

                split=
                    MLTimeHoldoutSplitContract(
                        time_column=
                            "event_time"
                    ),
            ),
        (
            ValidationError,
            ValueError,
        ),
    )


    expect_error(
        lambda:
            MLTrainingContract(
                workflow_id=
                    "workflow:test",

                dataset_id=
                    "dataset:test",

                problem_type=
                    "regression",

                target_column=
                    "target",

                feature_columns=
                    [
                        "feature",
                        "event_time",
                    ],

                estimator_key=
                    "linear_regression",

                split=
                    MLTimeHoldoutSplitContract(
                        time_column=
                            "event_time"
                    ),
            ),
        (
            ValidationError,
            ValueError,
        ),
    )


def test_server_time_readiness():
    datetime_readiness = (
        _column_ml_readiness(
            column_name=
                "event_time",

            series=
                pd.Series(
                    pd.to_datetime(
                        [
                            "2026-01-01",
                            "2026-01-02",
                            "2026-01-03",
                        ]
                    )
                ),
        )
    )


    assert (
        datetime_readiness[
            "analytical_type"
        ]
        ==
        "temporal"
    )


    assert (
        datetime_readiness[
            "analytical_subtype"
        ]
        ==
        "datetime"
    )


    assert (
        datetime_readiness[
            "ml_eligible_as_time"
        ]
        is True
    )


    birth_year_readiness = (
        _column_ml_readiness(
            column_name=
                "birth_year",

            series=
                pd.Series(
                    [
                        1980,
                        1990,
                        2000,
                    ]
                ),
        )
    )


    assert (
        birth_year_readiness[
            "analytical_type"
        ]
        ==
        "temporal"
    )


    assert (
        birth_year_readiness[
            "analytical_subtype"
        ]
        ==
        "birth_year"
    )


    assert (
        birth_year_readiness[
            "ml_eligible_as_time"
        ]
        is False
    )


    nullable_time = (
        pd.Series(
            pd.to_datetime(
                [
                    "2026-01-01",
                    None,
                    "2026-01-03",
                ]
            )
        )
    )


    nullable_readiness = (
        _column_ml_readiness(
            column_name=
                "event_time",

            series=
                nullable_time,
        )
    )


    assert (
        nullable_readiness[
            "ml_eligible_as_time"
        ]
        is False
    )


def test_chronological_holdout_and_boundary():
    (
        dataframe,
        x,
        y,
        contract,
    ) = material()


    (
        x_train,
        x_test,
        y_train,
        y_test,
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
        )
    )


    assert len(
        x_train
    ) == 7


    assert len(
        x_test
    ) == 3


    assert len(
        y_train
    ) == 7


    assert len(
        y_test
    ) == 3


    train_times = (
        dataframe.loc[
            x_train.index,
            "event_time",
        ]
    )


    test_times = (
        dataframe.loc[
            x_test.index,
            "event_time",
        ]
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


    assert (
        test_times.min()
        ==
        pd.Timestamp(
            "2026-01-08"
        )
    )


def test_time_validator_fails_closed():
    (
        dataframe,
        x,
        y,
        contract,
    ) = material()


    validated = (
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


    assert (
        pd.api.types
        .is_datetime64_any_dtype(
            validated.dtype
        )
    )


    string_dataframe = (
        dataframe.copy(
            deep=True
        )
    )


    string_dataframe[
        "event_time"
    ] = (
        string_dataframe[
            "event_time"
        ]
        .dt
        .strftime(
            "%Y-%m-%d"
        )
    )


    expect_error(
        lambda:
            _split_dataset(
                x=
                    string_dataframe[
                        [
                            "feature"
                        ]
                    ],

                y=
                    string_dataframe[
                        "target"
                    ],

                contract=
                    contract,

                dataframe=
                    string_dataframe,
            ),
        (
            ClassicalMLInputError,
        ),
    )


    missing_dataframe = (
        dataframe.copy(
            deep=True
        )
    )


    missing_dataframe.loc[
        3,
        "event_time",
    ] = pd.NaT


    expect_error(
        lambda:
            _split_dataset(
                x=
                    missing_dataframe[
                        [
                            "feature"
                        ]
                    ],

                y=
                    missing_dataframe[
                        "target"
                    ],

                contract=
                    contract,

                dataframe=
                    missing_dataframe,
            ),
        (
            ClassicalMLInputError,
        ),
    )


def test_temporal_cv_handoff_enabled():
    (
        dataframe,
        x,
        y,
        contract,
    ) = material()


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


def test_temporal_tuning_handoff_enabled():
    (
        dataframe,
        x,
        y,
        contract,
    ) = material()


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
        train_times.max()
        <
        test_times.min()
    )


    search_contract = (
        MLHyperparameterSearchContract(
            folds=
                2,

            shuffle=
                False,
        )
    )


    pairs = (
        _build_inner_cv_pairs(
            x_train=
                x_train,

            y_train=
                y_train,

            training_contract=
                contract,

            search_contract=
                search_contract,

            times_train=
                train_times,
        )
    )


    assert len(
        pairs
    ) == 2


    holdout_timestamps = set(
        test_times.tolist()
    )


    for (
        inner_train_indices,
        inner_validation_indices,
    ) in pairs:

        inner_times = (
            pd.concat(
                [
                    train_times.iloc[
                        inner_train_indices
                    ],
                    train_times.iloc[
                        inner_validation_indices
                    ],
                ]
            )
        )


        assert not (
            set(
                inner_times.tolist()
            )
            &
            holdout_timestamps
        )


def test_historical_split_contracts_preserved():
    row = (
        MLSplitContract()
    )


    group = (
        MLGroupHoldoutSplitContract(
            group_column=
                "client_id"
        )
    )


    assert (
        row.strategy
        ==
        "holdout"
    )


    assert (
        group.strategy
        ==
        "group_holdout"
    )


    assert (
        row.shuffle
        is True
    )


    assert (
        group.shuffle
        is True
    )


def test_frontend_temporal_contract():
    repo_root = (
        Path(
            __file__
        )
        .resolve()
        .parents[
            4
        ]
    )


    training_types = (
        repo_root
        /
        "apps/web/src/components/modelLab/modelTrainingTypes.ts"
    ).read_text(
        encoding=
            "utf-8"
    )


    lab_types = (
        repo_root
        /
        "apps/web/src/components/modelLab/modelLabTypes.ts"
    ).read_text(
        encoding=
            "utf-8"
    )


    client = (
        repo_root
        /
        "apps/web/src/app/model-lab/ModelLabClient.tsx"
    ).read_text(
        encoding=
            "utf-8"
    )


    assert (
        "ml_eligible_as_time"
        in
        training_types
    )


    assert (
        "ModelLabTimeHoldoutSplitContract"
        in
        lab_types
    )


    assert (
        '"time_holdout"'
        in
        lab_types
    )


    assert (
        "trainingTimeColumn"
        in
        client
    )


    assert (
        'value="time_holdout"'
        in
        client
    )


def main():
    print(
        "=== DATALENS TEMPORAL HOLDOUT / READINESS v0.1 ==="
    )


    test_temporal_contract_and_role_guards()

    print(
        "[PASS] temporal split contract + role guards"
    )


    test_server_time_readiness()

    print(
        "[PASS] server-owned observation-time readiness"
    )


    test_chronological_holdout_and_boundary()

    print(
        "[PASS] chronological holdout + timestamp boundary"
    )


    test_time_validator_fails_closed()

    print(
        "[PASS] datetime authority fails closed"
    )


    test_temporal_cv_handoff_enabled()

    print(
        "[PASS] temporal CV handoff enabled"
    )


    test_temporal_tuning_handoff_enabled()

    print(
        "[PASS] temporal tuning handoff enabled"
    )


    test_historical_split_contracts_preserved()

    print(
        "[PASS] historical row/group split contracts preserved"
    )


    test_frontend_temporal_contract()

    print(
        "[PASS] frontend temporal selection contract"
    )


    print()

    print(
        "TEMPORAL HOLDOUT / READINESS v0.1: PASS"
    )


if __name__ == "__main__":

    main()
