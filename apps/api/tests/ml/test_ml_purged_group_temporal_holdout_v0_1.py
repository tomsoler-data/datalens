from __future__ import annotations


import inspect


import pandas as pd


from app.ml.classical_executor import (
    ClassicalMLExecutionResult,
    ClassicalMLInputError,
    _split_dataset,
    execute_classical_ml,
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
    execute_ml_cross_validation,
)


from app.ml.hyperparameter_tuning import (
    MLHyperparameterSearchContract,
    hyperparameter_validation_strategy,
)


from app.ml.hyperparameter_tuning_executor import (
    MLHyperparameterTuningInputError,
    execute_ml_hyperparameter_tuning,
)


from tests.ml.test_classical_ml_executor_v0_1 import (
    isolated_environment,
    patched_handoff,
    seed_preparation_authority,
)


def expect_error(
    callback,
    error_type,
    message_fragment: str,
) -> None:

    try:

        callback()

    except error_type as error:

        assert (
            message_fragment
            in
            str(
                error
            )
        )

        return


    raise AssertionError(
        (
            f"Expected {error_type.__name__} "
            "was not raised."
        )
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
                        "2026-01-09",
                        "2026-01-10",
                        "2026-01-11",
                        "2026-01-12",
                    ]
                ),

            "client_id":
                [
                    "client_a",
                    "client_b",
                    "client_c",
                    "client_c",
                    "client_d",
                    "client_d",
                    "client_e",
                    "client_e",
                    "client_f",
                    "client_a",
                    "client_b",
                    "client_f",
                ],

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
                    11.0,
                    12.0,
                ],

            "target":
                [
                    11.0,
                    13.0,
                    15.0,
                    17.0,
                    19.0,
                    21.0,
                    23.0,
                    25.0,
                    27.0,
                    29.0,
                    31.0,
                    33.0,
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
                "workflow:purged-group-time",

            dataset_id=
                "dataset:purged-group-time",

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
                MLPurgedGroupTimeHoldoutSplitContract(
                    group_column=
                        "client_id",

                    time_column=
                        "event_time",

                    test_size=
                        0.25,
                ),
        )
    )


    return (
        dataframe,
        x,
        y,
        contract,
    )


def test_contract_and_role_guards():

    split = (
        MLPurgedGroupTimeHoldoutSplitContract(
            group_column=
                " client_id ",

            time_column=
                " event_time ",

            test_size=
                0.25,
        )
    )


    assert (
        split.strategy
        ==
        "purged_group_time_holdout"
    )


    assert (
        split.group_column
        ==
        "client_id"
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
            MLPurgedGroupTimeHoldoutSplitContract(
                group_column=
                    "same",

                time_column=
                    "same",
            ),

        ValueError,

        "different columns",
    )


    expect_error(
        lambda:
            MLTrainingContract(
                workflow_id=
                    "workflow:group-target-guard",

                dataset_id=
                    "dataset:group-target-guard",

                problem_type=
                    "regression",

                target_column=
                    "client_id",

                feature_columns=[
                    "feature",
                ],

                categorical_feature_columns=[],

                estimator_key=
                    "linear_regression",

                split=
                    MLPurgedGroupTimeHoldoutSplitContract(
                        group_column=
                            "client_id",

                        time_column=
                            "event_time",
                    ),
            ),

        ValueError,

        "group_column cannot also be target_column",
    )


    expect_error(
        lambda:
            MLTrainingContract(
                workflow_id=
                    "workflow:time-feature-guard",

                dataset_id=
                    "dataset:time-feature-guard",

                problem_type=
                    "regression",

                target_column=
                    "target",

                feature_columns=[
                    "feature",
                    "event_time",
                ],

                categorical_feature_columns=[],

                estimator_key=
                    "linear_regression",

                split=
                    MLPurgedGroupTimeHoldoutSplitContract(
                        group_column=
                            "client_id",

                        time_column=
                            "event_time",
                    ),
            ),

        ValueError,

        (
            "time_column cannot also "
            "be present in feature_columns"
        ),
    )


def test_temporal_cut_then_group_purge():

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
        train_groups,
        test_groups,
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

            return_group_partitions=
                True,

            return_time_partitions=
                True,
        )
    )


    assert (
        x_test.index.tolist()
        ==
        [
            9,
            10,
            11,
        ]
    )


    assert (
        test_times.tolist()
        ==
        pd.to_datetime(
            [
                "2026-01-10",
                "2026-01-11",
                "2026-01-12",
            ]
        ).tolist()
    )


    assert (
        set(
            test_groups.tolist()
        )
        ==
        {
            "client_a",
            "client_b",
            "client_f",
        }
    )


    assert (
        x_train.index.tolist()
        ==
        [
            2,
            3,
            4,
            5,
            6,
            7,
        ]
    )


    assert (
        len(
            x
        )
        ==
        (
            len(
                x_train
            )
            +
            len(
                x_test
            )
            +
            3
        )
    )


    assert (
        len(
            x_train
        )
        ==
        len(
            y_train
        )
    )


    assert (
        len(
            x_test
        )
        ==
        len(
            y_test
        )
    )


    assert (
        train_groups.index.equals(
            x_train.index
        )
    )


    assert (
        test_groups.index.equals(
            x_test.index
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


    assert not (
        set(
            train_groups.tolist()
        )
        &
        set(
            test_groups.tolist()
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


def test_determinism():

    (
        dataframe,
        x,
        y,
        contract,
    ) = material()


    first = (
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


    second = (
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


    assert (
        first[
            0
        ].index.tolist()
        ==
        second[
            0
        ].index.tolist()
    )


    assert (
        first[
            1
        ].index.tolist()
        ==
        second[
            1
        ].index.tolist()
    )


def test_joint_metadata_combined_only():

    (
        dataframe,
        x,
        y,
        _,
    ) = material()


    temporal_contract = (
        MLTrainingContract(
            workflow_id=
                "workflow:time-only",

            dataset_id=
                "dataset:time-only",

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
                ),
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
                    temporal_contract,

                dataframe=
                    dataframe,

                return_group_partitions=
                    True,

                return_time_partitions=
                    True,
            ),

        ClassicalMLInputError,

        "outside purged_group_time_holdout",
    )


def test_execution_result_surface():

    field = (
        ClassicalMLExecutionResult
        .model_fields[
            "purged_rows"
        ]
    )


    assert (
        field.default
        ==
        0
    )


    source = (
        inspect.getsource(
            execute_classical_ml
        )
    )


    assert (
        "purged_rows="
        in
        source
    )


    assert (
        "A non-purged holdout strategy"
        in
        source
    )


def test_full_classical_execution_reports_purged_rows():

    (
        dataframe,
        _,
        _,
        contract,
    ) = material()


    with isolated_environment():

        seed_preparation_authority(
            workflow_id=
                contract.workflow_id,

            dataset_id=
                contract.dataset_id,
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
                        contract,
                )
            )


        assert (
            result.train_rows
            ==
            6
        )


        assert (
            result.test_rows
            ==
            3
        )


        assert (
            result.purged_rows
            ==
            3
        )


        assert (
            result
            .model_artifact
            .training_contract
            .split
            .strategy
            ==
            "purged_group_time_holdout"
        )


        assert (
            result
            .model_artifact
            .train_rows
            ==
            6
        )


        assert (
            result
            .model_artifact
            .test_rows
            ==
            3
        )


        assert (
            result.experiment_provenance
            is not None
        )


        assert (
            result
            .experiment_provenance
            .train_rows
            ==
            6
        )


        assert (
            result
            .experiment_provenance
            .test_rows
            ==
            3
        )


def test_combined_cv_fail_closed():

    (
        _,
        _,
        _,
        contract,
    ) = material()


    expect_error(
        lambda:
            execute_ml_cross_validation(
                training_contract=
                    contract,

                cross_validation_contract=
                    MLCrossValidationContract(
                        folds=
                            2,

                        shuffle=
                            False,
                    ),
            ),

        MLCrossValidationInputError,

        "E15b",
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

        ValueError,

        "E15b",
    )


def test_combined_tuning_fail_closed():

    (
        _,
        _,
        _,
        contract,
    ) = material()


    expect_error(
        lambda:
            execute_ml_hyperparameter_tuning(
                training_contract=
                    contract,

                search_contract=
                    MLHyperparameterSearchContract(
                        folds=
                            2,

                        shuffle=
                            False,
                    ),
            ),

        MLHyperparameterTuningInputError,

        "E15b",
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

        ValueError,

        "E15b",
    )


def main():

    print(
        "=== DATALENS PURGED GROUP + TEMPORAL HOLDOUT v0.1 ==="
    )


    test_contract_and_role_guards()

    print(
        "[PASS] combined split contract + role guards"
    )


    test_temporal_cut_then_group_purge()

    print(
        "[PASS] chronological future TEST"
    )

    print(
        "[PASS] historical future-group purge"
    )

    print(
        "[PASS] train/test entity overlap = zero"
    )

    print(
        "[PASS] strict train_time < test_time"
    )

    print(
        "[PASS] source rows = train + test + purged"
    )


    test_determinism()

    print(
        "[PASS] deterministic combined holdout"
    )


    test_joint_metadata_combined_only()

    print(
        "[PASS] joint metadata combined-only"
    )


    test_execution_result_surface()

    print(
        "[PASS] execution result exposes purged_rows"
    )


    test_full_classical_execution_reports_purged_rows()

    print(
        "[PASS] full Classical ML execution reports purged_rows"
    )

    print(
        "[PASS] combined Training Contract persists in Model Artifact"
    )

    print(
        "[PASS] combined execution provenance remains valid"
    )


    test_combined_cv_fail_closed()

    print(
        "[PASS] combined CV fails closed to E15b"
    )


    test_combined_tuning_fail_closed()

    print(
        "[PASS] combined tuning fails closed to E15b"
    )


    print()
    print(
        "PURGED GROUP + TEMPORAL HOLDOUT v0.1: PASS"
    )


if __name__ == "__main__":

    main()
