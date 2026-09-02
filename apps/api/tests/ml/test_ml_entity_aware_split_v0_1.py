from __future__ import annotations


import ast


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
)


from app.ml.contracts import (
    MLGroupHoldoutSplitContract,
    MLSplitContract,
    MLTrainingContract,
)


from app.ml.cross_validation import (
    MLCrossValidationContract,
)


from app.ml.cross_validation_executor import (
    MLCrossValidationInputError,
    execute_ml_cross_validation,
)


from app.ml.hyperparameter_tuning import (
    MLHyperparameterSearchContract,
)


from app.ml.hyperparameter_tuning_executor import (
    MLHyperparameterTuningInputError,
    execute_ml_hyperparameter_tuning,
)


WORKFLOW_ID = "prep:entity-aware-split"

DATASET_ID = "dataset:entity-aware-split"


def build_dataframe(
) -> pd.DataFrame:

    return (
        pd.DataFrame(
            {
                "row_id":
                    list(
                        range(
                            80
                        )
                    ),

                "client_id":
                    [
                        f"client_{index // 4:02d}"

                        for index
                        in range(
                            80
                        )
                    ],

                "segment":
                    [
                        (
                            "consumer"
                            if (
                                index
                                //
                                4
                            )
                            %
                            2
                            ==
                            0
                            else
                            "business"
                        )

                        for index
                        in range(
                            80
                        )
                    ],

                "age":
                    [
                        20
                        +
                        (
                            (
                                index
                                //
                                4
                            )
                            %
                            35
                        )

                        for index
                        in range(
                            80
                        )
                    ],

                "categ":
                    [
                        index
                        %
                        2

                        for index
                        in range(
                            80
                        )
                    ],
            }
        )
    )


def row_contract(
) -> MLTrainingContract:

    return (
        MLTrainingContract(
            workflow_id=
                WORKFLOW_ID,

            dataset_id=
                DATASET_ID,

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
                MLSplitContract(),
        )
    )


def group_contract(
    *,
    group_column: str =
        "client_id",
) -> MLTrainingContract:

    return (
        MLTrainingContract(
            workflow_id=
                WORKFLOW_ID,

            dataset_id=
                DATASET_ID,

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
                        group_column,
                ),
        )
    )


def test_historical_holdout_serialization_preserved(
) -> None:

    payload = (
        row_contract()
        .model_dump(
            mode="json"
        )
    )


    split = (
        payload[
            "split"
        ]
    )


    assert split == {
        "strategy":
            "holdout",

        "test_size":
            0.2,

        "random_seed":
            42,

        "shuffle":
            True,

        "stratify":
            False,
    }


    assert (
        "group_column"
        not in
        split
    )


def test_group_contract_role_guards(
) -> None:

    contract = (
        group_contract()
    )


    assert (
        contract
        .split
        .strategy
        ==
        "group_holdout"
    )


    assert (
        contract
        .split
        .group_column
        ==
        "client_id"
    )


    assert (
        "client_id"
        not in
        contract.feature_columns
    )


    cases = (
        (
            "categ",
            [
                "age",
            ],
        ),
        (
            "age",
            [
                "age",
            ],
        ),
    )


    for (
        group_column,
        feature_columns,
    ) in cases:

        try:

            MLTrainingContract(
                workflow_id=
                    WORKFLOW_ID,

                dataset_id=
                    DATASET_ID,

                problem_type=
                    "classification",

                target_column=
                    "categ",

                feature_columns=
                    feature_columns,

                categorical_feature_columns=[],

                estimator_key=
                    "logistic_regression",

                split=
                    MLGroupHoldoutSplitContract(
                        group_column=
                            group_column,
                    ),
            )


        except ValidationError:

            continue


        raise AssertionError(
            (
                "Invalid group role accepted: "
                f"{group_column}"
            )
        )


def test_server_owned_group_readiness(
) -> None:

    dataframe = (
        build_dataframe()
    )


    client = (
        _column_ml_readiness(
            column_name=
                "client_id",

            series=
                dataframe[
                    "client_id"
                ],
        )
    )


    row = (
        _column_ml_readiness(
            column_name=
                "row_id",

            series=
                dataframe[
                    "row_id"
                ],
        )
    )


    segment = (
        _column_ml_readiness(
            column_name=
                "segment",

            series=
                dataframe[
                    "segment"
                ],
        )
    )


    assert (
        client[
            "analytical_type"
        ]
        ==
        "identifier"
    )


    assert (
        client[
            "analytical_subtype"
        ]
        ==
        "reference"
    )


    assert (
        client[
            "ml_eligible_as_group"
        ]
        is True
    )


    assert (
        row[
            "ml_eligible_as_group"
        ]
        is False
    )


    assert (
        segment[
            "ml_eligible_as_group"
        ]
        is False
    )


def test_group_holdout_has_zero_entity_overlap(
) -> None:

    dataframe = (
        build_dataframe()
    )


    x = (
        dataframe[
            [
                "age",
            ]
        ]
        .copy(
            deep=True
        )
    )


    y = (
        dataframe[
            "categ"
        ]
        .copy(
            deep=True
        )
    )


    first = (
        _split_dataset(
            x=
                x,

            y=
                y,

            contract=
                group_contract(),

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
                group_contract(),

            dataframe=
                dataframe,
        )
    )


    (
        x_train,
        x_test,
        y_train,
        y_test,
    ) = first


    assert (
        list(
            x_train.index
        )
        ==
        list(
            second[
                0
            ].index
        )
    )


    assert (
        list(
            x_test.index
        )
        ==
        list(
            second[
                1
            ].index
        )
    )


    train_entities = set(
        dataframe.loc[
            x_train.index,
            "client_id",
        ]
    )


    test_entities = set(
        dataframe.loc[
            x_test.index,
            "client_id",
        ]
    )


    assert train_entities

    assert test_entities

    assert (
        train_entities
        .isdisjoint(
            test_entities
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


def expect_split_error(
    *,
    dataframe: pd.DataFrame | None,
    group_column: str,
    expected_text: str,
) -> None:

    source = (
        build_dataframe()
    )


    try:

        _split_dataset(
            x=
                source[
                    [
                        "age",
                    ]
                ],

            y=
                source[
                    "categ"
                ],

            contract=
                group_contract(
                    group_column=
                        group_column
                ),

            dataframe=
                dataframe,
        )


    except ClassicalMLInputError as error:

        assert (
            expected_text
            in
            str(
                error
            )
        )

        return


    raise AssertionError(
        (
            "Expected split error: "
            f"{expected_text}"
        )
    )


def test_invalid_entity_groups_fail_closed(
) -> None:

    dataframe = (
        build_dataframe()
    )


    expect_split_error(
        dataframe=
            None,

        group_column=
            "client_id",

        expected_text=
            "server-owned source dataframe",
    )


    expect_split_error(
        dataframe=
            dataframe,

        group_column=
            "account_id",

        expected_text=
            "missing from the validated dataset",
    )


    expect_split_error(
        dataframe=
            dataframe,

        group_column=
            "row_id",

        expected_text=
            "repeated reference identifier",
    )


    expect_split_error(
        dataframe=
            dataframe,

        group_column=
            "segment",

        expected_text=
            "repeated reference identifier",
    )


    missing = (
        dataframe.copy(
            deep=True
        )
    )


    missing.loc[
        0,
        "client_id",
    ] = None


    expect_split_error(
        dataframe=
            missing,

        group_column=
            "client_id",

        expected_text=
            "contains missing values",
    )


def test_historical_row_holdout_remains_compatible(
) -> None:

    dataframe = (
        build_dataframe()
    )


    (
        x_train,
        x_test,
        y_train,
        y_test,
    ) = (
        _split_dataset(
            x=
                dataframe[
                    [
                        "age",
                    ]
                ],

            y=
                dataframe[
                    "categ"
                ],

            contract=
                row_contract(),
        )
    )


    assert (
        len(
            x_train
        )
        >
        0
    )


    assert (
        len(
            x_test
        )
        >
        0
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


def test_group_aware_cv_and_tuning_handoff(
) -> None:

    repo = (
        Path(
            __file__
        )
        .resolve()
        .parents[
            4
        ]
    )


    cv_source = (
        repo
        /
        "apps/api/app/ml/cross_validation_executor.py"
    ).read_text(
        encoding="utf-8"
    )


    tuning_source = (
        repo
        /
        "apps/api/app/ml/hyperparameter_tuning_executor.py"
    ).read_text(
        encoding="utf-8"
    )


    assert (
        "GroupKFold"
        in
        cv_source
    )


    assert (
        "StratifiedGroupKFold"
        in
        cv_source
    )


    assert (
        "_build_cross_validation_pairs"
        in
        cv_source
    )


    assert (
        "_build_cross_validation_pairs"
        in
        tuning_source
    )


    assert (
        "Entity-aware Cross-Validation "
        "is not supported"
        not in
        cv_source
    )


    assert (
        "Group-aware INNER "
        "Cross-Validation is required"
        not in
        tuning_source
    )


def test_production_reconstructors_receive_dataframe(
) -> None:

    repo = (
        Path(
            __file__
        )
        .resolve()
        .parents[
            4
        ]
    )


    relative_paths = (
        "apps/api/app/ml/classical_executor.py",
        "apps/api/app/ml/classification_diagnostics_executor.py",
        "apps/api/app/ml/decision_threshold_executor.py",
        "apps/api/app/ml/model_evaluation_summary_executor.py",
        "apps/api/app/ml/model_explainability_executor.py",
    )


    for relative in relative_paths:

        source = (
            repo
            /
            relative
        ).read_text(
            encoding="utf-8"
        )


        tree = (
            ast.parse(
                source
            )
        )


        calls = [
            node

            for node
            in ast.walk(
                tree
            )

            if (
                isinstance(
                    node,
                    ast.Call,
                )
                and
                isinstance(
                    node.func,
                    ast.Name,
                )
                and
                node.func.id
                ==
                "_split_dataset"
            )
        ]


        assert (
            len(
                calls
            )
            ==
            1
        ), relative


        keywords = {
            keyword.arg

            for keyword
            in calls[
                0
            ].keywords

            if keyword.arg
            is not None
        }


        assert (
            "dataframe"
            in
            keywords
        ), relative


def test_frontend_entity_split_contract(
) -> None:

    repo = (
        Path(
            __file__
        )
        .resolve()
        .parents[
            4
        ]
    )


    client = (
        repo
        /
        "apps/web/src/app/model-lab/ModelLabClient.tsx"
    ).read_text(
        encoding="utf-8"
    )


    model_lab_types = (
        repo
        /
        "apps/web/src/components/modelLab/modelLabTypes.ts"
    ).read_text(
        encoding="utf-8"
    )


    training_types = (
        repo
        /
        "apps/web/src/components/modelLab/modelTrainingTypes.ts"
    ).read_text(
        encoding="utf-8"
    )


    assert (
        "ml_eligible_as_group:"
        in
        training_types
    )


    assert (
        "ModelLabGroupHoldoutSplitContract"
        in
        model_lab_types
    )


    assert (
        '"group_holdout"'
        in
        model_lab_types
    )


    assert (
        "group_column:"
        in
        model_lab_types
    )


    assert (
        "trainingSplitStrategy"
        in
        client
    )


    assert (
        "trainingGroupColumn"
        in
        client
    )


    assert (
        "eligibleTrainingGroupColumns"
        in
        client
    )


    assert (
        "column.ml_eligible_as_group"
        in
        client
    )


    assert (
        "changeTrainingSplitStrategy"
        in
        client
    )


    assert (
        "group_column:"
        in
        client
    )


    assert (
        (
            "selectedDetail.split.strategy ==="
        )
        in
        client
    )


    assert (
        (
            "selectedDetail"
            "\n"
            "                                                              .split"
            "\n"
            "                                                              .group_column"
        )
        in
        client
    )


    split_label = (
        "S\u00e9paration train / test"
    )


    entity_label = (
        "Par entit\u00e9"
    )


    assert (
        split_label
        in
        client
    )


    assert (
        entity_label
        in
        client
    )


def main(
) -> None:

    print(
        (
            "=== DATALENS ENTITY-AWARE "
            "SPLIT BACKEND v0.1 ==="
        )
    )


    test_historical_holdout_serialization_preserved()

    print(
        "[PASS] historical holdout serialization preserved"
    )


    test_group_contract_role_guards()

    print(
        "[PASS] explicit group contract + role guards"
    )


    test_server_owned_group_readiness()

    print(
        "[PASS] server-owned repeated-reference readiness"
    )


    test_group_holdout_has_zero_entity_overlap()

    print(
        "[PASS] train/test entity overlap = zero"
    )


    test_invalid_entity_groups_fail_closed()

    print(
        "[PASS] invalid entity group roles fail closed"
    )


    test_historical_row_holdout_remains_compatible()

    print(
        "[PASS] historical row holdout remains compatible"
    )


    test_group_aware_cv_and_tuning_handoff()

    print(
        "[PASS] group-aware CV/tuning handoff"
    )


    test_production_reconstructors_receive_dataframe()

    print(
        "[PASS] downstream holdout reconstruction is group-aware"
    )


    test_frontend_entity_split_contract()

    print(
        "[PASS] frontend explicit entity split contract"
    )


    print()
    print(
        "ENTITY-AWARE SPLIT BACKEND v0.1: PASS"
    )


if __name__ == "__main__":
    main()
