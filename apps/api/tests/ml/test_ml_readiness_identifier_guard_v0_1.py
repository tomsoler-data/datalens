from __future__ import annotations


from pathlib import Path


from types import (
    SimpleNamespace,
)


import pandas as pd


import app.api.model_training_service as service


from app.api.model_training_service import (
    get_model_training_context,
)


from app.ml.classical_executor import (
    ClassicalMLInputError,
    _validate_and_extract_xy,
)


from app.ml.contracts import (
    MLTrainingContract,
)


WORKFLOW_ID = (
    "prep:ml-readiness-identifier"
)


DATASET_ID = (
    "dataset:ml-readiness-identifier"
)


class Patch:

    def __init__(
        self,
        name: str,
        value,
    ) -> None:

        self.name = name
        self.value = value

        self.original = getattr(
            service,
            name,
        )


    def __enter__(
        self,
    ):

        setattr(
            service,
            self.name,
            self.value,
        )

        return self


    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ):

        setattr(
            service,
            self.name,
            self.original,
        )


def build_dataframe(
) -> pd.DataFrame:

    row_count = 120

    return (
        pd.DataFrame(
            {
                "row_id":
                    list(
                        range(
                            row_count
                        )
                    ),

                "client_id":
                    [
                        f"c_{index // 4}"

                        for index
                        in range(
                            row_count
                        )
                    ],

                "session_id":
                    [
                        f"s_{index // 2}"

                        for index
                        in range(
                            row_count
                        )
                    ],

                "id_prod":
                    [
                        f"p_{index % 20}"

                        for index
                        in range(
                            row_count
                        )
                    ],

                "sex":
                    [
                        (
                            "f"
                            if index % 2 == 0
                            else "m"
                        )

                        for index
                        in range(
                            row_count
                        )
                    ],

                "birth":
                    [
                        1950
                        +
                        (
                            index
                            %
                            50
                        )

                        for index
                        in range(
                            row_count
                        )
                    ],

                "categ":
                    [
                        index
                        %
                        3

                        for index
                        in range(
                            row_count
                        )
                    ],
            }
        )
    )


def build_handoff(
):
    return (
        SimpleNamespace(
            workflow_id=
                WORKFLOW_ID,

            session_revision=
                41,

            dataset_records=[
                {
                    "dataset_id":
                        DATASET_ID,

                    "filename":
                        "lapage-like.csv",

                    "dataframe":
                        build_dataframe(),
                }
            ],
        )
    )


def contract(
    *,
    target_column: str,
    feature_columns: list[
        str
    ],
    categorical_feature_columns: list[
        str
    ],
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
                target_column,

            feature_columns=
                feature_columns,

            categorical_feature_columns=
                categorical_feature_columns,

            estimator_key=
                "logistic_regression",
        )
    )


def expect_identifier_error(
    *,
    target_column: str,
    feature_columns: list[
        str
    ],
    categorical_feature_columns: list[
        str
    ],
    expected_text: str,
) -> None:

    try:

        _validate_and_extract_xy(
            dataframe=
                build_dataframe(),

            contract=
                contract(
                    target_column=
                        target_column,

                    feature_columns=
                        feature_columns,

                    categorical_feature_columns=
                        categorical_feature_columns,
                ),
        )

    except ClassicalMLInputError as error:

        message = str(
            error
        )

        assert (
            expected_text
            in
            message
        )

        return


    raise AssertionError(
        "Expected identifier-role rejection."
    )


def test_context_exposes_identifier_readiness(
) -> None:

    with Patch(
        "load_validated_analysis_input",
        lambda **kwargs:
            build_handoff(),
    ):

        result = (
            get_model_training_context(
                workflow_id=
                    WORKFLOW_ID
            )
        )


    columns = {
        item.name:
            item

        for item
        in result.datasets[
            0
        ].columns
    }


    for name in (
        "row_id",
        "client_id",
        "session_id",
        "id_prod",
    ):

        item = (
            columns[
                name
            ]
        )

        assert (
            item.analytical_type
            ==
            "identifier"
        )

        assert (
            item.ml_eligible_as_target
            is False
        )

        assert (
            item.ml_eligible_as_feature
            is False
        )

        assert (
            item.exclusion_reason
            ==
            "identifier"
        )


    assert (
        columns[
            "sex"
        ]
        .ml_eligible_as_feature
        is True
    )

    assert (
        columns[
            "birth"
        ]
        .ml_eligible_as_feature
        is True
    )

    assert (
        columns[
            "categ"
        ]
        .ml_eligible_as_target
        is True
    )


def test_executor_rejects_identifier_feature(
) -> None:

    expect_identifier_error(
        target_column=
            "categ",

        feature_columns=[
            "client_id",
            "sex",
        ],

        categorical_feature_columns=[
            "client_id",
            "sex",
        ],

        expected_text=(
            "Identifier columns cannot be "
            "used as ML features"
        ),
    )


def test_executor_rejects_identifier_target(
) -> None:

    expect_identifier_error(
        target_column=
            "row_id",

        feature_columns=[
            "sex",
            "birth",
        ],

        categorical_feature_columns=[
            "sex",
        ],

        expected_text=(
            "Identifier columns cannot be "
            "used as ML targets"
        ),
    )


def test_valid_lapage_like_roles_remain_allowed(
) -> None:

    x, y = (
        _validate_and_extract_xy(
            dataframe=
                build_dataframe(),

            contract=
                contract(
                    target_column=
                        "categ",

                    feature_columns=[
                        "sex",
                        "birth",
                    ],

                    categorical_feature_columns=[
                        "sex",
                    ],
                ),
        )
    )


    assert (
        list(
            x.columns
        )
        ==
        [
            "sex",
            "birth",
        ]
    )

    assert (
        y.name
        ==
        "categ"
    )

    assert (
        len(
            x
        )
        ==
        120
    )


def test_frontend_consumes_server_readiness(
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


    client_source = (
        repo
        /
        "apps/web/src/app/model-lab/ModelLabClient.tsx"
    ).read_text(
        encoding="utf-8"
    )


    types_source = (
        repo
        /
        "apps/web/src/components/modelLab/modelTrainingTypes.ts"
    ).read_text(
        encoding="utf-8"
    )


    assert (
        "ml_eligible_as_target"
        in
        client_source
    )

    assert (
        "ml_eligible_as_feature"
        in
        client_source
    )

    assert (
        "identifierTrainingColumns"
        in
        client_source
    )

    assert (
        "Identifiants détectés par DataLens"
        in
        client_source
    )

    assert (
        "analytical_type:"
        in
        types_source
    )

    assert (
        "ml_eligible_as_target:"
        in
        types_source
    )

    assert (
        "ml_eligible_as_feature:"
        in
        types_source
    )


def main(
) -> None:

    print(
        (
            "=== DATALENS ML READINESS "
            "IDENTIFIER GUARD v0.1 ==="
        )
    )


    test_context_exposes_identifier_readiness()

    print(
        (
            "[PASS] context exposes "
            "server-owned identifier readiness"
        )
    )


    test_executor_rejects_identifier_feature()

    print(
        (
            "[PASS] identifier features "
            "rejected server-side"
        )
    )


    test_executor_rejects_identifier_target()

    print(
        (
            "[PASS] identifier targets "
            "rejected server-side"
        )
    )


    test_valid_lapage_like_roles_remain_allowed()

    print(
        "[PASS] valid Lapage-like roles preserved"
    )


    test_frontend_consumes_server_readiness()

    print(
        (
            "[PASS] frontend consumes "
            "server-owned readiness"
        )
    )


    print()
    print(
        (
            "ML Readiness Identifier Guard "
            "v0.1: PASS"
        )
    )


if __name__ == "__main__":
    main()
