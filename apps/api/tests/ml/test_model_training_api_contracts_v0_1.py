from __future__ import annotations


from pydantic import (
    ValidationError,
)


from app.api.model_training_contracts import (
    MODEL_TRAINING_API_CONTRACT_RULE_VERSION,
    MODEL_TRAINING_REQUEST_RULE_VERSION,
    ModelTrainingAPIErrorDetail,
    ModelTrainingColumn,
    ModelTrainingContextResponse,
    ModelTrainingDataset,
    ModelTrainingRequest,
)


from app.ml.contracts import (
    MLTrainingContract,
)


# ============================================================
# CONSTANTS
# ============================================================


WORKFLOW_ID = (
    "prep:model-training-contracts"
)

DATASET_ID = (
    "dataset:model-training-contracts"
)


# ============================================================
# ASSERTIONS
# ============================================================


def expect_validation_error(
    factory,
) -> None:

    try:
        factory()

    except ValidationError:
        return

    raise AssertionError(
        "Expected pydantic ValidationError."
    )


# ============================================================
# FIXTURES
# ============================================================


def column(
    *,
    name: str = "age",
    kind: str = "numeric",
    nullable: bool = False,
) -> ModelTrainingColumn:

    return (
        ModelTrainingColumn(
            name=
                name,

            kind=
                kind,

            nullable=
                nullable,
        )
    )


def dataset(
    *,
    dataset_id: str = DATASET_ID,
) -> ModelTrainingDataset:

    return (
        ModelTrainingDataset(
            dataset_id=
                dataset_id,

            filename=
                "training.csv",

            row_count=
                100,

            column_count=
                3,

            columns=[
                column(
                    name=
                        "age",

                    kind=
                        "numeric",
                ),

                column(
                    name=
                        "segment",

                    kind=
                        "categorical",
                ),

                column(
                    name=
                        "is_active",

                    kind=
                        "boolean",
                ),
            ],
        )
    )


def training_contract(
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
                "segment",

            feature_columns=[
                "age",
                "is_active",
            ],

            categorical_feature_columns=[
                "is_active",
            ],

            estimator_key=
                "logistic_regression",
        )
    )


# ============================================================
# COLUMN
# ============================================================


def test_valid_training_column(
) -> None:

    result = (
        column(
            name=
                " age "
        )
    )

    assert (
        result.name
        ==
        "age"
    )

    assert (
        result.kind
        ==
        "numeric"
    )

    assert (
        result.rule_version
        ==
        MODEL_TRAINING_API_CONTRACT_RULE_VERSION
    )


def test_training_column_rejects_blank_name(
) -> None:

    expect_validation_error(
        lambda:
            ModelTrainingColumn(
                name=
                    "   ",

                kind=
                    "numeric",

                nullable=
                    False,
            )
    )


# ============================================================
# DATASET
# ============================================================


def test_valid_training_dataset(
) -> None:

    result = (
        dataset()
    )

    assert (
        result.dataset_id
        ==
        DATASET_ID
    )

    assert (
        result.row_count
        ==
        100
    )

    assert (
        result.column_count
        ==
        3
    )

    assert (
        [
            item.name
            for item
            in result.columns
        ]
        ==
        [
            "age",
            "segment",
            "is_active",
        ]
    )


def test_dataset_column_count_is_bound(
) -> None:

    payload = (
        dataset()
        .model_dump(
            mode="python"
        )
    )

    payload[
        "column_count"
    ] = 2

    expect_validation_error(
        lambda:
            ModelTrainingDataset(
                **payload
            )
    )


def test_dataset_rejects_duplicate_columns(
) -> None:

    expect_validation_error(
        lambda:
            ModelTrainingDataset(
                dataset_id=
                    DATASET_ID,

                filename=
                    "training.csv",

                row_count=
                    100,

                column_count=
                    2,

                columns=[
                    column(
                        name=
                            "age"
                    ),

                    column(
                        name=
                            "age"
                    ),
                ],
            )
    )


# ============================================================
# CONTEXT
# ============================================================


def test_valid_training_context(
) -> None:

    result = (
        ModelTrainingContextResponse(
            workflow_id=
                WORKFLOW_ID,

            preparation_session_revision=
                12,

            dataset_count=
                1,

            datasets=[
                dataset()
            ],
        )
    )

    assert (
        result.workflow_id
        ==
        WORKFLOW_ID
    )

    assert (
        result.preparation_session_revision
        ==
        12
    )

    assert (
        result.dataset_count
        ==
        1
    )


def test_context_dataset_count_is_bound(
) -> None:

    expect_validation_error(
        lambda:
            ModelTrainingContextResponse(
                workflow_id=
                    WORKFLOW_ID,

                preparation_session_revision=
                    12,

                dataset_count=
                    2,

                datasets=[
                    dataset()
                ],
            )
    )


def test_context_rejects_duplicate_dataset_ids(
) -> None:

    expect_validation_error(
        lambda:
            ModelTrainingContextResponse(
                workflow_id=
                    WORKFLOW_ID,

                preparation_session_revision=
                    12,

                dataset_count=
                    2,

                datasets=[
                    dataset(),

                    dataset(),
                ],
            )
    )


# ============================================================
# TRAIN REQUEST
# ============================================================


def test_valid_training_request(
) -> None:

    result = (
        ModelTrainingRequest(
            training=
                training_contract(),

            expected_preparation_session_revision=
                12,
        )
    )

    assert (
        result.training.workflow_id
        ==
        WORKFLOW_ID
    )

    assert (
        result.training.dataset_id
        ==
        DATASET_ID
    )

    assert (
        result.expected_preparation_session_revision
        ==
        12
    )

    assert (
        result.rule_version
        ==
        MODEL_TRAINING_REQUEST_RULE_VERSION
    )


def test_training_request_rejects_negative_revision(
) -> None:

    expect_validation_error(
        lambda:
            ModelTrainingRequest(
                training=
                    training_contract(),

                expected_preparation_session_revision=
                    -1,
            )
    )


def test_training_request_rejects_extra_authority(
) -> None:

    payload = {
        "training":
            training_contract()
            .model_dump(
                mode="python"
            ),

        "expected_preparation_session_revision":
            12,

        "dataframe":
            "caller-owned-data",
    }

    expect_validation_error(
        lambda:
            ModelTrainingRequest
            .model_validate(
                payload
            )
    )


# ============================================================
# ERROR / VERSIONS
# ============================================================


def test_structured_training_error(
) -> None:

    result = (
        ModelTrainingAPIErrorDetail(
            error=
                "training_input_invalid",

            message=
                "Training input is invalid.",

            workflow_id=
                WORKFLOW_ID,
        )
    )

    assert (
        result.api_version
        ==
        "model_training_api_v0.1"
    )

    assert (
        result.retryable
        is False
    )

    assert (
        result.workflow_id
        ==
        WORKFLOW_ID
    )


def test_rule_versions(
) -> None:

    assert (
        MODEL_TRAINING_API_CONTRACT_RULE_VERSION
        ==
        "model_training_api_contract_v0.1"
    )

    assert (
        MODEL_TRAINING_REQUEST_RULE_VERSION
        ==
        "model_training_request_v0.1"
    )


# ============================================================
# MAIN
# ============================================================


def main(
) -> None:

    print(
        "=== DATALENS MODEL TRAINING API CONTRACTS v0.1 ==="
    )

    tests = [
        (
            "Valid normalized training column",
            test_valid_training_column,
        ),

        (
            "Blank training column blocked",
            test_training_column_rejects_blank_name,
        ),

        (
            "Valid server-owned training dataset",
            test_valid_training_dataset,
        ),

        (
            "Dataset column count bound",
            test_dataset_column_count_is_bound,
        ),

        (
            "Duplicate dataset columns blocked",
            test_dataset_rejects_duplicate_columns,
        ),

        (
            "Valid Preparation-derived training context",
            test_valid_training_context,
        ),

        (
            "Context dataset count bound",
            test_context_dataset_count_is_bound,
        ),

        (
            "Duplicate context dataset ids blocked",
            test_context_rejects_duplicate_dataset_ids,
        ),

        (
            "Valid training request",
            test_valid_training_request,
        ),

        (
            "Negative Preparation revision blocked",
            test_training_request_rejects_negative_revision,
        ),

        (
            "Caller-owned dataframe authority blocked",
            test_training_request_rejects_extra_authority,
        ),

        (
            "Structured Model Training API error",
            test_structured_training_error,
        ),

        (
            "Model Training contract rule versions",
            test_rule_versions,
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
        "PASS - Model Training API Contracts v0.1"
    )


if __name__ == "__main__":
    main()
