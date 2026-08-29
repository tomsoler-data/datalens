from __future__ import annotations


from types import (
    SimpleNamespace,
)


import pandas as pd


import app.api.model_training_service as service


from app.api.model_training_contracts import (
    ModelTrainingRequest,
)


from app.api.model_training_service import (
    MODEL_TRAINING_SERVICE_RULE_VERSION,
    ModelTrainingContextError,
    ModelTrainingEstimatorError,
    ModelTrainingExecutionError,
    ModelTrainingInputError,
    get_model_training_context,
    train_model,
)


from app.ml.classical_executor import (
    ClassicalMLEstimatorError,
    ClassicalMLExecutorError,
    ClassicalMLInputError,
)


from app.ml.contracts import (
    MLTrainingContract,
)


from app.preparation.analysis_input_handoff import (
    AnalysisInputHandoffError,
)


# ============================================================
# CONSTANTS
# ============================================================


WORKFLOW_ID = (
    "prep:model-training-service"
)

DATASET_ID = (
    "dataset:model-training-service"
)

MODEL_ID = (
    "model:model-training-service"
)


# ============================================================
# PATCH
# ============================================================


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


# ============================================================
# ASSERTIONS
# ============================================================


def expect_exception(
    exception_type,
    factory,
) -> None:

    try:
        factory()

    except exception_type:
        return

    raise AssertionError(
        (
            "Expected exception: "
            f"{exception_type.__name__}"
        )
    )


# ============================================================
# FIXTURES
# ============================================================


def build_dataframe(
) -> pd.DataFrame:

    return (
        pd.DataFrame(
            {
                "age":
                    pd.Series(
                        [
                            34.0,
                            None,
                            42.0,
                        ],
                        dtype="float64",
                    ),

                "quantity":
                    pd.Series(
                        [
                            1,
                            2,
                            3,
                        ],
                        dtype="int64",
                    ),

                "is_active":
                    pd.Series(
                        [
                            True,
                            False,
                            True,
                        ],
                        dtype="bool",
                    ),

                "segment":
                    pd.Series(
                        [
                            "SMB",
                            "Enterprise",
                            "Consumer",
                        ],
                        dtype="string",
                    ),

                "category":
                    pd.Series(
                        pd.Categorical(
                            [
                                "A",
                                "B",
                                "A",
                            ]
                        )
                    ),

                "created_at":
                    pd.to_datetime(
                        [
                            "2026-01-01",
                            "2026-01-02",
                            "2026-01-03",
                        ]
                    ),
            }
        )
    )


def build_handoff(
    *,
    dataframe=None,
):

    if dataframe is None:
        dataframe = (
            build_dataframe()
        )


    return (
        SimpleNamespace(
            workflow_id=
                WORKFLOW_ID,

            session_revision=
                17,

            dataset_records=[
                {
                    "dataset_id":
                        DATASET_ID,

                    "filename":
                        "training.csv",

                    "dataframe":
                        dataframe,
                }
            ],
        )
    )


def build_training_contract(
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
                "quantity",
                "is_active",
            ],

            categorical_feature_columns=[
                "is_active",
            ],

            estimator_key=
                "logistic_regression",
        )
    )


def build_request(
) -> ModelTrainingRequest:

    return (
        ModelTrainingRequest(
            training=
                build_training_contract(),

            expected_preparation_session_revision=
                17,
        )
    )


# ============================================================
# CONTEXT
# ============================================================


def test_context_uses_validated_handoff(
) -> None:

    calls = []


    def fake_load(
        *,
        workflow_id: str,
    ):

        calls.append(
            workflow_id
        )

        return (
            build_handoff()
        )


    with Patch(
        "load_validated_analysis_input",
        fake_load,
    ):

        result = (
            get_model_training_context(
                workflow_id=
                    WORKFLOW_ID
            )
        )


    assert (
        calls
        ==
        [
            WORKFLOW_ID
        ]
    )

    assert (
        result.workflow_id
        ==
        WORKFLOW_ID
    )

    assert (
        result.preparation_session_revision
        ==
        17
    )

    assert (
        result.dataset_count
        ==
        1
    )


def test_context_projects_safe_column_metadata(
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


    dataset = (
        result.datasets[
            0
        ]
    )


    assert (
        dataset.dataset_id
        ==
        DATASET_ID
    )

    assert (
        dataset.filename
        ==
        "training.csv"
    )

    assert (
        dataset.row_count
        ==
        3
    )

    assert (
        dataset.column_count
        ==
        6
    )


    metadata = {
        item.name:
            (
                item.kind,
                item.nullable,
            )

        for item
        in dataset.columns
    }


    assert (
        metadata["age"]
        ==
        (
            "numeric",
            True,
        )
    )

    assert (
        metadata["quantity"]
        ==
        (
            "numeric",
            False,
        )
    )

    assert (
        metadata["is_active"]
        ==
        (
            "boolean",
            False,
        )
    )

    assert (
        metadata["segment"]
        ==
        (
            "categorical",
            False,
        )
    )

    assert (
        metadata["category"]
        ==
        (
            "categorical",
            False,
        )
    )

    assert (
        metadata["created_at"]
        ==
        (
            "datetime",
            False,
        )
    )


def test_context_handoff_failure_maps_to_context_error(
) -> None:

    def fail(
        **kwargs,
    ):
        raise (
            AnalysisInputHandoffError(
                "handoff unavailable"
            )
        )


    with Patch(
        "load_validated_analysis_input",
        fail,
    ):

        expect_exception(
            ModelTrainingContextError,

            lambda:
                get_model_training_context(
                    workflow_id=
                        WORKFLOW_ID
                ),
        )


def test_context_rejects_invalid_dataset_record(
) -> None:

    handoff = (
        build_handoff()
    )

    handoff.dataset_records = [
        "not-a-record"
    ]


    with Patch(
        "load_validated_analysis_input",
        lambda **kwargs:
            handoff,
    ):

        expect_exception(
            ModelTrainingContextError,

            lambda:
                get_model_training_context(
                    workflow_id=
                        WORKFLOW_ID
                ),
        )


def test_context_rejects_non_dataframe(
) -> None:

    handoff = (
        build_handoff()
    )

    handoff.dataset_records[
        0
    ][
        "dataframe"
    ] = [
        {
            "age":
                42
        }
    ]


    with Patch(
        "load_validated_analysis_input",
        lambda **kwargs:
            handoff,
    ):

        expect_exception(
            ModelTrainingContextError,

            lambda:
                get_model_training_context(
                    workflow_id=
                        WORKFLOW_ID
                ),
        )


def test_context_rejects_non_string_columns(
) -> None:

    dataframe = (
        pd.DataFrame(
            [
                [
                    1,
                    2,
                ]
            ],
            columns=[
                "age",
                123,
            ],
        )
    )


    with Patch(
        "load_validated_analysis_input",
        lambda **kwargs:
            build_handoff(
                dataframe=
                    dataframe
            ),
    ):

        expect_exception(
            ModelTrainingContextError,

            lambda:
                get_model_training_context(
                    workflow_id=
                        WORKFLOW_ID
                ),
        )


def test_context_rejects_duplicate_normalized_columns(
) -> None:

    dataframe = (
        pd.DataFrame(
            [
                [
                    1,
                    2,
                ]
            ],
            columns=[
                "age",
                " age ",
            ],
        )
    )


    with Patch(
        "load_validated_analysis_input",
        lambda **kwargs:
            build_handoff(
                dataframe=
                    dataframe
            ),
    ):

        expect_exception(
            ModelTrainingContextError,

            lambda:
                get_model_training_context(
                    workflow_id=
                        WORKFLOW_ID
                ),
        )


def test_context_rejects_empty_dataset_collection(
) -> None:

    handoff = (
        build_handoff()
    )

    handoff.dataset_records = []


    with Patch(
        "load_validated_analysis_input",
        lambda **kwargs:
            handoff,
    ):

        expect_exception(
            ModelTrainingContextError,

            lambda:
                get_model_training_context(
                    workflow_id=
                        WORKFLOW_ID
                ),
        )


# ============================================================
# TRAINING
# ============================================================


def test_training_delegates_exact_authority_to_classical_ml(
) -> None:

    request = (
        build_request()
    )

    calls = []

    expected_detail = (
        object()
    )


    def fake_execute(
        *,
        training_contract,
        expected_preparation_session_revision,
    ):

        calls.append(
            (
                training_contract,
                expected_preparation_session_revision,
            )
        )

        return (
            SimpleNamespace(
                model_artifact=
                    SimpleNamespace(
                        model_id=
                            MODEL_ID
                    )
            )
        )


    detail_calls = []


    def fake_detail(
        *,
        workflow_id: str,
        model_id: str,
    ):

        detail_calls.append(
            (
                workflow_id,
                model_id,
            )
        )

        return (
            expected_detail
        )


    with Patch(
        "execute_classical_ml",
        fake_execute,
    ), Patch(
        "get_model_lab_model_detail",
        fake_detail,
    ):

        result = (
            train_model(
                request
            )
        )


    assert (
        result
        is
        expected_detail
    )

    assert (
        len(
            calls
        )
        ==
        1
    )

    called_contract, called_revision = (
        calls[
            0
        ]
    )

    assert (
        called_contract
        ==
        request.training
    )

    assert (
        called_revision
        ==
        17
    )

    assert (
        detail_calls
        ==
        [
            (
                WORKFLOW_ID,
                MODEL_ID,
            )
        ]
    )


def test_training_invalid_public_request_maps_to_input_error(
) -> None:

    expect_exception(
        ModelTrainingInputError,

        lambda:
            train_model(
                {
                    "training":
                        {
                            "workflow_id":
                                WORKFLOW_ID
                        },

                    "expected_preparation_session_revision":
                        17,
                }
            ),
    )


def test_classical_input_error_maps_to_training_input_error(
) -> None:

    def fail(
        **kwargs,
    ):
        raise (
            ClassicalMLInputError(
                "bad input"
            )
        )


    with Patch(
        "execute_classical_ml",
        fail,
    ):

        expect_exception(
            ModelTrainingInputError,

            lambda:
                train_model(
                    build_request()
                ),
        )


def test_classical_estimator_error_maps_to_training_estimator_error(
) -> None:

    def fail(
        **kwargs,
    ):
        raise (
            ClassicalMLEstimatorError(
                "bad estimator"
            )
        )


    with Patch(
        "execute_classical_ml",
        fail,
    ):

        expect_exception(
            ModelTrainingEstimatorError,

            lambda:
                train_model(
                    build_request()
                ),
        )


def test_classical_executor_error_maps_to_training_execution_error(
) -> None:

    def fail(
        **kwargs,
    ):
        raise (
            ClassicalMLExecutorError(
                "training failed"
            )
        )


    with Patch(
        "execute_classical_ml",
        fail,
    ):

        expect_exception(
            ModelTrainingExecutionError,

            lambda:
                train_model(
                    build_request()
                ),
        )


def test_persisted_detail_restore_failure_maps_to_execution_error(
) -> None:

    def fake_execute(
        **kwargs,
    ):

        return (
            SimpleNamespace(
                model_artifact=
                    SimpleNamespace(
                        model_id=
                            MODEL_ID
                    )
            )
        )


    def fail_detail(
        **kwargs,
    ):

        from app.api.model_lab_service import (
            ModelLabArtifactError,
        )

        raise (
            ModelLabArtifactError(
                "artifact unavailable"
            )
        )


    with Patch(
        "execute_classical_ml",
        fake_execute,
    ), Patch(
        "get_model_lab_model_detail",
        fail_detail,
    ):

        expect_exception(
            ModelTrainingExecutionError,

            lambda:
                train_model(
                    build_request()
                ),
        )


# ============================================================
# VERSION
# ============================================================


def test_rule_version(
) -> None:

    assert (
        MODEL_TRAINING_SERVICE_RULE_VERSION
        ==
        "model_training_service_v0.1"
    )


# ============================================================
# MAIN
# ============================================================


def main(
) -> None:

    print(
        "=== DATALENS MODEL TRAINING SERVICE v0.1 ==="
    )


    tests = [
        (
            "Context uses validated Preparation handoff",
            test_context_uses_validated_handoff,
        ),

        (
            "Context exposes safe typed column metadata",
            test_context_projects_safe_column_metadata,
        ),

        (
            "Handoff failure maps to context error",
            test_context_handoff_failure_maps_to_context_error,
        ),

        (
            "Invalid dataset record blocked",
            test_context_rejects_invalid_dataset_record,
        ),

        (
            "Non-DataFrame dataset blocked",
            test_context_rejects_non_dataframe,
        ),

        (
            "Non-string column names blocked",
            test_context_rejects_non_string_columns,
        ),

        (
            "Duplicate normalized column names blocked",
            test_context_rejects_duplicate_normalized_columns,
        ),

        (
            "Empty trainable dataset collection blocked",
            test_context_rejects_empty_dataset_collection,
        ),

        (
            "Training delegates exact server authority",
            test_training_delegates_exact_authority_to_classical_ml,
        ),

        (
            "Invalid public training request maps to input error",
            test_training_invalid_public_request_maps_to_input_error,
        ),

        (
            "Classical ML input error mapped",
            test_classical_input_error_maps_to_training_input_error,
        ),

        (
            "Classical ML estimator error mapped",
            test_classical_estimator_error_maps_to_training_estimator_error,
        ),

        (
            "Classical ML executor error mapped",
            test_classical_executor_error_maps_to_training_execution_error,
        ),

        (
            "Persisted detail restore failure mapped",
            test_persisted_detail_restore_failure_maps_to_execution_error,
        ),

        (
            "Model Training service rule version",
            test_rule_version,
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
        "PASS - Model Training Service v0.1"
    )


if __name__ == "__main__":
    main()
