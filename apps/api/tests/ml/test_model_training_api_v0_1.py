from __future__ import annotations


from fastapi import (
    FastAPI,
)


from fastapi.testclient import (
    TestClient,
)


import app.api.model_training as model_training_api


from app.api.model_lab_contracts import (
    ModelLabModelDetail,
)


from app.api.model_training import (
    MODEL_TRAINING_API_VERSION,
    router,
)


from app.api.model_training_contracts import (
    ModelTrainingColumn,
    ModelTrainingContextResponse,
    ModelTrainingDataset,
)


from app.api.model_training_service import (
    ModelTrainingContextError,
    ModelTrainingEstimatorError,
    ModelTrainingExecutionError,
    ModelTrainingInputError,
    ModelTrainingServiceError,
)


from app.ml.contracts import (
    MLPreprocessingContract,
    MLSplitContract,
)


from app.ml.estimator_contracts import (
    MLLogisticRegressionHyperparameters,
)


# ============================================================
# CONSTANTS
# ============================================================


WORKFLOW_ID = (
    "prep:model-training-api"
)

DATASET_ID = (
    "dataset:model-training-api"
)

MODEL_ID = (
    "model:model-training-api"
)

EXPERIMENT_ID = (
    "experiment:"
    +
    (
        "a"
        *
        32
    )
)

TRAINING_SHA = (
    "b"
    *
    64
)


# ============================================================
# TEST APP
# ============================================================


test_app = FastAPI()

test_app.include_router(
    router
)

client = TestClient(
    test_app
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
            model_training_api,
            name,
        )


    def __enter__(
        self,
    ):

        setattr(
            model_training_api,
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
            model_training_api,
            self.name,
            self.original,
        )


# ============================================================
# FIXTURES
# ============================================================


def context(
) -> ModelTrainingContextResponse:

    return (
        ModelTrainingContextResponse(
            workflow_id=
                WORKFLOW_ID,

            preparation_session_revision=
                12,

            dataset_count=
                1,

            datasets=[
                ModelTrainingDataset(
                    dataset_id=
                        DATASET_ID,

                    filename=
                        "training.csv",

                    row_count=
                        100,

                    column_count=
                        3,

                    columns=[
                        ModelTrainingColumn(
                            name=
                                "age",

                            kind=
                                "numeric",

                            nullable=
                                False,
                        ),

                        ModelTrainingColumn(
                            name=
                                "revenue",

                            kind=
                                "numeric",

                            nullable=
                                False,
                        ),

                        ModelTrainingColumn(
                            name=
                                "segment",

                            kind=
                                "categorical",

                            nullable=
                                False,
                        ),
                    ],
                )
            ],
        )
    )


def detail(
) -> ModelLabModelDetail:

    return (
        ModelLabModelDetail(
            model_id=
                MODEL_ID,

            workflow_id=
                WORKFLOW_ID,

            dataset_id=
                DATASET_ID,

            problem_type=
                "classification",

            target_column=
                "segment",

            estimator_key=
                "logistic_regression",

            feature_columns=[
                "age",
                "revenue",
            ],

            categorical_feature_columns=
                [],

            metrics={
                "accuracy":
                    0.80,

                "balanced_accuracy":
                    0.78,

                "f1_macro":
                    0.77,

                "precision_macro":
                    0.79,

                "recall_macro":
                    0.76,
            },

            train_rows=
                80,

            test_rows=
                20,

            created_at_utc=
                "2026-08-29T10:00:00+00:00",

            experiment_id=
                EXPERIMENT_ID,

            preparation_session_revision=
                12,

            training_contract_sha256=
                TRAINING_SHA,

            has_experiment_provenance=
                True,

            preprocessing=
                MLPreprocessingContract(),

            split=
                MLSplitContract(
                    stratify=
                        True
                ),

            effective_estimator_hyperparameters=
                MLLogisticRegressionHyperparameters(),
        )
    )


def valid_payload(
) -> dict:

    return {
        "training":
            {
                "workflow_id":
                    WORKFLOW_ID,

                "dataset_id":
                    DATASET_ID,

                "problem_type":
                    "classification",

                "target_column":
                    "segment",

                "feature_columns":
                    [
                        "age",
                        "revenue",
                    ],

                "categorical_feature_columns":
                    [],

                "estimator_key":
                    "logistic_regression",
            },

        "expected_preparation_session_revision":
            12,
    }


# ============================================================
# CONTEXT ROUTE
# ============================================================


def test_context_route(
) -> None:

    calls = []


    def fake_context(
        *,
        workflow_id: str,
    ):

        calls.append(
            workflow_id
        )

        return (
            context()
        )


    with Patch(
        "get_model_training_context",
        fake_context,
    ):

        response = client.get(
            "/model-training/context",
            params={
                "workflow_id":
                    WORKFLOW_ID
            },
        )


    assert (
        response.status_code
        ==
        200
    )

    assert (
        calls
        ==
        [
            WORKFLOW_ID
        ]
    )

    assert (
        response.json()
        ==
        context().model_dump(
            mode="json"
        )
    )


def test_context_requires_workflow_id(
) -> None:

    response = client.get(
        "/model-training/context"
    )

    assert (
        response.status_code
        ==
        422
    )


def test_context_error_maps_to_409(
) -> None:

    def fail(
        **kwargs,
    ):

        raise (
            ModelTrainingContextError(
                "Training context unavailable."
            )
        )


    with Patch(
        "get_model_training_context",
        fail,
    ):

        response = client.get(
            "/model-training/context",
            params={
                "workflow_id":
                    WORKFLOW_ID
            },
        )


    assert (
        response.status_code
        ==
        409
    )

    payload = (
        response.json()[
            "detail"
        ]
    )

    assert (
        payload["error"]
        ==
        "training_context_unavailable"
    )

    assert (
        payload["workflow_id"]
        ==
        WORKFLOW_ID
    )

    assert (
        payload["api_version"]
        ==
        MODEL_TRAINING_API_VERSION
    )


# ============================================================
# TRAIN ROUTE
# ============================================================


def test_train_route(
) -> None:

    calls = []


    def fake_train(
        request,
    ):

        calls.append(
            request
        )

        return (
            detail()
        )


    with Patch(
        "train_model",
        fake_train,
    ):

        response = client.post(
            "/model-training/train",
            json=
                valid_payload(),
        )


    assert (
        response.status_code
        ==
        200
    )

    assert (
        len(
            calls
        )
        ==
        1
    )

    request = (
        calls[
            0
        ]
    )

    assert (
        request.training.workflow_id
        ==
        WORKFLOW_ID
    )

    assert (
        request.training.dataset_id
        ==
        DATASET_ID
    )

    assert (
        request.expected_preparation_session_revision
        ==
        12
    )

    assert (
        response.json()
        ==
        detail().model_dump(
            mode="json"
        )
    )


def test_train_rejects_invalid_public_payload(
) -> None:

    payload = (
        valid_payload()
    )

    payload[
        "caller_dataframe"
    ] = "unsafe"


    response = client.post(
        "/model-training/train",
        json=
            payload,
    )


    assert (
        response.status_code
        ==
        422
    )


def test_training_input_error_maps_to_422(
) -> None:

    def fail(
        request,
    ):

        raise (
            ModelTrainingInputError(
                "Training input invalid."
            )
        )


    with Patch(
        "train_model",
        fail,
    ):

        response = client.post(
            "/model-training/train",
            json=
                valid_payload(),
        )


    assert (
        response.status_code
        ==
        422
    )

    payload = (
        response.json()[
            "detail"
        ]
    )

    assert (
        payload["error"]
        ==
        "training_input_invalid"
    )


def test_training_estimator_error_maps_to_422(
) -> None:

    def fail(
        request,
    ):

        raise (
            ModelTrainingEstimatorError(
                "Estimator invalid."
            )
        )


    with Patch(
        "train_model",
        fail,
    ):

        response = client.post(
            "/model-training/train",
            json=
                valid_payload(),
        )


    assert (
        response.status_code
        ==
        422
    )

    assert (
        response.json()[
            "detail"
        ][
            "error"
        ]
        ==
        "training_estimator_invalid"
    )


def test_training_execution_error_maps_to_409(
) -> None:

    def fail(
        request,
    ):

        raise (
            ModelTrainingExecutionError(
                "Training failed."
            )
        )


    with Patch(
        "train_model",
        fail,
    ):

        response = client.post(
            "/model-training/train",
            json=
                valid_payload(),
        )


    assert (
        response.status_code
        ==
        409
    )

    assert (
        response.json()[
            "detail"
        ][
            "error"
        ]
        ==
        "training_execution_failed"
    )


def test_generic_service_error_maps_to_400(
) -> None:

    def fail(
        request,
    ):

        raise (
            ModelTrainingServiceError(
                "Invalid request."
            )
        )


    with Patch(
        "train_model",
        fail,
    ):

        response = client.post(
            "/model-training/train",
            json=
                valid_payload(),
        )


    assert (
        response.status_code
        ==
        400
    )

    assert (
        response.json()[
            "detail"
        ][
            "error"
        ]
        ==
        "invalid_model_training_request"
    )


# ============================================================
# PUBLIC ERROR SHAPE
# ============================================================


def test_public_error_is_sanitized(
) -> None:

    internal_secret = (
        r"C:\private\models\secret.joblib"
    )


    def fail(
        request,
    ):

        raise (
            ModelTrainingExecutionError(
                "Training failed safely."
            )
        )


    with Patch(
        "train_model",
        fail,
    ):

        response = client.post(
            "/model-training/train",
            json=
                valid_payload(),
        )


    serialized = str(
        response.json()
    )


    assert (
        internal_secret
        not in serialized
    )

    assert (
        "training_contract"
        not in serialized
    )

    assert (
        "model_path"
        not in serialized
    )


# ============================================================
# ROUTE REGISTRATION
# ============================================================


def test_model_training_routes_are_exact(
) -> None:

    paths = set(
        test_app.openapi()[
            "paths"
        ]
    )

    expected = {
        "/model-training/context",
        "/model-training/train",
    }

    assert (
        paths
        ==
        expected
    )


def test_rule_version(
) -> None:

    assert (
        MODEL_TRAINING_API_VERSION
        ==
        "model_training_api_v0.1"
    )


# ============================================================
# MAIN
# ============================================================


def main(
) -> None:

    print(
        "=== DATALENS MODEL TRAINING API v0.1 ==="
    )


    tests = [
        (
            "GET training context",
            test_context_route,
        ),

        (
            "GET context requires workflow id",
            test_context_requires_workflow_id,
        ),

        (
            "Context error maps to HTTP 409",
            test_context_error_maps_to_409,
        ),

        (
            "POST real training contract boundary",
            test_train_route,
        ),

        (
            "Public payload extra authority blocked",
            test_train_rejects_invalid_public_payload,
        ),

        (
            "Training input error maps to HTTP 422",
            test_training_input_error_maps_to_422,
        ),

        (
            "Estimator error maps to HTTP 422",
            test_training_estimator_error_maps_to_422,
        ),

        (
            "Execution error maps to HTTP 409",
            test_training_execution_error_maps_to_409,
        ),

        (
            "Generic service error maps to HTTP 400",
            test_generic_service_error_maps_to_400,
        ),

        (
            "Public errors remain sanitized",
            test_public_error_is_sanitized,
        ),

        (
            "Model Training exposes exactly two routes",
            test_model_training_routes_are_exact,
        ),

        (
            "Model Training API rule version",
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
        "PASS - Model Training API v0.1"
    )


if __name__ == "__main__":
    main()
