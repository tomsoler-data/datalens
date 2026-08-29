from __future__ import annotations

from fastapi import (
    FastAPI,
)

from fastapi.testclient import (
    TestClient,
)

import app.api.model_lab as model_lab_api

from app.api.model_lab import (
    MODEL_LAB_API_VERSION,
    router,
)

from app.api.model_lab_contracts import (
    ModelLabModelCard,
    ModelLabModelDetail,
    ModelLabModelListResponse,
    ModelLabPredictResponse,
)

from app.api.model_lab_service import (
    ModelLabArtifactError,
    ModelLabEvaluationError,
    ModelLabModelNotFoundError,
    ModelLabPredictionExecutionError,
    ModelLabPredictionInputError,
    ModelLabWorkflowMismatchError,
)

from app.ml.contracts import (
    MLPreprocessingContract,
    MLSplitContract,
)

from app.ml.estimator_contracts import (
    MLLogisticRegressionHyperparameters,
)

from tests.ml.test_ml_model_evaluation_summary_contract_v0_1 import (
    valid_classification_summary,
)


# ============================================================
# CONSTANTS
# ============================================================


WORKFLOW_ID = "prep:model-lab-api"
MODEL_ID = "model:model-lab-api"
DATASET_ID = "dataset:model-lab-api"

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
# FIXTURES
# ============================================================


def card(
) -> ModelLabModelCard:

    return (
        ModelLabModelCard(
            model_id=
                MODEL_ID,

            workflow_id=
                WORKFLOW_ID,

            dataset_id=
                DATASET_ID,

            problem_type=
                "classification",

            target_column=
                "churned",

            estimator_key=
                "logistic_regression",

            feature_columns=[
                "age",
                "segment",
            ],

            categorical_feature_columns=[
                "segment",
            ],

            metrics={
                "accuracy":
                    0.80,

                "f1_macro":
                    0.78,

                "precision_macro":
                    0.79,

                "recall_macro":
                    0.77,

                "balanced_accuracy":
                    0.77,
            },

            train_rows=
                80,

            test_rows=
                20,

            created_at_utc=(
                "2026-08-29T10:00:00+00:00"
            ),

            experiment_id=
                EXPERIMENT_ID,

            preparation_session_revision=
                9,

            training_contract_sha256=
                TRAINING_SHA,

            has_experiment_provenance=
                True,
        )
    )


def detail(
) -> ModelLabModelDetail:

    return (
        ModelLabModelDetail(
            **card().model_dump(
                mode="python"
            ),

            preprocessing=(
                MLPreprocessingContract()
            ),

            split=(
                MLSplitContract(
                    stratify=
                        True
                )
            ),

            effective_estimator_hyperparameters=(
                MLLogisticRegressionHyperparameters()
            ),
        )
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
            model_lab_api,
            name,
        )


    def __enter__(
        self,
    ):

        setattr(
            model_lab_api,
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
            model_lab_api,
            self.name,
            self.original,
        )


# ============================================================
# LIST
# ============================================================


def test_get_model_list_route(
) -> None:

    expected = (
        ModelLabModelListResponse(
            workflow_id=
                WORKFLOW_ID,

            model_count=
                1,

            models=[
                card()
            ],
        )
    )

    calls = []

    def fake_list(
        *,
        workflow_id: str,
    ):

        calls.append(
            workflow_id
        )

        return expected

    with Patch(
        "list_model_lab_models",
        fake_list,
    ):

        response = client.get(
            "/model-lab/models",
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
        expected.model_dump(
            mode="json"
        )
    )


def test_list_requires_workflow_id(
) -> None:

    response = client.get(
        "/model-lab/models"
    )

    assert (
        response.status_code
        ==
        422
    )


# ============================================================
# DETAIL
# ============================================================


def test_get_model_detail_route(
) -> None:

    expected = detail()

    calls = []

    def fake_detail(
        *,
        workflow_id: str,
        model_id: str,
    ):

        calls.append(
            (
                workflow_id,
                model_id,
            )
        )

        return expected

    with Patch(
        "get_model_lab_model_detail",
        fake_detail,
    ):

        response = client.get(
            (
                "/model-lab/models/"
                f"{MODEL_ID}"
            ),
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
            (
                WORKFLOW_ID,
                MODEL_ID,
            )
        ]
    )

    assert (
        response.json()
        ==
        expected.model_dump(
            mode="json"
        )
    )


# ============================================================
# EVALUATE
# ============================================================


def test_evaluate_route(
) -> None:

    expected = (
        valid_classification_summary(
            with_threshold=
                False,

            selection_source=
                "standalone_model",
        )
    )

    calls = []

    def fake_evaluate(
        request,
    ):

        calls.append(
            request
        )

        return expected

    with Patch(
        "evaluate_model_lab_model",
        fake_evaluate,
    ):

        response = client.post(
            "/model-lab/evaluate",
            json={
                "workflow_id":
                    WORKFLOW_ID,

                "model_id":
                    MODEL_ID,
            },
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

    assert (
        calls[
            0
        ].workflow_id
        ==
        WORKFLOW_ID
    )

    assert (
        response.json()
        ==
        expected.model_dump(
            mode="json"
        )
    )


def test_evaluate_rejects_selection_context(
) -> None:

    response = client.post(
        "/model-lab/evaluate",
        json={
            "workflow_id":
                WORKFLOW_ID,

            "model_id":
                MODEL_ID,

            "selection_context":
                {
                    "source":
                        "model_comparison"
                },
        },
    )

    assert (
        response.status_code
        ==
        422
    )


# ============================================================
# PREDICT
# ============================================================


def test_predict_route(
) -> None:

    expected = (
        ModelLabPredictResponse(
            workflow_id=
                WORKFLOW_ID,

            model_id=
                MODEL_ID,

            problem_type=
                "classification",

            target_column=
                "churned",

            prediction_count=
                2,

            predictions=[
                "yes",
                "no",
            ],
        )
    )

    calls = []

    def fake_predict(
        request,
    ):

        calls.append(
            request
        )

        return expected

    with Patch(
        "predict_model_lab",
        fake_predict,
    ):

        response = client.post(
            "/model-lab/predict",
            json={
                "workflow_id":
                    WORKFLOW_ID,

                "model_id":
                    MODEL_ID,

                "rows":
                    [
                        {
                            "age":
                                42,

                            "segment":
                                "premium",
                        },
                        {
                            "age":
                                31,

                            "segment":
                                "standard",
                        },
                    ],
            },
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

    assert (
        response.json()
        ==
        expected.model_dump(
            mode="json"
        )
    )


# ============================================================
# ERROR SURFACES
# ============================================================


def test_missing_model_is_404(
) -> None:

    def fail(
        **kwargs,
    ):
        raise (
            ModelLabModelNotFoundError(
                "missing"
            )
        )

    with Patch(
        "get_model_lab_model_detail",
        fail,
    ):

        response = client.get(
            (
                "/model-lab/models/"
                f"{MODEL_ID}"
            ),
            params={
                "workflow_id":
                    WORKFLOW_ID
            },
        )

    assert (
        response.status_code
        ==
        404
    )

    detail_payload = (
        response.json()[
            "detail"
        ]
    )

    assert (
        detail_payload[
            "error"
        ]
        ==
        "model_not_found"
    )

    assert (
        detail_payload[
            "api_version"
        ]
        ==
        "model_lab_api_v0.1"
    )


def test_cross_workflow_model_is_same_404(
) -> None:

    def fail(
        **kwargs,
    ):
        raise (
            ModelLabWorkflowMismatchError(
                "cross workflow"
            )
        )

    with Patch(
        "get_model_lab_model_detail",
        fail,
    ):

        response = client.get(
            (
                "/model-lab/models/"
                f"{MODEL_ID}"
            ),
            params={
                "workflow_id":
                    WORKFLOW_ID
            },
        )

    assert (
        response.status_code
        ==
        404
    )

    assert (
        response.json()[
            "detail"
        ][
            "error"
        ]
        ==
        "model_not_found"
    )


def test_prediction_input_error_is_422(
) -> None:

    def fail(
        request,
    ):
        raise (
            ModelLabPredictionInputError(
                "missing=['age'], extra=[]"
            )
        )

    with Patch(
        "predict_model_lab",
        fail,
    ):

        response = client.post(
            "/model-lab/predict",
            json={
                "workflow_id":
                    WORKFLOW_ID,

                "model_id":
                    MODEL_ID,

                "rows":
                    [
                        {
                            "segment":
                                "premium"
                        }
                    ],
            },
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
        "prediction_input_invalid"
    )


def test_prediction_execution_error_is_422(
) -> None:

    def fail(
        request,
    ):
        raise (
            ModelLabPredictionExecutionError(
                "predict failed"
            )
        )

    with Patch(
        "predict_model_lab",
        fail,
    ):

        response = client.post(
            "/model-lab/predict",
            json={
                "workflow_id":
                    WORKFLOW_ID,

                "model_id":
                    MODEL_ID,

                "rows":
                    [
                        {
                            "age":
                                42,

                            "segment":
                                "premium",
                        }
                    ],
            },
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
        "prediction_execution_failed"
    )


def test_evaluation_error_is_409(
) -> None:

    def fail(
        request,
    ):
        raise (
            ModelLabEvaluationError(
                "stale evidence"
            )
        )

    with Patch(
        "evaluate_model_lab_model",
        fail,
    ):

        response = client.post(
            "/model-lab/evaluate",
            json={
                "workflow_id":
                    WORKFLOW_ID,

                "model_id":
                    MODEL_ID,
            },
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
        "model_evaluation_failed"
    )


def test_artifact_error_is_sanitized_500(
) -> None:

    secret = (
        "C:/secret/models/private.joblib"
    )

    def fail(
        **kwargs,
    ):
        raise (
            ModelLabArtifactError(
                secret
            )
        )

    with Patch(
        "get_model_lab_model_detail",
        fail,
    ):

        response = client.get(
            (
                "/model-lab/models/"
                f"{MODEL_ID}"
            ),
            params={
                "workflow_id":
                    WORKFLOW_ID
            },
        )

    assert (
        response.status_code
        ==
        500
    )

    payload = (
        response.json()
    )

    assert (
        payload[
            "detail"
        ][
            "error"
        ]
        ==
        "model_artifact_unavailable"
    )

    assert (
        secret
        not in
        str(
            payload
        )
    )


# ============================================================
# RESPONSE PRIVACY
# ============================================================


def _all_json_keys(
    value,
) -> set[
    str
]:

    keys = set()

    if isinstance(
        value,
        dict,
    ):

        for (
            key,
            nested,
        ) in value.items():

            keys.add(
                str(
                    key
                )
            )

            keys.update(
                _all_json_keys(
                    nested
                )
            )

    elif isinstance(
        value,
        list,
    ):

        for nested in value:

            keys.update(
                _all_json_keys(
                    nested
                )
            )

    return keys


def test_model_detail_response_is_privacy_minimal(
) -> None:

    expected = detail()

    with Patch(
        "get_model_lab_model_detail",
        lambda **kwargs:
            expected,
    ):

        response = client.get(
            (
                "/model-lab/models/"
                f"{MODEL_ID}"
            ),
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

    payload = (
        response.json()
    )

    forbidden_keys = {
        "model_path",
        "model_file_bytes",
        "model_sha256",
        "model_bytes",
        "training_contract",
        "estimator",
    }

    assert (
        forbidden_keys
        .isdisjoint(
            _all_json_keys(
                payload
            )
        )
    )

    # The public provenance fingerprint is intentionally safe
    # and must not be confused with the complete Training
    # Contract itself.
    assert (
        "training_contract_sha256"
        in
        _all_json_keys(
            payload
        )
    )


# ============================================================
# ROUTE REGISTRATION
# ============================================================


def test_model_lab_routes_are_registered_in_main(
) -> None:

    from app.main import (
        app,
    )

    schema = (
        app.openapi()
    )

    paths = (
        schema[
            "paths"
        ]
    )

    expected_operations = {
        "/model-lab/models":
            "get",

        "/model-lab/models/{model_id}":
            "get",

        "/model-lab/evaluate":
            "post",

        "/model-lab/predict":
            "post",
    }

    for (
        path,
        method,
    ) in expected_operations.items():

        assert (
            path
            in
            paths
        )

        assert (
            method
            in
            paths[
                path
            ]
        )


# ============================================================
# OPENAPI SURFACE
# ============================================================


def test_openapi_contains_only_expected_model_lab_paths(
) -> None:

    from app.main import (
        app,
    )

    schema = (
        app.openapi()
    )

    paths = {
        path
        for path
        in schema[
            "paths"
        ]
        if path.startswith(
            "/model-lab"
        )
    }

    assert (
        paths
        ==
        {
            "/model-lab/models",
            "/model-lab/models/{model_id}",
            "/model-lab/evaluate",
            "/model-lab/predict",
        }
    )


# ============================================================
# VERSION
# ============================================================


def test_api_version(
) -> None:

    assert (
        MODEL_LAB_API_VERSION
        ==
        "model_lab_api_v0.1"
    )


# ============================================================
# MAIN
# ============================================================


def main(
) -> None:

    print(
        "=== DATALENS MODEL LAB API v0.1 ==="
    )

    tests = [
        (
            "GET Model Lab model list",
            test_get_model_list_route,
        ),
        (
            "Model list requires workflow_id",
            test_list_requires_workflow_id,
        ),
        (
            "GET Model Lab model detail",
            test_get_model_detail_route,
        ),
        (
            "POST Model Lab evaluate",
            test_evaluate_route,
        ),
        (
            "Evaluate rejects public selection context",
            test_evaluate_rejects_selection_context,
        ),
        (
            "POST Model Lab predict",
            test_predict_route,
        ),
        (
            "Missing model maps to 404",
            test_missing_model_is_404,
        ),
        (
            "Cross-workflow model uses same 404",
            test_cross_workflow_model_is_same_404,
        ),
        (
            "Prediction input error maps to 422",
            test_prediction_input_error_is_422,
        ),
        (
            "Prediction execution error maps to 422",
            test_prediction_execution_error_is_422,
        ),
        (
            "Evaluation conflict maps to 409",
            test_evaluation_error_is_409,
        ),
        (
            "Artifact error is sanitized",
            test_artifact_error_is_sanitized_500,
        ),
        (
            "Model detail HTTP response privacy",
            test_model_detail_response_is_privacy_minimal,
        ),
        (
            "Model Lab routes registered in main",
            test_model_lab_routes_are_registered_in_main,
        ),
        (
            "OpenAPI Model Lab path surface",
            test_openapi_contains_only_expected_model_lab_paths,
        ),
        (
            "Model Lab API version",
            test_api_version,
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
        "PASS - Model Lab API v0.1"
    )


if __name__ == "__main__":
    main()
