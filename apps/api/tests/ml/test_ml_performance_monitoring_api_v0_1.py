from __future__ import annotations


from contextlib import (
    contextmanager,
)


from fastapi import (
    FastAPI,
)


from fastapi.testclient import (
    TestClient,
)


from pydantic import (
    ValidationError,
)


import app.api.ml_performance_monitoring as api_module


from app.api.ml_performance_monitoring import (
    ML_PERFORMANCE_MONITORING_API_VERSION,
    router,
)


from app.api.ml_performance_monitoring_contracts import (
    ML_PERFORMANCE_MONITORING_API_CONTRACT_RULE_VERSION,
    MLPerformanceMonitoringRunRequest,
)


from app.ml.performance_monitoring_service import (
    MLPerformanceMonitoringObservedDatasetError,
    MLPerformanceMonitoringServiceAuthorityError,
    MLPerformanceMonitoringServiceExecutionError,
    MLPerformanceMonitoringServiceInputError,
    MLPerformanceMonitoringTargetError,
)


from tests.ml.test_ml_performance_evaluation_contract_v0_1 import (
    MODEL_ID,
    classification_record,
)


# ============================================================
# CONSTANTS
# ============================================================


WORKFLOW_ID = (
    "prep:performance"
)


OBSERVED_DATASET_ID = (
    "dataset:observed"
)


# ============================================================
# CLIENT
# ============================================================


def build_client(
) -> TestClient:

    app = FastAPI()


    app.include_router(
        router
    )


    return (
        TestClient(
            app
        )
    )


# ============================================================
# PATCH SERVICE
# ============================================================


@contextmanager
def patched_service(
    replacement,
):

    original = (
        api_module
        .run_ml_performance_monitoring
    )


    api_module.run_ml_performance_monitoring = (
        replacement
    )


    try:
        yield

    finally:
        api_module.run_ml_performance_monitoring = (
            original
        )


# ============================================================
# CONTRACT
# ============================================================


def test_request_contract_accepts_identifiers_only(
) -> None:

    request = (
        MLPerformanceMonitoringRunRequest(
            workflow_id=
                "  prep:performance  ",

            model_id=
                f"  {MODEL_ID}  ",

            observed_dataset_id=
                "  dataset:observed  ",
        )
    )


    assert request.model_dump() == {
        "workflow_id":
            WORKFLOW_ID,

        "model_id":
            MODEL_ID,

        "observed_dataset_id":
            OBSERVED_DATASET_ID,
    }


    assert (
        set(
            MLPerformanceMonitoringRunRequest
            .model_fields
        )
        ==
        {
            "workflow_id",
            "model_id",
            "observed_dataset_id",
        }
    )


    try:
        MLPerformanceMonitoringRunRequest.model_validate(
            {
                "workflow_id":
                    WORKFLOW_ID,

                "model_id":
                    MODEL_ID,

                "observed_dataset_id":
                    OBSERVED_DATASET_ID,

                "predictions": [
                    0,
                    1,
                ],
            }
        )

    except ValidationError:
        pass

    else:
        raise AssertionError(
            (
                "Public Performance Monitoring "
                "request must reject prediction payloads."
            )
        )


# ============================================================
# SUCCESS
# ============================================================


def test_post_evaluate_delegates_identifiers_only(
) -> None:

    client = (
        build_client()
    )


    expected = (
        classification_record()
    )


    captured = {}


    def fake_service(
        *,
        workflow_id: str,
        model_id: str,
        observed_dataset_id: str,
    ):

        captured[
            "workflow_id"
        ] = workflow_id

        captured[
            "model_id"
        ] = model_id

        captured[
            "observed_dataset_id"
        ] = observed_dataset_id


        return expected


    with patched_service(
        fake_service
    ):

        response = (
            client.post(
                "/ml-monitoring/performance/evaluate",

                json={
                    "workflow_id":
                        WORKFLOW_ID,

                    "model_id":
                        MODEL_ID,

                    "observed_dataset_id":
                        OBSERVED_DATASET_ID,
                },
            )
        )


    assert (
        response.status_code
        ==
        200
    )


    assert captured == {
        "workflow_id":
            WORKFLOW_ID,

        "model_id":
            MODEL_ID,

        "observed_dataset_id":
            OBSERVED_DATASET_ID,
    }


    payload = (
        response.json()
    )


    assert (
        payload[
            "performance_evaluation_id"
        ]
        ==
        expected.performance_evaluation_id
    )


    assert (
        payload[
            "model_id"
        ]
        ==
        MODEL_ID
    )


    assert (
        payload[
            "workflow_id"
        ]
        ==
        WORKFLOW_ID
    )


# ============================================================
# HTTP REQUEST SURFACE
# ============================================================


def test_raw_supervised_payload_rejected_before_service(
) -> None:

    client = (
        build_client()
    )


    calls = {
        "count":
            0
    }


    def fake_service(
        **kwargs,
    ):

        calls[
            "count"
        ] += 1

        return (
            classification_record()
        )


    with patched_service(
        fake_service
    ):

        response = (
            client.post(
                "/ml-monitoring/performance/evaluate",

                json={
                    "workflow_id":
                        WORKFLOW_ID,

                    "model_id":
                        MODEL_ID,

                    "observed_dataset_id":
                        OBSERVED_DATASET_ID,

                    "y_true": [
                        0,
                        1,
                    ],

                    "predictions": [
                        0,
                        1,
                    ],
                },
            )
        )


    assert (
        response.status_code
        ==
        422
    )


    assert (
        calls[
            "count"
        ]
        ==
        0
    )


# ============================================================
# ERROR TRANSLATION
# ============================================================


def response_for_error(
    error: Exception,
):

    client = (
        build_client()
    )


    def fake_service(
        **kwargs,
    ):

        raise error


    with patched_service(
        fake_service
    ):

        return (
            client.post(
                "/ml-monitoring/performance/evaluate",

                json={
                    "workflow_id":
                        WORKFLOW_ID,

                    "model_id":
                        MODEL_ID,

                    "observed_dataset_id":
                        OBSERVED_DATASET_ID,
                },
            )
        )


def test_input_error_translation(
) -> None:

    response = (
        response_for_error(
            MLPerformanceMonitoringServiceInputError(
                "invalid identity"
            )
        )
    )


    assert (
        response.status_code
        ==
        422
    )


    detail = (
        response.json()[
            "detail"
        ]
    )


    assert (
        detail[
            "error"
        ]
        ==
        "performance_monitoring_input_invalid"
    )


def test_observed_dataset_error_translation(
) -> None:

    response = (
        response_for_error(
            MLPerformanceMonitoringObservedDatasetError(
                "internal observed dataset detail"
            )
        )
    )


    assert (
        response.status_code
        ==
        422
    )


    detail = (
        response.json()[
            "detail"
        ]
    )


    assert (
        detail[
            "error"
        ]
        ==
        "performance_observed_dataset_invalid"
    )


    assert (
        "internal observed dataset detail"
        not in
        detail[
            "message"
        ]
    )


def test_target_error_translation(
) -> None:

    response = (
        response_for_error(
            MLPerformanceMonitoringTargetError(
                "secret target implementation detail"
            )
        )
    )


    assert (
        response.status_code
        ==
        422
    )


    detail = (
        response.json()[
            "detail"
        ]
    )


    assert (
        detail[
            "error"
        ]
        ==
        "performance_ground_truth_invalid"
    )


    assert (
        "secret target implementation detail"
        not in
        detail[
            "message"
        ]
    )


def test_authority_error_is_non_enumerating(
) -> None:

    response = (
        response_for_error(
            MLPerformanceMonitoringServiceAuthorityError(
                (
                    "model exists in another "
                    "workflow at /private/model.joblib"
                )
            )
        )
    )


    assert (
        response.status_code
        ==
        409
    )


    detail = (
        response.json()[
            "detail"
        ]
    )


    assert (
        detail[
            "error"
        ]
        ==
        "performance_monitoring_authority_unavailable"
    )


    assert (
        "another workflow"
        not in
        detail[
            "message"
        ]
    )


    assert (
        "model.joblib"
        not in
        detail[
            "message"
        ]
    )


def test_execution_error_hides_internal_detail(
) -> None:

    response = (
        response_for_error(
            MLPerformanceMonitoringServiceExecutionError(
                "sqlite transaction failed at private path"
            )
        )
    )


    assert (
        response.status_code
        ==
        409
    )


    detail = (
        response.json()[
            "detail"
        ]
    )


    assert (
        detail[
            "error"
        ]
        ==
        "performance_monitoring_execution_failed"
    )


    assert (
        "sqlite transaction"
        not in
        detail[
            "message"
        ]
    )


    assert (
        detail[
            "retryable"
        ]
        is False
    )


# ============================================================
# APPLICATION REGISTRATION
# ============================================================


def test_router_registered_in_main_application(
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


    assert (
        "/ml-monitoring/performance/evaluate"
        in
        paths
    )


    assert (
        "post"
        in
        paths[
            "/ml-monitoring/performance/evaluate"
        ]
    )


# ============================================================
# VERSIONS
# ============================================================


def test_rule_versions(
) -> None:

    assert (
        ML_PERFORMANCE_MONITORING_API_VERSION
        ==
        "ml_performance_monitoring_api_v0.1"
    )


    assert (
        ML_PERFORMANCE_MONITORING_API_CONTRACT_RULE_VERSION
        ==
        "ml_performance_monitoring_api_contract_v0.1"
    )


# ============================================================
# MAIN
# ============================================================


def main(
) -> None:

    print(
        (
            "=== DATALENS ML PERFORMANCE "
            "MONITORING API v0.1 ==="
        )
    )

    print()


    tests = [
        (
            "Request contract accepts identifiers only",
            test_request_contract_accepts_identifiers_only,
        ),
        (
            "POST evaluate delegates identifiers only",
            test_post_evaluate_delegates_identifiers_only,
        ),
        (
            "Raw supervised payload rejected before service",
            test_raw_supervised_payload_rejected_before_service,
        ),
        (
            "Input error translated",
            test_input_error_translation,
        ),
        (
            "Observed dataset error translated",
            test_observed_dataset_error_translation,
        ),
        (
            "Ground truth error translated",
            test_target_error_translation,
        ),
        (
            "Authority error remains non-enumerating",
            test_authority_error_is_non_enumerating,
        ),
        (
            "Execution error hides internal detail",
            test_execution_error_hides_internal_detail,
        ),
        (
            "Performance router registered in main app",
            test_router_registered_in_main_application,
        ),
        (
            "Performance Monitoring API rule versions",
            test_rule_versions,
        ),
    ]


    for (
        label,
        callback,
    ) in tests:

        callback()

        print(
            f"[PASS] {label}"
        )


    print()

    print(
        (
            "PASS - ML Performance "
            "Monitoring API v0.1"
        )
    )


if __name__ == "__main__":
    main()
