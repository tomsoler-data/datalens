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


import app.api.ml_performance_monitoring as api_module


from app.api.ml_performance_monitoring import (
    ML_PERFORMANCE_MONITORING_HISTORY_API_VERSION,
    router,
)


from app.api.ml_performance_monitoring_contracts import (
    MLPerformanceMonitoringModelHistoryResponse,
    MLPerformanceMonitoringWorkflowHistoryResponse,
)


from app.ml.performance_monitoring_history_service import (
    MLPerformanceMonitoringHistoryInputError,
    MLPerformanceMonitoringHistoryNotFoundError,
    MLPerformanceMonitoringHistoryStorageError,
)


from tests.ml.test_ml_performance_evaluation_contract_v0_1 import (
    MODEL_ID,
    PERFORMANCE_ID,
    classification_record,
)


# ============================================================
# CONSTANTS
# ============================================================


WORKFLOW_ID = (
    "prep:performance"
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
# PATCH
# ============================================================


@contextmanager
def patched_history_function(
    *,
    function_name: str,
    replacement,
):

    original = getattr(
        api_module,
        function_name,
    )


    setattr(
        api_module,
        function_name,
        replacement,
    )


    try:
        yield

    finally:
        setattr(
            api_module,
            function_name,
            original,
        )


# ============================================================
# RESPONSE CONTRACTS
# ============================================================


def test_history_response_contracts(
) -> None:

    record = (
        classification_record()
    )


    model_response = (
        MLPerformanceMonitoringModelHistoryResponse(
            workflow_id=
                WORKFLOW_ID,

            model_id=
                MODEL_ID,

            evaluation_count=
                1,

            evaluations=[
                record
            ],
        )
    )


    workflow_response = (
        MLPerformanceMonitoringWorkflowHistoryResponse(
            workflow_id=
                WORKFLOW_ID,

            evaluation_count=
                1,

            evaluations=[
                record
            ],
        )
    )


    assert (
        model_response.api_version
        ==
        "ml_performance_monitoring_history_api_v0.1"
    )


    assert (
        workflow_response.api_version
        ==
        "ml_performance_monitoring_history_api_v0.1"
    )


# ============================================================
# DETAIL
# ============================================================


def test_get_evaluation_detail(
) -> None:

    client = (
        build_client()
    )


    expected = (
        classification_record()
    )


    captured = {}


    def fake_detail(
        *,
        workflow_id: str,
        performance_evaluation_id: str,
    ):

        captured[
            "workflow_id"
        ] = workflow_id

        captured[
            "performance_evaluation_id"
        ] = performance_evaluation_id


        return expected


    with patched_history_function(
        function_name=
            "get_ml_performance_monitoring_evaluation",

        replacement=
            fake_detail,
    ):

        response = (
            client.get(
                (
                    "/ml-monitoring/performance/"
                    f"evaluations/{PERFORMANCE_ID}"
                ),

                params={
                    "workflow_id":
                        WORKFLOW_ID
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

        "performance_evaluation_id":
            PERFORMANCE_ID,
    }


    assert (
        response.json()[
            "performance_evaluation_id"
        ]
        ==
        PERFORMANCE_ID
    )


# ============================================================
# MODEL HISTORY
# ============================================================


def test_get_model_history(
) -> None:

    client = (
        build_client()
    )


    expected = (
        classification_record()
    )


    captured = {}


    def fake_model_history(
        *,
        workflow_id: str,
        model_id: str,
    ):

        captured[
            "workflow_id"
        ] = workflow_id

        captured[
            "model_id"
        ] = model_id


        return [
            expected
        ]


    with patched_history_function(
        function_name=
            "list_ml_performance_monitoring_model_history",

        replacement=
            fake_model_history,
    ):

        response = (
            client.get(
                (
                    "/ml-monitoring/performance/"
                    f"models/{MODEL_ID}/history"
                ),

                params={
                    "workflow_id":
                        WORKFLOW_ID
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
    }


    payload = (
        response.json()
    )


    assert (
        payload[
            "evaluation_count"
        ]
        ==
        1
    )


    assert (
        len(
            payload[
                "evaluations"
            ]
        )
        ==
        1
    )


    assert (
        payload[
            "api_version"
        ]
        ==
        "ml_performance_monitoring_history_api_v0.1"
    )


# ============================================================
# WORKFLOW HISTORY
# ============================================================


def test_get_workflow_history(
) -> None:

    client = (
        build_client()
    )


    expected = (
        classification_record()
    )


    captured = {}


    def fake_workflow_history(
        *,
        workflow_id: str,
    ):

        captured[
            "workflow_id"
        ] = workflow_id


        return [
            expected
        ]


    with patched_history_function(
        function_name=(
            "list_ml_performance_monitoring_workflow_history"
        ),

        replacement=
            fake_workflow_history,
    ):

        response = (
            client.get(
                (
                    "/ml-monitoring/performance/"
                    f"workflows/{WORKFLOW_ID}/history"
                )
            )
        )


    assert (
        response.status_code
        ==
        200
    )


    assert captured == {
        "workflow_id":
            WORKFLOW_ID
    }


    payload = (
        response.json()
    )


    assert (
        payload[
            "evaluation_count"
        ]
        ==
        1
    )


    assert (
        payload[
            "workflow_id"
        ]
        ==
        WORKFLOW_ID
    )


# ============================================================
# NOT FOUND ? NON ENUMERATING
# ============================================================


def test_history_not_found_is_generic_404(
) -> None:

    client = (
        build_client()
    )


    def fake_detail(
        **kwargs,
    ):

        raise (
            MLPerformanceMonitoringHistoryNotFoundError(
                (
                    "secret: evaluation exists "
                    "in another workflow"
                )
            )
        )


    with patched_history_function(
        function_name=
            "get_ml_performance_monitoring_evaluation",

        replacement=
            fake_detail,
    ):

        response = (
            client.get(
                (
                    "/ml-monitoring/performance/"
                    f"evaluations/{PERFORMANCE_ID}"
                ),

                params={
                    "workflow_id":
                        WORKFLOW_ID
                },
            )
        )


    assert (
        response.status_code
        ==
        404
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
        "performance_monitoring_history_not_found"
    )


    assert (
        "another workflow"
        not in
        detail[
            "message"
        ]
    )


# ============================================================
# STORAGE FAILURE
# ============================================================


def test_history_storage_failure_is_generic_500(
) -> None:

    client = (
        build_client()
    )


    def fake_history(
        **kwargs,
    ):

        raise (
            MLPerformanceMonitoringHistoryStorageError(
                (
                    "sqlite corruption at "
                    "/private/datalens.sqlite3"
                )
            )
        )


    with patched_history_function(
        function_name=(
            "list_ml_performance_monitoring_workflow_history"
        ),

        replacement=
            fake_history,
    ):

        response = (
            client.get(
                (
                    "/ml-monitoring/performance/"
                    f"workflows/{WORKFLOW_ID}/history"
                )
            )
        )


    assert (
        response.status_code
        ==
        500
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
        "performance_monitoring_history_unavailable"
    )


    assert (
        "sqlite"
        not in
        detail[
            "message"
        ]
        .lower()
    )


    assert (
        "private"
        not in
        detail[
            "message"
        ]
        .lower()
    )


    assert (
        detail[
            "retryable"
        ]
        is False
    )


# ============================================================
# INPUT FAILURE
# ============================================================


def test_history_input_error_is_422(
) -> None:

    client = (
        build_client()
    )


    def fake_model_history(
        **kwargs,
    ):

        raise (
            MLPerformanceMonitoringHistoryInputError(
                "model_id cannot be empty."
            )
        )


    with patched_history_function(
        function_name=
            "list_ml_performance_monitoring_model_history",

        replacement=
            fake_model_history,
    ):

        response = (
            client.get(
                (
                    "/ml-monitoring/performance/"
                    f"models/{MODEL_ID}/history"
                ),

                params={
                    "workflow_id":
                        WORKFLOW_ID
                },
            )
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
        "performance_monitoring_history_input_invalid"
    )


# ============================================================
# MAIN APPLICATION ROUTES
# ============================================================


def test_history_routes_registered_in_main_application(
) -> None:

    from app.main import (
        app,
    )


    paths = (
        app.openapi()[
            "paths"
        ]
    )


    expected_paths = [
        (
            "/ml-monitoring/performance/"
            "evaluations/{performance_evaluation_id}"
        ),
        (
            "/ml-monitoring/performance/"
            "models/{model_id}/history"
        ),
        (
            "/ml-monitoring/performance/"
            "workflows/{workflow_id}/history"
        ),
    ]


    for path in (
        expected_paths
    ):

        assert (
            path
            in
            paths
        )


        assert (
            "get"
            in
            paths[
                path
            ]
        )


# ============================================================
# VERSION
# ============================================================


def test_history_api_version(
) -> None:

    assert (
        ML_PERFORMANCE_MONITORING_HISTORY_API_VERSION
        ==
        "ml_performance_monitoring_history_api_v0.1"
    )


# ============================================================
# MAIN
# ============================================================


def main(
) -> None:

    print(
        (
            "=== DATALENS ML PERFORMANCE MONITORING "
            "HISTORY API v0.1 ==="
        )
    )

    print()


    tests = [
        (
            "History response contracts",
            test_history_response_contracts,
        ),
        (
            "GET evaluation detail",
            test_get_evaluation_detail,
        ),
        (
            "GET model history",
            test_get_model_history,
        ),
        (
            "GET workflow history",
            test_get_workflow_history,
        ),
        (
            "History not-found remains generic 404",
            test_history_not_found_is_generic_404,
        ),
        (
            "History storage failure remains generic 500",
            test_history_storage_failure_is_generic_500,
        ),
        (
            "History input error maps to 422",
            test_history_input_error_is_422,
        ),
        (
            "History routes registered in main app",
            test_history_routes_registered_in_main_application,
        ),
        (
            "Performance History API version",
            test_history_api_version,
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
            "PASS - ML Performance Monitoring "
            "History API v0.1"
        )
    )


if __name__ == "__main__":
    main()
