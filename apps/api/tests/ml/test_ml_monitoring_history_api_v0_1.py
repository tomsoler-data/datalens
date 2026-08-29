from __future__ import annotations


from contextlib import (
    contextmanager,
)


from fastapi.testclient import (
    TestClient,
)


import app.api.ml_monitoring as api_module


from app.api.ml_monitoring import (
    ML_MONITORING_HISTORY_API_VERSION,
)


from app.api.ml_monitoring_contracts import (
    MLMonitoringModelHistoryResponse,
    MLMonitoringWorkflowHistoryResponse,
)


from app.main import (
    app,
)


from app.ml.monitoring_history_service import (
    MLMonitoringHistoryInputError,
    MLMonitoringHistoryNotFoundError,
    MLMonitoringHistoryStorageError,
)


from tests.ml.test_ml_drift_evaluation_store_v0_1 import (
    evaluation,
)


from tests.ml.test_ml_monitoring_profile_store_v0_1 import (
    isolated_environment,
    persisted_artifact_and_profile,
)


# ============================================================
# FIXTURE
# ============================================================


def valid_evaluation(
):

    with isolated_environment():

        (
            artifact,
            profile,
        ) = (
            persisted_artifact_and_profile()
        )


        return (
            evaluation(
                artifact=
                    artifact,

                profile=
                    profile,

                observed_dataset_id=
                    "dataset:observed",
            )
        )


# ============================================================
# PATCH
# ============================================================


@contextmanager
def patched_history(
    *,
    detail=None,
    model_history=None,
    workflow_history=None,
):

    original_detail = (
        api_module
        .get_ml_monitoring_evaluation
    )

    original_model = (
        api_module
        .list_ml_monitoring_model_history
    )

    original_workflow = (
        api_module
        .list_ml_monitoring_workflow_history
    )


    if detail is not None:
        api_module.get_ml_monitoring_evaluation = (
            detail
        )


    if model_history is not None:
        api_module.list_ml_monitoring_model_history = (
            model_history
        )


    if workflow_history is not None:
        api_module.list_ml_monitoring_workflow_history = (
            workflow_history
        )


    try:
        yield

    finally:
        api_module.get_ml_monitoring_evaluation = (
            original_detail
        )

        api_module.list_ml_monitoring_model_history = (
            original_model
        )

        api_module.list_ml_monitoring_workflow_history = (
            original_workflow
        )


# ============================================================
# ROUTES
# ============================================================


def test_history_routes_registered(
) -> None:

    paths = (
        app.openapi()
        .get(
            "paths",
            {}
        )
    )


    assert (
        "/ml-monitoring/evaluations/{evaluation_id}"
        in
        paths
    )


    assert (
        "/ml-monitoring/models/{model_id}/history"
        in
        paths
    )


    assert (
        "/ml-monitoring/workflows/{workflow_id}/history"
        in
        paths
    )


    assert (
        "get"
        in
        paths[
            "/ml-monitoring/evaluations/{evaluation_id}"
        ]
    )


# ============================================================
# DETAIL SUCCESS
# ============================================================


def test_evaluation_detail_success(
) -> None:

    result = (
        valid_evaluation()
    )


    captured = {}


    def fake_detail(
        *,
        workflow_id: str,
        evaluation_id: str,
    ):

        captured[
            "workflow_id"
        ] = workflow_id

        captured[
            "evaluation_id"
        ] = evaluation_id


        return result


    with patched_history(
        detail=
            fake_detail
    ):

        with TestClient(
            app
        ) as client:

            response = (
                client.get(
                    (
                        "/ml-monitoring/evaluations/"
                        f"{result.evaluation_id}"
                    ),

                    params={
                        "workflow_id":
                            "prep:history-api",
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
            "prep:history-api",

        "evaluation_id":
            result.evaluation_id,
    }


    assert (
        response.json()
        ==
        result.model_dump(
            mode="json"
        )
    )


# ============================================================
# MODEL HISTORY SUCCESS
# ============================================================


def test_model_history_success(
) -> None:

    result = (
        valid_evaluation()
    )


    def fake_history(
        *,
        workflow_id: str,
        model_id: str,
    ):

        assert (
            workflow_id
            ==
            "prep:history-api"
        )

        assert (
            model_id
            ==
            "model:history"
        )


        return [
            result
        ]


    with patched_history(
        model_history=
            fake_history
    ):

        with TestClient(
            app
        ) as client:

            response = (
                client.get(
                    (
                        "/ml-monitoring/models/"
                        "model:history/history"
                    ),

                    params={
                        "workflow_id":
                            "prep:history-api",
                    },
                )
            )


    assert (
        response.status_code
        ==
        200
    )


    payload = (
        response.json()
    )


    assert (
        payload[
            "workflow_id"
        ]
        ==
        "prep:history-api"
    )


    assert (
        payload[
            "model_id"
        ]
        ==
        "model:history"
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
            "evaluations"
        ]
        ==
        [
            result.model_dump(
                mode="json"
            )
        ]
    )


    MLMonitoringModelHistoryResponse.model_validate(
        payload
    )


# ============================================================
# WORKFLOW HISTORY SUCCESS
# ============================================================


def test_workflow_history_success(
) -> None:

    result = (
        valid_evaluation()
    )


    def fake_history(
        *,
        workflow_id: str,
    ):

        assert (
            workflow_id
            ==
            "prep:history-api"
        )


        return [
            result
        ]


    with patched_history(
        workflow_history=
            fake_history
    ):

        with TestClient(
            app
        ) as client:

            response = (
                client.get(
                    (
                        "/ml-monitoring/workflows/"
                        "prep:history-api/history"
                    )
                )
            )


    assert (
        response.status_code
        ==
        200
    )


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
            "evaluations"
        ]
        ==
        [
            result.model_dump(
                mode="json"
            )
        ]
    )


    MLMonitoringWorkflowHistoryResponse.model_validate(
        payload
    )


# ============================================================
# EMPTY HISTORY
# ============================================================


def test_empty_history_is_200(
) -> None:

    def fake_history(
        *,
        workflow_id: str,
    ):

        return []


    with patched_history(
        workflow_history=
            fake_history
    ):

        with TestClient(
            app
        ) as client:

            response = (
                client.get(
                    (
                        "/ml-monitoring/workflows/"
                        "prep:empty/history"
                    )
                )
            )


    assert (
        response.status_code
        ==
        200
    )


    assert (
        response.json()[
            "evaluation_count"
        ]
        ==
        0
    )


    assert (
        response.json()[
            "evaluations"
        ]
        ==
        []
    )


# ============================================================
# NON-ENUMERATING NOT FOUND
# ============================================================


def test_not_found_is_generic_404(
) -> None:

    secret = (
        "resource-exists-in-another-workflow"
    )


    def failing_detail(
        **_,
    ):

        raise (
            MLMonitoringHistoryNotFoundError(
                secret
            )
        )


    with patched_history(
        detail=
            failing_detail
    ):

        with TestClient(
            app
        ) as client:

            response = (
                client.get(
                    (
                        "/ml-monitoring/evaluations/"
                        "drift-evaluation:missing"
                    ),

                    params={
                        "workflow_id":
                            "prep:other",
                    },
                )
            )


    assert (
        response.status_code
        ==
        404
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
        "monitoring_history_not_found"
    )


    assert (
        secret
        not in
        str(
            payload
        )
    )


# ============================================================
# INPUT
# ============================================================


def test_history_input_error_maps_to_422(
) -> None:

    def failing_history(
        **_,
    ):

        raise (
            MLMonitoringHistoryInputError(
                "invalid history identity"
            )
        )


    with patched_history(
        workflow_history=
            failing_history
    ):

        with TestClient(
            app
        ) as client:

            response = (
                client.get(
                    (
                        "/ml-monitoring/workflows/"
                        "prep:any/history"
                    )
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
        "monitoring_history_input_invalid"
    )


# ============================================================
# STORAGE FAILURE
# ============================================================


def test_storage_failure_is_generic_500(
) -> None:

    secret = (
        "sqlite-internal-details"
    )


    def failing_history(
        **_,
    ):

        raise (
            MLMonitoringHistoryStorageError(
                secret
            )
        )


    with patched_history(
        model_history=
            failing_history
    ):

        with TestClient(
            app,
            raise_server_exceptions=False,
        ) as client:

            response = (
                client.get(
                    (
                        "/ml-monitoring/models/"
                        "model:any/history"
                    ),

                    params={
                        "workflow_id":
                            "prep:any",
                    },
                )
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
        "monitoring_history_unavailable"
    )


    assert (
        secret
        not in
        str(
            payload
        )
    )


# ============================================================
# VERSION
# ============================================================


def test_history_api_version(
) -> None:

    assert (
        ML_MONITORING_HISTORY_API_VERSION
        ==
        "ml_monitoring_history_api_v0.1"
    )


# ============================================================
# MAIN
# ============================================================


def main(
) -> None:

    print(
        (
            "=== DATALENS ML MONITORING "
            "HISTORY API v0.1 ==="
        )
    )

    print()


    tests = [
        (
            "History routes registered",
            test_history_routes_registered,
        ),
        (
            "Evaluation detail GET",
            test_evaluation_detail_success,
        ),
        (
            "Model history GET",
            test_model_history_success,
        ),
        (
            "Workflow history GET",
            test_workflow_history_success,
        ),
        (
            "Empty history returns 200",
            test_empty_history_is_200,
        ),
        (
            "Non-enumerating history 404",
            test_not_found_is_generic_404,
        ),
        (
            "History input error -> 422",
            test_history_input_error_maps_to_422,
        ),
        (
            "History storage failure -> generic 500",
            test_storage_failure_is_generic_500,
        ),
        (
            "Monitoring History API version",
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
            "PASS - ML Monitoring "
            "History API v0.1"
        )
    )


if __name__ == "__main__":
    main()
