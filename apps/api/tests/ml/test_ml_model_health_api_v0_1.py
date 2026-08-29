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


import app.api.ml_model_health as api_module


from app.api.ml_model_health import (
    ML_MODEL_HEALTH_API_VERSION,
    router,
)


from app.api.ml_model_health_contracts import (
    ML_MODEL_HEALTH_API_CONTRACT_RULE_VERSION,
)


from app.ml.model_health_service import (
    MLModelHealthServiceAuthorityError,
    MLModelHealthServiceInputError,
    MLModelHealthServiceNotFoundError,
    MLModelHealthServiceStorageError,
)


from tests.ml.test_ml_model_health_v0_1 import (
    MODEL_ID,
    WORKFLOW_ID,
    drift_record,
    performance_record,
    summary,
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
        .get_ml_model_health_summary
    )


    api_module.get_ml_model_health_summary = (
        replacement
    )


    try:
        yield

    finally:
        api_module.get_ml_model_health_summary = (
            original
        )


# ============================================================
# SUCCESS
# ============================================================


def test_get_health_delegates_identifiers_only(
) -> None:

    client = (
        build_client()
    )


    expected = (
        summary(
            drift=
                drift_record(
                    status="ok"
                ),

            performance=
                performance_record(
                    status="ok"
                ),
        )
    )


    captured = {}


    def fake_service(
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


        return expected


    with patched_service(
        fake_service
    ):

        response = (
            client.get(
                (
                    "/ml-monitoring/"
                    f"models/{MODEL_ID}/health"
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
            "health_status"
        ]
        ==
        "healthy"
    )


    assert (
        payload[
            "health_reason"
        ]
        ==
        "aligned_evidence_ok"
    )


    assert (
        payload[
            "evidence_alignment"
        ]
        ==
        "aligned"
    )


    assert (
        payload[
            "joint_interpretation_allowed"
        ]
        is True
    )


# ============================================================
# NO EVIDENCE
# ============================================================


def test_no_monitoring_evidence_returns_200_insufficient(
) -> None:

    client = (
        build_client()
    )


    expected = (
        summary()
    )


    def fake_service(
        **kwargs,
    ):

        return expected


    with patched_service(
        fake_service
    ):

        response = (
            client.get(
                (
                    "/ml-monitoring/"
                    f"models/{MODEL_ID}/health"
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


    payload = (
        response.json()
    )


    assert (
        payload[
            "health_status"
        ]
        ==
        "insufficient_evidence"
    )


    assert (
        payload[
            "health_reason"
        ]
        ==
        "no_monitoring_evidence"
    )


# ============================================================
# REQUIRED WORKFLOW
# ============================================================


def test_workflow_id_is_required_before_service(
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
            summary()
        )


    with patched_service(
        fake_service
    ):

        response = (
            client.get(
                (
                    "/ml-monitoring/"
                    f"models/{MODEL_ID}/health"
                )
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
# SERVICE ERROR HELPER
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
            client.get(
                (
                    "/ml-monitoring/"
                    f"models/{MODEL_ID}/health"
                ),

                params={
                    "workflow_id":
                        WORKFLOW_ID
                },
            )
        )


# ============================================================
# INPUT ERROR
# ============================================================


def test_input_error_maps_to_422(
) -> None:

    response = (
        response_for_error(
            MLModelHealthServiceInputError(
                "secret invalid identity detail"
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
        "model_health_input_invalid"
    )


# ============================================================
# NON-ENUMERATING NOT FOUND
# ============================================================


def test_not_found_is_generic_404(
) -> None:

    response = (
        response_for_error(
            MLModelHealthServiceNotFoundError(
                (
                    "model exists in another "
                    "workflow at private location"
                )
            )
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
        "model_health_not_found"
    )


    assert (
        "another workflow"
        not in
        detail[
            "message"
        ]
    )


    assert (
        "private"
        not in
        detail[
            "message"
        ]
    )


# ============================================================
# EVIDENCE CONFLICT
# ============================================================


def test_evidence_conflict_is_generic_409(
) -> None:

    response = (
        response_for_error(
            MLModelHealthServiceAuthorityError(
                (
                    "training fingerprint mismatch "
                    "for secret experiment"
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
        "model_health_evidence_conflict"
    )


    assert (
        "fingerprint"
        not in
        detail[
            "message"
        ]
    )


    assert (
        "experiment"
        not in
        detail[
            "message"
        ]
    )


# ============================================================
# STORAGE
# ============================================================


def test_storage_failure_is_generic_500(
) -> None:

    response = (
        response_for_error(
            MLModelHealthServiceStorageError(
                (
                    "sqlite failure at "
                    "/private/datalens.sqlite3"
                )
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
        "model_health_unavailable"
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
# PRIVACY SURFACE
# ============================================================


def test_health_response_remains_aggregate_only(
) -> None:

    client = (
        build_client()
    )


    expected = (
        summary(
            drift=
                drift_record(),

            performance=
                performance_record(),
        )
    )


    def fake_service(
        **kwargs,
    ):

        return expected


    with patched_service(
        fake_service
    ):

        response = (
            client.get(
                (
                    "/ml-monitoring/"
                    f"models/{MODEL_ID}/health"
                ),

                params={
                    "workflow_id":
                        WORKFLOW_ID
                },
            )
        )


    serialized = str(
        response.json()
    )


    assert (
        "predictions"
        not in
        serialized
    )


    assert (
        "probabilities"
        not in
        serialized
    )


    assert (
        "raw_values"
        not in
        serialized
    )


    assert (
        "model_bytes"
        not in
        serialized
    )


# ============================================================
# MAIN APPLICATION
# ============================================================


def test_router_registered_in_main_application(
) -> None:

    from app.main import (
        app,
    )


    paths = (
        app.openapi()[
            "paths"
        ]
    )


    path = (
        "/ml-monitoring/"
        "models/{model_id}/health"
    )


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


def test_rule_versions(
) -> None:

    assert (
        ML_MODEL_HEALTH_API_VERSION
        ==
        "ml_model_health_api_v0.1"
    )


    assert (
        ML_MODEL_HEALTH_API_CONTRACT_RULE_VERSION
        ==
        "ml_model_health_api_contract_v0.1"
    )


# ============================================================
# MAIN
# ============================================================


def main(
) -> None:

    print(
        (
            "=== DATALENS ML MODEL HEALTH "
            "API v0.1 ==="
        )
    )

    print()


    tests = [
        (
            "GET health delegates identifiers only",
            test_get_health_delegates_identifiers_only,
        ),
        (
            "No monitoring evidence returns 200 insufficient",
            test_no_monitoring_evidence_returns_200_insufficient,
        ),
        (
            "workflow_id is required before service",
            test_workflow_id_is_required_before_service,
        ),
        (
            "Input error maps to 422",
            test_input_error_maps_to_422,
        ),
        (
            "Not-found remains generic 404",
            test_not_found_is_generic_404,
        ),
        (
            "Evidence conflict remains generic 409",
            test_evidence_conflict_is_generic_409,
        ),
        (
            "Storage failure remains generic 500",
            test_storage_failure_is_generic_500,
        ),
        (
            "Health response remains aggregate-only",
            test_health_response_remains_aggregate_only,
        ),
        (
            "Model Health router registered in main app",
            test_router_registered_in_main_application,
        ),
        (
            "Model Health API rule versions",
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
            "PASS - ML Model Health "
            "API v0.1"
        )
    )


if __name__ == "__main__":
    main()
