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


import app.api.ml_monitoring_alert as api_module


from app.api.ml_monitoring_alert import (
    ML_MONITORING_ALERT_API_VERSION,
    router,
)


from app.api.ml_monitoring_alert_contracts import (
    ML_MONITORING_ALERT_API_CONTRACT_RULE_VERSION,
)


from app.ml.monitoring_alert import (
    build_ml_monitoring_alert_decision,
)


from app.ml.monitoring_alert_service import (
    MLMonitoringAlertServiceAuthorityError,
    MLMonitoringAlertServiceInputError,
    MLMonitoringAlertServiceNotFoundError,
    MLMonitoringAlertServiceStorageError,
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
        .get_ml_monitoring_alert_decision
    )


    api_module.get_ml_monitoring_alert_decision = (
        replacement
    )


    try:
        yield

    finally:
        api_module.get_ml_monitoring_alert_decision = (
            original
        )


# ============================================================
# SUCCESS
# ============================================================


def test_get_alert_delegates_identifiers_only(
) -> None:

    client = (
        build_client()
    )


    health = (
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


    expected = (
        build_ml_monitoring_alert_decision(
            model_health=
                health
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
                    f"models/{MODEL_ID}/alert"
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
            "alert_active"
        ]
        is False
    )


    assert (
        payload[
            "alert_category"
        ]
        ==
        "none"
    )


    assert (
        payload[
            "severity"
        ]
        ==
        "none"
    )


    assert (
        payload[
            "notification_recommended"
        ]
        is False
    )


# ============================================================
# MONITORING GAP
# ============================================================


def test_no_monitoring_evidence_returns_warning_gap(
) -> None:

    client = (
        build_client()
    )


    expected = (
        build_ml_monitoring_alert_decision(
            model_health=
                summary()
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
                    f"models/{MODEL_ID}/alert"
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
            "alert_category"
        ]
        ==
        "monitoring_gap"
    )


    assert (
        payload[
            "severity"
        ]
        ==
        "warning"
    )


    assert (
        payload[
            "recommended_action"
        ]
        ==
        "establish_monitoring_evidence"
    )


    assert (
        payload[
            "notification_recommended"
        ]
        is True
    )


# ============================================================
# CRITICAL PERFORMANCE
# ============================================================


def test_degraded_performance_returns_critical_decision(
) -> None:

    client = (
        build_client()
    )


    health = (
        summary(
            drift=
                drift_record(
                    status="ok"
                ),

            performance=
                performance_record(
                    status="degraded"
                ),
        )
    )


    expected = (
        build_ml_monitoring_alert_decision(
            model_health=
                health
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
                    f"models/{MODEL_ID}/alert"
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
            "alert_category"
        ]
        ==
        "performance_degradation"
    )


    assert (
        payload[
            "severity"
        ]
        ==
        "critical"
    )


    assert (
        payload[
            "notification_recommended"
        ]
        is True
    )


# ============================================================
# REQUIRED WORKFLOW
# ============================================================


def test_workflow_id_required_before_service(
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
            build_ml_monitoring_alert_decision(
                model_health=
                    summary()
            )
        )


    with patched_service(
        fake_service
    ):

        response = (
            client.get(
                (
                    "/ml-monitoring/"
                    f"models/{MODEL_ID}/alert"
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
# ERROR HELPER
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
                    f"models/{MODEL_ID}/alert"
                ),

                params={
                    "workflow_id":
                        WORKFLOW_ID
                },
            )
        )


# ============================================================
# INPUT
# ============================================================


def test_input_error_maps_to_422(
) -> None:

    response = (
        response_for_error(
            MLMonitoringAlertServiceInputError(
                "private invalid identity"
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
        "monitoring_alert_input_invalid"
    )


# ============================================================
# NON-ENUMERATION
# ============================================================


def test_not_found_remains_generic_404(
) -> None:

    response = (
        response_for_error(
            MLMonitoringAlertServiceNotFoundError(
                (
                    "model exists in another "
                    "workflow"
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
        "monitoring_alert_not_found"
    )


    assert (
        "another workflow"
        not in
        detail[
            "message"
        ]
    )


# ============================================================
# AUTHORITY
# ============================================================


def test_evidence_conflict_remains_generic_409(
) -> None:

    response = (
        response_for_error(
            MLMonitoringAlertServiceAuthorityError(
                (
                    "secret training "
                    "fingerprint mismatch"
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
        "monitoring_alert_evidence_conflict"
    )


    assert (
        "fingerprint"
        not in
        detail[
            "message"
        ]
    )


# ============================================================
# STORAGE
# ============================================================


def test_storage_failure_remains_generic_500(
) -> None:

    response = (
        response_for_error(
            MLMonitoringAlertServiceStorageError(
                (
                    "sqlite failed at "
                    "/private/database.sqlite3"
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
        "monitoring_alert_unavailable"
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
# PRIVACY
# ============================================================


def test_alert_response_remains_aggregate_only(
) -> None:

    client = (
        build_client()
    )


    expected = (
        build_ml_monitoring_alert_decision(
            model_health=
                summary(
                    drift=
                        drift_record(),

                    performance=
                        performance_record(),
                )
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
                    f"models/{MODEL_ID}/alert"
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


    forbidden = [
        "predictions",
        "probabilities",
        "raw_values",
        "model_bytes",
        "business_note",
    ]


    for value in forbidden:

        assert (
            value
            not in
            serialized
        )


# ============================================================
# MAIN APP
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
        "models/{model_id}/alert"
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
        ML_MONITORING_ALERT_API_VERSION
        ==
        "ml_monitoring_alert_api_v0.1"
    )


    assert (
        ML_MONITORING_ALERT_API_CONTRACT_RULE_VERSION
        ==
        "ml_monitoring_alert_api_contract_v0.1"
    )


# ============================================================
# MAIN
# ============================================================


def main(
) -> None:

    print(
        (
            "=== DATALENS ML MONITORING "
            "ALERT API v0.1 ==="
        )
    )

    print()


    tests = [
        (
            "GET alert delegates identifiers only",
            test_get_alert_delegates_identifiers_only,
        ),
        (
            "No monitoring evidence returns warning gap",
            test_no_monitoring_evidence_returns_warning_gap,
        ),
        (
            "Degraded Performance returns critical decision",
            test_degraded_performance_returns_critical_decision,
        ),
        (
            "workflow_id required before service",
            test_workflow_id_required_before_service,
        ),
        (
            "Input error maps to 422",
            test_input_error_maps_to_422,
        ),
        (
            "Not-found remains generic 404",
            test_not_found_remains_generic_404,
        ),
        (
            "Evidence conflict remains generic 409",
            test_evidence_conflict_remains_generic_409,
        ),
        (
            "Storage failure remains generic 500",
            test_storage_failure_remains_generic_500,
        ),
        (
            "Alert response remains aggregate-only",
            test_alert_response_remains_aggregate_only,
        ),
        (
            "Monitoring Alert router registered in main app",
            test_router_registered_in_main_application,
        ),
        (
            "Monitoring Alert API rule versions",
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
            "PASS - ML Monitoring "
            "Alert API v0.1"
        )
    )


if __name__ == "__main__":
    main()
