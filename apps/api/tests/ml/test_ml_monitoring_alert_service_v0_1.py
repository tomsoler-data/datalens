from __future__ import annotations


import inspect


from contextlib import (
    contextmanager,
)


import app.ml.monitoring_alert_service as service_module


from app.ml.model_health import (
    MLModelHealthSummary,
)


from app.ml.model_health_service import (
    MLModelHealthServiceAuthorityError,
    MLModelHealthServiceNotFoundError,
    MLModelHealthServiceStorageError,
)


from app.ml.monitoring_alert import (
    MLMonitoringAlertInputError,
)


from app.ml.monitoring_alert_service import (
    ML_MONITORING_ALERT_SERVICE_RULE_VERSION,
    MLMonitoringAlertServiceAuthorityError,
    MLMonitoringAlertServiceInputError,
    MLMonitoringAlertServiceNotFoundError,
    MLMonitoringAlertServiceStorageError,
    get_ml_monitoring_alert_decision,
)


from tests.ml.test_ml_model_health_v0_1 import (
    MODEL_ID,
    WORKFLOW_ID,
    drift_record,
    performance_record,
    summary,
)


# ============================================================
# HELPERS
# ============================================================


def expect_error(
    error_type,
    factory,
) -> None:

    try:
        factory()

    except error_type:
        return


    raise AssertionError(
        (
            "Expected "
            f"{error_type.__name__}."
        )
    )


@contextmanager
def patched_health_service(
    *,
    result=None,
    error: Exception | None = None,
):

    original = (
        service_module
        .get_ml_model_health_summary
    )


    captured = []


    def fake_health_service(
        *,
        workflow_id: str,
        model_id: str,
    ):

        captured.append(
            {
                "workflow_id":
                    workflow_id,

                "model_id":
                    model_id,
            }
        )


        if (
            error
            is not None
        ):
            raise error


        return result


    service_module.get_ml_model_health_summary = (
        fake_health_service
    )


    try:
        yield captured

    finally:
        service_module.get_ml_model_health_summary = (
            original
        )


# ============================================================
# PUBLIC SURFACE
# ============================================================


def test_public_service_accepts_identifiers_only(
) -> None:

    signature = (
        inspect.signature(
            get_ml_monitoring_alert_decision
        )
    )


    assert (
        list(
            signature.parameters
        )
        ==
        [
            "workflow_id",
            "model_id",
        ]
    )


    for parameter in (
        signature.parameters.values()
    ):

        assert (
            parameter.kind
            ==
            inspect.Parameter.KEYWORD_ONLY
        )


# ============================================================
# INPUT
# ============================================================


def test_empty_identities_blocked_before_health_read(
) -> None:

    with patched_health_service(
        result=
            summary()
    ) as captured:

        expect_error(
            MLMonitoringAlertServiceInputError,

            lambda:
                get_ml_monitoring_alert_decision(
                    workflow_id=
                        "   ",

                    model_id=
                        MODEL_ID,
                ),
        )


        expect_error(
            MLMonitoringAlertServiceInputError,

            lambda:
                get_ml_monitoring_alert_decision(
                    workflow_id=
                        WORKFLOW_ID,

                    model_id=
                        "   ",
                ),
        )


    assert (
        captured
        ==
        []
    )


# ============================================================
# DELEGATION
# ============================================================


def test_service_normalizes_and_delegates_identifiers(
) -> None:

    health = (
        summary()
    )


    with patched_health_service(
        result=
            health
    ) as captured:

        result = (
            get_ml_monitoring_alert_decision(
                workflow_id=
                    f"  {WORKFLOW_ID}  ",

                model_id=
                    f"  {MODEL_ID}  ",
            )
        )


    assert captured == [
        {
            "workflow_id":
                WORKFLOW_ID,

            "model_id":
                MODEL_ID,
        }
    ]


    assert (
        result.workflow_id
        ==
        WORKFLOW_ID
    )


    assert (
        result.model_id
        ==
        MODEL_ID
    )


# ============================================================
# HEALTHY
# ============================================================


def test_healthy_model_produces_no_alert(
) -> None:

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


    with patched_health_service(
        result=
            health
    ):

        result = (
            get_ml_monitoring_alert_decision(
                workflow_id=
                    WORKFLOW_ID,

                model_id=
                    MODEL_ID,
            )
        )


    assert (
        result.alert_active
        is False
    )


    assert (
        result.alert_category
        ==
        "none"
    )


    assert (
        result.severity
        ==
        "none"
    )


# ============================================================
# MONITORING GAP
# ============================================================


def test_missing_monitoring_evidence_produces_warning_gap(
) -> None:

    with patched_health_service(
        result=
            summary()
    ):

        result = (
            get_ml_monitoring_alert_decision(
                workflow_id=
                    WORKFLOW_ID,

                model_id=
                    MODEL_ID,
            )
        )


    assert (
        result.alert_active
        is True
    )


    assert (
        result.alert_category
        ==
        "monitoring_gap"
    )


    assert (
        result.severity
        ==
        "warning"
    )


    assert (
        result.notification_recommended
        is True
    )


# ============================================================
# DRIFT
# ============================================================


def test_strong_drift_with_good_performance_is_warning_only(
) -> None:

    health = (
        summary(
            drift=
                drift_record(
                    status="drift"
                ),

            performance=
                performance_record(
                    status="ok"
                ),
        )
    )


    with patched_health_service(
        result=
            health
    ):

        result = (
            get_ml_monitoring_alert_decision(
                workflow_id=
                    WORKFLOW_ID,

                model_id=
                    MODEL_ID,
            )
        )


    assert (
        result.alert_category
        ==
        "data_shift"
    )


    assert (
        result.severity
        ==
        "warning"
    )


    assert (
        result.severity
        !=
        "critical"
    )


# ============================================================
# PERFORMANCE
# ============================================================


def test_degraded_performance_produces_critical_alert(
) -> None:

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


    with patched_health_service(
        result=
            health
    ):

        result = (
            get_ml_monitoring_alert_decision(
                workflow_id=
                    WORKFLOW_ID,

                model_id=
                    MODEL_ID,
            )
        )


    assert (
        result.alert_category
        ==
        "performance_degradation"
    )


    assert (
        result.severity
        ==
        "critical"
    )


    assert (
        result.notification_recommended
        is True
    )


# ============================================================
# NON-ENUMERATION
# ============================================================


def test_health_not_found_remains_non_enumerating(
) -> None:

    with patched_health_service(
        error=
            MLModelHealthServiceNotFoundError(
                (
                    "model exists in another "
                    "workflow"
                )
            )
    ):

        expect_error(
            MLMonitoringAlertServiceNotFoundError,

            lambda:
                get_ml_monitoring_alert_decision(
                    workflow_id=
                        WORKFLOW_ID,

                    model_id=
                        MODEL_ID,
                ),
        )


# ============================================================
# STORAGE
# ============================================================


def test_health_storage_failure_translated(
) -> None:

    with patched_health_service(
        error=
            MLModelHealthServiceStorageError(
                (
                    "sqlite failure at "
                    "/private/database.sqlite3"
                )
            )
    ):

        expect_error(
            MLMonitoringAlertServiceStorageError,

            lambda:
                get_ml_monitoring_alert_decision(
                    workflow_id=
                        WORKFLOW_ID,

                    model_id=
                        MODEL_ID,
                ),
        )


# ============================================================
# AUTHORITY
# ============================================================


def test_health_authority_failure_translated(
) -> None:

    with patched_health_service(
        error=
            MLModelHealthServiceAuthorityError(
                (
                    "secret training "
                    "fingerprint mismatch"
                )
            )
    ):

        expect_error(
            MLMonitoringAlertServiceAuthorityError,

            lambda:
                get_ml_monitoring_alert_decision(
                    workflow_id=
                        WORKFLOW_ID,

                    model_id=
                        MODEL_ID,
                ),
        )


def test_invalid_health_surface_blocked(
) -> None:

    with patched_health_service(
        result=
            "not-a-health-summary"
    ):

        expect_error(
            MLMonitoringAlertServiceAuthorityError,

            lambda:
                get_ml_monitoring_alert_decision(
                    workflow_id=
                        WORKFLOW_ID,

                    model_id=
                        MODEL_ID,
                ),
        )


def test_cross_identity_health_summary_blocked(
) -> None:

    valid = (
        summary()
    )


    payload = (
        valid.model_dump(
            mode="json"
        )
    )


    payload[
        "model_id"
    ] = (
        "model:"
        +
        "9" * 32
    )


    wrong_identity = (
        MLModelHealthSummary
        .model_validate(
            payload
        )
    )


    with patched_health_service(
        result=
            wrong_identity
    ):

        expect_error(
            MLMonitoringAlertServiceAuthorityError,

            lambda:
                get_ml_monitoring_alert_decision(
                    workflow_id=
                        WORKFLOW_ID,

                    model_id=
                        MODEL_ID,
                ),
        )


# ============================================================
# BUILDER FAILURE
# ============================================================


def test_alert_builder_failure_translated(
) -> None:

    health = (
        summary()
    )


    original_builder = (
        service_module
        .build_ml_monitoring_alert_decision
    )


    def failing_builder(
        **kwargs,
    ):

        raise MLMonitoringAlertInputError(
            "injected policy failure"
        )


    service_module.build_ml_monitoring_alert_decision = (
        failing_builder
    )


    try:

        with patched_health_service(
            result=
                health
        ):

            expect_error(
                MLMonitoringAlertServiceAuthorityError,

                lambda:
                    get_ml_monitoring_alert_decision(
                        workflow_id=
                            WORKFLOW_ID,

                        model_id=
                            MODEL_ID,
                    ),
            )

    finally:

        service_module.build_ml_monitoring_alert_decision = (
            original_builder
        )


# ============================================================
# PRIVACY
# ============================================================


def test_service_output_remains_aggregate_only(
) -> None:

    health = (
        summary(
            drift=
                drift_record(),

            performance=
                performance_record(),
        )
    )


    with patched_health_service(
        result=
            health
    ):

        result = (
            get_ml_monitoring_alert_decision(
                workflow_id=
                    WORKFLOW_ID,

                model_id=
                    MODEL_ID,
            )
        )


    serialized = str(
        result.model_dump(
            mode="json"
        )
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
# VERSION
# ============================================================


def test_rule_version(
) -> None:

    assert (
        ML_MONITORING_ALERT_SERVICE_RULE_VERSION
        ==
        "ml_monitoring_alert_service_v0.1"
    )


# ============================================================
# MAIN
# ============================================================


def main(
) -> None:

    print(
        (
            "=== DATALENS ML MONITORING "
            "ALERT SERVICE v0.1 ==="
        )
    )

    print()


    tests = [
        (
            "Public service accepts identifiers only",
            test_public_service_accepts_identifiers_only,
        ),
        (
            "Empty identities blocked before Health read",
            test_empty_identities_blocked_before_health_read,
        ),
        (
            "Identifiers normalized and delegated",
            test_service_normalizes_and_delegates_identifiers,
        ),
        (
            "Healthy model produces no alert",
            test_healthy_model_produces_no_alert,
        ),
        (
            "Missing monitoring evidence produces warning gap",
            test_missing_monitoring_evidence_produces_warning_gap,
        ),
        (
            "Strong Drift with good Performance is warning only",
            test_strong_drift_with_good_performance_is_warning_only,
        ),
        (
            "Degraded Performance produces critical alert",
            test_degraded_performance_produces_critical_alert,
        ),
        (
            "Health not-found remains non-enumerating",
            test_health_not_found_remains_non_enumerating,
        ),
        (
            "Health storage failure translated",
            test_health_storage_failure_translated,
        ),
        (
            "Health authority failure translated",
            test_health_authority_failure_translated,
        ),
        (
            "Invalid Health surface blocked",
            test_invalid_health_surface_blocked,
        ),
        (
            "Cross-identity Health Summary blocked",
            test_cross_identity_health_summary_blocked,
        ),
        (
            "Alert builder failure translated",
            test_alert_builder_failure_translated,
        ),
        (
            "Service output remains aggregate-only",
            test_service_output_remains_aggregate_only,
        ),
        (
            "Monitoring Alert Service rule version",
            test_rule_version,
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
            "Alert Service v0.1"
        )
    )


if __name__ == "__main__":
    main()
