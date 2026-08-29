from __future__ import annotations


from pydantic import (
    ValidationError,
)


from app.ml.monitoring_alert import (
    ML_MONITORING_ALERT_RULE_VERSION,
    MLMonitoringAlertDecision,
    build_ml_monitoring_alert_decision,
)


from tests.ml.test_ml_model_health_v0_1 import (
    drift_record,
    performance_record,
    summary,
)


# ============================================================
# HEALTHY
# ============================================================


def test_healthy_model_has_no_alert(
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


    alert = (
        build_ml_monitoring_alert_decision(
            model_health=
                health
        )
    )


    assert (
        alert.alert_active
        is False
    )


    assert (
        alert.alert_category
        ==
        "none"
    )


    assert (
        alert.severity
        ==
        "none"
    )


    assert (
        alert.recommended_action
        ==
        "no_action"
    )


    assert (
        alert.notification_recommended
        is False
    )


# ============================================================
# NO MONITORING
# ============================================================


def test_no_monitoring_evidence_is_warning_gap(
) -> None:

    alert = (
        build_ml_monitoring_alert_decision(
            model_health=
                summary()
        )
    )


    assert (
        alert.alert_active
        is True
    )


    assert (
        alert.alert_category
        ==
        "monitoring_gap"
    )


    assert (
        alert.severity
        ==
        "warning"
    )


    assert (
        alert.recommended_action
        ==
        "establish_monitoring_evidence"
    )


    assert (
        alert.notification_recommended
        is True
    )


# ============================================================
# PARTIAL COVERAGE
# ============================================================


def test_single_ok_evidence_is_info_gap_without_notification(
) -> None:

    health = (
        summary(
            drift=
                drift_record(
                    status="ok"
                )
        )
    )


    alert = (
        build_ml_monitoring_alert_decision(
            model_health=
                health
        )
    )


    assert (
        alert.alert_category
        ==
        "monitoring_gap"
    )


    assert (
        alert.severity
        ==
        "info"
    )


    assert (
        alert.recommended_action
        ==
        "complete_monitoring_evidence"
    )


    assert (
        alert.notification_recommended
        is False
    )


# ============================================================
# ALIGNMENT GAP
# ============================================================


def test_misaligned_evidence_requires_warning(
) -> None:

    health = (
        summary(
            drift=
                drift_record(
                    status="ok",
                    observed_revision=11,
                ),

            performance=
                performance_record(
                    status="ok",
                    observed_revision=12,
                ),
        )
    )


    alert = (
        build_ml_monitoring_alert_decision(
            model_health=
                health
        )
    )


    assert (
        alert.alert_category
        ==
        "evidence_alignment_gap"
    )


    assert (
        alert.severity
        ==
        "warning"
    )


    assert (
        alert.recommended_action
        ==
        "align_monitoring_snapshots"
    )


    assert (
        alert.notification_recommended
        is True
    )


    assert (
        alert.joint_interpretation_allowed
        is False
    )


# ============================================================
# DRIFT
# ============================================================


def test_drift_warning_is_data_shift_warning(
) -> None:

    health = (
        summary(
            drift=
                drift_record(
                    status="warning"
                ),

            performance=
                performance_record(
                    status="ok"
                ),
        )
    )


    alert = (
        build_ml_monitoring_alert_decision(
            model_health=
                health
        )
    )


    assert (
        alert.alert_category
        ==
        "data_shift"
    )


    assert (
        alert.severity
        ==
        "warning"
    )


    assert (
        alert.notification_recommended
        is True
    )


def test_strong_drift_with_good_performance_is_not_critical(
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


    alert = (
        build_ml_monitoring_alert_decision(
            model_health=
                health
        )
    )


    assert (
        alert.alert_category
        ==
        "data_shift"
    )


    assert (
        alert.severity
        ==
        "warning"
    )


    assert (
        alert.severity
        !=
        "critical"
    )


# ============================================================
# PERFORMANCE WARNING
# ============================================================


def test_performance_warning_is_warning_alert(
) -> None:

    health = (
        summary(
            drift=
                drift_record(
                    status="ok"
                ),

            performance=
                performance_record(
                    status="warning"
                ),
        )
    )


    alert = (
        build_ml_monitoring_alert_decision(
            model_health=
                health
        )
    )


    assert (
        alert.alert_category
        ==
        "performance_warning"
    )


    assert (
        alert.severity
        ==
        "warning"
    )


    assert (
        alert.recommended_action
        ==
        "review_model_performance"
    )


# ============================================================
# PERFORMANCE DEGRADATION
# ============================================================


def test_performance_degradation_is_critical(
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


    alert = (
        build_ml_monitoring_alert_decision(
            model_health=
                health
        )
    )


    assert (
        alert.alert_category
        ==
        "performance_degradation"
    )


    assert (
        alert.severity
        ==
        "critical"
    )


    assert (
        alert.recommended_action
        ==
        "investigate_model_degradation"
    )


    assert (
        alert.notification_recommended
        is True
    )


# ============================================================
# DEGRADED + MISALIGNED
# ============================================================


def test_degraded_performance_remains_critical_without_causal_join(
) -> None:

    health = (
        summary(
            drift=
                drift_record(
                    status="drift",
                    observed_revision=11,
                ),

            performance=
                performance_record(
                    status="degraded",
                    observed_revision=12,
                ),
        )
    )


    alert = (
        build_ml_monitoring_alert_decision(
            model_health=
                health
        )
    )


    assert (
        alert.severity
        ==
        "critical"
    )


    assert (
        alert.alert_category
        ==
        "performance_degradation"
    )


    assert (
        alert.joint_interpretation_allowed
        is False
    )


# ============================================================
# EVIDENCE REFERENCES
# ============================================================


def test_alert_preserves_evidence_references(
) -> None:

    health = (
        summary(
            drift=
                drift_record(),

            performance=
                performance_record(),
        )
    )


    alert = (
        build_ml_monitoring_alert_decision(
            model_health=
                health
        )
    )


    assert (
        alert.drift_evaluation_id
        ==
        health.drift_evaluation_id
    )


    assert (
        alert.performance_evaluation_id
        ==
        health.performance_evaluation_id
    )


# ============================================================
# CONTRACT HARDENING
# ============================================================


def test_raw_payload_forbidden(
) -> None:

    valid = (
        build_ml_monitoring_alert_decision(
            model_health=
                summary()
        )
    )


    payload = (
        valid.model_dump(
            mode="json"
        )
    )


    payload[
        "predictions"
    ] = [
        0,
        1,
    ]


    try:
        MLMonitoringAlertDecision.model_validate(
            payload
        )

    except ValidationError:
        return


    raise AssertionError(
        "Raw prediction payload must be forbidden."
    )


def test_severity_tampering_blocked(
) -> None:

    valid = (
        build_ml_monitoring_alert_decision(
            model_health=
                summary(
                    drift=
                        drift_record(
                            status="drift"
                        )
                )
        )
    )


    payload = (
        valid.model_dump(
            mode="json"
        )
    )


    payload[
        "severity"
    ] = "critical"


    try:
        MLMonitoringAlertDecision.model_validate(
            payload
        )

    except ValidationError:
        return


    raise AssertionError(
        "Alert severity tampering must be blocked."
    )


def test_notification_tampering_blocked(
) -> None:

    valid = (
        build_ml_monitoring_alert_decision(
            model_health=
                summary(
                    drift=
                        drift_record(
                            status="ok"
                        )
                )
        )
    )


    payload = (
        valid.model_dump(
            mode="json"
        )
    )


    payload[
        "notification_recommended"
    ] = True


    try:
        MLMonitoringAlertDecision.model_validate(
            payload
        )

    except ValidationError:
        return


    raise AssertionError(
        (
            "notification_recommended tampering "
            "must be blocked."
        )
    )


def test_alert_decision_is_frozen(
) -> None:

    alert = (
        build_ml_monitoring_alert_decision(
            model_health=
                summary()
        )
    )


    try:
        alert.severity = "critical"

    except ValidationError:
        return


    raise AssertionError(
        "Monitoring Alert Decision must be frozen."
    )


# ============================================================
# PRIVACY
# ============================================================


def test_alert_surface_remains_aggregate_only(
) -> None:

    alert = (
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


    serialized = str(
        alert.model_dump(
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
        ML_MONITORING_ALERT_RULE_VERSION
        ==
        "ml_monitoring_alert_v0.1"
    )


# ============================================================
# MAIN
# ============================================================


def main(
) -> None:

    print(
        (
            "=== DATALENS ML MONITORING "
            "ALERT v0.1 ==="
        )
    )

    print()


    tests = [
        (
            "Healthy model has no alert",
            test_healthy_model_has_no_alert,
        ),
        (
            "No monitoring evidence creates warning gap",
            test_no_monitoring_evidence_is_warning_gap,
        ),
        (
            "Single OK evidence is informational gap",
            test_single_ok_evidence_is_info_gap_without_notification,
        ),
        (
            "Misaligned evidence creates warning",
            test_misaligned_evidence_requires_warning,
        ),
        (
            "Drift warning creates data-shift warning",
            test_drift_warning_is_data_shift_warning,
        ),
        (
            "Strong Drift with good Performance is not critical",
            test_strong_drift_with_good_performance_is_not_critical,
        ),
        (
            "Performance warning creates warning alert",
            test_performance_warning_is_warning_alert,
        ),
        (
            "Performance degradation is critical",
            test_performance_degradation_is_critical,
        ),
        (
            "Degraded Performance remains critical without causal join",
            test_degraded_performance_remains_critical_without_causal_join,
        ),
        (
            "Evidence references preserved",
            test_alert_preserves_evidence_references,
        ),
        (
            "Raw payload forbidden",
            test_raw_payload_forbidden,
        ),
        (
            "Severity tampering blocked",
            test_severity_tampering_blocked,
        ),
        (
            "Notification tampering blocked",
            test_notification_tampering_blocked,
        ),
        (
            "Monitoring Alert Decision frozen",
            test_alert_decision_is_frozen,
        ),
        (
            "Alert surface remains aggregate-only",
            test_alert_surface_remains_aggregate_only,
        ),
        (
            "Monitoring Alert rule version",
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
            "Alert v0.1"
        )
    )


if __name__ == "__main__":
    main()
