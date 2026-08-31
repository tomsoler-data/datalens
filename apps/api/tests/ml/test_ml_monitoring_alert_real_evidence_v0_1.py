from __future__ import annotations


from app.ml.monitoring_alert_service import (
    get_ml_monitoring_alert_decision,
)


from app.preparation.preparation_session import (
    record_validation_stage_signal,
)


from tests.ml.test_ml_model_health_real_evidence_v0_1 import (
    persist_real_drift_and_performance,
    persist_shared_model_and_profile,
)


from tests.ml.test_ml_monitoring_service_real_handoff_v0_1 import (
    build_ready_preparation_workflow,
    isolated_real_handoff_environment,
)


# ============================================================
# REAL MODEL WITHOUT MONITORING EVIDENCE
# ============================================================


def test_real_model_without_monitoring_evidence_produces_warning_gap(
) -> None:

    with isolated_real_handoff_environment():

        (
            session,
            dataset_id,
            dataframe,
        ) = (
            build_ready_preparation_workflow()
        )


        (
            artifact,
            _,
        ) = (
            persist_shared_model_and_profile(
                workflow_id=
                    session.workflow_id,

                dataset_id=
                    dataset_id,

                preparation_revision=
                    session.revision,

                dataframe=
                    dataframe,
            )
        )


        # ----------------------------------------------------
        # No Drift Evaluation and no Performance Evaluation
        # have been persisted.
        #
        # This is a valid Model Health state, not a missing
        # resource.
        # ----------------------------------------------------


        alert = (
            get_ml_monitoring_alert_decision(
                workflow_id=
                    session.workflow_id,

                model_id=
                    artifact.model_id,
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


        assert (
            alert.health_status
            ==
            "insufficient_evidence"
        )


        assert (
            alert.health_reason
            ==
            "no_monitoring_evidence"
        )


        assert (
            alert.drift_evaluation_id
            is None
        )


        assert (
            alert.performance_evaluation_id
            is None
        )


# ============================================================
# REAL ALIGNED MONITORING EVIDENCE
# ============================================================


def test_real_persisted_healthy_evidence_produces_no_alert(
) -> None:

    with isolated_real_handoff_environment():

        (
            session,
            dataset_id,
            dataframe,
        ) = (
            build_ready_preparation_workflow()
        )


        (
            artifact,
            _,
            drift,
            performance,
        ) = (
            persist_real_drift_and_performance(
                session=
                    session,

                dataset_id=
                    dataset_id,

                dataframe=
                    dataframe,
            )
        )


        # ----------------------------------------------------
        # Real evidence must itself be aligned and healthy
        # before the Alert layer derives its decision.
        # ----------------------------------------------------


        assert (
            drift.overall_status
            ==
            "ok"
        )


        assert (
            performance.performance_status
            ==
            "ok"
        )


        assert (
            drift.observed_dataset_id
            ==
            performance.observed_dataset_id
            ==
            dataset_id
        )


        assert (
            drift
            .observed_preparation_session_revision
            ==
            performance
            .observed_preparation_session_revision
            ==
            session.revision
        )


        alert = (
            get_ml_monitoring_alert_decision(
                workflow_id=
                    session.workflow_id,

                model_id=
                    artifact.model_id,
            )
        )


        assert (
            alert.health_status
            ==
            "healthy"
        )


        assert (
            alert.health_reason
            ==
            "aligned_evidence_ok"
        )


        assert (
            alert.evidence_alignment
            ==
            "aligned"
        )


        assert (
            alert.joint_interpretation_allowed
            is True
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


        assert (
            alert.drift_evaluation_id
            ==
            drift.evaluation_id
        )


        assert (
            alert.performance_evaluation_id
            ==
            performance
            .performance_evaluation_id
        )


        # ----------------------------------------------------
        # Aggregate-only derived surface.
        # ----------------------------------------------------


        serialized = str(
            alert.model_dump(
                mode="json"
            )
        )


        forbidden = [
            "business_note",
            "not-monitored",
            "predictions",
            "probabilities",
            "raw_values",
            "model_bytes",
        ]


        for value in forbidden:

            assert (
                value
                not in
                serialized
            )


# ============================================================
# HISTORY SURVIVES WORKFLOW LEAVING READY
# ============================================================


def test_real_alert_decision_remains_readable_after_workflow_leaves_ready(
) -> None:

    with isolated_real_handoff_environment():

        (
            session,
            dataset_id,
            dataframe,
        ) = (
            build_ready_preparation_workflow()
        )


        (
            artifact,
            _,
            drift,
            performance,
        ) = (
            persist_real_drift_and_performance(
                session=
                    session,

                dataset_id=
                    dataset_id,

                dataframe=
                    dataframe,
            )
        )


        before = (
            get_ml_monitoring_alert_decision(
                workflow_id=
                    session.workflow_id,

                model_id=
                    artifact.model_id,
            )
        )


        assert (
            before.alert_active
            is False
        )


        assert (
            before.health_status
            ==
            "healthy"
        )


        # ----------------------------------------------------
        # Advance Preparation and invalidate current readiness.
        #
        # No new Drift or Performance evaluation is executed.
        # ----------------------------------------------------


        changed = (
            record_validation_stage_signal(
                workflow_id=
                    session.workflow_id,

                completed=
                    True,

                passed=
                    False,

                dataset_ids=[
                    dataset_id
                ],

                evidence_refs=[
                    (
                        "test:"
                        "monitoring-alert-history"
                    )
                ],

                blocking_reasons=[
                    (
                        "test:"
                        "workflow-no-longer-ready"
                    )
                ],

                expected_revision=
                    session.revision,
            )
        )


        assert (
            changed.revision
            >
            session.revision
        )


        assert (
            changed
            .snapshot
            .ready_for_analysis
            is False
        )


        after = (
            get_ml_monitoring_alert_decision(
                workflow_id=
                    session.workflow_id,

                model_id=
                    artifact.model_id,
            )
        )


        # ----------------------------------------------------
        # Alert Decision is derived from the same immutable
        # historical evidence.
        # ----------------------------------------------------


        assert (
            after
            ==
            before
        )


        assert (
            after.drift_evaluation_id
            ==
            drift.evaluation_id
        )


        assert (
            after.performance_evaluation_id
            ==
            performance
            .performance_evaluation_id
        )


        assert (
            after.health_status
            ==
            "healthy"
        )


        assert (
            after.alert_category
            ==
            "none"
        )


        assert (
            after.severity
            ==
            "none"
        )


# ============================================================
# MAIN
# ============================================================


def main(
) -> None:

    print(
        (
            "=== DATALENS ML MONITORING ALERT "
            "REAL EVIDENCE v0.1 ==="
        )
    )

    print()


    tests = [
        (
            (
                "Real model without evidence "
                "produces monitoring gap"
            ),
            (
                test_real_model_without_monitoring_evidence_produces_warning_gap
            ),
        ),
        (
            (
                "Real persisted healthy evidence "
                "produces no alert"
            ),
            (
                test_real_persisted_healthy_evidence_produces_no_alert
            ),
        ),
        (
            (
                "Alert remains readable after "
                "Preparation leaves READY"
            ),
            (
                test_real_alert_decision_remains_readable_after_workflow_leaves_ready
            ),
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
            "PASS - ML Monitoring Alert "
            "Real Evidence v0.1"
        )
    )


if __name__ == "__main__":
    main()
