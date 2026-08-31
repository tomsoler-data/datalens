from __future__ import annotations


import inspect


from contextlib import (
    contextmanager,
)


import app.ml.model_health_service as service_module


from app.ml.drift_evaluation import (
    MLDriftEvaluationRecord,
)


from app.ml.model_health_service import (
    ML_MODEL_HEALTH_SERVICE_RULE_VERSION,
    MLModelHealthServiceAuthorityError,
    MLModelHealthServiceInputError,
    MLModelHealthServiceNotFoundError,
    MLModelHealthServiceStorageError,
    get_ml_model_health_summary,
)


from app.ml.monitoring_history_service import (
    MLMonitoringHistoryNotFoundError,
    MLMonitoringHistoryStorageError,
)


from app.ml.performance_evaluation import (
    MLPerformanceEvaluationRecord,
)


from app.ml.performance_monitoring_history_service import (
    MLPerformanceMonitoringHistoryStorageError,
)


from tests.ml.test_ml_model_health_v0_1 import (
    MODEL_ID,
    WORKFLOW_ID,
    drift_record,
    performance_record,
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


def drift_evidence(
    *,
    status: str,
    identity_character: str,
    evaluated_at_utc: str,
    observed_revision: int = 11,
) -> MLDriftEvaluationRecord:

    payload = (
        drift_record(
            status=
                status,

            observed_revision=
                observed_revision,
        )
        .model_dump(
            mode="json"
        )
    )


    payload[
        "evaluation_id"
    ] = (
        "drift-evaluation:"
        +
        identity_character * 32
    )


    payload[
        "evaluated_at_utc"
    ] = evaluated_at_utc


    return (
        MLDriftEvaluationRecord
        .model_validate(
            payload
        )
    )


def performance_evidence(
    *,
    status: str,
    identity_character: str,
    evaluated_at_utc: str,
    observed_revision: int = 11,
) -> MLPerformanceEvaluationRecord:

    payload = (
        performance_record(
            status=
                status,

            observed_revision=
                observed_revision,
        )
        .model_dump(
            mode="json"
        )
    )


    payload[
        "performance_evaluation_id"
    ] = (
        "performance-evaluation:"
        +
        identity_character * 32
    )


    payload[
        "evaluated_at_utc"
    ] = evaluated_at_utc


    return (
        MLPerformanceEvaluationRecord
        .model_validate(
            payload
        )
    )


@contextmanager
def patched_histories(
    *,
    drift_history=None,
    performance_history=None,
    drift_error: Exception | None = None,
    performance_error: Exception | None = None,
):

    original_drift = (
        service_module
        .list_ml_monitoring_model_history
    )


    original_performance = (
        service_module
        .list_ml_performance_monitoring_model_history
    )


    captured = {
        "drift_calls":
            [],

        "performance_calls":
            [],
    }


    def fake_drift_history(
        *,
        workflow_id: str,
        model_id: str,
    ):

        captured[
            "drift_calls"
        ].append(
            {
                "workflow_id":
                    workflow_id,

                "model_id":
                    model_id,
            }
        )


        if (
            drift_error
            is not None
        ):
            raise drift_error


        return (
            list(
                drift_history
                if drift_history is not None
                else []
            )
        )


    def fake_performance_history(
        *,
        workflow_id: str,
        model_id: str,
    ):

        captured[
            "performance_calls"
        ].append(
            {
                "workflow_id":
                    workflow_id,

                "model_id":
                    model_id,
            }
        )


        if (
            performance_error
            is not None
        ):
            raise performance_error


        return (
            list(
                performance_history
                if performance_history is not None
                else []
            )
        )


    service_module.list_ml_monitoring_model_history = (
        fake_drift_history
    )


    service_module.list_ml_performance_monitoring_model_history = (
        fake_performance_history
    )


    try:
        yield captured

    finally:
        service_module.list_ml_monitoring_model_history = (
            original_drift
        )


        service_module.list_ml_performance_monitoring_model_history = (
            original_performance
        )


# ============================================================
# PUBLIC SURFACE
# ============================================================


def test_public_service_accepts_identifiers_only(
) -> None:

    signature = (
        inspect.signature(
            get_ml_model_health_summary
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


def test_empty_identities_blocked_before_history_read(
) -> None:

    with patched_histories() as captured:

        expect_error(
            MLModelHealthServiceInputError,

            lambda:
                get_ml_model_health_summary(
                    workflow_id=
                        "   ",

                    model_id=
                        MODEL_ID,
                ),
        )


        expect_error(
            MLModelHealthServiceInputError,

            lambda:
                get_ml_model_health_summary(
                    workflow_id=
                        WORKFLOW_ID,

                    model_id=
                        "   ",
                ),
        )


    assert (
        captured[
            "drift_calls"
        ]
        ==
        []
    )


    assert (
        captured[
            "performance_calls"
        ]
        ==
        []
    )


# ============================================================
# LATEST PERSISTED EVIDENCE
# ============================================================


def test_latest_persisted_evidence_is_selected(
) -> None:

    old_drift = (
        drift_evidence(
            status=
                "ok",

            identity_character=
                "1",

            evaluated_at_utc=
                "2026-08-29T10:00:00+00:00",
        )
    )


    latest_drift = (
        drift_evidence(
            status=
                "drift",

            identity_character=
                "2",

            evaluated_at_utc=
                "2026-08-29T11:00:00+00:00",
        )
    )


    old_performance = (
        performance_evidence(
            status=
                "warning",

            identity_character=
                "3",

            evaluated_at_utc=
                "2026-08-29T10:00:00+00:00",
        )
    )


    latest_performance = (
        performance_evidence(
            status=
                "ok",

            identity_character=
                "4",

            evaluated_at_utc=
                "2026-08-29T11:00:00+00:00",
        )
    )


    with patched_histories(
        drift_history=[
            old_drift,
            latest_drift,
        ],

        performance_history=[
            old_performance,
            latest_performance,
        ],
    ) as captured:

        result = (
            get_ml_model_health_summary(
                workflow_id=
                    f"  {WORKFLOW_ID}  ",

                model_id=
                    f"  {MODEL_ID}  ",
            )
        )


    assert (
        captured[
            "drift_calls"
        ]
        ==
        [
            {
                "workflow_id":
                    WORKFLOW_ID,

                "model_id":
                    MODEL_ID,
            }
        ]
    )


    assert (
        captured[
            "performance_calls"
        ]
        ==
        [
            {
                "workflow_id":
                    WORKFLOW_ID,

                "model_id":
                    MODEL_ID,
            }
        ]
    )


    assert (
        result.drift_evaluation_id
        ==
        latest_drift.evaluation_id
    )


    assert (
        result.performance_evaluation_id
        ==
        latest_performance
        .performance_evaluation_id
    )


    assert (
        result.drift_status
        ==
        "drift"
    )


    assert (
        result.performance_status
        ==
        "ok"
    )


    assert (
        result.health_status
        ==
        "attention"
    )


    assert (
        result.health_reason
        ==
        "drift_signal"
    )


# ============================================================
# EMPTY HISTORY
# ============================================================


def test_existing_model_without_monitoring_evidence_is_insufficient(
) -> None:

    with patched_histories(
        drift_history=[],
        performance_history=[],
    ):

        result = (
            get_ml_model_health_summary(
                workflow_id=
                    WORKFLOW_ID,

                model_id=
                    MODEL_ID,
            )
        )


    assert (
        result.health_status
        ==
        "insufficient_evidence"
    )


    assert (
        result.health_reason
        ==
        "no_monitoring_evidence"
    )


    assert (
        result.evidence_alignment
        ==
        "none"
    )


# ============================================================
# SINGLE-SOURCE SIGNALS
# ============================================================


def test_drift_only_signal_is_attention(
) -> None:

    latest_drift = (
        drift_evidence(
            status=
                "drift",

            identity_character=
                "5",

            evaluated_at_utc=
                "2026-08-29T12:00:00+00:00",
        )
    )


    with patched_histories(
        drift_history=[
            latest_drift
        ],

        performance_history=[],
    ):

        result = (
            get_ml_model_health_summary(
                workflow_id=
                    WORKFLOW_ID,

                model_id=
                    MODEL_ID,
            )
        )


    assert (
        result.health_status
        ==
        "attention"
    )


    assert (
        result.health_reason
        ==
        "drift_signal"
    )


    assert (
        result.evidence_alignment
        ==
        "single_source"
    )


def test_performance_only_degradation_is_critical(
) -> None:

    latest_performance = (
        performance_evidence(
            status=
                "degraded",

            identity_character=
                "6",

            evaluated_at_utc=
                "2026-08-29T12:00:00+00:00",
        )
    )


    with patched_histories(
        drift_history=[],

        performance_history=[
            latest_performance
        ],
    ):

        result = (
            get_ml_model_health_summary(
                workflow_id=
                    WORKFLOW_ID,

                model_id=
                    MODEL_ID,
            )
        )


    assert (
        result.health_status
        ==
        "critical"
    )


    assert (
        result.health_reason
        ==
        "performance_degraded"
    )


    assert (
        result.evidence_alignment
        ==
        "single_source"
    )


# ============================================================
# SNAPSHOT MISALIGNMENT
# ============================================================


def test_latest_misaligned_ok_evidence_is_not_healthy(
) -> None:

    latest_drift = (
        drift_evidence(
            status=
                "ok",

            identity_character=
                "7",

            evaluated_at_utc=
                "2026-08-29T13:00:00+00:00",

            observed_revision=
                11,
        )
    )


    latest_performance = (
        performance_evidence(
            status=
                "ok",

            identity_character=
                "8",

            evaluated_at_utc=
                "2026-08-29T13:05:00+00:00",

            observed_revision=
                12,
        )
    )


    with patched_histories(
        drift_history=[
            latest_drift
        ],

        performance_history=[
            latest_performance
        ],
    ):

        result = (
            get_ml_model_health_summary(
                workflow_id=
                    WORKFLOW_ID,

                model_id=
                    MODEL_ID,
            )
        )


    assert (
        result.health_status
        ==
        "insufficient_evidence"
    )


    assert (
        result.health_reason
        ==
        "evidence_misaligned"
    )


    assert (
        result.evidence_alignment
        ==
        "misaligned"
    )


    assert (
        result.joint_interpretation_allowed
        is False
    )


# ============================================================
# NON-ENUMERATION
# ============================================================


def test_history_not_found_is_non_enumerating(
) -> None:

    with patched_histories(
        drift_error=
            MLMonitoringHistoryNotFoundError(
                (
                    "model exists in another "
                    "workflow"
                )
            )
    ):

        expect_error(
            MLModelHealthServiceNotFoundError,

            lambda:
                get_ml_model_health_summary(
                    workflow_id=
                        WORKFLOW_ID,

                    model_id=
                        MODEL_ID,
                ),
        )


# ============================================================
# STORAGE FAILURE
# ============================================================


def test_drift_history_storage_failure_translated(
) -> None:

    with patched_histories(
        drift_error=
            MLMonitoringHistoryStorageError(
                "sqlite drift failure"
            )
    ):

        expect_error(
            MLModelHealthServiceStorageError,

            lambda:
                get_ml_model_health_summary(
                    workflow_id=
                        WORKFLOW_ID,

                    model_id=
                        MODEL_ID,
                ),
        )


def test_performance_history_storage_failure_translated(
) -> None:

    with patched_histories(
        drift_history=[],

        performance_error=(
            MLPerformanceMonitoringHistoryStorageError(
                "sqlite performance failure"
            )
        ),
    ):

        expect_error(
            MLModelHealthServiceStorageError,

            lambda:
                get_ml_model_health_summary(
                    workflow_id=
                        WORKFLOW_ID,

                    model_id=
                        MODEL_ID,
                ),
        )


# ============================================================
# PERSISTED AUTHORITY INTEGRITY
# ============================================================


def test_incompatible_training_authority_is_blocked(
) -> None:

    latest_drift = (
        drift_evidence(
            status=
                "ok",

            identity_character=
                "9",

            evaluated_at_utc=
                "2026-08-29T14:00:00+00:00",
        )
    )


    performance_payload = (
        performance_evidence(
            status=
                "ok",

            identity_character=
                "a",

            evaluated_at_utc=
                "2026-08-29T14:00:00+00:00",
        )
        .model_dump(
            mode="json"
        )
    )


    performance_payload[
        "training_contract_sha256"
    ] = (
        "b" * 64
    )


    incompatible_performance = (
        MLPerformanceEvaluationRecord
        .model_validate(
            performance_payload
        )
    )


    with patched_histories(
        drift_history=[
            latest_drift
        ],

        performance_history=[
            incompatible_performance
        ],
    ):

        expect_error(
            MLModelHealthServiceAuthorityError,

            lambda:
                get_ml_model_health_summary(
                    workflow_id=
                        WORKFLOW_ID,

                    model_id=
                        MODEL_ID,
                ),
        )


# ============================================================
# INVALID HISTORY SURFACE
# ============================================================


def test_invalid_history_surface_blocked(
) -> None:

    original = (
        service_module
        .list_ml_monitoring_model_history
    )


    def invalid_history(
        **kwargs,
    ):

        return (
            "not-a-history-list"
        )


    service_module.list_ml_monitoring_model_history = (
        invalid_history
    )


    try:

        expect_error(
            MLModelHealthServiceAuthorityError,

            lambda:
                get_ml_model_health_summary(
                    workflow_id=
                        WORKFLOW_ID,

                    model_id=
                        MODEL_ID,
                ),
        )

    finally:

        service_module.list_ml_monitoring_model_history = (
            original
        )


# ============================================================
# VERSION
# ============================================================


def test_rule_version(
) -> None:

    assert (
        ML_MODEL_HEALTH_SERVICE_RULE_VERSION
        ==
        "ml_model_health_service_v0.1"
    )


# ============================================================
# MAIN
# ============================================================


def main(
) -> None:

    print(
        (
            "=== DATALENS ML MODEL HEALTH "
            "SERVICE v0.1 ==="
        )
    )

    print()


    tests = [
        (
            "Public service accepts identifiers only",
            test_public_service_accepts_identifiers_only,
        ),
        (
            "Empty identities blocked before history read",
            test_empty_identities_blocked_before_history_read,
        ),
        (
            "Latest persisted evidence is selected",
            test_latest_persisted_evidence_is_selected,
        ),
        (
            "No monitoring evidence is insufficient",
            test_existing_model_without_monitoring_evidence_is_insufficient,
        ),
        (
            "Drift-only signal requires attention",
            test_drift_only_signal_is_attention,
        ),
        (
            "Performance-only degradation is critical",
            test_performance_only_degradation_is_critical,
        ),
        (
            "Misaligned latest evidence is not healthy",
            test_latest_misaligned_ok_evidence_is_not_healthy,
        ),
        (
            "History not-found remains non-enumerating",
            test_history_not_found_is_non_enumerating,
        ),
        (
            "Drift storage failure translated",
            test_drift_history_storage_failure_translated,
        ),
        (
            "Performance storage failure translated",
            test_performance_history_storage_failure_translated,
        ),
        (
            "Incompatible training authority blocked",
            test_incompatible_training_authority_is_blocked,
        ),
        (
            "Invalid history surface blocked",
            test_invalid_history_surface_blocked,
        ),
        (
            "Model Health Service rule version",
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
            "PASS - ML Model Health "
            "Service v0.1"
        )
    )


if __name__ == "__main__":
    main()
