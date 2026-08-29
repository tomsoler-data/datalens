from __future__ import annotations


from pydantic import (
    ValidationError,
)


from app.ml.drift_evaluation import (
    MLDriftEvaluationRecord,
)


from app.ml.model_health import (
    ML_MODEL_HEALTH_RULE_VERSION,
    MLModelHealthAuthorityError,
    MLModelHealthSummary,
    build_ml_model_health_summary,
)


from app.ml.performance_evaluation import (
    MLPerformanceEvaluationRecord,
)


from tests.ml.test_ml_drift_evaluation_contract_v0_1 import (
    record as base_drift_record,
)


from tests.ml.test_ml_performance_evaluation_contract_v0_1 import (
    classification_record,
)


# ============================================================
# CONSTANT AUTHORITY
# ============================================================


WORKFLOW_ID = (
    "prep:drift-contract"
)


MODEL_ID = (
    "model:"
    +
    "3" * 32
)


REFERENCE_DATASET_ID = (
    "dataset:reference"
)


OBSERVED_DATASET_ID = (
    "dataset:observed"
)


EXPERIMENT_ID = (
    "experiment:"
    +
    "4" * 32
)


TRAINING_SHA = (
    "a" * 64
)


TRAINING_REVISION = 7


OBSERVED_REVISION = 11


OBSERVED_ROW_COUNT = 10


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


def drift_record(
    *,
    status: str = "ok",
    observed_dataset_id: str = OBSERVED_DATASET_ID,
    observed_revision=OBSERVED_REVISION,
) -> MLDriftEvaluationRecord:

    payload = (
        base_drift_record()
        .model_dump(
            mode="json"
        )
    )


    payload[
        "observed_dataset_id"
    ] = observed_dataset_id


    payload[
        "observed_preparation_session_revision"
    ] = observed_revision


    if (
        status
        ==
        "warning"
    ):

        numeric = (
            payload[
                "feature_results"
            ][
                0
            ]
        )


        numeric[
            "population_stability_index"
        ] = 0.15


        numeric[
            "distribution_status"
        ] = "warning"


        numeric[
            "status"
        ] = "warning"


        payload[
            "warning_feature_count"
        ] = 1


        payload[
            "drift_feature_count"
        ] = 0


        payload[
            "overall_status"
        ] = "warning"


    elif (
        status
        ==
        "drift"
    ):

        numeric = (
            payload[
                "feature_results"
            ][
                0
            ]
        )


        numeric[
            "population_stability_index"
        ] = 0.30


        numeric[
            "distribution_status"
        ] = "drift"


        numeric[
            "status"
        ] = "drift"


        payload[
            "warning_feature_count"
        ] = 0


        payload[
            "drift_feature_count"
        ] = 1


        payload[
            "overall_status"
        ] = "drift"


    elif (
        status
        !=
        "ok"
    ):

        raise AssertionError(
            (
                "Unsupported Drift status "
                f"for test: {status}"
            )
        )


    return (
        MLDriftEvaluationRecord
        .model_validate(
            payload
        )
    )


def performance_record(
    *,
    status: str = "ok",
    observed_dataset_id: str = OBSERVED_DATASET_ID,
    observed_revision: int = OBSERVED_REVISION,
) -> MLPerformanceEvaluationRecord:

    if (
        status
        ==
        "ok"
    ):
        observed_f1 = 0.80

    elif (
        status
        ==
        "warning"
    ):
        observed_f1 = 0.74

    elif (
        status
        ==
        "degraded"
    ):
        observed_f1 = 0.68

    else:
        raise AssertionError(
            (
                "Unsupported Performance status "
                f"for test: {status}"
            )
        )


    payload = (
        classification_record(
            observed_f1=
                observed_f1,

            status=
                status,
        )
        .model_dump(
            mode="json"
        )
    )


    payload[
        "workflow_id"
    ] = WORKFLOW_ID


    payload[
        "model_id"
    ] = MODEL_ID


    payload[
        "reference_dataset_id"
    ] = REFERENCE_DATASET_ID


    payload[
        "observed_dataset_id"
    ] = observed_dataset_id


    payload[
        "experiment_id"
    ] = EXPERIMENT_ID


    payload[
        "preparation_session_revision"
    ] = TRAINING_REVISION


    payload[
        "observed_preparation_session_revision"
    ] = observed_revision


    payload[
        "training_contract_sha256"
    ] = TRAINING_SHA


    payload[
        "observed_row_count"
    ] = OBSERVED_ROW_COUNT


    return (
        MLPerformanceEvaluationRecord
        .model_validate(
            payload
        )
    )


def summary(
    *,
    drift=None,
    performance=None,
):

    return (
        build_ml_model_health_summary(
            workflow_id=
                WORKFLOW_ID,

            model_id=
                MODEL_ID,

            latest_drift=
                drift,

            latest_performance=
                performance,
        )
    )


# ============================================================
# ALIGNED HEALTHY
# ============================================================


def test_aligned_ok_evidence_is_healthy(
) -> None:

    result = (
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


    assert (
        result.health_status
        ==
        "healthy"
    )


    assert (
        result.health_reason
        ==
        "aligned_evidence_ok"
    )


    assert (
        result.evidence_alignment
        ==
        "aligned"
    )


    assert (
        result.joint_interpretation_allowed
        is True
    )


# ============================================================
# DRIFT SIGNAL
# ============================================================


def test_warning_drift_is_attention(
) -> None:

    result = (
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


def test_strong_drift_with_good_performance_is_not_critical(
) -> None:

    result = (
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


    assert (
        result.health_status
        ==
        "attention"
    )


    assert (
        result.performance_status
        ==
        "ok"
    )


    assert (
        result.health_status
        !=
        "critical"
    )


# ============================================================
# PERFORMANCE SIGNAL
# ============================================================


def test_performance_warning_is_attention(
) -> None:

    result = (
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


    assert (
        result.health_status
        ==
        "attention"
    )


    assert (
        result.health_reason
        ==
        "performance_warning"
    )


def test_performance_degradation_is_critical(
) -> None:

    result = (
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


# ============================================================
# INSUFFICIENT EVIDENCE
# ============================================================


def test_no_evidence_is_insufficient(
) -> None:

    result = (
        summary()
    )


    assert (
        result.health_status
        ==
        "insufficient_evidence"
    )


    assert (
        result.evidence_alignment
        ==
        "none"
    )


def test_drift_only_ok_is_insufficient(
) -> None:

    result = (
        summary(
            drift=
                drift_record(
                    status="ok"
                )
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
        "drift_only_ok"
    )


def test_performance_only_ok_is_insufficient(
) -> None:

    result = (
        summary(
            performance=
                performance_record(
                    status="ok"
                )
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
        "performance_only_ok"
    )


def test_drift_only_signal_is_attention(
) -> None:

    result = (
        summary(
            drift=
                drift_record(
                    status="drift"
                )
        )
    )


    assert (
        result.health_status
        ==
        "attention"
    )


def test_performance_only_degraded_is_critical(
) -> None:

    result = (
        summary(
            performance=
                performance_record(
                    status="degraded"
                )
        )
    )


    assert (
        result.health_status
        ==
        "critical"
    )


# ============================================================
# SNAPSHOT ALIGNMENT
# ============================================================


def test_two_ok_but_misaligned_snapshots_are_not_healthy(
) -> None:

    result = (
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


def test_different_observed_dataset_is_misaligned(
) -> None:

    result = (
        summary(
            drift=
                drift_record(
                    status="ok",
                    observed_dataset_id=
                        "dataset:drift-snapshot",
                ),

            performance=
                performance_record(
                    status="ok",
                    observed_dataset_id=
                        "dataset:performance-snapshot",
                ),
        )
    )


    assert (
        result.evidence_alignment
        ==
        "misaligned"
    )


    assert (
        result.health_status
        ==
        "insufficient_evidence"
    )


def test_legacy_drift_revision_is_unverifiable(
) -> None:

    result = (
        summary(
            drift=
                drift_record(
                    status="ok",
                    observed_revision=None,
                ),

            performance=
                performance_record(
                    status="ok"
                ),
        )
    )


    assert (
        result.evidence_alignment
        ==
        "unverifiable"
    )


    assert (
        result.health_status
        ==
        "insufficient_evidence"
    )


    assert (
        result.health_reason
        ==
        "evidence_unverifiable"
    )


def test_degraded_performance_remains_critical_when_misaligned(
) -> None:

    result = (
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


    assert (
        result.health_status
        ==
        "critical"
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
# TRAINING AUTHORITY
# ============================================================


def test_training_fingerprint_mismatch_blocked(
) -> None:

    drift = (
        drift_record()
    )


    performance_payload = (
        performance_record()
        .model_dump(
            mode="json"
        )
    )


    performance_payload[
        "training_contract_sha256"
    ] = (
        "b" * 64
    )


    performance = (
        MLPerformanceEvaluationRecord
        .model_validate(
            performance_payload
        )
    )


    expect_error(
        MLModelHealthAuthorityError,

        lambda:
            summary(
                drift=
                    drift,

                performance=
                    performance,
            ),
    )


def test_cross_model_evidence_blocked(
) -> None:

    payload = (
        performance_record()
        .model_dump(
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


    wrong_model = (
        MLPerformanceEvaluationRecord
        .model_validate(
            payload
        )
    )


    expect_error(
        MLModelHealthAuthorityError,

        lambda:
            summary(
                drift=
                    drift_record(),

                performance=
                    wrong_model,
            ),
    )


# ============================================================
# CONTRACT HARDENING
# ============================================================


def test_summary_forbids_raw_payload(
) -> None:

    valid = (
        summary(
            drift=
                drift_record(),

            performance=
                performance_record(),
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
        MLModelHealthSummary.model_validate(
            payload
        )

    except ValidationError:
        return


    raise AssertionError(
        "Raw prediction payload must be forbidden."
    )


def test_summary_status_tampering_blocked(
) -> None:

    valid = (
        summary(
            drift=
                drift_record(),

            performance=
                performance_record(),
        )
    )


    payload = (
        valid.model_dump(
            mode="json"
        )
    )


    payload[
        "health_status"
    ] = "critical"


    try:
        MLModelHealthSummary.model_validate(
            payload
        )

    except ValidationError:
        return


    raise AssertionError(
        "Health status tampering must be blocked."
    )


def test_summary_is_frozen(
) -> None:

    result = (
        summary()
    )


    try:
        result.health_status = (
            "critical"
        )

    except ValidationError:
        return


    raise AssertionError(
        "Model Health Summary must be frozen."
    )


# ============================================================
# VERSION
# ============================================================


def test_rule_version(
) -> None:

    assert (
        ML_MODEL_HEALTH_RULE_VERSION
        ==
        "ml_model_health_v0.1"
    )


# ============================================================
# MAIN
# ============================================================


def main(
) -> None:

    print(
        (
            "=== DATALENS ML MODEL HEALTH v0.1 ==="
        )
    )

    print()


    tests = [
        (
            "Aligned OK evidence is healthy",
            test_aligned_ok_evidence_is_healthy,
        ),
        (
            "Drift warning requires attention",
            test_warning_drift_is_attention,
        ),
        (
            "Strong Drift with good Performance is not critical",
            test_strong_drift_with_good_performance_is_not_critical,
        ),
        (
            "Performance warning requires attention",
            test_performance_warning_is_attention,
        ),
        (
            "Performance degradation is critical",
            test_performance_degradation_is_critical,
        ),
        (
            "No evidence is insufficient",
            test_no_evidence_is_insufficient,
        ),
        (
            "Drift-only OK is insufficient",
            test_drift_only_ok_is_insufficient,
        ),
        (
            "Performance-only OK is insufficient",
            test_performance_only_ok_is_insufficient,
        ),
        (
            "Drift-only signal requires attention",
            test_drift_only_signal_is_attention,
        ),
        (
            "Performance-only degradation is critical",
            test_performance_only_degraded_is_critical,
        ),
        (
            "Two OK but misaligned snapshots are not healthy",
            test_two_ok_but_misaligned_snapshots_are_not_healthy,
        ),
        (
            "Different observed dataset is misaligned",
            test_different_observed_dataset_is_misaligned,
        ),
        (
            "Legacy Drift revision is unverifiable",
            test_legacy_drift_revision_is_unverifiable,
        ),
        (
            "Misaligned degraded Performance remains critical",
            test_degraded_performance_remains_critical_when_misaligned,
        ),
        (
            "Training fingerprint mismatch blocked",
            test_training_fingerprint_mismatch_blocked,
        ),
        (
            "Cross-model evidence blocked",
            test_cross_model_evidence_blocked,
        ),
        (
            "Raw payload forbidden",
            test_summary_forbids_raw_payload,
        ),
        (
            "Health status tampering blocked",
            test_summary_status_tampering_blocked,
        ),
        (
            "Model Health Summary frozen",
            test_summary_is_frozen,
        ),
        (
            "Model Health rule version",
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
        "PASS - ML Model Health v0.1"
    )


if __name__ == "__main__":
    main()
