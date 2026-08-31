from __future__ import annotations

from pydantic import (
    ValidationError,
)

from app.ml.drift_evaluation import (
    MLCategoricalDriftFeatureResult,
    MLDriftEvaluationRecord,
    MLNumericDriftFeatureResult,
    ML_DRIFT_EVALUATION_RULE_VERSION,
)


# ============================================================
# HELPERS
# ============================================================


def numeric_result(
) -> MLNumericDriftFeatureResult:

    return (
        MLNumericDriftFeatureResult(
            feature_name="age",
            reference_missing_rate=0.0,
            observed_total_count=10,
            observed_non_missing_count=10,
            observed_missing_count=0,
            observed_missing_rate=0.0,
            missing_rate_delta=0.0,
            absolute_missing_rate_delta=0.0,
            population_stability_index=0.0,
            distribution_status="ok",
            missingness_status="ok",
            outside_reference_range_count=0,
            outside_reference_range_rate=0.0,
            range_status="ok",
            status="ok",
        )
    )


def categorical_result(
) -> MLCategoricalDriftFeatureResult:

    return (
        MLCategoricalDriftFeatureResult(
            feature_name="segment",
            reference_missing_rate=0.0,
            observed_total_count=10,
            observed_non_missing_count=10,
            observed_missing_count=0,
            observed_missing_rate=0.0,
            missing_rate_delta=0.0,
            absolute_missing_rate_delta=0.0,
            population_stability_index=0.0,
            distribution_status="ok",
            missingness_status="ok",
            reference_other_rate=0.2,
            observed_untracked_count=2,
            observed_untracked_rate=0.2,
            untracked_rate_delta=0.0,
            absolute_untracked_rate_delta=0.0,
            status="ok",
        )
    )


def record(
) -> MLDriftEvaluationRecord:

    return (
        MLDriftEvaluationRecord(
            evaluation_id=(
                "drift-evaluation:"
                +
                ("1" * 32)
            ),
            profile_id=(
                "monitoring-profile:"
                +
                ("2" * 32)
            ),
            model_id=(
                "model:"
                +
                ("3" * 32)
            ),
            workflow_id="prep:drift-contract",
            reference_dataset_id="dataset:reference",
            observed_dataset_id="dataset:observed",
            experiment_id=(
                "experiment:"
                +
                ("4" * 32)
            ),
            preparation_session_revision=7,
            training_contract_sha256=(
                "a" * 64
            ),
            evaluated_at_utc=(
                "2026-08-29T13:00:00+00:00"
            ),
            observed_row_count=10,
            feature_results=[
                numeric_result(),
                categorical_result(),
            ],
            warning_feature_count=0,
            drift_feature_count=0,
            overall_status="ok",
        )
    )


def expect_validation_error(
    factory,
) -> None:

    try:
        factory()

    except ValidationError:
        return

    raise AssertionError(
        "Expected ValidationError."
    )


# ============================================================
# TESTS
# ============================================================


def test_valid_numeric_result(
) -> None:
    result = numeric_result()

    assert result.kind == "numeric"
    assert result.status == "ok"


def test_valid_categorical_result(
) -> None:
    result = categorical_result()

    assert result.kind == "categorical"
    assert (
        result.observed_untracked_rate
        ==
        0.2
    )


def test_invalid_numeric_count_consistency(
) -> None:

    payload = (
        numeric_result()
        .model_dump(
            mode="json"
        )
    )

    payload[
        "observed_missing_count"
    ] = 1

    expect_validation_error(
        lambda: (
            MLNumericDriftFeatureResult
            .model_validate(
                payload
            )
        )
    )


def test_invalid_status_policy(
) -> None:

    payload = (
        numeric_result()
        .model_dump(
            mode="json"
        )
    )

    payload[
        "population_stability_index"
    ] = 0.30

    payload[
        "distribution_status"
    ] = "ok"

    expect_validation_error(
        lambda: (
            MLNumericDriftFeatureResult
            .model_validate(
                payload
            )
        )
    )


def test_all_missing_requires_not_evaluable_psi(
) -> None:

    result = (
        MLNumericDriftFeatureResult(
            feature_name="age",
            reference_missing_rate=0.0,
            observed_total_count=10,
            observed_non_missing_count=0,
            observed_missing_count=10,
            observed_missing_rate=1.0,
            missing_rate_delta=1.0,
            absolute_missing_rate_delta=1.0,
            population_stability_index=None,
            distribution_status="not_evaluable",
            missingness_status="drift",
            outside_reference_range_count=0,
            outside_reference_range_rate=0.0,
            range_status="ok",
            status="drift",
        )
    )

    assert result.status == "drift"


def test_raw_payload_surface_forbidden(
) -> None:

    payload = (
        numeric_result()
        .model_dump(
            mode="json"
        )
    )

    payload[
        "raw_values"
    ] = [
        1.0,
        2.0,
    ]

    expect_validation_error(
        lambda: (
            MLNumericDriftFeatureResult
            .model_validate(
                payload
            )
        )
    )


def test_valid_record(
) -> None:
    evaluation = record()

    assert evaluation.overall_status == "ok"
    assert evaluation.privacy_scope == "aggregate_only"


def test_record_status_count_consistency(
) -> None:

    payload = (
        record()
        .model_dump(
            mode="json"
        )
    )

    payload[
        "drift_feature_count"
    ] = 1

    payload[
        "overall_status"
    ] = "drift"

    expect_validation_error(
        lambda: (
            MLDriftEvaluationRecord
            .model_validate(
                payload
            )
        )
    )


def test_record_is_frozen(
) -> None:

    evaluation = record()

    expect_validation_error(
        lambda: setattr(
            evaluation,
            "overall_status",
            "drift",
        )
    )


def test_rule_version(
) -> None:

    assert (
        ML_DRIFT_EVALUATION_RULE_VERSION
        ==
        "ml_drift_evaluation_v0.1"
    )


# ============================================================
# MAIN
# ============================================================


def main(
) -> None:

    print(
        "=== DATALENS ML DRIFT EVALUATION CONTRACT v0.1 ==="
    )
    print()

    test_valid_numeric_result()
    print("[PASS] Valid numeric drift result")

    test_valid_categorical_result()
    print("[PASS] Valid categorical drift result")

    test_invalid_numeric_count_consistency()
    print("[PASS] Numeric observed count consistency")

    test_invalid_status_policy()
    print("[PASS] Drift status policy consistency")

    test_all_missing_requires_not_evaluable_psi()
    print("[PASS] All-missing distribution is fail-closed")

    test_raw_payload_surface_forbidden()
    print("[PASS] Raw drift payload surface forbidden")

    test_valid_record()
    print("[PASS] Valid drift evaluation record")

    test_record_status_count_consistency()
    print("[PASS] Evaluation aggregate counts consistent")

    test_record_is_frozen()
    print("[PASS] Drift evaluation record frozen")

    test_rule_version()
    print("[PASS] Drift evaluation rule version")

    print()
    print(
        "PASS - ML Drift Evaluation Contract v0.1"
    )


if __name__ == "__main__":
    main()
