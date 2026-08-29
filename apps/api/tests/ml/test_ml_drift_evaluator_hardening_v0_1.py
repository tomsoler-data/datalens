from __future__ import annotations

import math

import numpy as np
import pandas as pd

from app.ml.drift_evaluation import (
    ML_DRIFT_MISSING_RATE_DRIFT_THRESHOLD,
    ML_DRIFT_MISSING_RATE_WARNING_THRESHOLD,
    ML_DRIFT_PSI_DRIFT_THRESHOLD,
    ML_DRIFT_PSI_WARNING_THRESHOLD,
    ml_drift_status_for_psi,
    ml_drift_status_for_rate_shift,
)

from app.ml.drift_evaluator import (
    MLDriftEvaluatorError,
    evaluate_ml_drift,
)

from app.ml.model_artifacts import (
    MLModelArtifactRecord,
)

from app.ml.monitoring_profile import (
    MLMonitoringProfileRecord,
)

from tests.ml.test_ml_drift_evaluator_v0_1 import (
    authority_fixture,
    feature_result,
)


# ============================================================
# HELPERS
# ============================================================


def expect_evaluator_error(
    factory,
) -> None:

    try:
        factory()

    except MLDriftEvaluatorError:
        return

    raise AssertionError(
        "Expected MLDriftEvaluatorError."
    )


# ============================================================
# POLICY BOUNDARIES
# ============================================================


def test_psi_threshold_boundaries_are_closed(
) -> None:

    assert (
        ml_drift_status_for_psi(
            ML_DRIFT_PSI_WARNING_THRESHOLD
            -
            1e-9
        )
        ==
        "ok"
    )

    assert (
        ml_drift_status_for_psi(
            ML_DRIFT_PSI_WARNING_THRESHOLD
        )
        ==
        "warning"
    )

    assert (
        ml_drift_status_for_psi(
            ML_DRIFT_PSI_DRIFT_THRESHOLD
            -
            1e-9
        )
        ==
        "warning"
    )

    assert (
        ml_drift_status_for_psi(
            ML_DRIFT_PSI_DRIFT_THRESHOLD
        )
        ==
        "drift"
    )


def test_missing_rate_threshold_boundaries_are_closed(
) -> None:

    assert (
        ml_drift_status_for_rate_shift(
            (
                ML_DRIFT_MISSING_RATE_WARNING_THRESHOLD
                -
                1e-9
            ),
            warning_threshold=(
                ML_DRIFT_MISSING_RATE_WARNING_THRESHOLD
            ),
            drift_threshold=(
                ML_DRIFT_MISSING_RATE_DRIFT_THRESHOLD
            ),
        )
        ==
        "ok"
    )

    assert (
        ml_drift_status_for_rate_shift(
            ML_DRIFT_MISSING_RATE_WARNING_THRESHOLD,
            warning_threshold=(
                ML_DRIFT_MISSING_RATE_WARNING_THRESHOLD
            ),
            drift_threshold=(
                ML_DRIFT_MISSING_RATE_DRIFT_THRESHOLD
            ),
        )
        ==
        "warning"
    )

    assert (
        ml_drift_status_for_rate_shift(
            ML_DRIFT_MISSING_RATE_DRIFT_THRESHOLD,
            warning_threshold=(
                ML_DRIFT_MISSING_RATE_WARNING_THRESHOLD
            ),
            drift_threshold=(
                ML_DRIFT_MISSING_RATE_DRIFT_THRESHOLD
            ),
        )
        ==
        "drift"
    )


# ============================================================
# INPUT SURFACE
# ============================================================


def test_non_dataframe_input_is_blocked(
) -> None:

    (
        _,
        artifact,
        profile,
    ) = authority_fixture()

    expect_evaluator_error(
        lambda: (
            evaluate_ml_drift(
                observed_features=[
                    {
                        "age": 20.0,
                        "income": 1000.0,
                        "segment": "standard",
                    }
                ],
                observed_dataset_id=(
                    "dataset:not-dataframe"
                ),
                monitoring_profile=profile,
                model_artifact=artifact,
            )
        )
    )


def test_empty_dataframe_is_blocked(
) -> None:

    (
        frame,
        artifact,
        profile,
    ) = authority_fixture()

    empty = (
        frame.iloc[
            0:0
        ]
        .copy(
            deep=True
        )
    )

    expect_evaluator_error(
        lambda: (
            evaluate_ml_drift(
                observed_features=empty,
                observed_dataset_id=(
                    "dataset:empty"
                ),
                monitoring_profile=profile,
                model_artifact=artifact,
            )
        )
    )


def test_empty_observed_dataset_id_is_blocked(
) -> None:

    (
        frame,
        artifact,
        profile,
    ) = authority_fixture()

    expect_evaluator_error(
        lambda: (
            evaluate_ml_drift(
                observed_features=frame,
                observed_dataset_id="   ",
                monitoring_profile=profile,
                model_artifact=artifact,
            )
        )
    )


def test_observed_dataset_id_is_normalized(
) -> None:

    (
        frame,
        artifact,
        profile,
    ) = authority_fixture()

    evaluation = (
        evaluate_ml_drift(
            observed_features=frame,
            observed_dataset_id=(
                "  dataset:normalized  "
            ),
            monitoring_profile=profile,
            model_artifact=artifact,
        )
    )

    assert (
        evaluation.observed_dataset_id
        ==
        "dataset:normalized"
    )


# ============================================================
# NON-FINITE NUMERIC
# ============================================================


def test_positive_infinity_numeric_is_blocked(
) -> None:

    (
        frame,
        artifact,
        profile,
    ) = authority_fixture()

    observed = (
        frame.copy(
            deep=True
        )
    )

    observed.loc[
        observed.index[
            0
        ],
        "age",
    ] = np.inf

    expect_evaluator_error(
        lambda: (
            evaluate_ml_drift(
                observed_features=observed,
                observed_dataset_id=(
                    "dataset:positive-infinity"
                ),
                monitoring_profile=profile,
                model_artifact=artifact,
            )
        )
    )


def test_negative_infinity_numeric_is_blocked(
) -> None:

    (
        frame,
        artifact,
        profile,
    ) = authority_fixture()

    observed = (
        frame.copy(
            deep=True
        )
    )

    observed.loc[
        observed.index[
            0
        ],
        "income",
    ] = -np.inf

    expect_evaluator_error(
        lambda: (
            evaluate_ml_drift(
                observed_features=observed,
                observed_dataset_id=(
                    "dataset:negative-infinity"
                ),
                monitoring_profile=profile,
                model_artifact=artifact,
            )
        )
    )


# ============================================================
# CATEGORICAL HARDENING
# ============================================================


def test_unsupported_categorical_family_is_blocked(
) -> None:

    (
        frame,
        artifact,
        profile,
    ) = authority_fixture()

    observed = (
        frame.copy(
            deep=True
        )
    )

    observed[
        "segment"
    ] = [
        {
            "private": index
        }
        for index
        in range(
            len(observed)
        )
    ]

    expect_evaluator_error(
        lambda: (
            evaluate_ml_drift(
                observed_features=observed,
                observed_dataset_id=(
                    "dataset:unsupported-category"
                ),
                monitoring_profile=profile,
                model_artifact=artifact,
            )
        )
    )


def test_all_missing_categorical_is_fail_closed(
) -> None:

    (
        frame,
        artifact,
        profile,
    ) = authority_fixture()

    observed = (
        frame.copy(
            deep=True
        )
    )

    observed[
        "segment"
    ] = None

    evaluation = (
        evaluate_ml_drift(
            observed_features=observed,
            observed_dataset_id=(
                "dataset:all-missing-category"
            ),
            monitoring_profile=profile,
            model_artifact=artifact,
        )
    )

    segment = feature_result(
        evaluation,
        "segment",
    )

    assert (
        segment.population_stability_index
        is None
    )

    assert (
        segment.distribution_status
        ==
        "not_evaluable"
    )

    assert (
        segment.observed_non_missing_count
        ==
        0
    )

    assert (
        segment.status
        ==
        "drift"
    )

    assert (
        evaluation.overall_status
        ==
        "drift"
    )


# ============================================================
# PROFILE / MODEL AUTHORITY
# ============================================================


def test_wrong_experiment_binding_is_blocked(
) -> None:

    (
        frame,
        artifact,
        profile,
    ) = authority_fixture()

    payload = (
        profile.model_dump(
            mode="json"
        )
    )

    payload[
        "experiment_id"
    ] = (
        "experiment:"
        +
        ("9" * 32)
    )

    wrong_profile = (
        MLMonitoringProfileRecord
        .model_validate(
            payload
        )
    )

    expect_evaluator_error(
        lambda: (
            evaluate_ml_drift(
                observed_features=frame,
                observed_dataset_id=(
                    "dataset:wrong-experiment"
                ),
                monitoring_profile=(
                    wrong_profile
                ),
                model_artifact=artifact,
            )
        )
    )


def test_wrong_training_fingerprint_is_blocked(
) -> None:

    (
        frame,
        artifact,
        profile,
    ) = authority_fixture()

    payload = (
        profile.model_dump(
            mode="json"
        )
    )

    payload[
        "training_contract_sha256"
    ] = (
        "b"
        *
        64
    )

    wrong_profile = (
        MLMonitoringProfileRecord
        .model_validate(
            payload
        )
    )

    expect_evaluator_error(
        lambda: (
            evaluate_ml_drift(
                observed_features=frame,
                observed_dataset_id=(
                    "dataset:wrong-fingerprint"
                ),
                monitoring_profile=(
                    wrong_profile
                ),
                model_artifact=artifact,
            )
        )
    )


def test_model_without_provenance_is_blocked(
) -> None:

    (
        frame,
        artifact,
        profile,
    ) = authority_fixture()

    payload = (
        artifact.model_dump(
            mode="json"
        )
    )

    payload[
        "experiment_provenance"
    ] = None

    legacy_artifact = (
        MLModelArtifactRecord
        .model_validate(
            payload
        )
    )

    expect_evaluator_error(
        lambda: (
            evaluate_ml_drift(
                observed_features=frame,
                observed_dataset_id=(
                    "dataset:no-provenance"
                ),
                monitoring_profile=profile,
                model_artifact=(
                    legacy_artifact
                ),
            )
        )
    )


# ============================================================
# DETERMINISM
# ============================================================


def test_drift_evidence_is_deterministic(
) -> None:

    (
        frame,
        artifact,
        profile,
    ) = authority_fixture()

    observed = (
        frame.copy(
            deep=True
        )
    )

    observed[
        "age"
    ] = (
        observed[
            "age"
        ]
        +
        15.0
    )

    first = (
        evaluate_ml_drift(
            observed_features=observed,
            observed_dataset_id=(
                "dataset:deterministic"
            ),
            monitoring_profile=profile,
            model_artifact=artifact,
        )
    )

    second = (
        evaluate_ml_drift(
            observed_features=observed,
            observed_dataset_id=(
                "dataset:deterministic"
            ),
            monitoring_profile=profile,
            model_artifact=artifact,
        )
    )

    first_features = [
        feature.model_dump(
            mode="json"
        )
        for feature
        in first.feature_results
    ]

    second_features = [
        feature.model_dump(
            mode="json"
        )
        for feature
        in second.feature_results
    ]

    assert (
        first_features
        ==
        second_features
    )

    assert (
        first.warning_feature_count
        ==
        second.warning_feature_count
    )

    assert (
        first.drift_feature_count
        ==
        second.drift_feature_count
    )

    assert (
        first.overall_status
        ==
        second.overall_status
    )

    # Identity and audit timestamp remain server-owned.
    assert (
        first.evaluation_id
        !=
        second.evaluation_id
    )


# ============================================================
# PSI FINITENESS
# ============================================================


def test_extreme_distribution_keeps_psi_finite(
) -> None:

    (
        frame,
        artifact,
        profile,
    ) = authority_fixture()

    observed = (
        frame.copy(
            deep=True
        )
    )

    observed[
        "income"
    ] = [
        10_000_000.0
        for _
        in range(
            len(observed)
        )
    ]

    evaluation = (
        evaluate_ml_drift(
            observed_features=observed,
            observed_dataset_id=(
                "dataset:extreme-distribution"
            ),
            monitoring_profile=profile,
            model_artifact=artifact,
        )
    )

    income = feature_result(
        evaluation,
        "income",
    )

    assert (
        income.population_stability_index
        is not None
    )

    assert math.isfinite(
        income.population_stability_index
    )

    assert (
        income.population_stability_index
        >
        0.0
    )

    assert (
        income.status
        ==
        "drift"
    )


# ============================================================
# PRIVACY SURFACE
# ============================================================


def test_no_raw_or_model_payload_surface(
) -> None:

    (
        frame,
        artifact,
        profile,
    ) = authority_fixture()

    secret = (
        "private-segment-hardening"
    )

    observed = (
        frame.copy(
            deep=True
        )
    )

    observed[
        "segment"
    ] = [
        secret
        for _
        in range(
            len(observed)
        )
    ]

    evaluation = (
        evaluate_ml_drift(
            observed_features=observed,
            observed_dataset_id=(
                "dataset:privacy-hardening"
            ),
            monitoring_profile=profile,
            model_artifact=artifact,
        )
    )

    payload = str(
        evaluation.model_dump(
            mode="json"
        )
    )

    forbidden = [
        secret,
        "model_bytes",
        "model_path",
        "predictions",
        "dataframe",
        "raw_rows",
        "raw_values",
        "x_train",
        "x_test",
        "y_train",
        "y_test",
    ]

    for value in forbidden:
        assert (
            value
            not in
            payload
        )


# ============================================================
# MAIN
# ============================================================


def main(
) -> None:

    print(
        "=== DATALENS ML DRIFT EVALUATOR HARDENING v0.1 ==="
    )
    print()

    test_psi_threshold_boundaries_are_closed()
    print("[PASS] PSI policy boundaries")

    test_missing_rate_threshold_boundaries_are_closed()
    print("[PASS] Missing-rate policy boundaries")

    test_non_dataframe_input_is_blocked()
    print("[PASS] Non-DataFrame input blocked")

    test_empty_dataframe_is_blocked()
    print("[PASS] Empty observed frame blocked")

    test_empty_observed_dataset_id_is_blocked()
    print("[PASS] Empty observed dataset identity blocked")

    test_observed_dataset_id_is_normalized()
    print("[PASS] Observed dataset identity normalized")

    test_positive_infinity_numeric_is_blocked()
    print("[PASS] Positive infinity numeric blocked")

    test_negative_infinity_numeric_is_blocked()
    print("[PASS] Negative infinity numeric blocked")

    test_unsupported_categorical_family_is_blocked()
    print("[PASS] Unsupported categorical family blocked")

    test_all_missing_categorical_is_fail_closed()
    print("[PASS] All-missing categorical fail-closed")

    test_wrong_experiment_binding_is_blocked()
    print("[PASS] Experiment authority binding")

    test_wrong_training_fingerprint_is_blocked()
    print("[PASS] Training fingerprint authority binding")

    test_model_without_provenance_is_blocked()
    print("[PASS] Missing Experiment Provenance blocked")

    test_drift_evidence_is_deterministic()
    print("[PASS] Deterministic aggregate drift evidence")

    test_extreme_distribution_keeps_psi_finite()
    print("[PASS] Extreme distribution PSI remains finite")

    test_no_raw_or_model_payload_surface()
    print("[PASS] Hardened privacy-minimal result surface")

    print()
    print(
        "PASS - ML Drift Evaluator Hardening v0.1"
    )


if __name__ == "__main__":
    main()
