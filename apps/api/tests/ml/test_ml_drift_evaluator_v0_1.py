from __future__ import annotations

import copy
import math

import numpy as np
import pandas as pd

from app.ml.drift_evaluation import (
    MLCategoricalDriftFeatureResult,
    MLNumericDriftFeatureResult,
)

from app.ml.drift_evaluator import (
    ML_DRIFT_EVALUATOR_RULE_VERSION,
    MLDriftEvaluatorError,
    evaluate_ml_drift,
)

from app.ml.monitoring_profile import (
    MLMonitoringProfileRecord,
)

from app.ml.monitoring_profile_builder import (
    build_ml_monitoring_profile,
)

from tests.ml.test_ml_monitoring_profile_builder_v0_1 import (
    mixed_training_frame,
    model_artifact,
    training_contract,
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


def authority_fixture(
):
    frame = (
        mixed_training_frame()
    )

    contract = (
        training_contract()
    )

    artifact = (
        model_artifact(
            contract=contract,
            train_rows=len(frame),
        )
    )

    profile = (
        build_ml_monitoring_profile(
            x_train=frame,
            model_artifact=artifact,
        )
    )

    return (
        frame,
        artifact,
        profile,
    )


def feature_result(
    evaluation,
    feature_name: str,
):
    for result in (
        evaluation.feature_results
    ):
        if (
            result.feature_name
            ==
            feature_name
        ):
            return result

    raise AssertionError(
        (
            "Missing feature result: "
            f"{feature_name}"
        )
    )


# ============================================================
# IDENTICAL DISTRIBUTION
# ============================================================


def test_identical_reference_is_ok(
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
                "dataset:observed-same"
            ),
            monitoring_profile=profile,
            model_artifact=artifact,
        )
    )

    assert (
        evaluation.overall_status
        ==
        "ok"
    )

    assert (
        evaluation.drift_feature_count
        ==
        0
    )

    for result in (
        evaluation.feature_results
    ):
        assert (
            result.status
            ==
            "ok"
        )

        assert (
            result.population_stability_index
            is not None
        )

        assert math.isclose(
            result.population_stability_index,
            0.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        )


# ============================================================
# NUMERIC SHIFT
# ============================================================


def test_numeric_shift_is_detected(
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
        1000.0
    )

    evaluation = (
        evaluate_ml_drift(
            observed_features=observed,
            observed_dataset_id=(
                "dataset:numeric-shift"
            ),
            monitoring_profile=profile,
            model_artifact=artifact,
        )
    )

    age = feature_result(
        evaluation,
        "age",
    )

    assert isinstance(
        age,
        MLNumericDriftFeatureResult,
    )

    assert age.status == "drift"

    assert (
        age.outside_reference_range_rate
        >
        0.9
    )

    assert (
        evaluation.overall_status
        ==
        "drift"
    )


# ============================================================
# CONSTANT FEATURE
# ============================================================


def test_constant_numeric_reference_shift_is_detected(
) -> None:

    contract = (
        training_contract(
            feature_columns=[
                "constant"
            ],
            categorical_feature_columns=[],
        )
    )

    reference_frame = (
        pd.DataFrame(
            {
                "constant": [
                    5.0
                    for _
                    in range(10)
                ]
            }
        )
    )

    artifact = (
        model_artifact(
            contract=contract,
            train_rows=10,
        )
    )

    profile = (
        build_ml_monitoring_profile(
            x_train=reference_frame,
            model_artifact=artifact,
        )
    )

    observed = (
        pd.DataFrame(
            {
                "constant": [
                    99.0
                    for _
                    in range(10)
                ]
            }
        )
    )

    evaluation = (
        evaluate_ml_drift(
            observed_features=observed,
            observed_dataset_id=(
                "dataset:constant-shift"
            ),
            monitoring_profile=profile,
            model_artifact=artifact,
        )
    )

    result = feature_result(
        evaluation,
        "constant",
    )

    assert isinstance(
        result,
        MLNumericDriftFeatureResult,
    )

    # One histogram bucket alone cannot detect this shift.
    assert math.isclose(
        result.population_stability_index,
        0.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    )

    # The explicit reference-range signal closes that gap.
    assert (
        result.outside_reference_range_rate
        ==
        1.0
    )

    assert result.range_status == "drift"
    assert result.status == "drift"


# ============================================================
# CATEGORICAL SHIFT
# ============================================================


def test_categorical_untracked_shift_is_detected(
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
        "brand-new-segment"
        for _
        in range(
            len(observed)
        )
    ]

    evaluation = (
        evaluate_ml_drift(
            observed_features=observed,
            observed_dataset_id=(
                "dataset:categorical-shift"
            ),
            monitoring_profile=profile,
            model_artifact=artifact,
        )
    )

    segment = feature_result(
        evaluation,
        "segment",
    )

    assert isinstance(
        segment,
        MLCategoricalDriftFeatureResult,
    )

    assert (
        segment.observed_untracked_rate
        ==
        1.0
    )

    assert (
        segment.distribution_status
        ==
        "drift"
    )

    assert segment.status == "drift"


# ============================================================
# ALL MISSING
# ============================================================


def test_all_missing_numeric_is_fail_closed_drift(
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
    ] = np.nan

    evaluation = (
        evaluate_ml_drift(
            observed_features=observed,
            observed_dataset_id=(
                "dataset:all-missing"
            ),
            monitoring_profile=profile,
            model_artifact=artifact,
        )
    )

    age = feature_result(
        evaluation,
        "age",
    )

    assert (
        age.population_stability_index
        is None
    )

    assert (
        age.distribution_status
        ==
        "not_evaluable"
    )

    assert age.status == "drift"


# ============================================================
# AUTHORITY BINDING
# ============================================================


def test_wrong_model_profile_binding_is_blocked(
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
        "model_id"
    ] = (
        "model:"
        +
        ("2" * 32)
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
                    "dataset:wrong-binding"
                ),
                monitoring_profile=(
                    wrong_profile
                ),
                model_artifact=artifact,
            )
        )
    )


# ============================================================
# FEATURE SURFACE
# ============================================================


def test_wrong_feature_order_is_blocked(
) -> None:

    (
        frame,
        artifact,
        profile,
    ) = authority_fixture()

    observed = (
        frame[
            [
                "income",
                "age",
                "segment",
            ]
        ]
        .copy(
            deep=True
        )
    )

    expect_evaluator_error(
        lambda: (
            evaluate_ml_drift(
                observed_features=observed,
                observed_dataset_id=(
                    "dataset:wrong-order"
                ),
                monitoring_profile=profile,
                model_artifact=artifact,
            )
        )
    )


def test_mixed_categorical_families_are_blocked(
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
        "standard",
        1,
        "standard",
        1,
        "premium",
        None,
        "standard",
        1,
        "premium",
        "standard",
    ]

    expect_evaluator_error(
        lambda: (
            evaluate_ml_drift(
                observed_features=observed,
                observed_dataset_id=(
                    "dataset:mixed-category"
                ),
                monitoring_profile=profile,
                model_artifact=artifact,
            )
        )
    )


# ============================================================
# IMMUTABILITY / PRIVACY
# ============================================================


def test_observed_frame_remains_immutable(
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

    before = copy.deepcopy(
        observed
    )

    evaluate_ml_drift(
        observed_features=observed,
        observed_dataset_id=(
            "dataset:immutable"
        ),
        monitoring_profile=profile,
        model_artifact=artifact,
    )

    pd.testing.assert_frame_equal(
        observed,
        before,
    )


def test_raw_categorical_label_absent_from_result(
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

    secret_label = (
        "secret-customer-segment"
    )

    observed[
        "segment"
    ] = [
        secret_label
        for _
        in range(
            len(observed)
        )
    ]

    evaluation = (
        evaluate_ml_drift(
            observed_features=observed,
            observed_dataset_id=(
                "dataset:privacy"
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

    assert (
        secret_label
        not in
        payload
    )

    assert (
        "raw_values"
        not in
        payload
    )

    # Deliberately named untracked, not unseen:
    # the reference `other` bucket does not retain every
    # historical category identity.
    segment = feature_result(
        evaluation,
        "segment",
    )

    assert hasattr(
        segment,
        "observed_untracked_rate",
    )

    assert not hasattr(
        segment,
        "unseen_rate",
    )


# ============================================================
# RULE VERSION
# ============================================================


def test_rule_version(
) -> None:

    assert (
        ML_DRIFT_EVALUATOR_RULE_VERSION
        ==
        "ml_drift_evaluator_v0.1"
    )


# ============================================================
# MAIN
# ============================================================


def main(
) -> None:

    print(
        "=== DATALENS ML DRIFT EVALUATOR v0.1 ==="
    )
    print()

    test_identical_reference_is_ok()
    print("[PASS] Identical reference distribution")

    test_numeric_shift_is_detected()
    print("[PASS] Numeric distribution / range drift")

    test_constant_numeric_reference_shift_is_detected()
    print("[PASS] Constant numeric drift gap closed")

    test_categorical_untracked_shift_is_detected()
    print("[PASS] Categorical untracked distribution drift")

    test_all_missing_numeric_is_fail_closed_drift()
    print("[PASS] All-missing numeric drift fail-closed")

    test_wrong_model_profile_binding_is_blocked()
    print("[PASS] Model / Monitoring Profile authority binding")

    test_wrong_feature_order_is_blocked()
    print("[PASS] Ordered feature surface binding")

    test_mixed_categorical_families_are_blocked()
    print("[PASS] Mixed categorical families blocked")

    test_observed_frame_remains_immutable()
    print("[PASS] Observed frame remains immutable")

    test_raw_categorical_label_absent_from_result()
    print("[PASS] Aggregate-only privacy surface")

    test_rule_version()
    print("[PASS] Drift Evaluator rule version")

    print()
    print(
        "PASS - ML Drift Evaluator v0.1"
    )


if __name__ == "__main__":
    main()
