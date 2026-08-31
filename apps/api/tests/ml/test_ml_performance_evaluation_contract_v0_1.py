from __future__ import annotations


import math


from app.ml.model_metrics import (
    ML_MODEL_METRICS_RULE_VERSION,
    ml_model_metric_direction,
    ml_model_metric_names,
    ml_model_primary_metric,
)


from app.ml.performance_evaluation import (
    ML_PERFORMANCE_CLASSIFICATION_DEGRADED_DROP,
    ML_PERFORMANCE_CLASSIFICATION_WARNING_DROP,
    ML_PERFORMANCE_EVALUATION_RULE_VERSION,
    ML_PERFORMANCE_REGRESSION_DEGRADED_INCREASE_RATIO,
    ML_PERFORMANCE_REGRESSION_WARNING_INCREASE_RATIO,
    MLPerformanceEvaluationRecord,
    MLPerformanceMetricComparison,
    ml_performance_status_for_primary_metric,
)


# ============================================================
# CONSTANT IDENTITIES
# ============================================================


PERFORMANCE_ID = (
    "performance-evaluation:"
    +
    "a" * 32
)


MODEL_ID = (
    "model:"
    +
    "b" * 32
)


EXPERIMENT_ID = (
    "experiment:"
    +
    "c" * 32
)


TRAINING_SHA = (
    "d" * 64
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


def assert_close(
    actual: float,
    expected: float,
) -> None:

    assert math.isclose(
        actual,
        expected,
        rel_tol=1e-12,
        abs_tol=1e-12,
    )


def comparison(
    *,
    problem_type: str,
    metric_name: str,
    reference_value: float,
    observed_value: float,
) -> MLPerformanceMetricComparison:

    direction = (
        ml_model_metric_direction(
            problem_type=
                problem_type,

            metric_name=
                metric_name,
        )
    )


    delta = (
        observed_value
        -
        reference_value
    )


    if (
        direction
        ==
        "higher_is_better"
    ):
        degradation = max(
            0.0,
            reference_value
            -
            observed_value,
        )

    else:
        degradation = max(
            0.0,
            observed_value
            -
            reference_value,
        )


    return (
        MLPerformanceMetricComparison(
            metric_name=
                metric_name,

            direction=
                direction,

            reference_value=
                reference_value,

            observed_value=
                observed_value,

            delta=
                delta,

            degradation_amount=
                degradation,
        )
    )


def classification_record(
    *,
    observed_f1: float = 0.74,
    status: str = "warning",
) -> MLPerformanceEvaluationRecord:

    reference = {
        "accuracy":
            0.84,

        "f1_macro":
            0.80,

        "precision_macro":
            0.81,

        "recall_macro":
            0.79,

        "balanced_accuracy":
            0.80,
    }


    observed = {
        "accuracy":
            0.80,

        "f1_macro":
            observed_f1,

        "precision_macro":
            0.77,

        "recall_macro":
            0.73,

        "balanced_accuracy":
            0.75,
    }


    results = [
        comparison(
            problem_type=
                "classification",

            metric_name=
                metric_name,

            reference_value=
                reference[
                    metric_name
                ],

            observed_value=
                observed[
                    metric_name
                ],
        )

        for metric_name
        in ml_model_metric_names(
            problem_type=
                "classification"
        )
    ]


    primary = next(
        item
        for item
        in results
        if (
            item.metric_name
            ==
            "f1_macro"
        )
    )


    return (
        MLPerformanceEvaluationRecord(
            performance_evaluation_id=
                PERFORMANCE_ID,

            model_id=
                MODEL_ID,

            workflow_id=
                "prep:performance",

            reference_dataset_id=
                "dataset:training",

            observed_dataset_id=
                "dataset:observed",

            experiment_id=
                EXPERIMENT_ID,

            preparation_session_revision=
                7,

            observed_preparation_session_revision=
                11,

            training_contract_sha256=
                TRAINING_SHA,

            problem_type=
                "classification",

            target_column=
                "target",

            reference_evaluation_row_count=
                20,

            observed_row_count=
                50,

            evaluated_at_utc=
                "2026-08-29T12:00:00+00:00",

            metric_results=
                results,

            primary_metric=
                "f1_macro",

            primary_metric_degradation_amount=(
                primary
                .degradation_amount
            ),

            primary_metric_degradation_ratio=
                None,

            degradation_basis=
                "absolute_points",

            performance_status=
                status,
        )
    )


def regression_record(
    *,
    observed_rmse: float = 13.0,
    status: str = "degraded",
) -> MLPerformanceEvaluationRecord:

    reference = {
        "mae":
            7.0,

        "rmse":
            10.0,

        "r2":
            0.80,

        "median_absolute_error":
            6.0,

        "explained_variance":
            0.82,
    }


    observed = {
        "mae":
            9.0,

        "rmse":
            observed_rmse,

        "r2":
            0.65,

        "median_absolute_error":
            8.0,

        "explained_variance":
            0.68,
    }


    results = [
        comparison(
            problem_type=
                "regression",

            metric_name=
                metric_name,

            reference_value=
                reference[
                    metric_name
                ],

            observed_value=
                observed[
                    metric_name
                ],
        )

        for metric_name
        in ml_model_metric_names(
            problem_type=
                "regression"
        )
    ]


    primary = next(
        item
        for item
        in results
        if (
            item.metric_name
            ==
            "rmse"
        )
    )


    ratio = (
        primary
        .degradation_amount
        /
        primary
        .reference_value
    )


    return (
        MLPerformanceEvaluationRecord(
            performance_evaluation_id=
                PERFORMANCE_ID,

            model_id=
                MODEL_ID,

            workflow_id=
                "prep:performance",

            reference_dataset_id=
                "dataset:training",

            observed_dataset_id=
                "dataset:observed",

            experiment_id=
                EXPERIMENT_ID,

            preparation_session_revision=
                7,

            observed_preparation_session_revision=
                11,

            training_contract_sha256=
                TRAINING_SHA,

            problem_type=
                "regression",

            target_column=
                "target",

            reference_evaluation_row_count=
                20,

            observed_row_count=
                50,

            evaluated_at_utc=
                "2026-08-29T12:00:00+00:00",

            metric_results=
                results,

            primary_metric=
                "rmse",

            primary_metric_degradation_amount=(
                primary
                .degradation_amount
            ),

            primary_metric_degradation_ratio=
                ratio,

            degradation_basis=
                "relative_increase",

            performance_status=
                status,
        )
    )


# ============================================================
# CANONICAL SEMANTICS
# ============================================================


def test_canonical_metric_semantics(
) -> None:

    assert (
        ml_model_primary_metric(
            problem_type=
                "regression"
        )
        ==
        "rmse"
    )


    assert (
        ml_model_primary_metric(
            problem_type=
                "classification"
        )
        ==
        "f1_macro"
    )


    assert (
        ml_model_metric_direction(
            problem_type=
                "regression",

            metric_name=
                "rmse",
        )
        ==
        "lower_is_better"
    )


    assert (
        ml_model_metric_direction(
            problem_type=
                "classification",

            metric_name=
                "f1_macro",
        )
        ==
        "higher_is_better"
    )


# ============================================================
# CLASSIFICATION
# ============================================================


def test_valid_classification_evaluation(
) -> None:

    record = (
        classification_record()
    )


    assert (
        record.primary_metric
        ==
        "f1_macro"
    )


    assert_close(
        record
        .primary_metric_degradation_amount,
        0.06,
    )


    assert (
        record
        .primary_metric_degradation_ratio
        is None
    )


    assert (
        record.degradation_basis
        ==
        "absolute_points"
    )


    assert (
        record.performance_status
        ==
        "warning"
    )


# ============================================================
# REGRESSION
# ============================================================


def test_valid_regression_evaluation(
) -> None:

    record = (
        regression_record()
    )


    assert (
        record.primary_metric
        ==
        "rmse"
    )


    assert_close(
        record
        .primary_metric_degradation_amount,
        3.0,
    )


    assert_close(
        record
        .primary_metric_degradation_ratio,
        0.30,
    )


    assert (
        record.performance_status
        ==
        "degraded"
    )


# ============================================================
# STATUS BOUNDARIES
# ============================================================


def test_classification_status_boundaries(
) -> None:

    assert (
        ML_PERFORMANCE_CLASSIFICATION_WARNING_DROP
        ==
        0.05
    )


    assert (
        ML_PERFORMANCE_CLASSIFICATION_DEGRADED_DROP
        ==
        0.10
    )


    assert (
        ml_performance_status_for_primary_metric(
            problem_type=
                "classification",

            reference_value=
                0.80,

            observed_value=
                0.80,
        )
        ==
        "ok"
    )


    assert (
        ml_performance_status_for_primary_metric(
            problem_type=
                "classification",

            reference_value=
                0.80,

            observed_value=
                0.75,
        )
        ==
        "warning"
    )


    assert (
        ml_performance_status_for_primary_metric(
            problem_type=
                "classification",

            reference_value=
                0.80,

            observed_value=
                0.70,
        )
        ==
        "degraded"
    )


    assert (
        ml_performance_status_for_primary_metric(
            problem_type=
                "classification",

            reference_value=
                0.80,

            observed_value=
                0.85,
        )
        ==
        "ok"
    )


def test_regression_status_boundaries(
) -> None:

    assert (
        ML_PERFORMANCE_REGRESSION_WARNING_INCREASE_RATIO
        ==
        0.10
    )


    assert (
        ML_PERFORMANCE_REGRESSION_DEGRADED_INCREASE_RATIO
        ==
        0.25
    )


    assert (
        ml_performance_status_for_primary_metric(
            problem_type=
                "regression",

            reference_value=
                10.0,

            observed_value=
                10.0,
        )
        ==
        "ok"
    )


    assert (
        ml_performance_status_for_primary_metric(
            problem_type=
                "regression",

            reference_value=
                10.0,

            observed_value=
                11.0,
        )
        ==
        "warning"
    )


    assert (
        ml_performance_status_for_primary_metric(
            problem_type=
                "regression",

            reference_value=
                10.0,

            observed_value=
                12.5,
        )
        ==
        "degraded"
    )


    assert (
        ml_performance_status_for_primary_metric(
            problem_type=
                "regression",

            reference_value=
                10.0,

            observed_value=
                8.0,
        )
        ==
        "ok"
    )


# ============================================================
# ZERO RMSE
# ============================================================


def test_zero_reference_rmse_is_fail_closed(
) -> None:

    assert (
        ml_performance_status_for_primary_metric(
            problem_type=
                "regression",

            reference_value=
                0.0,

            observed_value=
                0.0,
        )
        ==
        "ok"
    )


    assert (
        ml_performance_status_for_primary_metric(
            problem_type=
                "regression",

            reference_value=
                0.0,

            observed_value=
                0.01,
        )
        ==
        "degraded"
    )


# ============================================================
# METRIC CONSISTENCY
# ============================================================


def test_metric_delta_tampering_blocked(
) -> None:

    expect_error(
        ValueError,

        lambda:
            MLPerformanceMetricComparison(
                metric_name=
                    "f1_macro",

                direction=
                    "higher_is_better",

                reference_value=
                    0.8,

                observed_value=
                    0.7,

                delta=
                    0.1,

                degradation_amount=
                    0.1,
            ),
    )


def test_wrong_metric_direction_blocked(
) -> None:

    record = (
        classification_record()
    )


    payload = (
        record.model_dump(
            mode="python"
        )
    )


    payload[
        "metric_results"
    ][
        1
    ][
        "direction"
    ] = "lower_is_better"


    expect_error(
        ValueError,

        lambda:
            MLPerformanceEvaluationRecord
            .model_validate(
                payload
            ),
    )


def test_incomplete_metric_surface_blocked(
) -> None:

    record = (
        regression_record()
    )


    payload = (
        record.model_dump(
            mode="python"
        )
    )


    payload[
        "metric_results"
    ] = (
        payload[
            "metric_results"
        ][
            :-1
        ]
    )


    expect_error(
        ValueError,

        lambda:
            MLPerformanceEvaluationRecord
            .model_validate(
                payload
            ),
    )


def test_primary_metric_tampering_blocked(
) -> None:

    record = (
        classification_record()
    )


    payload = (
        record.model_dump(
            mode="python"
        )
    )


    payload[
        "primary_metric"
    ] = "rmse"


    expect_error(
        ValueError,

        lambda:
            MLPerformanceEvaluationRecord
            .model_validate(
                payload
            ),
    )


def test_status_tampering_blocked(
) -> None:

    record = (
        classification_record()
    )


    payload = (
        record.model_dump(
            mode="python"
        )
    )


    payload[
        "performance_status"
    ] = "ok"


    expect_error(
        ValueError,

        lambda:
            MLPerformanceEvaluationRecord
            .model_validate(
                payload
            ),
    )


# ============================================================
# PRIVACY / AUTHORITY SURFACE
# ============================================================


def test_raw_supervised_payload_forbidden(
) -> None:

    record = (
        classification_record()
    )


    payload = (
        record.model_dump(
            mode="python"
        )
    )


    payload[
        "y_true"
    ] = [
        0,
        1,
    ]


    payload[
        "predictions"
    ] = [
        0,
        1,
    ]


    expect_error(
        ValueError,

        lambda:
            MLPerformanceEvaluationRecord
            .model_validate(
                payload
            ),
    )


def test_server_shaped_identity_required(
) -> None:

    record = (
        classification_record()
    )


    payload = (
        record.model_dump(
            mode="python"
        )
    )


    payload[
        "performance_evaluation_id"
    ] = "client-created"


    expect_error(
        ValueError,

        lambda:
            MLPerformanceEvaluationRecord
            .model_validate(
                payload
            ),
    )


# ============================================================
# IMMUTABILITY
# ============================================================


def test_record_is_frozen(
) -> None:

    record = (
        classification_record()
    )


    expect_error(
        Exception,

        lambda:
            setattr(
                record,
                "performance_status",
                "ok",
            ),
    )


# ============================================================
# VERSIONS
# ============================================================


def test_rule_versions(
) -> None:

    assert (
        ML_MODEL_METRICS_RULE_VERSION
        ==
        "ml_model_metrics_v0.1"
    )


    assert (
        ML_PERFORMANCE_EVALUATION_RULE_VERSION
        ==
        "ml_performance_evaluation_v0.1"
    )


# ============================================================
# MAIN
# ============================================================


def main(
) -> None:

    print(
        (
            "=== DATALENS ML PERFORMANCE "
            "EVALUATION CONTRACT v0.1 ==="
        )
    )

    print()


    tests = [
        (
            "Canonical metric semantics",
            test_canonical_metric_semantics,
        ),
        (
            "Valid classification performance evaluation",
            test_valid_classification_evaluation,
        ),
        (
            "Valid regression performance evaluation",
            test_valid_regression_evaluation,
        ),
        (
            "Classification status boundaries",
            test_classification_status_boundaries,
        ),
        (
            "Regression status boundaries",
            test_regression_status_boundaries,
        ),
        (
            "Zero reference RMSE is fail-closed",
            test_zero_reference_rmse_is_fail_closed,
        ),
        (
            "Metric delta tampering blocked",
            test_metric_delta_tampering_blocked,
        ),
        (
            "Wrong metric direction blocked",
            test_wrong_metric_direction_blocked,
        ),
        (
            "Incomplete metric surface blocked",
            test_incomplete_metric_surface_blocked,
        ),
        (
            "Primary metric tampering blocked",
            test_primary_metric_tampering_blocked,
        ),
        (
            "Performance status tampering blocked",
            test_status_tampering_blocked,
        ),
        (
            "Raw supervised payload forbidden",
            test_raw_supervised_payload_forbidden,
        ),
        (
            "Server-shaped evaluation identity required",
            test_server_shaped_identity_required,
        ),
        (
            "Performance record frozen",
            test_record_is_frozen,
        ),
        (
            "Performance rule versions",
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
            "PASS - ML Performance "
            "Evaluation Contract v0.1"
        )
    )


if __name__ == "__main__":
    main()
