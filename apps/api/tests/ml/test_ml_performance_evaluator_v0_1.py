from __future__ import annotations


import copy
import math


import numpy as np
import pandas as pd


from app.ml.contracts import (
    MLTrainingContract,
)


from app.ml.model_artifact_store import (
    register_ml_model_artifact,
)


from app.ml.model_loader import (
    LoadedMLModel,
)


from app.ml.performance_evaluator import (
    ML_PERFORMANCE_EVALUATOR_RULE_VERSION,
    MLPerformanceEvaluatorAuthorityError,
    MLPerformanceEvaluatorInputError,
    MLPerformanceMetricsError,
    MLPerformancePredictionError,
    MLPerformanceTargetError,
    evaluate_ml_performance,
)


from tests.ml.test_ml_monitoring_service_real_handoff_v0_1 import (
    build_ready_preparation_workflow,
    isolated_real_handoff_environment,
)


# ============================================================
# ESTIMATOR
# ============================================================


class StaticEstimator:

    def __init__(
        self,
        predictions,
    ) -> None:

        self.predictions = list(
            predictions
        )


        self.call_count = 0


        self.last_columns = None


    def predict(
        self,
        features,
    ):

        self.call_count += 1


        self.last_columns = list(
            features.columns
        )


        return np.asarray(
            self.predictions
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


def classification_model(
    *,
    workflow_id: str,
    dataset_id: str,
    revision: int,
    predictions,
    metrics: dict | None = None,
):

    contract = (
        MLTrainingContract(
            workflow_id=
                workflow_id,

            dataset_id=
                dataset_id,

            problem_type=
                "classification",

            target_column=
                "target",

            feature_columns=[
                "age",
                "segment",
            ],

            categorical_feature_columns=[
                "segment"
            ],

            estimator_key=
                "logistic_regression",
        )
    )


    reference_metrics = (
        metrics

        if metrics is not None

        else {
            "accuracy":
                0.80,

            "f1_macro":
                0.80,

            "precision_macro":
                0.80,

            "recall_macro":
                0.80,

            "balanced_accuracy":
                0.80,
        }
    )


    artifact = (
        register_ml_model_artifact(
            training_contract=
                contract,

            metrics=
                reference_metrics,

            train_rows=
                6,

            test_rows=
                2,

            model_bytes=
                b"PERFORMANCE-CLASSIFIER",

            preparation_session_revision=
                revision,

            created_at_utc=
                "2026-08-29T20:00:00+00:00",
        )
    )


    estimator = (
        StaticEstimator(
            predictions
        )
    )


    return (
        LoadedMLModel(
            artifact=
                artifact,

            estimator=
                estimator,
        ),
        estimator,
    )


def regression_model(
    *,
    workflow_id: str,
    dataset_id: str,
    revision: int,
    predictions,
):

    contract = (
        MLTrainingContract(
            workflow_id=
                workflow_id,

            dataset_id=
                dataset_id,

            problem_type=
                "regression",

            target_column=
                "target",

            feature_columns=[
                "age",
                "segment",
            ],

            categorical_feature_columns=[
                "segment"
            ],

            estimator_key=
                "linear_regression",
        )
    )


    artifact = (
        register_ml_model_artifact(
            training_contract=
                contract,

            metrics={
                "mae":
                    0.75,

                "rmse":
                    1.00,

                "r2":
                    0.75,

                "median_absolute_error":
                    0.50,

                "explained_variance":
                    0.80,
            },

            train_rows=
                6,

            test_rows=
                2,

            model_bytes=
                b"PERFORMANCE-REGRESSOR",

            preparation_session_revision=
                revision,

            created_at_utc=
                "2026-08-29T20:00:00+00:00",
        )
    )


    estimator = (
        StaticEstimator(
            predictions
        )
    )


    return (
        LoadedMLModel(
            artifact=
                artifact,

            estimator=
                estimator,
        ),
        estimator,
    )


def observed_classification_frame(
) -> pd.DataFrame:

    return (
        pd.DataFrame(
            {
                "age": [
                    20.0,
                    25.0,
                    30.0,
                    35.0,
                ],

                "segment": [
                    "standard",
                    "premium",
                    "standard",
                    "premium",
                ],

                "target": [
                    0,
                    1,
                    0,
                    1,
                ],

                "business_note": [
                    "raw-private-value",
                    "raw-private-value",
                    "raw-private-value",
                    "raw-private-value",
                ],
            }
        )
    )


def observed_regression_frame(
) -> pd.DataFrame:

    return (
        pd.DataFrame(
            {
                "age": [
                    20.0,
                    25.0,
                    30.0,
                    35.0,
                ],

                "segment": [
                    "standard",
                    "premium",
                    "standard",
                    "premium",
                ],

                "target": [
                    1.0,
                    2.0,
                    3.0,
                    4.0,
                ],

                "business_note": [
                    "raw-private-value",
                    "raw-private-value",
                    "raw-private-value",
                    "raw-private-value",
                ],
            }
        )
    )


# ============================================================
# CLASSIFICATION ROUNDTRIP
# ============================================================


def test_classification_performance_roundtrip(
) -> None:

    with isolated_real_handoff_environment():

        (
            session,
            dataset_id,
            _,
        ) = (
            build_ready_preparation_workflow()
        )


        (
            trusted_model,
            estimator,
        ) = (
            classification_model(
                workflow_id=
                    session.workflow_id,

                dataset_id=
                    dataset_id,

                revision=
                    session.revision,

                predictions=[
                    0,
                    1,
                    0,
                    1,
                ],
            )
        )


        observed = (
            observed_classification_frame()
        )


        result = (
            evaluate_ml_performance(
                observed_dataframe=
                    observed,

                observed_dataset_id=
                    "dataset:observed-classification",

                observed_preparation_session_revision=
                    session.revision,

                trusted_model=
                    trusted_model,
            )
        )


        assert (
            result.workflow_id
            ==
            session.workflow_id
        )


        assert (
            result.model_id
            ==
            trusted_model
            .artifact
            .model_id
        )


        assert (
            result.reference_dataset_id
            ==
            dataset_id
        )


        assert (
            result.observed_dataset_id
            ==
            "dataset:observed-classification"
        )


        assert (
            result.problem_type
            ==
            "classification"
        )


        assert (
            result.target_column
            ==
            "target"
        )


        assert (
            result.primary_metric
            ==
            "f1_macro"
        )


        assert (
            result.performance_status
            ==
            "ok"
        )


        assert (
            result.observed_row_count
            ==
            4
        )


        assert (
            result.reference_evaluation_row_count
            ==
            2
        )


        assert (
            estimator.call_count
            ==
            1
        )


        assert (
            estimator.last_columns
            ==
            [
                "age",
                "segment",
            ]
        )


        assert [
            item.metric_name

            for item
            in result.metric_results
        ] == [
            "accuracy",
            "f1_macro",
            "precision_macro",
            "recall_macro",
            "balanced_accuracy",
        ]


        serialized = str(
            result.model_dump(
                mode="json"
            )
        )


        assert (
            "business_note"
            not in
            serialized
        )


        assert (
            "raw-private-value"
            not in
            serialized
        )


        assert (
            "y_true"
            not in
            serialized
        )


        assert (
            "predictions"
            not in
            serialized
        )


# ============================================================
# DEGRADED CLASSIFICATION
# ============================================================


def test_classification_degradation_detected(
) -> None:

    with isolated_real_handoff_environment():

        (
            session,
            dataset_id,
            _,
        ) = (
            build_ready_preparation_workflow()
        )


        (
            trusted_model,
            _,
        ) = (
            classification_model(
                workflow_id=
                    session.workflow_id,

                dataset_id=
                    dataset_id,

                revision=
                    session.revision,

                predictions=[
                    0,
                    0,
                    0,
                    0,
                ],
            )
        )


        result = (
            evaluate_ml_performance(
                observed_dataframe=
                    observed_classification_frame(),

                observed_dataset_id=
                    "dataset:degraded",

                observed_preparation_session_revision=
                    session.revision,

                trusted_model=
                    trusted_model,
            )
        )


        assert (
            result.performance_status
            ==
            "degraded"
        )


        assert (
            result
            .primary_metric_degradation_amount
            >
            0.10
        )


# ============================================================
# REGRESSION ROUNDTRIP
# ============================================================


def test_regression_performance_roundtrip(
) -> None:

    with isolated_real_handoff_environment():

        (
            session,
            dataset_id,
            _,
        ) = (
            build_ready_preparation_workflow()
        )


        (
            trusted_model,
            estimator,
        ) = (
            regression_model(
                workflow_id=
                    session.workflow_id,

                dataset_id=
                    dataset_id,

                revision=
                    session.revision,

                predictions=[
                    4.0,
                    4.0,
                    4.0,
                    4.0,
                ],
            )
        )


        result = (
            evaluate_ml_performance(
                observed_dataframe=
                    observed_regression_frame(),

                observed_dataset_id=
                    "dataset:observed-regression",

                observed_preparation_session_revision=
                    session.revision,

                trusted_model=
                    trusted_model,
            )
        )


        assert (
            result.problem_type
            ==
            "regression"
        )


        assert (
            result.primary_metric
            ==
            "rmse"
        )


        assert (
            result.performance_status
            ==
            "degraded"
        )


        assert (
            result
            .primary_metric_degradation_ratio
            is not None
        )


        assert (
            result
            .primary_metric_degradation_ratio
            >
            0.25
        )


        assert (
            estimator.call_count
            ==
            1
        )


# ============================================================
# TARGET REQUIRED BEFORE PREDICT
# ============================================================


def test_missing_target_blocks_before_predict(
) -> None:

    with isolated_real_handoff_environment():

        (
            session,
            dataset_id,
            _,
        ) = (
            build_ready_preparation_workflow()
        )


        (
            trusted_model,
            estimator,
        ) = (
            classification_model(
                workflow_id=
                    session.workflow_id,

                dataset_id=
                    dataset_id,

                revision=
                    session.revision,

                predictions=[
                    0,
                    1,
                    0,
                    1,
                ],
            )
        )


        observed = (
            observed_classification_frame()
            .drop(
                columns=[
                    "target"
                ]
            )
        )


        expect_error(
            MLPerformanceTargetError,

            lambda:
                evaluate_ml_performance(
                    observed_dataframe=
                        observed,

                    observed_dataset_id=
                        "dataset:no-target",

                    observed_preparation_session_revision=
                        session.revision,

                    trusted_model=
                        trusted_model,
                ),
        )


        assert (
            estimator.call_count
            ==
            0
        )


def test_missing_target_values_block_before_predict(
) -> None:

    with isolated_real_handoff_environment():

        (
            session,
            dataset_id,
            _,
        ) = (
            build_ready_preparation_workflow()
        )


        (
            trusted_model,
            estimator,
        ) = (
            classification_model(
                workflow_id=
                    session.workflow_id,

                dataset_id=
                    dataset_id,

                revision=
                    session.revision,

                predictions=[
                    0,
                    1,
                    0,
                    1,
                ],
            )
        )


        observed = (
            observed_classification_frame()
        )


        observed.loc[
            2,
            "target",
        ] = None


        expect_error(
            MLPerformanceTargetError,

            lambda:
                evaluate_ml_performance(
                    observed_dataframe=
                        observed,

                    observed_dataset_id=
                        "dataset:missing-label",

                    observed_preparation_session_revision=
                        session.revision,

                    trusted_model=
                        trusted_model,
                ),
        )


        assert (
            estimator.call_count
            ==
            0
        )


# ============================================================
# INVALID REGRESSION TARGET
# ============================================================


def test_non_finite_regression_target_blocked(
) -> None:

    with isolated_real_handoff_environment():

        (
            session,
            dataset_id,
            _,
        ) = (
            build_ready_preparation_workflow()
        )


        (
            trusted_model,
            estimator,
        ) = (
            regression_model(
                workflow_id=
                    session.workflow_id,

                dataset_id=
                    dataset_id,

                revision=
                    session.revision,

                predictions=[
                    1.0,
                    2.0,
                    3.0,
                    4.0,
                ],
            )
        )


        observed = (
            observed_regression_frame()
        )


        observed.loc[
            1,
            "target",
        ] = np.inf


        expect_error(
            MLPerformanceTargetError,

            lambda:
                evaluate_ml_performance(
                    observed_dataframe=
                        observed,

                    observed_dataset_id=
                        "dataset:non-finite-target",

                    observed_preparation_session_revision=
                        session.revision,

                    trusted_model=
                        trusted_model,
                ),
        )


        assert (
            estimator.call_count
            ==
            0
        )


# ============================================================
# FEATURE CONTRACT
# ============================================================


def test_missing_feature_blocked(
) -> None:

    with isolated_real_handoff_environment():

        (
            session,
            dataset_id,
            _,
        ) = (
            build_ready_preparation_workflow()
        )


        (
            trusted_model,
            estimator,
        ) = (
            classification_model(
                workflow_id=
                    session.workflow_id,

                dataset_id=
                    dataset_id,

                revision=
                    session.revision,

                predictions=[
                    0,
                    1,
                    0,
                    1,
                ],
            )
        )


        observed = (
            observed_classification_frame()
            .drop(
                columns=[
                    "age"
                ]
            )
        )


        expect_error(
            MLPerformanceEvaluatorInputError,

            lambda:
                evaluate_ml_performance(
                    observed_dataframe=
                        observed,

                    observed_dataset_id=
                        "dataset:missing-feature",

                    observed_preparation_session_revision=
                        session.revision,

                    trusted_model=
                        trusted_model,
                ),
        )


        assert (
            estimator.call_count
            ==
            0
        )


def test_duplicate_required_feature_blocked(
) -> None:

    with isolated_real_handoff_environment():

        (
            session,
            dataset_id,
            _,
        ) = (
            build_ready_preparation_workflow()
        )


        (
            trusted_model,
            estimator,
        ) = (
            classification_model(
                workflow_id=
                    session.workflow_id,

                dataset_id=
                    dataset_id,

                revision=
                    session.revision,

                predictions=[
                    0,
                    1,
                    0,
                    1,
                ],
            )
        )


        observed = (
            observed_classification_frame()
        )


        observed = pd.concat(
            [
                observed,
                observed[
                    [
                        "age"
                    ]
                ],
            ],
            axis=1,
        )


        expect_error(
            MLPerformanceEvaluatorInputError,

            lambda:
                evaluate_ml_performance(
                    observed_dataframe=
                        observed,

                    observed_dataset_id=
                        "dataset:duplicate-feature",

                    observed_preparation_session_revision=
                        session.revision,

                    trusted_model=
                        trusted_model,
                ),
        )


        assert (
            estimator.call_count
            ==
            0
        )


# ============================================================
# PREDICTION CONTRACT
# ============================================================


def test_wrong_prediction_count_blocked(
) -> None:

    with isolated_real_handoff_environment():

        (
            session,
            dataset_id,
            _,
        ) = (
            build_ready_preparation_workflow()
        )


        (
            trusted_model,
            _,
        ) = (
            classification_model(
                workflow_id=
                    session.workflow_id,

                dataset_id=
                    dataset_id,

                revision=
                    session.revision,

                predictions=[
                    0,
                    1,
                ],
            )
        )


        expect_error(
            MLPerformancePredictionError,

            lambda:
                evaluate_ml_performance(
                    observed_dataframe=
                        observed_classification_frame(),

                    observed_dataset_id=
                        "dataset:bad-prediction-count",

                    observed_preparation_session_revision=
                        session.revision,

                    trusted_model=
                        trusted_model,
                ),
        )


# ============================================================
# REFERENCE METRICS AUTHORITY
# ============================================================


def test_incomplete_reference_metrics_blocked(
) -> None:

    with isolated_real_handoff_environment():

        (
            session,
            dataset_id,
            _,
        ) = (
            build_ready_preparation_workflow()
        )


        (
            trusted_model,
            estimator,
        ) = (
            classification_model(
                workflow_id=
                    session.workflow_id,

                dataset_id=
                    dataset_id,

                revision=
                    session.revision,

                predictions=[
                    0,
                    1,
                    0,
                    1,
                ],

                metrics={
                    "accuracy":
                        0.80,

                    "f1_macro":
                        0.80,
                },
            )
        )


        expect_error(
            MLPerformanceEvaluatorAuthorityError,

            lambda:
                evaluate_ml_performance(
                    observed_dataframe=
                        observed_classification_frame(),

                    observed_dataset_id=
                        "dataset:incomplete-reference",

                    observed_preparation_session_revision=
                        session.revision,

                    trusted_model=
                        trusted_model,
                ),
        )


        assert (
            estimator.call_count
            ==
            0
        )


# ============================================================
# REVISION AUTHORITY
# ============================================================


def test_observed_revision_validation(
) -> None:

    with isolated_real_handoff_environment():

        (
            session,
            dataset_id,
            _,
        ) = (
            build_ready_preparation_workflow()
        )


        (
            trusted_model,
            _,
        ) = (
            classification_model(
                workflow_id=
                    session.workflow_id,

                dataset_id=
                    dataset_id,

                revision=
                    session.revision,

                predictions=[
                    0,
                    1,
                    0,
                    1,
                ],
            )
        )


        for invalid_revision in (
            True,
            -1,
            None,
            "7",
        ):

            expect_error(
                MLPerformanceEvaluatorInputError,

                lambda invalid_revision=invalid_revision:
                    evaluate_ml_performance(
                        observed_dataframe=
                            observed_classification_frame(),

                        observed_dataset_id=
                            "dataset:revision",

                        observed_preparation_session_revision=(
                            invalid_revision
                        ),

                        trusted_model=
                            trusted_model,
                    ),
            )


# ============================================================
# INPUT IMMUTABILITY
# ============================================================


def test_observed_dataframe_remains_immutable(
) -> None:

    with isolated_real_handoff_environment():

        (
            session,
            dataset_id,
            _,
        ) = (
            build_ready_preparation_workflow()
        )


        (
            trusted_model,
            _,
        ) = (
            classification_model(
                workflow_id=
                    session.workflow_id,

                dataset_id=
                    dataset_id,

                revision=
                    session.revision,

                predictions=[
                    0,
                    1,
                    0,
                    1,
                ],
            )
        )


        observed = (
            observed_classification_frame()
        )


        before = (
            observed.copy(
                deep=True
            )
        )


        evaluate_ml_performance(
            observed_dataframe=
                observed,

            observed_dataset_id=
                "dataset:immutable",

            observed_preparation_session_revision=
                session.revision,

            trusted_model=
                trusted_model,
        )


        pd.testing.assert_frame_equal(
            observed,
            before,
        )


# ============================================================
# GENERATED AUTHORITY
# ============================================================


def test_server_generates_evaluation_identity_and_time(
) -> None:

    with isolated_real_handoff_environment():

        (
            session,
            dataset_id,
            _,
        ) = (
            build_ready_preparation_workflow()
        )


        (
            trusted_model,
            _,
        ) = (
            classification_model(
                workflow_id=
                    session.workflow_id,

                dataset_id=
                    dataset_id,

                revision=
                    session.revision,

                predictions=[
                    0,
                    1,
                    0,
                    1,
                ],
            )
        )


        result = (
            evaluate_ml_performance(
                observed_dataframe=
                    observed_classification_frame(),

                observed_dataset_id=
                    "dataset:generated",

                observed_preparation_session_revision=
                    session.revision,

                trusted_model=
                    trusted_model,
            )
        )


        assert (
            result
            .performance_evaluation_id
            .startswith(
                "performance-evaluation:"
            )
        )


        assert (
            len(
                result
                .performance_evaluation_id
                .split(
                    ":",
                    1,
                )[
                    1
                ]
            )
            ==
            32
        )


        assert (
            result.evaluated_at_utc
        )


# ============================================================
# VERSION
# ============================================================


def test_rule_version(
) -> None:

    assert (
        ML_PERFORMANCE_EVALUATOR_RULE_VERSION
        ==
        "ml_performance_evaluator_v0.1"
    )


# ============================================================
# MAIN
# ============================================================


def main(
) -> None:

    print(
        (
            "=== DATALENS ML PERFORMANCE "
            "EVALUATOR v0.1 ==="
        )
    )

    print()


    tests = [
        (
            "Classification performance roundtrip",
            test_classification_performance_roundtrip,
        ),
        (
            "Classification degradation detected",
            test_classification_degradation_detected,
        ),
        (
            "Regression performance roundtrip",
            test_regression_performance_roundtrip,
        ),
        (
            "Missing target blocks before predict",
            test_missing_target_blocks_before_predict,
        ),
        (
            "Missing target values block before predict",
            test_missing_target_values_block_before_predict,
        ),
        (
            "Non-finite regression target blocked",
            test_non_finite_regression_target_blocked,
        ),
        (
            "Missing required feature blocked",
            test_missing_feature_blocked,
        ),
        (
            "Duplicate required feature blocked",
            test_duplicate_required_feature_blocked,
        ),
        (
            "Wrong prediction count blocked",
            test_wrong_prediction_count_blocked,
        ),
        (
            "Incomplete reference metrics blocked",
            test_incomplete_reference_metrics_blocked,
        ),
        (
            "Observed revision authority",
            test_observed_revision_validation,
        ),
        (
            "Observed DataFrame remains immutable",
            test_observed_dataframe_remains_immutable,
        ),
        (
            "Server-generated evaluation identity / time",
            test_server_generates_evaluation_identity_and_time,
        ),
        (
            "Performance Evaluator rule version",
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
            "PASS - ML Performance "
            "Evaluator v0.1"
        )
    )


if __name__ == "__main__":
    main()
