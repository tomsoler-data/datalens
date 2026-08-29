from __future__ import annotations

import math

from pydantic import ValidationError

from app.ml.baseline import (
    MLBaselineComparisonResult,
    MLBaselineEvaluationResult,
)

from app.ml.classification_diagnostics import (
    MLClassificationClassDiagnostics,
    MLClassificationDiagnosticsResult,
    MLClassificationMetricAverage,
)

from app.ml.decision_threshold import (
    MLDecisionThresholdContract,
    MLDecisionThresholdResult,
)

from app.ml.model_evaluation_summary import (
    ML_MODEL_EVALUATION_SUMMARY_RULE_VERSION,
    ML_MODEL_SELECTION_EVIDENCE_RULE_VERSION,
    MLModelEvaluationSummaryContract,
    MLModelEvaluationSummaryResult,
    MLModelSelectionEvidence,
    expected_model_evaluation_limitations,
)

from app.ml.model_explainability import (
    MLFeatureImportanceResult,
    MLModelExplainabilityResult,
)


# ============================================================
# CONSTANTS
# ============================================================


WORKFLOW_ID = (
    "prep:model-evaluation-summary"
)

DATASET_ID = (
    "dataset:model-evaluation-summary"
)

MODEL_ID = (
    "model:model-evaluation-summary"
)

EXPERIMENT_ID = (
    "experiment:"
    +
    (
        "a"
        *
        32
    )
)

TRAINING_SHA = (
    "b"
    *
    64
)

PREPARATION_REVISION = 7

TRAIN_ROWS = 80

TEST_ROWS = 20


# ============================================================
# HELPERS
# ============================================================


def safe_ratio(
    numerator: int,
    denominator: int,
) -> float:

    if denominator <= 0:
        return 0.0

    return float(
        numerator
        /
        denominator
    )


def f1_value(
    precision: float,
    recall: float,
) -> float:

    denominator = (
        precision
        +
        recall
    )

    if denominator <= 0.0:
        return 0.0

    return float(
        2.0
        *
        precision
        *
        recall
        /
        denominator
    )


def expect_validation_error(
    factory,
) -> None:

    try:
        factory()

    except ValidationError:
        return

    raise AssertionError(
        "Expected pydantic ValidationError."
    )


# ============================================================
# CLASSIFICATION DIAGNOSTICS
# ============================================================


def classification_diagnostics(
) -> MLClassificationDiagnosticsResult:

    labels = [
        "negative",
        "positive",
    ]

    matrix = [
        [
            8,
            2,
        ],
        [
            1,
            9,
        ],
    ]

    evaluation_rows = 20

    per_class = []

    for index, label in enumerate(labels):

        true_positive = (
            matrix[
                index
            ][
                index
            ]
        )

        false_negative = (
            sum(
                matrix[
                    index
                ]
            )
            -
            true_positive
        )

        false_positive = (
            sum(
                row[
                    index
                ]
                for row
                in matrix
            )
            -
            true_positive
        )

        true_negative = (
            evaluation_rows
            -
            true_positive
            -
            false_negative
            -
            false_positive
        )

        support = (
            true_positive
            +
            false_negative
        )

        precision = safe_ratio(
            true_positive,
            (
                true_positive
                +
                false_positive
            ),
        )

        recall = safe_ratio(
            true_positive,
            support,
        )

        f1 = f1_value(
            precision,
            recall,
        )

        per_class.append(
            MLClassificationClassDiagnostics(
                class_label=
                    label,

                precision=
                    precision,

                recall=
                    recall,

                f1=
                    f1,

                support=
                    support,

                true_positive=
                    true_positive,

                false_positive=
                    false_positive,

                false_negative=
                    false_negative,

                true_negative=
                    true_negative,
            )
        )

    macro_precision = (
        sum(
            item.precision
            for item
            in per_class
        )
        /
        len(per_class)
    )

    macro_recall = (
        sum(
            item.recall
            for item
            in per_class
        )
        /
        len(per_class)
    )

    macro_f1 = (
        sum(
            item.f1
            for item
            in per_class
        )
        /
        len(per_class)
    )

    weighted_precision = (
        sum(
            item.precision
            *
            item.support
            for item
            in per_class
        )
        /
        evaluation_rows
    )

    weighted_recall = (
        sum(
            item.recall
            *
            item.support
            for item
            in per_class
        )
        /
        evaluation_rows
    )

    weighted_f1 = (
        sum(
            item.f1
            *
            item.support
            for item
            in per_class
        )
        /
        evaluation_rows
    )

    accuracy = (
        (
            matrix[0][0]
            +
            matrix[1][1]
        )
        /
        evaluation_rows
    )

    balanced_accuracy = (
        macro_recall
    )

    return (
        MLClassificationDiagnosticsResult(
            workflow_id=
                WORKFLOW_ID,

            dataset_id=
                DATASET_ID,

            model_id=
                MODEL_ID,

            experiment_id=
                EXPERIMENT_ID,

            target_column=
                "churned",

            estimator_key=
                "logistic_regression",

            preparation_session_revision=(
                PREPARATION_REVISION
            ),

            training_contract_sha256=(
                TRAINING_SHA
            ),

            evaluation_rows=
                evaluation_rows,

            class_count=
                2,

            class_labels=
                labels,

            confusion_matrix=
                matrix,

            per_class=
                per_class,

            accuracy=
                accuracy,

            balanced_accuracy=(
                balanced_accuracy
            ),

            macro_average=(
                MLClassificationMetricAverage(
                    precision=
                        macro_precision,

                    recall=
                        macro_recall,

                    f1=
                        macro_f1,
                )
            ),

            weighted_average=(
                MLClassificationMetricAverage(
                    precision=
                        weighted_precision,

                    recall=
                        weighted_recall,

                    f1=
                        weighted_f1,
                )
            ),
        )
    )


def classification_metrics(
) -> dict[
    str,
    float,
]:

    diagnostics = (
        classification_diagnostics()
    )

    return {
        "accuracy":
            diagnostics.accuracy,

        "f1_macro":
            diagnostics
            .macro_average
            .f1,

        "precision_macro":
            diagnostics
            .macro_average
            .precision,

        "recall_macro":
            diagnostics
            .macro_average
            .recall,

        "balanced_accuracy":
            diagnostics
            .balanced_accuracy,
    }


# ============================================================
# DECISION THRESHOLD
# ============================================================


def threshold_result(
    *,
    threshold: float = 0.70,
    negative_label: str = "negative",
    positive_label: str = "positive",
) -> MLDecisionThresholdResult:

    true_negative = 9
    false_positive = 1
    false_negative = 4
    true_positive = 6

    negative_support = (
        true_negative
        +
        false_positive
    )

    positive_support = (
        false_negative
        +
        true_positive
    )

    precision = safe_ratio(
        true_positive,
        (
            true_positive
            +
            false_positive
        ),
    )

    recall = safe_ratio(
        true_positive,
        positive_support,
    )

    specificity = safe_ratio(
        true_negative,
        negative_support,
    )

    f1 = f1_value(
        precision,
        recall,
    )

    accuracy = (
        (
            true_negative
            +
            true_positive
        )
        /
        TEST_ROWS
    )

    balanced_accuracy = (
        (
            specificity
            +
            recall
        )
        /
        2.0
    )

    positive_prediction_rate = (
        (
            true_positive
            +
            false_positive
        )
        /
        TEST_ROWS
    )

    return (
        MLDecisionThresholdResult(
            workflow_id=
                WORKFLOW_ID,

            dataset_id=
                DATASET_ID,

            model_id=
                MODEL_ID,

            experiment_id=
                EXPERIMENT_ID,

            target_column=
                "churned",

            estimator_key=
                "logistic_regression",

            preparation_session_revision=(
                PREPARATION_REVISION
            ),

            training_contract_sha256=(
                TRAINING_SHA
            ),

            evaluation_rows=
                TEST_ROWS,

            threshold=
                threshold,

            negative_class_label=(
                negative_label
            ),

            positive_class_label=(
                positive_label
            ),

            confusion_matrix=[
                [
                    true_negative,
                    false_positive,
                ],
                [
                    false_negative,
                    true_positive,
                ],
            ],

            negative_support=
                negative_support,

            positive_support=
                positive_support,

            true_negative=
                true_negative,

            false_positive=
                false_positive,

            false_negative=
                false_negative,

            true_positive=
                true_positive,

            precision=
                precision,

            recall=
                recall,

            f1=
                f1,

            specificity=
                specificity,

            accuracy=
                accuracy,

            balanced_accuracy=(
                balanced_accuracy
            ),

            positive_prediction_rate=(
                positive_prediction_rate
            ),
        )
    )


# ============================================================
# REGRESSION METRICS
# ============================================================


def regression_metrics(
) -> dict[
    str,
    float,
]:

    return {
        "mae":
            2.0,

        "rmse":
            3.0,

        "r2":
            0.50,

        "median_absolute_error":
            1.50,

        "explained_variance":
            0.60,
    }


# ============================================================
# BASELINE
# ============================================================


def baseline_result(
    problem_type: str,
) -> MLBaselineEvaluationResult:

    if problem_type == "regression":

        return (
            MLBaselineEvaluationResult(
                problem_type=
                    "regression",

                strategy=
                    "mean_train_target",

                primary_metric=
                    "rmse",

                train_rows=
                    TRAIN_ROWS,

                test_rows=
                    TEST_ROWS,

                metrics={
                    "mae":
                        4.0,

                    "rmse":
                        5.0,

                    "r2":
                        -0.10,
                },
            )
        )

    return (
        MLBaselineEvaluationResult(
            problem_type=
                "classification",

            strategy=
                "majority_train_class",

            primary_metric=
                "f1_macro",

            train_rows=
                TRAIN_ROWS,

            test_rows=
                TEST_ROWS,

            metrics={
                "accuracy":
                    0.50,

                "f1_macro":
                    0.45,
            },
        )
    )


def baseline_comparison(
    problem_type: str,
    *,
    model_primary_value: (
        float
        |
        None
    ) = None,
) -> MLBaselineComparisonResult:

    if problem_type == "regression":

        if model_primary_value is None:
            model_primary_value = 3.0

        baseline_primary_value = 5.0
        improvement = (
            baseline_primary_value
            -
            model_primary_value
        )

        primary_metric = "rmse"

    else:

        if model_primary_value is None:
            model_primary_value = (
                classification_metrics()[
                    "f1_macro"
                ]
            )

        baseline_primary_value = 0.45
        improvement = (
            model_primary_value
            -
            baseline_primary_value
        )

        primary_metric = "f1_macro"

    relative = (
        improvement
        /
        abs(
            baseline_primary_value
        )
        *
        100.0
    )

    return (
        MLBaselineComparisonResult(
            problem_type=
                problem_type,

            primary_metric=
                primary_metric,

            model_primary_metric_value=(
                model_primary_value
            ),

            baseline_primary_metric_value=(
                baseline_primary_value
            ),

            absolute_improvement=
                improvement,

            relative_improvement_pct=
                relative,

            beats_baseline=(
                improvement
                >
                0.0
            ),
        )
    )


# ============================================================
# EXPLAINABILITY
# ============================================================


def explainability_result(
    problem_type: str,
    *,
    model_id: str = MODEL_ID,
) -> MLModelExplainabilityResult:

    if problem_type == "regression":

        estimator_key = (
            "linear_regression"
        )

        scoring = (
            "neg_root_mean_squared_error"
        )

    else:

        estimator_key = (
            "logistic_regression"
        )

        scoring = (
            "f1_macro"
        )

    return (
        MLModelExplainabilityResult(
            workflow_id=
                WORKFLOW_ID,

            dataset_id=
                DATASET_ID,

            model_id=
                model_id,

            experiment_id=
                EXPERIMENT_ID,

            problem_type=
                problem_type,

            estimator_key=
                estimator_key,

            preparation_session_revision=(
                PREPARATION_REVISION
            ),

            training_contract_sha256=(
                TRAINING_SHA
            ),

            scoring=
                scoring,

            n_repeats=
                10,

            random_seed=
                42,

            evaluation_rows=
                TEST_ROWS,

            feature_importances=[
                MLFeatureImportanceResult(
                    feature_name=
                        "age",

                    rank=
                        1,

                    importance_mean=
                        0.40,

                    importance_std=
                        0.05,
                ),
                MLFeatureImportanceResult(
                    feature_name=
                        "tenure",

                    rank=
                        2,

                    importance_mean=
                        0.10,

                    importance_std=
                        0.02,
                ),
            ],
        )
    )


# ============================================================
# SELECTION EVIDENCE
# ============================================================


def selection_evidence(
    problem_type: str,
    source: str,
    *,
    primary_value: (
        float
        |
        None
    ) = None,
) -> MLModelSelectionEvidence:

    if source == "standalone_model":

        return (
            MLModelSelectionEvidence(
                source=
                    "standalone_model",

                status=(
                    "selection_not_available"
                ),

                metric_scope=
                    "not_available",
            )
        )

    primary_metric = (
        "rmse"
        if problem_type
        ==
        "regression"
        else
        "f1_macro"
    )

    if source == "model_comparison":

        if primary_value is None:

            primary_value = (
                regression_metrics()[
                    "rmse"
                ]
                if problem_type
                ==
                "regression"
                else
                classification_metrics()[
                    "f1_macro"
                ]
            )

        policy = (
            "regression_rmse_v0.1"
            if problem_type
            ==
            "regression"
            else
            "classification_f1_macro_v0.1"
        )

        return (
            MLModelSelectionEvidence(
                source=
                    "model_comparison",

                status=
                    "verified_selected",

                rank=
                    1,

                selection_policy=
                    policy,

                primary_metric=(
                    primary_metric
                ),

                primary_metric_value=(
                    primary_value
                ),

                metric_scope=
                    "final_holdout",
            )
        )

    if primary_value is None:

        primary_value = (
            2.75
            if problem_type
            ==
            "regression"
            else
            0.72
        )

    return (
        MLModelSelectionEvidence(
            source=
                "tuned_model_promotion",

            status=
                "verified_selected",

            rank=
                1,

            selection_policy=
                "rank_1_only",

            primary_metric=
                primary_metric,

            primary_metric_value=(
                primary_value
            ),

            metric_scope=(
                "inner_cross_validation"
            ),
        )
    )


# ============================================================
# VALID SUMMARY FIXTURES
# ============================================================


def valid_regression_summary(
    *,
    selection_source: str = (
        "standalone_model"
    ),
) -> MLModelEvaluationSummaryResult:

    contract = (
        MLModelEvaluationSummaryContract()
    )

    selection = (
        selection_evidence(
            "regression",
            selection_source,
        )
    )

    limitations = (
        expected_model_evaluation_limitations(
            problem_type=
                "regression",

            selection_source=
                selection.source,

            threshold_requested=
                False,
        )
    )

    return (
        MLModelEvaluationSummaryResult(
            workflow_id=
                WORKFLOW_ID,

            dataset_id=
                DATASET_ID,

            model_id=
                MODEL_ID,

            experiment_id=
                EXPERIMENT_ID,

            problem_type=
                "regression",

            target_column=
                "revenue",

            estimator_key=
                "linear_regression",

            preparation_session_revision=(
                PREPARATION_REVISION
            ),

            training_contract_sha256=(
                TRAINING_SHA
            ),

            train_rows=
                TRAIN_ROWS,

            test_rows=
                TEST_ROWS,

            summary_contract=
                contract,

            metrics=
                regression_metrics(),

            baseline=(
                baseline_result(
                    "regression"
                )
            ),

            baseline_comparison=(
                baseline_comparison(
                    "regression"
                )
            ),

            selection_evidence=
                selection,

            explainability=(
                explainability_result(
                    "regression"
                )
            ),

            limitations=
                limitations,
        )
    )


def valid_classification_summary(
    *,
    with_threshold: bool = False,
    selection_source: str = (
        "model_comparison"
    ),
) -> MLModelEvaluationSummaryResult:

    metrics = (
        classification_metrics()
    )

    threshold_contract = (
        MLDecisionThresholdContract(
            threshold=
                0.70
        )
        if with_threshold
        else
        None
    )

    contract = (
        MLModelEvaluationSummaryContract(
            decision_threshold=(
                threshold_contract
            )
        )
    )

    selection = (
        selection_evidence(
            "classification",
            selection_source,
        )
    )

    limitations = (
        expected_model_evaluation_limitations(
            problem_type=
                "classification",

            selection_source=
                selection.source,

            threshold_requested=(
                with_threshold
            ),
        )
    )

    return (
        MLModelEvaluationSummaryResult(
            workflow_id=
                WORKFLOW_ID,

            dataset_id=
                DATASET_ID,

            model_id=
                MODEL_ID,

            experiment_id=
                EXPERIMENT_ID,

            problem_type=
                "classification",

            target_column=
                "churned",

            estimator_key=
                "logistic_regression",

            preparation_session_revision=(
                PREPARATION_REVISION
            ),

            training_contract_sha256=(
                TRAINING_SHA
            ),

            train_rows=
                TRAIN_ROWS,

            test_rows=
                TEST_ROWS,

            summary_contract=
                contract,

            metrics=
                metrics,

            baseline=(
                baseline_result(
                    "classification"
                )
            ),

            baseline_comparison=(
                baseline_comparison(
                    "classification"
                )
            ),

            selection_evidence=
                selection,

            classification_diagnostics=(
                classification_diagnostics()
            ),

            decision_threshold_evaluation=(
                threshold_result()
                if with_threshold
                else
                None
            ),

            explainability=(
                explainability_result(
                    "classification"
                )
            ),

            limitations=
                limitations,
        )
    )


# ============================================================
# CONTRACT TESTS
# ============================================================


def test_contract_defaults_and_authority(
) -> None:

    contract = (
        MLModelEvaluationSummaryContract()
    )

    assert (
        contract.decision_threshold
        is None
    )

    assert (
        contract.method
        ==
        "trusted_model_evaluation_summary"
    )

    assert (
        contract.evaluation_scope
        ==
        "persisted_model_holdout"
    )

    assert (
        contract.evidence_policy
        ==
        "server_reconstructed_only"
    )

    assert (
        contract.selection_policy
        ==
        "preserve_upstream_selection_only"
    )

    assert (
        contract.explainability_policy
        ==
        "default_permutation_importance_v0.1"
    )

    assert (
        contract.threshold_policy
        ==
        "explicit_requested_threshold_only"
    )


def test_contract_accepts_explicit_threshold(
) -> None:

    contract = (
        MLModelEvaluationSummaryContract(
            decision_threshold=(
                MLDecisionThresholdContract(
                    threshold=
                        0.65
                )
            )
        )
    )

    assert (
        contract.decision_threshold
        is not None
    )

    assert (
        contract
        .decision_threshold
        .threshold
        ==
        0.65
    )


def test_contract_is_strict_and_frozen(
) -> None:

    expect_validation_error(
        lambda:
            MLModelEvaluationSummaryContract
            .model_validate(
                {
                    "unexpected":
                        True
                }
            )
    )

    contract = (
        MLModelEvaluationSummaryContract()
    )

    try:
        contract.method = "other"

    except ValidationError:
        pass

    else:
        raise AssertionError(
            "Contract must be frozen."
        )


# ============================================================
# SELECTION EVIDENCE TESTS
# ============================================================


def test_standalone_selection_evidence(
) -> None:

    evidence = (
        selection_evidence(
            "regression",
            "standalone_model",
        )
    )

    assert (
        evidence.status
        ==
        "selection_not_available"
    )

    assert evidence.rank is None
    assert evidence.primary_metric is None

    assert (
        evidence.metric_scope
        ==
        "not_available"
    )


def test_model_comparison_selection_evidence(
) -> None:

    evidence = (
        selection_evidence(
            "classification",
            "model_comparison",
        )
    )

    assert (
        evidence.status
        ==
        "verified_selected"
    )

    assert evidence.rank == 1

    assert (
        evidence.primary_metric
        ==
        "f1_macro"
    )

    assert (
        evidence.metric_scope
        ==
        "final_holdout"
    )


def test_tuned_selection_evidence(
) -> None:

    evidence = (
        selection_evidence(
            "regression",
            "tuned_model_promotion",
        )
    )

    assert evidence.rank == 1

    assert (
        evidence.selection_policy
        ==
        "rank_1_only"
    )

    assert (
        evidence.metric_scope
        ==
        "inner_cross_validation"
    )


def test_invalid_selection_claim_fails_closed(
) -> None:

    expect_validation_error(
        lambda:
            MLModelSelectionEvidence(
                source=
                    "model_comparison",

                status=
                    "verified_selected",

                rank=
                    2,

                selection_policy=(
                    "classification_f1_macro_v0.1"
                ),

                primary_metric=
                    "f1_macro",

                primary_metric_value=
                    0.8,

                metric_scope=
                    "final_holdout",
            )
    )


# ============================================================
# VALID SUMMARIES
# ============================================================


def test_valid_regression_summary(
) -> None:

    result = (
        valid_regression_summary()
    )

    assert (
        result.problem_type
        ==
        "regression"
    )

    assert result.metrics["rmse"] == 3.0

    assert (
        result.classification_diagnostics
        is None
    )

    assert (
        result.decision_threshold_evaluation
        is None
    )

    assert (
        result.evaluation_status
        ==
        "complete"
    )


def test_valid_classification_summary_without_threshold(
) -> None:

    result = (
        valid_classification_summary()
    )

    assert (
        result.classification_diagnostics
        is not None
    )

    assert (
        result.decision_threshold_evaluation
        is None
    )

    assert (
        result.selection_evidence.source
        ==
        "model_comparison"
    )


def test_valid_classification_summary_with_threshold(
) -> None:

    result = (
        valid_classification_summary(
            with_threshold=
                True
        )
    )

    assert (
        result.decision_threshold_evaluation
        is not None
    )

    assert (
        result
        .decision_threshold_evaluation
        .threshold
        ==
        0.70
    )


# ============================================================
# PROBLEM-TYPE AUTHORITY
# ============================================================


def test_regression_rejects_threshold_request(
) -> None:

    result = (
        valid_regression_summary()
    )

    payload = (
        result.model_dump(
            mode="python"
        )
    )

    payload["summary_contract"] = (
        MLModelEvaluationSummaryContract(
            decision_threshold=(
                MLDecisionThresholdContract(
                    threshold=
                        0.50
                )
            )
        )
        .model_dump(
            mode="python"
        )
    )

    expect_validation_error(
        lambda:
            MLModelEvaluationSummaryResult(
                **payload
            )
    )


def test_classification_requires_diagnostics(
) -> None:

    result = (
        valid_classification_summary()
    )

    payload = (
        result.model_dump(
            mode="python"
        )
    )

    payload[
        "classification_diagnostics"
    ] = None

    expect_validation_error(
        lambda:
            MLModelEvaluationSummaryResult(
                **payload
            )
    )


def test_requested_threshold_requires_evidence(
) -> None:

    result = (
        valid_classification_summary(
            with_threshold=
                True
        )
    )

    payload = (
        result.model_dump(
            mode="python"
        )
    )

    payload[
        "decision_threshold_evaluation"
    ] = None

    expect_validation_error(
        lambda:
            MLModelEvaluationSummaryResult(
                **payload
            )
    )


def test_unrequested_threshold_evidence_fails_closed(
) -> None:

    result = (
        valid_classification_summary()
    )

    payload = (
        result.model_dump(
            mode="python"
        )
    )

    payload[
        "decision_threshold_evaluation"
    ] = (
        threshold_result()
        .model_dump(
            mode="python"
        )
    )

    expect_validation_error(
        lambda:
            MLModelEvaluationSummaryResult(
                **payload
            )
    )


# ============================================================
# IDENTITY / METRIC BINDING
# ============================================================


def test_explainability_identity_is_bound(
) -> None:

    result = (
        valid_regression_summary()
    )

    payload = (
        result.model_dump(
            mode="python"
        )
    )

    payload["explainability"] = (
        explainability_result(
            "regression",
            model_id=
                "model:other",
        )
        .model_dump(
            mode="python"
        )
    )

    expect_validation_error(
        lambda:
            MLModelEvaluationSummaryResult(
                **payload
            )
    )


def test_baseline_model_metric_is_bound(
) -> None:

    result = (
        valid_regression_summary()
    )

    payload = (
        result.model_dump(
            mode="python"
        )
    )

    payload[
        "baseline_comparison"
    ] = (
        baseline_comparison(
            "regression",
            model_primary_value=
                2.50,
        )
        .model_dump(
            mode="python"
        )
    )

    expect_validation_error(
        lambda:
            MLModelEvaluationSummaryResult(
                **payload
            )
    )


def test_classification_metrics_bind_diagnostics(
) -> None:

    result = (
        valid_classification_summary()
    )

    payload = (
        result.model_dump(
            mode="python"
        )
    )

    metrics = dict(
        result.metrics
    )

    metrics["f1_macro"] = (
        metrics["f1_macro"]
        -
        0.01
    )

    payload["metrics"] = metrics

    payload[
        "baseline_comparison"
    ] = (
        baseline_comparison(
            "classification",
            model_primary_value=(
                metrics[
                    "f1_macro"
                ]
            ),
        )
        .model_dump(
            mode="python"
        )
    )

    payload[
        "selection_evidence"
    ] = (
        selection_evidence(
            "classification",
            "model_comparison",
            primary_value=(
                metrics[
                    "f1_macro"
                ]
            ),
        )
        .model_dump(
            mode="python"
        )
    )

    expect_validation_error(
        lambda:
            MLModelEvaluationSummaryResult(
                **payload
            )
    )


def test_model_comparison_metric_binds_final_holdout(
) -> None:

    result = (
        valid_classification_summary()
    )

    payload = (
        result.model_dump(
            mode="python"
        )
    )

    wrong_value = (
        result.metrics[
            "f1_macro"
        ]
        -
        0.02
    )

    payload[
        "selection_evidence"
    ] = (
        selection_evidence(
            "classification",
            "model_comparison",
            primary_value=
                wrong_value,
        )
        .model_dump(
            mode="python"
        )
    )

    expect_validation_error(
        lambda:
            MLModelEvaluationSummaryResult(
                **payload
            )
    )


def test_tuning_selection_metric_is_not_final_holdout_metric(
) -> None:

    result = (
        valid_classification_summary(
            selection_source=(
                "tuned_model_promotion"
            )
        )
    )

    assert (
        result.selection_evidence
        .metric_scope
        ==
        "inner_cross_validation"
    )

    assert not math.isclose(
        float(
            result
            .selection_evidence
            .primary_metric_value
        ),
        float(
            result.metrics[
                "f1_macro"
            ]
        ),
        rel_tol=1e-12,
        abs_tol=1e-12,
    )


def test_threshold_class_order_binds_diagnostics(
) -> None:

    result = (
        valid_classification_summary(
            with_threshold=
                True
        )
    )

    payload = (
        result.model_dump(
            mode="python"
        )
    )

    payload[
        "decision_threshold_evaluation"
    ] = (
        threshold_result(
            negative_label=
                "positive",

            positive_label=
                "negative",
        )
        .model_dump(
            mode="python"
        )
    )

    expect_validation_error(
        lambda:
            MLModelEvaluationSummaryResult(
                **payload
            )
    )


# ============================================================
# LIMITATIONS
# ============================================================


def test_limitations_are_server_derived(
) -> None:

    regression = (
        valid_regression_summary()
    )

    assert (
        regression.limitations
        ==
        [
            "single_holdout_evaluation",
            "no_external_validation",
            "feature_importance_not_causal",
            "selection_evidence_not_available",
        ]
    )

    classification = (
        valid_classification_summary(
            with_threshold=
                True
        )
    )

    assert (
        classification.limitations
        ==
        [
            "single_holdout_evaluation",
            "no_external_validation",
            "feature_importance_not_causal",
            "requested_threshold_not_optimized",
        ]
    )

    payload = (
        classification.model_dump(
            mode="python"
        )
    )

    payload["limitations"] = [
        "single_holdout_evaluation",
        "no_external_validation",
        "feature_importance_not_causal",
    ]

    expect_validation_error(
        lambda:
            MLModelEvaluationSummaryResult(
                **payload
            )
    )


# ============================================================
# PRIVACY
# ============================================================


def all_keys(
    value,
) -> set[
    str
]:

    keys = set()

    if isinstance(value, dict):

        for (
            key,
            nested,
        ) in value.items():

            keys.add(
                str(key)
            )

            keys.update(
                all_keys(
                    nested
                )
            )

    elif isinstance(value, list):

        for nested in value:
            keys.update(
                all_keys(
                    nested
                )
            )

    return keys


def test_result_is_privacy_minimal(
) -> None:

    result = (
        valid_classification_summary(
            with_threshold=
                True
        )
    )

    payload = (
        result.model_dump(
            mode="json"
        )
    )

    forbidden = {
        "model_artifact",
        "model_path",
        "model_file_bytes",
        "model_sha256",
        "model_bytes",
        "estimator",
        "training_contract",
        "raw_rows",
        "rows",
        "predictions",
        "holdout_predictions",
        "probabilities",
        "positive_probabilities",
        "negative_probabilities",
        "decision_scores",
        "scores",
        "y_true",
        "y_pred",
        "x_train",
        "x_test",
        "y_train",
        "y_test",
    }

    assert (
        forbidden.isdisjoint(
            all_keys(
                payload
            )
        )
    )


# ============================================================
# RULE VERSION
# ============================================================


def test_rule_versions(
) -> None:

    assert (
        ML_MODEL_EVALUATION_SUMMARY_RULE_VERSION
        ==
        "ml_model_evaluation_summary_v0.1"
    )

    assert (
        ML_MODEL_SELECTION_EVIDENCE_RULE_VERSION
        ==
        "ml_model_selection_evidence_v0.1"
    )


# ============================================================
# MAIN
# ============================================================


def main(
) -> None:

    print(
        "=== DATALENS ML MODEL EVALUATION SUMMARY CONTRACT v0.1 ==="
    )

    tests = [
        (
            "Summary contract defaults and authority",
            test_contract_defaults_and_authority,
        ),
        (
            "Explicit Decision Threshold contract accepted",
            test_contract_accepts_explicit_threshold,
        ),
        (
            "Summary contract strict and frozen",
            test_contract_is_strict_and_frozen,
        ),
        (
            "Standalone selection evidence",
            test_standalone_selection_evidence,
        ),
        (
            "Model Comparison selection evidence",
            test_model_comparison_selection_evidence,
        ),
        (
            "Tuned promotion selection evidence",
            test_tuned_selection_evidence,
        ),
        (
            "Invalid selection claim fails closed",
            test_invalid_selection_claim_fails_closed,
        ),
        (
            "Valid regression evaluation summary",
            test_valid_regression_summary,
        ),
        (
            "Valid classification summary without threshold",
            test_valid_classification_summary_without_threshold,
        ),
        (
            "Valid classification summary with threshold",
            test_valid_classification_summary_with_threshold,
        ),
        (
            "Regression rejects threshold request",
            test_regression_rejects_threshold_request,
        ),
        (
            "Classification requires diagnostics",
            test_classification_requires_diagnostics,
        ),
        (
            "Requested threshold requires evidence",
            test_requested_threshold_requires_evidence,
        ),
        (
            "Unrequested threshold evidence fails closed",
            test_unrequested_threshold_evidence_fails_closed,
        ),
        (
            "Explainability identity bound",
            test_explainability_identity_is_bound,
        ),
        (
            "Baseline model metric bound",
            test_baseline_model_metric_is_bound,
        ),
        (
            "Classification metrics bind diagnostics",
            test_classification_metrics_bind_diagnostics,
        ),
        (
            "Model Comparison metric binds final holdout",
            test_model_comparison_metric_binds_final_holdout,
        ),
        (
            "Tuning metric remains inner-CV evidence",
            test_tuning_selection_metric_is_not_final_holdout_metric,
        ),
        (
            "Threshold class order binds diagnostics",
            test_threshold_class_order_binds_diagnostics,
        ),
        (
            "Limitations are server-derived",
            test_limitations_are_server_derived,
        ),
        (
            "Summary result privacy-minimal",
            test_result_is_privacy_minimal,
        ),
        (
            "Rule versions",
            test_rule_versions,
        ),
    ]

    for (
        label,
        test,
    ) in tests:

        test()

        print(
            f"[PASS] {label}"
        )

    print()

    print(
        "PASS - ML Model Evaluation Summary Contract v0.1"
    )


if __name__ == "__main__":
    main()
