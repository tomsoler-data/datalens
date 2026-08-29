from __future__ import annotations


import math


import pandas as pd


import app.ml.model_evaluation_summary_executor as summary_executor


from app.ml.baseline import (
    build_ml_baseline_evaluation,
    build_ml_baseline_predictions,
    compare_model_to_baseline,
)


from app.ml.classical_executor import (
    _baseline_metrics_v0_1,
    _classification_metrics,
    _regression_metrics,
)


from app.ml.classification_diagnostics import (
    MLClassificationClassDiagnostics,
    MLClassificationDiagnosticsResult,
    MLClassificationMetricAverage,
)


from app.ml.contracts import (
    MLSplitContract,
    MLTrainingContract,
)


from app.ml.decision_threshold import (
    MLDecisionThresholdContract,
    MLDecisionThresholdResult,
)


from app.ml.experiment_provenance import (
    MLExperimentProvenanceRecord,
    ml_training_contract_sha256,
)


from app.ml.model_artifacts import (
    MLModelArtifactRecord,
)


from app.ml.model_evaluation_summary import (
    MLModelEvaluationSummaryContract,
)


from app.ml.model_evaluation_summary_executor import (
    ML_MODEL_EVALUATION_SUMMARY_EXECUTOR_RULE_VERSION,
    MLModelEvaluationSummaryArtifactError,
    MLModelEvaluationSummaryInputError,
    MLModelEvaluationSummarySelectionError,
    execute_ml_model_evaluation_summary,
)


from app.ml.model_explainability import (
    MLFeatureImportanceResult,
    MLModelExplainabilityResult,
)


from app.ml.model_loader import (
    LoadedMLModel,
)


# ============================================================
# CONSTANTS
# ============================================================


WORKFLOW_ID = (
    "prep:evaluation-summary"
)

DATASET_ID = (
    "dataset:evaluation-summary"
)

MODEL_ID = (
    "model:evaluation-summary"
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

PREPARATION_REVISION = 9


# ============================================================
# ASSERTIONS
# ============================================================


def expect_exception(
    exception_type,
    factory,
) -> None:

    try:
        factory()

    except exception_type:
        return

    raise AssertionError(
        (
            "Expected exception: "
            f"{exception_type.__name__}"
        )
    )


# ============================================================
# HOLDOUTS
# ============================================================


def regression_holdout():

    x_train = pd.DataFrame(
        {
            "feature":
                [
                    1.0,
                    2.0,
                    3.0,
                    4.0,
                ]
        }
    )

    x_test = pd.DataFrame(
        {
            "feature":
                [
                    5.0,
                    6.0,
                ]
        }
    )

    y_train = pd.Series(
        [
            1.0,
            2.0,
            3.0,
            4.0,
        ],
        name="target",
    )

    y_test = pd.Series(
        [
            2.0,
            4.0,
        ],
        name="target",
    )

    return (
        x_train,
        x_test,
        y_train,
        y_test,
    )


def classification_holdout():

    x_train = pd.DataFrame(
        {
            "feature":
                [
                    1.0,
                    2.0,
                    3.0,
                    4.0,
                ]
        }
    )

    x_test = pd.DataFrame(
        {
            "feature":
                [
                    5.0,
                    6.0,
                    7.0,
                    8.0,
                ]
        }
    )

    y_train = pd.Series(
        [
            "negative",
            "negative",
            "positive",
            "negative",
        ],
        name="target",
    )

    y_test = pd.Series(
        [
            "negative",
            "positive",
            "positive",
            "negative",
        ],
        name="target",
    )

    return (
        x_train,
        x_test,
        y_train,
        y_test,
    )


# ============================================================
# TRAINING CONTRACTS
# ============================================================


def training_contract(
    problem_type: str,
) -> MLTrainingContract:

    return (
        MLTrainingContract(
            workflow_id=
                WORKFLOW_ID,

            dataset_id=
                DATASET_ID,

            problem_type=
                problem_type,

            target_column=
                "target",

            feature_columns=[
                "feature",
            ],

            estimator_key=(
                "linear_regression"
                if problem_type
                ==
                "regression"
                else
                "logistic_regression"
            ),

            split=(
                MLSplitContract(
                    test_size=
                        0.33,

                    random_seed=
                        42,

                    shuffle=
                        True,

                    stratify=
                        False,
                )
            ),
        )
    )


# ============================================================
# METRICS
# ============================================================


def regression_model_metrics():

    return {
        "mae":
            0.50,

        "rmse":
            0.60,

        "r2":
            0.80,

        "median_absolute_error":
            0.50,

        "explained_variance":
            0.90,
    }


def classification_diagnostics(
) -> MLClassificationDiagnosticsResult:

    matrix = [
        [
            2,
            0,
        ],
        [
            1,
            1,
        ],
    ]

    labels = [
        "negative",
        "positive",
    ]

    evaluation_rows = 4

    per_class = []

    for index, label in enumerate(labels):

        tp = (
            matrix[
                index
            ][
                index
            ]
        )

        fn = (
            sum(
                matrix[
                    index
                ]
            )
            -
            tp
        )

        fp = (
            sum(
                row[
                    index
                ]
                for row
                in matrix
            )
            -
            tp
        )

        tn = (
            evaluation_rows
            -
            tp
            -
            fn
            -
            fp
        )

        support = (
            tp
            +
            fn
        )

        precision = (
            0.0
            if (
                tp
                +
                fp
            )
            ==
            0
            else
            tp
            /
            (
                tp
                +
                fp
            )
        )

        recall = (
            0.0
            if support
            ==
            0
            else
            tp
            /
            support
        )

        f1 = (
            0.0
            if (
                precision
                +
                recall
            )
            ==
            0.0
            else
            2.0
            *
            precision
            *
            recall
            /
            (
                precision
                +
                recall
            )
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
                    tp,

                false_positive=
                    fp,

                false_negative=
                    fn,

                true_negative=
                    tn,
            )
        )


    macro_precision = (
        sum(
            item.precision
            for item
            in per_class
        )
        /
        2.0
    )

    macro_recall = (
        sum(
            item.recall
            for item
            in per_class
        )
        /
        2.0
    )

    macro_f1 = (
        sum(
            item.f1
            for item
            in per_class
        )
        /
        2.0
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
                "target",

            estimator_key=
                "logistic_regression",

            preparation_session_revision=(
                PREPARATION_REVISION
            ),

            training_contract_sha256=(
                ml_training_contract_sha256(
                    training_contract(
                        "classification"
                    )
                )
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
                0.75,

            balanced_accuracy=
                0.75,

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


def classification_model_metrics():

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
# ARTIFACT
# ============================================================


def build_artifact(
    problem_type: str,
) -> MLModelArtifactRecord:

    contract = (
        training_contract(
            problem_type
        )
    )

    metrics = (
        regression_model_metrics()
        if problem_type
        ==
        "regression"
        else
        classification_model_metrics()
    )

    holdout = (
        regression_holdout()
        if problem_type
        ==
        "regression"
        else
        classification_holdout()
    )

    (
        x_train,
        x_test,
        _,
        _,
    ) = holdout

    provenance = (
        MLExperimentProvenanceRecord(
            experiment_id=
                EXPERIMENT_ID,

            workflow_id=
                WORKFLOW_ID,

            dataset_id=
                DATASET_ID,

            preparation_session_revision=(
                PREPARATION_REVISION
            ),

            training_contract_sha256=(
                ml_training_contract_sha256(
                    contract
                )
            ),

            model_id=
                MODEL_ID,

            train_rows=
                len(
                    x_train
                ),

            test_rows=
                len(
                    x_test
                ),

            metrics=
                metrics,
        )
    )

    return (
        MLModelArtifactRecord(
            model_id=
                MODEL_ID,

            workflow_id=
                WORKFLOW_ID,

            dataset_id=
                DATASET_ID,

            training_contract=
                contract,

            experiment_provenance=
                provenance,

            metrics=
                metrics,

            train_rows=
                len(
                    x_train
                ),

            test_rows=
                len(
                    x_test
                ),

            created_at_utc=(
                "2026-08-29T00:00:00+00:00"
            ),

            serialization_format=
                "joblib",

            model_path=(
                "models/"
                "evaluation-summary.joblib"
            ),

            model_file_bytes=
                123,

            model_sha256=(
                "c"
                *
                64
            ),
        )
    )


# ============================================================
# EXPLAINABILITY
# ============================================================


def explainability_result(
    problem_type: str,
) -> MLModelExplainabilityResult:

    contract = (
        training_contract(
            problem_type
        )
    )

    return (
        MLModelExplainabilityResult(
            workflow_id=
                WORKFLOW_ID,

            dataset_id=
                DATASET_ID,

            model_id=
                MODEL_ID,

            experiment_id=
                EXPERIMENT_ID,

            problem_type=
                problem_type,

            estimator_key=(
                contract.estimator_key
            ),

            preparation_session_revision=(
                PREPARATION_REVISION
            ),

            training_contract_sha256=(
                ml_training_contract_sha256(
                    contract
                )
            ),

            scoring=(
                "neg_root_mean_squared_error"
                if problem_type
                ==
                "regression"
                else
                "f1_macro"
            ),

            n_repeats=
                10,

            random_seed=
                42,

            evaluation_rows=(
                2
                if problem_type
                ==
                "regression"
                else
                4
            ),

            feature_importances=[
                MLFeatureImportanceResult(
                    feature_name=
                        "feature",

                    rank=
                        1,

                    importance_mean=
                        0.10,

                    importance_std=
                        0.01,
                )
            ],
        )
    )


# ============================================================
# THRESHOLD
# ============================================================


def threshold_result(
) -> MLDecisionThresholdResult:

    contract = (
        training_contract(
            "classification"
        )
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
                "target",

            estimator_key=
                "logistic_regression",

            preparation_session_revision=(
                PREPARATION_REVISION
            ),

            training_contract_sha256=(
                ml_training_contract_sha256(
                    contract
                )
            ),

            evaluation_rows=
                4,

            threshold=
                0.70,

            negative_class_label=
                "negative",

            positive_class_label=
                "positive",

            confusion_matrix=[
                [
                    2,
                    0,
                ],
                [
                    1,
                    1,
                ],
            ],

            negative_support=
                2,

            positive_support=
                2,

            true_negative=
                2,

            false_positive=
                0,

            false_negative=
                1,

            true_positive=
                1,

            precision=
                1.0,

            recall=
                0.5,

            f1=(
                2.0
                /
                3.0
            ),

            specificity=
                1.0,

            accuracy=
                0.75,

            balanced_accuracy=
                0.75,

            positive_prediction_rate=
                0.25,
        )
    )


# ============================================================
# EXPECTED BASELINE
# ============================================================


def expected_baseline(
    problem_type: str,
):

    holdout = (
        regression_holdout()
        if problem_type
        ==
        "regression"
        else
        classification_holdout()
    )

    (
        _,
        _,
        y_train,
        y_test,
    ) = holdout

    bundle = (
        build_ml_baseline_predictions(
            problem_type=
                problem_type,

            y_train=
                y_train,

            test_rows=
                len(
                    y_test
                ),
        )
    )

    if problem_type == "regression":

        richer = (
            _regression_metrics(
                y_true=
                    y_test,

                predictions=
                    bundle.predictions,
            )
        )

        model_metrics = (
            regression_model_metrics()
        )

    else:

        richer = (
            _classification_metrics(
                y_true=
                    y_test,

                predictions=
                    bundle.predictions,
            )
        )

        model_metrics = (
            classification_model_metrics()
        )

    metrics = (
        _baseline_metrics_v0_1(
            problem_type=
                problem_type,

            metrics=
                richer,
        )
    )

    baseline = (
        build_ml_baseline_evaluation(
            problem_type=
                problem_type,

            strategy=
                bundle.strategy,

            metrics=
                metrics,

            train_rows=
                len(
                    y_train
                ),

            test_rows=
                len(
                    y_test
                ),
        )
    )

    comparison = (
        compare_model_to_baseline(
            problem_type=
                problem_type,

            model_metrics=
                model_metrics,

            baseline_metrics=
                baseline.metrics,
        )
    )

    return (
        baseline,
        comparison,
    )


# ============================================================
# FAKE RUNTIME
# ============================================================


class NeverFitEstimator:

    def __init__(
        self,
    ) -> None:

        self.fit_calls = 0


    def fit(
        self,
        *args,
        **kwargs,
    ):

        self.fit_calls += 1

        raise AssertionError(
            (
                "Model Evaluation Summary must "
                "never fit or refit."
            )
        )


def run_fake_runtime(
    problem_type: str,
    *,
    summary_contract=None,
    current_revision=PREPARATION_REVISION,
    artifact=None,
    selection_context=None,
):

    if artifact is None:

        artifact = (
            build_artifact(
                problem_type
            )
        )

    holdout = (
        regression_holdout()
        if problem_type
        ==
        "regression"
        else
        classification_holdout()
    )

    (
        x_train,
        x_test,
        y_train,
        y_test,
    ) = holdout

    estimator = (
        NeverFitEstimator()
    )

    loaded = (
        LoadedMLModel(
            artifact=
                artifact,

            estimator=
                estimator,
        )
    )

    calls = {
        "explainability":
            0,

        "diagnostics":
            0,

        "threshold":
            0,
    }

    originals = {
        "loader":
            summary_executor
            .load_trusted_ml_model,

        "dataframe":
            summary_executor
            ._load_authorized_dataframe,

        "xy":
            summary_executor
            ._validate_and_extract_xy,

        "split":
            summary_executor
            ._split_dataset,

        "explainability":
            summary_executor
            .execute_ml_model_explainability,

        "diagnostics":
            summary_executor
            .execute_ml_classification_diagnostics,

        "threshold":
            summary_executor
            .execute_ml_decision_threshold,
    }

    summary_executor.load_trusted_ml_model = (
        lambda **kwargs:
            loaded
    )

    summary_executor._load_authorized_dataframe = (
        lambda **kwargs:
            (
                pd.DataFrame(
                    {
                        "feature":
                            list(
                                range(
                                    len(
                                        x_train
                                    )
                                    +
                                    len(
                                        x_test
                                    )
                                )
                            ),

                        "target":
                            list(
                                y_train
                            )
                            +
                            list(
                                y_test
                            ),
                    }
                ),
                current_revision,
            )
    )

    summary_executor._validate_and_extract_xy = (
        lambda **kwargs:
            (
                pd.concat(
                    [
                        x_train,
                        x_test,
                    ],
                    ignore_index=True,
                ),

                pd.concat(
                    [
                        y_train,
                        y_test,
                    ],
                    ignore_index=True,
                ),
            )
    )

    summary_executor._split_dataset = (
        lambda **kwargs:
            (
                x_train,
                x_test,
                y_train,
                y_test,
            )
    )

    def fake_explainability(
        **kwargs,
    ):

        calls[
            "explainability"
        ] += 1

        return (
            explainability_result(
                problem_type
            )
        )

    def fake_diagnostics(
        **kwargs,
    ):

        calls[
            "diagnostics"
        ] += 1

        return (
            classification_diagnostics()
        )

    def fake_threshold(
        **kwargs,
    ):

        calls[
            "threshold"
        ] += 1

        return (
            threshold_result()
        )

    summary_executor.execute_ml_model_explainability = (
        fake_explainability
    )

    summary_executor.execute_ml_classification_diagnostics = (
        fake_diagnostics
    )

    summary_executor.execute_ml_decision_threshold = (
        fake_threshold
    )

    if summary_contract is None:

        summary_contract = (
            MLModelEvaluationSummaryContract()
        )

    try:

        result = (
            execute_ml_model_evaluation_summary(
                workflow_id=
                    WORKFLOW_ID,

                model_id=
                    MODEL_ID,

                summary_contract=(
                    summary_contract
                ),

                selection_context=(
                    selection_context
                ),
            )
        )

    finally:

        summary_executor.load_trusted_ml_model = (
            originals[
                "loader"
            ]
        )

        summary_executor._load_authorized_dataframe = (
            originals[
                "dataframe"
            ]
        )

        summary_executor._validate_and_extract_xy = (
            originals[
                "xy"
            ]
        )

        summary_executor._split_dataset = (
            originals[
                "split"
            ]
        )

        summary_executor.execute_ml_model_explainability = (
            originals[
                "explainability"
            ]
        )

        summary_executor.execute_ml_classification_diagnostics = (
            originals[
                "diagnostics"
            ]
        )

        summary_executor.execute_ml_decision_threshold = (
            originals[
                "threshold"
            ]
        )

    return (
        result,
        calls,
        estimator,
    )


# ============================================================
# HAPPY PATH
# ============================================================


def test_regression_standalone_summary(
) -> None:

    (
        result,
        calls,
        estimator,
    ) = (
        run_fake_runtime(
            "regression"
        )
    )

    expected, expected_comparison = (
        expected_baseline(
            "regression"
        )
    )

    assert (
        result.problem_type
        ==
        "regression"
    )

    assert (
        result.baseline
        ==
        expected
    )

    assert (
        result.baseline_comparison
        ==
        expected_comparison
    )

    assert (
        result.selection_evidence.source
        ==
        "standalone_model"
    )

    assert (
        result.classification_diagnostics
        is None
    )

    assert (
        result.decision_threshold_evaluation
        is None
    )

    assert calls == {
        "explainability":
            1,

        "diagnostics":
            0,

        "threshold":
            0,
    }

    assert (
        estimator.fit_calls
        ==
        0
    )


def test_classification_summary_without_threshold(
) -> None:

    (
        result,
        calls,
        estimator,
    ) = (
        run_fake_runtime(
            "classification"
        )
    )

    assert (
        result.classification_diagnostics
        is not None
    )

    assert (
        result.decision_threshold_evaluation
        is None
    )

    assert calls == {
        "explainability":
            1,

        "diagnostics":
            1,

        "threshold":
            0,
    }

    assert (
        estimator.fit_calls
        ==
        0
    )


def test_explicit_threshold_is_delegated_once(
) -> None:

    contract = (
        MLModelEvaluationSummaryContract(
            decision_threshold=(
                MLDecisionThresholdContract(
                    threshold=
                        0.70
                )
            )
        )
    )

    (
        result,
        calls,
        _,
    ) = (
        run_fake_runtime(
            "classification",
            summary_contract=
                contract,
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

    assert (
        calls[
            "threshold"
        ]
        ==
        1
    )


# ============================================================
# REQUEST / REVISION GATES
# ============================================================


def test_regression_threshold_fails_before_posthoc_evidence(
) -> None:

    contract = (
        MLModelEvaluationSummaryContract(
            decision_threshold=(
                MLDecisionThresholdContract(
                    threshold=
                        0.50
                )
            )
        )
    )

    expect_exception(
        MLModelEvaluationSummaryInputError,
        lambda:
            run_fake_runtime(
                "regression",
                summary_contract=
                    contract,
            ),
    )


def test_stale_preparation_revision_fails_closed(
) -> None:

    expect_exception(
        MLModelEvaluationSummaryInputError,
        lambda:
            run_fake_runtime(
                "regression",
                current_revision=(
                    PREPARATION_REVISION
                    +
                    1
                ),
            ),
    )


# ============================================================
# ARTIFACT PROVENANCE
# ============================================================


def test_training_contract_sha_is_recomputed(
) -> None:

    artifact = (
        build_artifact(
            "regression"
        )
    )

    provenance = (
        artifact
        .experiment_provenance
    )

    assert (
        provenance
        is not None
    )

    bad_provenance = (
        provenance.model_copy(
            update={
                "training_contract_sha256":
                    (
                        "d"
                        *
                        64
                    )
            }
        )
    )

    bad_artifact = (
        artifact.model_copy(
            update={
                "experiment_provenance":
                    bad_provenance
            }
        )
    )

    expect_exception(
        MLModelEvaluationSummaryArtifactError,
        lambda:
            run_fake_runtime(
                "regression",
                artifact=
                    bad_artifact,
            ),
    )


def test_provenance_metrics_bind_artifact(
) -> None:

    artifact = (
        build_artifact(
            "regression"
        )
    )

    provenance = (
        artifact
        .experiment_provenance
    )

    assert provenance is not None

    bad_provenance = (
        provenance.model_copy(
            update={
                "metrics":
                    {
                        **provenance.metrics,
                        "rmse":
                            999.0,
                    }
            }
        )
    )

    bad_artifact = (
        artifact.model_copy(
            update={
                "experiment_provenance":
                    bad_provenance
            }
        )
    )

    expect_exception(
        MLModelEvaluationSummaryArtifactError,
        lambda:
            run_fake_runtime(
                "regression",
                artifact=
                    bad_artifact,
            ),
    )


# ============================================================
# SELECTION CONTEXT TYPE
# ============================================================


def test_unknown_selection_context_is_not_inferred(
) -> None:

    expect_exception(
        MLModelEvaluationSummarySelectionError,
        lambda:
            run_fake_runtime(
                "regression",
                selection_context={
                    "source":
                        "model_comparison"
                },
            ),
    )


# ============================================================
# BASELINE
# ============================================================


def test_classification_baseline_is_reconstructed(
) -> None:

    (
        result,
        _,
        _,
    ) = (
        run_fake_runtime(
            "classification"
        )
    )

    expected, comparison = (
        expected_baseline(
            "classification"
        )
    )

    assert (
        result.baseline
        ==
        expected
    )

    assert (
        result.baseline_comparison
        ==
        comparison
    )


# ============================================================
# PRIVACY / LIMITATIONS
# ============================================================


def all_keys(
    value,
) -> set[
    str
]:

    keys = set()

    if isinstance(value, dict):

        for key, nested in value.items():

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

    (
        result,
        _,
        _,
    ) = (
        run_fake_runtime(
            "classification"
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
        "probabilities",
        "scores",
        "decision_scores",
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
                result.model_dump(
                    mode="json"
                )
            )
        )
    )


def test_limitations_are_server_owned(
) -> None:

    (
        regression,
        _,
        _,
    ) = (
        run_fake_runtime(
            "regression"
        )
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

    contract = (
        MLModelEvaluationSummaryContract(
            decision_threshold=(
                MLDecisionThresholdContract(
                    threshold=
                        0.70
                )
            )
        )
    )

    (
        classification,
        _,
        _,
    ) = (
        run_fake_runtime(
            "classification",
            summary_contract=
                contract,
        )
    )

    assert (
        classification.limitations
        ==
        [
            "single_holdout_evaluation",
            "no_external_validation",
            "feature_importance_not_causal",
            "selection_evidence_not_available",
            "requested_threshold_not_optimized",
        ]
    )


# ============================================================
# RULE VERSION
# ============================================================


def test_rule_version(
) -> None:

    assert (
        ML_MODEL_EVALUATION_SUMMARY_EXECUTOR_RULE_VERSION
        ==
        "ml_model_evaluation_summary_executor_v0.1"
    )


# ============================================================
# MAIN
# ============================================================


def main(
) -> None:

    print(
        "=== DATALENS ML MODEL EVALUATION SUMMARY EXECUTOR v0.1 ==="
    )

    tests = [
        (
            "Regression standalone summary",
            test_regression_standalone_summary,
        ),
        (
            "Classification summary without threshold",
            test_classification_summary_without_threshold,
        ),
        (
            "Explicit threshold delegated exactly once",
            test_explicit_threshold_is_delegated_once,
        ),
        (
            "Regression threshold blocked",
            test_regression_threshold_fails_before_posthoc_evidence,
        ),
        (
            "Stale Preparation revision blocked",
            test_stale_preparation_revision_fails_closed,
        ),
        (
            "Training Contract SHA recomputed",
            test_training_contract_sha_is_recomputed,
        ),
        (
            "Provenance metrics bind artifact",
            test_provenance_metrics_bind_artifact,
        ),
        (
            "Unknown selection context not inferred",
            test_unknown_selection_context_is_not_inferred,
        ),
        (
            "Classification baseline reconstructed",
            test_classification_baseline_is_reconstructed,
        ),
        (
            "Summary result privacy-minimal",
            test_result_is_privacy_minimal,
        ),
        (
            "Limitations server-owned",
            test_limitations_are_server_owned,
        ),
        (
            "Executor rule version",
            test_rule_version,
        ),
    ]

    for label, test in tests:

        test()

        print(
            f"[PASS] {label}"
        )

    print()

    print(
        "PASS - ML Model Evaluation Summary Executor v0.1"
    )


if __name__ == "__main__":
    main()
