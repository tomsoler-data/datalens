from __future__ import annotations


import math


import numpy as np


from fastapi.testclient import (
    TestClient,
)


from sklearn.metrics import (
    confusion_matrix,
)


# ============================================================
# REAL ISOLATED PRODUCT ENVIRONMENT
#
# Import this Golden Path first because it establishes the
# isolated SQLite / Preparation / Model Artifact environment.
# ============================================================


from tests.e2e.test_ml_classification_diagnostics_golden_path_v0_1 import (
    WORKFLOW_ROOT_DATASET_ID,
    app,
    build_real_classification_contract,
    create_preparation_session,
    ml_model_artifact_count,
    reconstruct_real_holdout,
    reset_product_state,
    run_real_cleaning_plan,
    run_real_quality,
    select_analysis_output,
    train_real_classifier,
    validate_preparation,
    verify_preparation_persistence,
    verify_real_classifier_reload,
    verify_real_handoff,
)


# ============================================================
# DECISION THRESHOLD PRODUCTION MODULES
# ============================================================


from app.ml.decision_threshold import (
    ML_DECISION_THRESHOLD_RULE_VERSION,
    MLDecisionThresholdContract,
)


import app.ml.decision_threshold_executor as threshold_executor


from app.ml.decision_threshold_executor import (
    ML_DECISION_THRESHOLD_EXECUTOR_RULE_VERSION,
    execute_ml_decision_threshold,
)


from app.ml.experiment_provenance import (
    ml_training_contract_sha256,
)


from app.preparation.analysis_readiness_gate import (
    require_analysis_readiness,
)


# ============================================================
# VERSION
# ============================================================


ML_DECISION_THRESHOLD_GOLDEN_PATH_RULE_VERSION = (
    "ml_decision_threshold_golden_path_v0.1"
)


# ============================================================
# METRIC HELPERS
# ============================================================


def _safe_ratio(
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


def _f1(
    *,
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


def expected_values_from_matrix(
    matrix,
) -> dict[
    str,
    float
    |
    int
]:

    array = np.asarray(
        matrix,
        dtype=np.int64,
    )


    assert (
        array.shape
        ==
        (
            2,
            2,
        )
    )


    tn = int(
        array[
            0,
            0
        ]
    )


    fp = int(
        array[
            0,
            1
        ]
    )


    fn = int(
        array[
            1,
            0
        ]
    )


    tp = int(
        array[
            1,
            1
        ]
    )


    total = (
        tn
        +
        fp
        +
        fn
        +
        tp
    )


    negative_support = (
        tn
        +
        fp
    )


    positive_support = (
        fn
        +
        tp
    )


    precision = (
        _safe_ratio(
            tp,
            tp
            +
            fp,
        )
    )


    recall = (
        _safe_ratio(
            tp,
            tp
            +
            fn,
        )
    )


    specificity = (
        _safe_ratio(
            tn,
            tn
            +
            fp,
        )
    )


    f1 = (
        _f1(
            precision=
                precision,

            recall=
                recall,
        )
    )


    accuracy = (
        _safe_ratio(
            tp
            +
            tn,
            total,
        )
    )


    supported_recalls = []


    if (
        negative_support
        >
        0
    ):
        supported_recalls.append(
            specificity
        )


    if (
        positive_support
        >
        0
    ):
        supported_recalls.append(
            recall
        )


    balanced_accuracy = float(
        sum(
            supported_recalls
        )
        /
        len(
            supported_recalls
        )
    )


    positive_prediction_rate = (
        _safe_ratio(
            tp
            +
            fp,
            total,
        )
    )


    return {
        "negative_support":
            negative_support,

        "positive_support":
            positive_support,

        "true_negative":
            tn,

        "false_positive":
            fp,

        "false_negative":
            fn,

        "true_positive":
            tp,

        "precision":
            precision,

        "recall":
            recall,

        "f1":
            f1,

        "specificity":
            specificity,

        "accuracy":
            accuracy,

        "balanced_accuracy":
            balanced_accuracy,

        "positive_prediction_rate":
            positive_prediction_rate,
    }


# ============================================================
# REAL THRESHOLD EXECUTION
# ============================================================


def run_real_threshold_evaluations(
    *,
    workflow_id: str,
    execution_result,
):

    artifacts_before = (
        ml_model_artifact_count(
            workflow_id=
                workflow_id
        )
    )


    assert (
        artifacts_before
        ==
        1
    )


    readiness_before = (
        require_analysis_readiness(
            workflow_id=
                workflow_id
        )
    )


    real_loader = (
        threshold_executor
        .load_trusted_ml_model
    )


    guarded_loads = []


    def guarded_loader(
        **kwargs,
    ):

        loaded = (
            real_loader(
                **kwargs
            )
        )


        def forbidden_fit(
            *args,
            **fit_kwargs,
        ):

            raise AssertionError(
                (
                    "Decision Threshold must never "
                    "fit or refit the trusted model."
                )
            )


        def forbidden_predict(
            *args,
            **predict_kwargs,
        ):

            raise AssertionError(
                (
                    "Decision Threshold must never "
                    "use native predict()."
                )
            )


        loaded.estimator.fit = (
            forbidden_fit
        )


        loaded.estimator.predict = (
            forbidden_predict
        )


        guarded_loads.append(
            loaded
        )


        return loaded


    threshold_executor.load_trusted_ml_model = (
        guarded_loader
    )


    try:

        threshold_050 = (
            execute_ml_decision_threshold(
                workflow_id=
                    workflow_id,

                model_id=(
                    execution_result
                    .model_artifact
                    .model_id
                ),

                threshold_contract=(
                    MLDecisionThresholdContract(
                        threshold=
                            0.50
                    )
                ),
            )
        )


        threshold_070 = (
            execute_ml_decision_threshold(
                workflow_id=
                    workflow_id,

                model_id=(
                    execution_result
                    .model_artifact
                    .model_id
                ),

                threshold_contract=(
                    MLDecisionThresholdContract(
                        threshold=
                            0.70
                    )
                ),
            )
        )


        threshold_050_repeat = (
            execute_ml_decision_threshold(
                workflow_id=
                    workflow_id,

                model_id=(
                    execution_result
                    .model_artifact
                    .model_id
                ),

                threshold_contract=(
                    MLDecisionThresholdContract(
                        threshold=
                            0.50
                    )
                ),
            )
        )


    finally:

        threshold_executor.load_trusted_ml_model = (
            real_loader
        )


    readiness_after = (
        require_analysis_readiness(
            workflow_id=
                workflow_id
        )
    )


    artifacts_after = (
        ml_model_artifact_count(
            workflow_id=
                workflow_id
        )
    )


    assert (
        len(
            guarded_loads
        )
        ==
        3
    )


    assert (
        artifacts_after
        ==
        artifacts_before
        ==
        1
    )


    assert (
        readiness_after
        .session_revision
        ==
        readiness_before
        .session_revision
    )


    assert (
        threshold_050
        .model_dump(
            mode="json"
        )
        ==
        threshold_050_repeat
        .model_dump(
            mode="json"
        )
    )


    print(
        (
            "[PASS] Decision Threshold crossed real trusted "
            "Model Artifact reload"
        )
    )


    print(
        (
            "[PASS] real threshold evaluation completed with "
            "fit() and native predict() explicitly forbidden"
        )
    )


    print(
        (
            "[PASS] thresholds 0.50 and 0.70 evaluated against "
            "the same persisted classifier"
        )
    )


    print(
        (
            "[PASS] repeated threshold 0.50 evaluation is "
            "exactly deterministic"
        )
    )


    print(
        (
            "[PASS] threshold evaluation created zero additional "
            "Model Artifacts / Experiments"
        )
    )


    print(
        (
            "[PASS] threshold evaluation did not mutate "
            "Preparation revision"
        )
    )


    return (
        threshold_050,
        threshold_070,
    )


# ============================================================
# AUTHORITY
# ============================================================


def verify_result_authority(
    *,
    workflow_id: str,
    training_contract,
    execution_result,
    result,
    expected_threshold: float,
) -> None:

    provenance = (
        execution_result
        .experiment_provenance
    )


    assert (
        result.workflow_id
        ==
        workflow_id
    )


    assert (
        result.dataset_id
        ==
        WORKFLOW_ROOT_DATASET_ID
    )


    assert (
        result.model_id
        ==
        execution_result
        .model_artifact
        .model_id
    )


    assert (
        result.experiment_id
        ==
        provenance
        .experiment_id
    )


    assert (
        result.problem_type
        ==
        "classification"
    )


    assert (
        result.target_column
        ==
        "segment"
    )


    assert (
        result.estimator_key
        ==
        "logistic_regression"
    )


    assert (
        result
        .preparation_session_revision
        ==
        provenance
        .preparation_session_revision
    )


    assert (
        result
        .training_contract_sha256
        ==
        ml_training_contract_sha256(
            training_contract
        )
    )


    assert math.isclose(
        result.threshold,
        expected_threshold,
        rel_tol=0.0,
        abs_tol=0.0,
    )


    assert (
        result.method
        ==
        "holdout_binary_decision_threshold"
    )


    assert (
        result.score_source
        ==
        "predict_proba"
    )


    assert (
        result.positive_class_policy
        ==
        "estimator_classes_index_1"
    )


    assert (
        result.comparison_operator
        ==
        "greater_than_or_equal"
    )


    assert (
        result.threshold_selection_policy
        ==
        "evaluate_requested_threshold_only"
    )


    assert (
        result.zero_division_policy
        ==
        "zero"
    )


# ============================================================
# INDEPENDENT REAL PROBABILITY CHECK
# ============================================================


def verify_real_thresholds(
    *,
    training_contract,
    execution_result,
    loaded_model,
    result_050,
    result_070,
) -> None:

    (
        preparation_revision,
        x_test,
        y_test,
    ) = (
        reconstruct_real_holdout(
            training_contract=
                training_contract
        )
    )


    assert (
        preparation_revision
        ==
        execution_result
        .experiment_provenance
        .preparation_session_revision
    )


    classifier = (
        loaded_model
        .estimator
        .named_steps[
            "estimator"
        ]
    )


    raw_classes = [
        value.item()
        if isinstance(
            value,
            np.generic,
        )
        else value

        for value
        in np.asarray(
            classifier.classes_,
            dtype=object,
        ).tolist()
    ]


    assert (
        len(
            raw_classes
        )
        ==
        2
    )


    probabilities = np.asarray(
        loaded_model
        .estimator
        .predict_proba(
            x_test
        ),
        dtype=np.float64,
    )


    assert (
        probabilities.shape
        ==
        (
            6,
            2,
        )
    )


    assert (
        np.isfinite(
            probabilities
        )
        .all()
    )


    assert np.allclose(
        probabilities.sum(
            axis=1
        ),
        np.ones(
            6,
            dtype=np.float64,
        ),
        rtol=1e-9,
        atol=1e-9,
    )


    assert (
        result_050
        .negative_class_label
        ==
        str(
            raw_classes[
                0
            ]
        )
    )


    assert (
        result_050
        .positive_class_label
        ==
        str(
            raw_classes[
                1
            ]
        )
    )


    assert (
        result_070
        .negative_class_label
        ==
        str(
            raw_classes[
                0
            ]
        )
    )


    assert (
        result_070
        .positive_class_label
        ==
        str(
            raw_classes[
                1
            ]
        )
    )


    for (
        threshold,
        result,
    ) in [
        (
            0.50,
            result_050,
        ),
        (
            0.70,
            result_070,
        ),
    ]:

        thresholded_predictions = (
            np.where(
                (
                    probabilities[
                        :,
                        1
                    ]
                    >=
                    threshold
                ),
                raw_classes[
                    1
                ],
                raw_classes[
                    0
                ],
            )
        )


        matrix = (
            confusion_matrix(
                y_test,
                thresholded_predictions,
                labels=
                    raw_classes,
            )
        )


        expected_matrix = [
            [
                int(
                    matrix[
                        0,
                        0
                    ]
                ),
                int(
                    matrix[
                        0,
                        1
                    ]
                ),
            ],
            [
                int(
                    matrix[
                        1,
                        0
                    ]
                ),
                int(
                    matrix[
                        1,
                        1
                    ]
                ),
            ],
        ]


        expected = (
            expected_values_from_matrix(
                expected_matrix
            )
        )


        assert (
            result.confusion_matrix
            ==
            expected_matrix
        )


        assert (
            result.evaluation_rows
            ==
            6
        )


        for field_name in [
            "negative_support",
            "positive_support",
            "true_negative",
            "false_positive",
            "false_negative",
            "true_positive",
        ]:

            assert (
                getattr(
                    result,
                    field_name,
                )
                ==
                expected[
                    field_name
                ]
            )


        for field_name in [
            "precision",
            "recall",
            "f1",
            "specificity",
            "accuracy",
            "balanced_accuracy",
            "positive_prediction_rate",
        ]:

            assert math.isclose(
                float(
                    getattr(
                        result,
                        field_name,
                    )
                ),
                float(
                    expected[
                        field_name
                    ]
                ),
                rel_tol=1e-12,
                abs_tol=1e-12,
            )


    print(
        (
            "[PASS] real predict_proba() produced exact "
            "six-row binary holdout probabilities"
        )
    )


    print(
        (
            "[PASS] positive class is exactly fitted "
            "LogisticRegression classes_[1]"
        )
    )


    print(
        (
            "[PASS] threshold 0.50 confusion matrix and metrics "
            "match independent probability reconstruction"
        )
    )


    print(
        (
            "[PASS] threshold 0.70 confusion matrix and metrics "
            "match independent probability reconstruction"
        )
    )


# ============================================================
# NATIVE 0.50 CONSISTENCY
# ============================================================


def verify_native_threshold_consistency(
    *,
    execution_result,
    result_050,
) -> None:

    persisted_metrics = (
        execution_result
        .model_artifact
        .metrics
    )


    assert math.isclose(
        result_050.accuracy,
        float(
            persisted_metrics[
                "accuracy"
            ]
        ),
        rel_tol=1e-12,
        abs_tol=1e-12,
    )


    assert math.isclose(
        result_050.balanced_accuracy,
        float(
            persisted_metrics[
                "balanced_accuracy"
            ]
        ),
        rel_tol=1e-12,
        abs_tol=1e-12,
    )


    print(
        (
            "[PASS] explicit probability threshold 0.50 "
            "matches persisted native classifier accuracy "
            "and balanced accuracy"
        )
    )


# ============================================================
# PRIVACY
# ============================================================


def _all_keys(
    value,
) -> set[
    str
]:

    keys = set()


    if isinstance(
        value,
        dict,
    ):

        for (
            key,
            nested,
        ) in value.items():

            keys.add(
                str(
                    key
                )
            )


            keys.update(
                _all_keys(
                    nested
                )
            )


    elif isinstance(
        value,
        list,
    ):

        for nested in value:

            keys.update(
                _all_keys(
                    nested
                )
            )


    return keys


def verify_privacy_minimal(
    *,
    result,
) -> None:

    payload = (
        result
        .model_dump(
            mode="json"
        )
    )


    forbidden = {
        "rows",
        "raw_rows",
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
        "model_bytes",
        "model_path",
        "estimator",
        "training_contract",
    }


    assert (
        forbidden.isdisjoint(
            _all_keys(
                payload
            )
        )
    )


# ============================================================
# RULE VERSIONS
# ============================================================


def verify_rule_versions(
) -> None:

    assert (
        ML_DECISION_THRESHOLD_RULE_VERSION
        ==
        "ml_decision_threshold_v0.1"
    )


    assert (
        ML_DECISION_THRESHOLD_EXECUTOR_RULE_VERSION
        ==
        "ml_decision_threshold_executor_v0.1"
    )


    assert (
        ML_DECISION_THRESHOLD_GOLDEN_PATH_RULE_VERSION
        ==
        "ml_decision_threshold_golden_path_v0.1"
    )


    print(
        "[PASS] Decision Threshold rule versions"
    )


# ============================================================
# GOLDEN PATH
# ============================================================


def test_ml_decision_threshold_golden_path_v0_1(
) -> None:

    reset_product_state()


    with TestClient(
        app
    ) as client:

        # ----------------------------------------------------
        # REAL PREPARATION
        # ----------------------------------------------------

        workflow_id = (
            create_preparation_session(
                client
            )
        )


        run_real_quality(
            client,
            workflow_id=
                workflow_id,
        )


        run_real_cleaning_plan(
            client,
            workflow_id=
                workflow_id,
        )


        select_analysis_output(
            client,
            workflow_id=
                workflow_id,
        )


        validate_preparation(
            client,
            workflow_id=
                workflow_id,
        )


        verify_preparation_persistence(
            workflow_id=
                workflow_id,
        )


        verify_real_handoff(
            workflow_id=
                workflow_id,
        )


        # ----------------------------------------------------
        # REAL CLASSIFICATION TRAINING
        # ----------------------------------------------------

        training_contract = (
            build_real_classification_contract(
                workflow_id=
                    workflow_id
            )
        )


        execution_result = (
            train_real_classifier(
                workflow_id=
                    workflow_id,

                training_contract=
                    training_contract,
            )
        )


        # ----------------------------------------------------
        # REAL TRUSTED RELOAD
        # ----------------------------------------------------

        loaded_model = (
            verify_real_classifier_reload(
                workflow_id=
                    workflow_id,

                execution_result=
                    execution_result,
            )
        )


        # ----------------------------------------------------
        # REAL THRESHOLD EVALUATION
        # ----------------------------------------------------

        (
            result_050,
            result_070,
        ) = (
            run_real_threshold_evaluations(
                workflow_id=
                    workflow_id,

                execution_result=
                    execution_result,
            )
        )


        verify_result_authority(
            workflow_id=
                workflow_id,

            training_contract=
                training_contract,

            execution_result=
                execution_result,

            result=
                result_050,

            expected_threshold=
                0.50,
        )


        verify_result_authority(
            workflow_id=
                workflow_id,

            training_contract=
                training_contract,

            execution_result=
                execution_result,

            result=
                result_070,

            expected_threshold=
                0.70,
        )


        verify_real_thresholds(
            training_contract=
                training_contract,

            execution_result=
                execution_result,

            loaded_model=
                loaded_model,

            result_050=
                result_050,

            result_070=
                result_070,
        )


        verify_native_threshold_consistency(
            execution_result=
                execution_result,

            result_050=
                result_050,
        )


        verify_privacy_minimal(
            result=
                result_050
        )


        verify_privacy_minimal(
            result=
                result_070
        )


        print(
            (
                "[PASS] real threshold results remain "
                "privacy-minimal"
            )
        )


        verify_rule_versions()


# ============================================================
# MAIN
# ============================================================


def main(
) -> None:

    print()


    print(
        "="
        *
        78
    )


    print(
        "DATALENS ML DECISION THRESHOLD GOLDEN PATH E2E v0.1"
    )


    print(
        "="
        *
        78
    )


    print(
        "Preparation : real validated 30-row mixed CSV"
    )


    print(
        "Target      : segment = premium / standard"
    )


    print(
        "Estimator   : real LogisticRegression"
    )


    print(
        "Holdout     : deterministic stratified 24/6"
    )


    print(
        "Scores      : real predict_proba(x_test) only"
    )


    print(
        "Thresholds  : explicit 0.50 + 0.70"
    )


    print(
        "Positive    : fitted estimator classes_[1]"
    )


    print(
        "Persistence : exactly one training Model Artifact"
    )


    print(
        "Safety      : no fit / no refit / no executor predict()"
    )


    print()


    test_ml_decision_threshold_golden_path_v0_1()


    print()


    print(
        "="
        *
        78
    )


    print(
        (
            "PASS - Preparation -> Classification Training -> "
            "1 Model Artifact + Experiment Provenance -> "
            "Trusted Reload -> Exact 24/6 Holdout -> "
            "predict_proba Only -> Explicit Thresholds "
            "0.50 / 0.70 -> Binary Metrics -> "
            "Deterministic Repeat -> No Refit -> "
            "No Additional Persistence"
        )
    )


    print(
        "="
        *
        78
    )


if __name__ == "__main__":
    main()
