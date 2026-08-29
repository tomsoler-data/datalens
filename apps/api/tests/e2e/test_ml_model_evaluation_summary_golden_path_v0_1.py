from __future__ import annotations


import math


from fastapi.testclient import (
    TestClient,
)


# ============================================================
# REAL ISOLATED PRODUCT ENVIRONMENT
#
# Reuse the established classification Golden Path:
#
# Preparation
# -> real LogisticRegression
# -> persisted Model Artifact + Experiment Provenance
# -> trusted reload
# ============================================================


from tests.e2e.test_ml_classification_diagnostics_golden_path_v0_1 import (
    WORKFLOW_ROOT_DATASET_ID,
    app,
    build_real_classification_contract,
    create_preparation_session,
    ml_model_artifact_count,
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
# PRODUCT IMPORTS
# ============================================================


from app.ml.decision_threshold import (
    MLDecisionThresholdContract,
)


from app.ml.experiment_provenance import (
    ml_training_contract_sha256,
)


from app.ml.model_evaluation_summary import (
    ML_MODEL_EVALUATION_SUMMARY_RULE_VERSION,
    ML_MODEL_SELECTION_EVIDENCE_RULE_VERSION,
    MLModelEvaluationSummaryContract,
)


from app.ml.model_evaluation_summary_executor import (
    ML_MODEL_EVALUATION_SUMMARY_EXECUTOR_RULE_VERSION,
    execute_ml_model_evaluation_summary,
)


from app.preparation.analysis_readiness_gate import (
    require_analysis_readiness,
)


# ============================================================
# VERSION
# ============================================================


ML_MODEL_EVALUATION_SUMMARY_GOLDEN_PATH_RULE_VERSION = (
    "ml_model_evaluation_summary_golden_path_v0.1"
)


# ============================================================
# REAL SUMMARY
# ============================================================


def run_real_summary(
    *,
    workflow_id: str,
    training_result,
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


    result = (
        execute_ml_model_evaluation_summary(
            workflow_id=
                workflow_id,

            model_id=(
                training_result
                .model_artifact
                .model_id
            ),

            summary_contract=
                contract,
        )
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


    print(
        (
            "[PASS] real Model Evaluation Summary crossed "
            "trusted persisted Model Artifact authority"
        )
    )


    print(
        (
            "[PASS] complete Summary created zero additional "
            "Model Artifacts / Experiments"
        )
    )


    print(
        (
            "[PASS] complete Summary did not mutate "
            "Preparation revision"
        )
    )


    return result


# ============================================================
# AUTHORITY
# ============================================================


def verify_summary_authority(
    *,
    workflow_id: str,
    training_contract,
    training_result,
    summary_result,
) -> None:

    artifact = (
        training_result
        .model_artifact
    )


    provenance = (
        training_result
        .experiment_provenance
    )


    assert (
        summary_result.workflow_id
        ==
        workflow_id
    )


    assert (
        summary_result.dataset_id
        ==
        WORKFLOW_ROOT_DATASET_ID
    )


    assert (
        summary_result.model_id
        ==
        artifact.model_id
    )


    assert (
        summary_result.experiment_id
        ==
        provenance.experiment_id
    )


    assert (
        summary_result.problem_type
        ==
        "classification"
    )


    assert (
        summary_result.target_column
        ==
        training_contract.target_column
    )


    assert (
        summary_result.estimator_key
        ==
        training_contract.estimator_key
        ==
        "logistic_regression"
    )


    assert (
        summary_result
        .preparation_session_revision
        ==
        provenance
        .preparation_session_revision
    )


    assert (
        summary_result
        .training_contract_sha256
        ==
        ml_training_contract_sha256(
            training_contract
        )
    )


    assert (
        summary_result.train_rows
        ==
        training_result.train_rows
        ==
        24
    )


    assert (
        summary_result.test_rows
        ==
        training_result.test_rows
        ==
        6
    )


    assert (
        summary_result.metrics
        ==
        artifact.metrics
        ==
        provenance.metrics
    )


    print(
        (
            "[PASS] Summary identity is bound to workflow / "
            "dataset / model / experiment / Training Contract SHA"
        )
    )


    print(
        (
            "[PASS] Summary preserves exact persisted "
            "24/6 holdout metric surface"
        )
    )


# ============================================================
# BASELINE EVIDENCE
# ============================================================


def verify_baseline(
    *,
    training_result,
    summary_result,
) -> None:

    baseline = (
        summary_result.baseline
    )


    comparison = (
        summary_result
        .baseline_comparison
    )


    assert (
        baseline.problem_type
        ==
        "classification"
    )


    assert (
        baseline.strategy
        ==
        "majority_train_class"
    )


    assert (
        baseline.primary_metric
        ==
        "f1_macro"
    )


    assert (
        baseline.train_rows
        ==
        24
    )


    assert (
        baseline.test_rows
        ==
        6
    )


    assert (
        comparison.problem_type
        ==
        "classification"
    )


    assert (
        comparison.primary_metric
        ==
        "f1_macro"
    )


    assert math.isclose(
        float(
            comparison
            .model_primary_metric_value
        ),
        float(
            training_result
            .metrics[
                "f1_macro"
            ]
        ),
        rel_tol=1e-12,
        abs_tol=1e-12,
    )


    assert math.isclose(
        float(
            comparison
            .baseline_primary_metric_value
        ),
        float(
            baseline.metrics[
                "f1_macro"
            ]
        ),
        rel_tol=1e-12,
        abs_tol=1e-12,
    )


    print(
        (
            "[PASS] Summary reconstructed the real "
            "training-only majority-class baseline"
        )
    )


    print(
        (
            "[PASS] model-to-baseline comparison is bound "
            "to persisted final holdout F1 macro"
        )
    )


# ============================================================
# SELECTION EVIDENCE
# ============================================================


def verify_standalone_selection(
    *,
    summary_result,
) -> None:

    evidence = (
        summary_result
        .selection_evidence
    )


    assert (
        evidence.source
        ==
        "standalone_model"
    )


    assert (
        evidence.status
        ==
        "selection_not_available"
    )


    assert evidence.rank is None


    assert (
        evidence.selection_policy
        is None
    )


    assert (
        evidence.primary_metric
        is None
    )


    assert (
        evidence.primary_metric_value
        is None
    )


    assert (
        evidence.metric_scope
        ==
        "not_available"
    )


    print(
        (
            "[PASS] Summary does not invent upstream "
            "selection provenance when none was supplied"
        )
    )


# ============================================================
# CLASSIFICATION DIAGNOSTICS
# ============================================================


def verify_classification_diagnostics(
    *,
    training_result,
    summary_result,
) -> None:

    diagnostics = (
        summary_result
        .classification_diagnostics
    )


    assert (
        diagnostics
        is not None
    )


    assert (
        diagnostics.model_id
        ==
        summary_result.model_id
    )


    assert (
        diagnostics.experiment_id
        ==
        summary_result.experiment_id
    )


    assert (
        diagnostics
        .preparation_session_revision
        ==
        summary_result
        .preparation_session_revision
    )


    assert (
        diagnostics
        .training_contract_sha256
        ==
        summary_result
        .training_contract_sha256
    )


    assert (
        diagnostics.evaluation_rows
        ==
        summary_result.test_rows
        ==
        6
    )


    assert (
        diagnostics.class_count
        ==
        2
    )


    assert (
        len(
            diagnostics.class_labels
        )
        ==
        2
    )


    assert math.isclose(
        diagnostics.accuracy,
        float(
            training_result.metrics[
                "accuracy"
            ]
        ),
        rel_tol=1e-12,
        abs_tol=1e-12,
    )


    assert math.isclose(
        diagnostics.balanced_accuracy,
        float(
            training_result.metrics[
                "balanced_accuracy"
            ]
        ),
        rel_tol=1e-12,
        abs_tol=1e-12,
    )


    assert math.isclose(
        diagnostics
        .macro_average
        .precision,
        float(
            training_result.metrics[
                "precision_macro"
            ]
        ),
        rel_tol=1e-12,
        abs_tol=1e-12,
    )


    assert math.isclose(
        diagnostics
        .macro_average
        .recall,
        float(
            training_result.metrics[
                "recall_macro"
            ]
        ),
        rel_tol=1e-12,
        abs_tol=1e-12,
    )


    assert math.isclose(
        diagnostics
        .macro_average
        .f1,
        float(
            training_result.metrics[
                "f1_macro"
            ]
        ),
        rel_tol=1e-12,
        abs_tol=1e-12,
    )


    print(
        (
            "[PASS] Summary embedded real Classification "
            "Diagnostics on the exact persisted holdout"
        )
    )


    print(
        (
            "[PASS] diagnostic metrics exactly match "
            "persisted native classifier metrics"
        )
    )


# ============================================================
# DECISION THRESHOLD
# ============================================================


def verify_decision_threshold(
    *,
    summary_result,
) -> None:

    threshold = (
        summary_result
        .decision_threshold_evaluation
    )


    diagnostics = (
        summary_result
        .classification_diagnostics
    )


    assert threshold is not None
    assert diagnostics is not None


    assert math.isclose(
        threshold.threshold,
        0.70,
        rel_tol=0.0,
        abs_tol=0.0,
    )


    assert (
        threshold.evaluation_rows
        ==
        summary_result.test_rows
        ==
        6
    )


    assert (
        threshold.model_id
        ==
        summary_result.model_id
    )


    assert (
        threshold.experiment_id
        ==
        summary_result.experiment_id
    )


    assert (
        threshold
        .training_contract_sha256
        ==
        summary_result
        .training_contract_sha256
    )


    assert (
        [
            threshold
            .negative_class_label,

            threshold
            .positive_class_label,
        ]
        ==
        diagnostics.class_labels
    )


    assert (
        threshold.score_source
        ==
        "predict_proba"
    )


    assert (
        threshold
        .positive_class_policy
        ==
        "estimator_classes_index_1"
    )


    assert (
        threshold
        .comparison_operator
        ==
        "greater_than_or_equal"
    )


    assert (
        threshold
        .threshold_selection_policy
        ==
        "evaluate_requested_threshold_only"
    )


    print(
        (
            "[PASS] Summary embedded real explicit "
            "Decision Threshold 0.70 evaluation"
        )
    )


    print(
        (
            "[PASS] threshold class order is bound to "
            "Classification Diagnostics estimator classes"
        )
    )


# ============================================================
# EXPLAINABILITY
# ============================================================


def verify_explainability(
    *,
    training_contract,
    summary_result,
) -> None:

    explainability = (
        summary_result
        .explainability
    )


    assert (
        explainability.workflow_id
        ==
        summary_result.workflow_id
    )


    assert (
        explainability.dataset_id
        ==
        summary_result.dataset_id
    )


    assert (
        explainability.model_id
        ==
        summary_result.model_id
    )


    assert (
        explainability.experiment_id
        ==
        summary_result.experiment_id
    )


    assert (
        explainability.problem_type
        ==
        "classification"
    )


    assert (
        explainability.estimator_key
        ==
        "logistic_regression"
    )


    assert (
        explainability
        .preparation_session_revision
        ==
        summary_result
        .preparation_session_revision
    )


    assert (
        explainability
        .training_contract_sha256
        ==
        summary_result
        .training_contract_sha256
    )


    assert (
        explainability.evaluation_rows
        ==
        summary_result.test_rows
        ==
        6
    )


    assert (
        {
            item.feature_name

            for item
            in explainability
            .feature_importances
        }
        ==
        set(
            training_contract
            .feature_columns
        )
    )


    assert (
        [
            item.rank

            for item
            in explainability
            .feature_importances
        ]
        ==
        list(
            range(
                1,
                len(
                    training_contract
                    .feature_columns
                )
                +
                1,
            )
        )
    )


    print(
        (
            "[PASS] Summary embedded real deterministic "
            "holdout permutation importance"
        )
    )


    print(
        (
            "[PASS] explainability preserves exact original "
            "Training Contract feature identities"
        )
    )


# ============================================================
# LIMITATIONS
# ============================================================


def verify_limitations(
    *,
    summary_result,
) -> None:

    assert (
        summary_result.limitations
        ==
        [
            "single_holdout_evaluation",
            "no_external_validation",
            "feature_importance_not_causal",
            "selection_evidence_not_available",
            "requested_threshold_not_optimized",
        ]
    )


    print(
        (
            "[PASS] limitations are explicit and "
            "server-derived"
        )
    )


# ============================================================
# DETERMINISM
# ============================================================


def verify_deterministic_repeat(
    *,
    workflow_id: str,
    training_result,
    first_result,
) -> None:

    second_result = (
        execute_ml_model_evaluation_summary(
            workflow_id=
                workflow_id,

            model_id=(
                training_result
                .model_artifact
                .model_id
            ),

            summary_contract=(
                MLModelEvaluationSummaryContract(
                    decision_threshold=(
                        MLDecisionThresholdContract(
                            threshold=
                                0.70
                        )
                    )
                )
            ),
        )
    )


    assert (
        first_result
        .model_dump(
            mode="json"
        )
        ==
        second_result
        .model_dump(
            mode="json"
        )
    )


    assert (
        ml_model_artifact_count(
            workflow_id=
                workflow_id
        )
        ==
        1
    )


    print(
        (
            "[PASS] repeated complete Summary is "
            "exactly deterministic"
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
    summary_result,
) -> None:

    payload = (
        summary_result
        .model_dump(
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
        forbidden
        .isdisjoint(
            _all_keys(
                payload
            )
        )
    )


    print(
        (
            "[PASS] complete real Summary remains "
            "privacy-minimal"
        )
    )


# ============================================================
# CONTRACT / RULE POLICIES
# ============================================================


def verify_summary_policies(
    *,
    summary_result,
) -> None:

    contract = (
        summary_result
        .summary_contract
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


    assert (
        summary_result.evaluation_status
        ==
        "complete"
    )


    print(
        (
            "[PASS] Summary preserves all fixed "
            "v0.1 evaluation policies"
        )
    )


# ============================================================
# RULE VERSIONS
# ============================================================


def verify_rule_versions(
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


    assert (
        ML_MODEL_EVALUATION_SUMMARY_EXECUTOR_RULE_VERSION
        ==
        "ml_model_evaluation_summary_executor_v0.1"
    )


    assert (
        ML_MODEL_EVALUATION_SUMMARY_GOLDEN_PATH_RULE_VERSION
        ==
        "ml_model_evaluation_summary_golden_path_v0.1"
    )


    print(
        "[PASS] Model Evaluation Summary rule versions"
    )


# ============================================================
# GOLDEN PATH
# ============================================================


def test_ml_model_evaluation_summary_golden_path_v0_1(
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
        # REAL TRAINING
        # ----------------------------------------------------

        training_contract = (
            build_real_classification_contract(
                workflow_id=
                    workflow_id
            )
        )


        training_result = (
            train_real_classifier(
                workflow_id=
                    workflow_id,

                training_contract=
                    training_contract,
            )
        )


        # ----------------------------------------------------
        # TRUSTED RELOAD
        # ----------------------------------------------------

        verify_real_classifier_reload(
            workflow_id=
                workflow_id,

            execution_result=
                training_result,
        )


        # ----------------------------------------------------
        # COMPLETE REAL MODEL EVALUATION SUMMARY
        # ----------------------------------------------------

        summary_result = (
            run_real_summary(
                workflow_id=
                    workflow_id,

                training_result=
                    training_result,
            )
        )


        verify_summary_authority(
            workflow_id=
                workflow_id,

            training_contract=
                training_contract,

            training_result=
                training_result,

            summary_result=
                summary_result,
        )


        verify_baseline(
            training_result=
                training_result,

            summary_result=
                summary_result,
        )


        verify_standalone_selection(
            summary_result=
                summary_result
        )


        verify_classification_diagnostics(
            training_result=
                training_result,

            summary_result=
                summary_result,
        )


        verify_decision_threshold(
            summary_result=
                summary_result
        )


        verify_explainability(
            training_contract=
                training_contract,

            summary_result=
                summary_result,
        )


        verify_limitations(
            summary_result=
                summary_result
        )


        verify_summary_policies(
            summary_result=
                summary_result
        )


        verify_privacy_minimal(
            summary_result=
                summary_result
        )


        verify_deterministic_repeat(
            workflow_id=
                workflow_id,

            training_result=
                training_result,

            first_result=
                summary_result,
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
        (
            "DATALENS ML MODEL EVALUATION SUMMARY "
            "GOLDEN PATH E2E v0.1"
        )
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
        "Estimator   : real LogisticRegression"
    )


    print(
        "Artifact    : one persisted trusted Model Artifact"
    )


    print(
        "Holdout     : deterministic stratified 24/6"
    )


    print(
        "Baseline    : majority class learned from y_train only"
    )


    print(
        "Diagnostics : real native Classification Diagnostics"
    )


    print(
        "Threshold   : explicit predict_proba threshold 0.70"
    )


    print(
        "Explain     : real holdout permutation importance"
    )


    print(
        "Selection   : standalone / no provenance inferred"
    )


    print(
        "Persistence : zero additional Model Artifacts"
    )


    print()


    test_ml_model_evaluation_summary_golden_path_v0_1()


    print()


    print(
        "="
        *
        78
    )


    print(
        (
            "PASS - Preparation -> Classification Training -> "
            "Trusted Model Artifact -> Baseline Reconstruction -> "
            "Classification Diagnostics -> Explicit Threshold 0.70 -> "
            "Permutation Explainability -> Deterministic "
            "Evaluation Summary -> Privacy-Minimal -> "
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
