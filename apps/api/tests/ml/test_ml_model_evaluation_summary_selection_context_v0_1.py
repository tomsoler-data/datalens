from __future__ import annotations


import math


from app.ml.baseline import (
    compare_model_to_baseline,
)


from app.ml.contracts import (
    MLTrainingContract,
)


from app.ml.estimator_contracts import (
    MLLogisticRegressionHyperparameters,
)


from app.ml.experiment_provenance import (
    MLExperimentProvenanceRecord,
    ml_training_contract_sha256,
)


from app.ml.model_artifacts import (
    MLModelArtifactRecord,
)


from app.ml.model_comparison_contracts import (
    MLModelComparisonContract,
)


from app.ml.model_comparison_executor import (
    MLModelComparisonCandidateResult,
    MLModelComparisonExecutionResult,
)


from app.ml.model_evaluation_summary_executor import (
    MLModelEvaluationSummarySelectionError,
)


from app.ml.tuned_model_promotion_executor import (
    MLTunedModelPromotionExecutionResult,
)


from tests.ml.test_ml_model_evaluation_summary_executor_v0_1 import (
    DATASET_ID,
    EXPERIMENT_ID,
    MODEL_ID,
    PREPARATION_REVISION,
    WORKFLOW_ID,
    build_artifact,
    classification_model_metrics,
    expected_baseline,
    expect_exception,
    run_fake_runtime,
    training_contract,
)


# ============================================================
# VERSION
# ============================================================


ML_MODEL_EVALUATION_SUMMARY_SELECTION_CONTEXT_TEST_VERSION = (
    "ml_model_evaluation_summary_selection_context_v0.1"
)


# ============================================================
# SECOND COMPARISON CANDIDATE
# ============================================================


SECOND_MODEL_ID = (
    "model:evaluation-summary-random-forest"
)


SECOND_EXPERIMENT_ID = (
    "experiment:"
    +
    (
        "d"
        *
        32
    )
)


def random_forest_training_contract(
) -> MLTrainingContract:

    base = (
        training_contract(
            "classification"
        )
    )


    payload = (
        base.model_dump(
            mode="python"
        )
    )


    payload[
        "estimator_key"
    ] = (
        "random_forest_classifier"
    )


    payload[
        "estimator_hyperparameters"
    ] = None


    return (
        MLTrainingContract
        .model_validate(
            payload
        )
    )


def loser_metrics(
) -> dict[
    str,
    float,
]:

    return {
        "accuracy":
            0.50,

        "f1_macro":
            0.40,

        "precision_macro":
            0.42,

        "recall_macro":
            0.40,

        "balanced_accuracy":
            0.40,
    }


def build_loser_artifact(
) -> MLModelArtifactRecord:

    contract = (
        random_forest_training_contract()
    )


    metrics = (
        loser_metrics()
    )


    provenance = (
        MLExperimentProvenanceRecord(
            experiment_id=
                SECOND_EXPERIMENT_ID,

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
                SECOND_MODEL_ID,

            train_rows=
                4,

            test_rows=
                4,

            metrics=
                metrics,
        )
    )


    return (
        MLModelArtifactRecord(
            model_id=
                SECOND_MODEL_ID,

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
                4,

            test_rows=
                4,

            created_at_utc=(
                "2026-08-29T00:00:00+00:00"
            ),

            serialization_format=
                "joblib",

            model_path=(
                "models/"
                "evaluation-summary-random-forest.joblib"
            ),

            model_file_bytes=
                456,

            model_sha256=(
                "e"
                *
                64
            ),
        )
    )


# ============================================================
# REAL PYDANTIC MODEL COMPARISON CONTEXT
# ============================================================


def model_comparison_context(
) -> MLModelComparisonExecutionResult:

    winner_artifact = (
        build_artifact(
            "classification"
        )
    )


    loser_artifact = (
        build_loser_artifact()
    )


    winner_provenance = (
        winner_artifact
        .experiment_provenance
    )


    loser_provenance = (
        loser_artifact
        .experiment_provenance
    )


    assert (
        winner_provenance
        is not None
    )


    assert (
        loser_provenance
        is not None
    )


    (
        baseline,
        winner_baseline_comparison,
    ) = (
        expected_baseline(
            "classification"
        )
    )


    loser_baseline_comparison = (
        compare_model_to_baseline(
            problem_type=
                "classification",

            model_metrics=(
                loser_artifact
                .metrics
            ),

            baseline_metrics=(
                baseline.metrics
            ),
        )
    )


    winner_contract = (
        winner_artifact
        .training_contract
    )


    loser_contract = (
        loser_artifact
        .training_contract
    )


    comparison_contract = (
        MLModelComparisonContract(
            candidates=[
                winner_contract,
                loser_contract,
            ]
        )
    )


    winner = (
        MLModelComparisonCandidateResult(
            rank=
                1,

            estimator_key=(
                winner_contract
                .estimator_key
            ),

            primary_metric=
                "f1_macro",

            primary_metric_value=(
                winner_artifact
                .metrics[
                    "f1_macro"
                ]
            ),

            metrics=
                dict(
                    winner_artifact
                    .metrics
                ),

            train_rows=
                4,

            test_rows=
                4,

            baseline_comparison=(
                winner_baseline_comparison
            ),

            experiment_provenance=(
                winner_provenance
            ),

            model_artifact=(
                winner_artifact
            ),
        )
    )


    loser = (
        MLModelComparisonCandidateResult(
            rank=
                2,

            estimator_key=(
                loser_contract
                .estimator_key
            ),

            primary_metric=
                "f1_macro",

            primary_metric_value=(
                loser_artifact
                .metrics[
                    "f1_macro"
                ]
            ),

            metrics=
                dict(
                    loser_artifact
                    .metrics
                ),

            train_rows=
                4,

            test_rows=
                4,

            baseline_comparison=(
                loser_baseline_comparison
            ),

            experiment_provenance=(
                loser_provenance
            ),

            model_artifact=(
                loser_artifact
            ),
        )
    )


    return (
        MLModelComparisonExecutionResult(
            workflow_id=
                WORKFLOW_ID,

            dataset_id=
                DATASET_ID,

            preparation_session_revision=(
                PREPARATION_REVISION
            ),

            problem_type=
                "classification",

            comparison_contract=(
                comparison_contract
            ),

            primary_metric=
                "f1_macro",

            ranking_policy=(
                "classification_f1_macro_v0.1"
            ),

            baseline=
                baseline,

            candidates=[
                winner,
                loser,
            ],

            selected_estimator_key=(
                winner
                .estimator_key
            ),

            selected_experiment_id=(
                winner_provenance
                .experiment_id
            ),

            selected_model_id=(
                winner_artifact
                .model_id
            ),
        )
    )


# ============================================================
# REAL PYDANTIC TUNED PROMOTION CONTEXT
# ============================================================


def tuned_promotion_context(
) -> MLTunedModelPromotionExecutionResult:

    artifact = (
        build_artifact(
            "classification"
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


    promoted_sha256 = (
        ml_training_contract_sha256(
            artifact.training_contract
        )
    )


    return (
        MLTunedModelPromotionExecutionResult(
            workflow_id=
                WORKFLOW_ID,

            dataset_id=
                DATASET_ID,

            problem_type=
                "classification",

            estimator_key=
                "logistic_regression",

            preparation_session_revision=(
                PREPARATION_REVISION
            ),

            base_training_contract_sha256=(
                "f"
                *
                64
            ),

            promoted_training_contract_sha256=(
                promoted_sha256
            ),

            selected_candidate_index=
                2,

            selected_hyperparameters=(
                MLLogisticRegressionHyperparameters(
                    inverse_regularization_strength=
                        1.0,

                    fit_intercept=
                        True,

                    max_iter=
                        1000,

                    class_weight=
                        None,
                )
            ),

            tuning_primary_metric=
                "f1_macro",

            tuning_primary_metric_mean=
                0.71,

            tuning_primary_metric_std=
                0.03,

            train_rows=
                artifact.train_rows,

            test_rows=
                artifact.test_rows,

            final_metrics=
                dict(
                    artifact.metrics
                ),

            model_id=
                artifact.model_id,

            experiment_id=(
                provenance
                .experiment_id
            ),
        )
    )


# ============================================================
# MODEL COMPARISON
# ============================================================


def test_model_comparison_context_is_preserved(
) -> None:

    comparison = (
        model_comparison_context()
    )


    (
        result,
        _,
        _,
    ) = (
        run_fake_runtime(
            "classification",

            selection_context=(
                comparison
            ),
        )
    )


    evidence = (
        result
        .selection_evidence
    )


    assert (
        evidence.source
        ==
        "model_comparison"
    )


    assert (
        evidence.status
        ==
        "verified_selected"
    )


    assert (
        evidence.rank
        ==
        1
    )


    assert (
        evidence.selection_policy
        ==
        "classification_f1_macro_v0.1"
    )


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


    assert math.isclose(
        float(
            evidence
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


    assert (
        "selection_evidence_not_available"
        not in
        result.limitations
    )


def test_model_comparison_nonwinner_fails_closed(
) -> None:

    comparison = (
        model_comparison_context()
    )


    tampered = (
        comparison.model_copy(
            update={
                "selected_model_id":
                    SECOND_MODEL_ID
            }
        )
    )


    expect_exception(
        MLModelEvaluationSummarySelectionError,

        lambda:
            run_fake_runtime(
                "classification",

                selection_context=(
                    tampered
                ),
            ),
    )


# ============================================================
# TUNED PROMOTION
# ============================================================


def test_tuned_promotion_context_is_preserved(
) -> None:

    promotion = (
        tuned_promotion_context()
    )


    (
        result,
        _,
        _,
    ) = (
        run_fake_runtime(
            "classification",

            selection_context=(
                promotion
            ),
        )
    )


    evidence = (
        result
        .selection_evidence
    )


    assert (
        evidence.source
        ==
        "tuned_model_promotion"
    )


    assert (
        evidence.status
        ==
        "verified_selected"
    )


    assert (
        evidence.rank
        ==
        1
    )


    assert (
        evidence.selection_policy
        ==
        "rank_1_only"
    )


    assert (
        evidence.primary_metric
        ==
        "f1_macro"
    )


    assert (
        evidence.metric_scope
        ==
        "inner_cross_validation"
    )


    assert math.isclose(
        float(
            evidence
            .primary_metric_value
        ),
        0.71,
        rel_tol=1e-12,
        abs_tol=1e-12,
    )


    # Critical methodological boundary:
    # tuning evidence is NOT represented as final holdout
    # performance.
    assert not math.isclose(
        float(
            evidence
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


    assert (
        "selection_evidence_not_available"
        not in
        result.limitations
    )


def test_tuned_promotion_identity_mismatch_fails_closed(
) -> None:

    promotion = (
        tuned_promotion_context()
    )


    tampered = (
        promotion.model_copy(
            update={
                "model_id":
                    "model:other"
            }
        )
    )


    expect_exception(
        MLModelEvaluationSummarySelectionError,

        lambda:
            run_fake_runtime(
                "classification",

                selection_context=(
                    tampered
                ),
            ),
    )


# ============================================================
# FINAL AUTHORITY
# ============================================================


def test_context_does_not_change_evaluated_model(
) -> None:

    comparison = (
        model_comparison_context()
    )


    (
        result,
        _,
        _,
    ) = (
        run_fake_runtime(
            "classification",

            selection_context=(
                comparison
            ),
        )
    )


    assert (
        result.model_id
        ==
        MODEL_ID
    )


    assert (
        result.experiment_id
        ==
        EXPERIMENT_ID
    )


    assert (
        result.metrics
        ==
        classification_model_metrics()
    )


# ============================================================
# RULE VERSION
# ============================================================


def test_rule_version(
) -> None:

    assert (
        ML_MODEL_EVALUATION_SUMMARY_SELECTION_CONTEXT_TEST_VERSION
        ==
        "ml_model_evaluation_summary_selection_context_v0.1"
    )


# ============================================================
# MAIN
# ============================================================


def main(
) -> None:

    print(
        "=== DATALENS ML MODEL EVALUATION SUMMARY "
        "SELECTION CONTEXT v0.1 ==="
    )


    tests = [
        (
            "Model Comparison rank-1 context preserved",
            test_model_comparison_context_is_preserved,
        ),
        (
            "Model Comparison non-winner fails closed",
            test_model_comparison_nonwinner_fails_closed,
        ),
        (
            "Tuned Promotion context preserved",
            test_tuned_promotion_context_is_preserved,
        ),
        (
            "Tuned Promotion identity mismatch fails closed",
            test_tuned_promotion_identity_mismatch_fails_closed,
        ),
        (
            "Selection context cannot change evaluated model",
            test_context_does_not_change_evaluated_model,
        ),
        (
            "Selection-context test rule version",
            test_rule_version,
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
        (
            "PASS - ML Model Evaluation Summary "
            "Selection Context v0.1"
        )
    )


if __name__ == "__main__":
    main()
