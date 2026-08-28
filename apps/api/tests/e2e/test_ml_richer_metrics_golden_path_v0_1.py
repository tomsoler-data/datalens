from __future__ import annotations


import math


from fastapi.testclient import (
    TestClient,
)


from app.ml.classical_executor import (
    ML_RICHER_METRICS_RULE_VERSION,
)


from app.ml.model_artifact_store import (
    get_ml_model_artifact,
)


from tests.e2e.test_ml_experiment_provenance_golden_path_v0_1 import (
    verify_experiment_provenance_survives_real_restore,
    verify_real_candidate_experiment_provenance,
    verify_selected_experiment_identity,
    verify_selected_reload_preserves_provenance,
)


from tests.e2e.test_ml_model_comparison_golden_path_v0_1 import (
    app,
    create_preparation_session,
    reset_product_state,
    run_real_cleaning_plan,
    run_real_model_comparison,
    run_real_quality,
    select_analysis_output,
    validate_preparation,
    verify_all_candidates_reload,
    verify_candidate_artifacts,
    verify_preparation_persistence,
    verify_real_handoff,
    verify_selected_known_predictions,
    verify_selected_pipeline,
    verify_selected_unseen_category,
)


# ============================================================
# VERSION
# ============================================================


ML_RICHER_METRICS_GOLDEN_PATH_RULE_VERSION = (
    "ml_richer_metrics_golden_path_v0.1"
)


# ============================================================
# EXPECTED METRIC SURFACES
# ============================================================


REGRESSION_METRIC_NAMES = {
    "mae",
    "rmse",
    "r2",
    "median_absolute_error",
    "explained_variance",
}


REGRESSION_BASELINE_METRIC_NAMES = {
    "mae",
    "rmse",
    "r2",
}


# ============================================================
# REAL COMPARISON METRICS
# ============================================================


def verify_real_richer_metric_surface(
    *,
    comparison_result,
) -> None:

    assert (
        comparison_result.problem_type
        ==
        "regression"
    )


    assert (
        comparison_result.primary_metric
        ==
        "rmse"
    )


    assert (
        comparison_result.ranking_policy
        ==
        "regression_rmse_v0.1"
    )


    assert (
        set(
            comparison_result
            .baseline
            .metrics
        )
        ==
        REGRESSION_BASELINE_METRIC_NAMES
    )


    assert (
        len(
            comparison_result.candidates
        )
        ==
        3
    )


    for candidate in (
        comparison_result.candidates
    ):

        assert (
            set(
                candidate.metrics
            )
            ==
            REGRESSION_METRIC_NAMES
        )


        assert (
            candidate.primary_metric
            ==
            "rmse"
        )


        assert math.isclose(
            candidate.primary_metric_value,
            candidate.metrics[
                "rmse"
            ],
            rel_tol=1e-12,
            abs_tol=1e-12,
        )


        assert (
            candidate.model_artifact.metrics
            ==
            candidate.metrics
        )


        assert (
            candidate
            .experiment_provenance
            .metrics
            ==
            candidate.metrics
        )


        assert (
            candidate
            .model_artifact
            .experiment_provenance
            ==
            candidate
            .experiment_provenance
        )


        for metric_value in (
            candidate.metrics.values()
        ):

            assert math.isfinite(
                float(
                    metric_value
                )
            )


    winner = (
        comparison_result
        .candidates[
            0
        ]
    )


    assert (
        comparison_result
        .selected_estimator_key
        ==
        winner.estimator_key
    )


    assert (
        comparison_result
        .selected_experiment_id
        ==
        winner
        .experiment_provenance
        .experiment_id
    )


    assert (
        comparison_result
        .selected_model_id
        ==
        winner
        .model_artifact
        .model_id
    )


    print(
        (
            "[PASS] three real candidates expose five "
            "finite richer regression metrics"
        )
    )


    print(
        (
            "[PASS] baseline remains on exact v0.1 "
            "three-metric surface"
        )
    )


    print(
        (
            "[PASS] Model Comparison ranking remains "
            "regression_rmse_v0.1"
        )
    )


# ============================================================
# SQLITE RESTORE
# ============================================================


def verify_richer_metrics_survive_real_restore(
    *,
    workflow_id: str,
    comparison_result,
) -> None:

    for candidate in (
        comparison_result.candidates
    ):

        restored = (
            get_ml_model_artifact(
                workflow_id=
                    workflow_id,

                model_id=
                    candidate
                    .model_artifact
                    .model_id,
            )
        )


        assert (
            set(
                restored.metrics
            )
            ==
            REGRESSION_METRIC_NAMES
        )


        assert (
            restored.metrics
            ==
            candidate.metrics
        )


        assert (
            restored
            .experiment_provenance
            is not None
        )


        assert (
            restored
            .experiment_provenance
            .metrics
            ==
            candidate.metrics
        )


    print(
        (
            "[PASS] five richer metrics survived "
            "real SQLite Model Artifact restore"
        )
    )


# ============================================================
# TRUSTED RELOAD
# ============================================================


def verify_selected_reload_preserves_richer_metrics(
    *,
    selected_loaded,
    comparison_result,
) -> None:

    winner = (
        comparison_result
        .candidates[
            0
        ]
    )


    assert (
        selected_loaded
        .artifact
        .model_id
        ==
        comparison_result
        .selected_model_id
    )


    assert (
        selected_loaded
        .artifact
        .metrics
        ==
        winner.metrics
    )


    assert (
        set(
            selected_loaded
            .artifact
            .metrics
        )
        ==
        REGRESSION_METRIC_NAMES
    )


    assert (
        selected_loaded
        .artifact
        .experiment_provenance
        is not None
    )


    assert (
        selected_loaded
        .artifact
        .experiment_provenance
        .metrics
        ==
        winner.metrics
    )


    print(
        (
            "[PASS] trusted reload preserved selected "
            "richer metrics + provenance"
        )
    )


# ============================================================
# VERSION
# ============================================================


def verify_rule_versions(
) -> None:

    assert (
        ML_RICHER_METRICS_RULE_VERSION
        ==
        "ml_richer_metrics_v0.1"
    )


    assert (
        ML_RICHER_METRICS_GOLDEN_PATH_RULE_VERSION
        ==
        "ml_richer_metrics_golden_path_v0.1"
    )


    print(
        "[PASS] Richer ML Metrics rule versions"
    )


# ============================================================
# GOLDEN PATH
# ============================================================


def test_ml_richer_metrics_golden_path_v0_1(
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
        # REAL 3-MODEL COMPARISON
        # ----------------------------------------------------

        (
            comparison_contract,
            comparison_result,
        ) = (
            run_real_model_comparison(
                workflow_id=
                    workflow_id
            )
        )


        verify_real_richer_metric_surface(
            comparison_result=
                comparison_result
        )


        # ----------------------------------------------------
        # MODEL ARTIFACT + EXPERIMENT PROVENANCE
        # ----------------------------------------------------

        verify_candidate_artifacts(
            workflow_id=
                workflow_id,

            contract=
                comparison_contract,

            result=
                comparison_result,
        )


        verify_real_candidate_experiment_provenance(
            workflow_id=
                workflow_id,

            comparison_result=
                comparison_result,
        )


        verify_selected_experiment_identity(
            comparison_result=
                comparison_result
        )


        # ----------------------------------------------------
        # REAL SQLITE RESTORE
        # ----------------------------------------------------

        verify_experiment_provenance_survives_real_restore(
            workflow_id=
                workflow_id,

            comparison_result=
                comparison_result,
        )


        verify_richer_metrics_survive_real_restore(
            workflow_id=
                workflow_id,

            comparison_result=
                comparison_result,
        )


        # ----------------------------------------------------
        # TRUSTED RELOAD
        # ----------------------------------------------------

        selected_loaded = (
            verify_all_candidates_reload(
                workflow_id=
                    workflow_id,

                result=
                    comparison_result,
            )
        )


        verify_selected_reload_preserves_provenance(
            selected_loaded=
                selected_loaded,

            comparison_result=
                comparison_result,
        )


        verify_selected_reload_preserves_richer_metrics(
            selected_loaded=
                selected_loaded,

            comparison_result=
                comparison_result,
        )


        verify_selected_pipeline(
            selected_loaded=
                selected_loaded
        )


        # ----------------------------------------------------
        # REAL PREDICTIONS
        # ----------------------------------------------------

        verify_selected_known_predictions(
            selected_loaded=
                selected_loaded
        )


        verify_selected_unseen_category(
            selected_loaded=
                selected_loaded
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
        "DATALENS RICHER ML METRICS GOLDEN PATH E2E v0.1"
    )


    print(
        "="
        *
        78
    )


    print(
        "Preparation : real validated mixed-type CSV"
    )


    print(
        "Candidates  : Linear + Ridge + Random Forest"
    )


    print(
        "Metrics     : five richer regression metrics"
    )


    print(
        "Baseline    : compatible three-metric v0.1 surface"
    )


    print(
        "Ranking     : unchanged regression_rmse_v0.1"
    )


    print(
        "Artifacts   : real persisted Model Artifacts"
    )


    print(
        "Provenance  : richer metrics persisted"
    )


    print(
        "Reload      : trusted SHA-verified joblib boundary"
    )


    print(
        "Prediction  : known + unseen category"
    )


    print()


    test_ml_richer_metrics_golden_path_v0_1()


    print()

    print(
        "="
        *
        78
    )


    print(
        (
            "PASS - Preparation -> 3 Models -> "
            "5 Richer Metrics -> Model Artifacts -> "
            "Experiment Provenance -> SQLite Restore -> "
            "Trusted Reload -> Prediction"
        )
    )


    print(
        "="
        *
        78
    )


if __name__ == "__main__":
    main()
