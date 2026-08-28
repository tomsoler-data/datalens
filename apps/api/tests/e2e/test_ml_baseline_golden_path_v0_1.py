from __future__ import annotations


import math


from fastapi.testclient import (
    TestClient,
)


# ============================================================
# REUSE REAL MODEL COMPARISON GOLDEN PATH
#
# Importing this module first also establishes the isolated
# product environment before application execution.
# ============================================================


from tests.e2e.test_ml_model_comparison_golden_path_v0_1 import (
    WORKFLOW_ROOT_DATASET_ID,
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


from app.ml.model_artifact_store import (
    list_ml_model_artifacts,
)


# ============================================================
# VERSION
# ============================================================


ML_BASELINE_GOLDEN_PATH_RULE_VERSION = (
    "ml_baseline_golden_path_v0.1"
)


# ============================================================
# SHARED BASELINE
# ============================================================


def verify_real_shared_baseline(
    *,
    comparison_result,
) -> None:

    baseline = (
        comparison_result
        .baseline
    )


    assert (
        baseline.problem_type
        ==
        "regression"
    )


    assert (
        baseline.strategy
        ==
        "mean_train_target"
    )


    assert (
        baseline.primary_metric
        ==
        "rmse"
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
        set(
            baseline.metrics
        )
        ==
        {
            "mae",
            "rmse",
            "r2",
        }
    )


    for metric_value in (
        baseline.metrics.values()
    ):

        assert math.isfinite(
            float(
                metric_value
            )
        )


    baseline_rmse = float(
        baseline.metrics[
            "rmse"
        ]
    )


    assert (
        baseline_rmse
        >
        0.0
    )


    print(
        (
            "[PASS] real baseline was learned only "
            "from the common training holdout"
        )
    )


# ============================================================
# CANDIDATE VS SHARED BASELINE
# ============================================================


def verify_candidate_baseline_comparisons(
    *,
    comparison_result,
) -> None:

    baseline_rmse = float(
        comparison_result
        .baseline
        .metrics[
            "rmse"
        ]
    )


    for candidate in (
        comparison_result.candidates
    ):

        comparison = (
            candidate
            .baseline_comparison
        )


        assert (
            comparison.problem_type
            ==
            "regression"
        )


        assert (
            comparison.primary_metric
            ==
            "rmse"
        )


        assert math.isclose(
            (
                comparison
                .baseline_primary_metric_value
            ),
            baseline_rmse,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )


        candidate_rmse = float(
            candidate.metrics[
                "rmse"
            ]
        )


        assert math.isclose(
            (
                comparison
                .model_primary_metric_value
            ),
            candidate_rmse,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )


        expected_improvement = (
            baseline_rmse
            -
            candidate_rmse
        )


        assert math.isclose(
            (
                comparison
                .absolute_improvement
            ),
            expected_improvement,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )


        assert (
            comparison.beats_baseline
            ==
            (
                candidate_rmse
                <
                baseline_rmse
            )
        )


    print(
        (
            "[PASS] every ranked model was evaluated "
            "against exactly one shared baseline"
        )
    )


# ============================================================
# SELECTED MODEL VS BASELINE
# ============================================================


def verify_selected_model_beats_baseline(
    *,
    comparison_result,
) -> None:

    winner = (
        comparison_result
        .candidates[
            0
        ]
    )


    assert (
        winner.estimator_key
        ==
        "linear_regression"
    )


    assert (
        winner
        .baseline_comparison
        .beats_baseline
        is True
    )


    assert (
        winner.metrics[
            "rmse"
        ]
        <
        comparison_result
        .baseline
        .metrics[
            "rmse"
        ]
    )


    assert (
        winner
        .baseline_comparison
        .absolute_improvement
        >
        0.0
    )


    assert (
        winner
        .baseline_comparison
        .relative_improvement_pct
        is not None
    )


    assert (
        winner
        .baseline_comparison
        .relative_improvement_pct
        >
        0.0
    )


    print(
        (
            "[PASS] selected LinearRegression proves "
            "measurable improvement over baseline"
        )
    )


# ============================================================
# MODEL ARTIFACT BOUNDARY
# ============================================================


def verify_baseline_created_no_model_artifact(
    *,
    workflow_id: str,
    comparison_result,
) -> None:

    artifacts = (
        list_ml_model_artifacts(
            workflow_id=
                workflow_id
        )
    )


    candidate_model_ids = {
        candidate
        .model_artifact
        .model_id

        for candidate
        in comparison_result.candidates
    }


    persisted_model_ids = {
        artifact.model_id

        for artifact
        in artifacts
    }


    assert (
        len(
            artifacts
        )
        ==
        3
    )


    assert (
        persisted_model_ids
        ==
        candidate_model_ids
    )


    assert not hasattr(
        comparison_result.baseline,
        "model_id",
    )


    print(
        (
            "[PASS] baseline created no fourth "
            "Model Artifact"
        )
    )


# ============================================================
# RULE VERSION
# ============================================================


def verify_rule_version(
) -> None:

    assert (
        ML_BASELINE_GOLDEN_PATH_RULE_VERSION
        ==
        "ml_baseline_golden_path_v0.1"
    )


    print(
        "[PASS] ML Baseline Golden Path rule version"
    )


# ============================================================
# GOLDEN PATH
# ============================================================


def test_ml_baseline_golden_path_v0_1(
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
        # REAL MODEL COMPARISON + BASELINE
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


        verify_real_shared_baseline(
            comparison_result=
                comparison_result
        )


        verify_candidate_baseline_comparisons(
            comparison_result=
                comparison_result
        )


        verify_selected_model_beats_baseline(
            comparison_result=
                comparison_result
        )


        # ----------------------------------------------------
        # REAL MODEL ARTIFACTS
        # ----------------------------------------------------

        verify_candidate_artifacts(
            workflow_id=
                workflow_id,

            contract=
                comparison_contract,

            result=
                comparison_result,
        )


        verify_baseline_created_no_model_artifact(
            workflow_id=
                workflow_id,

            comparison_result=
                comparison_result,
        )


        # ----------------------------------------------------
        # TRUSTED RELOAD OF REAL MODELS
        # ----------------------------------------------------

        selected_loaded = (
            verify_all_candidates_reload(
                workflow_id=
                    workflow_id,

                result=
                    comparison_result,
            )
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


        assert (
            comparison_result.dataset_id
            ==
            WORKFLOW_ROOT_DATASET_ID
        )


        verify_rule_version()


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
        "DATALENS ML BASELINE GOLDEN PATH E2E v0.1"
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
        "Baseline    : mean(y_train) on common holdout"
    )

    print(
        "Candidates  : Linear + Ridge + Random Forest"
    )

    print(
        "Comparison  : every model vs shared baseline"
    )

    print(
        "Artifacts   : real models only, no baseline artifact"
    )

    print(
        "Reload      : trusted SHA-verified joblib boundary"
    )

    print(
        "Prediction  : known + unseen category"
    )

    print()


    test_ml_baseline_golden_path_v0_1()


    print()

    print(
        "="
        *
        78
    )


    print(
        (
            "PASS - Mixed CSV -> Preparation -> VALIDATE -> "
            "Shared Baseline -> 3 Models -> Model vs Baseline -> "
            "Deterministic Ranking -> Real Model Artifacts -> "
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
