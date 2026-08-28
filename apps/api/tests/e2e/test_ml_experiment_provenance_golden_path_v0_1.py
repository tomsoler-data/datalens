from __future__ import annotations


from fastapi.testclient import (
    TestClient,
)


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


from app.ml.experiment_provenance import (
    ML_EXPERIMENT_PROVENANCE_RULE_VERSION,
    ml_training_contract_sha256,
)


from app.ml.model_artifact_store import (
    get_ml_model_artifact,
    list_ml_model_artifacts,
)


# ============================================================
# VERSION
# ============================================================


ML_EXPERIMENT_PROVENANCE_GOLDEN_PATH_RULE_VERSION = (
    "ml_experiment_provenance_golden_path_v0.1"
)


# ============================================================
# REAL CANDIDATE PROVENANCE
# ============================================================


def verify_real_candidate_experiment_provenance(
    *,
    workflow_id: str,
    comparison_result,
) -> None:

    candidates = (
        comparison_result.candidates
    )


    assert (
        len(
            candidates
        )
        ==
        3
    )


    experiment_ids = {
        candidate
        .experiment_provenance
        .experiment_id

        for candidate
        in candidates
    }


    model_ids = {
        candidate
        .model_artifact
        .model_id

        for candidate
        in candidates
    }


    assert (
        len(
            experiment_ids
        )
        ==
        3
    )


    assert (
        len(
            model_ids
        )
        ==
        3
    )


    for candidate in (
        candidates
    ):

        provenance = (
            candidate
            .experiment_provenance
        )


        artifact = (
            candidate
            .model_artifact
        )


        assert (
            provenance
            ==
            artifact.experiment_provenance
        )


        assert (
            provenance.experiment_id
            .startswith(
                "experiment:"
            )
        )


        assert (
            provenance.workflow_id
            ==
            workflow_id
        )


        assert (
            provenance.dataset_id
            ==
            WORKFLOW_ROOT_DATASET_ID
        )


        assert (
            provenance
            .preparation_session_revision
            ==
            comparison_result
            .preparation_session_revision
        )


        assert (
            provenance.model_id
            ==
            artifact.model_id
        )


        assert (
            provenance.train_rows
            ==
            candidate.train_rows
        )


        assert (
            provenance.test_rows
            ==
            candidate.test_rows
        )


        assert (
            provenance.metrics
            ==
            candidate.metrics
        )


        assert (
            provenance.training_contract_sha256
            ==
            ml_training_contract_sha256(
                artifact.training_contract
            )
        )


        assert (
            provenance.rule_version
            ==
            ML_EXPERIMENT_PROVENANCE_RULE_VERSION
        )


    print(
        (
            "[PASS] three real candidates produced "
            "three distinct Experiment Provenance records"
        )
    )


# ============================================================
# WINNER IDENTITY
# ============================================================


def verify_selected_experiment_identity(
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


    assert (
        winner
        .experiment_provenance
        .model_id
        ==
        comparison_result
        .selected_model_id
    )


    print(
        (
            "[PASS] selected experiment and selected "
            "Model Artifact identify the same winner"
        )
    )


# ============================================================
# SQLITE RESTORE
# ============================================================


def verify_experiment_provenance_survives_real_restore(
    *,
    workflow_id: str,
    comparison_result,
) -> None:

    restored_artifacts = (
        list_ml_model_artifacts(
            workflow_id=
                workflow_id
        )
    )


    assert (
        len(
            restored_artifacts
        )
        ==
        3
    )


    restored_by_model_id = {
        artifact.model_id:
            artifact

        for artifact
        in restored_artifacts
    }


    restored_experiment_ids = set()


    for candidate in (
        comparison_result.candidates
    ):

        model_id = (
            candidate
            .model_artifact
            .model_id
        )


        assert (
            model_id
            in
            restored_by_model_id
        )


        restored = (
            get_ml_model_artifact(
                workflow_id=
                    workflow_id,

                model_id=
                    model_id,
            )
        )


        provenance = (
            restored
            .experiment_provenance
        )


        assert (
            provenance
            is not None
        )


        assert (
            provenance
            ==
            candidate
            .experiment_provenance
        )


        assert (
            provenance.model_id
            ==
            restored.model_id
        )


        assert (
            provenance.training_contract_sha256
            ==
            ml_training_contract_sha256(
                restored.training_contract
            )
        )


        restored_experiment_ids.add(
            provenance.experiment_id
        )


    assert (
        len(
            restored_experiment_ids
        )
        ==
        3
    )


    print(
        (
            "[PASS] Experiment Provenance survived "
            "real SQLite Model Artifact restore"
        )
    )


# ============================================================
# PRIVACY SURFACE
# ============================================================


def verify_real_provenance_is_privacy_minimal(
    *,
    comparison_result,
) -> None:

    forbidden_fields = {
        "dataframe",
        "predictions",
        "x_train",
        "x_test",
        "y_train",
        "y_test",
        "model_bytes",
        "model_path",
    }


    for candidate in (
        comparison_result.candidates
    ):

        payload = (
            candidate
            .experiment_provenance
            .model_dump(
                mode="json"
            )
        )


        assert (
            forbidden_fields
            .isdisjoint(
                payload
            )
        )


    print(
        (
            "[PASS] real Experiment Provenance remains "
            "privacy-minimal"
        )
    )


# ============================================================
# TRUSTED RELOAD PROVENANCE
# ============================================================


def verify_selected_reload_preserves_provenance(
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
        .experiment_provenance
        ==
        winner
        .experiment_provenance
    )


    assert (
        selected_loaded
        .artifact
        .experiment_provenance
        .experiment_id
        ==
        comparison_result
        .selected_experiment_id
    )


    print(
        (
            "[PASS] trusted reload preserved selected "
            "Experiment Provenance"
        )
    )


# ============================================================
# RULE VERSION
# ============================================================


def verify_rule_version(
) -> None:

    assert (
        ML_EXPERIMENT_PROVENANCE_GOLDEN_PATH_RULE_VERSION
        ==
        "ml_experiment_provenance_golden_path_v0.1"
    )


    print(
        "[PASS] Experiment Provenance Golden Path rule version"
    )


# ============================================================
# GOLDEN PATH
# ============================================================


def test_ml_experiment_provenance_golden_path_v0_1(
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
        # REAL MODEL COMPARISON
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


        # ----------------------------------------------------
        # REAL EXPERIMENT PROVENANCE
        # ----------------------------------------------------

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


        verify_real_provenance_is_privacy_minimal(
            comparison_result=
                comparison_result
        )


        # ----------------------------------------------------
        # REAL MODEL ARTIFACT RESTORE
        # ----------------------------------------------------

        verify_candidate_artifacts(
            workflow_id=
                workflow_id,

            contract=
                comparison_contract,

            result=
                comparison_result,
        )


        verify_experiment_provenance_survives_real_restore(
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
        "DATALENS ML EXPERIMENT PROVENANCE GOLDEN PATH E2E v0.1"
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
        "Snapshot    : real server-owned Preparation revision"
    )


    print(
        "Experiments : three server-generated experiment IDs"
    )


    print(
        "Contracts   : deterministic SHA-256 fingerprints"
    )


    print(
        "Artifacts   : three real persisted Model Artifacts"
    )


    print(
        "Reload      : trusted SHA-verified joblib boundary"
    )


    print(
        "Prediction  : known + unseen category"
    )


    print()


    test_ml_experiment_provenance_golden_path_v0_1()


    print()

    print(
        "="
        *
        78
    )


    print(
        (
            "PASS - Preparation -> Pinned Revision -> "
            "3 Experiments -> Contract SHA-256 -> "
            "3 Model Artifacts -> SQLite Restore -> "
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
