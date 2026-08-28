from __future__ import annotations


import math


from fastapi.testclient import (
    TestClient,
)


# ============================================================
# REUSE REAL ML PRODUCT ENVIRONMENT
#
# Importing this module first establishes the isolated E2E
# environment before the DataLens application/persistence
# modules below are imported.
# ============================================================


from tests.e2e.test_ml_golden_path_v0_1 import (
    WORKFLOW_ROOT_DATASET_ID,
    app,
    create_preparation_session,
    reset_ml_product_state,
    run_real_cleaning_plan,
    run_real_quality,
    select_analysis_output,
    validate_preparation,
    verify_preparation_persistence,
    verify_real_handoff,
)


from app.ml.classical_executor import (
    execute_classical_ml,
)


from app.ml.contracts import (
    MLTrainingContract,
)


from app.ml.experiment_provenance import (
    ml_training_contract_sha256,
)


from app.ml.model_explainability import (
    ML_MODEL_EXPLAINABILITY_RULE_VERSION,
    MLModelExplainabilityContract,
)


from app.ml.model_explainability_executor import (
    ML_MODEL_EXPLAINABILITY_EXECUTOR_RULE_VERSION,
    execute_ml_model_explainability,
)


from app.ml.model_loader import (
    load_trusted_ml_model,
)


from app.persistence.sqlite_database import (
    sqlite_connection,
)


from app.preparation.analysis_readiness_gate import (
    require_analysis_readiness,
)


# ============================================================
# VERSION
# ============================================================


ML_MODEL_EXPLAINABILITY_GOLDEN_PATH_RULE_VERSION = (
    "ml_model_explainability_golden_path_v0.1"
)


# ============================================================
# MODEL ARTIFACT COUNT
# ============================================================


def model_artifact_count(
    *,
    workflow_id: str,
) -> int:

    with sqlite_connection(
        write=False
    ) as connection:

        row = (
            connection.execute(
                """
                SELECT COUNT(*) AS count
                FROM ml_model_artifacts

                WHERE
                    workflow_id = ?
                """,
                (
                    workflow_id,
                ),
            )
            .fetchone()
        )


    assert (
        row
        is not None
    )


    return int(
        row[
            "count"
        ]
    )


# ============================================================
# TRAINING CONTRACT
# ============================================================


def build_training_contract(
    *,
    workflow_id: str,
) -> MLTrainingContract:

    return (
        MLTrainingContract(
            workflow_id=
                workflow_id,

            dataset_id=
                WORKFLOW_ROOT_DATASET_ID,

            problem_type=
                "regression",

            target_column=
                "revenue",

            feature_columns=[
                "customer_age",
                "tenure_months",
            ],

            estimator_key=
                "linear_regression",
        )
    )


# ============================================================
# REAL TRAINING
# ============================================================


def train_and_persist_real_model(
    *,
    workflow_id: str,
):

    contract = (
        build_training_contract(
            workflow_id=
                workflow_id
        )
    )


    readiness_before = (
        require_analysis_readiness(
            workflow_id=
                workflow_id
        )
    )


    artifacts_before = (
        model_artifact_count(
            workflow_id=
                workflow_id
        )
    )


    assert (
        artifacts_before
        ==
        0
    )


    result = (
        execute_classical_ml(
            training_contract=
                contract
        )
    )


    readiness_after = (
        require_analysis_readiness(
            workflow_id=
                workflow_id
        )
    )


    assert (
        readiness_before
        .session_revision
        ==
        readiness_after
        .session_revision
    )


    assert (
        result
        .experiment_provenance
        .preparation_session_revision
        ==
        readiness_before
        .session_revision
    )


    assert (
        model_artifact_count(
            workflow_id=
                workflow_id
        )
        ==
        1
    )


    print(
        (
            "[PASS] real Classical ML training produced "
            "one persisted Model Artifact"
        )
    )


    print(
        (
            "[PASS] trained model remained pinned "
            "to one Preparation revision"
        )
    )


    return (
        contract,
        result,
    )


# ============================================================
# TRUSTED RELOAD
# ============================================================


def verify_real_trusted_reload(
    *,
    workflow_id: str,
    training_result,
):

    loaded = (
        load_trusted_ml_model(
            workflow_id=
                workflow_id,

            model_id=
                training_result
                .model_artifact
                .model_id,
        )
    )


    assert (
        loaded
        .artifact
        .model_id
        ==
        training_result
        .model_artifact
        .model_id
    )


    assert (
        loaded
        .artifact
        .experiment_provenance
        ==
        training_result
        .experiment_provenance
    )


    print(
        (
            "[PASS] trained Model Artifact crossed "
            "trusted SHA-verified reload"
        )
    )


    return loaded


# ============================================================
# REAL EXPLAINABILITY
# ============================================================


def run_real_explainability(
    *,
    workflow_id: str,
    training_result,
):

    artifacts_before = (
        model_artifact_count(
            workflow_id=
                workflow_id
        )
    )


    assert (
        artifacts_before
        ==
        1
    )


    result = (
        execute_ml_model_explainability(
            workflow_id=
                workflow_id,

            model_id=
                training_result
                .model_artifact
                .model_id,

            explainability_contract=(
                MLModelExplainabilityContract(
                    n_repeats=
                        10,

                    random_seed=
                        73,
                )
            ),
        )
    )


    artifacts_after = (
        model_artifact_count(
            workflow_id=
                workflow_id
        )
    )


    assert (
        artifacts_after
        ==
        artifacts_before
    )


    print(
        (
            "[PASS] Explainability remained "
            "evaluation-only with no new Model Artifact"
        )
    )


    return result


# ============================================================
# RESULT AUTHORITY
# ============================================================


def verify_real_explainability_authority(
    *,
    training_contract,
    training_result,
    explainability_result,
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
        explainability_result.workflow_id
        ==
        artifact.workflow_id
    )


    assert (
        explainability_result.dataset_id
        ==
        artifact.dataset_id
    )


    assert (
        explainability_result.model_id
        ==
        artifact.model_id
    )


    assert (
        explainability_result.experiment_id
        ==
        provenance.experiment_id
    )


    assert (
        explainability_result
        .preparation_session_revision
        ==
        provenance
        .preparation_session_revision
    )


    assert (
        explainability_result
        .training_contract_sha256
        ==
        ml_training_contract_sha256(
            training_contract
        )
    )


    assert (
        explainability_result.problem_type
        ==
        "regression"
    )


    assert (
        explainability_result.estimator_key
        ==
        "linear_regression"
    )


    assert (
        explainability_result.method
        ==
        "permutation_importance"
    )


    assert (
        explainability_result.scoring
        ==
        "neg_root_mean_squared_error"
    )


    print(
        (
            "[PASS] explanation is bound to the "
            "real Model Artifact + Experiment Provenance"
        )
    )


# ============================================================
# HOLDOUT-ONLY FEATURE IMPORTANCE
# ============================================================


def verify_real_feature_importances(
    *,
    training_result,
    explainability_result,
) -> None:

    assert (
        explainability_result.evaluation_rows
        ==
        training_result.test_rows
    )


    assert (
        explainability_result.evaluation_rows
        ==
        6
    )


    importances = (
        explainability_result
        .feature_importances
    )


    assert (
        len(
            importances
        )
        ==
        2
    )


    assert (
        {
            item.feature_name

            for item
            in importances
        }
        ==
        {
            "customer_age",
            "tenure_months",
        }
    )


    assert (
        [
            item.rank

            for item
            in importances
        ]
        ==
        [
            1,
            2,
        ]
    )


    for item in importances:

        assert math.isfinite(
            float(
                item.importance_mean
            )
        )


        assert math.isfinite(
            float(
                item.importance_std
            )
        )


        assert (
            item.importance_std
            >=
            0.0
        )


    print(
        (
            "[PASS] permutation importance used the "
            "real six-row persisted holdout"
        )
    )


    print(
        (
            "[PASS] explanation preserves the two "
            "original Training Contract feature names"
        )
    )


# ============================================================
# DETERMINISM
# ============================================================


def verify_real_explainability_is_deterministic(
    *,
    workflow_id: str,
    training_result,
    first_result,
) -> None:

    second_result = (
        execute_ml_model_explainability(
            workflow_id=
                workflow_id,

            model_id=
                training_result
                .model_artifact
                .model_id,

            explainability_contract=(
                MLModelExplainabilityContract(
                    n_repeats=
                        10,

                    random_seed=
                        73,
                )
            ),
        )
    )


    assert (
        first_result.model_dump(
            mode="json"
        )
        ==
        second_result.model_dump(
            mode="json"
        )
    )


    assert (
        model_artifact_count(
            workflow_id=
                workflow_id
        )
        ==
        1
    )


    print(
        (
            "[PASS] repeated real explanation with "
            "the same seed is exactly deterministic"
        )
    )


# ============================================================
# PRIVACY
# ============================================================


def verify_privacy_minimal_result(
    *,
    explainability_result,
) -> None:

    payload = (
        explainability_result
        .model_dump(
            mode="json"
        )
    )


    forbidden = {
        "raw_rows",
        "predictions",
        "permuted_rows",
        "x_train",
        "x_test",
        "y_train",
        "y_test",
        "estimator",
        "model_bytes",
        "model_path",
    }


    assert (
        forbidden
        .isdisjoint(
            payload
        )
    )


    print(
        (
            "[PASS] real explanation result remains "
            "privacy-minimal"
        )
    )


# ============================================================
# RULE VERSIONS
# ============================================================


def verify_rule_versions(
) -> None:

    assert (
        ML_MODEL_EXPLAINABILITY_RULE_VERSION
        ==
        "ml_model_explainability_v0.1"
    )


    assert (
        ML_MODEL_EXPLAINABILITY_EXECUTOR_RULE_VERSION
        ==
        "ml_model_explainability_executor_v0.1"
    )


    assert (
        ML_MODEL_EXPLAINABILITY_GOLDEN_PATH_RULE_VERSION
        ==
        "ml_model_explainability_golden_path_v0.1"
    )


    print(
        "[PASS] Model Explainability rule versions"
    )


# ============================================================
# GOLDEN PATH
# ============================================================


def test_ml_model_explainability_golden_path_v0_1(
) -> None:

    reset_ml_product_state()


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
        # REAL TRAINING + PERSISTENCE
        # ----------------------------------------------------

        (
            training_contract,
            training_result,
        ) = (
            train_and_persist_real_model(
                workflow_id=
                    workflow_id
            )
        )


        # ----------------------------------------------------
        # REAL TRUSTED RELOAD
        # ----------------------------------------------------

        verify_real_trusted_reload(
            workflow_id=
                workflow_id,

            training_result=
                training_result,
        )


        # ----------------------------------------------------
        # REAL EXPLAINABILITY
        # ----------------------------------------------------

        explainability_result = (
            run_real_explainability(
                workflow_id=
                    workflow_id,

                training_result=
                    training_result,
            )
        )


        verify_real_explainability_authority(
            training_contract=
                training_contract,

            training_result=
                training_result,

            explainability_result=
                explainability_result,
        )


        verify_real_feature_importances(
            training_result=
                training_result,

            explainability_result=
                explainability_result,
        )


        verify_privacy_minimal_result(
            explainability_result=
                explainability_result
        )


        # ----------------------------------------------------
        # REAL DETERMINISTIC REPEAT
        # ----------------------------------------------------

        verify_real_explainability_is_deterministic(
            workflow_id=
                workflow_id,

            training_result=
                training_result,

            first_result=
                explainability_result,
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
        "DATALENS ML MODEL EXPLAINABILITY GOLDEN PATH E2E v0.1"
    )


    print(
        "="
        *
        78
    )


    print(
        "Preparation : real FastAPI + SQLite + Artifact Store"
    )


    print(
        "Training    : real scikit-learn LinearRegression"
    )


    print(
        "Artifact    : real persisted Model Artifact"
    )


    print(
        "Reload      : trusted SHA-verified joblib boundary"
    )


    print(
        "Explain     : holdout-only Permutation Feature Importance"
    )


    print(
        "Features    : original Training Contract columns"
    )


    print(
        "Persistence : no additional Model Artifact / Experiment"
    )


    print()


    test_ml_model_explainability_golden_path_v0_1()


    print()

    print(
        "="
        *
        78
    )


    print(
        (
            "PASS - Preparation -> Train -> Model Artifact -> "
            "Trusted Reload -> Holdout Permutation Importance -> "
            "Artifact/Experiment Provenance -> "
            "Deterministic Repeat -> No New Persistence"
        )
    )


    print(
        "="
        *
        78
    )


if __name__ == "__main__":
    main()
