from __future__ import annotations


import math


import numpy as np
import pandas as pd


from fastapi.testclient import (
    TestClient,
)


from sklearn.compose import (
    ColumnTransformer,
)


from sklearn.linear_model import (
    Ridge,
)


from sklearn.pipeline import (
    Pipeline,
)


# ============================================================
# REAL ISOLATED PRODUCT ENVIRONMENT
# ============================================================
#
# Import this Golden Path first because it establishes the
# isolated Preparation / SQLite / Model Artifact environment
# used by the real ML E2E suite.
# ============================================================


from tests.e2e.test_ml_hyperparameter_tuning_golden_path_v0_1 import (
    WORKFLOW_ROOT_DATASET_ID,
    app,
    build_real_tuning_training_contract,
    create_preparation_session,
    ml_model_artifact_count,
    reset_product_state,
    run_real_cleaning_plan,
    run_real_quality,
    select_analysis_output,
    validate_preparation,
    verify_preparation_persistence,
    verify_real_handoff,
    verify_real_tuning_handoff,
)


# ============================================================
# PRODUCT IMPORTS
# ============================================================


import app.ml.tuned_model_promotion_executor as promotion_executor


from app.ml.classical_executor import (
    ClassicalMLInputError,
    execute_classical_ml,
)


from app.ml.experiment_provenance import (
    ml_training_contract_sha256,
)


from app.ml.hyperparameter_tuning import (
    MLHyperparameterSearchContract,
    expected_hyperparameter_metric_names,
)


from app.ml.hyperparameter_tuning_executor import (
    execute_ml_hyperparameter_tuning,
)


from app.ml.model_loader import (
    load_trusted_ml_model,
)


from app.ml.tuned_model_promotion import (
    ML_TUNED_MODEL_PROMOTION_RULE_VERSION,
    MLTunedModelPromotionContract,
    build_promoted_training_contract,
)


from app.ml.tuned_model_promotion_executor import (
    ML_TUNED_MODEL_PROMOTION_EXECUTOR_RULE_VERSION,
    execute_ml_tuned_model_promotion,
)


from app.preparation.analysis_readiness_gate import (
    require_analysis_readiness,
)


# ============================================================
# VERSION
# ============================================================


ML_TUNED_MODEL_PROMOTION_GOLDEN_PATH_RULE_VERSION = (
    "ml_tuned_model_promotion_golden_path_v0.1"
)


# ============================================================
# SEARCH CONTRACT
# ============================================================


def build_real_search_contract(
) -> MLHyperparameterSearchContract:

    return (
        MLHyperparameterSearchContract(
            folds=
                5,

            shuffle=
                True,

            random_seed=
                73,
        )
    )


# ============================================================
# REAL TUNING PREVIEW
# ============================================================


def run_real_tuning_preview(
    *,
    workflow_id: str,
    training_contract,
    search_contract,
):
    """
    Execute the real tuning engine before promotion so the
    Golden Path has an independent deterministic reference
    for the expected rank-1 candidate.

    This preview remains evaluation-only.

    It must create zero Model Artifacts and therefore zero
    persisted Experiment Provenance.
    """

    readiness_before = (
        require_analysis_readiness(
            workflow_id=
                workflow_id
        )
    )


    artifacts_before = (
        ml_model_artifact_count(
            workflow_id=
                workflow_id
        )
    )


    assert (
        artifacts_before
        ==
        0
    )


    base_before = (
        training_contract.model_dump(
            mode="json"
        )
    )


    base_sha_before = (
        ml_training_contract_sha256(
            training_contract
        )
    )


    result = (
        execute_ml_hyperparameter_tuning(
            training_contract=
                training_contract,

            search_contract=
                search_contract,
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
        readiness_before.session_revision
        ==
        readiness_after.session_revision
    )


    assert (
        result.preparation_session_revision
        ==
        readiness_before.session_revision
    )


    assert (
        artifacts_after
        ==
        artifacts_before
        ==
        0
    )


    assert (
        training_contract.model_dump(
            mode="json"
        )
        ==
        base_before
    )


    assert (
        ml_training_contract_sha256(
            training_contract
        )
        ==
        base_sha_before
    )


    assert (
        result.base_training_contract_sha256
        ==
        base_sha_before
    )


    assert (
        result.outer_train_rows
        ==
        24
    )


    assert (
        result.holdout_test_rows
        ==
        6
    )


    print(
        (
            "[PASS] real tuning preview remained "
            "evaluation-only with zero Model Artifacts"
        )
    )


    print(
        (
            "[PASS] real tuning preview isolated "
            "24 outer-train rows from 6 untouched holdout rows"
        )
    )


    return result


# ============================================================
# REAL PROMOTION
# ============================================================


def run_real_promotion(
    *,
    workflow_id: str,
    training_contract,
    search_contract,
    tuning_preview,
):
    """
    Execute the complete production promotion flow.

    Wrappers count calls but delegate immediately to the real
    production functions.

    Therefore this remains a real Golden Path while proving
    that one promotion invokes:

    - server-owned tuning exactly once;
    - final Classical ML exactly once.
    """

    expected_promoted_contract = (
        build_promoted_training_contract(
            base_training_contract=
                training_contract,

            tuning_result=
                tuning_preview,
        )
    )


    winner = (
        tuning_preview.candidate_results[
            0
        ]
    )


    assert (
        winner.rank
        ==
        1
    )


    assert (
        ml_training_contract_sha256(
            expected_promoted_contract
        )
        ==
        winner.training_contract_sha256
    )


    readiness_before = (
        require_analysis_readiness(
            workflow_id=
                workflow_id
        )
    )


    artifacts_before = (
        ml_model_artifact_count(
            workflow_id=
                workflow_id
        )
    )


    assert (
        artifacts_before
        ==
        0
    )


    base_before = (
        training_contract.model_dump(
            mode="json"
        )
    )


    base_sha_before = (
        ml_training_contract_sha256(
            training_contract
        )
    )


    real_tuning = (
        promotion_executor
        .execute_ml_hyperparameter_tuning
    )


    real_final_training = (
        promotion_executor
        .execute_classical_ml
    )


    tuning_calls = []
    final_training_calls = []


    def counting_real_tuning(
        *,
        training_contract,
        search_contract,
    ):

        result = (
            real_tuning(
                training_contract=
                    training_contract,

                search_contract=
                    search_contract,
            )
        )


        tuning_calls.append(
            result
        )


        return result


    def counting_real_final_training(
        *,
        training_contract,
        expected_preparation_session_revision=None,
    ):

        final_training_calls.append(
            {
                "training_contract":
                    training_contract,

                "expected_preparation_session_revision":
                    expected_preparation_session_revision,
            }
        )


        return (
            real_final_training(
                training_contract=
                    training_contract,

                expected_preparation_session_revision=(
                    expected_preparation_session_revision
                ),
            )
        )


    promotion_executor.execute_ml_hyperparameter_tuning = (
        counting_real_tuning
    )


    promotion_executor.execute_classical_ml = (
        counting_real_final_training
    )


    try:
        result = (
            execute_ml_tuned_model_promotion(
                promotion_contract=(
                    MLTunedModelPromotionContract(
                        base_training_contract=
                            training_contract,

                        search_contract=
                            search_contract,
                    )
                )
            )
        )

    finally:
        promotion_executor.execute_ml_hyperparameter_tuning = (
            real_tuning
        )

        promotion_executor.execute_classical_ml = (
            real_final_training
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


    # ========================================================
    # EXACT ORCHESTRATION COUNT
    # ========================================================


    assert (
        len(
            tuning_calls
        )
        ==
        1
    )


    assert (
        len(
            final_training_calls
        )
        ==
        1
    )


    print(
        (
            "[PASS] promotion replayed server-owned "
            "Hyperparameter Tuning exactly once"
        )
    )


    print(
        (
            "[PASS] promotion invoked final "
            "Classical ML exactly once"
        )
    )


    # ========================================================
    # DETERMINISTIC TUNING REPLAY
    # ========================================================


    internal_tuning = (
        tuning_calls[
            0
        ]
    )


    assert (
        internal_tuning.model_dump(
            mode="json"
        )
        ==
        tuning_preview.model_dump(
            mode="json"
        )
    )


    print(
        (
            "[PASS] promotion tuning replay exactly "
            "matched the deterministic tuning preview"
        )
    )


    # ========================================================
    # FINAL CONTRACT + REVISION PIN
    # ========================================================


    final_call = (
        final_training_calls[
            0
        ]
    )


    assert (
        final_call[
            "training_contract"
        ]
        ==
        expected_promoted_contract
    )


    assert (
        final_call[
            "expected_preparation_session_revision"
        ]
        ==
        tuning_preview
        .preparation_session_revision
    )


    print(
        (
            "[PASS] rank-1 Training Contract was "
            "materialized server-side with no caller override"
        )
    )


    print(
        (
            "[PASS] tuning Preparation revision was "
            "pinned into final training"
        )
    )


    # ========================================================
    # PRODUCT STATE
    # ========================================================


    assert (
        readiness_before.session_revision
        ==
        readiness_after.session_revision
        ==
        tuning_preview.preparation_session_revision
    )


    assert (
        artifacts_after
        ==
        1
    )


    assert (
        artifacts_before
        ==
        0
    )


    print(
        (
            "[PASS] complete promotion remained pinned "
            "to one Preparation revision"
        )
    )


    print(
        (
            "[PASS] promotion persisted exactly "
            "one final Model Artifact"
        )
    )


    # ========================================================
    # BASE CONTRACT IMMUTABILITY
    # ========================================================


    assert (
        training_contract.model_dump(
            mode="json"
        )
        ==
        base_before
    )


    assert (
        ml_training_contract_sha256(
            training_contract
        )
        ==
        base_sha_before
    )


    print(
        (
            "[PASS] base Training Contract and SHA-256 "
            "remained immutable during promotion"
        )
    )


    return (
        result,
        expected_promoted_contract,
    )


# ============================================================
# RESULT AUTHORITY
# ============================================================


def verify_real_promotion_result(
    *,
    result,
    training_contract,
    promoted_contract,
    tuning_preview,
) -> None:

    winner = (
        tuning_preview.candidate_results[
            0
        ]
    )


    primary_summary = (
        winner.metric_summary[
            tuning_preview.primary_metric
        ]
    )


    assert (
        result.workflow_id
        ==
        training_contract.workflow_id
    )


    assert (
        result.dataset_id
        ==
        WORKFLOW_ROOT_DATASET_ID
        ==
        training_contract.dataset_id
    )


    assert (
        result.problem_type
        ==
        "regression"
    )


    assert (
        result.estimator_key
        ==
        "ridge_regression"
    )


    assert (
        result.preparation_session_revision
        ==
        tuning_preview.preparation_session_revision
    )


    assert (
        result.base_training_contract_sha256
        ==
        ml_training_contract_sha256(
            training_contract
        )
    )


    assert (
        result.promoted_training_contract_sha256
        ==
        ml_training_contract_sha256(
            promoted_contract
        )
        ==
        winner.training_contract_sha256
    )


    assert (
        result.base_training_contract_sha256
        !=
        result.promoted_training_contract_sha256
    )


    assert (
        result.selected_candidate_index
        ==
        tuning_preview.best_candidate_index
        ==
        winner.candidate_index
    )


    assert (
        result.selected_hyperparameters
        ==
        winner.hyperparameters
        ==
        promoted_contract.estimator_hyperparameters
    )


    assert (
        result.tuning_primary_metric
        ==
        "rmse"
        ==
        tuning_preview.primary_metric
    )


    assert math.isclose(
        result.tuning_primary_metric_mean,
        primary_summary.mean,
        rel_tol=0.0,
        abs_tol=0.0,
    )


    assert math.isclose(
        result.tuning_primary_metric_std,
        primary_summary.std,
        rel_tol=0.0,
        abs_tol=0.0,
    )


    assert (
        result.selection_policy
        ==
        "rank_1_only"
    )


    assert (
        result.holdout_policy
        ==
        "single_final_evaluation"
    )


    print(
        (
            "[PASS] promotion result is bound to "
            "the deterministic tuning rank-1 candidate"
        )
    )


    print(
        (
            "[PASS] promoted Training Contract SHA-256 "
            "matches the tuning winner fingerprint"
        )
    )


# ============================================================
# HOLDOUT + FINAL METRICS
# ============================================================


def verify_real_final_holdout(
    *,
    result,
    tuning_preview,
) -> None:

    assert (
        result.train_rows
        ==
        tuning_preview.outer_train_rows
        ==
        24
    )


    assert (
        result.test_rows
        ==
        tuning_preview.holdout_test_rows
        ==
        6
    )


    assert (
        result.train_rows
        +
        result.test_rows
        ==
        30
    )


    expected_metrics = set(
        expected_hyperparameter_metric_names(
            problem_type=
                "regression"
        )
    )


    assert (
        set(
            result.final_metrics
        )
        ==
        expected_metrics
        ==
        {
            "mae",
            "rmse",
            "r2",
            "median_absolute_error",
            "explained_variance",
        }
    )


    for metric_value in (
        result.final_metrics.values()
    ):

        assert math.isfinite(
            float(
                metric_value
            )
        )


    print(
        (
            "[PASS] final model reused the exact "
            "24/6 outer holdout isolated before tuning"
        )
    )


    print(
        (
            "[PASS] untouched holdout produced exactly "
            "five finite final regression metrics"
        )
    )


# ============================================================
# PERSISTED ARTIFACT + PROVENANCE
# ============================================================


def verify_real_artifact_and_provenance(
    *,
    workflow_id: str,
    result,
    promoted_contract,
):

    assert (
        ml_model_artifact_count(
            workflow_id=
                workflow_id
        )
        ==
        1
    )


    loaded = (
        load_trusted_ml_model(
            workflow_id=
                workflow_id,

            model_id=
                result.model_id,
        )
    )


    artifact = (
        loaded.artifact
    )


    provenance = (
        artifact.experiment_provenance
    )


    assert (
        provenance
        is not None
    )


    assert (
        artifact.workflow_id
        ==
        workflow_id
    )


    assert (
        artifact.dataset_id
        ==
        WORKFLOW_ROOT_DATASET_ID
    )


    assert (
        artifact.model_id
        ==
        result.model_id
    )


    assert (
        artifact.training_contract
        ==
        promoted_contract
    )


    assert (
        ml_training_contract_sha256(
            artifact.training_contract
        )
        ==
        result.promoted_training_contract_sha256
    )


    assert (
        artifact.metrics
        ==
        result.final_metrics
    )


    assert (
        artifact.serialization_format
        ==
        "joblib"
    )


    assert (
        artifact.model_file_bytes
        >
        0
    )


    assert (
        len(
            artifact.model_sha256
        )
        ==
        64
    )


    # --------------------------------------------------------
    # EXPERIMENT PROVENANCE
    # --------------------------------------------------------


    assert (
        provenance.experiment_id
        ==
        result.experiment_id
    )


    assert (
        provenance.model_id
        ==
        result.model_id
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
        provenance.preparation_session_revision
        ==
        result.preparation_session_revision
    )


    assert (
        provenance.training_contract_sha256
        ==
        result.promoted_training_contract_sha256
    )


    assert (
        provenance.train_rows
        ==
        result.train_rows
        ==
        24
    )


    assert (
        provenance.test_rows
        ==
        result.test_rows
        ==
        6
    )


    assert (
        provenance.metrics
        ==
        result.final_metrics
    )


    print(
        (
            "[PASS] final promotion persisted one "
            "Model Artifact with the promoted Training Contract"
        )
    )


    print(
        (
            "[PASS] final Model Artifact carries one "
            "Experiment Provenance bound to the promoted SHA-256"
        )
    )


    print(
        (
            "[PASS] promoted Model Artifact crossed "
            "trusted SHA-verified reload"
        )
    )


    return loaded


# ============================================================
# TRUSTED PIPELINE
# ============================================================


def verify_real_promoted_pipeline(
    *,
    loaded,
) -> None:

    pipeline = (
        loaded.estimator
    )


    assert isinstance(
        pipeline,
        Pipeline,
    )


    assert (
        "preprocessor"
        in
        pipeline.named_steps
    )


    assert (
        "estimator"
        in
        pipeline.named_steps
    )


    assert isinstance(
        pipeline.named_steps[
            "preprocessor"
        ],
        ColumnTransformer,
    )


    assert isinstance(
        pipeline.named_steps[
            "estimator"
        ],
        Ridge,
    )


    print(
        (
            "[PASS] trusted reload restored the "
            "promoted Ridge + ColumnTransformer pipeline"
        )
    )


# ============================================================
# TRUSTED PREDICTION
# ============================================================


def verify_real_promoted_predictions(
    *,
    loaded,
) -> None:

    known_features = (
        pd.DataFrame(
            {
                "age": [
                    33.0,
                    44.0,
                ],

                "tenure": [
                    6.0,
                    11.0,
                ],

                "segment": [
                    "standard",
                    "premium",
                ],
            }
        )
    )


    first_predictions = (
        np.asarray(
            loaded.predict(
                known_features
            ),
            dtype=np.float64,
        )
    )


    second_predictions = (
        np.asarray(
            loaded.predict(
                known_features
            ),
            dtype=np.float64,
        )
    )


    assert (
        first_predictions.shape
        ==
        (
            2,
        )
    )


    assert (
        np.isfinite(
            first_predictions
        )
        .all()
    )


    assert np.array_equal(
        first_predictions,
        second_predictions,
    )


    unseen_features = (
        pd.DataFrame(
            {
                "age": [
                    36.0,
                ],

                "tenure": [
                    8.0,
                ],

                "segment": [
                    "enterprise",
                ],
            }
        )
    )


    unseen_prediction = (
        np.asarray(
            loaded.predict(
                unseen_features
            ),
            dtype=np.float64,
        )
    )


    assert (
        unseen_prediction.shape
        ==
        (
            1,
        )
    )


    assert (
        np.isfinite(
            unseen_prediction
        )
        .all()
    )


    print(
        (
            "[PASS] promoted trusted model produces "
            "deterministic finite predictions"
        )
    )


    print(
        (
            "[PASS] persisted tuned preprocessing handles "
            "an unseen category without refit"
        )
    )


# ============================================================
# REAL STALE-REVISION GUARD
# ============================================================


def verify_real_stale_revision_is_blocked(
    *,
    workflow_id: str,
    result,
    promoted_contract,
) -> None:
    """
    Exercise the new Classical ML revision pin against the
    real validated Preparation handoff.

    A deliberately stale/incorrect expected revision must fail
    before a second model can be fitted or persisted.
    """

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


    try:
        execute_classical_ml(
            training_contract=
                promoted_contract,

            expected_preparation_session_revision=(
                result.preparation_session_revision
                +
                1
            ),
        )

    except ClassicalMLInputError:
        pass

    else:
        raise AssertionError(
            (
                "Real stale Preparation revision "
                "must fail closed before training."
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


    print(
        (
            "[PASS] real stale Preparation revision "
            "was blocked before a second Model Artifact"
        )
    )


# ============================================================
# PRIVACY
# ============================================================


def _collect_keys(
    value,
) -> set[
    str
]:

    keys: set[
        str
    ] = set()


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
                _collect_keys(
                    nested
                )
            )


    elif isinstance(
        value,
        list,
    ):

        for nested in value:

            keys.update(
                _collect_keys(
                    nested
                )
            )


    return keys


def verify_real_result_is_privacy_minimal(
    *,
    result,
) -> None:

    payload = (
        result.model_dump(
            mode="json"
        )
    )


    forbidden = {
        "raw_rows",
        "rows",
        "predictions",
        "fold_predictions",
        "holdout_predictions",
        "x_train",
        "x_test",
        "y_train",
        "y_test",
        "model_bytes",
        "model_path",
        "estimator",
        "tuning_result",
        "candidate_results",
    }


    payload_keys = (
        _collect_keys(
            payload
        )
    )


    assert (
        forbidden.isdisjoint(
            payload_keys
        )
    )


    print(
        (
            "[PASS] promotion result remains privacy-minimal "
            "without rows, predictions or model bytes"
        )
    )


# ============================================================
# RULE VERSIONS
# ============================================================


def verify_rule_versions(
) -> None:

    assert (
        ML_TUNED_MODEL_PROMOTION_RULE_VERSION
        ==
        "ml_tuned_model_promotion_v0.1"
    )


    assert (
        ML_TUNED_MODEL_PROMOTION_EXECUTOR_RULE_VERSION
        ==
        "ml_tuned_model_promotion_executor_v0.1"
    )


    assert (
        ML_TUNED_MODEL_PROMOTION_GOLDEN_PATH_RULE_VERSION
        ==
        "ml_tuned_model_promotion_golden_path_v0.1"
    )


    print(
        (
            "[PASS] Tuned Model Promotion "
            "rule versions"
        )
    )


# ============================================================
# GOLDEN PATH
# ============================================================


def test_ml_tuned_model_promotion_golden_path_v0_1(
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


        row_count = (
            verify_real_tuning_handoff(
                workflow_id=
                    workflow_id
            )
        )


        assert (
            row_count
            ==
            30
        )


        # ----------------------------------------------------
        # BASE CONTRACT + SEARCH CONTRACT
        # ----------------------------------------------------

        training_contract = (
            build_real_tuning_training_contract(
                workflow_id=
                    workflow_id
            )
        )


        search_contract = (
            build_real_search_contract()
        )


        # ----------------------------------------------------
        # INDEPENDENT REAL TUNING PREVIEW
        # ----------------------------------------------------

        tuning_preview = (
            run_real_tuning_preview(
                workflow_id=
                    workflow_id,

                training_contract=
                    training_contract,

                search_contract=
                    search_contract,
            )
        )


        # ----------------------------------------------------
        # COMPLETE REAL PROMOTION
        # ----------------------------------------------------

        (
            promotion_result,
            promoted_contract,
        ) = (
            run_real_promotion(
                workflow_id=
                    workflow_id,

                training_contract=
                    training_contract,

                search_contract=
                    search_contract,

                tuning_preview=
                    tuning_preview,
            )
        )


        # ----------------------------------------------------
        # RESULT AUTHORITY
        # ----------------------------------------------------

        verify_real_promotion_result(
            result=
                promotion_result,

            training_contract=
                training_contract,

            promoted_contract=
                promoted_contract,

            tuning_preview=
                tuning_preview,
        )


        # ----------------------------------------------------
        # FINAL HOLDOUT
        # ----------------------------------------------------

        verify_real_final_holdout(
            result=
                promotion_result,

            tuning_preview=
                tuning_preview,
        )


        # ----------------------------------------------------
        # PERSISTENCE + TRUSTED RELOAD
        # ----------------------------------------------------

        loaded = (
            verify_real_artifact_and_provenance(
                workflow_id=
                    workflow_id,

                result=
                    promotion_result,

                promoted_contract=
                    promoted_contract,
            )
        )


        verify_real_promoted_pipeline(
            loaded=
                loaded
        )


        verify_real_promoted_predictions(
            loaded=
                loaded
        )


        # ----------------------------------------------------
        # REAL REVISION FAIL-CLOSED
        # ----------------------------------------------------

        verify_real_stale_revision_is_blocked(
            workflow_id=
                workflow_id,

            result=
                promotion_result,

            promoted_contract=
                promoted_contract,
        )


        # ----------------------------------------------------
        # PRIVACY + VERSIONS
        # ----------------------------------------------------

        verify_real_result_is_privacy_minimal(
            result=
                promotion_result
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
        "DATALENS ML TUNED MODEL PROMOTION GOLDEN PATH E2E v0.1"
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
        "Tuning      : real train-only 5-fold inner CV"
    )


    print(
        "Selection   : deterministic rank #1 only"
    )


    print(
        "Promotion   : server-owned Training Contract"
    )


    print(
        "Revision    : pinned from tuning to final training"
    )


    print(
        "Final fit   : one real Classical ML execution"
    )


    print(
        "Holdout     : original untouched 24/6 split"
    )


    print(
        "Persistence : exactly one final Model Artifact"
    )


    print(
        "Provenance  : final promoted Training Contract SHA-256"
    )


    print(
        "Reload      : trusted SHA-verified joblib boundary"
    )


    print()


    test_ml_tuned_model_promotion_golden_path_v0_1()


    print()

    print(
        "="
        *
        78
    )


    print(
        (
            "PASS - Preparation -> Train-only Tuning -> "
            "Rank #1 -> Promoted Training Contract -> "
            "Pinned Revision -> One Final Holdout Training -> "
            "1 Model Artifact + Experiment Provenance -> "
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
