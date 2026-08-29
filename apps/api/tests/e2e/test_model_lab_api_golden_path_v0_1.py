from __future__ import annotations


import math


import pandas as pd


from fastapi.testclient import (
    TestClient,
)


# ============================================================
# REAL ISOLATED PRODUCT ENVIRONMENT
#
# IMPORTANT:
# Import the established ML Golden Path before importing
# DataLens production modules.
#
# This establishes isolated:
# - SQLite
# - Preparation Artifact Store
# - Model Artifact Store
# - runtime trace
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


from app.api.model_lab import (
    MODEL_LAB_API_VERSION,
)


from app.api.model_lab_contracts import (
    MODEL_LAB_API_CONTRACT_RULE_VERSION,
    ModelLabModelDetail,
    ModelLabModelListResponse,
    ModelLabPredictResponse,
)


from app.api.model_lab_service import (
    MODEL_LAB_SERVICE_RULE_VERSION,
)


from app.ml.experiment_provenance import (
    ml_training_contract_sha256,
)


from app.ml.model_evaluation_summary import (
    MLModelEvaluationSummaryResult,
)


from app.ml.model_loader import (
    load_trusted_ml_model,
)


from app.preparation.analysis_readiness_gate import (
    require_analysis_readiness,
)


# ============================================================
# VERSION
# ============================================================


MODEL_LAB_API_GOLDEN_PATH_RULE_VERSION = (
    "model_lab_api_golden_path_v0.1"
)


# ============================================================
# JSON PRIVACY
# ============================================================


def _all_json_keys(
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
                _all_json_keys(
                    nested
                )
            )

    elif isinstance(
        value,
        list,
    ):

        for nested in value:

            keys.update(
                _all_json_keys(
                    nested
                )
            )

    return keys


def assert_no_internal_artifact_keys(
    payload,
) -> None:

    forbidden = {
        "model_path",
        "model_file_bytes",
        "model_sha256",
        "model_bytes",
        "training_contract",
        "estimator",
    }

    assert (
        forbidden
        .isdisjoint(
            _all_json_keys(
                payload
            )
        )
    )


# ============================================================
# GET /models
# ============================================================


def verify_real_model_list_http(
    *,
    client: TestClient,
    workflow_id: str,
    training_result,
) -> ModelLabModelListResponse:

    response = client.get(
        "/model-lab/models",
        params={
            "workflow_id":
                workflow_id
        },
    )

    assert (
        response.status_code
        ==
        200
    )

    payload = (
        response.json()
    )

    result = (
        ModelLabModelListResponse
        .model_validate(
            payload
        )
    )

    artifact = (
        training_result
        .model_artifact
    )

    provenance = (
        training_result
        .experiment_provenance
    )

    assert (
        result.workflow_id
        ==
        workflow_id
    )

    assert (
        result.model_count
        ==
        1
    )

    assert (
        len(
            result.models
        )
        ==
        1
    )

    card = (
        result.models[
            0
        ]
    )

    assert (
        card.model_id
        ==
        artifact.model_id
    )

    assert (
        card.dataset_id
        ==
        WORKFLOW_ROOT_DATASET_ID
    )

    assert (
        card.problem_type
        ==
        "classification"
    )

    assert (
        card.target_column
        ==
        "segment"
    )

    assert (
        card.estimator_key
        ==
        "logistic_regression"
    )

    assert (
        card.feature_columns
        ==
        [
            "age",
            "tenure",
            "revenue",
        ]
    )

    assert (
        card.categorical_feature_columns
        ==
        []
    )

    assert (
        card.metrics
        ==
        artifact.metrics
    )

    assert (
        card.train_rows
        ==
        24
    )

    assert (
        card.test_rows
        ==
        6
    )

    assert (
        card.experiment_id
        ==
        provenance.experiment_id
    )

    assert (
        card.preparation_session_revision
        ==
        provenance
        .preparation_session_revision
    )

    assert (
        card.training_contract_sha256
        ==
        provenance
        .training_contract_sha256
    )

    assert (
        card.has_experiment_provenance
        is True
    )

    assert_no_internal_artifact_keys(
        payload
    )

    print(
        (
            "[PASS] GET /model-lab/models returned "
            "exactly one trusted privacy-minimal model"
        )
    )

    print(
        (
            "[PASS] Model Lab list preserved persisted "
            "model + experiment provenance identity"
        )
    )

    return result


# ============================================================
# GET /models/{model_id}
# ============================================================


def verify_real_model_detail_http(
    *,
    client: TestClient,
    workflow_id: str,
    training_contract,
    training_result,
) -> ModelLabModelDetail:

    model_id = (
        training_result
        .model_artifact
        .model_id
    )

    response = client.get(
        (
            "/model-lab/models/"
            f"{model_id}"
        ),
        params={
            "workflow_id":
                workflow_id
        },
    )

    assert (
        response.status_code
        ==
        200
    )

    payload = (
        response.json()
    )

    result = (
        ModelLabModelDetail
        .model_validate(
            payload
        )
    )

    assert (
        result.model_id
        ==
        model_id
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
        result.preprocessing
        ==
        training_contract.preprocessing
    )

    assert (
        result.split
        ==
        training_contract.split
    )

    assert (
        result
        .effective_estimator_hyperparameters
        ==
        training_contract
        .effective_estimator_hyperparameters
    )

    assert (
        result
        .effective_estimator_hyperparameters
        .kind
        ==
        "logistic_regression"
    )

    assert_no_internal_artifact_keys(
        payload
    )

    print(
        (
            "[PASS] GET /model-lab/models/{model_id} "
            "returned safe expanded training configuration"
        )
    )

    print(
        (
            "[PASS] Model Detail exposed effective "
            "hyperparameters without exposing Training Contract"
        )
    )

    return result


# ============================================================
# POST /predict
# ============================================================


def verify_real_prediction_http(
    *,
    client: TestClient,
    workflow_id: str,
    training_result,
) -> ModelLabPredictResponse:

    model_id = (
        training_result
        .model_artifact
        .model_id
    )

    rows = [
        {
            # Input order intentionally differs from the
            # persisted feature order.
            "revenue":
                120.0,

            "tenure":
                2,

            "age":
                22,
        },
        {
            "revenue":
                95.0,

            "tenure":
                1,

            "age":
                21,
        },
    ]

    response = client.post(
        "/model-lab/predict",
        json={
            "workflow_id":
                workflow_id,

            "model_id":
                model_id,

            "rows":
                rows,
        },
    )

    assert (
        response.status_code
        ==
        200
    )

    result = (
        ModelLabPredictResponse
        .model_validate(
            response.json()
        )
    )

    # --------------------------------------------------------
    # INDEPENDENT TRUSTED EXPECTATION
    # --------------------------------------------------------

    loaded = (
        load_trusted_ml_model(
            workflow_id=
                workflow_id,

            model_id=
                model_id,
        )
    )

    expected_frame = (
        pd.DataFrame(
            rows,
            columns=[
                "age",
                "tenure",
                "revenue",
            ],
        )
    )

    expected_raw = (
        loaded.predict(
            expected_frame
        )
    )

    expected_predictions = [
        (
            value.item()
            if callable(
                getattr(
                    value,
                    "item",
                    None,
                )
            )
            else value
        )

        for value
        in expected_raw
    ]

    assert (
        result.workflow_id
        ==
        workflow_id
    )

    assert (
        result.model_id
        ==
        model_id
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
        result.prediction_count
        ==
        2
    )

    assert (
        result.predictions
        ==
        expected_predictions
    )

    assert (
        result.method
        ==
        "trusted_native_predict"
    )

    for prediction in (
        result.predictions
    ):

        assert (
            prediction
            in
            {
                "premium",
                "standard",
            }
        )

    print(
        (
            "[PASS] POST /model-lab/predict crossed "
            "real trusted SHA-verified model reload"
        )
    )

    print(
        (
            "[PASS] HTTP prediction reconstructed exact "
            "persisted feature order before native predict()"
        )
    )

    print(
        (
            "[PASS] HTTP predictions exactly match an "
            "independent trusted native prediction"
        )
    )

    return result


# ============================================================
# INVALID PREDICTION INPUT
# ============================================================


def verify_real_prediction_contract_guard(
    *,
    client: TestClient,
    workflow_id: str,
    training_result,
) -> None:

    model_id = (
        training_result
        .model_artifact
        .model_id
    )

    response = client.post(
        "/model-lab/predict",
        json={
            "workflow_id":
                workflow_id,

            "model_id":
                model_id,

            "rows":
                [
                    {
                        "age":
                            22,

                        "tenure":
                            2,

                        # revenue deliberately missing
                    }
                ],
        },
    )

    assert (
        response.status_code
        ==
        422
    )

    payload = (
        response.json()
    )

    assert (
        payload[
            "detail"
        ][
            "error"
        ]
        ==
        "prediction_input_invalid"
    )

    assert (
        payload[
            "detail"
        ][
            "workflow_id"
        ]
        ==
        workflow_id
    )

    assert (
        payload[
            "detail"
        ][
            "model_id"
        ]
        ==
        model_id
    )

    print(
        (
            "[PASS] real HTTP prediction rejects "
            "missing persisted features before predict()"
        )
    )


# ============================================================
# POST /evaluate
# ============================================================


def verify_real_evaluation_http(
    *,
    client: TestClient,
    workflow_id: str,
    training_contract,
    training_result,
) -> MLModelEvaluationSummaryResult:

    artifact = (
        training_result
        .model_artifact
    )

    provenance = (
        training_result
        .experiment_provenance
    )

    response = client.post(
        "/model-lab/evaluate",
        json={
            "workflow_id":
                workflow_id,

            "model_id":
                artifact.model_id,

            "evaluation":
                {
                    "decision_threshold":
                        {
                            "threshold":
                                0.70
                        }
                },
        },
    )

    assert (
        response.status_code
        ==
        200
    )

    payload = (
        response.json()
    )

    result = (
        MLModelEvaluationSummaryResult
        .model_validate(
            payload
        )
    )

    # --------------------------------------------------------
    # AUTHORITY
    # --------------------------------------------------------

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
        artifact.model_id
    )

    assert (
        result.experiment_id
        ==
        provenance.experiment_id
    )

    assert (
        result.problem_type
        ==
        "classification"
    )

    assert (
        result.target_column
        ==
        training_contract.target_column
    )

    assert (
        result.estimator_key
        ==
        "logistic_regression"
    )

    assert (
        result.preparation_session_revision
        ==
        provenance
        .preparation_session_revision
    )

    assert (
        result.training_contract_sha256
        ==
        ml_training_contract_sha256(
            training_contract
        )
    )

    assert (
        result.train_rows
        ==
        24
    )

    assert (
        result.test_rows
        ==
        6
    )

    assert (
        result.metrics
        ==
        artifact.metrics
    )

    # --------------------------------------------------------
    # SERVER-OWNED EVALUATION EVIDENCE
    # --------------------------------------------------------

    assert (
        result.evaluation_status
        ==
        "complete"
    )

    assert (
        result.method
        ==
        "trusted_model_evaluation_summary"
    )

    assert (
        result.selection_evidence.source
        ==
        "standalone_model"
    )

    assert (
        result.selection_evidence.status
        ==
        "selection_not_available"
    )

    assert (
        result.selection_evidence.rank
        is None
    )

    assert (
        result.selection_evidence.metric_scope
        ==
        "not_available"
    )

    assert (
        result.classification_diagnostics
        is not None
    )

    assert (
        result
        .classification_diagnostics
        .evaluation_rows
        ==
        6
    )

    assert (
        result.decision_threshold_evaluation
        is not None
    )

    assert (
        result.summary_contract
        .decision_threshold
        is not None
    )

    assert math.isclose(
        float(
            result
            .summary_contract
            .decision_threshold
            .threshold
        ),
        0.70,
        rel_tol=0.0,
        abs_tol=1e-12,
    )

    assert (
        result.explainability
        .evaluation_rows
        ==
        6
    )

    assert (
        len(
            result.limitations
        )
        >=
        3
    )

    # --------------------------------------------------------
    # PRIVACY
    # --------------------------------------------------------

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
            _all_json_keys(
                payload
            )
        )
    )

    print(
        (
            "[PASS] POST /model-lab/evaluate produced "
            "real deterministic Evaluation Summary"
        )
    )

    print(
        (
            "[PASS] public evaluate remained standalone "
            "and inferred no selection provenance"
        )
    )

    print(
        (
            "[PASS] HTTP evaluation included real "
            "diagnostics + explicit threshold 0.70 + explainability"
        )
    )

    print(
        (
            "[PASS] HTTP Evaluation Summary remained "
            "privacy-minimal"
        )
    )

    return result


# ============================================================
# NO MUTATION / NO ADDITIONAL PERSISTENCE
# ============================================================


def verify_read_only_model_lab(
    *,
    workflow_id: str,
    expected_revision: int,
) -> None:

    readiness_after = (
        require_analysis_readiness(
            workflow_id=
                workflow_id
        )
    )

    assert (
        readiness_after
        .session_revision
        ==
        expected_revision
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
            "[PASS] Model Lab HTTP reads / evaluate / predict "
            "did not mutate Preparation revision"
        )
    )

    print(
        (
            "[PASS] Model Lab HTTP created zero additional "
            "Model Artifacts"
        )
    )


# ============================================================
# VERSIONS
# ============================================================


def verify_rule_versions(
) -> None:

    assert (
        MODEL_LAB_API_CONTRACT_RULE_VERSION
        ==
        "model_lab_api_contract_v0.1"
    )

    assert (
        MODEL_LAB_SERVICE_RULE_VERSION
        ==
        "model_lab_service_v0.1"
    )

    assert (
        MODEL_LAB_API_VERSION
        ==
        "model_lab_api_v0.1"
    )

    assert (
        MODEL_LAB_API_GOLDEN_PATH_RULE_VERSION
        ==
        "model_lab_api_golden_path_v0.1"
    )

    print(
        "[PASS] Model Lab API v0.1 rule versions"
    )


# ============================================================
# GOLDEN PATH
# ============================================================


def test_model_lab_api_golden_path_v0_1(
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

        training_result = (
            train_real_classifier(
                workflow_id=
                    workflow_id,

                training_contract=
                    training_contract,
            )
        )

        verify_real_classifier_reload(
            workflow_id=
                workflow_id,

            execution_result=
                training_result,
        )

        readiness_before = (
            require_analysis_readiness(
                workflow_id=
                    workflow_id
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

        # ----------------------------------------------------
        # REAL MODEL LAB HTTP
        # ----------------------------------------------------

        verify_real_model_list_http(
            client=
                client,

            workflow_id=
                workflow_id,

            training_result=
                training_result,
        )

        verify_real_model_detail_http(
            client=
                client,

            workflow_id=
                workflow_id,

            training_contract=
                training_contract,

            training_result=
                training_result,
        )

        verify_real_prediction_http(
            client=
                client,

            workflow_id=
                workflow_id,

            training_result=
                training_result,
        )

        verify_real_prediction_contract_guard(
            client=
                client,

            workflow_id=
                workflow_id,

            training_result=
                training_result,
        )

        verify_real_evaluation_http(
            client=
                client,

            workflow_id=
                workflow_id,

            training_contract=
                training_contract,

            training_result=
                training_result,
        )

        # ----------------------------------------------------
        # READ-ONLY GUARANTEE
        # ----------------------------------------------------

        verify_read_only_model_lab(
            workflow_id=
                workflow_id,

            expected_revision=(
                readiness_before
                .session_revision
            ),
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
        "DATALENS MODEL LAB API GOLDEN PATH E2E v0.1"
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
        "List        : real GET /model-lab/models"
    )

    print(
        "Detail      : real GET /model-lab/models/{model_id}"
    )

    print(
        "Predict     : real POST /model-lab/predict"
    )

    print(
        "Evaluate    : real POST /model-lab/evaluate"
    )

    print(
        "Threshold   : explicit 0.70"
    )

    print(
        "Persistence : read-only / zero additional artifacts"
    )

    print()

    test_model_lab_api_golden_path_v0_1()

    print()

    print(
        "="
        *
        78
    )

    print(
        (
            "PASS - Preparation -> Classification Training -> "
            "Trusted Model Artifact -> Model Lab List -> "
            "Model Detail -> Trusted HTTP Prediction -> "
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
