from __future__ import annotations


import pandas as pd


from fastapi.testclient import (
    TestClient,
)


# ============================================================
# REAL ISOLATED PRODUCT ENVIRONMENT
#
# Reuse the established Model Lab Golden Path environment:
# - isolated SQLite
# - isolated Preparation Artifact Store
# - isolated Model Artifact Store
# - real DataLens FastAPI app
# ============================================================


from tests.e2e.test_model_lab_api_golden_path_v0_1 import (
    WORKFLOW_ROOT_DATASET_ID,
    app,
    build_real_classification_contract,
    create_preparation_session,
    ml_model_artifact_count,
    reset_product_state,
    run_real_cleaning_plan,
    run_real_quality,
    select_analysis_output,
    validate_preparation,
    verify_preparation_persistence,
    verify_real_handoff,
)


# ============================================================
# PRODUCT IMPORTS
# ============================================================


from app.api.model_lab_contracts import (
    ModelLabModelDetail,
    ModelLabModelListResponse,
    ModelLabPredictResponse,
)


from app.api.model_training import (
    MODEL_TRAINING_API_VERSION,
)


from app.api.model_training_contracts import (
    MODEL_TRAINING_API_CONTRACT_RULE_VERSION,
    MODEL_TRAINING_REQUEST_RULE_VERSION,
    ModelTrainingContextResponse,
)


from app.api.model_training_service import (
    MODEL_TRAINING_SERVICE_RULE_VERSION,
)


from app.preparation.analysis_input_handoff import (
    load_validated_analysis_input,
)


from app.preparation.analysis_readiness_gate import (
    require_analysis_readiness,
)


# ============================================================
# VERSION
# ============================================================


MODEL_TRAINING_API_GOLDEN_PATH_RULE_VERSION = (
    "model_training_api_golden_path_v0.1"
)


# ============================================================
# PRIVACY
# ============================================================


def all_json_keys(
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
                all_json_keys(
                    nested
                )
            )


    elif isinstance(
        value,
        list,
    ):

        for nested in value:

            keys.update(
                all_json_keys(
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
        "dataframe",
        "raw_rows",
        "x_train",
        "x_test",
        "y_train",
        "y_test",
    }


    assert (
        forbidden
        .isdisjoint(
            all_json_keys(
                payload
            )
        )
    )


# ============================================================
# PREPARATION
# ============================================================


def build_real_ready_workflow(
    *,
    client: TestClient,
) -> tuple[
    str,
    int,
]:

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


    readiness = (
        require_analysis_readiness(
            workflow_id=
                workflow_id
        )
    )


    assert (
        readiness.ready_for_analysis
        is True
    )


    assert (
        readiness.session_revision
        >
        0
    )


    assert (
        ml_model_artifact_count(
            workflow_id=
                workflow_id
        )
        ==
        0
    )


    print(
        "[PASS] Real Preparation workflow is READY"
    )


    print(
        (
            "[PASS] Model Artifact Store starts empty "
            "for the workflow"
        )
    )


    return (
        workflow_id,
        readiness.session_revision,
    )


# ============================================================
# GET /model-training/context
# ============================================================


def verify_real_training_context_http(
    *,
    client: TestClient,
    workflow_id: str,
    expected_revision: int,
) -> ModelTrainingContextResponse:

    response = client.get(
        "/model-training/context",
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
        ModelTrainingContextResponse
        .model_validate(
            payload
        )
    )


    assert (
        result.workflow_id
        ==
        workflow_id
    )


    assert (
        result.preparation_session_revision
        ==
        expected_revision
    )


    assert (
        result.dataset_count
        ==
        1
    )


    assert (
        len(
            result.datasets
        )
        ==
        1
    )


    dataset = (
        result.datasets[
            0
        ]
    )


    assert (
        dataset.dataset_id
        ==
        WORKFLOW_ROOT_DATASET_ID
    )


    assert (
        dataset.row_count
        ==
        30
    )


    column_metadata = {
        column.name:
            column

        for column
        in dataset.columns
    }


    for column_name in [
        "age",
        "tenure",
        "revenue",
        "segment",
    ]:

        assert (
            column_name
            in
            column_metadata
        )


    assert (
        column_metadata[
            "age"
        ].kind
        ==
        "numeric"
    )


    assert (
        column_metadata[
            "tenure"
        ].kind
        ==
        "numeric"
    )


    assert (
        column_metadata[
            "revenue"
        ].kind
        ==
        "numeric"
    )


    assert (
        column_metadata[
            "segment"
        ].kind
        ==
        "categorical"
    )


    assert_no_internal_artifact_keys(
        payload
    )


    print(
        (
            "[PASS] GET /model-training/context returned "
            "the real server-owned Preparation dataset"
        )
    )


    print(
        (
            "[PASS] Training context exposed schema metadata "
            "without raw rows or model internals"
        )
    )


    return result


# ============================================================
# STALE REVISION GUARD
# ============================================================


def verify_stale_revision_fails_closed(
    *,
    client: TestClient,
    workflow_id: str,
    expected_revision: int,
    training_contract,
) -> None:

    stale_revision = (
        expected_revision
        -
        1
    )


    assert (
        stale_revision
        >=
        0
    )


    response = client.post(
        "/model-training/train",
        json={
            "training":
                training_contract
                .model_dump(
                    mode="json"
                ),

            "expected_preparation_session_revision":
                stale_revision,
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


    detail = (
        payload[
            "detail"
        ]
    )


    assert (
        detail[
            "error"
        ]
        ==
        "training_input_invalid"
    )


    assert (
        detail[
            "workflow_id"
        ]
        ==
        workflow_id
    )


    assert (
        ml_model_artifact_count(
            workflow_id=
                workflow_id
        )
        ==
        0
    )


    readiness = (
        require_analysis_readiness(
            workflow_id=
                workflow_id
        )
    )


    assert (
        readiness.session_revision
        ==
        expected_revision
    )


    assert_no_internal_artifact_keys(
        payload
    )


    print(
        (
            "[PASS] Stale Preparation revision was rejected "
            "before training"
        )
    )


    print(
        (
            "[PASS] Failed stale training created zero "
            "Model Artifacts"
        )
    )


# ============================================================
# POST /model-training/train
# ============================================================


def verify_real_training_http(
    *,
    client: TestClient,
    workflow_id: str,
    expected_revision: int,
    training_contract,
) -> ModelLabModelDetail:

    response = client.post(
        "/model-training/train",
        json={
            "training":
                training_contract
                .model_dump(
                    mode="json"
                ),

            "expected_preparation_session_revision":
                expected_revision,
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
        result.feature_columns
        ==
        [
            "age",
            "tenure",
            "revenue",
        ]
    )


    assert (
        result.categorical_feature_columns
        ==
        []
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
        result.has_experiment_provenance
        is True
    )


    assert (
        result.preparation_session_revision
        ==
        expected_revision
    )


    assert (
        result.experiment_id
        is not None
    )


    assert (
        result.training_contract_sha256
        is not None
    )


    assert (
        len(
            result.training_contract_sha256
        )
        ==
        64
    )


    assert (
        result.metrics
    )


    assert (
        ml_model_artifact_count(
            workflow_id=
                workflow_id
        )
        ==
        1
    )


    assert_no_internal_artifact_keys(
        payload
    )


    print(
        (
            "[PASS] POST /model-training/train performed "
            "real Classical ML training"
        )
    )


    print(
        (
            "[PASS] HTTP training persisted exactly one "
            "trusted Model Artifact"
        )
    )


    print(
        (
            "[PASS] Training response preserved experiment "
            "provenance and remained privacy-minimal"
        )
    )


    return result


# ============================================================
# MODEL LAB RELOAD
# ============================================================


def verify_model_lab_sees_trained_model(
    *,
    client: TestClient,
    workflow_id: str,
    trained: ModelLabModelDetail,
) -> None:

    list_response = client.get(
        "/model-lab/models",
        params={
            "workflow_id":
                workflow_id
        },
    )


    assert (
        list_response.status_code
        ==
        200
    )


    list_payload = (
        list_response.json()
    )


    inventory = (
        ModelLabModelListResponse
        .model_validate(
            list_payload
        )
    )


    assert (
        inventory.model_count
        ==
        1
    )


    assert (
        inventory.models[
            0
        ].model_id
        ==
        trained.model_id
    )


    detail_response = client.get(
        (
            "/model-lab/models/"
            +
            trained.model_id
        ),
        params={
            "workflow_id":
                workflow_id
        },
    )


    assert (
        detail_response.status_code
        ==
        200
    )


    detail_payload = (
        detail_response.json()
    )


    restored = (
        ModelLabModelDetail
        .model_validate(
            detail_payload
        )
    )


    assert (
        restored.model_id
        ==
        trained.model_id
    )


    assert (
        restored.workflow_id
        ==
        trained.workflow_id
    )


    assert (
        restored.dataset_id
        ==
        trained.dataset_id
    )


    assert (
        restored.training_contract_sha256
        ==
        trained.training_contract_sha256
    )


    assert (
        restored.metrics
        ==
        trained.metrics
    )


    assert_no_internal_artifact_keys(
        list_payload
    )


    assert_no_internal_artifact_keys(
        detail_payload
    )


    print(
        (
            "[PASS] Model Lab inventory discovered the "
            "HTTP-trained Model Artifact"
        )
    )


    print(
        (
            "[PASS] Model Lab detail restored the exact "
            "persisted model identity"
        )
    )


# ============================================================
# TRUSTED PREDICTION AFTER HTTP TRAINING
# ============================================================


def python_scalar(
    value,
):

    if hasattr(
        value,
        "item",
    ):

        value = (
            value.item()
        )


    if pd.isna(
        value
    ):

        return None


    return value


def real_prediction_row(
    *,
    workflow_id: str,
    training_contract,
) -> tuple[
    dict,
    object,
]:

    handoff = (
        load_validated_analysis_input(
            workflow_id=
                workflow_id
        )
    )


    matching = [
        record

        for record
        in handoff.dataset_records

        if (
            isinstance(
                record,
                dict,
            )
            and
            record.get(
                "dataset_id"
            )
            ==
            training_contract.dataset_id
        )
    ]


    assert (
        len(
            matching
        )
        ==
        1
    )


    dataframe = (
        matching[
            0
        ][
            "dataframe"
        ]
    )


    assert isinstance(
        dataframe,
        pd.DataFrame,
    )


    required = [
        *training_contract.feature_columns,
        training_contract.target_column,
    ]


    usable = (
        dataframe
        .dropna(
            subset=
                required
        )
    )


    assert (
        not usable.empty
    )


    source_row = (
        usable.iloc[
            0
        ]
    )


    request_row = {
        feature:
            python_scalar(
                source_row[
                    feature
                ]
            )

        for feature
        in training_contract
        .feature_columns
    }


    actual_target = (
        python_scalar(
            source_row[
                training_contract
                .target_column
            ]
        )
    )


    return (
        request_row,
        actual_target,
    )


def verify_trusted_prediction_after_training(
    *,
    client: TestClient,
    workflow_id: str,
    trained: ModelLabModelDetail,
    training_contract,
) -> None:

    (
        row,
        actual_target,
    ) = (
        real_prediction_row(
            workflow_id=
                workflow_id,

            training_contract=
                training_contract,
        )
    )


    assert (
        set(
            row
        )
        ==
        set(
            training_contract
            .feature_columns
        )
    )


    response = client.post(
        "/model-lab/predict",
        json={
            "workflow_id":
                workflow_id,

            "model_id":
                trained.model_id,

            "rows":
                [
                    row
                ],
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
        ModelLabPredictResponse
        .model_validate(
            payload
        )
    )


    assert (
        result.workflow_id
        ==
        workflow_id
    )


    assert (
        result.model_id
        ==
        trained.model_id
    )


    assert (
        result.target_column
        ==
        "segment"
    )


    assert (
        result.prediction_count
        ==
        1
    )


    assert (
        len(
            result.predictions
        )
        ==
        1
    )


    assert (
        result.method
        ==
        "trusted_native_predict"
    )


    assert (
        ml_model_artifact_count(
            workflow_id=
                workflow_id
        )
        ==
        1
    )


    assert_no_internal_artifact_keys(
        payload
    )


    print(
        (
            "[PASS] Persisted HTTP-trained model was "
            "reloaded for trusted native prediction"
        )
    )


    print(
        (
            "[INFO] Golden Path actual target="
            f"{actual_target!r}, "
            "prediction="
            f"{result.predictions[0]!r}"
        )
    )


# ============================================================
# NO PREPARATION MUTATION
# ============================================================


def verify_preparation_not_mutated(
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
        readiness_after.ready_for_analysis
        is True
    )


    assert (
        readiness_after.session_revision
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
            "[PASS] Model Training + Model Lab prediction "
            "did not mutate Preparation revision"
        )
    )


    print(
        (
            "[PASS] Exactly one Model Artifact remains "
            "persisted"
        )
    )


# ============================================================
# VERSIONS
# ============================================================


def verify_rule_versions(
) -> None:

    assert (
        MODEL_TRAINING_API_CONTRACT_RULE_VERSION
        ==
        "model_training_api_contract_v0.1"
    )


    assert (
        MODEL_TRAINING_REQUEST_RULE_VERSION
        ==
        "model_training_request_v0.1"
    )


    assert (
        MODEL_TRAINING_SERVICE_RULE_VERSION
        ==
        "model_training_service_v0.1"
    )


    assert (
        MODEL_TRAINING_API_VERSION
        ==
        "model_training_api_v0.1"
    )


    assert (
        MODEL_TRAINING_API_GOLDEN_PATH_RULE_VERSION
        ==
        "model_training_api_golden_path_v0.1"
    )


    print(
        "[PASS] Model Training API v0.1 rule versions"
    )


# ============================================================
# GOLDEN PATH
# ============================================================


def test_model_training_api_golden_path_v0_1(
) -> None:

    reset_product_state()


    with TestClient(
        app
    ) as client:

        (
            workflow_id,
            preparation_revision,
        ) = (
            build_real_ready_workflow(
                client=
                    client
            )
        )


        # ----------------------------------------------------
        # REAL CONTEXT
        # ----------------------------------------------------


        context = (
            verify_real_training_context_http(
                client=
                    client,

                workflow_id=
                    workflow_id,

                expected_revision=
                    preparation_revision,
            )
        )


        # ----------------------------------------------------
        # REAL TRAINING CONTRACT
        # ----------------------------------------------------


        training_contract = (
            build_real_classification_contract(
                workflow_id=
                    workflow_id
            )
        )


        assert (
            training_contract.dataset_id
            ==
            context.datasets[
                0
            ].dataset_id
        )


        # ----------------------------------------------------
        # STALE REVISION MUST FAIL CLOSED
        # ----------------------------------------------------


        verify_stale_revision_fails_closed(
            client=
                client,

            workflow_id=
                workflow_id,

            expected_revision=
                preparation_revision,

            training_contract=
                training_contract,
        )


        # ----------------------------------------------------
        # REAL HTTP TRAINING
        # ----------------------------------------------------


        trained = (
            verify_real_training_http(
                client=
                    client,

                workflow_id=
                    workflow_id,

                expected_revision=
                    preparation_revision,

                training_contract=
                    training_contract,
            )
        )


        # ----------------------------------------------------
        # MODEL LAB MUST SEE PERSISTED MODEL
        # ----------------------------------------------------


        verify_model_lab_sees_trained_model(
            client=
                client,

            workflow_id=
                workflow_id,

            trained=
                trained,
        )


        # ----------------------------------------------------
        # TRUSTED MODEL RELOAD / PREDICTION
        # ----------------------------------------------------


        verify_trusted_prediction_after_training(
            client=
                client,

            workflow_id=
                workflow_id,

            trained=
                trained,

            training_contract=
                training_contract,
        )


        # ----------------------------------------------------
        # PREPARATION MUST REMAIN IMMUTABLE
        # ----------------------------------------------------


        verify_preparation_not_mutated(
            workflow_id=
                workflow_id,

            expected_revision=
                preparation_revision,
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
        "DATALENS MODEL TRAINING API GOLDEN PATH E2E v0.1"
    )


    print(
        "="
        *
        78
    )


    print(
        "Preparation : real validated 30-row dataset"
    )

    print(
        "Context     : real GET /model-training/context"
    )

    print(
        "Revision    : stale revision rejected before fit"
    )

    print(
        "Training    : real POST /model-training/train"
    )

    print(
        "Estimator   : real LogisticRegression"
    )

    print(
        "Persistence : exactly one trusted Model Artifact"
    )

    print(
        "Reload      : real Model Lab list + detail"
    )

    print(
        "Prediction  : trusted_native_predict"
    )

    print(
        "Privacy     : no raw rows / serialized model exposure"
    )

    print()


    test_model_training_api_golden_path_v0_1()


    print()

    print(
        "="
        *
        78
    )


    print(
        (
            "PASS - Preparation READY -> Training Context -> "
            "Stale Revision Guard -> HTTP Training -> "
            "Trusted Model Artifact -> Model Lab Reload -> "
            "Native Prediction -> No Preparation Mutation"
        )
    )


    print(
        "="
        *
        78
    )


if __name__ == "__main__":
    main()
