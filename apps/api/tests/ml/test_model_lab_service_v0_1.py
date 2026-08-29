from __future__ import annotations

import math

import numpy as np

import app.api.model_lab_service as model_lab_service

from app.api.model_lab_contracts import (
    ModelLabEvaluateRequest,
    ModelLabPredictRequest,
)

from app.api.model_lab_service import (
    MODEL_LAB_SERVICE_RULE_VERSION,
    ModelLabArtifactError,
    ModelLabPredictionExecutionError,
    ModelLabPredictionInputError,
    evaluate_model_lab_model,
    get_model_lab_model_detail,
    list_model_lab_models,
    predict_model_lab,
)

from app.ml.contracts import (
    MLTrainingContract,
)

from app.ml.experiment_provenance import (
    MLExperimentProvenanceRecord,
    ml_training_contract_sha256,
)

from app.ml.model_artifacts import (
    MLModelArtifactRecord,
)

from app.ml.model_loader import (
    LoadedMLModel,
)

from tests.ml.test_ml_model_evaluation_summary_contract_v0_1 import (
    valid_classification_summary,
)


# ============================================================
# CONSTANTS
# ============================================================


WORKFLOW_ID = "prep:model-lab-service"
DATASET_ID = "dataset:model-lab-service"

MODEL_ID = "model:model-lab-service"

EXPERIMENT_ID = (
    "experiment:"
    +
    (
        "a"
        *
        32
    )
)


# ============================================================
# ASSERTIONS
# ============================================================


def expect_exception(
    exception_type,
    factory,
) -> None:

    try:
        factory()

    except exception_type:
        return

    raise AssertionError(
        (
            "Expected exception: "
            f"{exception_type.__name__}"
        )
    )


# ============================================================
# ARTIFACT FIXTURE
# ============================================================


def build_contract(
    *,
    estimator_key: str = (
        "logistic_regression"
    ),
) -> MLTrainingContract:

    return (
        MLTrainingContract(
            workflow_id=
                WORKFLOW_ID,

            dataset_id=
                DATASET_ID,

            problem_type=
                "classification",

            target_column=
                "churned",

            feature_columns=[
                "age",
                "segment",
            ],

            categorical_feature_columns=[
                "segment",
            ],

            estimator_key=
                estimator_key,
        )
    )


def build_artifact(
    *,
    model_id: str = MODEL_ID,
    created_at_utc: str = (
        "2026-08-29T10:00:00+00:00"
    ),
    with_provenance: bool = True,
    estimator_key: str = (
        "logistic_regression"
    ),
) -> MLModelArtifactRecord:

    contract = (
        build_contract(
            estimator_key=
                estimator_key
        )
    )

    metrics = {
        "accuracy":
            0.80,

        "f1_macro":
            0.78,

        "precision_macro":
            0.79,

        "recall_macro":
            0.77,

        "balanced_accuracy":
            0.77,
    }

    provenance = None

    if with_provenance:

        provenance = (
            MLExperimentProvenanceRecord(
                experiment_id=
                    EXPERIMENT_ID,

                workflow_id=
                    WORKFLOW_ID,

                dataset_id=
                    DATASET_ID,

                preparation_session_revision=
                    9,

                training_contract_sha256=(
                    ml_training_contract_sha256(
                        contract
                    )
                ),

                model_id=
                    model_id,

                train_rows=
                    80,

                test_rows=
                    20,

                metrics=
                    metrics,
            )
        )

    return (
        MLModelArtifactRecord(
            model_id=
                model_id,

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
                80,

            test_rows=
                20,

            created_at_utc=
                created_at_utc,

            serialization_format=
                "joblib",

            model_path=(
                "models/"
                f"{model_id.replace(':', '-')}.joblib"
            ),

            model_file_bytes=
                123,

            model_sha256=(
                "b"
                *
                64
            ),
        )
    )


# ============================================================
# MODEL CARD / DETAIL
# ============================================================


def test_safe_model_detail_projection(
) -> None:

    artifact = (
        build_artifact()
    )

    original = (
        model_lab_service
        .get_ml_model_artifact
    )

    model_lab_service.get_ml_model_artifact = (
        lambda **kwargs:
            artifact
    )

    try:
        detail = (
            get_model_lab_model_detail(
                workflow_id=
                    WORKFLOW_ID,

                model_id=
                    MODEL_ID,
            )
        )

    finally:
        model_lab_service.get_ml_model_artifact = (
            original
        )

    assert (
        detail.model_id
        ==
        MODEL_ID
    )

    assert (
        detail.feature_columns
        ==
        [
            "age",
            "segment",
        ]
    )

    assert (
        detail
        .categorical_feature_columns
        ==
        [
            "segment",
        ]
    )

    assert (
        detail
        .effective_estimator_hyperparameters
        .kind
        ==
        "logistic_regression"
    )

    payload = (
        detail.model_dump(
            mode="json"
        )
    )

    forbidden = {
        "model_path",
        "model_file_bytes",
        "model_sha256",
        "training_contract",
        "estimator",
        "model_bytes",
    }

    assert (
        forbidden.isdisjoint(
            payload
        )
    )


def test_legacy_artifact_projects_without_provenance(
) -> None:

    artifact = (
        build_artifact(
            with_provenance=
                False
        )
    )

    original = (
        model_lab_service
        .get_ml_model_artifact
    )

    model_lab_service.get_ml_model_artifact = (
        lambda **kwargs:
            artifact
    )

    try:
        detail = (
            get_model_lab_model_detail(
                workflow_id=
                    WORKFLOW_ID,

                model_id=
                    MODEL_ID,
            )
        )

    finally:
        model_lab_service.get_ml_model_artifact = (
            original
        )

    assert (
        detail.has_experiment_provenance
        is False
    )

    assert (
        detail.experiment_id
        is None
    )

    assert (
        detail.training_contract_sha256
        is None
    )


def test_unknown_effective_hyperparameters_fail_closed(
) -> None:

    artifact = (
        build_artifact(
            estimator_key=
                "future_estimator"
        )
    )

    original = (
        model_lab_service
        .get_ml_model_artifact
    )

    model_lab_service.get_ml_model_artifact = (
        lambda **kwargs:
            artifact
    )

    try:
        expect_exception(
            ModelLabArtifactError,

            lambda:
                get_model_lab_model_detail(
                    workflow_id=
                        WORKFLOW_ID,

                    model_id=
                        MODEL_ID,
                ),
        )

    finally:
        model_lab_service.get_ml_model_artifact = (
            original
        )


# ============================================================
# LIST
# ============================================================


def test_model_list_is_deterministically_ordered(
) -> None:

    newest_b = (
        build_artifact(
            model_id=
                "model:b",

            created_at_utc=(
                "2026-08-29T12:00:00+00:00"
            ),
        )
    )

    oldest = (
        build_artifact(
            model_id=
                "model:z",

            created_at_utc=(
                "2026-08-29T10:00:00+00:00"
            ),
        )
    )

    newest_a = (
        build_artifact(
            model_id=
                "model:a",

            created_at_utc=(
                "2026-08-29T12:00:00+00:00"
            ),
        )
    )

    artifacts = {
        newest_b.model_id:
            newest_b,

        oldest.model_id:
            oldest,

        newest_a.model_id:
            newest_a,
    }

    originals = {
        "index":
            model_lab_service
            .load_ml_model_artifact_index_workflow,

        "get":
            model_lab_service
            .get_ml_model_artifact,
    }

    model_lab_service.load_ml_model_artifact_index_workflow = (
        lambda **kwargs:
            [
                {
                    "model_id":
                        "model:z"
                },
                {
                    "model_id":
                        "model:b"
                },
                {
                    "model_id":
                        "model:a"
                },
            ]
    )

    model_lab_service.get_ml_model_artifact = (
        lambda **kwargs:
            artifacts[
                kwargs[
                    "model_id"
                ]
            ]
    )

    try:
        result = (
            list_model_lab_models(
                workflow_id=
                    WORKFLOW_ID
            )
        )

    finally:
        model_lab_service.load_ml_model_artifact_index_workflow = (
            originals[
                "index"
            ]
        )

        model_lab_service.get_ml_model_artifact = (
            originals[
                "get"
            ]
        )

    assert (
        result.model_count
        ==
        3
    )

    assert (
        [
            item.model_id
            for item
            in result.models
        ]
        ==
        [
            "model:a",
            "model:b",
            "model:z",
        ]
    )


def test_empty_model_list_is_valid(
) -> None:

    original = (
        model_lab_service
        .load_ml_model_artifact_index_workflow
    )

    model_lab_service.load_ml_model_artifact_index_workflow = (
        lambda **kwargs:
            []
    )

    try:
        result = (
            list_model_lab_models(
                workflow_id=
                    WORKFLOW_ID
            )
        )

    finally:
        model_lab_service.load_ml_model_artifact_index_workflow = (
            original
        )

    assert (
        result.model_count
        ==
        0
    )

    assert (
        result.models
        ==
        []
    )


def test_duplicate_index_model_id_fails_closed(
) -> None:

    original = (
        model_lab_service
        .load_ml_model_artifact_index_workflow
    )

    model_lab_service.load_ml_model_artifact_index_workflow = (
        lambda **kwargs:
            [
                {
                    "model_id":
                        MODEL_ID
                },
                {
                    "model_id":
                        MODEL_ID
                },
            ]
    )

    try:
        expect_exception(
            ModelLabArtifactError,

            lambda:
                list_model_lab_models(
                    workflow_id=
                        WORKFLOW_ID
                ),
        )

    finally:
        model_lab_service.load_ml_model_artifact_index_workflow = (
            original
        )


# ============================================================
# TRUSTED PREDICTION
# ============================================================


class CapturingLoadedModel:

    def __init__(
        self,
        artifact,
        predictions,
    ) -> None:

        self.artifact = artifact
        self.predictions = predictions
        self.calls = []


    def predict(
        self,
        features,
    ):

        self.calls.append(
            features.copy()
        )

        return self.predictions


def test_prediction_uses_exact_feature_contract(
) -> None:

    artifact = (
        build_artifact()
    )

    loaded = (
        CapturingLoadedModel(
            artifact=
                artifact,

            predictions=(
                np.asarray(
                    [
                        "yes",
                        "no",
                    ],
                    dtype=object,
                )
            ),
        )
    )

    original = (
        model_lab_service
        .load_trusted_ml_model
    )

    model_lab_service.load_trusted_ml_model = (
        lambda **kwargs:
            loaded
    )

    try:
        result = (
            predict_model_lab(
                ModelLabPredictRequest(
                    workflow_id=
                        WORKFLOW_ID,

                    model_id=
                        MODEL_ID,

                    rows=[
                        {
                            # Deliberately reversed input order.
                            "segment":
                                "premium",

                            "age":
                                42,
                        },
                        {
                            "segment":
                                "standard",

                            "age":
                                31,
                        },
                    ],
                )
            )
        )

    finally:
        model_lab_service.load_trusted_ml_model = (
            original
        )

    assert (
        len(
            loaded.calls
        )
        ==
        1
    )

    dataframe = (
        loaded.calls[
            0
        ]
    )

    assert (
        list(
            dataframe.columns
        )
        ==
        [
            "age",
            "segment",
        ]
    )

    assert (
        result.predictions
        ==
        [
            "yes",
            "no",
        ]
    )

    assert (
        result.method
        ==
        "trusted_native_predict"
    )


def test_prediction_missing_feature_fails_before_predict(
) -> None:

    artifact = (
        build_artifact()
    )

    loaded = (
        CapturingLoadedModel(
            artifact=
                artifact,

            predictions=
                [],
        )
    )

    original = (
        model_lab_service
        .load_trusted_ml_model
    )

    model_lab_service.load_trusted_ml_model = (
        lambda **kwargs:
            loaded
    )

    try:
        expect_exception(
            ModelLabPredictionInputError,

            lambda:
                predict_model_lab(
                    ModelLabPredictRequest(
                        workflow_id=
                            WORKFLOW_ID,

                        model_id=
                            MODEL_ID,

                        rows=[
                            {
                                "age":
                                    42
                            }
                        ],
                    )
                ),
        )

    finally:
        model_lab_service.load_trusted_ml_model = (
            original
        )

    assert (
        loaded.calls
        ==
        []
    )


def test_prediction_extra_feature_fails_before_predict(
) -> None:

    artifact = (
        build_artifact()
    )

    loaded = (
        CapturingLoadedModel(
            artifact=
                artifact,

            predictions=
                [],
        )
    )

    original = (
        model_lab_service
        .load_trusted_ml_model
    )

    model_lab_service.load_trusted_ml_model = (
        lambda **kwargs:
            loaded
    )

    try:
        expect_exception(
            ModelLabPredictionInputError,

            lambda:
                predict_model_lab(
                    ModelLabPredictRequest(
                        workflow_id=
                            WORKFLOW_ID,

                        model_id=
                            MODEL_ID,

                        rows=[
                            {
                                "age":
                                    42,

                                "segment":
                                    "premium",

                                "invented":
                                    1,
                            }
                        ],
                    )
                ),
        )

    finally:
        model_lab_service.load_trusted_ml_model = (
            original
        )

    assert (
        loaded.calls
        ==
        []
    )


def test_prediction_count_mismatch_fails_closed(
) -> None:

    artifact = (
        build_artifact()
    )

    loaded = (
        CapturingLoadedModel(
            artifact=
                artifact,

            predictions=[
                "yes",
            ],
        )
    )

    original = (
        model_lab_service
        .load_trusted_ml_model
    )

    model_lab_service.load_trusted_ml_model = (
        lambda **kwargs:
            loaded
    )

    try:
        expect_exception(
            ModelLabPredictionExecutionError,

            lambda:
                predict_model_lab(
                    ModelLabPredictRequest(
                        workflow_id=
                            WORKFLOW_ID,

                        model_id=
                            MODEL_ID,

                        rows=[
                            {
                                "age":
                                    42,

                                "segment":
                                    "premium",
                            },
                            {
                                "age":
                                    31,

                                "segment":
                                    "standard",
                            },
                        ],
                    )
                ),
        )

    finally:
        model_lab_service.load_trusted_ml_model = (
            original
        )


def test_nonfinite_prediction_fails_closed(
) -> None:

    artifact = (
        build_artifact()
    )

    loaded = (
        CapturingLoadedModel(
            artifact=
                artifact,

            predictions=
                np.asarray(
                    [
                        math.inf
                    ]
                ),
        )
    )

    original = (
        model_lab_service
        .load_trusted_ml_model
    )

    model_lab_service.load_trusted_ml_model = (
        lambda **kwargs:
            loaded
    )

    try:
        expect_exception(
            ModelLabPredictionExecutionError,

            lambda:
                predict_model_lab(
                    ModelLabPredictRequest(
                        workflow_id=
                            WORKFLOW_ID,

                        model_id=
                            MODEL_ID,

                        rows=[
                            {
                                "age":
                                    42,

                                "segment":
                                    "premium",
                            }
                        ],
                    )
                ),
        )

    finally:
        model_lab_service.load_trusted_ml_model = (
            original
        )


# ============================================================
# EVALUATION
# ============================================================


def test_evaluation_delegates_without_selection_context(
) -> None:

    expected = (
        valid_classification_summary(
            with_threshold=
                False,

            selection_source=
                "standalone_model",
        )
    )

    calls = []

    original = (
        model_lab_service
        .execute_ml_model_evaluation_summary
    )

    def fake_summary_executor(
        **kwargs,
    ):

        calls.append(
            kwargs
        )

        return expected

    model_lab_service.execute_ml_model_evaluation_summary = (
        fake_summary_executor
    )

    try:
        result = (
            evaluate_model_lab_model(
                ModelLabEvaluateRequest(
                    workflow_id=
                        WORKFLOW_ID,

                    model_id=
                        MODEL_ID,
                )
            )
        )

    finally:
        model_lab_service.execute_ml_model_evaluation_summary = (
            original
        )

    assert (
        result
        ==
        expected
    )

    assert (
        len(
            calls
        )
        ==
        1
    )

    assert (
        set(
            calls[
                0
            ]
        )
        ==
        {
            "workflow_id",
            "model_id",
            "summary_contract",
        }
    )

    assert (
        "selection_context"
        not in
        calls[
            0
        ]
    )


# ============================================================
# PRIVACY
# ============================================================


def test_model_card_projection_does_not_leak_artifact_fields(
) -> None:

    artifact = (
        build_artifact()
    )

    card = (
        model_lab_service
        ._model_card_from_artifact(
            artifact
        )
    )

    payload = (
        card.model_dump(
            mode="json"
        )
    )

    forbidden = {
        "model_path",
        "model_file_bytes",
        "model_sha256",
        "model_bytes",
        "training_contract",
        "estimator",
    }

    assert (
        forbidden.isdisjoint(
            payload
        )
    )


# ============================================================
# RULE VERSION
# ============================================================


def test_rule_version(
) -> None:

    assert (
        MODEL_LAB_SERVICE_RULE_VERSION
        ==
        "model_lab_service_v0.1"
    )


# ============================================================
# MAIN
# ============================================================


def main(
) -> None:

    print(
        "=== DATALENS MODEL LAB SERVICE v0.1 ==="
    )

    tests = [
        (
            "Safe Model Detail projection",
            test_safe_model_detail_projection,
        ),
        (
            "Legacy artifact projection",
            test_legacy_artifact_projects_without_provenance,
        ),
        (
            "Unknown effective hyperparameters fail closed",
            test_unknown_effective_hyperparameters_fail_closed,
        ),
        (
            "Model list deterministic ordering",
            test_model_list_is_deterministically_ordered,
        ),
        (
            "Empty Model Lab list",
            test_empty_model_list_is_valid,
        ),
        (
            "Duplicate index model id blocked",
            test_duplicate_index_model_id_fails_closed,
        ),
        (
            "Prediction exact feature contract",
            test_prediction_uses_exact_feature_contract,
        ),
        (
            "Prediction missing feature blocked",
            test_prediction_missing_feature_fails_before_predict,
        ),
        (
            "Prediction extra feature blocked",
            test_prediction_extra_feature_fails_before_predict,
        ),
        (
            "Prediction count mismatch blocked",
            test_prediction_count_mismatch_fails_closed,
        ),
        (
            "Non-finite prediction blocked",
            test_nonfinite_prediction_fails_closed,
        ),
        (
            "Evaluation delegates without selection context",
            test_evaluation_delegates_without_selection_context,
        ),
        (
            "Model card projection remains privacy-minimal",
            test_model_card_projection_does_not_leak_artifact_fields,
        ),
        (
            "Model Lab Service rule version",
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
        "PASS - Model Lab Service v0.1"
    )


if __name__ == "__main__":
    main()
