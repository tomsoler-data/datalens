from __future__ import annotations

from typing import (
    Any,
)

import pandas as pd

from app.api.model_lab_contracts import (
    ModelLabEvaluateRequest,
    ModelLabModelCard,
    ModelLabModelDetail,
    ModelLabModelListResponse,
    ModelLabPredictRequest,
    ModelLabPredictResponse,
)

from app.ml.model_artifact_index import (
    MLModelArtifactIndexError,
    load_ml_model_artifact_index_workflow,
)

from app.ml.model_artifact_store import (
    MLModelArtifactNotFoundError,
    MLModelArtifactStoreError,
    MLModelArtifactWorkflowMismatchError,
    get_ml_model_artifact,
    resolve_ml_model_artifact_store_path,
)

from app.ml.model_artifacts import (
    MLModelArtifactRecord,
)

from app.ml.model_evaluation_summary import (
    MLModelEvaluationSummaryResult,
)

from app.ml.model_evaluation_summary_executor import (
    MLModelEvaluationSummaryExecutorError,
    execute_ml_model_evaluation_summary,
)

from app.ml.model_loader import (
    MLModelLoaderError,
    load_trusted_ml_model,
)


# ============================================================
# VERSION
# ============================================================


MODEL_LAB_SERVICE_RULE_VERSION = (
    "model_lab_service_v0.1"
)


# ============================================================
# ERRORS
# ============================================================


class ModelLabServiceError(
    RuntimeError
):
    pass


class ModelLabArtifactError(
    ModelLabServiceError
):
    pass


class ModelLabModelNotFoundError(
    ModelLabServiceError
):
    pass


class ModelLabWorkflowMismatchError(
    ModelLabServiceError
):
    pass


class ModelLabPredictionInputError(
    ModelLabServiceError
):
    pass


class ModelLabPredictionExecutionError(
    ModelLabServiceError
):
    pass


class ModelLabEvaluationError(
    ModelLabServiceError
):
    pass


# ============================================================
# IDENTIFIERS
# ============================================================


def _required_identifier(
    value: object,
    *,
    field_name: str,
) -> str:

    normalized = str(
        value
        if value is not None
        else ""
    ).strip()

    if not normalized:
        raise ModelLabServiceError(
            f"{field_name} cannot be empty."
        )

    return normalized


# ============================================================
# TRUSTED ARTIFACT LOOKUP
# ============================================================


def _get_artifact(
    *,
    workflow_id: str,
    model_id: str,
) -> MLModelArtifactRecord:

    normalized_workflow_id = (
        _required_identifier(
            workflow_id,
            field_name=
                "workflow_id",
        )
    )

    normalized_model_id = (
        _required_identifier(
            model_id,
            field_name=
                "model_id",
        )
    )

    try:
        return (
            get_ml_model_artifact(
                workflow_id=
                    normalized_workflow_id,

                model_id=
                    normalized_model_id,
            )
        )

    except MLModelArtifactNotFoundError as error:
        raise ModelLabModelNotFoundError(
            (
                "Model Lab could not find the "
                "requested Model Artifact."
            )
        ) from error

    except (
        MLModelArtifactWorkflowMismatchError
    ) as error:
        raise ModelLabWorkflowMismatchError(
            (
                "Model Artifact does not belong "
                "to the requested workflow."
            )
        ) from error

    except (
        MLModelArtifactStoreError,
        ValueError,
    ) as error:
        raise ModelLabArtifactError(
            (
                "Model Lab could not restore "
                "trusted Model Artifact metadata."
            )
        ) from error


# ============================================================
# SAFE PROJECTION
# ============================================================


def _model_card_from_artifact(
    artifact: MLModelArtifactRecord,
) -> ModelLabModelCard:

    contract = (
        artifact.training_contract
    )

    provenance = (
        artifact.experiment_provenance
    )

    return (
        ModelLabModelCard(
            model_id=
                artifact.model_id,

            workflow_id=
                artifact.workflow_id,

            dataset_id=
                artifact.dataset_id,

            problem_type=
                contract.problem_type,

            target_column=
                contract.target_column,

            estimator_key=
                contract.estimator_key,

            feature_columns=
                list(
                    contract.feature_columns
                ),

            categorical_feature_columns=
                list(
                    contract
                    .categorical_feature_columns
                ),

            metrics=
                dict(
                    artifact.metrics
                ),

            train_rows=
                artifact.train_rows,

            test_rows=
                artifact.test_rows,

            created_at_utc=
                artifact.created_at_utc,

            experiment_id=(
                provenance.experiment_id
                if provenance
                is not None
                else None
            ),

            preparation_session_revision=(
                provenance
                .preparation_session_revision

                if provenance
                is not None

                else None
            ),

            training_contract_sha256=(
                provenance
                .training_contract_sha256

                if provenance
                is not None

                else None
            ),

            has_experiment_provenance=(
                provenance
                is not None
            ),
        )
    )


def _model_detail_from_artifact(
    artifact: MLModelArtifactRecord,
) -> ModelLabModelDetail:

    card = (
        _model_card_from_artifact(
            artifact
        )
    )

    contract = (
        artifact.training_contract
    )

    effective_hyperparameters = (
        contract
        .effective_estimator_hyperparameters
    )

    if effective_hyperparameters is None:
        raise ModelLabArtifactError(
            (
                "Model Artifact estimator does not "
                "have a supported effective "
                "hyperparameter contract."
            )
        )

    return (
        ModelLabModelDetail(
            **card.model_dump(
                mode="python"
            ),

            preprocessing=
                contract.preprocessing,

            split=
                contract.split,

            effective_estimator_hyperparameters=(
                effective_hyperparameters
            ),
        )
    )


# ============================================================
# LIST INDEX NORMALIZATION
# ============================================================


def _normalize_index_entries(
    value: object,
) -> list[
    dict[
        str,
        Any,
    ]
]:

    if isinstance(
        value,
        dict,
    ):
        raw_entries = list(
            value.values()
        )

    elif isinstance(
        value,
        (
            list,
            tuple,
        ),
    ):
        raw_entries = list(
            value
        )

    else:
        raise ModelLabArtifactError(
            (
                "Model Artifact workflow index "
                "returned an invalid collection."
            )
        )

    entries: list[
        dict[
            str,
            Any,
        ]
    ] = []

    for entry in raw_entries:

        if not isinstance(
            entry,
            dict,
        ):
            raise ModelLabArtifactError(
                (
                    "Model Artifact workflow index "
                    "contains an invalid entry."
                )
            )

        model_id = str(
            entry.get(
                "model_id",
                ""
            )
        ).strip()

        if not model_id:
            raise ModelLabArtifactError(
                (
                    "Model Artifact workflow index "
                    "entry is missing model_id."
                )
            )

        entries.append(
            entry
        )

    return entries


# ============================================================
# LIST MODELS
# ============================================================


def list_model_lab_models(
    *,
    workflow_id: str,
) -> ModelLabModelListResponse:

    normalized_workflow_id = (
        _required_identifier(
            workflow_id,
            field_name=
                "workflow_id",
        )
    )

    store_path = (
        resolve_ml_model_artifact_store_path()
    )

    try:
        raw_entries = (
            load_ml_model_artifact_index_workflow(
                store_path=
                    store_path,

                workflow_id=
                    normalized_workflow_id,
            )
        )

    except MLModelArtifactIndexError as error:
        raise ModelLabArtifactError(
            (
                "Model Lab could not read the "
                "Model Artifact workflow index."
            )
        ) from error

    entries = (
        _normalize_index_entries(
            raw_entries
        )
    )

    # ========================================================
    # FIRST PASS ? INDEX STRUCTURAL INTEGRITY
    #
    # Validate the complete identifier surface before touching
    # the authoritative Model Artifact Store.
    #
    # This ensures that a malformed / duplicated index fails
    # closed before any artifact lookup occurs.
    # ========================================================

    model_ids: list[
        str
    ] = []

    seen_model_ids: set[
        str
    ] = set()

    for entry in entries:

        model_id = str(
            entry[
                "model_id"
            ]
        ).strip()

        if model_id in seen_model_ids:
            raise ModelLabArtifactError(
                (
                    "Model Artifact workflow index "
                    "contains a duplicate model_id."
                )
            )

        seen_model_ids.add(
            model_id
        )

        model_ids.append(
            model_id
        )

    # ========================================================
    # SECOND PASS ? AUTHORITATIVE ARTIFACT RESTORE
    #
    # The index is discovery-only.
    #
    # Public metadata is still reconstructed exclusively from
    # the authoritative Model Artifact Store.
    # ========================================================

    cards: list[
        ModelLabModelCard
    ] = []

    for model_id in model_ids:

        artifact = (
            _get_artifact(
                workflow_id=
                    normalized_workflow_id,

                model_id=
                    model_id,
            )
        )

        cards.append(
            _model_card_from_artifact(
                artifact
            )
        )

    # Stable two-pass ordering:
    #
    # 1. model_id ascending for deterministic tie-breaking
    # 2. created_at descending
    cards.sort(
        key=lambda item:
            item.model_id
    )

    cards.sort(
        key=lambda item:
            item.created_at_utc,
        reverse=True,
    )

    return (
        ModelLabModelListResponse(
            workflow_id=
                normalized_workflow_id,

            model_count=
                len(
                    cards
                ),

            models=
                cards,
        )
    )


# ============================================================
# MODEL DETAIL
# ============================================================


def get_model_lab_model_detail(
    *,
    workflow_id: str,
    model_id: str,
) -> ModelLabModelDetail:

    artifact = (
        _get_artifact(
            workflow_id=
                workflow_id,

            model_id=
                model_id,
        )
    )

    return (
        _model_detail_from_artifact(
            artifact
        )
    )


# ============================================================
# EVALUATION
# ============================================================


def evaluate_model_lab_model(
    request: ModelLabEvaluateRequest,
) -> MLModelEvaluationSummaryResult:

    try:
        config = (
            ModelLabEvaluateRequest
            .model_validate(
                request
            )
        )

    except Exception as error:
        raise ModelLabEvaluationError(
            (
                "Model Lab evaluation request "
                "is invalid."
            )
        ) from error

    try:
        result = (
            execute_ml_model_evaluation_summary(
                workflow_id=
                    config.workflow_id,

                model_id=
                    config.model_id,

                summary_contract=
                    config.evaluation,
            )
        )

    except (
        MLModelEvaluationSummaryExecutorError
    ) as error:
        raise ModelLabEvaluationError(
            (
                "Model Lab could not evaluate "
                "the trusted Model Artifact."
            )
        ) from error

    try:
        return (
            MLModelEvaluationSummaryResult
            .model_validate(
                result.model_dump(
                    mode="python"
                )
            )
        )

    except Exception as error:
        raise ModelLabEvaluationError(
            (
                "Model Evaluation Summary returned "
                "an invalid result."
            )
        ) from error


# ============================================================
# PREDICTION FEATURE AUTHORITY
# ============================================================


def _prediction_dataframe(
    *,
    rows: list[
        dict[
            str,
            Any,
        ]
    ],
    feature_columns: list[
        str
    ],
) -> pd.DataFrame:

    expected = list(
        feature_columns
    )

    expected_set = set(
        expected
    )

    if not expected:
        raise ModelLabArtifactError(
            (
                "Model Artifact contains an empty "
                "feature contract."
            )
        )

    for (
        row_index,
        row,
    ) in enumerate(
        rows
    ):

        actual_set = set(
            row
        )

        missing = (
            expected_set
            -
            actual_set
        )

        extra = (
            actual_set
            -
            expected_set
        )

        if (
            missing
            or
            extra
        ):
            raise ModelLabPredictionInputError(
                (
                    "Prediction row does not match "
                    "the exact Model Artifact feature "
                    "contract. "
                    f"row={row_index}, "
                    f"missing={sorted(missing)}, "
                    f"extra={sorted(extra)}"
                )
            )

    try:
        dataframe = (
            pd.DataFrame(
                rows,
                columns=
                    expected,
            )
        )

    except Exception as error:
        raise ModelLabPredictionInputError(
            (
                "Prediction rows could not be "
                "materialized safely."
            )
        ) from error

    if (
        list(
            dataframe.columns
        )
        !=
        expected
    ):
        raise ModelLabPredictionInputError(
            (
                "Prediction feature ordering "
                "could not be reconstructed."
            )
        )

    if (
        len(
            dataframe
        )
        !=
        len(
            rows
        )
    ):
        raise ModelLabPredictionInputError(
            (
                "Prediction row count changed "
                "during materialization."
            )
        )

    return dataframe


# ============================================================
# PREDICTION NORMALIZATION
# ============================================================


def _python_prediction_scalar(
    value: object,
) -> object:

    item_method = getattr(
        value,
        "item",
        None,
    )

    if callable(
        item_method
    ):
        try:
            value = (
                item_method()
            )

        except Exception:
            pass

    if isinstance(
        value,
        (
            str,
            int,
            float,
            bool,
        ),
    ):
        return value

    raise ModelLabPredictionExecutionError(
        (
            "Trusted model produced a prediction "
            "that is not a supported JSON scalar."
        )
    )


def _prediction_list(
    *,
    predictions: object,
    expected_rows: int,
) -> list[
    object
]:

    tolist = getattr(
        predictions,
        "tolist",
        None,
    )

    if callable(
        tolist
    ):
        try:
            raw_values = (
                tolist()
            )

        except Exception as error:
            raise ModelLabPredictionExecutionError(
                (
                    "Trusted model predictions "
                    "could not be normalized."
                )
            ) from error

    else:
        try:
            raw_values = list(
                predictions
            )

        except Exception as error:
            raise ModelLabPredictionExecutionError(
                (
                    "Trusted model predictions "
                    "are not a one-dimensional "
                    "prediction sequence."
                )
            ) from error

    if not isinstance(
        raw_values,
        list,
    ):
        raise ModelLabPredictionExecutionError(
            (
                "Trusted model predictions "
                "must normalize to a list."
            )
        )

    if (
        len(
            raw_values
        )
        !=
        expected_rows
    ):
        raise ModelLabPredictionExecutionError(
            (
                "Trusted model prediction count "
                "does not match request row count."
            )
        )

    normalized = [
        _python_prediction_scalar(
            value
        )
        for value
        in raw_values
    ]

    return normalized


# ============================================================
# PREDICT
# ============================================================


def predict_model_lab(
    request: ModelLabPredictRequest,
) -> ModelLabPredictResponse:

    try:
        config = (
            ModelLabPredictRequest
            .model_validate(
                request
            )
        )

    except Exception as error:
        raise ModelLabPredictionInputError(
            (
                "Model Lab prediction request "
                "is invalid."
            )
        ) from error

    try:
        loaded_model = (
            load_trusted_ml_model(
                workflow_id=
                    config.workflow_id,

                model_id=
                    config.model_id,
            )
        )

    except MLModelLoaderError as error:
        raise ModelLabArtifactError(
            (
                "Model Lab could not restore "
                "the trusted predictor."
            )
        ) from error

    artifact = (
        loaded_model.artifact
    )

    if (
        artifact.workflow_id
        !=
        config.workflow_id
        or
        artifact.model_id
        !=
        config.model_id
    ):
        raise ModelLabArtifactError(
            (
                "Trusted Model Artifact identity "
                "does not match the prediction request."
            )
        )

    contract = (
        artifact.training_contract
    )

    dataframe = (
        _prediction_dataframe(
            rows=
                config.rows,

            feature_columns=
                list(
                    contract.feature_columns
                ),
        )
    )

    try:
        predictions = (
            loaded_model.predict(
                dataframe
            )
        )

    except Exception as error:
        raise ModelLabPredictionExecutionError(
            (
                "Trusted model native predict() "
                "execution failed."
            )
        ) from error

    normalized_predictions = (
        _prediction_list(
            predictions=
                predictions,

            expected_rows=
                len(
                    config.rows
                ),
        )
    )

    try:
        return (
            ModelLabPredictResponse(
                workflow_id=
                    artifact.workflow_id,

                model_id=
                    artifact.model_id,

                problem_type=
                    contract.problem_type,

                target_column=
                    contract.target_column,

                prediction_count=
                    len(
                        normalized_predictions
                    ),

                predictions=
                    normalized_predictions,
            )
        )

    except Exception as error:
        raise ModelLabPredictionExecutionError(
            (
                "Trusted model produced an invalid "
                "Model Lab prediction response."
            )
        ) from error
