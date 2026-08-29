from __future__ import annotations

from fastapi import (
    APIRouter,
    HTTPException,
    Query,
    status,
)

from app.api.model_lab_contracts import (
    ModelLabAPIErrorDetail,
    ModelLabEvaluateRequest,
    ModelLabModelDetail,
    ModelLabModelListResponse,
    ModelLabPredictRequest,
    ModelLabPredictResponse,
)

from app.api.model_lab_service import (
    ModelLabArtifactError,
    ModelLabEvaluationError,
    ModelLabModelNotFoundError,
    ModelLabPredictionExecutionError,
    ModelLabPredictionInputError,
    ModelLabServiceError,
    ModelLabWorkflowMismatchError,
    evaluate_model_lab_model,
    get_model_lab_model_detail,
    list_model_lab_models,
    predict_model_lab,
)

from app.ml.model_evaluation_summary import (
    MLModelEvaluationSummaryResult,
)


# ============================================================
# VERSION
# ============================================================


MODEL_LAB_API_VERSION = (
    "model_lab_api_v0.1"
)


# ============================================================
# ROUTER
# ============================================================


router = APIRouter(
    prefix="/model-lab",
    tags=[
        "model-lab",
    ],
)


# ============================================================
# ERROR DETAIL
# ============================================================


def _error_detail(
    *,
    error: str,
    message: str,
    workflow_id: str | None = None,
    model_id: str | None = None,
    retryable: bool = False,
) -> dict:

    return (
        ModelLabAPIErrorDetail(
            error=
                error,

            message=
                message,

            workflow_id=
                workflow_id,

            model_id=
                model_id,

            retryable=
                retryable,
        )
        .model_dump(
            mode="json"
        )
    )


# ============================================================
# ERROR TRANSLATION
# ============================================================


def _raise_service_error(
    *,
    error: Exception,
    workflow_id: str | None = None,
    model_id: str | None = None,
) -> None:

    # --------------------------------------------------------
    # NOT FOUND / CROSS-WORKFLOW
    #
    # A workflow mismatch intentionally uses the same public
    # 404 shape as a missing model.
    #
    # The API must not reveal that a model_id exists inside
    # another Preparation workflow.
    # --------------------------------------------------------

    if isinstance(
        error,
        (
            ModelLabModelNotFoundError,
            ModelLabWorkflowMismatchError,
        ),
    ):
        raise HTTPException(
            status_code=
                status.HTTP_404_NOT_FOUND,

            detail=
                _error_detail(
                    error=
                        "model_not_found",

                    message=(
                        "The requested Model Lab "
                        "model was not found."
                    ),

                    workflow_id=
                        workflow_id,

                    model_id=
                        model_id,
                ),
        ) from error

    # --------------------------------------------------------
    # PREDICTION INPUT
    # --------------------------------------------------------

    if isinstance(
        error,
        ModelLabPredictionInputError,
    ):
        raise HTTPException(
            status_code=
                status.HTTP_422_UNPROCESSABLE_CONTENT,

            detail=
                _error_detail(
                    error=
                        "prediction_input_invalid",

                    message=
                        str(
                            error
                        ),

                    workflow_id=
                        workflow_id,

                    model_id=
                        model_id,
                ),
        ) from error

    # --------------------------------------------------------
    # PREDICTION EXECUTION
    # --------------------------------------------------------

    if isinstance(
        error,
        ModelLabPredictionExecutionError,
    ):
        raise HTTPException(
            status_code=
                status.HTTP_422_UNPROCESSABLE_CONTENT,

            detail=
                _error_detail(
                    error=
                        "prediction_execution_failed",

                    message=(
                        "The trusted model could not "
                        "produce valid predictions for "
                        "the supplied feature values."
                    ),

                    workflow_id=
                        workflow_id,

                    model_id=
                        model_id,
                ),
        ) from error

    # --------------------------------------------------------
    # EVALUATION
    # --------------------------------------------------------

    if isinstance(
        error,
        ModelLabEvaluationError,
    ):
        raise HTTPException(
            status_code=
                status.HTTP_409_CONFLICT,

            detail=
                _error_detail(
                    error=
                        "model_evaluation_failed",

                    message=(
                        "The trusted model could not be "
                        "evaluated against the current "
                        "server-owned evidence."
                    ),

                    workflow_id=
                        workflow_id,

                    model_id=
                        model_id,
                ),
        ) from error

    # --------------------------------------------------------
    # TRUSTED ARTIFACT / LOADER FAILURE
    # --------------------------------------------------------

    if isinstance(
        error,
        ModelLabArtifactError,
    ):
        raise HTTPException(
            status_code=
                status.HTTP_500_INTERNAL_SERVER_ERROR,

            detail=
                _error_detail(
                    error=
                        "model_artifact_unavailable",

                    message=(
                        "Trusted Model Artifact state "
                        "is unavailable or invalid."
                    ),

                    workflow_id=
                        workflow_id,

                    model_id=
                        model_id,

                    retryable=
                        False,
                ),
        ) from error

    # --------------------------------------------------------
    # GENERIC MODEL LAB SERVICE INPUT
    # --------------------------------------------------------

    if isinstance(
        error,
        ModelLabServiceError,
    ):
        raise HTTPException(
            status_code=
                status.HTTP_400_BAD_REQUEST,

            detail=
                _error_detail(
                    error=
                        "invalid_model_lab_request",

                    message=
                        str(
                            error
                        ),

                    workflow_id=
                        workflow_id,

                    model_id=
                        model_id,
                ),
        ) from error

    raise error


# ============================================================
# LIST MODELS
# ============================================================


@router.get(
    "/models",
    response_model=
        ModelLabModelListResponse,
)
def list_models(
    workflow_id: str = Query(
        ...,
        min_length=1,
    ),
) -> ModelLabModelListResponse:

    try:
        return (
            list_model_lab_models(
                workflow_id=
                    workflow_id
            )
        )

    except ModelLabServiceError as error:
        _raise_service_error(
            error=
                error,

            workflow_id=
                workflow_id,
        )

        raise AssertionError(
            "unreachable"
        )


# ============================================================
# MODEL DETAIL
# ============================================================


@router.get(
    "/models/{model_id}",
    response_model=
        ModelLabModelDetail,
)
def get_model(
    model_id: str,
    workflow_id: str = Query(
        ...,
        min_length=1,
    ),
) -> ModelLabModelDetail:

    try:
        return (
            get_model_lab_model_detail(
                workflow_id=
                    workflow_id,

                model_id=
                    model_id,
            )
        )

    except ModelLabServiceError as error:
        _raise_service_error(
            error=
                error,

            workflow_id=
                workflow_id,

            model_id=
                model_id,
        )

        raise AssertionError(
            "unreachable"
        )


# ============================================================
# EVALUATE
# ============================================================


@router.post(
    "/evaluate",
    response_model=
        MLModelEvaluationSummaryResult,
)
def evaluate_model(
    request: ModelLabEvaluateRequest,
) -> MLModelEvaluationSummaryResult:

    try:
        return (
            evaluate_model_lab_model(
                request
            )
        )

    except ModelLabServiceError as error:
        _raise_service_error(
            error=
                error,

            workflow_id=
                request.workflow_id,

            model_id=
                request.model_id,
        )

        raise AssertionError(
            "unreachable"
        )


# ============================================================
# PREDICT
# ============================================================


@router.post(
    "/predict",
    response_model=
        ModelLabPredictResponse,
)
def predict_model(
    request: ModelLabPredictRequest,
) -> ModelLabPredictResponse:

    try:
        return (
            predict_model_lab(
                request
            )
        )

    except ModelLabServiceError as error:
        _raise_service_error(
            error=
                error,

            workflow_id=
                request.workflow_id,

            model_id=
                request.model_id,
        )

        raise AssertionError(
            "unreachable"
        )
