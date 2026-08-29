from __future__ import annotations


from fastapi import (
    APIRouter,
    HTTPException,
    Query,
    status,
)


from app.api.model_lab_contracts import (
    ModelLabModelDetail,
)


from app.api.model_training_contracts import (
    ModelTrainingAPIErrorDetail,
    ModelTrainingContextResponse,
    ModelTrainingRequest,
)


from app.api.model_training_service import (
    ModelTrainingContextError,
    ModelTrainingEstimatorError,
    ModelTrainingExecutionError,
    ModelTrainingInputError,
    ModelTrainingServiceError,
    get_model_training_context,
    train_model,
)


# ============================================================
# VERSION
# ============================================================


MODEL_TRAINING_API_VERSION = (
    "model_training_api_v0.1"
)


# ============================================================
# ROUTER
# ============================================================


router = APIRouter(
    prefix="/model-training",
    tags=[
        "model-training",
    ],
)


# ============================================================
# ERROR
# ============================================================


def _error_detail(
    *,
    error: str,
    message: str,
    workflow_id: str | None = None,
    retryable: bool = False,
) -> dict:

    return (
        ModelTrainingAPIErrorDetail(
            error=
                error,

            message=
                message,

            workflow_id=
                workflow_id,

            retryable=
                retryable,
        )
        .model_dump(
            mode="json"
        )
    )


def _raise_service_error(
    *,
    error: Exception,
    workflow_id: str | None = None,
) -> None:

    if isinstance(
        error,
        ModelTrainingContextError,
    ):
        raise HTTPException(
            status_code=
                status.HTTP_409_CONFLICT,

            detail=
                _error_detail(
                    error=
                        "training_context_unavailable",

                    message=
                        str(
                            error
                        ),

                    workflow_id=
                        workflow_id,
                ),
        ) from error

    if isinstance(
        error,
        ModelTrainingInputError,
    ):
        raise HTTPException(
            status_code=
                status.HTTP_422_UNPROCESSABLE_CONTENT,

            detail=
                _error_detail(
                    error=
                        "training_input_invalid",

                    message=
                        str(
                            error
                        ),

                    workflow_id=
                        workflow_id,
                ),
        ) from error

    if isinstance(
        error,
        ModelTrainingEstimatorError,
    ):
        raise HTTPException(
            status_code=
                status.HTTP_422_UNPROCESSABLE_CONTENT,

            detail=
                _error_detail(
                    error=
                        "training_estimator_invalid",

                    message=
                        str(
                            error
                        ),

                    workflow_id=
                        workflow_id,
                ),
        ) from error

    if isinstance(
        error,
        ModelTrainingExecutionError,
    ):
        raise HTTPException(
            status_code=
                status.HTTP_409_CONFLICT,

            detail=
                _error_detail(
                    error=
                        "training_execution_failed",

                    message=
                        str(
                            error
                        ),

                    workflow_id=
                        workflow_id,
                ),
        ) from error

    if isinstance(
        error,
        ModelTrainingServiceError,
    ):
        raise HTTPException(
            status_code=
                status.HTTP_400_BAD_REQUEST,

            detail=
                _error_detail(
                    error=
                        "invalid_model_training_request",

                    message=
                        str(
                            error
                        ),

                    workflow_id=
                        workflow_id,
                ),
        ) from error

    raise error


# ============================================================
# CONTEXT
# ============================================================


@router.get(
    "/context",
    response_model=
        ModelTrainingContextResponse,
)
def training_context(
    workflow_id: str = Query(
        ...,
        min_length=1,
    ),
) -> ModelTrainingContextResponse:

    try:
        return (
            get_model_training_context(
                workflow_id=
                    workflow_id
            )
        )

    except ModelTrainingServiceError as error:
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
# TRAIN
# ============================================================


@router.post(
    "/train",
    response_model=
        ModelLabModelDetail,
)
def train(
    request: ModelTrainingRequest,
) -> ModelLabModelDetail:

    workflow_id = (
        request
        .training
        .workflow_id
    )

    try:
        return (
            train_model(
                request
            )
        )

    except ModelTrainingServiceError as error:
        _raise_service_error(
            error=
                error,

            workflow_id=
                workflow_id,
        )

        raise AssertionError(
            "unreachable"
        )
