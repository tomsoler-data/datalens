from __future__ import annotations


from fastapi import (
    APIRouter,
    HTTPException,
    Query,
    status,
)


from app.api.ml_model_health_contracts import (
    MLModelHealthAPIErrorDetail,
)


from app.ml.model_health import (
    MLModelHealthSummary,
)


from app.ml.model_health_service import (
    MLModelHealthServiceAuthorityError,
    MLModelHealthServiceError,
    MLModelHealthServiceInputError,
    MLModelHealthServiceNotFoundError,
    MLModelHealthServiceStorageError,
    get_ml_model_health_summary,
)


# ============================================================
# VERSION
# ============================================================


ML_MODEL_HEALTH_API_VERSION = (
    "ml_model_health_api_v0.1"
)


# ============================================================
# ROUTER
# ============================================================


router = APIRouter(
    prefix=
        "/ml-monitoring",

    tags=[
        "ml-monitoring",
        "ml-model-health",
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
        MLModelHealthAPIErrorDetail(
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
    # PUBLIC INPUT
    # --------------------------------------------------------


    if isinstance(
        error,
        MLModelHealthServiceInputError,
    ):
        raise HTTPException(
            status_code=
                status.HTTP_422_UNPROCESSABLE_CONTENT,

            detail=
                _error_detail(
                    error=
                        "model_health_input_invalid",

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
    # NON-ENUMERATING RESOURCE FAILURE
    #
    # Missing workflow/model and cross-workflow model access
    # deliberately look identical.
    # --------------------------------------------------------


    if isinstance(
        error,
        MLModelHealthServiceNotFoundError,
    ):
        raise HTTPException(
            status_code=
                status.HTTP_404_NOT_FOUND,

            detail=
                _error_detail(
                    error=
                        "model_health_not_found",

                    message=(
                        "The requested Model Health "
                        "resource was not found."
                    ),

                    workflow_id=
                        workflow_id,

                    model_id=
                        model_id,
                ),
        ) from error


    # --------------------------------------------------------
    # PERSISTED EVIDENCE CONFLICT
    #
    # The resource exists, but persisted monitoring evidence
    # cannot safely be combined.
    # --------------------------------------------------------


    if isinstance(
        error,
        MLModelHealthServiceAuthorityError,
    ):
        raise HTTPException(
            status_code=
                status.HTTP_409_CONFLICT,

            detail=
                _error_detail(
                    error=
                        "model_health_evidence_conflict",

                    message=(
                        "Persisted monitoring evidence "
                        "cannot be combined safely into "
                        "a Model Health Summary."
                    ),

                    workflow_id=
                        workflow_id,

                    model_id=
                        model_id,
                ),
        ) from error


    # --------------------------------------------------------
    # STORAGE / HISTORY FAILURE
    # --------------------------------------------------------


    if isinstance(
        error,
        MLModelHealthServiceStorageError,
    ):
        raise HTTPException(
            status_code=
                status.HTTP_500_INTERNAL_SERVER_ERROR,

            detail=
                _error_detail(
                    error=
                        "model_health_unavailable",

                    message=(
                        "Persisted monitoring evidence "
                        "is currently unavailable or "
                        "invalid."
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
    # GENERIC MODEL HEALTH FAILURE
    # --------------------------------------------------------


    if isinstance(
        error,
        MLModelHealthServiceError,
    ):
        raise HTTPException(
            status_code=
                status.HTTP_400_BAD_REQUEST,

            detail=
                _error_detail(
                    error=
                        "invalid_model_health_request",

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
# MODEL HEALTH
# ============================================================


@router.get(
    "/models/{model_id}/health",

    response_model=
        MLModelHealthSummary,
)
def get_model_health(
    model_id: str,

    workflow_id: str = Query(
        ...,
        min_length=1,
    ),
) -> MLModelHealthSummary:
    """
    Return the current derived Model Health interpretation from
    already-persisted Drift and Performance evidence.

    This endpoint does not:
    - execute monitoring;
    - load a model;
    - call predict();
    - require Preparation READY;
    - persist a Health Summary.
    """

    try:
        return (
            get_ml_model_health_summary(
                workflow_id=
                    workflow_id,

                model_id=
                    model_id,
            )
        )

    except MLModelHealthServiceError as error:

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
