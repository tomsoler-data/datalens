from __future__ import annotations


from fastapi import (
    APIRouter,
    HTTPException,
    status,
)


from app.api.ml_monitoring_contracts import (
    MLMonitoringAPIErrorDetail,
    MLMonitoringRunRequest,
)


from app.ml.drift_evaluation import (
    MLDriftEvaluationRecord,
)


from app.ml.monitoring_service import (
    MLMonitoringObservedDatasetError,
    MLMonitoringServiceAuthorityError,
    MLMonitoringServiceError,
    MLMonitoringServiceExecutionError,
    MLMonitoringServiceInputError,
    run_ml_monitoring,
)


# ============================================================
# VERSION
# ============================================================


ML_MONITORING_API_VERSION = (
    "ml_monitoring_api_v0.1"
)


# ============================================================
# ROUTER
# ============================================================


router = APIRouter(
    prefix="/ml-monitoring",

    tags=[
        "ml-monitoring",
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
    observed_dataset_id: str | None = None,
    retryable: bool = False,
) -> dict:

    return (
        MLMonitoringAPIErrorDetail(
            error=
                error,

            message=
                message,

            workflow_id=
                workflow_id,

            model_id=
                model_id,

            observed_dataset_id=
                observed_dataset_id,

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
    observed_dataset_id: str | None = None,
) -> None:

    # --------------------------------------------------------
    # PUBLIC INPUT
    # --------------------------------------------------------


    if isinstance(
        error,
        MLMonitoringServiceInputError,
    ):
        raise HTTPException(
            status_code=
                status.HTTP_422_UNPROCESSABLE_CONTENT,

            detail=
                _error_detail(
                    error=
                        "monitoring_input_invalid",

                    message=
                        str(
                            error
                        ),

                    workflow_id=
                        workflow_id,

                    model_id=
                        model_id,

                    observed_dataset_id=
                        observed_dataset_id,
                ),
        ) from error


    # --------------------------------------------------------
    # OBSERVED DATASET REQUEST
    # --------------------------------------------------------


    if isinstance(
        error,
        MLMonitoringObservedDatasetError,
    ):
        raise HTTPException(
            status_code=
                status.HTTP_422_UNPROCESSABLE_CONTENT,

            detail=
                _error_detail(
                    error=
                        "observed_dataset_invalid",

                    message=
                        str(
                            error
                        ),

                    workflow_id=
                        workflow_id,

                    model_id=
                        model_id,

                    observed_dataset_id=
                        observed_dataset_id,
                ),
        ) from error


    # --------------------------------------------------------
    # SERVER-OWNED AUTHORITY
    #
    # Deliberately generic.
    #
    # Do not reveal whether a model exists in another workflow
    # or which internal authority component failed.
    # --------------------------------------------------------


    if isinstance(
        error,
        MLMonitoringServiceAuthorityError,
    ):
        raise HTTPException(
            status_code=
                status.HTTP_409_CONFLICT,

            detail=
                _error_detail(
                    error=
                        "monitoring_authority_unavailable",

                    message=(
                        "The requested monitoring operation "
                        "is not authorized by the current "
                        "server-owned workflow state."
                    ),

                    workflow_id=
                        workflow_id,

                    model_id=
                        model_id,

                    observed_dataset_id=
                        observed_dataset_id,
                ),
        ) from error


    # --------------------------------------------------------
    # DRIFT COMPUTATION / COMMIT
    # --------------------------------------------------------


    if isinstance(
        error,
        MLMonitoringServiceExecutionError,
    ):
        raise HTTPException(
            status_code=
                status.HTTP_409_CONFLICT,

            detail=
                _error_detail(
                    error=
                        "monitoring_execution_failed",

                    message=(
                        "The monitoring evaluation could "
                        "not be completed against the "
                        "current server-owned evidence."
                    ),

                    workflow_id=
                        workflow_id,

                    model_id=
                        model_id,

                    observed_dataset_id=
                        observed_dataset_id,

                    retryable=
                        False,
                ),
        ) from error


    # --------------------------------------------------------
    # GENERIC SERVICE ERROR
    # --------------------------------------------------------


    if isinstance(
        error,
        MLMonitoringServiceError,
    ):
        raise HTTPException(
            status_code=
                status.HTTP_400_BAD_REQUEST,

            detail=
                _error_detail(
                    error=
                        "invalid_monitoring_request",

                    message=
                        str(
                            error
                        ),

                    workflow_id=
                        workflow_id,

                    model_id=
                        model_id,

                    observed_dataset_id=
                        observed_dataset_id,
                ),
        ) from error


    raise error


# ============================================================
# EVALUATE
# ============================================================


@router.post(
    "/evaluate",

    response_model=
        MLDriftEvaluationRecord,
)
def evaluate_monitoring(
    request: MLMonitoringRunRequest,
) -> MLDriftEvaluationRecord:

    try:
        return (
            run_ml_monitoring(
                workflow_id=
                    request.workflow_id,

                model_id=
                    request.model_id,

                observed_dataset_id=
                    request.observed_dataset_id,
            )
        )


    except MLMonitoringServiceError as error:

        _raise_service_error(
            error=
                error,

            workflow_id=
                request.workflow_id,

            model_id=
                request.model_id,

            observed_dataset_id=
                request.observed_dataset_id,
        )


        raise AssertionError(
            "unreachable"
        )
