from __future__ import annotations


from fastapi import (
    APIRouter,
    HTTPException,
    Query,
    status,
)


from app.api.ml_monitoring_contracts import (
    MLMonitoringAPIErrorDetail,
    MLMonitoringModelHistoryResponse,
    MLMonitoringRunRequest,
    MLMonitoringWorkflowHistoryResponse,
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



from app.ml.monitoring_history_service import (
    MLMonitoringHistoryInputError,
    MLMonitoringHistoryNotFoundError,
    MLMonitoringHistoryServiceError,
    MLMonitoringHistoryStorageError,
    get_ml_monitoring_evaluation,
    list_ml_monitoring_model_history,
    list_ml_monitoring_workflow_history,
)


# ============================================================
# VERSION
# ============================================================


ML_MONITORING_API_VERSION = (
    "ml_monitoring_api_v0.1"
)



ML_MONITORING_HISTORY_API_VERSION = (
    "ml_monitoring_history_api_v0.1"
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


# ============================================================
# HISTORY ERROR TRANSLATION
# ============================================================


def _raise_history_service_error(
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
        MLMonitoringHistoryInputError,
    ):
        raise HTTPException(
            status_code=
                status.HTTP_422_UNPROCESSABLE_CONTENT,

            detail=
                _error_detail(
                    error=
                        "monitoring_history_input_invalid",

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
    # NON-ENUMERATING NOT FOUND
    # --------------------------------------------------------


    if isinstance(
        error,
        MLMonitoringHistoryNotFoundError,
    ):
        raise HTTPException(
            status_code=
                status.HTTP_404_NOT_FOUND,

            detail=
                _error_detail(
                    error=
                        "monitoring_history_not_found",

                    message=(
                        "The requested monitoring "
                        "history resource was not found."
                    ),

                    workflow_id=
                        workflow_id,

                    model_id=
                        model_id,
                ),
        ) from error


    # --------------------------------------------------------
    # SERVER-OWNED STORAGE / AUTHORITY FAILURE
    # --------------------------------------------------------


    if isinstance(
        error,
        MLMonitoringHistoryStorageError,
    ):
        raise HTTPException(
            status_code=
                status.HTTP_500_INTERNAL_SERVER_ERROR,

            detail=
                _error_detail(
                    error=
                        "monitoring_history_unavailable",

                    message=(
                        "Persisted monitoring history "
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


    if isinstance(
        error,
        MLMonitoringHistoryServiceError,
    ):
        raise HTTPException(
            status_code=
                status.HTTP_400_BAD_REQUEST,

            detail=
                _error_detail(
                    error=
                        "invalid_monitoring_history_request",

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
# HISTORY ? EVALUATION DETAIL
# ============================================================


@router.get(
    "/evaluations/{evaluation_id}",

    response_model=
        MLDriftEvaluationRecord,
)
def get_monitoring_evaluation(
    evaluation_id: str,

    workflow_id: str = Query(
        ...,
        min_length=1,
    ),
) -> MLDriftEvaluationRecord:

    try:
        return (
            get_ml_monitoring_evaluation(
                workflow_id=
                    workflow_id,

                evaluation_id=
                    evaluation_id,
            )
        )


    except MLMonitoringHistoryServiceError as error:

        _raise_history_service_error(
            error=
                error,

            workflow_id=
                workflow_id,
        )


        raise AssertionError(
            "unreachable"
        )


# ============================================================
# HISTORY ? MODEL
# ============================================================


@router.get(
    "/models/{model_id}/history",

    response_model=
        MLMonitoringModelHistoryResponse,
)
def get_monitoring_model_history(
    model_id: str,

    workflow_id: str = Query(
        ...,
        min_length=1,
    ),
) -> MLMonitoringModelHistoryResponse:

    try:
        evaluations = (
            list_ml_monitoring_model_history(
                workflow_id=
                    workflow_id,

                model_id=
                    model_id,
            )
        )


        return (
            MLMonitoringModelHistoryResponse(
                workflow_id=
                    workflow_id.strip(),

                model_id=
                    model_id.strip(),

                evaluation_count=
                    len(
                        evaluations
                    ),

                evaluations=
                    evaluations,
            )
        )


    except MLMonitoringHistoryServiceError as error:

        _raise_history_service_error(
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
# HISTORY ? WORKFLOW
# ============================================================


@router.get(
    "/workflows/{workflow_id}/history",

    response_model=
        MLMonitoringWorkflowHistoryResponse,
)
def get_monitoring_workflow_history(
    workflow_id: str,
) -> MLMonitoringWorkflowHistoryResponse:

    try:
        evaluations = (
            list_ml_monitoring_workflow_history(
                workflow_id=
                    workflow_id
            )
        )


        return (
            MLMonitoringWorkflowHistoryResponse(
                workflow_id=
                    workflow_id.strip(),

                evaluation_count=
                    len(
                        evaluations
                    ),

                evaluations=
                    evaluations,
            )
        )


    except MLMonitoringHistoryServiceError as error:

        _raise_history_service_error(
            error=
                error,

            workflow_id=
                workflow_id,
        )


        raise AssertionError(
            "unreachable"
        )

