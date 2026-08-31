from __future__ import annotations


from fastapi import (
    APIRouter,
    HTTPException,
    Query,
    status,
)


from app.api.ml_performance_monitoring_contracts import (
    MLPerformanceMonitoringAPIErrorDetail,
    MLPerformanceMonitoringModelHistoryResponse,
    MLPerformanceMonitoringRunRequest,
    MLPerformanceMonitoringWorkflowHistoryResponse,
)


from app.ml.performance_evaluation import (
    MLPerformanceEvaluationRecord,
)


from app.ml.performance_monitoring_service import (
    MLPerformanceMonitoringObservedDatasetError,
    MLPerformanceMonitoringServiceAuthorityError,
    MLPerformanceMonitoringServiceError,
    MLPerformanceMonitoringServiceExecutionError,
    MLPerformanceMonitoringServiceInputError,
    MLPerformanceMonitoringTargetError,
    run_ml_performance_monitoring,
)



from app.ml.performance_monitoring_history_service import (
    MLPerformanceMonitoringHistoryInputError,
    MLPerformanceMonitoringHistoryNotFoundError,
    MLPerformanceMonitoringHistoryServiceError,
    MLPerformanceMonitoringHistoryStorageError,
    get_ml_performance_monitoring_evaluation,
    list_ml_performance_monitoring_model_history,
    list_ml_performance_monitoring_workflow_history,
)


# ============================================================
# VERSION
# ============================================================


ML_PERFORMANCE_MONITORING_API_VERSION = (
    "ml_performance_monitoring_api_v0.1"
)



ML_PERFORMANCE_MONITORING_HISTORY_API_VERSION = (
    "ml_performance_monitoring_history_api_v0.1"
)


# ============================================================
# ROUTER
# ============================================================


router = APIRouter(
    prefix=
        "/ml-monitoring/performance",

    tags=[
        "ml-monitoring",
        "ml-performance-monitoring",
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
        MLPerformanceMonitoringAPIErrorDetail(
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
    # PUBLIC REQUEST INPUT
    # --------------------------------------------------------


    if isinstance(
        error,
        MLPerformanceMonitoringServiceInputError,
    ):
        raise HTTPException(
            status_code=
                status.HTTP_422_UNPROCESSABLE_CONTENT,

            detail=
                _error_detail(
                    error=
                        "performance_monitoring_input_invalid",

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
    # OBSERVED DATASET
    # --------------------------------------------------------


    if isinstance(
        error,
        MLPerformanceMonitoringObservedDatasetError,
    ):
        raise HTTPException(
            status_code=
                status.HTTP_422_UNPROCESSABLE_CONTENT,

            detail=
                _error_detail(
                    error=
                        "performance_observed_dataset_invalid",

                    message=(
                        "The observed dataset is not "
                        "compatible with the requested "
                        "Performance Monitoring operation."
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
    # SUPERVISED GROUND TRUTH
    # --------------------------------------------------------


    if isinstance(
        error,
        MLPerformanceMonitoringTargetError,
    ):
        raise HTTPException(
            status_code=
                status.HTTP_422_UNPROCESSABLE_CONTENT,

            detail=
                _error_detail(
                    error=
                        "performance_ground_truth_invalid",

                    message=(
                        "The observed dataset does not "
                        "provide valid complete ground "
                        "truth for Performance Monitoring."
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
    # Deliberately non-enumerating.
    #
    # Do not expose:
    # - whether a model exists in another workflow;
    # - which internal authority check failed;
    # - artifact paths or persistence details.
    # --------------------------------------------------------


    if isinstance(
        error,
        MLPerformanceMonitoringServiceAuthorityError,
    ):
        raise HTTPException(
            status_code=
                status.HTTP_409_CONFLICT,

            detail=
                _error_detail(
                    error=(
                        "performance_monitoring_"
                        "authority_unavailable"
                    ),

                    message=(
                        "The requested Performance "
                        "Monitoring operation is not "
                        "authorized by the current "
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
    # PREDICTION / METRIC COMPUTATION / DURABLE COMMIT
    # --------------------------------------------------------


    if isinstance(
        error,
        MLPerformanceMonitoringServiceExecutionError,
    ):
        raise HTTPException(
            status_code=
                status.HTTP_409_CONFLICT,

            detail=
                _error_detail(
                    error=(
                        "performance_monitoring_"
                        "execution_failed"
                    ),

                    message=(
                        "The Performance Monitoring "
                        "evaluation could not be "
                        "completed against the current "
                        "server-owned evidence."
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
        MLPerformanceMonitoringServiceError,
    ):
        raise HTTPException(
            status_code=
                status.HTTP_400_BAD_REQUEST,

            detail=
                _error_detail(
                    error=(
                        "invalid_performance_"
                        "monitoring_request"
                    ),

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
        MLPerformanceEvaluationRecord,
)
def evaluate_performance_monitoring(
    request: MLPerformanceMonitoringRunRequest,
) -> MLPerformanceEvaluationRecord:

    try:
        return (
            run_ml_performance_monitoring(
                workflow_id=
                    request.workflow_id,

                model_id=
                    request.model_id,

                observed_dataset_id=
                    request.observed_dataset_id,
            )
        )


    except MLPerformanceMonitoringServiceError as error:

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
        MLPerformanceMonitoringHistoryInputError,
    ):
        raise HTTPException(
            status_code=
                status.HTTP_422_UNPROCESSABLE_CONTENT,

            detail=
                _error_detail(
                    error=(
                        "performance_monitoring_"
                        "history_input_invalid"
                    ),

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
        MLPerformanceMonitoringHistoryNotFoundError,
    ):
        raise HTTPException(
            status_code=
                status.HTTP_404_NOT_FOUND,

            detail=
                _error_detail(
                    error=(
                        "performance_monitoring_"
                        "history_not_found"
                    ),

                    message=(
                        "The requested Performance "
                        "Monitoring history resource "
                        "was not found."
                    ),

                    workflow_id=
                        workflow_id,

                    model_id=
                        model_id,
                ),
        ) from error


    # --------------------------------------------------------
    # PERSISTED HISTORY / AUTHORITY STORAGE FAILURE
    # --------------------------------------------------------


    if isinstance(
        error,
        MLPerformanceMonitoringHistoryStorageError,
    ):
        raise HTTPException(
            status_code=
                status.HTTP_500_INTERNAL_SERVER_ERROR,

            detail=
                _error_detail(
                    error=(
                        "performance_monitoring_"
                        "history_unavailable"
                    ),

                    message=(
                        "Persisted Performance "
                        "Monitoring history is "
                        "unavailable or invalid."
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
    # GENERIC HISTORY SERVICE ERROR
    # --------------------------------------------------------


    if isinstance(
        error,
        MLPerformanceMonitoringHistoryServiceError,
    ):
        raise HTTPException(
            status_code=
                status.HTTP_400_BAD_REQUEST,

            detail=
                _error_detail(
                    error=(
                        "invalid_performance_"
                        "monitoring_history_request"
                    ),

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
    "/evaluations/{performance_evaluation_id}",

    response_model=
        MLPerformanceEvaluationRecord,
)
def get_performance_monitoring_evaluation(
    performance_evaluation_id: str,

    workflow_id: str = Query(
        ...,
        min_length=1,
    ),
) -> MLPerformanceEvaluationRecord:

    try:
        return (
            get_ml_performance_monitoring_evaluation(
                workflow_id=
                    workflow_id,

                performance_evaluation_id=
                    performance_evaluation_id,
            )
        )


    except MLPerformanceMonitoringHistoryServiceError as error:

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
        MLPerformanceMonitoringModelHistoryResponse,
)
def get_performance_monitoring_model_history(
    model_id: str,

    workflow_id: str = Query(
        ...,
        min_length=1,
    ),
) -> MLPerformanceMonitoringModelHistoryResponse:

    try:
        evaluations = (
            list_ml_performance_monitoring_model_history(
                workflow_id=
                    workflow_id,

                model_id=
                    model_id,
            )
        )


        return (
            MLPerformanceMonitoringModelHistoryResponse(
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


    except MLPerformanceMonitoringHistoryServiceError as error:

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
        MLPerformanceMonitoringWorkflowHistoryResponse,
)
def get_performance_monitoring_workflow_history(
    workflow_id: str,
) -> MLPerformanceMonitoringWorkflowHistoryResponse:

    try:
        evaluations = (
            list_ml_performance_monitoring_workflow_history(
                workflow_id=
                    workflow_id
            )
        )


        return (
            MLPerformanceMonitoringWorkflowHistoryResponse(
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


    except MLPerformanceMonitoringHistoryServiceError as error:

        _raise_history_service_error(
            error=
                error,

            workflow_id=
                workflow_id,
        )


        raise AssertionError(
            "unreachable"
        )
