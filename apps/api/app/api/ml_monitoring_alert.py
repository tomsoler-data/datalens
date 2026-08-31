from __future__ import annotations


from fastapi import (
    APIRouter,
    HTTPException,
    Query,
    status,
)


from app.api.ml_monitoring_alert_contracts import (
    MLMonitoringAlertAPIErrorDetail,
)


from app.ml.monitoring_alert import (
    MLMonitoringAlertDecision,
)


from app.ml.monitoring_alert_service import (
    MLMonitoringAlertServiceAuthorityError,
    MLMonitoringAlertServiceError,
    MLMonitoringAlertServiceInputError,
    MLMonitoringAlertServiceNotFoundError,
    MLMonitoringAlertServiceStorageError,
    get_ml_monitoring_alert_decision,
)


# ============================================================
# VERSION
# ============================================================


ML_MONITORING_ALERT_API_VERSION = (
    "ml_monitoring_alert_api_v0.1"
)


# ============================================================
# ROUTER
# ============================================================


router = APIRouter(
    prefix=
        "/ml-monitoring",

    tags=[
        "ml-monitoring",
        "ml-monitoring-alert",
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
        MLMonitoringAlertAPIErrorDetail(
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
        MLMonitoringAlertServiceInputError,
    ):
        raise HTTPException(
            status_code=
                status.HTTP_422_UNPROCESSABLE_CONTENT,

            detail=
                _error_detail(
                    error=
                        "monitoring_alert_input_invalid",

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
    # --------------------------------------------------------


    if isinstance(
        error,
        MLMonitoringAlertServiceNotFoundError,
    ):
        raise HTTPException(
            status_code=
                status.HTTP_404_NOT_FOUND,

            detail=
                _error_detail(
                    error=
                        "monitoring_alert_not_found",

                    message=(
                        "The requested Monitoring "
                        "Alert resource was not found."
                    ),

                    workflow_id=
                        workflow_id,

                    model_id=
                        model_id,
                ),
        ) from error


    # --------------------------------------------------------
    # SERVER-OWNED EVIDENCE CONFLICT
    # --------------------------------------------------------


    if isinstance(
        error,
        MLMonitoringAlertServiceAuthorityError,
    ):
        raise HTTPException(
            status_code=
                status.HTTP_409_CONFLICT,

            detail=
                _error_detail(
                    error=(
                        "monitoring_alert_"
                        "evidence_conflict"
                    ),

                    message=(
                        "Server-owned monitoring "
                        "evidence cannot be resolved "
                        "safely into an Alert Decision."
                    ),

                    workflow_id=
                        workflow_id,

                    model_id=
                        model_id,
                ),
        ) from error


    # --------------------------------------------------------
    # STORAGE / HISTORICAL EVIDENCE FAILURE
    # --------------------------------------------------------


    if isinstance(
        error,
        MLMonitoringAlertServiceStorageError,
    ):
        raise HTTPException(
            status_code=
                status.HTTP_500_INTERNAL_SERVER_ERROR,

            detail=
                _error_detail(
                    error=
                        "monitoring_alert_unavailable",

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
    # GENERIC SERVICE FAILURE
    # --------------------------------------------------------


    if isinstance(
        error,
        MLMonitoringAlertServiceError,
    ):
        raise HTTPException(
            status_code=
                status.HTTP_400_BAD_REQUEST,

            detail=
                _error_detail(
                    error=(
                        "invalid_monitoring_"
                        "alert_request"
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
# ALERT DECISION
# ============================================================


@router.get(
    "/models/{model_id}/alert",

    response_model=
        MLMonitoringAlertDecision,
)
def get_monitoring_alert(
    model_id: str,

    workflow_id: str = Query(
        ...,
        min_length=1,
    ),
) -> MLMonitoringAlertDecision:
    """
    Return the current derived Monitoring Alert Decision.

    This endpoint does not:
    - execute Drift Monitoring;
    - execute Performance Monitoring;
    - load a model;
    - call predict();
    - require Preparation READY;
    - persist an alert;
    - send a notification.

    notification_recommended is policy evidence only.
    """

    try:
        return (
            get_ml_monitoring_alert_decision(
                workflow_id=
                    workflow_id,

                model_id=
                    model_id,
            )
        )

    except MLMonitoringAlertServiceError as error:

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
