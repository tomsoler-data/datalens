from __future__ import annotations


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


from app.ml.monitoring_alert import (
    MLMonitoringAlertDecision,
    MLMonitoringAlertError,
    MLMonitoringAlertInputError,
    build_ml_monitoring_alert_decision,
)


# ============================================================
# VERSION
# ============================================================


ML_MONITORING_ALERT_SERVICE_RULE_VERSION = (
    "ml_monitoring_alert_service_v0.1"
)


# ============================================================
# ERRORS
# ============================================================


class MLMonitoringAlertServiceError(
    RuntimeError
):
    pass


class MLMonitoringAlertServiceInputError(
    MLMonitoringAlertServiceError
):
    pass


class MLMonitoringAlertServiceNotFoundError(
    MLMonitoringAlertServiceError
):
    """
    Public non-enumerating identity failure.

    Missing workflows, missing models and cross-workflow model
    access deliberately use the same public error.
    """

    pass


class MLMonitoringAlertServiceStorageError(
    MLMonitoringAlertServiceError
):
    pass


class MLMonitoringAlertServiceAuthorityError(
    MLMonitoringAlertServiceError
):
    """
    Server-owned Model Health evidence exists but cannot be
    trusted or transformed safely into an Alert Decision.
    """

    pass


# ============================================================
# TEXT
# ============================================================


def _required_text(
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
        raise MLMonitoringAlertServiceInputError(
            (
                f"{field_name} "
                "cannot be empty."
            )
        )


    return normalized


# ============================================================
# MODEL HEALTH
# ============================================================


def _read_model_health(
    *,
    workflow_id: str,
    model_id: str,
) -> MLModelHealthSummary:

    try:
        health = (
            get_ml_model_health_summary(
                workflow_id=
                    workflow_id,

                model_id=
                    model_id,
            )
        )

    except MLModelHealthServiceInputError as error:
        raise MLMonitoringAlertServiceInputError(
            (
                "Monitoring Alert Model Health "
                "identity is invalid."
            )
        ) from error

    except MLModelHealthServiceNotFoundError as error:
        raise MLMonitoringAlertServiceNotFoundError(
            (
                "Monitoring Alert resource "
                "was not found."
            )
        ) from error

    except MLModelHealthServiceStorageError as error:
        raise MLMonitoringAlertServiceStorageError(
            (
                "Model Health evidence "
                "could not be read."
            )
        ) from error

    except MLModelHealthServiceAuthorityError as error:
        raise MLMonitoringAlertServiceAuthorityError(
            (
                "Model Health evidence "
                "cannot be trusted."
            )
        ) from error

    except MLModelHealthServiceError as error:
        raise MLMonitoringAlertServiceAuthorityError(
            (
                "Model Health evidence "
                "could not be resolved safely."
            )
        ) from error


    if not isinstance(
        health,
        MLModelHealthSummary,
    ):
        raise MLMonitoringAlertServiceAuthorityError(
            (
                "Model Health Service returned "
                "an invalid summary surface."
            )
        )


    # ========================================================
    # DEFENSIVE IDENTITY REVALIDATION
    # ========================================================


    if (
        health.workflow_id
        !=
        workflow_id
    ):
        raise MLMonitoringAlertServiceAuthorityError(
            (
                "Model Health Summary returned "
                "an inconsistent workflow identity."
            )
        )


    if (
        health.model_id
        !=
        model_id
    ):
        raise MLMonitoringAlertServiceAuthorityError(
            (
                "Model Health Summary returned "
                "an inconsistent Model Artifact identity."
            )
        )


    return health


# ============================================================
# PUBLIC SERVICE
# ============================================================


def get_ml_monitoring_alert_decision(
    *,
    workflow_id: str,
    model_id: str,
) -> MLMonitoringAlertDecision:
    """
    Derive the current operational Monitoring Alert Decision
    from server-owned Model Health evidence.

    Public authority surface:
        workflow_id + model_id only.

    This service performs no:
    - Preparation readiness evaluation;
    - Analysis Input Handoff;
    - model loading;
    - prediction;
    - Drift computation;
    - Performance metric computation;
    - Alert persistence;
    - notification delivery.

    notification_recommended is policy evidence only.
    """

    normalized_workflow_id = (
        _required_text(
            workflow_id,
            field_name=
                "workflow_id",
        )
    )


    normalized_model_id = (
        _required_text(
            model_id,
            field_name=
                "model_id",
        )
    )


    health = (
        _read_model_health(
            workflow_id=
                normalized_workflow_id,

            model_id=
                normalized_model_id,
        )
    )


    try:
        return (
            build_ml_monitoring_alert_decision(
                model_health=
                    health
            )
        )

    except (
        MLMonitoringAlertInputError,
        MLMonitoringAlertError,
    ) as error:
        raise MLMonitoringAlertServiceAuthorityError(
            (
                "Model Health evidence cannot "
                "be transformed safely into a "
                "Monitoring Alert Decision."
            )
        ) from error
