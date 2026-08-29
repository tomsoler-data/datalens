from __future__ import annotations


from app.ml.model_health import (
    MLModelHealthAuthorityError,
    MLModelHealthError,
    MLModelHealthInputError,
    MLModelHealthSummary,
    build_ml_model_health_summary,
)


from app.ml.monitoring_history_service import (
    MLMonitoringHistoryInputError,
    MLMonitoringHistoryNotFoundError,
    MLMonitoringHistoryServiceError,
    MLMonitoringHistoryStorageError,
    list_ml_monitoring_model_history,
)


from app.ml.performance_monitoring_history_service import (
    MLPerformanceMonitoringHistoryInputError,
    MLPerformanceMonitoringHistoryNotFoundError,
    MLPerformanceMonitoringHistoryServiceError,
    MLPerformanceMonitoringHistoryStorageError,
    list_ml_performance_monitoring_model_history,
)


# ============================================================
# VERSION
# ============================================================


ML_MODEL_HEALTH_SERVICE_RULE_VERSION = (
    "ml_model_health_service_v0.1"
)


# ============================================================
# ERRORS
# ============================================================


class MLModelHealthServiceError(
    RuntimeError
):
    pass


class MLModelHealthServiceInputError(
    MLModelHealthServiceError
):
    pass


class MLModelHealthServiceNotFoundError(
    MLModelHealthServiceError
):
    """
    Public non-enumerating identity failure.

    Missing workflows, missing Model Artifacts and Model
    Artifacts belonging to another workflow deliberately map
    to the same service error.
    """

    pass


class MLModelHealthServiceStorageError(
    MLModelHealthServiceError
):
    pass


class MLModelHealthServiceAuthorityError(
    MLModelHealthServiceError
):
    """
    Persisted Drift / Performance evidence exists but cannot be
    combined safely under Model Health v0.1 authority rules.
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
        raise MLModelHealthServiceInputError(
            (
                f"{field_name} "
                "cannot be empty."
            )
        )


    return normalized


# ============================================================
# DRIFT HISTORY
# ============================================================


def _read_drift_history(
    *,
    workflow_id: str,
    model_id: str,
):

    try:
        history = (
            list_ml_monitoring_model_history(
                workflow_id=
                    workflow_id,

                model_id=
                    model_id,
            )
        )

    except MLMonitoringHistoryInputError as error:
        raise MLModelHealthServiceInputError(
            (
                "Model Health Drift history "
                "identity is invalid."
            )
        ) from error

    except MLMonitoringHistoryNotFoundError as error:
        raise MLModelHealthServiceNotFoundError(
            (
                "Model Health resource "
                "was not found."
            )
        ) from error

    except MLMonitoringHistoryStorageError as error:
        raise MLModelHealthServiceStorageError(
            (
                "Persisted Drift history "
                "could not be read."
            )
        ) from error

    except MLMonitoringHistoryServiceError as error:
        raise MLModelHealthServiceStorageError(
            (
                "Persisted Drift history "
                "could not be resolved."
            )
        ) from error


    if not isinstance(
        history,
        list,
    ):
        raise MLModelHealthServiceAuthorityError(
            (
                "Drift History Service returned "
                "an invalid history surface."
            )
        )


    return history


# ============================================================
# PERFORMANCE HISTORY
# ============================================================


def _read_performance_history(
    *,
    workflow_id: str,
    model_id: str,
):

    try:
        history = (
            list_ml_performance_monitoring_model_history(
                workflow_id=
                    workflow_id,

                model_id=
                    model_id,
            )
        )

    except MLPerformanceMonitoringHistoryInputError as error:
        raise MLModelHealthServiceInputError(
            (
                "Model Health Performance history "
                "identity is invalid."
            )
        ) from error

    except MLPerformanceMonitoringHistoryNotFoundError as error:
        raise MLModelHealthServiceNotFoundError(
            (
                "Model Health resource "
                "was not found."
            )
        ) from error

    except MLPerformanceMonitoringHistoryStorageError as error:
        raise MLModelHealthServiceStorageError(
            (
                "Persisted Performance history "
                "could not be read."
            )
        ) from error

    except MLPerformanceMonitoringHistoryServiceError as error:
        raise MLModelHealthServiceStorageError(
            (
                "Persisted Performance history "
                "could not be resolved."
            )
        ) from error


    if not isinstance(
        history,
        list,
    ):
        raise MLModelHealthServiceAuthorityError(
            (
                "Performance History Service returned "
                "an invalid history surface."
            )
        )


    return history


# ============================================================
# PUBLIC SERVICE
# ============================================================


def get_ml_model_health_summary(
    *,
    workflow_id: str,
    model_id: str,
) -> MLModelHealthSummary:
    """
    Build the current operational Model Health Summary from the
    latest already-persisted Drift and Performance evidence.

    Public authority surface:
        workflow_id + model_id only.

    This service performs no:
    - Analysis Readiness check;
    - Analysis Input Handoff;
    - model loading;
    - prediction;
    - Drift computation;
    - Performance metric computation;
    - monitoring persistence.

    The underlying History Services verify server-owned
    workflow / Model Artifact identity without requiring the
    current Preparation workflow to remain READY.
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


    # ========================================================
    # PERSISTED HISTORY ONLY
    # ========================================================


    drift_history = (
        _read_drift_history(
            workflow_id=
                normalized_workflow_id,

            model_id=
                normalized_model_id,
        )
    )


    performance_history = (
        _read_performance_history(
            workflow_id=
                normalized_workflow_id,

            model_id=
                normalized_model_id,
        )
    )


    # History Stores already provide chronological ordering:
    #
    # evaluated_at_utc ASC,
    # evaluation identity ASC.
    #
    # Therefore the last element is the latest persisted
    # evidence in each monitoring branch.


    latest_drift = (
        drift_history[
            -1
        ]

        if drift_history

        else None
    )


    latest_performance = (
        performance_history[
            -1
        ]

        if performance_history

        else None
    )


    # ========================================================
    # PURE DETERMINISTIC DERIVATION
    # ========================================================


    try:
        return (
            build_ml_model_health_summary(
                workflow_id=
                    normalized_workflow_id,

                model_id=
                    normalized_model_id,

                latest_drift=
                    latest_drift,

                latest_performance=
                    latest_performance,
            )
        )

    except (
        MLModelHealthInputError,
        MLModelHealthAuthorityError,
    ) as error:
        raise MLModelHealthServiceAuthorityError(
            (
                "Persisted monitoring evidence "
                "cannot be combined safely into "
                "a Model Health Summary."
            )
        ) from error

    except MLModelHealthError as error:
        raise MLModelHealthServiceAuthorityError(
            (
                "Model Health derivation failed "
                "against persisted monitoring "
                "evidence."
            )
        ) from error
