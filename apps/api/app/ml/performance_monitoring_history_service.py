from __future__ import annotations


from app.ml.model_artifact_store import (
    MLModelArtifactNotFoundError,
    MLModelArtifactStoreError,
    MLModelArtifactWorkflowMismatchError,
    get_ml_model_artifact,
)


from app.ml.performance_evaluation import (
    MLPerformanceEvaluationRecord,
)


from app.ml.performance_evaluation_store import (
    MLPerformanceEvaluationNotFoundError,
    MLPerformanceEvaluationStoreError,
    MLPerformanceEvaluationWorkflowMismatchError,
    get_ml_performance_evaluation,
    list_ml_performance_evaluations_for_model,
    list_ml_performance_evaluations_for_workflow,
)


from app.preparation.preparation_session import (
    PreparationSessionNotFoundError,
    PreparationSessionStoreError,
    get_preparation_session,
)


# ============================================================
# VERSION
# ============================================================


ML_PERFORMANCE_MONITORING_HISTORY_SERVICE_RULE_VERSION = (
    "ml_performance_monitoring_history_service_v0.1"
)


# ============================================================
# ERRORS
# ============================================================


class MLPerformanceMonitoringHistoryServiceError(
    RuntimeError
):
    pass


class MLPerformanceMonitoringHistoryInputError(
    MLPerformanceMonitoringHistoryServiceError
):
    pass


class MLPerformanceMonitoringHistoryNotFoundError(
    MLPerformanceMonitoringHistoryServiceError
):
    """
    Public non-enumerating identity failure.

    Missing resources and resources belonging to another
    workflow deliberately use the same public error.
    """

    pass


class MLPerformanceMonitoringHistoryStorageError(
    MLPerformanceMonitoringHistoryServiceError
):
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
        raise MLPerformanceMonitoringHistoryInputError(
            (
                f"{field_name} "
                "cannot be empty."
            )
        )


    return normalized


# ============================================================
# WORKFLOW AUTHORITY
# ============================================================


def _require_server_owned_workflow(
    *,
    workflow_id: str,
) -> None:
    """
    Verify that the Preparation workflow still exists.

    Historical evidence is intentionally independent from the
    current Preparation readiness state.

    This function MUST NOT call:

    - Analysis Readiness Gate;
    - Analysis Input Handoff;
    - Performance Evaluator;
    - trusted model prediction.

    Historical reads use persisted evidence only.
    """

    try:
        session = (
            get_preparation_session(
                workflow_id
            )
        )

    except PreparationSessionNotFoundError as error:
        raise MLPerformanceMonitoringHistoryNotFoundError(
            (
                "Performance Monitoring history "
                "resource was not found."
            )
        ) from error

    except PreparationSessionStoreError as error:
        raise MLPerformanceMonitoringHistoryStorageError(
            (
                "Preparation workflow authority "
                "could not be read."
            )
        ) from error


    if (
        session.workflow_id
        !=
        workflow_id
    ):
        raise MLPerformanceMonitoringHistoryStorageError(
            (
                "Preparation workflow authority "
                "returned an inconsistent identity."
            )
        )


# ============================================================
# EVALUATION DETAIL
# ============================================================


def get_ml_performance_monitoring_evaluation(
    *,
    workflow_id: str,
    performance_evaluation_id: str,
) -> MLPerformanceEvaluationRecord:
    """
    Read one persisted Performance Evaluation.

    Cross-workflow access intentionally looks identical to a
    missing evaluation.
    """

    normalized_workflow_id = (
        _required_text(
            workflow_id,
            field_name=
                "workflow_id",
        )
    )


    normalized_evaluation_id = (
        _required_text(
            performance_evaluation_id,
            field_name=
                "performance_evaluation_id",
        )
    )


    _require_server_owned_workflow(
        workflow_id=
            normalized_workflow_id
    )


    try:
        record = (
            get_ml_performance_evaluation(
                performance_evaluation_id=
                    normalized_evaluation_id,

                workflow_id=
                    normalized_workflow_id,
            )
        )

    except (
        MLPerformanceEvaluationNotFoundError,
        MLPerformanceEvaluationWorkflowMismatchError,
    ) as error:
        raise MLPerformanceMonitoringHistoryNotFoundError(
            (
                "Performance Monitoring history "
                "resource was not found."
            )
        ) from error

    except MLPerformanceEvaluationStoreError as error:
        raise MLPerformanceMonitoringHistoryStorageError(
            (
                "Performance Monitoring history "
                "could not be read."
            )
        ) from error


    if (
        record.workflow_id
        !=
        normalized_workflow_id
    ):
        raise MLPerformanceMonitoringHistoryStorageError(
            (
                "Persisted Performance history "
                "returned an inconsistent "
                "workflow identity."
            )
        )


    if (
        record.performance_evaluation_id
        !=
        normalized_evaluation_id
    ):
        raise MLPerformanceMonitoringHistoryStorageError(
            (
                "Persisted Performance history "
                "returned an inconsistent "
                "evaluation identity."
            )
        )


    return record


# ============================================================
# MODEL HISTORY
# ============================================================


def list_ml_performance_monitoring_model_history(
    *,
    workflow_id: str,
    model_id: str,
) -> list[
    MLPerformanceEvaluationRecord
]:
    """
    Read persisted Performance history for one server-owned
    Model Artifact.

    Current Preparation readiness is deliberately irrelevant.
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


    _require_server_owned_workflow(
        workflow_id=
            normalized_workflow_id
    )


    # ========================================================
    # MODEL IDENTITY AUTHORITY
    # ========================================================


    try:
        artifact = (
            get_ml_model_artifact(
                model_id=
                    normalized_model_id,

                workflow_id=
                    normalized_workflow_id,
            )
        )

    except (
        MLModelArtifactNotFoundError,
        MLModelArtifactWorkflowMismatchError,
    ) as error:
        raise MLPerformanceMonitoringHistoryNotFoundError(
            (
                "Performance Monitoring history "
                "resource was not found."
            )
        ) from error

    except MLModelArtifactStoreError as error:
        raise MLPerformanceMonitoringHistoryStorageError(
            (
                "Model Artifact authority "
                "could not be read."
            )
        ) from error


    if (
        artifact.workflow_id
        !=
        normalized_workflow_id
        or
        artifact.model_id
        !=
        normalized_model_id
    ):
        raise MLPerformanceMonitoringHistoryStorageError(
            (
                "Model Artifact authority "
                "returned an inconsistent identity."
            )
        )


    # ========================================================
    # PERSISTED HISTORY
    # ========================================================


    try:
        history = (
            list_ml_performance_evaluations_for_model(
                model_id=
                    normalized_model_id,

                workflow_id=
                    normalized_workflow_id,
            )
        )

    except MLPerformanceEvaluationStoreError as error:
        raise MLPerformanceMonitoringHistoryStorageError(
            (
                "Performance Monitoring model "
                "history could not be read."
            )
        ) from error


    for record in history:

        if (
            record.workflow_id
            !=
            normalized_workflow_id
            or
            record.model_id
            !=
            normalized_model_id
        ):
            raise MLPerformanceMonitoringHistoryStorageError(
                (
                    "Persisted Performance model "
                    "history contains an "
                    "inconsistent identity."
                )
            )


    return list(
        history
    )


# ============================================================
# WORKFLOW HISTORY
# ============================================================


def list_ml_performance_monitoring_workflow_history(
    *,
    workflow_id: str,
) -> list[
    MLPerformanceEvaluationRecord
]:
    """
    Read all persisted Performance Evaluation evidence belonging
    to one existing server-owned Preparation workflow.

    This is an historical read only.

    No current READY state is required.
    """

    normalized_workflow_id = (
        _required_text(
            workflow_id,
            field_name=
                "workflow_id",
        )
    )


    _require_server_owned_workflow(
        workflow_id=
            normalized_workflow_id
    )


    try:
        history = (
            list_ml_performance_evaluations_for_workflow(
                workflow_id=
                    normalized_workflow_id
            )
        )

    except MLPerformanceEvaluationStoreError as error:
        raise MLPerformanceMonitoringHistoryStorageError(
            (
                "Performance Monitoring workflow "
                "history could not be read."
            )
        ) from error


    for record in history:

        if (
            record.workflow_id
            !=
            normalized_workflow_id
        ):
            raise MLPerformanceMonitoringHistoryStorageError(
                (
                    "Persisted Performance workflow "
                    "history contains an "
                    "inconsistent identity."
                )
            )


    return list(
        history
    )
