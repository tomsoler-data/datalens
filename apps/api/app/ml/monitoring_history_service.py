from __future__ import annotations


from app.ml.drift_evaluation import (
    MLDriftEvaluationRecord,
)


from app.ml.drift_evaluation_store import (
    MLDriftEvaluationNotFoundError,
    MLDriftEvaluationStoreError,
    MLDriftEvaluationWorkflowMismatchError,
    get_ml_drift_evaluation,
    list_ml_drift_evaluations_for_model,
    list_ml_drift_evaluations_for_workflow,
)


from app.ml.model_artifact_store import (
    MLModelArtifactNotFoundError,
    MLModelArtifactStoreError,
    MLModelArtifactWorkflowMismatchError,
    get_ml_model_artifact,
)


from app.preparation.preparation_session import (
    PreparationSessionNotFoundError,
    PreparationSessionStoreError,
    get_preparation_session,
)


# ============================================================
# VERSION
# ============================================================


ML_MONITORING_HISTORY_SERVICE_RULE_VERSION = (
    "ml_monitoring_history_service_v0.1"
)


# ============================================================
# ERRORS
# ============================================================


class MLMonitoringHistoryServiceError(
    RuntimeError
):
    pass


class MLMonitoringHistoryInputError(
    MLMonitoringHistoryServiceError
):
    pass


class MLMonitoringHistoryNotFoundError(
    MLMonitoringHistoryServiceError
):
    """
    Public non-enumerating identity failure.

    Missing resources and cross-workflow resources deliberately
    use the same service error.
    """

    pass


class MLMonitoringHistoryStorageError(
    MLMonitoringHistoryServiceError
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
        raise MLMonitoringHistoryInputError(
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
    Verify server-side workflow ownership without requiring the
    workflow to remain READY FOR ANALYSIS.

    Historical evidence must remain readable after Preparation
    has advanced to a newer state.
    """

    try:
        session = (
            get_preparation_session(
                workflow_id
            )
        )

    except PreparationSessionNotFoundError as error:
        raise MLMonitoringHistoryNotFoundError(
            (
                "Monitoring history resource "
                "was not found."
            )
        ) from error

    except PreparationSessionStoreError as error:
        raise MLMonitoringHistoryStorageError(
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
        raise MLMonitoringHistoryStorageError(
            (
                "Preparation workflow authority "
                "returned an inconsistent identity."
            )
        )


# ============================================================
# EVALUATION DETAIL
# ============================================================


def get_ml_monitoring_evaluation(
    *,
    workflow_id: str,
    evaluation_id: str,
) -> MLDriftEvaluationRecord:
    """
    Read one persisted Drift Evaluation.

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
            evaluation_id,
            field_name=
                "evaluation_id",
        )
    )


    _require_server_owned_workflow(
        workflow_id=
            normalized_workflow_id
    )


    try:
        record = (
            get_ml_drift_evaluation(
                evaluation_id=
                    normalized_evaluation_id,

                workflow_id=
                    normalized_workflow_id,
            )
        )

    except (
        MLDriftEvaluationNotFoundError,
        MLDriftEvaluationWorkflowMismatchError,
    ) as error:
        raise MLMonitoringHistoryNotFoundError(
            (
                "Monitoring history resource "
                "was not found."
            )
        ) from error

    except MLDriftEvaluationStoreError as error:
        raise MLMonitoringHistoryStorageError(
            (
                "Monitoring history could "
                "not be read."
            )
        ) from error


    if (
        record.workflow_id
        !=
        normalized_workflow_id
    ):
        raise MLMonitoringHistoryStorageError(
            (
                "Persisted Monitoring history "
                "returned an inconsistent "
                "workflow identity."
            )
        )


    return record


# ============================================================
# MODEL HISTORY
# ============================================================


def list_ml_monitoring_model_history(
    *,
    workflow_id: str,
    model_id: str,
) -> list[
    MLDriftEvaluationRecord
]:
    """
    Read persisted Drift history for one trusted Model Artifact.

    A model from another workflow is not enumerable.
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
        raise MLMonitoringHistoryNotFoundError(
            (
                "Monitoring history resource "
                "was not found."
            )
        ) from error

    except MLModelArtifactStoreError as error:
        raise MLMonitoringHistoryStorageError(
            (
                "Model Artifact authority "
                "could not be read."
            )
        ) from error


    if (
        artifact.workflow_id
        !=
        normalized_workflow_id
    ):
        raise MLMonitoringHistoryStorageError(
            (
                "Model Artifact authority "
                "returned an inconsistent "
                "workflow identity."
            )
        )


    try:
        history = (
            list_ml_drift_evaluations_for_model(
                model_id=
                    normalized_model_id,

                workflow_id=
                    normalized_workflow_id,
            )
        )

    except MLDriftEvaluationStoreError as error:
        raise MLMonitoringHistoryStorageError(
            (
                "Monitoring model history "
                "could not be read."
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
            raise MLMonitoringHistoryStorageError(
                (
                    "Persisted Monitoring model "
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


def list_ml_monitoring_workflow_history(
    *,
    workflow_id: str,
) -> list[
    MLDriftEvaluationRecord
]:
    """
    Read all persisted Drift Evaluation evidence belonging to
    one existing server-owned Preparation workflow.

    Current Preparation readiness is deliberately irrelevant.
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
            list_ml_drift_evaluations_for_workflow(
                workflow_id=
                    normalized_workflow_id
            )
        )

    except MLDriftEvaluationStoreError as error:
        raise MLMonitoringHistoryStorageError(
            (
                "Monitoring workflow history "
                "could not be read."
            )
        ) from error


    for record in history:

        if (
            record.workflow_id
            !=
            normalized_workflow_id
        ):
            raise MLMonitoringHistoryStorageError(
                (
                    "Persisted Monitoring workflow "
                    "history contains an "
                    "inconsistent identity."
                )
            )


    return list(
        history
    )
