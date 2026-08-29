from __future__ import annotations


from app.ml.model_loader import (
    MLModelLoaderError,
    load_trusted_ml_model,
)


from app.ml.observed_dataset_authority import (
    MLObservedDatasetAuthorityError,
    MLObservedDatasetNotAuthorizedError,
    resolve_server_owned_observed_dataframe,
)


from app.ml.performance_evaluation import (
    MLPerformanceEvaluationRecord,
)


from app.ml.performance_evaluation_store import (
    MLPerformanceEvaluationStoreError,
    register_ml_performance_evaluation,
)


from app.ml.performance_evaluator import (
    MLPerformanceEvaluatorAuthorityError,
    MLPerformanceEvaluatorError,
    MLPerformanceEvaluatorInputError,
    MLPerformanceMetricsError,
    MLPerformancePredictionError,
    MLPerformanceTargetError,
    evaluate_ml_performance,
)


from app.preparation.analysis_input_handoff import (
    AnalysisInputHandoffError,
    load_validated_analysis_input,
)


from app.preparation.analysis_readiness_gate import (
    AnalysisReadinessError,
)


# ============================================================
# VERSION
# ============================================================


ML_PERFORMANCE_MONITORING_SERVICE_RULE_VERSION = (
    "ml_performance_monitoring_service_v0.1"
)


# ============================================================
# ERRORS
# ============================================================


class MLPerformanceMonitoringServiceError(
    RuntimeError
):
    pass


class MLPerformanceMonitoringServiceInputError(
    MLPerformanceMonitoringServiceError
):
    pass


class MLPerformanceMonitoringServiceAuthorityError(
    MLPerformanceMonitoringServiceError
):
    pass


class MLPerformanceMonitoringObservedDatasetError(
    MLPerformanceMonitoringServiceError
):
    pass


class MLPerformanceMonitoringTargetError(
    MLPerformanceMonitoringServiceError
):
    pass


class MLPerformanceMonitoringServiceExecutionError(
    MLPerformanceMonitoringServiceError
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
        raise MLPerformanceMonitoringServiceInputError(
            (
                f"{field_name} "
                "cannot be empty."
            )
        )


    return normalized


# ============================================================
# PUBLIC SERVICE
# ============================================================


def run_ml_performance_monitoring(
    *,
    workflow_id: str,
    model_id: str,
    observed_dataset_id: str,
) -> MLPerformanceEvaluationRecord:
    """
    Execute one server-owned supervised Performance Monitoring
    operation.

    Caller-controlled authority is limited to:

        workflow_id
        model_id
        observed_dataset_id

    The caller cannot provide:

        - raw rows;
        - a DataFrame;
        - true labels separately;
        - predictions;
        - model bytes;
        - filesystem paths;
        - a Model Artifact;
        - a Training Contract;
        - reference metrics;
        - a Preparation revision;
        - an evaluation identity.

    Trust boundary:

        requested identities
            ?
        trusted Model Loader
            ?
        validated Analysis Input Handoff
            ?
        trusted observed DataFrame
            ?
        target from Training Contract
            ?
        trusted model predict()
            ?
        canonical Performance Evaluator
            ?
        transactional Performance Store
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


    normalized_observed_dataset_id = (
        _required_text(
            observed_dataset_id,
            field_name=
                "observed_dataset_id",
        )
    )


    # ========================================================
    # 1. TRUSTED MODEL AUTHORITY
    # ========================================================


    try:
        trusted_model = (
            load_trusted_ml_model(
                workflow_id=
                    normalized_workflow_id,

                model_id=
                    normalized_model_id,
            )
        )

    except (
        MLModelLoaderError,
        ValueError,
    ) as error:
        raise MLPerformanceMonitoringServiceAuthorityError(
            (
                "Requested trusted Model Artifact "
                "is not available in the "
                "requested workflow."
            )
        ) from error


    artifact = (
        trusted_model.artifact
    )


    if (
        artifact.workflow_id
        !=
        normalized_workflow_id
        or
        artifact.model_id
        !=
        normalized_model_id
    ):
        raise MLPerformanceMonitoringServiceAuthorityError(
            (
                "Trusted Model Artifact identity "
                "does not match the requested "
                "Performance Monitoring authority."
            )
        )


    # ========================================================
    # 2. CURRENT VALIDATED OBSERVED SNAPSHOT
    # ========================================================


    try:
        handoff = (
            load_validated_analysis_input(
                workflow_id=
                    normalized_workflow_id
            )
        )

    except (
        AnalysisInputHandoffError,
        AnalysisReadinessError,
    ) as error:
        raise MLPerformanceMonitoringServiceAuthorityError(
            (
                "Current Preparation output "
                "is not authorized for ML "
                "Performance Monitoring."
            )
        ) from error


    try:
        (
            observed_dataframe,
            observed_revision,
        ) = (
            resolve_server_owned_observed_dataframe(
                handoff=
                    handoff,

                workflow_id=
                    normalized_workflow_id,

                observed_dataset_id=
                    normalized_observed_dataset_id,
            )
        )

    except MLObservedDatasetNotAuthorizedError as error:
        raise MLPerformanceMonitoringObservedDatasetError(
            (
                "Requested observed dataset "
                "is not authorized by the "
                "current validated Analysis "
                "Input Handoff."
            )
        ) from error

    except MLObservedDatasetAuthorityError as error:
        raise MLPerformanceMonitoringServiceAuthorityError(
            (
                "Analysis Input Handoff "
                "authority is invalid."
            )
        ) from error


    # ========================================================
    # 3. SUPERVISED PERFORMANCE EVALUATION
    #
    # Target identity comes only from the trusted Training
    # Contract embedded in the loaded Model Artifact.
    # ========================================================


    try:
        evaluation = (
            evaluate_ml_performance(
                observed_dataframe=
                    observed_dataframe,

                observed_dataset_id=
                    normalized_observed_dataset_id,

                observed_preparation_session_revision=
                    observed_revision,

                trusted_model=
                    trusted_model,
            )
        )

    except MLPerformanceTargetError as error:
        raise MLPerformanceMonitoringTargetError(
            (
                "Observed dataset does not contain "
                "valid complete ground truth for "
                "Performance Monitoring."
            )
        ) from error

    except MLPerformanceEvaluatorInputError as error:
        raise MLPerformanceMonitoringObservedDatasetError(
            (
                "Observed dataset is not compatible "
                "with the trusted Model Artifact."
            )
        ) from error

    except MLPerformanceEvaluatorAuthorityError as error:
        raise MLPerformanceMonitoringServiceAuthorityError(
            (
                "Trusted model authority is invalid "
                "for Performance Monitoring."
            )
        ) from error

    except (
        MLPerformancePredictionError,
        MLPerformanceMetricsError,
    ) as error:
        raise MLPerformanceMonitoringServiceExecutionError(
            (
                "ML Performance Evaluation "
                "could not be computed."
            )
        ) from error

    except MLPerformanceEvaluatorError as error:
        raise MLPerformanceMonitoringServiceExecutionError(
            (
                "ML Performance Evaluation "
                "failed."
            )
        ) from error


    # ========================================================
    # 4. DURABLE COMMIT
    #
    # The Performance Store revalidates:
    #
    # - Model Artifact authority;
    # - Experiment Provenance;
    # - Training Contract SHA;
    # - reference holdout metrics;
    # - observed Preparation revision;
    #
    # inside the SQLite write transaction.
    # ========================================================


    try:
        persisted = (
            register_ml_performance_evaluation(
                evaluation=
                    evaluation
            )
        )

    except MLPerformanceEvaluationStoreError as error:
        raise MLPerformanceMonitoringServiceExecutionError(
            (
                "ML Performance Evaluation "
                "could not be committed."
            )
        ) from error


    return persisted
