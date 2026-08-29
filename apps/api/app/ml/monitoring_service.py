from __future__ import annotations


import pandas as pd


from app.ml.drift_evaluation import (
    MLDriftEvaluationRecord,
)


from app.ml.drift_evaluation_store import (
    MLDriftEvaluationStoreError,
    register_ml_drift_evaluation,
)


from app.ml.drift_evaluator import (
    MLDriftEvaluatorError,
    evaluate_ml_drift,
)


from app.ml.model_artifact_store import (
    MLModelArtifactStoreError,
    get_ml_model_artifact,
)


from app.ml.monitoring_profile_store import (
    MLMonitoringProfileStoreError,
    get_ml_monitoring_profile,
)



from app.ml.observed_dataset_authority import (
    MLObservedDatasetAuthorityError,
    MLObservedDatasetNotAuthorizedError,
    resolve_server_owned_observed_dataframe,
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


ML_MONITORING_SERVICE_RULE_VERSION = (
    "ml_monitoring_service_v0.1"
)


# ============================================================
# ERRORS
# ============================================================


class MLMonitoringServiceError(
    RuntimeError
):
    pass


class MLMonitoringServiceInputError(
    MLMonitoringServiceError
):
    pass


class MLMonitoringServiceAuthorityError(
    MLMonitoringServiceError
):
    pass


class MLMonitoringObservedDatasetError(
    MLMonitoringServiceError
):
    pass


class MLMonitoringServiceExecutionError(
    MLMonitoringServiceError
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
        raise MLMonitoringServiceInputError(
            (
                f"{field_name} "
                "cannot be empty."
            )
        )


    return normalized


# ============================================================
# FEATURE SURFACE
# ============================================================


def _build_observed_feature_surface(
    *,
    dataframe: pd.DataFrame,
    feature_columns: list[
        str
    ],
) -> pd.DataFrame:
    """
    Select exactly the model Training Contract feature surface.

    Target columns and unrelated dataset columns are deliberately
    excluded from drift evaluation.
    """

    normalized_features = [
        _required_text(
            feature_name,
            field_name=(
                "training feature"
            ),
        )

        for feature_name
        in feature_columns
    ]


    if not normalized_features:
        raise MLMonitoringServiceAuthorityError(
            (
                "Model Training Contract has "
                "no monitoring feature surface."
            )
        )


    if (
        len(
            normalized_features
        )
        !=
        len(
            set(
                normalized_features
            )
        )
    ):
        raise MLMonitoringServiceAuthorityError(
            (
                "Model Training Contract "
                "contains duplicate features."
            )
        )


    observed_columns = list(
        dataframe.columns
    )


    for feature_name in (
        normalized_features
    ):
        occurrences = sum(
            1
            for column_name
            in observed_columns
            if (
                column_name
                ==
                feature_name
            )
        )


        if occurrences == 0:
            raise MLMonitoringObservedDatasetError(
                (
                    "Observed dataset is missing "
                    "a required model feature. "
                    f"feature={feature_name}"
                )
            )


        if occurrences > 1:
            raise MLMonitoringObservedDatasetError(
                (
                    "Observed dataset contains "
                    "a duplicate required model "
                    "feature. "
                    f"feature={feature_name}"
                )
            )


    try:
        feature_surface = (
            dataframe.loc[
                :,
                normalized_features,
            ]
            .copy(
                deep=True
            )
        )

    except Exception as error:
        raise MLMonitoringObservedDatasetError(
            (
                "Observed feature surface "
                "could not be materialized."
            )
        ) from error


    return feature_surface


# ============================================================
# PUBLIC SERVICE
# ============================================================


def run_ml_monitoring(
    *,
    workflow_id: str,
    model_id: str,
    observed_dataset_id: str,
) -> MLDriftEvaluationRecord:
    """
    Execute one server-owned ML drift monitoring operation.

    Caller-controlled authority is intentionally limited to
    identities:

        workflow_id
        model_id
        observed_dataset_id

    The caller cannot provide:

        - raw rows;
        - a DataFrame;
        - a Model Artifact;
        - a Monitoring Profile;
        - a Training Contract fingerprint;
        - a Preparation revision;
        - an evaluation_id.

    Trust boundary:

        requested IDs
            ?
        Model Artifact Store
            ?
        Monitoring Profile Store
            ?
        validated Analysis Input Handoff
            ?
        exact model feature surface
            ?
        deterministic Drift Evaluator
            ?
        atomic Drift Evaluation Store
    """

    normalized_workflow_id = (
        _required_text(
            workflow_id,
            field_name="workflow_id",
        )
    )


    normalized_model_id = (
        _required_text(
            model_id,
            field_name="model_id",
        )
    )


    normalized_observed_dataset_id = (
        _required_text(
            observed_dataset_id,
            field_name=(
                "observed_dataset_id"
            ),
        )
    )


    # ========================================================
    # 1. MODEL AUTHORITY
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

    except MLModelArtifactStoreError as error:
        raise MLMonitoringServiceAuthorityError(
            (
                "Requested Model Artifact "
                "is not available in the "
                "requested workflow."
            )
        ) from error


    # ========================================================
    # 2. MONITORING REFERENCE AUTHORITY
    # ========================================================


    try:
        profile = (
            get_ml_monitoring_profile(
                model_id=
                    normalized_model_id,

                workflow_id=
                    normalized_workflow_id,
            )
        )

    except MLMonitoringProfileStoreError as error:
        raise MLMonitoringServiceAuthorityError(
            (
                "Trusted Monitoring Profile "
                "is not available for the "
                "requested Model Artifact."
            )
        ) from error


    # ========================================================
    # 3. CURRENT VALIDATED OBSERVED SNAPSHOT
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
        raise MLMonitoringServiceAuthorityError(
            (
                "Current Preparation output "
                "is not authorized for ML "
                "monitoring."
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
        raise MLMonitoringObservedDatasetError(
            (
                "Requested observed dataset "
                "is not authorized by the "
                "current validated Analysis "
                "Input Handoff."
            )
        ) from error

    except MLObservedDatasetAuthorityError as error:
        raise MLMonitoringServiceAuthorityError(
            (
                "Analysis Input Handoff "
                "authority is invalid."
            )
        ) from error


    # ========================================================
    # 4. EXACT TRAINING FEATURE SURFACE
    # ========================================================


    observed_features = (
        _build_observed_feature_surface(
            dataframe=
                observed_dataframe,

            feature_columns=
                list(
                    artifact
                    .training_contract
                    .feature_columns
                ),
        )
    )


    # ========================================================
    # 5. DETERMINISTIC DRIFT EVALUATION
    # ========================================================


    try:
        evaluation = (
            evaluate_ml_drift(
                observed_features=
                    observed_features,

                observed_dataset_id=
                    normalized_observed_dataset_id,

                observed_preparation_session_revision=
                    observed_revision,

                monitoring_profile=
                    profile,

                model_artifact=
                    artifact,
            )
        )

    except MLDriftEvaluatorError as error:
        raise MLMonitoringServiceExecutionError(
            (
                "ML Drift Evaluation "
                "could not be computed."
            )
        ) from error


    # ========================================================
    # 6. DURABLE COMMIT
    #
    # Drift Evaluation Store v12 re-checks the observed
    # Preparation revision inside its SQLite write transaction.
    # ========================================================


    try:
        persisted = (
            register_ml_drift_evaluation(
                evaluation=
                    evaluation
            )
        )

    except MLDriftEvaluationStoreError as error:
        raise MLMonitoringServiceExecutionError(
            (
                "ML Drift Evaluation "
                "could not be committed."
            )
        ) from error


    return persisted
