from __future__ import annotations


from typing import (
    Literal,
)


from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)



from app.ml.performance_evaluation import (
    MLPerformanceEvaluationRecord,
)


# ============================================================
# VERSION
# ============================================================


ML_PERFORMANCE_MONITORING_API_CONTRACT_RULE_VERSION = (
    "ml_performance_monitoring_api_contract_v0.1"
)


# ============================================================
# HELPERS
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
        raise ValueError(
            (
                f"{field_name} "
                "cannot be empty."
            )
        )


    return normalized


# ============================================================
# RUN REQUEST
# ============================================================


class MLPerformanceMonitoringRunRequest(
    BaseModel
):
    """
    Public supervised Performance Monitoring authority surface.

    The client may provide identities only.

    The client cannot provide:

    - raw rows;
    - DataFrames;
    - ground-truth values;
    - predictions;
    - model bytes;
    - filesystem paths;
    - Model Artifact metadata;
    - Training Contracts;
    - reference metrics;
    - Preparation revisions;
    - Performance Evaluation identities.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )


    workflow_id: str = Field(
        min_length=1,
    )


    model_id: str = Field(
        min_length=1,
    )


    observed_dataset_id: str = Field(
        min_length=1,
    )


    @field_validator(
        "workflow_id",
        "model_id",
        "observed_dataset_id",
        mode="before",
    )
    @classmethod
    def validate_identity(
        cls,
        value: object,
        info,
    ) -> str:

        return (
            _required_text(
                value,
                field_name=
                    info.field_name,
            )
        )


# ============================================================
# ERROR DETAIL
# ============================================================


class MLPerformanceMonitoringAPIErrorDetail(
    BaseModel
):

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )


    error: str = Field(
        min_length=1,
    )


    message: str = Field(
        min_length=1,
    )


    workflow_id: (
        str
        |
        None
    ) = None


    model_id: (
        str
        |
        None
    ) = None


    observed_dataset_id: (
        str
        |
        None
    ) = None


    retryable: bool = False


    api_version: Literal[
        "ml_performance_monitoring_api_v0.1"
    ] = (
        "ml_performance_monitoring_api_v0.1"
    )


# ============================================================
# MODEL HISTORY RESPONSE
# ============================================================


class MLPerformanceMonitoringModelHistoryResponse(
    BaseModel
):

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )


    workflow_id: str = Field(
        min_length=1,
    )


    model_id: str = Field(
        min_length=1,
    )


    evaluation_count: int = Field(
        ge=0,
        strict=True,
    )


    evaluations: list[
        MLPerformanceEvaluationRecord
    ]


    api_version: Literal[
        "ml_performance_monitoring_history_api_v0.1"
    ] = (
        "ml_performance_monitoring_history_api_v0.1"
    )


# ============================================================
# WORKFLOW HISTORY RESPONSE
# ============================================================


class MLPerformanceMonitoringWorkflowHistoryResponse(
    BaseModel
):

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )


    workflow_id: str = Field(
        min_length=1,
    )


    evaluation_count: int = Field(
        ge=0,
        strict=True,
    )


    evaluations: list[
        MLPerformanceEvaluationRecord
    ]


    api_version: Literal[
        "ml_performance_monitoring_history_api_v0.1"
    ] = (
        "ml_performance_monitoring_history_api_v0.1"
    )
