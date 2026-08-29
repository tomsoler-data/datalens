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


# ============================================================
# VERSION
# ============================================================


ML_MONITORING_API_CONTRACT_RULE_VERSION = (
    "ml_monitoring_api_contract_v0.1"
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
            f"{field_name} cannot be empty."
        )


    return normalized


# ============================================================
# RUN REQUEST
# ============================================================


class MLMonitoringRunRequest(
    BaseModel
):

    """
    Public monitoring authority surface.

    The client may provide identities only.

    It cannot provide:

    - raw data;
    - DataFrames;
    - model metadata;
    - Monitoring Profiles;
    - Training Contract fingerprints;
    - Preparation revisions;
    - evaluation identities.
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


class MLMonitoringAPIErrorDetail(
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
        "ml_monitoring_api_v0.1"
    ] = "ml_monitoring_api_v0.1"
