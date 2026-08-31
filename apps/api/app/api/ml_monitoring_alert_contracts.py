from __future__ import annotations


from typing import (
    Literal,
)


from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)


# ============================================================
# VERSION
# ============================================================


ML_MONITORING_ALERT_API_CONTRACT_RULE_VERSION = (
    "ml_monitoring_alert_api_contract_v0.1"
)


# ============================================================
# ERROR DETAIL
# ============================================================


class MLMonitoringAlertAPIErrorDetail(
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


    retryable: bool = False


    api_version: Literal[
        "ml_monitoring_alert_api_v0.1"
    ] = "ml_monitoring_alert_api_v0.1"
