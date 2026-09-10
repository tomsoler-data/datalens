from __future__ import annotations


from typing import (
    Literal,
)


from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


from app.ml.contracts import (
    MLTimeHoldoutSplitContract,
)


# ============================================================
# VERSION
# ============================================================


ML_TIME_SERIES_FORECASTING_CONTRACT_RULE_VERSION = (
    "ml_time_series_forecasting_contract_v0.1"
)


# ============================================================
# FORECASTING TASK CONTRACT
# ============================================================


class MLTimeSeriesForecastingContract(
    BaseModel
):
    """
    Controlled DataLens definition of one univariate
    time-series forecasting task.

    v0.1 deliberately supports only:

    - supervised regression;
    - one numeric target series;
    - one explicit chronological observation-time column;
    - one-step-ahead forecasting;
    - one explicit lookback window;
    - chronological holdout;
    - no shuffling;
    - no stratification.

    This contract describes the forecasting problem only.

    It intentionally does not contain:

    - raw observations;
    - generated windows;
    - predictions;
    - learned preprocessing state;
    - fitted model state;
    - random seed;
    - execution device;
    - optimizer configuration;
    - loss configuration;
    - neural-network architecture.

    Model-specific configuration belongs to later estimator
    contracts.

    The observation-time column is split metadata and is not a
    model feature.

    The historical target itself supplies the univariate
    sequence values in v0.1.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )


    workflow_id: str = Field(
        min_length=1,
    )


    dataset_id: str = Field(
        min_length=1,
    )


    task_family: Literal[
        "time_series_forecasting"
    ] = "time_series_forecasting"


    problem_type: Literal[
        "regression"
    ] = "regression"


    target_column: str = Field(
        min_length=1,
    )


    lookback: int = Field(
        ge=1,
        le=4096,
        strict=True,
    )


    forecast_horizon: Literal[
        1
    ] = 1


    split: MLTimeHoldoutSplitContract


    rule_version: Literal[
        "ml_time_series_forecasting_contract_v0.1"
    ] = (
        ML_TIME_SERIES_FORECASTING_CONTRACT_RULE_VERSION
    )


    # ========================================================
    # TEXT NORMALIZATION
    # ========================================================


    @field_validator(
        "workflow_id",
        "dataset_id",
        "target_column",
        mode="before",
    )
    @classmethod
    def normalize_required_text(
        cls,
        value: object,
    ) -> str:

        normalized = str(
            value
            if value is not None
            else ""
        ).strip()


        if not normalized:

            raise ValueError(
                "value cannot be empty"
            )


        return normalized


    # ========================================================
    # CROSS-FIELD AUTHORITY
    # ========================================================


    @model_validator(
        mode="after"
    )
    def validate_forecasting_contract(
        self,
    ) -> "MLTimeSeriesForecastingContract":

        if (
            self.target_column
            ==
            self.split.time_column
        ):

            raise ValueError(
                (
                    "target_column and time_column "
                    "must reference different columns"
                )
            )


        return self


    # ========================================================
    # DERIVED AUTHORITY
    # ========================================================


    @property
    def time_column(
        self,
    ) -> str:

        return (
            self.split.time_column
        )
