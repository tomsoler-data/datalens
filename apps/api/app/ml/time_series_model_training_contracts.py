from __future__ import annotations


from typing import (
    Literal,
    Union,
)


from pydantic import (
    BaseModel,
    ConfigDict,
    field_validator,
)


from app.deep_learning.time_series_contracts import (
    DLTimeSeriesLSTMRegressorHyperparameters,
    DLTimeSeriesMLPRegressorHyperparameters,
    DLTimeSeriesRNNRegressorHyperparameters,
)


from app.ml.time_series_contracts import (
    MLTimeSeriesForecastingContract,
)


# ============================================================
# VERSION
# ============================================================


ML_TIME_SERIES_MODEL_TRAINING_CONTRACT_RULE_VERSION = (
    "ml_time_series_model_training_contract_v0.1"
)


# ============================================================
# ESTIMATOR FAMILY
# ============================================================


MLTimeSeriesEstimatorHyperparameters = Union[
    DLTimeSeriesMLPRegressorHyperparameters,
    DLTimeSeriesRNNRegressorHyperparameters,
    DLTimeSeriesLSTMRegressorHyperparameters,
]


# ============================================================
# ESTIMATOR VALIDATION
# ============================================================


def validate_time_series_estimator_hyperparameters(
    value: object,
) -> MLTimeSeriesEstimatorHyperparameters:
    """
    Resolve exactly one supported DL-5 forecasting estimator.

    Dispatch is based only on the explicit estimator kind.
    """

    if isinstance(
        value,
        DLTimeSeriesMLPRegressorHyperparameters,
    ):

        return (
            DLTimeSeriesMLPRegressorHyperparameters
            .model_validate(
                value
            )
        )


    if isinstance(
        value,
        DLTimeSeriesRNNRegressorHyperparameters,
    ):

        return (
            DLTimeSeriesRNNRegressorHyperparameters
            .model_validate(
                value
            )
        )


    if isinstance(
        value,
        DLTimeSeriesLSTMRegressorHyperparameters,
    ):

        return (
            DLTimeSeriesLSTMRegressorHyperparameters
            .model_validate(
                value
            )
        )


    if not isinstance(
        value,
        dict,
    ):

        raise ValueError(
            (
                "Forecast estimator hyperparameters must "
                "be a supported validated contract or "
                "JSON-style object."
            )
        )


    kind = str(
        value.get(
            "kind",
            "",
        )
    ).strip()


    if (
        kind
        ==
        "time_series_mlp_regressor"
    ):

        return (
            DLTimeSeriesMLPRegressorHyperparameters
            .model_validate(
                value
            )
        )


    if (
        kind
        ==
        "time_series_rnn_regressor"
    ):

        return (
            DLTimeSeriesRNNRegressorHyperparameters
            .model_validate(
                value
            )
        )


    if (
        kind
        ==
        "time_series_lstm_regressor"
    ):

        return (
            DLTimeSeriesLSTMRegressorHyperparameters
            .model_validate(
                value
            )
        )


    raise ValueError(
        (
            "Unsupported forecasting estimator kind. "
            f"kind={kind!r}"
        )
    )


# ============================================================
# MODEL TRAINING CONTRACT
# ============================================================


class MLTimeSeriesModelTrainingContract(
    BaseModel
):
    """
    Per-model training contract for one already-defined
    forecasting task.

    The Forecasting Task Contract remains model-neutral.

    This contract composes:

        forecasting task
        +
        estimator hyperparameters

    It deliberately contains no:

    - raw rows;
    - generated windows;
    - predictions;
    - fitted preprocessing statistics;
    - learned weights;
    - execution device;
    - random seed;
    - optimizer state.

    Those remain server-owned execution or Artifact concerns.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )


    task_contract: (
        MLTimeSeriesForecastingContract
    )


    estimator_hyperparameters: (
        MLTimeSeriesEstimatorHyperparameters
    )


    rule_version: Literal[
        "ml_time_series_model_training_contract_v0.1"
    ] = (
        ML_TIME_SERIES_MODEL_TRAINING_CONTRACT_RULE_VERSION
    )


    # ========================================================
    # TASK CONTRACT
    # ========================================================


    @field_validator(
        "task_contract",
        mode="before",
    )
    @classmethod
    def validate_task_contract(
        cls,
        value: object,
    ) -> MLTimeSeriesForecastingContract:

        return (
            MLTimeSeriesForecastingContract
            .model_validate(
                value
            )
        )


    # ========================================================
    # ESTIMATOR CONTRACT
    # ========================================================


    @field_validator(
        "estimator_hyperparameters",
        mode="before",
    )
    @classmethod
    def validate_estimator_hyperparameters(
        cls,
        value: object,
    ) -> MLTimeSeriesEstimatorHyperparameters:

        return (
            validate_time_series_estimator_hyperparameters(
                value
            )
        )


    # ========================================================
    # GENERIC MODEL LAB SURFACE
    # ========================================================


    @property
    def workflow_id(
        self,
    ) -> str:

        return (
            self.task_contract.workflow_id
        )


    @property
    def dataset_id(
        self,
    ) -> str:

        return (
            self.task_contract.dataset_id
        )


    @property
    def task_family(
        self,
    ) -> Literal[
        "time_series_forecasting"
    ]:

        return (
            self.task_contract.task_family
        )


    @property
    def problem_type(
        self,
    ) -> Literal[
        "regression"
    ]:

        return (
            self.task_contract.problem_type
        )


    @property
    def target_column(
        self,
    ) -> str:

        return (
            self.task_contract.target_column
        )


    @property
    def time_column(
        self,
    ) -> str:

        return (
            self.task_contract.time_column
        )


    @property
    def lookback(
        self,
    ) -> int:

        return int(
            self.task_contract.lookback
        )


    @property
    def forecast_horizon(
        self,
    ) -> int:

        return int(
            self.task_contract.forecast_horizon
        )


    @property
    def split(
        self,
    ):

        return (
            self.task_contract.split
        )


    @property
    def estimator_key(
        self,
    ) -> str:

        return str(
            self.estimator_hyperparameters.kind
        )


    @property
    def effective_estimator_hyperparameters(
        self,
    ) -> MLTimeSeriesEstimatorHyperparameters:

        return (
            self.estimator_hyperparameters
        )
