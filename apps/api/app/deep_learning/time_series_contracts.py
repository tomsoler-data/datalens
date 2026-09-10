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


DL_TIME_SERIES_MLP_CONTRACT_RULE_VERSION = (
    "dl_time_series_mlp_contract_v0.1"
)


# ============================================================
# SERVER-OWNED EXECUTION AUTHORITY
# ============================================================


DL_TIME_SERIES_MODEL_RANDOM_SEED = 42


# ============================================================
# TEMPORAL MLP
# ============================================================


class DLTimeSeriesMLPRegressorHyperparameters(
    BaseModel
):
    """
    Controlled DataLens configuration for one feed-forward
    one-step time-series regression candidate.

    The network receives one already-created lookback window as
    a flat feature vector.

    Therefore:

        input_features = forecasting lookback

    The contract contains model configuration only.

    It intentionally does not expose:

    - dataset rows;
    - generated windows;
    - random seed;
    - execution device;
    - optimizer choice;
    - loss choice;
    - activation choice;
    - preprocessing statistics;
    - predictions;
    - learned model state.

    Those remain server-owned execution concerns.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )


    kind: Literal[
        "time_series_mlp_regressor"
    ] = "time_series_mlp_regressor"


    hidden_features: int = Field(
        default=32,
        ge=1,
        le=2048,
        strict=True,
    )


    epochs: int = Field(
        default=200,
        ge=1,
        le=2000,
        strict=True,
    )


    batch_size: int = Field(
        default=32,
        ge=1,
        le=65536,
        strict=True,
    )


    learning_rate: float = Field(
        default=0.01,
        gt=0.0,
        le=1.0,
    )


    rule_version: Literal[
        "dl_time_series_mlp_contract_v0.1"
    ] = (
        DL_TIME_SERIES_MLP_CONTRACT_RULE_VERSION
    )


# ============================================================
# SIMPLE RNN VERSION
# ============================================================


DL_TIME_SERIES_RNN_CONTRACT_RULE_VERSION = (
    "dl_time_series_rnn_contract_v0.1"
)


# ============================================================
# SIMPLE RNN
# ============================================================


class DLTimeSeriesRNNRegressorHyperparameters(
    BaseModel
):
    """
    Controlled DataLens hyperparameters for one simple recurrent
    one-step forecasting candidate.

    v0.1 deliberately fixes:

    - one recurrent layer;
    - input_size = 1;
    - tanh recurrent activation;
    - zero dropout;
    - one scalar output.

    Random seed, execution device, loss, optimizer family and
    learned state remain server-owned.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )


    kind: Literal[
        "time_series_rnn_regressor"
    ] = "time_series_rnn_regressor"


    hidden_size: int = Field(
        default=32,
        ge=1,
        le=2048,
        strict=True,
    )


    epochs: int = Field(
        default=200,
        ge=1,
        le=2000,
        strict=True,
    )


    batch_size: int = Field(
        default=32,
        ge=1,
        le=65536,
        strict=True,
    )


    learning_rate: float = Field(
        default=0.01,
        gt=0.0,
        le=1.0,
    )


    rule_version: Literal[
        "dl_time_series_rnn_contract_v0.1"
    ] = (
        DL_TIME_SERIES_RNN_CONTRACT_RULE_VERSION
    )


# ============================================================
# LSTM VERSION
# ============================================================


DL_TIME_SERIES_LSTM_CONTRACT_RULE_VERSION = (
    "dl_time_series_lstm_contract_v0.1"
)


# ============================================================
# LSTM
# ============================================================


class DLTimeSeriesLSTMRegressorHyperparameters(
    BaseModel
):
    """
    Controlled DataLens hyperparameters for one univariate
    one-step LSTM forecasting candidate.

    v0.1 deliberately fixes:

    - input_size = 1;
    - one recurrent layer;
    - zero dropout;
    - unidirectional recurrence;
    - one scalar regression output.

    Random seed, execution device, optimizer family, loss,
    preprocessing state and learned parameters remain
    server-owned.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )


    kind: Literal[
        "time_series_lstm_regressor"
    ] = "time_series_lstm_regressor"


    hidden_size: int = Field(
        default=32,
        ge=1,
        le=2048,
        strict=True,
    )


    epochs: int = Field(
        default=200,
        ge=1,
        le=2000,
        strict=True,
    )


    batch_size: int = Field(
        default=32,
        ge=1,
        le=65536,
        strict=True,
    )


    learning_rate: float = Field(
        default=0.01,
        gt=0.0,
        le=1.0,
    )


    rule_version: Literal[
        "dl_time_series_lstm_contract_v0.1"
    ] = (
        DL_TIME_SERIES_LSTM_CONTRACT_RULE_VERSION
    )
