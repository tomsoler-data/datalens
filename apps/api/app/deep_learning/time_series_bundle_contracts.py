from __future__ import annotations


from typing import (
    Literal,
)


from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)


# ============================================================
# VERSION
# ============================================================


DL_TIME_SERIES_NEURAL_BUNDLE_RULE_VERSION = (
    "dl_time_series_neural_bundle_v0.1"
)


DL_TIME_SERIES_STANDARDIZER_SNAPSHOT_RULE_VERSION = (
    "dl_time_series_standardizer_snapshot_v0.1"
)


# ============================================================
# ESTIMATOR / NETWORK AUTHORITIES
# ============================================================


DLTimeSeriesBundleEstimatorKey = Literal[
    "time_series_mlp_regressor",
    "time_series_rnn_regressor",
    "time_series_lstm_regressor",
]


DLTimeSeriesBundleNetworkClass = Literal[
    "FeedForwardRegressor",
    "TimeSeriesRNNRegressor",
    "TimeSeriesLSTMRegressor",
]


# ============================================================
# STANDARDIZER SNAPSHOT
# ============================================================


class DLTimeSeriesStandardizerSnapshot(
    BaseModel
):
    """
    JSON-safe persisted state for the TRAIN-only temporal scaler.

    This snapshot intentionally contains only the minimum state
    required to reproduce transform() and inverse_transform():

    - mean;
    - population standard deviation;
    - number of unique TRAIN observations fitted;
    - scaling rule version.

    Raw rows, windows, targets and predictions are excluded.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )


    snapshot_version: Literal[
        "dl_time_series_standardizer_snapshot_v0.1"
    ] = (
        DL_TIME_SERIES_STANDARDIZER_SNAPSHOT_RULE_VERSION
    )


    mean: float = Field(
        strict=True,
        allow_inf_nan=False,
    )


    standard_deviation: float = Field(
        gt=0.0,
        strict=True,
        allow_inf_nan=False,
    )


    fitted_observation_count: int = Field(
        ge=2,
        strict=True,
    )


    scaling_rule_version: Literal[
        "dl_time_series_scaling_v0.1"
    ]


# ============================================================
# BUNDLE MANIFEST
# ============================================================


class DLTimeSeriesNeuralBundleManifest(
    BaseModel
):
    """
    Privacy-minimal structural manifest for one temporal neural
    PyTorch bundle.

    The exact Training Contract is bound by SHA-256 rather than
    duplicated inside the binary bundle.

    Learned weights and scaler values are stored in separate
    checksummed members.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )


    bundle_version: Literal[
        "dl_time_series_neural_bundle_v0.1"
    ] = (
        DL_TIME_SERIES_NEURAL_BUNDLE_RULE_VERSION
    )


    estimator_key: (
        DLTimeSeriesBundleEstimatorKey
    )


    network_class: (
        DLTimeSeriesBundleNetworkClass
    )


    state_dict_format: Literal[
        "torch_state_dict"
    ] = "torch_state_dict"


    standardizer_format: Literal[
        "datalens_json"
    ] = "datalens_json"


    lookback: int = Field(
        gt=0,
        strict=True,
    )


    hidden_width: int = Field(
        gt=0,
        strict=True,
    )


    training_contract_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$",
    )


    state_dict_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$",
    )


    standardizer_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$",
    )


    # ========================================================
    # ESTIMATOR / NETWORK CONSISTENCY
    # ========================================================


    @model_validator(
        mode="after"
    )
    def validate_network_identity(
        self,
    ):

        expected_network = {
            "time_series_mlp_regressor":
                "FeedForwardRegressor",

            "time_series_rnn_regressor":
                "TimeSeriesRNNRegressor",

            "time_series_lstm_regressor":
                "TimeSeriesLSTMRegressor",
        }[
            self.estimator_key
        ]


        if (
            self.network_class
            !=
            expected_network
        ):

            raise ValueError(
                (
                    "Temporal bundle estimator/network "
                    "identity mismatch. "
                    f"estimator_key={self.estimator_key!r}, "
                    f"network_class={self.network_class!r}"
                )
            )


        return self
