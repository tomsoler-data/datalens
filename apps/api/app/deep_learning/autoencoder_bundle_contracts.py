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
# AUTHORITY
# ============================================================


DL_TABULAR_AUTOENCODER_BUNDLE_RULE_VERSION = (
    "dl_tabular_autoencoder_bundle_v0.1"
)


# ============================================================
# MANIFEST
# ============================================================


class TabularAutoencoderBundleManifest(
    BaseModel
):
    """
    Privacy-minimal structural manifest for one trusted
    DataLens Tabular Autoencoder PyTorch bundle.

    The manifest contains the information required to rebuild
    the inference function without retraining, refitting the
    preprocessor or refitting the anomaly threshold.

    It deliberately contains no raw rows, reconstruction
    errors, anomaly flags or learned preprocessing statistics.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
    )


    bundle_version: Literal[
        "dl_tabular_autoencoder_bundle_v0.1"
    ] = DL_TABULAR_AUTOENCODER_BUNDLE_RULE_VERSION


    estimator_key: Literal[
        "tabular_autoencoder"
    ]


    network_class: Literal[
        "TabularAutoencoder"
    ]


    state_dict_format: Literal[
        "torch_state_dict"
    ]


    preprocessor_format: Literal[
        "joblib"
    ]


    input_features: int = Field(
        gt=0,
    )


    hidden_features: int = Field(
        gt=0,
    )


    latent_features: int = Field(
        gt=0,
    )


    training_contract_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$",
    )


    threshold_rule_version: Literal[
        "autoencoder_threshold_v0.1"
    ]


    threshold_quantile: float = Field(
        gt=0.0,
        lt=1.0,
    )


    anomaly_threshold: float = Field(
        ge=0.0,
    )


    threshold_train_rows: int = Field(
        ge=2,
    )


    threshold_method: Literal[
        "train_reconstruction_error_quantile"
    ]


    threshold_comparison_operator: Literal[
        "greater_than"
    ]


    state_dict_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$",
    )


    preprocessor_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$",
    )


    @model_validator(
        mode="after"
    )
    def validate_architecture(
        self,
    ) -> TabularAutoencoderBundleManifest:

        if (
            self.latent_features
            >
            self.hidden_features
        ):
            raise ValueError(
                (
                    "latent_features cannot exceed "
                    "hidden_features."
                )
            )


        return self
