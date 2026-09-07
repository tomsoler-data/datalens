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


DL_TABULAR_AUTOENCODER_CONTRACT_RULE_VERSION = (
    "dl_tabular_autoencoder_contract_v0.1"
)


# ============================================================
# HYPERPARAMETERS
# ============================================================


class DLTabularAutoencoderHyperparameters(
    BaseModel
):
    """
    Torch-free configuration contract for the v0.1
    tabular autoencoder.

    Execution controls such as device, random seed,
    optimizer implementation, loss implementation,
    DataLoader workers and anomaly threshold policy
    remain server-owned.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )


    kind: Literal[
        "tabular_autoencoder"
    ] = "tabular_autoencoder"


    hidden_features: int = Field(
        default=64,
        ge=1,
        le=2048,
    )


    latent_features: int = Field(
        default=16,
        ge=1,
        le=2048,
    )


    epochs: int = Field(
        default=100,
        ge=1,
        le=2000,
    )


    batch_size: int = Field(
        default=64,
        ge=1,
        le=65536,
    )


    learning_rate: float = Field(
        default=0.001,
        gt=0.0,
        le=1.0,
    )


    rule_version: Literal[
        "dl_tabular_autoencoder_contract_v0.1"
    ] = (
        DL_TABULAR_AUTOENCODER_CONTRACT_RULE_VERSION
    )


    @model_validator(
        mode="after",
    )
    def validate_architecture(
        self,
    ) -> "DLTabularAutoencoderHyperparameters":

        if (
            self.latent_features
            >
            self.hidden_features
        ):
            raise ValueError(
                (
                    "latent_features must be less "
                    "than or equal to hidden_features."
                )
            )

        return self
