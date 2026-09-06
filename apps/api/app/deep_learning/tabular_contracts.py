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


DL_TABULAR_MLP_CONTRACT_RULE_VERSION = (
    "dl_tabular_mlp_contract_v0.1"
)


# ============================================================
# TABULAR MLP REGRESSOR
# ============================================================


class DLTabularMLPRegressorHyperparameters(
    BaseModel
):
    """
    Controlled DataLens configuration for one tabular MLP
    regression candidate.

    This contract contains configuration only.

    It intentionally does not expose:

    - random seed;
    - execution device;
    - optimizer selection;
    - loss-function selection;
    - activation selection;
    - DataLoader worker count.

    Those execution decisions remain server-owned.

    v0.1 corresponds to the neural-network foundation already
    implemented by DataLens:

        input
        -> Linear
        -> ReLU
        -> Linear(1)

    with MSE loss and SGD optimization.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )


    kind: Literal[
        "tabular_mlp_regressor"
    ] = "tabular_mlp_regressor"


    hidden_features: int = Field(
        default=64,
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
        default=0.01,
        gt=0.0,
        le=1.0,
    )


    rule_version: Literal[
        "dl_tabular_mlp_contract_v0.1"
    ] = DL_TABULAR_MLP_CONTRACT_RULE_VERSION
