from __future__ import annotations


import torch


from torch import nn


DEEP_LEARNING_NETWORK_RULE_VERSION = (
    "deep_learning_network_v0.1"
)


def _validate_positive_integer(
    value: int,
    *,
    name: str,
) -> None:

    if (
        isinstance(
            value,
            bool,
        )
        or
        not isinstance(
            value,
            int,
        )
    ):

        raise TypeError(
            f"{name} must be an integer."
        )


    if value <= 0:

        raise ValueError(
            f"{name} must be greater than zero."
        )


class FeedForwardRegressor(
    nn.Module,
):
    """Minimal feed-forward regression network foundation."""

    def __init__(
        self,
        *,
        input_features: int,
        hidden_features: int,
    ) -> None:

        super().__init__()


        _validate_positive_integer(
            input_features,
            name="input_features",
        )

        _validate_positive_integer(
            hidden_features,
            name="hidden_features",
        )


        self.input_features = (
            input_features
        )

        self.hidden_features = (
            hidden_features
        )


        self.input_layer = nn.Linear(
            input_features,
            hidden_features,
        )

        self.activation = nn.ReLU()

        self.output_layer = nn.Linear(
            hidden_features,
            1,
        )


    def forward(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:

        if not isinstance(
            features,
            torch.Tensor,
        ):

            raise TypeError(
                "features must be a torch.Tensor."
            )


        if features.ndim != 2:

            raise ValueError(
                "features must be two-dimensional."
            )


        if features.dtype != torch.float32:

            raise TypeError(
                "features must use torch.float32."
            )


        if (
            features.shape[1]
            !=
            self.input_features
        ):

            raise ValueError(
                (
                    "features column count must match "
                    "input_features."
                )
            )


        hidden = self.input_layer(
            features
        )

        activated = self.activation(
            hidden
        )

        prediction = self.output_layer(
            activated
        )


        return prediction.squeeze(
            dim=-1
        )