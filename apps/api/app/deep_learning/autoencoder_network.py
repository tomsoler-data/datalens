from __future__ import annotations


import torch


from torch import nn


TABULAR_AUTOENCODER_NETWORK_RULE_VERSION = (
    "tabular_autoencoder_network_v0.1"
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


def _validate_feature_matrix(
    features: torch.Tensor,
    *,
    expected_columns: int,
    name: str,
) -> None:

    if not isinstance(
        features,
        torch.Tensor,
    ):

        raise TypeError(
            f"{name} must be a torch.Tensor."
        )


    if features.ndim != 2:

        raise ValueError(
            f"{name} must be two-dimensional."
        )


    if (
        features.dtype
        !=
        torch.float32
    ):

        raise TypeError(
            f"{name} must use torch.float32."
        )


    if (
        features.shape[1]
        !=
        expected_columns
    ):

        raise ValueError(
            (
                f"{name} column count must match "
                f"{expected_columns}."
            )
        )


class TabularAutoencoder(
    nn.Module,
):
    """
    Symmetric feed-forward autoencoder for preprocessed
    tabular feature matrices.

    The network reconstructs X from X. It contains no target
    semantics and performs no anomaly-threshold decision.
    """

    def __init__(
        self,
        *,
        input_features: int,
        hidden_features: int,
        latent_features: int,
    ) -> None:

        super().__init__()


        _validate_positive_integer(
            input_features,
            name=
                "input_features",
        )

        _validate_positive_integer(
            hidden_features,
            name=
                "hidden_features",
        )

        _validate_positive_integer(
            latent_features,
            name=
                "latent_features",
        )


        if (
            latent_features
            >
            hidden_features
        ):

            raise ValueError(
                (
                    "latent_features cannot exceed "
                    "hidden_features."
                )
            )


        self.input_features = (
            input_features
        )

        self.hidden_features = (
            hidden_features
        )

        self.latent_features = (
            latent_features
        )


        self.encoder_input = nn.Linear(
            input_features,
            hidden_features,
        )

        self.encoder_activation = (
            nn.ReLU()
        )

        self.latent_layer = nn.Linear(
            hidden_features,
            latent_features,
        )


        self.decoder_hidden = nn.Linear(
            latent_features,
            hidden_features,
        )

        self.decoder_activation = (
            nn.ReLU()
        )

        self.output_layer = nn.Linear(
            hidden_features,
            input_features,
        )


    def encode(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:

        _validate_feature_matrix(
            features,
            expected_columns=
                self.input_features,
            name=
                "features",
        )


        hidden = (
            self.encoder_input(
                features
            )
        )


        activated = (
            self.encoder_activation(
                hidden
            )
        )


        return self.latent_layer(
            activated
        )


    def decode(
        self,
        latent: torch.Tensor,
    ) -> torch.Tensor:

        _validate_feature_matrix(
            latent,
            expected_columns=
                self.latent_features,
            name=
                "latent",
        )


        hidden = (
            self.decoder_hidden(
                latent
            )
        )


        activated = (
            self.decoder_activation(
                hidden
            )
        )


        return self.output_layer(
            activated
        )


    def forward(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:

        latent = self.encode(
            features
        )


        return self.decode(
            latent
        )
