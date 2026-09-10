from __future__ import annotations


import torch


from torch import (
    nn,
)


# ============================================================
# VERSION
# ============================================================


DL_TIME_SERIES_RNN_NETWORK_RULE_VERSION = (
    "dl_time_series_rnn_network_v0.1"
)


# ============================================================
# VALIDATION
# ============================================================


def _validate_positive_integer(
    value: int,
    *,
    name: str,
) -> None:

    if isinstance(
        value,
        bool,
    ):

        raise TypeError(
            f"{name} must be an integer."
        )


    if not isinstance(
        value,
        int,
    ):

        raise TypeError(
            f"{name} must be an integer."
        )


    if value <= 0:

        raise ValueError(
            f"{name} must be greater than zero."
        )


# ============================================================
# SIMPLE RECURRENT REGRESSOR
# ============================================================


class TimeSeriesRNNRegressor(
    nn.Module
):
    """
    One-layer recurrent regressor for univariate lookback windows.

    External input:
        batch x lookback

    Internal sequence:
        batch x lookback x 1

    Each forecasting window is independent. Hidden state is not
    carried between separate samples or batches.
    """

    rule_version = (
        DL_TIME_SERIES_RNN_NETWORK_RULE_VERSION
    )


    def __init__(
        self,
        *,
        lookback: int,
        hidden_size: int,
    ) -> None:

        super().__init__()


        _validate_positive_integer(
            lookback,
            name="lookback",
        )


        _validate_positive_integer(
            hidden_size,
            name="hidden_size",
        )


        self.lookback = lookback
        self.hidden_size = hidden_size


        self.recurrent = nn.RNN(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=1,
            nonlinearity="tanh",
            batch_first=True,
            dropout=0.0,
            bidirectional=False,
        )


        self.output_layer = nn.Linear(
            hidden_size,
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
                (
                    "RNN forecasting features must be "
                    "two-dimensional: batch x lookback."
                )
            )


        if features.dtype != torch.float32:

            raise TypeError(
                "RNN features must use torch.float32."
            )


        if int(
            features.shape[
                1
            ]
        ) != self.lookback:

            raise ValueError(
                (
                    "RNN sequence length must match "
                    "the forecasting lookback."
                )
            )


        sequence = features.unsqueeze(
            dim=-1
        )


        recurrent_output, _ = self.recurrent(
            sequence
        )


        final_state = recurrent_output[
            :,
            -1,
            :,
        ]


        prediction = self.output_layer(
            final_state
        )


        return prediction.squeeze(
            dim=-1
        )
