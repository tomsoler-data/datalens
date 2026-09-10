from __future__ import annotations


import torch


from torch import (
    nn,
)


# ============================================================
# VERSION
# ============================================================


DL_TIME_SERIES_LSTM_NETWORK_RULE_VERSION = (
    "dl_time_series_lstm_network_v0.1"
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
# LSTM REGRESSOR
# ============================================================


class TimeSeriesLSTMRegressor(
    nn.Module
):
    """
    One-layer LSTM regressor for univariate forecasting windows.

    External input:

        batch x lookback

    Internal recurrent sequence:

        batch x lookback x 1

    PyTorch LSTM maintains two recurrent states:

        h = hidden state
        c = cell state

    Each forecasting window is independent. Recurrent state is
    not carried from one unrelated window or batch to another.
    """

    rule_version = (
        DL_TIME_SERIES_LSTM_NETWORK_RULE_VERSION
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


        self.recurrent = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=1,
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
                    "LSTM forecasting features must be "
                    "two-dimensional: batch x lookback."
                )
            )


        if features.dtype != torch.float32:

            raise TypeError(
                "LSTM features must use torch.float32."
            )


        if int(
            features.shape[
                1
            ]
        ) != self.lookback:

            raise ValueError(
                (
                    "LSTM sequence length must match "
                    "the forecasting lookback."
                )
            )


        sequence = features.unsqueeze(
            dim=-1
        )


        _, (
            hidden_state,
            _,
        ) = self.recurrent(
            sequence
        )


        final_hidden = hidden_state[
            -1,
            :,
            :,
        ]


        prediction = self.output_layer(
            final_hidden
        )


        return prediction.squeeze(
            dim=-1
        )
