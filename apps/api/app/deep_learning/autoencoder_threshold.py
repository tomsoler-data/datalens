from __future__ import annotations


from dataclasses import (
    dataclass,
)


import math
import torch


AUTOENCODER_THRESHOLD_RULE_VERSION = (
    "autoencoder_threshold_v0.1"
)


AUTOENCODER_THRESHOLD_METHOD = (
    "train_reconstruction_error_quantile"
)


AUTOENCODER_THRESHOLD_COMPARISON = (
    "greater_than"
)


@dataclass(
    frozen=True,
)
class ReconstructionErrorThreshold:

    quantile: float

    threshold: float

    train_rows: int

    method: str = (
        AUTOENCODER_THRESHOLD_METHOD
    )

    comparison_operator: str = (
        AUTOENCODER_THRESHOLD_COMPARISON
    )


def _validate_errors(
    errors: torch.Tensor,
    *,
    name: str,
    minimum_rows: int = 2,
) -> None:

    if not isinstance(
        errors,
        torch.Tensor,
    ):

        raise TypeError(
            f"{name} must be a torch.Tensor."
        )


    if errors.ndim != 1:

        raise ValueError(
            f"{name} must be one-dimensional."
        )


    if (
        isinstance(
            minimum_rows,
            bool,
        )
        or
        not isinstance(
            minimum_rows,
            int,
        )
        or
        minimum_rows
        <
        1
    ):

        raise ValueError(
            "minimum_rows must be a positive integer."
        )


    if errors.numel() < minimum_rows:

        raise ValueError(
            (
                f"{name} must contain at least "
                f"{minimum_rows} row(s)."
            )
        )


    if (
        errors.dtype
        !=
        torch.float32
    ):

        raise TypeError(
            f"{name} must use torch.float32."
        )


    if (
        errors.device.type
        !=
        "cpu"
    ):

        raise ValueError(
            f"{name} must remain on CPU."
        )


    if not bool(
        torch.isfinite(
            errors
        ).all()
    ):

        raise ValueError(
            f"{name} must contain only finite values."
        )


    if bool(
        (
            errors
            <
            0.0
        )
        .any()
    ):

        raise ValueError(
            (
                f"{name} cannot contain negative "
                "reconstruction errors."
            )
        )


def fit_reconstruction_error_threshold(
    *,
    train_errors: torch.Tensor,
    quantile: float,
) -> ReconstructionErrorThreshold:
    """
    Fit exactly one anomaly threshold from TRAIN reconstruction
    errors.

    No TEST reconstruction error is accepted by this API.
    """

    _validate_errors(
        train_errors,
        name=
            "train_errors",
    )


    if isinstance(
        quantile,
        bool,
    ):

        raise TypeError(
            "quantile must be numeric."
        )


    if not isinstance(
        quantile,
        (
            int,
            float,
        ),
    ):

        raise TypeError(
            "quantile must be numeric."
        )


    resolved_quantile = float(
        quantile
    )


    if not math.isfinite(
        resolved_quantile
    ):

        raise ValueError(
            "quantile must be finite."
        )


    if not (
        0.0
        <
        resolved_quantile
        <
        1.0
    ):

        raise ValueError(
            (
                "quantile must be strictly between "
                "zero and one."
            )
        )


    threshold_tensor = (
        torch.quantile(
            train_errors,
            q=
                resolved_quantile,
            interpolation=
                "linear",
        )
    )


    threshold = float(
        threshold_tensor.item()
    )


    if not math.isfinite(
        threshold
    ):

        raise ValueError(
            "resolved anomaly threshold is non-finite."
        )


    if threshold < 0.0:

        raise ValueError(
            "resolved anomaly threshold is negative."
        )


    return (
        ReconstructionErrorThreshold(
            quantile=
                resolved_quantile,
            threshold=
                threshold,
            train_rows=
                int(
                    train_errors.numel()
                ),
        )
    )


def apply_reconstruction_error_threshold(
    *,
    reconstruction_errors: torch.Tensor,
    threshold: ReconstructionErrorThreshold,
) -> torch.Tensor:
    """
    Apply a previously TRAIN-fitted threshold.

    A row is anomalous only when its reconstruction error is
    strictly greater than the fitted TRAIN quantile threshold.
    """

    _validate_errors(
        reconstruction_errors,
        name=
            "reconstruction_errors",
        minimum_rows=
            1,
    )


    if not isinstance(
        threshold,
        ReconstructionErrorThreshold,
    ):

        raise TypeError(
            (
                "threshold must be a "
                "ReconstructionErrorThreshold."
            )
        )


    if (
        not math.isfinite(
            threshold.threshold
        )
        or
        threshold.threshold
        <
        0.0
    ):

        raise ValueError(
            "threshold value is invalid."
        )


    return (
        reconstruction_errors
        >
        threshold.threshold
    )
