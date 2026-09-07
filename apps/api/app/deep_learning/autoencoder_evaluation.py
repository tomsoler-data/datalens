from __future__ import annotations


from dataclasses import (
    dataclass,
)


import torch


from torch import nn
from torch.utils.data import DataLoader


AUTOENCODER_EVALUATION_RULE_VERSION = (
    "autoencoder_evaluation_v0.1"
)


@dataclass(
    frozen=True,
)
class ReconstructionEvaluation:

    mean_loss: float

    reconstruction_errors: (
        torch.Tensor
    )


def evaluate_reconstruction_loader(
    model: nn.Module,
    *,
    data_loader: DataLoader,
    device: torch.device | str,
) -> ReconstructionEvaluation:
    """
    Evaluate reconstruction without gradients.

    Returns one mean squared reconstruction error per source
    row, in DataLoader iteration order.
    """

    if not isinstance(
        model,
        nn.Module,
    ):

        raise TypeError(
            "model must be a torch.nn.Module."
        )


    if not isinstance(
        data_loader,
        DataLoader,
    ):

        raise TypeError(
            "data_loader must be a torch DataLoader."
        )


    resolved_device = torch.device(
        device
    )


    parameter = next(
        model.parameters(),
        None,
    )


    if parameter is None:

        raise ValueError(
            "model must contain parameters."
        )


    model_device = (
        parameter.device
    )


    if (
        model_device.type
        !=
        resolved_device.type
    ):

        raise ValueError(
            (
                "model parameters must already be on "
                "the requested execution device."
            )
        )


    if (
        resolved_device.type
        ==
        "cuda"
        and
        resolved_device.index
        is not None
        and
        model_device.index
        !=
        resolved_device.index
    ):

        raise ValueError(
            (
                "model parameters must already be on "
                "the requested CUDA device."
            )
        )


    model.eval()


    error_batches: list[
        torch.Tensor
    ] = []


    with torch.inference_mode():

        for batch_features in data_loader:

            if not isinstance(
                batch_features,
                torch.Tensor,
            ):

                raise TypeError(
                    (
                        "reconstruction data_loader must "
                        "yield feature tensors only."
                    )
                )


            if (
                batch_features.ndim
                !=
                2
            ):

                raise ValueError(
                    (
                        "reconstruction feature batches "
                        "must be two-dimensional."
                    )
                )


            if (
                batch_features.dtype
                !=
                torch.float32
            ):

                raise TypeError(
                    (
                        "reconstruction feature batches "
                        "must use torch.float32."
                    )
                )


            batch_features = (
                batch_features.to(
                    resolved_device
                )
            )


            reconstruction = model(
                batch_features
            )


            if (
                reconstruction.shape
                !=
                batch_features.shape
            ):

                raise ValueError(
                    (
                        "autoencoder reconstruction shape "
                        "must match the feature batch shape."
                    )
                )


            row_errors = (
                (
                    reconstruction
                    -
                    batch_features
                )
                .pow(
                    2
                )
                .mean(
                    dim=1
                )
            )


            if not bool(
                torch.isfinite(
                    row_errors
                ).all()
            ):

                raise ValueError(
                    (
                        "reconstruction evaluation "
                        "produced non-finite errors."
                    )
                )


            error_batches.append(
                row_errors.detach().cpu()
            )


    if not error_batches:

        raise ValueError(
            "data_loader must yield at least one row."
        )


    reconstruction_errors = (
        torch.cat(
            error_batches,
            dim=0,
        )
    )


    return (
        ReconstructionEvaluation(
            mean_loss=
                float(
                    reconstruction_errors
                    .mean()
                    .item()
                ),
            reconstruction_errors=
                reconstruction_errors,
        )
    )
