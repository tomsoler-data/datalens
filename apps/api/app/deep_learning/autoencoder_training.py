from __future__ import annotations


import torch


from torch import nn
from torch.utils.data import DataLoader


AUTOENCODER_TRAINING_RULE_VERSION = (
    "autoencoder_training_v0.1"
)


def train_reconstruction_batch(
    model: nn.Module,
    *,
    features: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    loss_function: nn.Module,
    device: torch.device | str,
) -> float:
    """
    Train an autoencoder on one feature-only batch.

    The reconstruction target is the input batch itself and is
    never represented as a supervised target tensor.
    """

    if not isinstance(
        model,
        nn.Module,
    ):

        raise TypeError(
            "model must be a torch.nn.Module."
        )


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


    if (
        features.dtype
        !=
        torch.float32
    ):

        raise TypeError(
            "features must use torch.float32."
        )


    if not bool(
        torch.isfinite(
            features
        ).all()
    ):

        raise ValueError(
            "features must contain only finite values."
        )


    if not isinstance(
        optimizer,
        torch.optim.Optimizer,
    ):

        raise TypeError(
            "optimizer must be a torch optimizer."
        )


    if not isinstance(
        loss_function,
        nn.Module,
    ):

        raise TypeError(
            "loss_function must be a torch.nn.Module."
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
            "model must contain trainable parameters."
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


    batch_features = (
        features.to(
            resolved_device
        )
    )


    model.train()


    optimizer.zero_grad(
        set_to_none=True
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


    loss = loss_function(
        reconstruction,
        batch_features,
    )


    if not bool(
        torch.isfinite(
            loss.detach()
        ).all()
    ):

        raise ValueError(
            "reconstruction loss became non-finite."
        )


    loss.backward()


    optimizer.step()


    return float(
        loss.detach().cpu().item()
    )


def train_reconstruction_epochs(
    model: nn.Module,
    *,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_function: nn.Module,
    epochs: int,
    device: torch.device | str,
) -> list[float]:
    """
    Train a reconstruction network across feature-only batches.
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


    if not isinstance(
        optimizer,
        torch.optim.Optimizer,
    ):

        raise TypeError(
            "optimizer must be a torch optimizer."
        )


    if not isinstance(
        loss_function,
        nn.Module,
    ):

        raise TypeError(
            "loss_function must be a torch.nn.Module."
        )


    if (
        isinstance(
            epochs,
            bool,
        )
        or
        not isinstance(
            epochs,
            int,
        )
    ):

        raise TypeError(
            "epochs must be an integer."
        )


    if epochs <= 0:

        raise ValueError(
            "epochs must be greater than zero."
        )


    epoch_losses: list[
        float
    ] = []


    for _ in range(
        epochs
    ):

        weighted_loss_sum = 0.0
        row_count = 0


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


            batch_loss = (
                train_reconstruction_batch(
                    model,
                    features=
                        batch_features,
                    optimizer=
                        optimizer,
                    loss_function=
                        loss_function,
                    device=
                        device,
                )
            )


            batch_rows = int(
                batch_features.shape[0]
            )


            weighted_loss_sum += (
                batch_loss
                *
                batch_rows
            )

            row_count += (
                batch_rows
            )


        if row_count == 0:

            raise ValueError(
                "data_loader must yield at least one row."
            )


        epoch_loss = (
            weighted_loss_sum
            /
            row_count
        )


        if not torch.isfinite(
            torch.tensor(
                epoch_loss,
                dtype=torch.float64,
            )
        ):

            raise ValueError(
                "epoch reconstruction loss is non-finite."
            )


        epoch_losses.append(
            float(
                epoch_loss
            )
        )


    return epoch_losses
