from __future__ import annotations


import torch


from torch import nn


DEEP_LEARNING_TRAINING_RULE_VERSION = (
    "deep_learning_training_v0.1"
)


def build_regression_loss(
) -> nn.Module:
    """Build the regression loss used by DL foundations."""

    return nn.MSELoss()


def build_sgd_optimizer(
    model: nn.Module,
    *,
    learning_rate: float,
) -> torch.optim.Optimizer:
    """Build a basic SGD optimizer for a neural network."""

    if not isinstance(
        model,
        nn.Module,
    ):

        raise TypeError(
            "model must be a torch.nn.Module."
        )


    if isinstance(
        learning_rate,
        bool,
    ):

        raise TypeError(
            "learning_rate must be numeric."
        )


    if not isinstance(
        learning_rate,
        (
            int,
            float,
        ),
    ):

        raise TypeError(
            "learning_rate must be numeric."
        )


    if learning_rate <= 0:

        raise ValueError(
            "learning_rate must be greater than zero."
        )


    return torch.optim.SGD(
        model.parameters(),
        lr=float(
            learning_rate
        ),
    )


def train_regression_batch(
    model: nn.Module,
    *,
    features: torch.Tensor,
    targets: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    loss_function: nn.Module,
    device: torch.device | str,
) -> float:
    """Train a regression network on exactly one batch."""

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


    if not isinstance(
        targets,
        torch.Tensor,
    ):

        raise TypeError(
            "targets must be a torch.Tensor."
        )


    if features.ndim != 2:

        raise ValueError(
            "features must be two-dimensional."
        )


    if targets.ndim != 1:

        raise ValueError(
            "targets must be one-dimensional."
        )


    if (
        features.shape[0]
        !=
        targets.shape[0]
    ):

        raise ValueError(
            (
                "features and targets must contain "
                "the same number of rows."
            )
        )


    if features.dtype != torch.float32:

        raise TypeError(
            "features must use torch.float32."
        )


    if targets.dtype != torch.float32:

        raise TypeError(
            "targets must use torch.float32."
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


    model_parameter = next(
        model.parameters(),
        None,
    )

    if model_parameter is None:

        raise ValueError(
            "model must contain trainable parameters."
        )


    model_device = model_parameter.device


    if model_device.type != resolved_device.type:

        raise ValueError(
            (
                "model parameters must already be on "
                "the requested execution device."
            )
        )


    if (
        resolved_device.type == "cuda"
        and
        resolved_device.index is not None
        and
        model_device.index != resolved_device.index
    ):

        raise ValueError(
            (
                "model parameters must already be on "
                "the requested CUDA device."
            )
        )


    batch_features = features.to(
        resolved_device
    )

    batch_targets = targets.to(
        resolved_device
    )


    model.train()


    optimizer.zero_grad(
        set_to_none=True
    )


    predictions = model(
        batch_features
    )


    loss = loss_function(
        predictions,
        batch_targets
    )


    loss.backward()


    optimizer.step()


    return float(
        loss.detach().cpu().item()
    )

def train_regression_epochs(
    model: nn.Module,
    *,
    data_loader,
    optimizer: torch.optim.Optimizer,
    loss_function: nn.Module,
    epochs: int,
    device: torch.device | str,
) -> list[float]:
    """Train a regression network across batches and epochs."""

    if not isinstance(
        model,
        nn.Module,
    ):

        raise TypeError(
            "model must be a torch.nn.Module."
        )


    if not isinstance(
        data_loader,
        torch.utils.data.DataLoader,
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


    epoch_losses: list[float] = []


    for _ in range(
        epochs
    ):

        weighted_loss_sum = 0.0
        row_count = 0


        for (
            batch_features,
            batch_targets,
        ) in data_loader:

            batch_loss = train_regression_batch(
                model,
                features=batch_features,
                targets=batch_targets,
                optimizer=optimizer,
                loss_function=loss_function,
                device=device,
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


        epoch_losses.append(
            weighted_loss_sum
            /
            row_count
        )


    return epoch_losses