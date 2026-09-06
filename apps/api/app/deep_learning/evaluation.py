from __future__ import annotations


from dataclasses import (
    dataclass,
)


import torch


from torch import nn
from torch.utils.data import DataLoader


DEEP_LEARNING_EVALUATION_RULE_VERSION = (
    "deep_learning_evaluation_v0.1"
)


@dataclass(
    frozen=True,
)
class RegressionEvaluation:
    mean_loss: float
    predictions: torch.Tensor
    targets: torch.Tensor


def evaluate_regression_loader(
    model: nn.Module,
    *,
    data_loader: DataLoader,
    loss_function: nn.Module,
    device: torch.device | str,
) -> RegressionEvaluation:
    """Evaluate a regression network without updating parameters."""

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
            "model must contain parameters."
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


    model.eval()


    weighted_loss_sum = 0.0
    row_count = 0

    prediction_batches: list[torch.Tensor] = []
    target_batches: list[torch.Tensor] = []


    with torch.inference_mode():

        for (
            batch_features,
            batch_targets,
        ) in data_loader:

            batch_features = batch_features.to(
                resolved_device
            )

            batch_targets = batch_targets.to(
                resolved_device
            )


            predictions = model(
                batch_features
            )


            loss = loss_function(
                predictions,
                batch_targets
            )


            batch_rows = int(
                batch_features.shape[0]
            )


            weighted_loss_sum += (
                float(
                    loss.detach().cpu().item()
                )
                *
                batch_rows
            )

            row_count += (
                batch_rows
            )


            prediction_batches.append(
                predictions.detach().cpu()
            )

            target_batches.append(
                batch_targets.detach().cpu()
            )


    if row_count == 0:

        raise ValueError(
            "data_loader must yield at least one row."
        )


    return RegressionEvaluation(
        mean_loss=(
            weighted_loss_sum
            /
            row_count
        ),
        predictions=torch.cat(
            prediction_batches,
            dim=0,
        ),
        targets=torch.cat(
            target_batches,
            dim=0,
        ),
    )