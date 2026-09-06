from __future__ import annotations


from collections.abc import (
    Sequence,
)


import torch


from app.deep_learning.runtime import (
    resolve_device,
)


DEEP_LEARNING_TENSOR_RULE_VERSION = (
    "deep_learning_tensor_v0.1"
)


def build_float_tensor(
    rows: Sequence[
        Sequence[
            float
        ]
    ],
    *,
    device: (
        str
        | torch.device
        | None
    ) = None,
) -> torch.Tensor:
    """Create a non-empty rectangular 2-D float32 tensor."""

    if len(rows) == 0:

        raise ValueError(
            "rows must not be empty."
        )


    first_row = rows[0]


    if (
        isinstance(
            first_row,
            (
                str,
                bytes,
            ),
        )
        or
        not isinstance(
            first_row,
            Sequence,
        )
    ):

        raise ValueError(
            "rows must contain row sequences."
        )


    width = len(
        first_row
    )


    if width == 0:

        raise ValueError(
            "rows must not contain empty rows."
        )


    for row in rows:

        if (
            isinstance(
                row,
                (
                    str,
                    bytes,
                ),
            )
            or
            not isinstance(
                row,
                Sequence,
            )
        ):

            raise ValueError(
                "rows must contain row sequences."
            )


        if len(row) != width:

            raise ValueError(
                "rows must form a rectangular matrix."
            )


    tensor = torch.tensor(
        rows,
        dtype=torch.float32,
    )


    if tensor.ndim != 2:

        raise ValueError(
            "rows must produce a two-dimensional tensor."
        )


    if device is not None:

        tensor = tensor.to(
            device
        )


    return tensor


def move_tensor(
    tensor: torch.Tensor,
    *,
    device: (
        str
        | torch.device
    ),
) -> torch.Tensor:
    """Move one PyTorch tensor to an explicit device."""

    if not isinstance(
        tensor,
        torch.Tensor,
    ):

        raise TypeError(
            "tensor must be a torch.Tensor."
        )


    return tensor.to(
        device
    )


def scalar_square_gradient(
    value: float,
    *,
    device: (
        str
        | torch.device
        | None
    ) = None,
) -> tuple[
    float,
    float,
]:
    """Compute y=x² and dy/dx through PyTorch autograd."""

    selected_device = (
        torch.device(
            device
        )
        if device is not None
        else
        resolve_device()
    )


    x = torch.tensor(
        float(
            value
        ),
        dtype=torch.float32,
        device=selected_device,
        requires_grad=True,
    )


    y = x ** 2

    y.backward()


    if x.grad is None:

        raise RuntimeError(
            "Autograd did not produce a gradient."
        )


    output = float(
        y
        .detach()
        .cpu()
        .item()
    )


    gradient = float(
        x.grad
        .detach()
        .cpu()
        .item()
    )


    return (
        output,
        gradient,
    )