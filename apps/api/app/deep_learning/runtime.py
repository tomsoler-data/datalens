from __future__ import annotations


from dataclasses import (
    dataclass,
)


import torch


DEEP_LEARNING_RUNTIME_RULE_VERSION = (
    "deep_learning_runtime_v0.1"
)


@dataclass(
    frozen=True,
)
class TorchRuntimeInfo:
    torch_version: str
    cuda_build: str | None
    cuda_available: bool
    cuda_device_count: int
    cuda_device_name: str | None
    selected_device: str


def resolve_device(
    *,
    require_cuda: bool = False,
) -> torch.device:
    """Resolve the authoritative PyTorch execution device."""

    if torch.cuda.is_available():

        return torch.device(
            "cuda"
        )


    if require_cuda:

        raise RuntimeError(
            (
                "CUDA is required for this deep-learning "
                "execution but is not available."
            )
        )


    return torch.device(
        "cpu"
    )


def get_runtime_info(
    *,
    require_cuda: bool = False,
) -> TorchRuntimeInfo:
    """Return the current PyTorch execution authority."""

    device = resolve_device(
        require_cuda=require_cuda
    )

    cuda_available = (
        torch.cuda.is_available()
    )

    cuda_device_count = (
        torch.cuda.device_count()
        if cuda_available
        else
        0
    )

    cuda_device_name = (
        torch.cuda.get_device_name(
            0
        )
        if cuda_device_count > 0
        else
        None
    )


    return TorchRuntimeInfo(
        torch_version=
            torch.__version__,

        cuda_build=
            torch.version.cuda,

        cuda_available=
            cuda_available,

        cuda_device_count=
            cuda_device_count,

        cuda_device_name=
            cuda_device_name,

        selected_device=
            str(
                device
            ),
    )


def seed_torch(
    seed: int,
) -> None:
    """Seed PyTorch deterministically on CPU and CUDA."""

    if isinstance(
        seed,
        bool,
    ):

        raise TypeError(
            "seed must be an integer."
        )


    if not isinstance(
        seed,
        int,
    ):

        raise TypeError(
            "seed must be an integer."
        )


    if seed < 0:

        raise ValueError(
            "seed must be greater than or equal to zero."
        )


    torch.manual_seed(
        seed
    )


    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            seed
        )


    torch.use_deterministic_algorithms(
        True
    )


    if hasattr(
        torch.backends,
        "cudnn",
    ):

        torch.backends.cudnn.benchmark = (
            False
        )

        torch.backends.cudnn.deterministic = (
            True
        )