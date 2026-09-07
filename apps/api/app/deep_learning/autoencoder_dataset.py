from __future__ import annotations


import torch


from torch.utils.data import (
    Dataset,
)


AUTOENCODER_DATASET_RULE_VERSION = (
    "autoencoder_dataset_v0.1"
)


class ReconstructionTensorDataset(
    Dataset,
):
    """
    CPU-owned feature-only dataset for reconstruction models.

    Each item is one feature vector. No target tensor exists.
    """

    def __init__(
        self,
        *,
        features: torch.Tensor,
    ) -> None:

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


        if features.shape[0] == 0:

            raise ValueError(
                "dataset must contain at least one row."
            )


        if features.shape[1] == 0:

            raise ValueError(
                "dataset must contain at least one feature."
            )


        if (
            features.dtype
            !=
            torch.float32
        ):

            raise TypeError(
                "features must use torch.float32."
            )


        if (
            features.device.type
            !=
            "cpu"
        ):

            raise ValueError(
                (
                    "dataset features must remain on CPU; "
                    "move batches to the execution device "
                    "explicitly."
                )
            )


        if not bool(
            torch.isfinite(
                features
            ).all()
        ):

            raise ValueError(
                (
                    "dataset features must contain only "
                    "finite values."
                )
            )


        self._features = (
            features
            .detach()
            .clone()
        )


    def __len__(
        self,
    ) -> int:

        return int(
            self._features.shape[0]
        )


    def __getitem__(
        self,
        index: int,
    ) -> torch.Tensor:

        return self._features[
            index
        ]
