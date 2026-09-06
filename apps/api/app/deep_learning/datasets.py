from __future__ import annotations


import torch


from torch.utils.data import (
    DataLoader,
    Dataset,
)


DEEP_LEARNING_DATASET_RULE_VERSION = (
    "deep_learning_dataset_v0.1"
)


class TabularTensorDataset(
    Dataset,
):
    """CPU-owned tabular tensor dataset.

    Features remain on CPU while data is loaded.

    Device transfer belongs to the training or inference
    execution layer, where batches are moved explicitly
    to the selected PyTorch device.
    """

    def __init__(
        self,
        *,
        features: torch.Tensor,
        targets: torch.Tensor,
    ) -> None:

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


        if features.shape[0] == 0:

            raise ValueError(
                "dataset must contain at least one row."
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


        if features.device.type != "cpu":

            raise ValueError(
                (
                    "dataset features must remain on CPU; "
                    "move batches to the execution device "
                    "explicitly."
                )
            )


        if targets.device.type != "cpu":

            raise ValueError(
                (
                    "dataset targets must remain on CPU; "
                    "move batches to the execution device "
                    "explicitly."
                )
            )


        self._features = features
        self._targets = targets


    def __len__(
        self,
    ) -> int:

        return int(
            self._features.shape[0]
        )


    def __getitem__(
        self,
        index: int,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:

        return (
            self._features[index],
            self._targets[index],
        )


def build_data_loader(
    dataset: Dataset,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    drop_last: bool = False,
) -> DataLoader:
    """Build a deterministic single-process DataLoader."""

    if not isinstance(
        dataset,
        Dataset,
    ):

        raise TypeError(
            "dataset must be a torch.utils.data.Dataset."
        )


    if (
        isinstance(
            batch_size,
            bool,
        )
        or
        not isinstance(
            batch_size,
            int,
        )
    ):

        raise TypeError(
            "batch_size must be an integer."
        )


    if batch_size <= 0:

        raise ValueError(
            "batch_size must be greater than zero."
        )


    if not isinstance(
        shuffle,
        bool,
    ):

        raise TypeError(
            "shuffle must be a boolean."
        )


    if (
        isinstance(
            seed,
            bool,
        )
        or
        not isinstance(
            seed,
            int,
        )
    ):

        raise TypeError(
            "seed must be an integer."
        )


    if seed < 0:

        raise ValueError(
            "seed must be greater than or equal to zero."
        )


    if not isinstance(
        drop_last,
        bool,
    ):

        raise TypeError(
            "drop_last must be a boolean."
        )


    generator = torch.Generator(
        device="cpu"
    )

    generator.manual_seed(
        seed
    )


    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=0,
        generator=generator,
    )