from __future__ import annotations


import math


import torch


from app.deep_learning.autoencoder_dataset import (
    AUTOENCODER_DATASET_RULE_VERSION,
    ReconstructionTensorDataset,
)


from app.deep_learning.autoencoder_evaluation import (
    AUTOENCODER_EVALUATION_RULE_VERSION,
    ReconstructionEvaluation,
    evaluate_reconstruction_loader,
)


from app.deep_learning.autoencoder_network import (
    TabularAutoencoder,
)


from app.deep_learning.autoencoder_training import (
    AUTOENCODER_TRAINING_RULE_VERSION,
    train_reconstruction_batch,
    train_reconstruction_epochs,
)


from app.deep_learning.datasets import (
    build_data_loader,
)


from app.deep_learning.runtime import (
    seed_torch,
)


from app.deep_learning.training import (
    build_regression_loss,
    build_sgd_optimizer,
)


def expect_error(
    callback,
    expected_exception,
) -> None:

    try:

        callback()

    except expected_exception:
        return


    raise AssertionError(
        (
            "Expected exception: "
            +
            expected_exception.__name__
        )
    )


def build_features(
) -> torch.Tensor:

    rows = []


    for value in range(
        1,
        17,
    ):

        x = float(
            value
        ) / 16.0

        rows.append(
            [
                x,
                2.0 * x,
                x + 0.25,
                2.0 * x + 0.25,
            ]
        )


    return torch.tensor(
        rows,
        dtype=
            torch.float32,
    )


def build_loader(
    *,
    shuffle: bool,
):

    dataset = (
        ReconstructionTensorDataset(
            features=
                build_features(),
        )
    )


    return build_data_loader(
        dataset,
        batch_size=
            4,
        shuffle=
            shuffle,
        seed=
            42,
    )


def build_model(
    *,
    device: str,
) -> TabularAutoencoder:

    seed_torch(
        42
    )


    return (
        TabularAutoencoder(
            input_features=
                4,
            hidden_features=
                8,
            latent_features=
                2,
        )
        .to(
            device
        )
    )


# ============================================================
# FEATURE-ONLY DATASET
# ============================================================


def test_reconstruction_dataset(
) -> None:

    source = build_features()


    dataset = (
        ReconstructionTensorDataset(
            features=
                source,
        )
    )


    assert (
        len(
            dataset
        )
        ==
        16
    )


    item = dataset[
        0
    ]


    assert isinstance(
        item,
        torch.Tensor,
    )


    assert tuple(
        item.shape
    ) == (
        4,
    )


    assert (
        item.device.type
        ==
        "cpu"
    )


    source[
        0,
        0,
    ] = 999.0


    assert (
        float(
            dataset[
                0
            ][
                0
            ]
        )
        !=
        999.0
    )


    expect_error(
        lambda:
            ReconstructionTensorDataset(
                features=
                    torch.tensor(
                        [
                            [
                                1.0,
                                math.nan,
                            ],
                        ],
                        dtype=
                            torch.float32,
                    )
            ),
        ValueError,
    )


# ============================================================
# SINGLE BATCH
# ============================================================


def test_single_batch_training(
    device: str,
) -> None:

    model = build_model(
        device=
            device,
    )


    optimizer = (
        build_sgd_optimizer(
            model,
            learning_rate=
                0.05,
        )
    )


    loss_function = (
        build_regression_loss()
    )


    features = (
        build_features()[
            :4
        ]
    )


    before = [
        parameter
        .detach()
        .clone()

        for parameter
        in model.parameters()
    ]


    loss = (
        train_reconstruction_batch(
            model,
            features=
                features,
            optimizer=
                optimizer,
            loss_function=
                loss_function,
            device=
                device,
        )
    )


    after = [
        parameter
        .detach()
        .clone()

        for parameter
        in model.parameters()
    ]


    assert (
        loss
        >
        0.0
    )


    assert (
        model.training
        is True
    )


    assert any(
        not torch.equal(
            first,
            second,
        )

        for first, second
        in zip(
            before,
            after,
        )
    )


# ============================================================
# EPOCH TRAINING + LOSS REDUCTION
# ============================================================


def train_model(
    *,
    device: str,
):

    model = build_model(
        device=
            device,
    )


    optimizer = (
        build_sgd_optimizer(
            model,
            learning_rate=
                0.05,
        )
    )


    loss_function = (
        build_regression_loss()
    )


    losses = (
        train_reconstruction_epochs(
            model,
            data_loader=
                build_loader(
                    shuffle=True,
                ),
            optimizer=
                optimizer,
            loss_function=
                loss_function,
            epochs=
                120,
            device=
                device,
        )
    )


    return (
        model,
        losses,
    )


def test_epoch_training(
    device: str,
) -> None:

    model, losses = (
        train_model(
            device=
                device,
        )
    )


    assert (
        len(
            losses
        )
        ==
        120
    )


    assert all(
        math.isfinite(
            loss
        )
        for loss
        in losses
    )


    assert (
        losses[
            -1
        ]
        <
        losses[
            0
        ]
    )


    assert (
        losses[
            -1
        ]
        <
        (
            losses[
                0
            ]
            *
            0.50
        )
    )


    assert (
        model.training
        is True
    )


# ============================================================
# EVALUATION / ROW ERRORS
# ============================================================


def test_evaluation(
    device: str,
) -> None:

    model, _ = (
        train_model(
            device=
                device,
        )
    )


    loader = build_loader(
        shuffle=False,
    )


    before = [
        parameter
        .detach()
        .clone()

        for parameter
        in model.parameters()
    ]


    result = (
        evaluate_reconstruction_loader(
            model,
            data_loader=
                loader,
            device=
                device,
        )
    )


    after = [
        parameter
        .detach()
        .clone()

        for parameter
        in model.parameters()
    ]


    assert isinstance(
        result,
        ReconstructionEvaluation,
    )


    assert (
        result.mean_loss
        >=
        0.0
    )


    assert (
        result.reconstruction_errors.shape
        ==
        torch.Size(
            [
                16,
            ]
        )
    )


    assert (
        result.reconstruction_errors.dtype
        ==
        torch.float32
    )


    assert (
        result.reconstruction_errors.device.type
        ==
        "cpu"
    )


    assert (
        result.reconstruction_errors.requires_grad
        is False
    )


    assert bool(
        torch.isfinite(
            result.reconstruction_errors
        ).all()
    )


    assert abs(
        result.mean_loss
        -
        float(
            result.reconstruction_errors
            .mean()
            .item()
        )
    ) < 1e-8


    assert (
        model.training
        is False
    )


    assert all(
        torch.equal(
            first,
            second,
        )

        for first, second
        in zip(
            before,
            after,
        )
    )


# ============================================================
# ERROR ORDER IS DATA-LOADER ORDER
# ============================================================


def test_evaluation_order(
) -> None:

    model = build_model(
        device=
            "cpu",
    )


    source = build_features()


    full_loader = build_data_loader(
        ReconstructionTensorDataset(
            features=
                source,
        ),
        batch_size=
            16,
        shuffle=
            False,
        seed=
            42,
    )


    result = (
        evaluate_reconstruction_loader(
            model,
            data_loader=
                full_loader,
            device=
                "cpu",
        )
    )


    model.eval()


    with torch.inference_mode():

        reconstruction = model(
            source
        )


        expected = (
            (
                reconstruction
                -
                source
            )
            .pow(
                2
            )
            .mean(
                dim=1
            )
            .cpu()
        )


    assert torch.equal(
        result.reconstruction_errors,
        expected,
    )


# ============================================================
# NO SUPERVISED BATCH SEMANTICS
# ============================================================


def test_feature_only_loader(
) -> None:

    batch = next(
        iter(
            build_loader(
                shuffle=False,
            )
        )
    )


    assert isinstance(
        batch,
        torch.Tensor,
    )


    assert (
        batch.ndim
        ==
        2
    )


# ============================================================
# GUARDS
# ============================================================


def test_training_guards(
) -> None:

    model = build_model(
        device=
            "cpu",
    )


    optimizer = build_sgd_optimizer(
        model,
        learning_rate=
            0.01,
    )


    loss_function = (
        build_regression_loss()
    )


    expect_error(
        lambda:
            train_reconstruction_epochs(
                model,
                data_loader=
                    build_loader(
                        shuffle=False,
                    ),
                optimizer=
                    optimizer,
                loss_function=
                    loss_function,
                epochs=
                    0,
                device=
                    "cpu",
            ),
        ValueError,
    )


    expect_error(
        lambda:
            train_reconstruction_batch(
                model,
                features=
                    torch.tensor(
                        [
                            [
                                1.0,
                                math.inf,
                                2.0,
                                3.0,
                            ]
                        ],
                        dtype=
                            torch.float32,
                    ),
                optimizer=
                    optimizer,
                loss_function=
                    loss_function,
                device=
                    "cpu",
            ),
        ValueError,
    )


# ============================================================
# RULE VERSIONS
# ============================================================


def test_rule_versions(
) -> None:

    assert (
        AUTOENCODER_DATASET_RULE_VERSION
        ==
        "autoencoder_dataset_v0.1"
    )


    assert (
        AUTOENCODER_TRAINING_RULE_VERSION
        ==
        "autoencoder_training_v0.1"
    )


    assert (
        AUTOENCODER_EVALUATION_RULE_VERSION
        ==
        "autoencoder_evaluation_v0.1"
    )


# ============================================================
# DIRECT ACCEPTANCE
# ============================================================


def main(
) -> None:

    print(
        "=== DATALENS TABULAR AUTOENCODER RECONSTRUCTION v0.1 ==="
    )

    print()


    test_reconstruction_dataset()

    print(
        "Feature-only CPU dataset: PASS"
    )


    test_feature_only_loader()

    print(
        "Target-free DataLoader batches: PASS"
    )


    test_single_batch_training(
        "cpu"
    )

    print(
        "CPU reconstruction backward / optimizer: PASS"
    )


    if not torch.cuda.is_available():

        raise RuntimeError(
            (
                "DL-4 acceptance requires the "
                "validated CUDA environment."
            )
        )


    test_single_batch_training(
        "cuda"
    )

    print(
        "CUDA reconstruction backward / optimizer: PASS"
    )


    test_epoch_training(
        "cpu"
    )

    print(
        "CPU reconstruction loss reduction: PASS"
    )


    test_epoch_training(
        "cuda"
    )

    print(
        "CUDA reconstruction loss reduction: PASS"
    )


    test_evaluation(
        "cpu"
    )

    print(
        "CPU per-row reconstruction errors: PASS"
    )


    test_evaluation(
        "cuda"
    )

    print(
        "CUDA per-row reconstruction errors: PASS"
    )


    test_evaluation_order()

    print(
        "Reconstruction error row ordering: PASS"
    )


    test_training_guards()

    print(
        "Reconstruction training guards: PASS"
    )


    test_rule_versions()

    print(
        "Reconstruction rule versions: PASS"
    )


    print()

    print(
        "PASS - DataLens Tabular Autoencoder Reconstruction v0.1"
    )


if __name__ == "__main__":

    main()
