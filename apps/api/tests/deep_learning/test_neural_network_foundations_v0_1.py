from __future__ import annotations


import torch


from torch import nn


from app.deep_learning.datasets import (
    TabularTensorDataset,
    build_data_loader,
)


from app.deep_learning.evaluation import (
    DEEP_LEARNING_EVALUATION_RULE_VERSION,
    RegressionEvaluation,
    evaluate_regression_loader,
)


from app.deep_learning.networks import (
    DEEP_LEARNING_NETWORK_RULE_VERSION,
    FeedForwardRegressor,
)


from app.deep_learning.runtime import (
    seed_torch,
)


from app.deep_learning.training import (
    DEEP_LEARNING_TRAINING_RULE_VERSION,
    build_regression_loss,
    build_sgd_optimizer,
    train_regression_batch,
    train_regression_epochs,
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
            + expected_exception.__name__
        )
    )


def build_foundation_data(
) -> tuple[
    torch.Tensor,
    torch.Tensor,
]:

    features = torch.tensor(
        [
            [1.0, 1.0],
            [1.0, 2.0],
            [2.0, 1.0],
            [2.0, 2.0],
            [3.0, 1.0],
            [3.0, 2.0],
            [4.0, 1.0],
            [4.0, 2.0],
        ],
        dtype=torch.float32,
    )

    targets = torch.tensor(
        [
            3.0,
            5.0,
            4.0,
            6.0,
            5.0,
            7.0,
            6.0,
            8.0,
        ],
        dtype=torch.float32,
    )


    return (
        features,
        targets,
    )


def build_foundation_loader(
):
    features, targets = (
        build_foundation_data()
    )

    dataset = TabularTensorDataset(
        features=features,
        targets=targets,
    )


    return build_data_loader(
        dataset,
        batch_size=2,
        shuffle=False,
        seed=42,
    )


# ============================================================
# NETWORK
# ============================================================


def test_network_structure_and_forward(
) -> None:

    seed_torch(
        42
    )

    model = FeedForwardRegressor(
        input_features=2,
        hidden_features=8,
    )


    assert isinstance(
        model,
        nn.Module,
    )

    assert isinstance(
        model.input_layer,
        nn.Linear,
    )

    assert isinstance(
        model.activation,
        nn.ReLU,
    )

    assert isinstance(
        model.output_layer,
        nn.Linear,
    )


    features, _ = (
        build_foundation_data()
    )

    predictions = model(
        features
    )


    assert tuple(
        predictions.shape
    ) == (8,)

    assert (
        predictions.dtype
        ==
        torch.float32
    )

    assert (
        predictions.device.type
        ==
        "cpu"
    )


def test_network_initialization_determinism(
) -> None:

    seed_torch(
        42
    )

    first = FeedForwardRegressor(
        input_features=2,
        hidden_features=8,
    )


    seed_torch(
        42
    )

    second = FeedForwardRegressor(
        input_features=2,
        hidden_features=8,
    )


    for (
        first_parameter,
        second_parameter,
    ) in zip(
        first.parameters(),
        second.parameters(),
    ):

        assert torch.equal(
            first_parameter,
            second_parameter,
        )


# ============================================================
# SINGLE BATCH TRAINING
# ============================================================


def _assert_single_batch_training(
    device: str,
) -> None:

    features, targets = (
        build_foundation_data()
    )


    seed_torch(
        42
    )

    model = FeedForwardRegressor(
        input_features=2,
        hidden_features=8,
    ).to(
        device
    )


    optimizer = build_sgd_optimizer(
        model,
        learning_rate=0.01,
    )

    loss_function = (
        build_regression_loss()
    )


    before = [
        parameter.detach().clone()
        for parameter in model.parameters()
    ]


    loss = train_regression_batch(
        model,
        features=features,
        targets=targets,
        optimizer=optimizer,
        loss_function=loss_function,
        device=device,
    )


    after = [
        parameter.detach().clone()
        for parameter in model.parameters()
    ]


    assert loss > 0.0

    assert model.training is True


    assert any(
        parameter.grad is not None
        for parameter in model.parameters()
    )


    assert any(
        not torch.equal(
            before_parameter,
            after_parameter,
        )
        for (
            before_parameter,
            after_parameter,
        ) in zip(
            before,
            after,
        )
    )


def test_single_batch_training_cpu(
) -> None:

    _assert_single_batch_training(
        "cpu"
    )


def test_single_batch_training_cuda(
) -> None:

    _assert_single_batch_training(
        "cuda"
    )


# ============================================================
# EPOCH TRAINING
# ============================================================


def _train_epochs(
    device: str,
) -> tuple[
    FeedForwardRegressor,
    list[float],
]:

    loader = (
        build_foundation_loader()
    )


    seed_torch(
        42
    )

    model = FeedForwardRegressor(
        input_features=2,
        hidden_features=8,
    ).to(
        device
    )


    optimizer = build_sgd_optimizer(
        model,
        learning_rate=0.01,
    )

    loss_function = (
        build_regression_loss()
    )


    losses = train_regression_epochs(
        model,
        data_loader=loader,
        optimizer=optimizer,
        loss_function=loss_function,
        epochs=40,
        device=device,
    )


    return (
        model,
        losses,
    )


def test_epoch_training_cpu_and_cuda(
) -> None:

    cpu_model, cpu_losses = (
        _train_epochs(
            "cpu"
        )
    )

    cuda_model, cuda_losses = (
        _train_epochs(
            "cuda"
        )
    )


    assert len(
        cpu_losses
    ) == 40

    assert len(
        cuda_losses
    ) == 40


    assert (
        cpu_losses[-1]
        <
        cpu_losses[0]
    )

    assert (
        cuda_losses[-1]
        <
        cuda_losses[0]
    )


    assert cpu_model.training is True
    assert cuda_model.training is True


    assert torch.allclose(
        torch.tensor(
            cpu_losses
        ),
        torch.tensor(
            cuda_losses
        ),
        atol=1e-5,
        rtol=1e-5,
    )


# ============================================================
# EVALUATION
# ============================================================


def _assert_evaluation(
    device: str,
) -> RegressionEvaluation:

    loader = (
        build_foundation_loader()
    )


    model, _ = (
        _train_epochs(
            device
        )
    )


    loss_function = (
        build_regression_loss()
    )


    before = [
        parameter.detach().clone()
        for parameter in model.parameters()
    ]


    result = evaluate_regression_loader(
        model,
        data_loader=loader,
        loss_function=loss_function,
        device=device,
    )


    after = [
        parameter.detach().clone()
        for parameter in model.parameters()
    ]


    assert isinstance(
        result,
        RegressionEvaluation,
    )

    assert result.mean_loss >= 0.0


    assert tuple(
        result.predictions.shape
    ) == (8,)

    assert tuple(
        result.targets.shape
    ) == (8,)


    assert (
        result.predictions.device.type
        ==
        "cpu"
    )

    assert (
        result.targets.device.type
        ==
        "cpu"
    )


    assert (
        result.predictions.requires_grad
        is False
    )


    assert model.training is False


    assert all(
        torch.equal(
            before_parameter,
            after_parameter,
        )
        for (
            before_parameter,
            after_parameter,
        ) in zip(
            before,
            after,
        )
    )


    return result


def test_evaluation_cpu_and_cuda(
) -> None:

    cpu_result = (
        _assert_evaluation(
            "cpu"
        )
    )

    cuda_result = (
        _assert_evaluation(
            "cuda"
        )
    )


    assert torch.allclose(
        cpu_result.predictions,
        cuda_result.predictions,
        atol=1e-5,
        rtol=1e-5,
    )


    assert abs(
        cpu_result.mean_loss
        -
        cuda_result.mean_loss
    ) < 1e-5


# ============================================================
# GUARDS
# ============================================================


def test_network_guards(
) -> None:

    expect_error(
        lambda: FeedForwardRegressor(
            input_features=0,
            hidden_features=8,
        ),
        ValueError,
    )


    expect_error(
        lambda: FeedForwardRegressor(
            input_features=True,
            hidden_features=8,
        ),
        TypeError,
    )


    model = FeedForwardRegressor(
        input_features=2,
        hidden_features=8,
    )


    expect_error(
        lambda: model(
            torch.tensor(
                [1.0, 2.0],
                dtype=torch.float32,
            )
        ),
        ValueError,
    )


    expect_error(
        lambda: model(
            torch.tensor(
                [
                    [1.0, 2.0, 3.0],
                ],
                dtype=torch.float32,
            )
        ),
        ValueError,
    )


def test_training_guards(
) -> None:

    model = FeedForwardRegressor(
        input_features=2,
        hidden_features=8,
    )


    expect_error(
        lambda: build_sgd_optimizer(
            model,
            learning_rate=0,
        ),
        ValueError,
    )


    expect_error(
        lambda: build_sgd_optimizer(
            model,
            learning_rate=True,
        ),
        TypeError,
    )


    loader = (
        build_foundation_loader()
    )

    optimizer = build_sgd_optimizer(
        model,
        learning_rate=0.01,
    )

    loss_function = (
        build_regression_loss()
    )


    expect_error(
        lambda: train_regression_epochs(
            model,
            data_loader=loader,
            optimizer=optimizer,
            loss_function=loss_function,
            epochs=0,
            device="cpu",
        ),
        ValueError,
    )


    expect_error(
        lambda: train_regression_epochs(
            model,
            data_loader=loader,
            optimizer=optimizer,
            loss_function=loss_function,
            epochs=True,
            device="cpu",
        ),
        TypeError,
    )


    expect_error(
        lambda: train_regression_epochs(
            model,
            data_loader="not-a-loader",
            optimizer=optimizer,
            loss_function=loss_function,
            epochs=1,
            device="cpu",
        ),
        TypeError,
    )


def test_device_authority_guard(
) -> None:

    features, targets = (
        build_foundation_data()
    )


    model = FeedForwardRegressor(
        input_features=2,
        hidden_features=8,
    )


    optimizer = build_sgd_optimizer(
        model,
        learning_rate=0.01,
    )

    loss_function = (
        build_regression_loss()
    )


    expect_error(
        lambda: train_regression_batch(
            model,
            features=features,
            targets=targets,
            optimizer=optimizer,
            loss_function=loss_function,
            device="cuda",
        ),
        ValueError,
    )


# ============================================================
# RULE VERSIONS
# ============================================================


def test_rule_versions(
) -> None:

    assert (
        DEEP_LEARNING_NETWORK_RULE_VERSION
        ==
        "deep_learning_network_v0.1"
    )

    assert (
        DEEP_LEARNING_TRAINING_RULE_VERSION
        ==
        "deep_learning_training_v0.1"
    )

    assert (
        DEEP_LEARNING_EVALUATION_RULE_VERSION
        ==
        "deep_learning_evaluation_v0.1"
    )


# ============================================================
# DIRECT ACCEPTANCE RUNNER
# ============================================================


def main(
) -> None:

    print(
        "=== DATALENS NEURAL NETWORK FOUNDATIONS v0.1 ==="
    )

    print()


    test_network_structure_and_forward()

    print(
        "nn.Module / Linear / ReLU: PASS"
    )


    test_network_initialization_determinism()

    print(
        "Deterministic initialization: PASS"
    )


    test_single_batch_training_cpu()

    print(
        "CPU forward / loss / backward / optimizer: PASS"
    )


    test_single_batch_training_cuda()

    print(
        "CUDA forward / loss / backward / optimizer: PASS"
    )


    test_epoch_training_cpu_and_cuda()

    print(
        "CPU / CUDA epoch training: PASS"
    )

    print(
        "Loss reduction across epochs: PASS"
    )


    test_evaluation_cpu_and_cuda()

    print(
        "train() / eval() separation: PASS"
    )

    print(
        "Inference without gradients: PASS"
    )

    print(
        "Evaluation preserves parameters: PASS"
    )

    print(
        "CPU / CUDA evaluation consistency: PASS"
    )


    test_network_guards()

    print(
        "Network guards: PASS"
    )


    test_training_guards()

    print(
        "Training guards: PASS"
    )


    test_device_authority_guard()

    print(
        "Execution device authority: PASS"
    )


    test_rule_versions()

    print(
        "Rule versions: PASS"
    )


    print()

    print(
        "PASS - DataLens Neural Network Foundations v0.1"
    )


if __name__ == "__main__":

    main()