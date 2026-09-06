from __future__ import annotations


import torch


from app.deep_learning.datasets import (
    DEEP_LEARNING_DATASET_RULE_VERSION,
    TabularTensorDataset,
    build_data_loader,
)


from app.deep_learning.runtime import (
    DEEP_LEARNING_RUNTIME_RULE_VERSION,
    get_runtime_info,
    resolve_device,
    seed_torch,
)


from app.deep_learning.tensors import (
    DEEP_LEARNING_TENSOR_RULE_VERSION,
    build_float_tensor,
    move_tensor,
    scalar_square_gradient,
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


# ============================================================
# RUNTIME
# ============================================================


def test_cuda_runtime_authority(
) -> None:

    info = get_runtime_info(
        require_cuda=True
    )

    assert info.cuda_available is True
    assert info.cuda_device_count >= 1
    assert info.cuda_device_name is not None
    assert info.selected_device == "cuda"

    device = resolve_device(
        require_cuda=True
    )

    assert device.type == "cuda"


# ============================================================
# DETERMINISM
# ============================================================


def test_cpu_seed_determinism(
) -> None:

    seed_torch(42)

    first = torch.rand(8)

    seed_torch(42)

    second = torch.rand(8)

    assert torch.equal(
        first,
        second,
    )


def test_cuda_seed_determinism(
) -> None:

    seed_torch(42)

    first = torch.rand(
        8,
        device="cuda",
    )

    seed_torch(42)

    second = torch.rand(
        8,
        device="cuda",
    )

    assert torch.equal(
        first,
        second,
    )


# ============================================================
# TENSORS
# ============================================================


def test_tensor_shape_dtype_and_device(
) -> None:

    tensor = build_float_tensor(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ]
    )

    assert tuple(tensor.shape) == (2, 2)
    assert tensor.dtype == torch.float32
    assert tensor.device.type == "cpu"


def test_tensor_cpu_cuda_round_trip(
) -> None:

    cpu_tensor = build_float_tensor(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ]
    )

    cuda_tensor = move_tensor(
        cpu_tensor,
        device="cuda",
    )

    returned = move_tensor(
        cuda_tensor,
        device="cpu",
    )

    assert cuda_tensor.device.type == "cuda"
    assert returned.device.type == "cpu"

    assert torch.equal(
        returned,
        cpu_tensor,
    )


# ============================================================
# AUTOGRAD
# ============================================================


def test_autograd_cpu_and_cuda(
) -> None:

    cpu_output, cpu_gradient = (
        scalar_square_gradient(
            3.0,
            device="cpu",
        )
    )

    cuda_output, cuda_gradient = (
        scalar_square_gradient(
            3.0,
            device="cuda",
        )
    )

    assert cpu_output == 9.0
    assert cpu_gradient == 6.0

    assert cuda_output == 9.0
    assert cuda_gradient == 6.0


# ============================================================
# DATASET
# ============================================================


def test_tabular_dataset_authority(
) -> None:

    features = torch.tensor(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ],
        dtype=torch.float32,
    )

    targets = torch.tensor(
        [
            10.0,
            20.0,
            30.0,
        ],
        dtype=torch.float32,
    )

    dataset = TabularTensorDataset(
        features=features,
        targets=targets,
    )

    feature_row, target = dataset[1]

    assert len(dataset) == 3
    assert tuple(feature_row.shape) == (2,)
    assert float(target.item()) == 20.0
    assert feature_row.device.type == "cpu"
    assert target.device.type == "cpu"


# ============================================================
# DATALOADER
# ============================================================


def test_dataloader_batch_authority(
) -> None:

    features = torch.tensor(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0],
        ],
        dtype=torch.float32,
    )

    targets = torch.tensor(
        [
            10.0,
            20.0,
            30.0,
            40.0,
        ],
        dtype=torch.float32,
    )

    dataset = TabularTensorDataset(
        features=features,
        targets=targets,
    )

    loader = build_data_loader(
        dataset,
        batch_size=2,
        shuffle=False,
        seed=42,
    )

    batch_features, batch_targets = (
        next(
            iter(loader)
        )
    )

    assert tuple(batch_features.shape) == (2, 2)
    assert tuple(batch_targets.shape) == (2,)

    assert batch_features.device.type == "cpu"
    assert batch_targets.device.type == "cpu"


def test_dataloader_shuffle_is_deterministic(
) -> None:

    features = torch.arange(
        16,
        dtype=torch.float32,
    ).reshape(
        8,
        2,
    )

    targets = torch.arange(
        8,
        dtype=torch.int64,
    )

    dataset = TabularTensorDataset(
        features=features,
        targets=targets,
    )

    first_loader = build_data_loader(
        dataset,
        batch_size=2,
        shuffle=True,
        seed=42,
    )

    second_loader = build_data_loader(
        dataset,
        batch_size=2,
        shuffle=True,
        seed=42,
    )

    first_order = torch.cat(
        [
            batch_targets
            for _, batch_targets
            in first_loader
        ]
    )

    second_order = torch.cat(
        [
            batch_targets
            for _, batch_targets
            in second_loader
        ]
    )

    assert torch.equal(
        first_order,
        second_order,
    )


# ============================================================
# GUARDS
# ============================================================


def test_seed_guards(
) -> None:

    expect_error(
        lambda: seed_torch(True),
        TypeError,
    )

    expect_error(
        lambda: seed_torch(-1),
        ValueError,
    )


def test_tensor_shape_guards(
) -> None:

    expect_error(
        lambda: build_float_tensor([]),
        ValueError,
    )

    expect_error(
        lambda: build_float_tensor(
            [1.0, 2.0]
        ),
        ValueError,
    )

    expect_error(
        lambda: build_float_tensor(
            [
                [1.0, 2.0],
                [3.0],
            ]
        ),
        ValueError,
    )


def test_dataset_guards(
) -> None:

    expect_error(
        lambda: TabularTensorDataset(
            features=torch.empty(
                (0, 2),
                dtype=torch.float32,
            ),
            targets=torch.empty(
                (0,),
                dtype=torch.float32,
            ),
        ),
        ValueError,
    )

    expect_error(
        lambda: TabularTensorDataset(
            features=torch.tensor(
                [
                    [1.0],
                    [2.0],
                ],
                dtype=torch.float32,
            ),
            targets=torch.tensor(
                [1.0],
                dtype=torch.float32,
            ),
        ),
        ValueError,
    )


def test_cuda_dataset_is_blocked(
) -> None:

    features = torch.tensor(
        [
            [1.0, 2.0],
        ],
        dtype=torch.float32,
        device="cuda",
    )

    targets = torch.tensor(
        [
            1.0,
        ],
        dtype=torch.float32,
        device="cuda",
    )

    expect_error(
        lambda: TabularTensorDataset(
            features=features,
            targets=targets,
        ),
        ValueError,
    )


def test_loader_parameter_guards(
) -> None:

    dataset = TabularTensorDataset(
        features=torch.tensor(
            [
                [1.0],
            ],
            dtype=torch.float32,
        ),
        targets=torch.tensor(
            [
                1.0,
            ],
            dtype=torch.float32,
        ),
    )

    expect_error(
        lambda: build_data_loader(
            dataset,
            batch_size=0,
            shuffle=False,
            seed=42,
        ),
        ValueError,
    )

    expect_error(
        lambda: build_data_loader(
            dataset,
            batch_size=1,
            shuffle=False,
            seed=True,
        ),
        TypeError,
    )


# ============================================================
# RULE VERSIONS
# ============================================================


def test_rule_versions(
) -> None:

    assert (
        DEEP_LEARNING_RUNTIME_RULE_VERSION
        ==
        "deep_learning_runtime_v0.1"
    )

    assert (
        DEEP_LEARNING_TENSOR_RULE_VERSION
        ==
        "deep_learning_tensor_v0.1"
    )

    assert (
        DEEP_LEARNING_DATASET_RULE_VERSION
        ==
        "deep_learning_dataset_v0.1"
    )


# ============================================================
# MAIN
# ============================================================


def main(
) -> None:

    print(
        "=== DATALENS PYTORCH FOUNDATIONS v0.1 ==="
    )

    print()

    test_cuda_runtime_authority()
    print("CUDA runtime authority: PASS")

    test_cpu_seed_determinism()
    print("CPU seed determinism: PASS")

    test_cuda_seed_determinism()
    print("CUDA seed determinism: PASS")

    test_tensor_shape_dtype_and_device()
    print("Tensor shape / dtype / device: PASS")

    test_tensor_cpu_cuda_round_trip()
    print("Tensor CPU / CUDA round trip: PASS")

    test_autograd_cpu_and_cuda()
    print("CPU / CUDA autograd: PASS")

    test_tabular_dataset_authority()
    print("Tabular dataset authority: PASS")

    test_dataloader_batch_authority()
    print("DataLoader batch authority: PASS")

    test_dataloader_shuffle_is_deterministic()
    print("Deterministic DataLoader shuffle: PASS")

    test_seed_guards()
    print("Seed guards: PASS")

    test_tensor_shape_guards()
    print("Tensor shape guards: PASS")

    test_dataset_guards()
    print("Dataset guards: PASS")

    test_cuda_dataset_is_blocked()
    print("CPU dataset ownership guard: PASS")

    test_loader_parameter_guards()
    print("DataLoader parameter guards: PASS")

    test_rule_versions()
    print("Rule versions: PASS")

    print()
    print(
        "PASS - DataLens PyTorch Foundations v0.1"
    )


if __name__ == "__main__":

    main()