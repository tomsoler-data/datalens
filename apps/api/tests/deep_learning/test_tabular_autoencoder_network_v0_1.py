from __future__ import annotations


import torch


from torch import nn


from app.deep_learning.autoencoder_network import (
    TABULAR_AUTOENCODER_NETWORK_RULE_VERSION,
    TabularAutoencoder,
)


from app.deep_learning.runtime import (
    seed_torch,
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

    return (
        torch.tensor(
            [
                [1.0, 2.0, 3.0, 4.0],
                [2.0, 3.0, 4.0, 5.0],
                [3.0, 4.0, 5.0, 6.0],
                [4.0, 5.0, 6.0, 7.0],
                [5.0, 6.0, 7.0, 8.0],
                [6.0, 7.0, 8.0, 9.0],
            ],
            dtype=
                torch.float32,
        )
    )


# ============================================================
# NETWORK STRUCTURE
# ============================================================


def test_network_structure(
) -> None:

    model = TabularAutoencoder(
        input_features=
            4,
        hidden_features=
            8,
        latent_features=
            3,
    )


    assert isinstance(
        model,
        nn.Module,
    )


    assert isinstance(
        model.encoder_input,
        nn.Linear,
    )

    assert isinstance(
        model.encoder_activation,
        nn.ReLU,
    )

    assert isinstance(
        model.latent_layer,
        nn.Linear,
    )

    assert isinstance(
        model.decoder_hidden,
        nn.Linear,
    )

    assert isinstance(
        model.decoder_activation,
        nn.ReLU,
    )

    assert isinstance(
        model.output_layer,
        nn.Linear,
    )


    assert (
        model.input_features
        ==
        4
    )

    assert (
        model.hidden_features
        ==
        8
    )

    assert (
        model.latent_features
        ==
        3
    )


# ============================================================
# ENCODE / DECODE / RECONSTRUCTION SHAPES
# ============================================================


def test_encode_decode_forward_shapes(
    device: str,
) -> None:

    model = (
        TabularAutoencoder(
            input_features=
                4,
            hidden_features=
                8,
            latent_features=
                3,
        )
        .to(
            device
        )
    )


    features = (
        build_features()
        .to(
            device
        )
    )


    latent = model.encode(
        features
    )


    assert tuple(
        latent.shape
    ) == (
        6,
        3,
    )


    assert (
        latent.dtype
        ==
        torch.float32
    )


    assert (
        latent.device.type
        ==
        device
    )


    reconstructed = model.decode(
        latent
    )


    assert tuple(
        reconstructed.shape
    ) == (
        6,
        4,
    )


    assert (
        reconstructed.dtype
        ==
        torch.float32
    )


    assert (
        reconstructed.device.type
        ==
        device
    )


    forward_result = model(
        features
    )


    assert tuple(
        forward_result.shape
    ) == (
        6,
        4,
    )


    assert torch.equal(
        reconstructed,
        forward_result,
    )


# ============================================================
# DETERMINISTIC INITIALIZATION
# ============================================================


def test_initialization_determinism(
) -> None:

    seed_torch(
        42
    )


    first = TabularAutoencoder(
        input_features=
            4,
        hidden_features=
            8,
        latent_features=
            3,
    )


    seed_torch(
        42
    )


    second = TabularAutoencoder(
        input_features=
            4,
        hidden_features=
            8,
        latent_features=
            3,
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
# GRADIENT FLOW
# ============================================================


def test_reconstruction_gradient_flow(
    device: str,
) -> None:

    seed_torch(
        42
    )


    model = (
        TabularAutoencoder(
            input_features=
                4,
            hidden_features=
                8,
            latent_features=
                3,
        )
        .to(
            device
        )
    )


    features = (
        build_features()
        .to(
            device
        )
    )


    reconstruction = model(
        features
    )


    loss = torch.mean(
        (
            reconstruction
            -
            features
        )
        **
        2
    )


    loss.backward()


    assert (
        float(
            loss.detach().cpu()
        )
        >
        0.0
    )


    parameters = list(
        model.parameters()
    )


    assert parameters


    assert all(
        parameter.grad
        is not None
        for parameter
        in parameters
    )


    assert any(
        bool(
            torch.any(
                parameter.grad
                !=
                0
            )
        )
        for parameter
        in parameters
    )


# ============================================================
# INPUT / ARCHITECTURE GUARDS
# ============================================================


def test_network_guards(
) -> None:

    expect_error(
        lambda:
            TabularAutoencoder(
                input_features=
                    0,
                hidden_features=
                    8,
                latent_features=
                    3,
            ),
        ValueError,
    )


    expect_error(
        lambda:
            TabularAutoencoder(
                input_features=
                    True,
                hidden_features=
                    8,
                latent_features=
                    3,
            ),
        TypeError,
    )


    expect_error(
        lambda:
            TabularAutoencoder(
                input_features=
                    4,
                hidden_features=
                    3,
                latent_features=
                    4,
            ),
        ValueError,
    )


    model = TabularAutoencoder(
        input_features=
            4,
        hidden_features=
            8,
        latent_features=
            3,
    )


    expect_error(
        lambda:
            model(
                torch.tensor(
                    [
                        1.0,
                        2.0,
                        3.0,
                        4.0,
                    ],
                    dtype=
                        torch.float32,
                )
            ),
        ValueError,
    )


    expect_error(
        lambda:
            model(
                torch.zeros(
                    (
                        2,
                        5,
                    ),
                    dtype=
                        torch.float32,
                )
            ),
        ValueError,
    )


    expect_error(
        lambda:
            model(
                torch.zeros(
                    (
                        2,
                        4,
                    ),
                    dtype=
                        torch.float64,
                )
            ),
        TypeError,
    )


    expect_error(
        lambda:
            model.decode(
                torch.zeros(
                    (
                        2,
                        4,
                    ),
                    dtype=
                        torch.float32,
                )
            ),
        ValueError,
    )


# ============================================================
# NO TARGET SEMANTICS
# ============================================================


def test_no_target_authority(
) -> None:

    model = TabularAutoencoder(
        input_features=
            4,
        hidden_features=
            8,
        latent_features=
            3,
    )


    assert not hasattr(
        model,
        "target",
    )

    assert not hasattr(
        model,
        "target_column",
    )


# ============================================================
# RULE VERSION
# ============================================================


def test_rule_version(
) -> None:

    assert (
        TABULAR_AUTOENCODER_NETWORK_RULE_VERSION
        ==
        "tabular_autoencoder_network_v0.1"
    )


# ============================================================
# DIRECT ACCEPTANCE
# ============================================================


def main(
) -> None:

    print(
        "=== DATALENS TABULAR AUTOENCODER NETWORK v0.1 ==="
    )

    print()


    test_network_structure()

    print(
        "Encoder / latent / decoder structure: PASS"
    )


    test_encode_decode_forward_shapes(
        "cpu"
    )

    print(
        "CPU latent / reconstruction shapes: PASS"
    )


    if torch.cuda.is_available():

        test_encode_decode_forward_shapes(
            "cuda"
        )

        print(
            "CUDA latent / reconstruction shapes: PASS"
        )

    else:

        raise RuntimeError(
            (
                "DL-4 acceptance requires the "
                "validated CUDA environment."
            )
        )


    test_initialization_determinism()

    print(
        "Deterministic initialization: PASS"
    )


    test_reconstruction_gradient_flow(
        "cpu"
    )

    print(
        "CPU reconstruction gradient flow: PASS"
    )


    test_reconstruction_gradient_flow(
        "cuda"
    )

    print(
        "CUDA reconstruction gradient flow: PASS"
    )


    test_network_guards()

    print(
        "Network guards: PASS"
    )


    test_no_target_authority()

    print(
        "Target-free network semantics: PASS"
    )


    test_rule_version()

    print(
        "Network rule version: PASS"
    )


    print()

    print(
        "PASS - DataLens Tabular Autoencoder Network v0.1"
    )


if __name__ == "__main__":

    main()
