from __future__ import annotations


from pathlib import (
    Path,
)


from pydantic import (
    ValidationError,
)


from app.deep_learning.tabular_contracts import (
    DLTabularMLPRegressorHyperparameters,
    DL_TABULAR_MLP_CONTRACT_RULE_VERSION,
)


# ============================================================
# HELPERS
# ============================================================


def expect_validation_error(
    callback,
) -> None:

    try:
        callback()

    except ValidationError:
        return


    raise AssertionError(
        "Expected Pydantic ValidationError."
    )


# ============================================================
# DEFAULT CONTRACT
# ============================================================


def test_default_contract(
) -> None:

    contract = (
        DLTabularMLPRegressorHyperparameters()
    )


    assert (
        contract.kind
        ==
        "tabular_mlp_regressor"
    )


    assert (
        contract.hidden_features
        ==
        64
    )


    assert (
        contract.epochs
        ==
        100
    )


    assert (
        contract.batch_size
        ==
        64
    )


    assert (
        contract.learning_rate
        ==
        0.01
    )


# ============================================================
# EXPLICIT CONFIGURATION
# ============================================================


def test_explicit_contract(
) -> None:

    contract = (
        DLTabularMLPRegressorHyperparameters(
            hidden_features=128,
            epochs=250,
            batch_size=32,
            learning_rate=0.005,
        )
    )


    assert (
        contract.hidden_features
        ==
        128
    )


    assert (
        contract.epochs
        ==
        250
    )


    assert (
        contract.batch_size
        ==
        32
    )


    assert (
        contract.learning_rate
        ==
        0.005
    )


# ============================================================
# BOUNDS
# ============================================================


def test_hidden_feature_bounds(
) -> None:

    def zero_hidden(
    ) -> None:
        DLTabularMLPRegressorHyperparameters(
            hidden_features=0
        )


    def excessive_hidden(
    ) -> None:
        DLTabularMLPRegressorHyperparameters(
            hidden_features=2049
        )


    expect_validation_error(
        zero_hidden
    )


    expect_validation_error(
        excessive_hidden
    )


def test_epoch_bounds(
) -> None:

    def zero_epochs(
    ) -> None:
        DLTabularMLPRegressorHyperparameters(
            epochs=0
        )


    def excessive_epochs(
    ) -> None:
        DLTabularMLPRegressorHyperparameters(
            epochs=2001
        )


    expect_validation_error(
        zero_epochs
    )


    expect_validation_error(
        excessive_epochs
    )


def test_batch_size_bounds(
) -> None:

    def zero_batch_size(
    ) -> None:
        DLTabularMLPRegressorHyperparameters(
            batch_size=0
        )


    def excessive_batch_size(
    ) -> None:
        DLTabularMLPRegressorHyperparameters(
            batch_size=65537
        )


    expect_validation_error(
        zero_batch_size
    )


    expect_validation_error(
        excessive_batch_size
    )


def test_learning_rate_bounds(
) -> None:

    def zero_learning_rate(
    ) -> None:
        DLTabularMLPRegressorHyperparameters(
            learning_rate=0.0
        )


    def excessive_learning_rate(
    ) -> None:
        DLTabularMLPRegressorHyperparameters(
            learning_rate=1.000001
        )


    expect_validation_error(
        zero_learning_rate
    )


    expect_validation_error(
        excessive_learning_rate
    )


# ============================================================
# SERVER-OWNED EXECUTION CONTROLS
# ============================================================


def test_execution_controls_are_not_client_configurable(
) -> None:

    forbidden_fields = (
        ("random_seed", 123),
        ("device", "cuda"),
        ("optimizer", "adam"),
        ("loss", "mae"),
        ("activation", "gelu"),
        ("num_workers", 4),
    )


    for (
        field_name,
        field_value,
    ) in forbidden_fields:

        def build(
            field_name=field_name,
            field_value=field_value,
        ) -> None:

            DLTabularMLPRegressorHyperparameters(
                **{
                    field_name:
                        field_value
                }
            )


        expect_validation_error(
            build
        )


# ============================================================
# IMMUTABILITY
# ============================================================


def test_contract_is_frozen(
) -> None:

    contract = (
        DLTabularMLPRegressorHyperparameters()
    )


    def mutate(
    ) -> None:

        contract.hidden_features = 32


    expect_validation_error(
        mutate
    )


# ============================================================
# DETERMINISTIC SERIALIZATION
# ============================================================


def test_serialization_is_deterministic(
) -> None:

    first = (
        DLTabularMLPRegressorHyperparameters(
            hidden_features=96,
            epochs=150,
            batch_size=48,
            learning_rate=0.02,
        )
    )


    second = (
        DLTabularMLPRegressorHyperparameters(
            hidden_features=96,
            epochs=150,
            batch_size=48,
            learning_rate=0.02,
        )
    )


    assert (
        first.model_dump(
            mode="json"
        )
        ==
        second.model_dump(
            mode="json"
        )
    )


# ============================================================
# TORCH-FREE CONTRACT SURFACE
# ============================================================


def test_contract_module_has_no_torch_dependency(
) -> None:

    path = (
        Path(__file__)
        .parents[2]
        /
        "app"
        /
        "deep_learning"
        /
        "tabular_contracts.py"
    )


    source = path.read_text(
        encoding="utf-8"
    )


    forbidden_tokens = (
        "import torch",
        "from torch",
        "torch.",
    )


    for token in forbidden_tokens:

        assert (
            token
            not in
            source
        )


# ============================================================
# RULE VERSION
# ============================================================


def test_rule_version(
) -> None:

    assert (
        DL_TABULAR_MLP_CONTRACT_RULE_VERSION
        ==
        "dl_tabular_mlp_contract_v0.1"
    )


    assert (
        DLTabularMLPRegressorHyperparameters()
        .rule_version
        ==
        DL_TABULAR_MLP_CONTRACT_RULE_VERSION
    )


# ============================================================
# MAIN
# ============================================================


def main(
) -> None:

    print(
        "=== DATALENS TABULAR MLP CONTRACT v0.1 ==="
    )

    print()


    test_default_contract()

    print(
        "Default MLP contract: PASS"
    )


    test_explicit_contract()

    print(
        "Explicit MLP hyperparameters: PASS"
    )


    test_hidden_feature_bounds()

    print(
        "Hidden-feature bounds: PASS"
    )


    test_epoch_bounds()

    print(
        "Epoch bounds: PASS"
    )


    test_batch_size_bounds()

    print(
        "Batch-size bounds: PASS"
    )


    test_learning_rate_bounds()

    print(
        "Learning-rate bounds: PASS"
    )


    test_execution_controls_are_not_client_configurable()

    print(
        "Server-owned execution controls: PASS"
    )


    test_contract_is_frozen()

    print(
        "Frozen contract: PASS"
    )


    test_serialization_is_deterministic()

    print(
        "Deterministic serialization: PASS"
    )


    test_contract_module_has_no_torch_dependency()

    print(
        "Torch-free contract surface: PASS"
    )


    test_rule_version()

    print(
        "Rule version: PASS"
    )


    print()

    print(
        "PASS - DataLens Tabular MLP Contract v0.1"
    )


if __name__ == "__main__":
    main()
