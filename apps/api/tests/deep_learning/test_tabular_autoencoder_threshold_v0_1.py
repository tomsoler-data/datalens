from __future__ import annotations


import inspect
import math


import torch


from app.deep_learning.autoencoder_threshold import (
    AUTOENCODER_THRESHOLD_COMPARISON,
    AUTOENCODER_THRESHOLD_METHOD,
    AUTOENCODER_THRESHOLD_RULE_VERSION,
    ReconstructionErrorThreshold,
    apply_reconstruction_error_threshold,
    fit_reconstruction_error_threshold,
)


from app.ml.anomaly_contracts import (
    MLAnomalyTrainingContract,
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


def train_errors(
) -> torch.Tensor:

    return torch.tensor(
        [
            0.01,
            0.02,
            0.03,
            0.04,
            0.05,
            0.06,
            0.07,
            0.08,
            0.09,
            0.10,
        ],
        dtype=
            torch.float32,
    )


# ============================================================
# CONTRACT POLICY
# ============================================================


def test_contract_threshold_quantile(
) -> None:

    default_contract = (
        MLAnomalyTrainingContract(
            workflow_id=
                "workflow:a6-default",
            dataset_id=
                "dataset:a6-default",
            feature_columns=[
                "x",
            ],
        )
    )


    assert (
        default_contract.threshold_quantile
        ==
        0.99
    )


    explicit_contract = (
        MLAnomalyTrainingContract(
            workflow_id=
                "workflow:a6-explicit",
            dataset_id=
                "dataset:a6-explicit",
            feature_columns=[
                "x",
            ],
        threshold_quantile=
            0.90,
        )
    )


    assert (
        explicit_contract.threshold_quantile
        ==
        0.90
    )


# ============================================================
# TRAIN-ONLY API
# ============================================================


def test_fit_api_is_train_only(
) -> None:

    signature = inspect.signature(
        fit_reconstruction_error_threshold
    )


    names = set(
        signature.parameters
    )


    assert (
        names
        ==
        {
            "train_errors",
            "quantile",
        }
    )


    for forbidden in (
        "test_errors",
        "evaluation_errors",
        "labels",
        "targets",
        "y",
    ):

        assert (
            forbidden
            not in
            names
        )


# ============================================================
# EXACT QUANTILE
# ============================================================


def test_exact_quantile_threshold(
) -> None:

    errors = train_errors()


    result = (
        fit_reconstruction_error_threshold(
            train_errors=
                errors,
            quantile=
                0.90,
        )
    )


    expected = float(
        torch.quantile(
            errors,
            q=
                0.90,
            interpolation=
                "linear",
        )
        .item()
    )


    assert isinstance(
        result,
        ReconstructionErrorThreshold,
    )


    assert (
        result.quantile
        ==
        0.90
    )


    assert (
        result.train_rows
        ==
        10
    )


    assert math.isclose(
        result.threshold,
        expected,
        rel_tol=
            0.0,
        abs_tol=
            0.0,
    )


    assert (
        result.method
        ==
        "train_reconstruction_error_quantile"
    )


    assert (
        result.comparison_operator
        ==
        "greater_than"
    )


# ============================================================
# DETERMINISM
# ============================================================


def test_threshold_determinism(
) -> None:

    first = (
        fit_reconstruction_error_threshold(
            train_errors=
                train_errors(),
            quantile=
                0.90,
        )
    )


    second = (
        fit_reconstruction_error_threshold(
            train_errors=
                train_errors(),
            quantile=
                0.90,
        )
    )


    assert (
        first
        ==
        second
    )


# ============================================================
# TEST DATA CANNOT CHANGE FITTED THRESHOLD
# ============================================================


def test_test_population_is_application_only(
) -> None:

    fitted = (
        fit_reconstruction_error_threshold(
            train_errors=
                train_errors(),
            quantile=
                0.90,
        )
    )


    original_threshold = (
        fitted.threshold
    )


    first_test = torch.tensor(
        [
            0.01,
            0.02,
            5.00,
        ],
        dtype=
            torch.float32,
    )


    second_test = torch.tensor(
        [
            100.0,
            200.0,
            300.0,
        ],
        dtype=
            torch.float32,
    )


    first_flags = (
        apply_reconstruction_error_threshold(
            reconstruction_errors=
                first_test,
            threshold=
                fitted,
        )
    )


    second_flags = (
        apply_reconstruction_error_threshold(
            reconstruction_errors=
                second_test,
            threshold=
                fitted,
        )
    )


    assert (
        fitted.threshold
        ==
        original_threshold
    )


    assert torch.equal(
        first_flags,
        torch.tensor(
            [
                False,
                False,
                True,
            ]
        ),
    )


    assert torch.equal(
        second_flags,
        torch.tensor(
            [
                True,
                True,
                True,
            ]
        ),
    )


# ============================================================
# STRICT GREATER-THAN POLICY
# ============================================================


def test_threshold_comparison_policy(
) -> None:

    fitted = (
        ReconstructionErrorThreshold(
            quantile=
                0.90,
            threshold=
                0.10,
            train_rows=
                10,
        )
    )


    flags = (
        apply_reconstruction_error_threshold(
            reconstruction_errors=
                torch.tensor(
                    [
                        0.09,
                        0.10,
                        0.11,
                    ],
                    dtype=
                        torch.float32,
                ),
            threshold=
                fitted,
        )
    )


    assert torch.equal(
        flags,
        torch.tensor(
            [
                False,
                False,
                True,
            ]
        ),
    )


# ============================================================
# RESULT SHAPE / TYPE
# ============================================================


def test_flag_surface(
) -> None:

    fitted = (
        fit_reconstruction_error_threshold(
            train_errors=
                train_errors(),
            quantile=
                0.90,
        )
    )


    flags = (
        apply_reconstruction_error_threshold(
            reconstruction_errors=
                train_errors(),
            threshold=
                fitted,
        )
    )


    assert (
        flags.dtype
        ==
        torch.bool
    )


    assert (
        flags.device.type
        ==
        "cpu"
    )


    assert (
        flags.shape
        ==
        torch.Size(
            [
                10,
            ]
        )
    )


    assert (
        flags.requires_grad
        is False
    )


# ============================================================
# GUARDS
# ============================================================


def test_threshold_guards(
) -> None:

    expect_error(
        lambda:
            fit_reconstruction_error_threshold(
                train_errors=
                    torch.tensor(
                        [
                            0.1,
                        math.nan,
                    ],
                    dtype=
                        torch.float32,
                ),
                quantile=
                    0.90,
            ),
        ValueError,
    )


    expect_error(
        lambda:
            fit_reconstruction_error_threshold(
                train_errors=
                    torch.tensor(
                        [
                            -0.1,
                            0.2,
                        ],
                        dtype=
                            torch.float32,
                ),
                quantile=
                    0.90,
            ),
        ValueError,
    )


    expect_error(
        lambda:
            fit_reconstruction_error_threshold(
                train_errors=
                    train_errors(),
                quantile=
                    1.0,
            ),
        ValueError,
    )


    expect_error(
        lambda:
            fit_reconstruction_error_threshold(
                train_errors=
                    train_errors(),
                quantile=
                    True,
            ),
        TypeError,
    )


# ============================================================
# FIXED POLICIES
# ============================================================


def test_single_row_threshold_application(
) -> None:

    threshold = (
        ReconstructionErrorThreshold(
            quantile=
                0.99,
            threshold=
                0.5,
            train_rows=
                10,
        )
    )


    anomalous = (
        apply_reconstruction_error_threshold(
            reconstruction_errors=
                torch.tensor(
                    [
                        0.75,
                    ],
                    dtype=torch.float32,
                ),
            threshold=
                threshold,
        )
    )


    normal = (
        apply_reconstruction_error_threshold(
            reconstruction_errors=
                torch.tensor(
                    [
                        0.25,
                    ],
                    dtype=torch.float32,
                ),
            threshold=
                threshold,
        )
    )


    equal_to_threshold = (
        apply_reconstruction_error_threshold(
            reconstruction_errors=
                torch.tensor(
                    [
                        0.5,
                    ],
                    dtype=torch.float32,
                ),
            threshold=
                threshold,
        )
    )


    assert (
        anomalous.shape
        ==
        torch.Size(
            [
                1,
            ]
        )
    )


    assert (
        anomalous.dtype
        ==
        torch.bool
    )


    assert (
        bool(
            anomalous[0].item()
        )
        is True
    )


    assert (
        bool(
            normal[0].item()
        )
        is False
    )


    # Comparison semantics remain strictly greater-than.

    assert (
        bool(
            equal_to_threshold[0].item()
        )
        is False
    )


def test_single_train_row_still_cannot_fit_threshold(
) -> None:

    try:

        fit_reconstruction_error_threshold(
            train_errors=
                torch.tensor(
                    [
                        0.25,
                    ],
                    dtype=torch.float32,
                ),
            quantile=
                0.99,
        )

    except ValueError:
        return


    raise AssertionError(
        (
            "Threshold fitting must still require "
            "at least two TRAIN rows."
        )
    )


def test_rule_versions(
) -> None:

    assert (
        AUTOENCODER_THRESHOLD_RULE_VERSION
        ==
        "autoencoder_threshold_v0.1"
    )


    assert (
        AUTOENCODER_THRESHOLD_METHOD
        ==
        "train_reconstruction_error_quantile"
    )


    assert (
        AUTOENCODER_THRESHOLD_COMPARISON
        ==
        "greater_than"
    )


# ============================================================
# DIRECT ACCEPTANCE
# ============================================================


def main(
) -> None:

    print(
        "=== DATALENS TABULAR AUTOENCODER THRESHOLD v0.1 ==="
    )

    print()


    test_contract_threshold_quantile()

    print(
        "Threshold quantile contract policy: PASS"
    )


    test_fit_api_is_train_only()

    print(
        "TRAIN-only fitting API: PASS"
    )


    test_exact_quantile_threshold()

    print(
        "Deterministic TRAIN quantile threshold: PASS"
    )


    test_threshold_determinism()

    print(
        "Threshold determinism: PASS"
    )


    test_test_population_is_application_only()

    print(
        "TEST population cannot fit threshold: PASS"
    )


    test_threshold_comparison_policy()

    print(
        "Strict greater-than anomaly policy: PASS"
    )


    test_flag_surface()

    print(
        "Boolean anomaly-flag surface: PASS"
    )


    test_threshold_guards()

    print(
        "Threshold fail-closed guards: PASS"
    )


    test_single_row_threshold_application()

    print(
        "Single-row threshold application: PASS"
    )


    test_single_train_row_still_cannot_fit_threshold()

    print(
        "TRAIN minimum-row fit guard preserved: PASS"
    )


    test_rule_versions()

    print(
        "Threshold rule versions: PASS"
    )


    print()

    print(
        "PASS - DataLens Tabular Autoencoder Threshold v0.1"
    )


if __name__ == "__main__":

    main()
