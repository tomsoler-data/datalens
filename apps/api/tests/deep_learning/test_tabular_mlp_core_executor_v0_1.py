from __future__ import annotations


import math
import sys


from pathlib import (
    Path,
)


import numpy as np
import pandas as pd


import app.deep_learning.tabular_executor as executor_module


from app.deep_learning.runtime import (
    resolve_device,
)


from app.deep_learning.tabular_executor import (
    DL_TABULAR_MLP_CORE_EXECUTOR_RULE_VERSION,
    TabularMLPCoreExecutorError,
    TabularMLPEstimatorError,
    TabularMLPInputError,
    execute_tabular_mlp_core,
)


from app.ml.contracts import (
    MLTrainingContract,
)


from app.ml.model_metrics import (
    compute_ml_regression_metrics,
)


from app.ml.splitting import (
    resolve_ml_holdout_partition,
)


# ============================================================
# FIXTURE
# ============================================================


def build_xy(
) -> tuple[
    pd.DataFrame,
    pd.Series,
]:

    rows = 120


    feature_a = np.linspace(
        -2.0,
        2.0,
        rows,
        dtype=np.float64,
    )


    feature_b = np.linspace(
        1.5,
        -1.5,
        rows,
        dtype=np.float64,
    )


    segment = [
        (
            "business"
            if
            index % 2 == 0
            else
            "consumer"
        )

        for index
        in range(
            rows
        )
    ]


    segment_effect = np.asarray(
        [
            (
                0.35
                if
                value == "business"
                else
                -0.20
            )

            for value
            in segment
        ],
        dtype=np.float64,
    )


    target = (
        0.70
        *
        feature_a
        -
        0.45
        *
        feature_b
        +
        segment_effect
        +
        0.10
    )


    source_index = [
        10_000
        +
        index
        *
        7

        for index
        in range(
            rows
        )
    ]


    x = pd.DataFrame(
        {
            "feature_a":
                feature_a,

            "feature_b":
                feature_b,

            "segment":
                segment,
        },
        index=
            source_index,
    )


    y = pd.Series(
        target,
        index=
            source_index,
        name=
            "target",
    )


    return (
        x,
        y,
    )


def build_contract(
    *,
    epochs: int = 140,
) -> MLTrainingContract:

    return (
        MLTrainingContract(
            workflow_id=
                "prep:tabular-mlp-core",

            dataset_id=
                "dataset:validated",

            problem_type=
                "regression",

            target_column=
                "target",

            feature_columns=[
                "feature_a",
                "feature_b",
                "segment",
            ],

            categorical_feature_columns=[
                "segment",
            ],

            estimator_key=
                "tabular_mlp_regressor",

            estimator_hyperparameters={
                "kind":
                    "tabular_mlp_regressor",

                "hidden_features":
                    16,

                "epochs":
                    epochs,

                "batch_size":
                    18,

                "learning_rate":
                    0.03,
            },

            preprocessing={
                "numeric_imputation":
                    "error",

                "categorical_imputation":
                    "error",

                "categorical_encoding":
                    "one_hot",

                "handle_unknown_categories":
                    "ignore",

                "scale_numeric":
                    True,
            },

            split={
                "strategy":
                    "holdout",

                "test_size":
                    0.25,

                "random_seed":
                    19,

                "shuffle":
                    True,

                "stratify":
                    False,
            },
        )
    )


# ============================================================
# END-TO-END CORE EXECUTION
# ============================================================


def test_core_mlp_executes_end_to_end(
) -> None:

    (
        x,
        y,
    ) = (
        build_xy()
    )


    contract = (
        build_contract()
    )


    expected_partition = (
        resolve_ml_holdout_partition(
            x=x,
            y=y,
            contract=contract,
        )
    )


    result = (
        execute_tabular_mlp_core(
            x=x,
            y=y,
            contract=contract,
            execution_device=
                "cpu",
        )
    )


    assert (
        result.estimator_key
        ==
        "tabular_mlp_regressor"
    )


    assert (
        result.partition
        ==
        expected_partition
    )


    assert (
        result.train_rows
        ==
        90
    )


    assert (
        result.test_rows
        ==
        30
    )


    assert (
        result.purged_rows
        ==
        0
    )


    assert (
        result.random_seed
        ==
        19
    )


    assert (
        result.execution_device
        ==
        "cpu"
    )


    assert (
        result.input_features
        >
        0
    )


    assert (
        len(
            result.epoch_losses
        )
        ==
        140
    )


    assert (
        len(
            result.predictions
        )
        ==
        result.test_rows
    )


    assert (
        result.epoch_losses[
            -1
        ]
        <
        result.epoch_losses[
            0
        ]
    )


    assert math.isfinite(
        result.test_loss
    )


    assert (
        set(
            result.metrics
        )
        ==
        {
            "mae",
            "rmse",
            "r2",
            "median_absolute_error",
            "explained_variance",
        }
    )


    for value in (
        result.metrics.values()
    ):

        assert math.isfinite(
            value
        )


# ============================================================
# CANONICAL METRICS
# ============================================================


def test_core_uses_canonical_model_lab_metrics(
) -> None:

    (
        x,
        y,
    ) = (
        build_xy()
    )


    contract = (
        build_contract(
            epochs=80
        )
    )


    result = (
        execute_tabular_mlp_core(
            x=x,
            y=y,
            contract=contract,
            execution_device=
                "cpu",
        )
    )


    y_test = (
        y.iloc[
            list(
                result
                .partition
                .test_positions
            )
        ]
    )


    expected = (
        compute_ml_regression_metrics(
            y_true=y_test,
            predictions=np.asarray(
                result.predictions,
                dtype=np.float64,
            ),
        )
    )


    assert (
        result.metrics
        ==
        expected
    )


# ============================================================
# DETERMINISM
# ============================================================


def test_core_execution_is_deterministic_on_cpu(
) -> None:

    (
        x,
        y,
    ) = (
        build_xy()
    )


    contract = (
        build_contract(
            epochs=80
        )
    )


    first = (
        execute_tabular_mlp_core(
            x=x,
            y=y,
            contract=contract,
            execution_device=
                "cpu",
        )
    )


    second = (
        execute_tabular_mlp_core(
            x=x,
            y=y,
            contract=contract,
            execution_device=
                "cpu",
        )
    )


    assert (
        first.partition
        ==
        second.partition
    )


    assert (
        first.epoch_losses
        ==
        second.epoch_losses
    )


    assert (
        first.predictions
        ==
        second.predictions
    )


    assert (
        first.metrics
        ==
        second.metrics
    )


# ============================================================
# TRAIN-ONLY PREPROCESSING
# ============================================================


def test_preprocessor_learns_from_train_only(
) -> None:

    (
        x,
        y,
    ) = (
        build_xy()
    )


    contract = (
        build_contract(
            epochs=5
        )
    )


    initial_partition = (
        resolve_ml_holdout_partition(
            x=x,
            y=y,
            contract=contract,
        )
    )


    test_position = (
        initial_partition
        .test_positions[
            0
        ]
    )


    modified_x = (
        x.copy(
            deep=True
        )
    )


    modified_x.iloc[
        test_position,
        modified_x.columns.get_loc(
            "feature_a"
        ),
    ] = 1_000_000.0


    modified_x.iloc[
        test_position,
        modified_x.columns.get_loc(
            "segment"
        ),
    ] = "test_only_category"


    result = (
        execute_tabular_mlp_core(
            x=
                modified_x,

            y=
                y,

            contract=
                contract,

            execution_device=
                "cpu",
        )
    )


    assert (
        result.partition
        ==
        initial_partition
    )


    numeric_pipeline = (
        result
        .preprocessor
        .named_transformers_[
            "numeric"
        ]
    )


    scaler = (
        numeric_pipeline
        .named_steps[
            "scaler"
        ]
    )


    train_frame = (
        modified_x.iloc[
            list(
                result
                .partition
                .train_positions
            )
        ]
    )


    expected_train_mean = (
        train_frame[
            [
                "feature_a",
                "feature_b",
            ]
        ]
        .mean()
        .to_numpy(
            dtype=np.float64
        )
    )


    whole_dataset_mean = (
        modified_x[
            [
                "feature_a",
                "feature_b",
            ]
        ]
        .mean()
        .to_numpy(
            dtype=np.float64
        )
    )


    assert np.allclose(
        scaler.mean_,
        expected_train_mean,
        rtol=0.0,
        atol=1e-12,
    )


    assert not np.allclose(
        scaler.mean_,
        whole_dataset_mean,
        rtol=0.0,
        atol=1e-12,
    )


    assert (
        len(
            result.predictions
        )
        ==
        result.test_rows
    )


# ============================================================
# ESTIMATOR AUTHORITY
# ============================================================


def test_classical_estimator_is_refused(
) -> None:

    (
        x,
        y,
    ) = (
        build_xy()
    )


    contract = (
        MLTrainingContract(
            workflow_id=
                "prep:not-mlp",

            dataset_id=
                "dataset:validated",

            problem_type=
                "regression",

            target_column=
                "target",

            feature_columns=[
                "feature_a",
                "feature_b",
            ],

            estimator_key=
                "linear_regression",
        )
    )


    try:

        execute_tabular_mlp_core(
            x=
                x[
                    [
                        "feature_a",
                        "feature_b",
                    ]
                ],

            y=
                y,

            contract=
                contract,

            execution_device=
                "cpu",
        )

    except TabularMLPEstimatorError:
        return


    raise AssertionError(
        (
            "Classical estimator must be refused "
            "by the MLP core executor."
        )
    )


# ============================================================
# INPUT ALIGNMENT
# ============================================================


def test_misaligned_input_fails_closed(
) -> None:

    (
        x,
        y,
    ) = (
        build_xy()
    )


    contract = (
        build_contract(
            epochs=5
        )
    )


    broken_y = (
        y.copy(
            deep=True
        )
    )


    broken_y.index = list(
        reversed(
            broken_y.index.tolist()
        )
    )


    try:

        execute_tabular_mlp_core(
            x=x,
            y=broken_y,
            contract=contract,
            execution_device=
                "cpu",
        )

    except TabularMLPInputError:
        return


    raise AssertionError(
        (
            "Misaligned x/y must fail closed."
        )
    )


# ============================================================
# NUMERICAL STABILITY
# ============================================================


def test_non_finite_training_loss_fails_closed(
) -> None:

    (
        x,
        y,
    ) = (
        build_xy()
    )


    contract = (
        build_contract(
            epochs=5
        )
    )


    original = (
        executor_module
        .train_regression_epochs
    )


    def fake_train_regression_epochs(
        *args,
        **kwargs,
    ):

        return [
            1.0,
            float(
                "nan"
            ),
        ]


    executor_module.train_regression_epochs = (
        fake_train_regression_epochs
    )


    try:

        try:

            execute_tabular_mlp_core(
                x=x,
                y=y,
                contract=contract,
                execution_device=
                    "cpu",
            )

        except TabularMLPCoreExecutorError as error:

            assert (
                "non-finite loss"
                in
                str(
                    error
                )
            )

        else:

            raise AssertionError(
                (
                    "Non-finite training loss "
                    "must fail closed."
                )
            )

    finally:

        executor_module.train_regression_epochs = (
            original
        )


# ============================================================
# DEFAULT DEVICE AUTHORITY
# ============================================================


def test_default_device_authority_executes(
) -> None:

    (
        x,
        y,
    ) = (
        build_xy()
    )


    contract = (
        build_contract(
            epochs=5
        )
    )


    result = (
        execute_tabular_mlp_core(
            x=x,
            y=y,
            contract=contract,
        )
    )


    assert (
        result.execution_device
        ==
        str(
            resolve_device()
        )
    )


    assert (
        len(
            result.predictions
        )
        ==
        result.test_rows
    )


# ============================================================
# RUNTIME BOUNDARY
# ============================================================


def test_torch_lives_only_on_deep_learning_side(
) -> None:

    executor_path = (
        Path(__file__)
        .parents[2]
        /
        "app"
        /
        "deep_learning"
        /
        "tabular_executor.py"
    )


    assert (
        "torch"
        in
        executor_path.read_text(
            encoding="utf-8"
        )
    )


    for relative_path in (
        (
            "app/ml/"
            "training_estimator_contracts.py"
        ),
        (
            "app/ml/"
            "contracts.py"
        ),
        (
            "app/ml/"
            "splitting.py"
        ),
    ):

        path = (
            Path(__file__)
            .parents[2]
            /
            relative_path
        )


        source = path.read_text(
            encoding="utf-8"
        )


        for token in (
            "import torch",
            "from torch",
            "torch.",
        ):

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
        DL_TABULAR_MLP_CORE_EXECUTOR_RULE_VERSION
        ==
        "dl_tabular_mlp_core_executor_v0.1"
    )


# ============================================================
# MAIN
# ============================================================


def main(
) -> None:

    print(
        "=== DATALENS TABULAR MLP CORE EXECUTOR v0.1 ==="
    )

    print()


    test_core_mlp_executes_end_to_end()

    print(
        "End-to-end MLP regression: PASS"
    )


    test_core_uses_canonical_model_lab_metrics()

    print(
        "Canonical Model Lab metrics: PASS"
    )


    test_core_execution_is_deterministic_on_cpu()

    print(
        "Deterministic CPU execution: PASS"
    )


    test_preprocessor_learns_from_train_only()

    print(
        "TRAIN-only preprocessing / leakage guard: PASS"
    )


    test_classical_estimator_is_refused()

    print(
        "MLP estimator authority: PASS"
    )


    test_misaligned_input_fails_closed()

    print(
        "Input alignment fail-closed guard: PASS"
    )


    test_non_finite_training_loss_fails_closed()

    print(
        "Non-finite training fail-closed guard: PASS"
    )


    test_default_device_authority_executes()

    print(
        "Default CPU/CUDA device authority: PASS"
    )


    test_torch_lives_only_on_deep_learning_side()

    print(
        "Runtime / PyTorch isolation boundary: PASS"
    )


    test_rule_version()

    print(
        "Core executor rule version: PASS"
    )


    print()

    print(
        "PASS - DataLens Tabular MLP Core Executor v0.1"
    )


if __name__ == "__main__":
    main()
