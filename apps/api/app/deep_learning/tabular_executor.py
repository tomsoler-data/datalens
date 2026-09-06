from __future__ import annotations


import math


from dataclasses import (
    dataclass,
)


from typing import (
    Any,
)


import numpy as np
import pandas as pd
import torch


from app.deep_learning.datasets import (
    TabularTensorDataset,
    build_data_loader,
)


from app.deep_learning.evaluation import (
    evaluate_regression_loader,
)


from app.deep_learning.networks import (
    FeedForwardRegressor,
)


from app.deep_learning.runtime import (
    resolve_device,
    seed_torch,
)


from app.deep_learning.tabular_contracts import (
    DLTabularMLPRegressorHyperparameters,
)


from app.deep_learning.training import (
    build_regression_loss,
    build_sgd_optimizer,
    train_regression_epochs,
)


from app.ml.contracts import (
    MLTrainingContract,
)


from app.ml.model_metrics import (
    MLModelMetricsError,
    compute_ml_regression_metrics,
)


from app.ml.preprocessing import (
    MLPreprocessingRuntimeError,
    build_ml_preprocessor,
    validate_ml_feature_frame,
)


from app.ml.splitting import (
    MLHoldoutPartition,
    MLSplitError,
    resolve_ml_holdout_partition,
)


# ============================================================
# VERSION
# ============================================================


DL_TABULAR_MLP_CORE_EXECUTOR_RULE_VERSION = (
    "dl_tabular_mlp_core_executor_v0.1"
)


# ============================================================
# ERRORS
# ============================================================


class TabularMLPCoreExecutorError(
    RuntimeError
):
    pass


class TabularMLPInputError(
    TabularMLPCoreExecutorError
):
    pass


class TabularMLPEstimatorError(
    TabularMLPCoreExecutorError
):
    pass


# ============================================================
# RESULT
# ============================================================


@dataclass(
    frozen=True,
)
class TabularMLPCoreExecutionResult:
    """
    Internal trained MLP execution bundle.

    This is not yet the persisted Model Lab result contract.

    model and preprocessor are intentionally retained here so
    the later artifact milestone can serialize the already
    trained execution rather than retraining it.
    """

    estimator_key: str

    partition: MLHoldoutPartition

    input_features: int

    random_seed: int

    execution_device: str

    epoch_losses: tuple[
        float,
        ...
    ]

    test_loss: float

    metrics: dict[
        str,
        float,
    ]

    predictions: tuple[
        float,
        ...
    ]

    model: FeedForwardRegressor

    preprocessor: Any


    @property
    def train_rows(
        self,
    ) -> int:
        return (
            self.partition.train_rows
        )


    @property
    def test_rows(
        self,
    ) -> int:
        return (
            self.partition.test_rows
        )


    @property
    def purged_rows(
        self,
    ) -> int:
        return (
            self.partition.purged_rows
        )


# ============================================================
# INPUT VALIDATION
# ============================================================


def _validate_tabular_mlp_inputs(
    *,
    x: pd.DataFrame,
    y: pd.Series,
    contract: MLTrainingContract,
) -> tuple[
    pd.DataFrame,
    pd.Series,
    DLTabularMLPRegressorHyperparameters,
]:

    if not isinstance(
        contract,
        MLTrainingContract,
    ):
        raise (
            TabularMLPInputError(
                (
                    "contract must be an "
                    "MLTrainingContract."
                )
            )
        )


    if (
        contract.estimator_key
        !=
        "tabular_mlp_regressor"
    ):
        raise (
            TabularMLPEstimatorError(
                (
                    "Tabular MLP core executor only "
                    "supports estimator_key="
                    "'tabular_mlp_regressor'. "
                    f"estimator_key={contract.estimator_key}"
                )
            )
        )


    if (
        contract.problem_type
        !=
        "regression"
    ):
        raise (
            TabularMLPEstimatorError(
                (
                    "Tabular MLP core executor only "
                    "supports regression."
                )
            )
        )


    hyperparameters = (
        contract
        .effective_estimator_hyperparameters
    )


    if not isinstance(
        hyperparameters,
        DLTabularMLPRegressorHyperparameters,
    ):
        raise (
            TabularMLPEstimatorError(
                (
                    "Tabular MLP estimator did not "
                    "resolve to its typed "
                    "hyperparameter contract."
                )
            )
        )


    if not isinstance(
        x,
        pd.DataFrame,
    ):
        raise (
            TabularMLPInputError(
                "x must be a pandas DataFrame."
            )
        )


    if not isinstance(
        y,
        pd.Series,
    ):
        raise (
            TabularMLPInputError(
                "y must be a pandas Series."
            )
        )


    if x.empty:
        raise (
            TabularMLPInputError(
                "x cannot be empty."
            )
        )


    if (
        len(
            x
        )
        !=
        len(
            y
        )
    ):
        raise (
            TabularMLPInputError(
                (
                    "x and y must contain the "
                    "same number of rows."
                )
            )
        )


    if not x.index.equals(
        y.index
    ):
        raise (
            TabularMLPInputError(
                (
                    "x and y indexes must be "
                    "row-aligned."
                )
            )
        )


    try:

        validated_x = (
            validate_ml_feature_frame(
                features=x,
                contract=contract,
            )
        )

    except MLPreprocessingRuntimeError as error:

        raise (
            TabularMLPInputError(
                str(
                    error
                )
            )
        ) from error


    if bool(
        y.isna().any()
    ):
        raise (
            TabularMLPInputError(
                (
                    "Tabular MLP regression target "
                    "cannot contain missing values."
                )
            )
        )


    if (
        pd.api.types
        .is_bool_dtype(
            y.dtype
        )
        or
        not pd.api.types
        .is_numeric_dtype(
            y.dtype
        )
    ):
        raise (
            TabularMLPInputError(
                (
                    "Tabular MLP regression target "
                    "must be numeric and non-boolean."
                )
            )
        )


    try:

        numeric_target = (
            y.to_numpy(
                dtype=np.float64,
                copy=True,
            )
        )

    except Exception as error:

        raise (
            TabularMLPInputError(
                (
                    "Tabular MLP regression target "
                    "could not be converted to "
                    "floating-point values."
                )
            )
        ) from error


    if not (
        np.isfinite(
            numeric_target
        )
        .all()
    ):
        raise (
            TabularMLPInputError(
                (
                    "Tabular MLP regression target "
                    "contains non-finite values."
                )
            )
        )


    if (
        int(
            y.nunique(
                dropna=False
            )
        )
        <
        2
    ):
        raise (
            TabularMLPInputError(
                (
                    "Tabular MLP regression target "
                    "must contain at least two "
                    "distinct values."
                )
            )
        )


    return (
        validated_x,
        y.copy(
            deep=True
        ),
        hyperparameters,
    )


# ============================================================
# FLOAT32 MATRIX AUTHORITY
# ============================================================


def _as_float32_feature_matrix(
    values: Any,
    *,
    name: str,
) -> np.ndarray:

    try:

        array = np.asarray(
            values,
            dtype=np.float32,
        )

    except Exception as error:

        raise (
            TabularMLPInputError(
                (
                    f"{name} could not be converted "
                    "to a float32 feature matrix."
                )
            )
        ) from error


    if array.ndim != 2:
        raise (
            TabularMLPInputError(
                (
                    f"{name} must be "
                    "two-dimensional."
                )
            )
        )


    if (
        array.shape[
            0
        ]
        <=
        0
        or
        array.shape[
            1
        ]
        <=
        0
    ):
        raise (
            TabularMLPInputError(
                (
                    f"{name} must contain at least "
                    "one row and one feature."
                )
            )
        )


    if not (
        np.isfinite(
            array
        )
        .all()
    ):
        raise (
            TabularMLPInputError(
                (
                    f"{name} contains non-finite "
                    "transformed values."
                )
            )
        )


    return np.ascontiguousarray(
        array,
        dtype=np.float32,
    )


def _as_float32_target_vector(
    values: pd.Series,
    *,
    name: str,
) -> np.ndarray:

    try:

        array = (
            values.to_numpy(
                dtype=np.float32,
                copy=True,
            )
        )

    except Exception as error:

        raise (
            TabularMLPInputError(
                (
                    f"{name} could not be converted "
                    "to a float32 target vector."
                )
            )
        ) from error


    if array.ndim != 1:
        raise (
            TabularMLPInputError(
                (
                    f"{name} must be "
                    "one-dimensional."
                )
            )
        )


    if array.shape[0] <= 0:
        raise (
            TabularMLPInputError(
                (
                    f"{name} cannot be empty."
                )
            )
        )


    if not (
        np.isfinite(
            array
        )
        .all()
    ):
        raise (
            TabularMLPInputError(
                (
                    f"{name} contains non-finite "
                    "target values."
                )
            )
        )


    return np.ascontiguousarray(
        array,
        dtype=np.float32,
    )


# ============================================================
# DEVICE AUTHORITY
# ============================================================


def _resolve_execution_device(
    execution_device: (
        str
        |
        torch.device
        |
        None
    ),
) -> torch.device:

    if execution_device is None:

        return (
            resolve_device()
        )


    try:

        device = torch.device(
            execution_device
        )

    except Exception as error:

        raise (
            TabularMLPCoreExecutorError(
                (
                    "Invalid internal PyTorch "
                    "execution device."
                )
            )
        ) from error


    if (
        device.type
        ==
        "cuda"
        and
        not torch.cuda.is_available()
    ):
        raise (
            TabularMLPCoreExecutorError(
                (
                    "CUDA execution was requested "
                    "but CUDA is unavailable."
                )
            )
        )


    return device


# ============================================================
# EXECUTION
# ============================================================


def execute_tabular_mlp_core(
    *,
    x: pd.DataFrame,
    y: pd.Series,
    contract: MLTrainingContract,
    dataframe: pd.DataFrame | None = None,
    execution_device: (
        str
        |
        torch.device
        |
        None
    ) = None,
) -> TabularMLPCoreExecutionResult:
    """
    Train and evaluate one tabular MLP on the exact Model Lab
    outer holdout population.

    Important boundaries:

    - split authority belongs to app.ml.splitting;
    - preprocessing is fit on TRAIN only;
    - train/test feature transformation uses the same fitted
      preprocessor;
    - PyTorch receives only float32 transformed matrices;
    - metrics come from app.ml.model_metrics;
    - device selection is internal execution state and is not
      part of the client-controlled Training Contract.
    """

    (
        validated_x,
        validated_y,
        hyperparameters,
    ) = (
        _validate_tabular_mlp_inputs(
            x=x,
            y=y,
            contract=contract,
        )
    )


    try:

        partition = (
            resolve_ml_holdout_partition(
                x=validated_x,
                y=validated_y,
                contract=contract,
                dataframe=dataframe,
            )
        )

    except MLSplitError as error:

        raise (
            TabularMLPInputError(
                str(
                    error
                )
            )
        ) from error


    train_positions = list(
        partition.train_positions
    )

    test_positions = list(
        partition.test_positions
    )


    x_train = (
        validated_x.iloc[
            train_positions
        ]
        .copy(
            deep=True
        )
    )


    x_test = (
        validated_x.iloc[
            test_positions
        ]
        .copy(
            deep=True
        )
    )


    y_train = (
        validated_y.iloc[
            train_positions
        ]
        .copy(
            deep=True
        )
    )


    y_test = (
        validated_y.iloc[
            test_positions
        ]
        .copy(
            deep=True
        )
    )


    # ========================================================
    # LEAKAGE-SAFE PREPROCESSING
    # ========================================================


    try:

        preprocessor = (
            build_ml_preprocessor(
                contract=contract
            )
        )


        transformed_train = (
            preprocessor.fit_transform(
                x_train
            )
        )


        transformed_test = (
            preprocessor.transform(
                x_test
            )
        )

    except (
        MLPreprocessingRuntimeError,
        ValueError,
        TypeError,
    ) as error:

        raise (
            TabularMLPInputError(
                (
                    "Tabular MLP preprocessing "
                    "failed."
                )
            )
        ) from error


    train_features = (
        _as_float32_feature_matrix(
            transformed_train,
            name="transformed_train",
        )
    )


    test_features = (
        _as_float32_feature_matrix(
            transformed_test,
            name="transformed_test",
        )
    )


    if (
        train_features.shape[
            1
        ]
        !=
        test_features.shape[
            1
        ]
    ):
        raise (
            TabularMLPCoreExecutorError(
                (
                    "Train/test transformed feature "
                    "dimensions do not match."
                )
            )
        )


    train_targets = (
        _as_float32_target_vector(
            y_train,
            name="y_train",
        )
    )


    test_targets = (
        _as_float32_target_vector(
            y_test,
            name="y_test",
        )
    )


    # ========================================================
    # CPU-OWNED DATASETS
    # ========================================================


    train_dataset = (
        TabularTensorDataset(
            features=torch.from_numpy(
                train_features
            ),
            targets=torch.from_numpy(
                train_targets
            ),
        )
    )


    test_dataset = (
        TabularTensorDataset(
            features=torch.from_numpy(
                test_features
            ),
            targets=torch.from_numpy(
                test_targets
            ),
        )
    )


    random_seed = int(
        contract.split.random_seed
    )


    train_loader = (
        build_data_loader(
            train_dataset,
            batch_size=
                hyperparameters.batch_size,
            shuffle=True,
            seed=random_seed,
        )
    )


    test_loader = (
        build_data_loader(
            test_dataset,
            batch_size=
                hyperparameters.batch_size,
            shuffle=False,
            seed=random_seed,
        )
    )


    # ========================================================
    # PYTORCH EXECUTION
    # ========================================================


    seed_torch(
        random_seed
    )


    device = (
        _resolve_execution_device(
            execution_device
        )
    )


    model = (
        FeedForwardRegressor(
            input_features=int(
                train_features.shape[
                    1
                ]
            ),
            hidden_features=
                hyperparameters.hidden_features,
        )
        .to(
            device
        )
    )


    loss_function = (
        build_regression_loss()
    )


    optimizer = (
        build_sgd_optimizer(
            model,
            learning_rate=
                hyperparameters.learning_rate,
        )
    )


    epoch_losses = (
        train_regression_epochs(
            model,
            data_loader=train_loader,
            optimizer=optimizer,
            loss_function=loss_function,
            epochs=
                hyperparameters.epochs,
            device=device,
        )
    )


    if (
        not epoch_losses
        or
        not all(
            math.isfinite(
                float(
                    value
                )
            )

            for value
            in epoch_losses
        )
    ):
        raise (
            TabularMLPCoreExecutorError(
                (
                    "Tabular MLP training produced "
                    "a non-finite loss."
                )
            )
        )


    evaluation = (
        evaluate_regression_loader(
            model,
            data_loader=test_loader,
            loss_function=loss_function,
            device=device,
        )
    )


    if not math.isfinite(
        float(
            evaluation.mean_loss
        )
    ):
        raise (
            TabularMLPCoreExecutorError(
                (
                    "Tabular MLP evaluation produced "
                    "a non-finite test loss."
                )
            )
        )


    predictions_array = (
        evaluation
        .predictions
        .numpy()
        .astype(
            np.float64,
            copy=True,
        )
    )


    if not (
        np.isfinite(
            predictions_array
        )
        .all()
    ):
        raise (
            TabularMLPCoreExecutorError(
                (
                    "Tabular MLP evaluation produced "
                    "non-finite predictions."
                )
            )
        )


    predictions = tuple(
        float(
            value
        )

        for value
        in predictions_array.tolist()
    )


    try:

        metrics = (
            compute_ml_regression_metrics(
                y_true=y_test,
                predictions=
                    predictions_array,
            )
        )

    except MLModelMetricsError as error:

        raise (
            TabularMLPCoreExecutorError(
                (
                    "Canonical Model Lab regression "
                    "metrics could not be computed."
                )
            )
        ) from error


    return (
        TabularMLPCoreExecutionResult(
            estimator_key=
                contract.estimator_key,

            partition=
                partition,

            input_features=int(
                train_features.shape[
                    1
                ]
            ),

            random_seed=
                random_seed,

            execution_device=
                str(
                    device
                ),

            epoch_losses=tuple(
                float(
                    value
                )

                for value
                in epoch_losses
            ),

            test_loss=float(
                evaluation.mean_loss
            ),

            metrics=dict(
                metrics
            ),

            predictions=
                predictions,

            model=
                model,

            preprocessor=
                preprocessor,
        )
    )
