from __future__ import annotations


from dataclasses import (
    dataclass,
)


import math
import numpy as np
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


from app.deep_learning.time_series_contracts import (
    DL_TIME_SERIES_MODEL_RANDOM_SEED,
    DLTimeSeriesMLPRegressorHyperparameters,
)


from app.deep_learning.time_series_scaling import (
    DLTimeSeriesScalingError,
    DLTimeSeriesStandardizer,
    fit_time_series_train_standardizer,
)


from app.deep_learning.training import (
    build_regression_loss,
    build_sgd_optimizer,
    train_regression_epochs,
)


from app.ml.time_series_evaluation import (
    MLForecastRegressionMetrics,
    evaluate_forecast_predictions,
)


from app.ml.time_series_windows import (
    MLTimeSeriesSupervisedWindows,
)


# ============================================================
# VERSION
# ============================================================


DL_TIME_SERIES_MLP_EXECUTOR_RULE_VERSION = (
    "dl_time_series_mlp_executor_v0.1"
)


# ============================================================
# ERRORS
# ============================================================


class DLTimeSeriesMLPExecutorError(
    RuntimeError
):
    pass


# ============================================================
# RESULT
# ============================================================


@dataclass(
    frozen=True
)
class DLTimeSeriesMLPExecutionResult:
    """
    Internal trained temporal-MLP execution result.

    model and standardizer are deliberately retained so later
    artifact work can persist the already-trained state without
    retraining it.
    """

    estimator_key: str

    random_seed: int

    execution_device: str

    input_features: int

    epoch_losses: tuple[
        float,
        ...
    ]

    scaled_test_loss: float

    predictions: np.ndarray

    targets: np.ndarray

    target_positions: tuple[
        int,
        ...
    ]

    metrics: MLForecastRegressionMetrics

    model: FeedForwardRegressor

    standardizer: DLTimeSeriesStandardizer

    rule_version: str = (
        DL_TIME_SERIES_MLP_EXECUTOR_RULE_VERSION
    )



    @property
    def test_samples(
        self,
    ) -> int:

        return int(
            self.predictions.size
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
            DLTimeSeriesMLPExecutorError(
                "Invalid internal PyTorch execution device."
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
            DLTimeSeriesMLPExecutorError(
                (
                    "CUDA execution was requested but "
                    "CUDA is unavailable."
                )
            )
        )


    return device


# ============================================================
# FLOAT32 AUTHORITY
# ============================================================


def _as_float32_matrix(
    values: object,
    *,
    label: str,
) -> np.ndarray:

    try:

        array = np.asarray(
            values,
            dtype=np.float32,
        )

    except Exception as error:

        raise (
            DLTimeSeriesMLPExecutorError(
                (
                    f"{label} could not be converted "
                    "to float32."
                )
            )
        ) from error


    if (
        array.ndim
        !=
        2
        or
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
            DLTimeSeriesMLPExecutorError(
                (
                    f"{label} must be a non-empty "
                    "two-dimensional matrix."
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
            DLTimeSeriesMLPExecutorError(
                f"{label} contains non-finite values."
            )
        )


    return np.ascontiguousarray(
        array,
        dtype=np.float32,
    )


def _as_float32_vector(
    values: object,
    *,
    label: str,
) -> np.ndarray:

    try:

        array = np.asarray(
            values,
            dtype=np.float32,
        )

    except Exception as error:

        raise (
            DLTimeSeriesMLPExecutorError(
                (
                    f"{label} could not be converted "
                    "to float32."
                )
            )
        ) from error


    if (
        array.ndim
        !=
        1
        or
        array.shape[
            0
        ]
        <=
        0
    ):

        raise (
            DLTimeSeriesMLPExecutorError(
                (
                    f"{label} must be a non-empty "
                    "one-dimensional vector."
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
            DLTimeSeriesMLPExecutorError(
                f"{label} contains non-finite values."
            )
        )


    return np.ascontiguousarray(
        array,
        dtype=np.float32,
    )


# ============================================================
# EXECUTION
# ============================================================


def execute_time_series_mlp(
    *,
    windows: MLTimeSeriesSupervisedWindows,
    hyperparameters: DLTimeSeriesMLPRegressorHyperparameters,
    execution_device: (
        str
        |
        torch.device
        |
        None
    ) = None,
) -> DLTimeSeriesMLPExecutionResult:
    """
    Train and evaluate one temporal feed-forward MLP.

    Important boundaries:

    - supervised samples belong to app.ml.time_series_windows;
    - scaling is fitted from TRAIN observations only;
    - TEST values never influence fitting;
    - existing DL-2 FeedForwardRegressor is reused;
    - model input width is exactly forecasting lookback;
    - PyTorch receives float32 CPU-owned datasets;
    - predictions are inverse-transformed to original units;
    - final metrics use the shared A5 forecasting evaluator.
    """

    if not isinstance(
        windows,
        MLTimeSeriesSupervisedWindows,
    ):

        raise (
            DLTimeSeriesMLPExecutorError(
                (
                    "Temporal MLP requires validated "
                    "supervised forecasting windows."
                )
            )
        )


    hyperparameters = (
        DLTimeSeriesMLPRegressorHyperparameters
        .model_validate(
            hyperparameters
        )
    )


    train = (
        windows.train
    )

    test = (
        windows.test
    )


    if (
        train.sample_count
        <=
        0
        or
        test.sample_count
        <=
        0
    ):

        raise (
            DLTimeSeriesMLPExecutorError(
                (
                    "Temporal MLP requires non-empty "
                    "TRAIN and TEST sample populations."
                )
            )
        )


    if (
        train.lookback
        !=
        windows.lookback
        or
        test.lookback
        !=
        windows.lookback
    ):

        raise (
            DLTimeSeriesMLPExecutorError(
                (
                    "Temporal MLP window dimensions do not "
                    "match the forecasting lookback."
                )
            )
        )


    if (
        len(
            test.target_positions
        )
        !=
        test.sample_count
    ):

        raise (
            DLTimeSeriesMLPExecutorError(
                (
                    "Temporal MLP TEST target-position "
                    "provenance is not sample-aligned."
                )
            )
        )


    # ========================================================
    # TRAIN-ONLY STANDARDIZATION
    # ========================================================

    try:

        standardizer = (
            fit_time_series_train_standardizer(
                windows=
                    windows
            )
        )

    except DLTimeSeriesScalingError as error:

        raise (
            DLTimeSeriesMLPExecutorError(
                str(
                    error
                )
            )
        ) from error


    try:

        scaled_train_inputs = (
            standardizer.transform(
                train.inputs
            )
        )

        scaled_train_targets = (
            standardizer.transform(
                train.targets
            )
        )

        scaled_test_inputs = (
            standardizer.transform(
                test.inputs
            )
        )

        scaled_test_targets = (
            standardizer.transform(
                test.targets
            )
        )

    except DLTimeSeriesScalingError as error:

        raise (
            DLTimeSeriesMLPExecutorError(
                str(
                    error
                )
            )
        ) from error


    train_features = (
        _as_float32_matrix(
            scaled_train_inputs,
            label=
                "scaled_train_inputs",
        )
    )


    train_targets = (
        _as_float32_vector(
            scaled_train_targets,
            label=
                "scaled_train_targets",
        )
    )


    test_features = (
        _as_float32_matrix(
            scaled_test_inputs,
            label=
                "scaled_test_inputs",
        )
    )


    test_targets_scaled = (
        _as_float32_vector(
            scaled_test_targets,
            label=
                "scaled_test_targets",
        )
    )


    # ========================================================
    # CPU-OWNED DATASETS
    # ========================================================

    train_dataset = (
        TabularTensorDataset(
            features=
                torch.from_numpy(
                    train_features
                ),

            targets=
                torch.from_numpy(
                    train_targets
                ),
        )
    )


    test_dataset = (
        TabularTensorDataset(
            features=
                torch.from_numpy(
                    test_features
                ),

            targets=
                torch.from_numpy(
                    test_targets_scaled
                ),
        )
    )


    random_seed = (
        DL_TIME_SERIES_MODEL_RANDOM_SEED
    )


    train_loader = (
        build_data_loader(
            train_dataset,
            batch_size=
                hyperparameters.batch_size,

            shuffle=
                False,

            seed=
                random_seed,
        )
    )


    test_loader = (
        build_data_loader(
            test_dataset,
            batch_size=
                hyperparameters.batch_size,

            shuffle=
                False,

            seed=
                random_seed,
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
            input_features=
                int(
                    windows.lookback
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
            data_loader=
                train_loader,

            optimizer=
                optimizer,

            loss_function=
                loss_function,

            epochs=
                hyperparameters.epochs,

            device=
                device,
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
            DLTimeSeriesMLPExecutorError(
                (
                    "Temporal MLP training produced "
                    "non-finite loss values."
                )
            )
        )


    evaluation = (
        evaluate_regression_loader(
            model,
            data_loader=
                test_loader,

            loss_function=
                loss_function,

            device=
                device,
        )
    )


    if not math.isfinite(
        float(
            evaluation.mean_loss
        )
    ):

        raise (
            DLTimeSeriesMLPExecutorError(
                (
                    "Temporal MLP evaluation produced "
                    "a non-finite scaled TEST loss."
                )
            )
        )


    scaled_predictions = (
        evaluation.predictions
        .numpy()
        .astype(
            np.float64,
            copy=True,
        )
    )


    if not (
        np.isfinite(
            scaled_predictions
        )
        .all()
    ):

        raise (
            DLTimeSeriesMLPExecutorError(
                (
                    "Temporal MLP evaluation produced "
                    "non-finite predictions."
                )
            )
        )


    try:

        predictions = (
            standardizer.inverse_transform(
                scaled_predictions
            )
        )

    except DLTimeSeriesScalingError as error:

        raise (
            DLTimeSeriesMLPExecutorError(
                str(
                    error
                )
            )
        ) from error


    targets = np.asarray(
        test.targets,
        dtype=np.float64,
    ).copy()


    metrics = (
        evaluate_forecast_predictions(
            predictions=
                predictions,

            targets=
                targets,
        )
    )


    predictions.setflags(
        write=False
    )


    targets.setflags(
        write=False
    )


    return (
        DLTimeSeriesMLPExecutionResult(
            estimator_key=
                hyperparameters.kind,

            random_seed=
                random_seed,

            execution_device=
                str(
                    device
                ),

            input_features=
                int(
                    windows.lookback
                ),

            epoch_losses=
                tuple(
                    float(
                        value
                    )
                    for value
                    in epoch_losses
                ),

            scaled_test_loss=
                float(
                    evaluation.mean_loss
                ),

            predictions=
                predictions,

            targets=
                targets,

            target_positions=
                tuple(
                    int(
                        position
                    )
                    for position
                    in test.target_positions
                ),

            metrics=
                metrics,

            model=
                model,

            standardizer=
                standardizer,
        )
    )
