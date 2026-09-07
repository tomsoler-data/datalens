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


from app.deep_learning.autoencoder_contracts import (
    DLTabularAutoencoderHyperparameters,
)


from app.deep_learning.autoencoder_dataset import (
    ReconstructionTensorDataset,
)


from app.deep_learning.autoencoder_evaluation import (
    ReconstructionEvaluation,
    evaluate_reconstruction_loader,
)


from app.deep_learning.autoencoder_network import (
    TabularAutoencoder,
)


from app.deep_learning.autoencoder_threshold import (
    ReconstructionErrorThreshold,
    apply_reconstruction_error_threshold,
    fit_reconstruction_error_threshold,
)


from app.deep_learning.autoencoder_training import (
    train_reconstruction_epochs,
)


from app.deep_learning.datasets import (
    build_data_loader,
)


from app.deep_learning.runtime import (
    resolve_device,
    seed_torch,
)


from app.deep_learning.training import (
    build_regression_loss,
    build_sgd_optimizer,
)


from app.ml.anomaly_contracts import (
    MLAnomalyTrainingContract,
)


from app.ml.preprocessing import (
    MLPreprocessingRuntimeError,
    build_ml_preprocessor,
    validate_ml_feature_frame,
)


from app.ml.splitting import (
    MLHoldoutPartition,
    MLSplitError,
    resolve_ml_feature_holdout_partition,
)


# ============================================================
# VERSION
# ============================================================


DL_TABULAR_AUTOENCODER_CORE_EXECUTOR_RULE_VERSION = (
    "dl_tabular_autoencoder_core_executor_v0.1"
)


# ============================================================
# ERRORS
# ============================================================


class TabularAutoencoderCoreExecutorError(
    RuntimeError
):
    pass


class TabularAutoencoderInputError(
    TabularAutoencoderCoreExecutorError
):
    pass


class TabularAutoencoderEstimatorError(
    TabularAutoencoderCoreExecutorError
):
    pass


# ============================================================
# INTERNAL RESULT
# ============================================================


@dataclass(
    frozen=True,
)
class TabularAutoencoderCoreExecutionResult:
    """
    Internal trained autoencoder execution bundle.

    Learned model state, fitted preprocessing state, row-level
    reconstruction errors and row-level anomaly flags remain
    internal.

    A8 may serialize the already-trained model/preprocessor
    without repeating split or training.
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

    train_evaluation: ReconstructionEvaluation

    test_evaluation: ReconstructionEvaluation

    threshold: ReconstructionErrorThreshold

    train_flags: torch.Tensor

    test_flags: torch.Tensor

    model: TabularAutoencoder

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


    @property
    def train_anomaly_count(
        self,
    ) -> int:

        return int(
            torch.count_nonzero(
                self.train_flags
            )
            .item()
        )


    @property
    def test_anomaly_count(
        self,
    ) -> int:

        return int(
            torch.count_nonzero(
                self.test_flags
            )
            .item()
        )


    @property
    def train_anomaly_rate(
        self,
    ) -> float:

        return float(
            self.train_anomaly_count
            /
            self.train_rows
        )


    @property
    def test_anomaly_rate(
        self,
    ) -> float:

        return float(
            self.test_anomaly_count
            /
            self.test_rows
        )


# ============================================================
# INPUT VALIDATION
# ============================================================


def _validate_autoencoder_inputs(
    *,
    x: pd.DataFrame,
    contract: MLAnomalyTrainingContract,
) -> tuple[
    pd.DataFrame,
    DLTabularAutoencoderHyperparameters,
]:

    if not isinstance(
        contract,
        MLAnomalyTrainingContract,
    ):

        raise (
            TabularAutoencoderInputError(
                (
                    "contract must be an "
                    "MLAnomalyTrainingContract."
                )
            )
        )


    if (
        contract.problem_type
        !=
        "anomaly_detection"
    ):

        raise (
            TabularAutoencoderEstimatorError(
                (
                    "Tabular autoencoder core executor "
                    "only supports anomaly_detection."
                )
            )
        )


    if (
        contract.estimator_key
        !=
        "tabular_autoencoder"
    ):

        raise (
            TabularAutoencoderEstimatorError(
                (
                    "Tabular autoencoder core executor "
                    "requires estimator_key="
                    "'tabular_autoencoder'."
                )
            )
        )


    hyperparameters = (
        contract.estimator_hyperparameters
    )


    if not isinstance(
        hyperparameters,
        DLTabularAutoencoderHyperparameters,
    ):

        raise (
            TabularAutoencoderEstimatorError(
                (
                    "Tabular autoencoder estimator did "
                    "not resolve to its typed "
                    "hyperparameter contract."
                )
            )
        )


    if not isinstance(
        x,
        pd.DataFrame,
    ):

        raise (
            TabularAutoencoderInputError(
                "x must be a pandas DataFrame."
            )
        )


    try:

        validated_x = (
            validate_ml_feature_frame(
                features=
                    x,
                contract=
                    contract,
            )
        )

    except MLPreprocessingRuntimeError as error:

        raise (
            TabularAutoencoderInputError(
                str(
                    error
                )
            )
        ) from error


    return (
        validated_x,
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
            dtype=
                np.float32,
        )

    except Exception as error:

        raise (
            TabularAutoencoderInputError(
                (
                    f"{name} could not be converted "
                    "to a float32 feature matrix."
                )
            )
        ) from error


    if array.ndim != 2:

        raise (
            TabularAutoencoderInputError(
                f"{name} must be two-dimensional."
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
            TabularAutoencoderInputError(
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
            TabularAutoencoderInputError(
                (
                    f"{name} contains non-finite "
                    "transformed values."
                )
            )
        )


    return np.ascontiguousarray(
        array,
        dtype=
            np.float32,
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
            TabularAutoencoderCoreExecutorError(
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
            TabularAutoencoderCoreExecutorError(
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


def execute_tabular_autoencoder_core(
    *,
    x: pd.DataFrame,
    contract: MLAnomalyTrainingContract,
    dataframe: pd.DataFrame | None = None,
    execution_device: (
        str
        |
        torch.device
        |
        None
    ) = None,
) -> TabularAutoencoderCoreExecutionResult:
    """
    Train and evaluate one feature-only tabular autoencoder.

    Boundaries:
    - no y / target exists;
    - split authority is target-free;
    - preprocessing is fit on TRAIN only;
    - threshold is fit from TRAIN reconstruction errors only;
    - TEST is application-only for the frozen threshold.
    """

    (
        validated_x,
        hyperparameters,
    ) = (
        _validate_autoencoder_inputs(
            x=
                x,
            contract=
                contract,
        )
    )


    try:

        partition = (
            resolve_ml_feature_holdout_partition(
                x=
                    validated_x,
                contract=
                    contract,
                dataframe=
                    dataframe,
            )
        )

    except MLSplitError as error:

        raise (
            TabularAutoencoderInputError(
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


    # ========================================================
    # TRAIN-ONLY PREPROCESSING
    # ========================================================


    try:

        preprocessor = (
            build_ml_preprocessor(
                contract=
                    contract,
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
            TabularAutoencoderInputError(
                (
                    "Tabular autoencoder preprocessing "
                    "failed."
                )
            )
        ) from error


    train_features = (
        _as_float32_feature_matrix(
            transformed_train,
            name=
                "transformed_train",
        )
    )


    test_features = (
        _as_float32_feature_matrix(
            transformed_test,
            name=
                "transformed_test",
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
            TabularAutoencoderCoreExecutorError(
                (
                    "Train/test transformed feature "
                    "dimensions do not match."
                )
            )
        )


    # ========================================================
    # CPU-OWNED FEATURE-ONLY DATASETS
    # ========================================================


    train_dataset = (
        ReconstructionTensorDataset(
            features=
                torch.from_numpy(
                    train_features
                ),
        )
    )


    test_dataset = (
        ReconstructionTensorDataset(
            features=
                torch.from_numpy(
                    test_features
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
            shuffle=
                True,
            seed=
                random_seed,
        )
    )


    train_evaluation_loader = (
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


    test_evaluation_loader = (
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
        TabularAutoencoder(
            input_features=
                int(
                    train_features.shape[
                        1
                    ]
                ),
            hidden_features=
                hyperparameters.hidden_features,
            latent_features=
                hyperparameters.latent_features,
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
        train_reconstruction_epochs(
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
            TabularAutoencoderCoreExecutorError(
                (
                    "Tabular autoencoder training "
                    "produced a non-finite loss."
                )
            )
        )


    # ========================================================
    # RECONSTRUCTION EVALUATION
    # ========================================================


    train_evaluation = (
        evaluate_reconstruction_loader(
            model,
            data_loader=
                train_evaluation_loader,
            device=
                device,
        )
    )


    test_evaluation = (
        evaluate_reconstruction_loader(
            model,
            data_loader=
                test_evaluation_loader,
            device=
                device,
        )
    )


    if (
        not math.isfinite(
            float(
                train_evaluation.mean_loss
            )
        )
        or
        not math.isfinite(
            float(
                test_evaluation.mean_loss
            )
        )
    ):

        raise (
            TabularAutoencoderCoreExecutorError(
                (
                    "Tabular autoencoder evaluation "
                    "produced a non-finite loss."
                )
            )
        )


    # ========================================================
    # TRAIN-ONLY THRESHOLD
    # ========================================================


    threshold = (
        fit_reconstruction_error_threshold(
            train_errors=
                train_evaluation
                .reconstruction_errors,
            quantile=
                contract.threshold_quantile,
        )
    )


    train_flags = (
        apply_reconstruction_error_threshold(
            reconstruction_errors=
                train_evaluation
                .reconstruction_errors,
            threshold=
                threshold,
        )
    )


    test_flags = (
        apply_reconstruction_error_threshold(
            reconstruction_errors=
                test_evaluation
                .reconstruction_errors,
            threshold=
                threshold,
        )
    )


    return (
        TabularAutoencoderCoreExecutionResult(
            estimator_key=
                contract.estimator_key,
            partition=
                partition,
            input_features=
                int(
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
            epoch_losses=
                tuple(
                    float(
                        value
                    )

                    for value
                    in epoch_losses
                ),
            train_evaluation=
                train_evaluation,
            test_evaluation=
                test_evaluation,
            threshold=
                threshold,
            train_flags=
                train_flags,
            test_flags=
                test_flags,
            model=
                model,
            preprocessor=
                preprocessor,
        )
    )
