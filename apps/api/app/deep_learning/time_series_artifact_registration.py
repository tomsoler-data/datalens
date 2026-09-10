from __future__ import annotations


import math
import numpy as np


from app.deep_learning.time_series_bundle import (
    DLTimeSeriesNeuralBundleError,
    serialize_time_series_neural_bundle,
)


from app.deep_learning.time_series_lstm_executor import (
    DLTimeSeriesLSTMExecutionResult,
)


from app.deep_learning.time_series_mlp_executor import (
    DLTimeSeriesMLPExecutionResult,
)


from app.deep_learning.time_series_rnn_executor import (
    DLTimeSeriesRNNExecutionResult,
)


from app.ml.model_artifact_store import (
    MLModelArtifactStoreError,
    register_ml_model_artifact,
)


from app.ml.model_artifacts import (
    MLModelArtifactRecord,
)


from app.ml.time_series_model_training_contracts import (
    MLTimeSeriesModelTrainingContract,
)


from app.ml.time_series_windows import (
    MLTimeSeriesSupervisedWindows,
)


# ============================================================
# VERSION
# ============================================================


DL_TIME_SERIES_ARTIFACT_REGISTRATION_RULE_VERSION = (
    "dl_time_series_artifact_registration_v0.1"
)


# ============================================================
# ERRORS
# ============================================================


class DLTimeSeriesArtifactRegistrationError(
    RuntimeError
):
    pass


class DLTimeSeriesArtifactRegistrationContractError(
    DLTimeSeriesArtifactRegistrationError
):
    pass


class DLTimeSeriesArtifactRegistrationStoreError(
    DLTimeSeriesArtifactRegistrationError
):
    pass


# ============================================================
# EXECUTION FAMILY
# ============================================================


DLTimeSeriesPersistableExecution = (
    DLTimeSeriesMLPExecutionResult
    |
    DLTimeSeriesRNNExecutionResult
    |
    DLTimeSeriesLSTMExecutionResult
)


# ============================================================
# EXPECTED EXECUTION TYPE
# ============================================================


def _expected_execution_type(
    estimator_key: str,
):

    if (
        estimator_key
        ==
        "time_series_mlp_regressor"
    ):

        return (
            DLTimeSeriesMLPExecutionResult
        )


    if (
        estimator_key
        ==
        "time_series_rnn_regressor"
    ):

        return (
            DLTimeSeriesRNNExecutionResult
        )


    if (
        estimator_key
        ==
        "time_series_lstm_regressor"
    ):

        return (
            DLTimeSeriesLSTMExecutionResult
        )


    raise DLTimeSeriesArtifactRegistrationContractError(
        (
            "Unsupported forecasting Artifact estimator. "
            f"estimator_key={estimator_key!r}"
        )
    )


# ============================================================
# METRICS
# ============================================================


def _artifact_metrics(
    execution: DLTimeSeriesPersistableExecution,
) -> dict[
    str,
    float,
]:

    metrics = {
        "mae":
            float(
                execution.metrics.mae
            ),

        "mse":
            float(
                execution.metrics.mse
            ),

        "rmse":
            float(
                execution.metrics.rmse
            ),

        "mean_error":
            float(
                execution.metrics.mean_error
            ),
    }


    if not all(
        math.isfinite(
            value
        )
        for value
        in metrics.values()
    ):

        raise DLTimeSeriesArtifactRegistrationContractError(
            (
                "Forecast Artifact metrics must "
                "all be finite."
            )
        )


    return metrics


# ============================================================
# EXECUTION / WINDOW AUTHORITY
# ============================================================


def _validate_execution_evidence(
    *,
    windows: MLTimeSeriesSupervisedWindows,
    contract: MLTimeSeriesModelTrainingContract,
    execution: object,
) -> DLTimeSeriesPersistableExecution:

    expected_type = (
        _expected_execution_type(
            contract.estimator_key
        )
    )


    if not isinstance(
        execution,
        expected_type,
    ):

        raise DLTimeSeriesArtifactRegistrationContractError(
            (
                "Forecast execution result type does not "
                "match the Model Training Contract. "
                f"estimator_key={contract.estimator_key!r}"
            )
        )


    if (
        execution.estimator_key
        !=
        contract.estimator_key
    ):

        raise DLTimeSeriesArtifactRegistrationContractError(
            (
                "Forecast execution estimator identity "
                "does not match the Model Training Contract."
            )
        )


    if (
        execution.metrics.sample_count
        !=
        windows.test.sample_count
    ):

        raise DLTimeSeriesArtifactRegistrationContractError(
            (
                "Forecast execution metric sample count "
                "does not match the A4 TEST population."
            )
        )


    if (
        execution.target_positions
        !=
        windows.test.target_positions
    ):

        raise DLTimeSeriesArtifactRegistrationContractError(
            (
                "Forecast execution target positions do "
                "not match the A4 TEST population."
            )
        )


    if not np.array_equal(
        np.asarray(
            execution.targets,
            dtype=np.float64,
        ),
        np.asarray(
            windows.test.targets,
            dtype=np.float64,
        ),
    ):

        raise DLTimeSeriesArtifactRegistrationContractError(
            (
                "Forecast execution targets do not match "
                "the exact A4 TEST population."
            )
        )


    prediction_values = np.asarray(
        execution.predictions,
        dtype=np.float64,
    )


    if (
        prediction_values.ndim
        !=
        1
    ):

        raise DLTimeSeriesArtifactRegistrationContractError(
            (
                "Forecast execution predictions must "
                "be one-dimensional."
            )
        )


    if (
        prediction_values.size
        !=
        windows.test.sample_count
    ):

        raise DLTimeSeriesArtifactRegistrationContractError(
            (
                "Forecast execution prediction count "
                "does not match the A4 TEST population."
            )
        )


    if not np.isfinite(
        prediction_values
    ).all():

        raise DLTimeSeriesArtifactRegistrationContractError(
            (
                "Forecast execution predictions "
                "contain non-finite values."
            )
        )


    return execution


# ============================================================
# PUBLIC REGISTRATION
# ============================================================


def register_time_series_neural_execution_artifact(
    *,
    windows: MLTimeSeriesSupervisedWindows,
    training_contract: MLTimeSeriesModelTrainingContract,
    execution: DLTimeSeriesPersistableExecution,
    preparation_session_revision: int,
) -> MLModelArtifactRecord:
    """
    Persist one already-trained DL-5 forecasting neural model.

    Lifecycle authority:

        exact A4 windows
                |
        already-trained execution
                |
        per-model Forecast Training Contract
                |
        temporal .ptbundle
                |
        generic Model Artifact Store
                |
        generic Experiment Provenance

    There is deliberately no training, split generation or
    metric recomputation in this adapter.
    """

    if not isinstance(
        windows,
        MLTimeSeriesSupervisedWindows,
    ):

        raise DLTimeSeriesArtifactRegistrationContractError(
            (
                "Forecast Artifact registration requires "
                "validated A4 supervised windows."
            )
        )


    try:

        contract = (
            MLTimeSeriesModelTrainingContract
            .model_validate(
                training_contract
            )
        )

    except Exception as error:

        raise DLTimeSeriesArtifactRegistrationContractError(
            (
                "Forecast Artifact registration requires "
                "a valid per-model Training Contract."
            )
        ) from error


    if isinstance(
        preparation_session_revision,
        bool,
    ):

        raise DLTimeSeriesArtifactRegistrationContractError(
            (
                "Preparation revision must be "
                "a non-negative integer."
            )
        )


    if not isinstance(
        preparation_session_revision,
        int,
    ):

        raise DLTimeSeriesArtifactRegistrationContractError(
            (
                "Preparation revision must be "
                "a non-negative integer."
            )
        )


    if (
        preparation_session_revision
        <
        0
    ):

        raise DLTimeSeriesArtifactRegistrationContractError(
            (
                "Preparation revision must be "
                "a non-negative integer."
            )
        )


    validated_execution = (
        _validate_execution_evidence(
            windows=
                windows,

            contract=
                contract,

            execution=
                execution,
        )
    )


    metrics = (
        _artifact_metrics(
            validated_execution
        )
    )


    try:

        model_bytes = (
            serialize_time_series_neural_bundle(
                model=
                    validated_execution.model,

                standardizer=
                    validated_execution.standardizer,

                training_contract=
                    contract,
            )
        )

    except DLTimeSeriesNeuralBundleError as error:

        raise DLTimeSeriesArtifactRegistrationContractError(
            (
                "Forecast training completed but its "
                "temporal PyTorch bundle could not "
                "be serialized."
            )
        ) from error


    try:

        artifact = (
            register_ml_model_artifact(
                training_contract=
                    contract,

                metrics=
                    metrics,

                train_rows=
                    windows.train.sample_count,

                test_rows=
                    windows.test.sample_count,

                model_bytes=
                    model_bytes,

                serialization_format=
                    "pytorch_bundle",

                preparation_session_revision=
                    preparation_session_revision,
            )
        )

    except MLModelArtifactStoreError as error:

        raise DLTimeSeriesArtifactRegistrationStoreError(
            (
                "Forecast temporal PyTorch Artifact "
                "could not be persisted."
            )
        ) from error


    if (
        artifact.experiment_provenance
        is None
    ):

        raise DLTimeSeriesArtifactRegistrationStoreError(
            (
                "Forecast Artifact did not persist "
                "required Experiment Provenance."
            )
        )


    if (
        artifact.training_contract.estimator_key
        !=
        contract.estimator_key
    ):

        raise DLTimeSeriesArtifactRegistrationStoreError(
            (
                "Persisted Forecast Artifact estimator "
                "identity changed unexpectedly."
            )
        )


    return artifact
