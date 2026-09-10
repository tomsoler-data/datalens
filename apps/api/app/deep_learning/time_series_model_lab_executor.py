from __future__ import annotations


import math


from typing import (
    Literal,
)


from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)


from app.deep_learning.time_series_artifact_registration import (
    DLTimeSeriesArtifactRegistrationError,
    register_time_series_neural_execution_artifact,
)


from app.deep_learning.time_series_bundle_contracts import (
    DLTimeSeriesBundleEstimatorKey,
)


from app.deep_learning.time_series_lstm_executor import (
    DLTimeSeriesLSTMExecutorError,
    execute_time_series_lstm,
)


from app.deep_learning.time_series_mlp_executor import (
    DLTimeSeriesMLPExecutorError,
    execute_time_series_mlp,
)


from app.deep_learning.time_series_rnn_executor import (
    DLTimeSeriesRNNExecutorError,
    execute_time_series_rnn,
)


from app.ml.experiment_provenance import (
    MLExperimentProvenanceRecord,
)


from app.ml.model_artifacts import (
    MLModelArtifactRecord,
)


from app.ml.time_series_baseline import (
    MLTimeSeriesNaiveBaselineError,
    evaluate_last_value_baseline,
)


from app.ml.time_series_holdout import (
    resolve_time_series_holdout,
)


from app.ml.time_series_model_training_contracts import (
    MLTimeSeriesModelTrainingContract,
)


from app.ml.time_series_training_input import (
    validate_and_extract_time_series,
)


from app.ml.time_series_windows import (
    build_time_series_supervised_windows,
)


from app.ml.training_input import (
    MLTrainingInputError,
    load_authorized_ml_dataframe,
)


# ============================================================
# VERSION
# ============================================================


DL_TIME_SERIES_MODEL_LAB_EXECUTOR_RULE_VERSION = (
    "dl_time_series_model_lab_executor_v0.1"
)


# ============================================================
# ERRORS
# ============================================================


class DLTimeSeriesModelLabExecutionError(
    RuntimeError
):
    pass


class DLTimeSeriesModelLabInputError(
    DLTimeSeriesModelLabExecutionError
):
    pass


class DLTimeSeriesModelLabEstimatorError(
    DLTimeSeriesModelLabExecutionError
):
    pass


class DLTimeSeriesModelLabArtifactError(
    DLTimeSeriesModelLabExecutionError
):
    pass


# ============================================================
# PRIVACY-MINIMAL RESULT
# ============================================================


class DLTimeSeriesModelLabExecutionResult(
    BaseModel
):
    """
    Production-facing Model Lab result for one forecasting
    estimator.

    Deliberately absent:

    - raw dataframe;
    - time-series observations;
    - TRAIN / TEST windows;
    - source-row positions;
    - predictions;
    - targets;
    - fitted standardizer;
    - fitted model;
    - epoch losses.

    Learned state lives only in the server-owned Model Artifact.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )


    workflow_id: str = Field(
        min_length=1,
    )


    dataset_id: str = Field(
        min_length=1,
    )


    preparation_session_revision: int = Field(
        ge=0,
        strict=True,
    )


    task_family: Literal[
        "time_series_forecasting"
    ] = "time_series_forecasting"


    problem_type: Literal[
        "regression"
    ] = "regression"


    estimator_key: (
        DLTimeSeriesBundleEstimatorKey
    )


    train_rows: int = Field(
        gt=0,
        strict=True,
    )


    test_rows: int = Field(
        gt=0,
        strict=True,
    )


    metrics: dict[
        str,
        float,
    ]


    naive_baseline_metrics: dict[
        str,
        float,
    ]


    beats_naive_baseline: bool


    rmse_delta_vs_naive: float


    experiment_provenance: (
        MLExperimentProvenanceRecord
    )


    model_artifact: (
        MLModelArtifactRecord
    )


    rule_version: Literal[
        "dl_time_series_model_lab_executor_v0.1"
    ] = (
        DL_TIME_SERIES_MODEL_LAB_EXECUTOR_RULE_VERSION
    )


# ============================================================
# PREPARATION REVISION PIN
# ============================================================


def _validate_preparation_revision_pin(
    *,
    actual_revision: int,
    expected_revision: int | None,
) -> None:

    if expected_revision is None:
        return


    if isinstance(
        expected_revision,
        bool,
    ):

        raise DLTimeSeriesModelLabInputError(
            (
                "Expected Preparation revision must "
                "be a non-negative integer."
            )
        )


    if not isinstance(
        expected_revision,
        int,
    ):

        raise DLTimeSeriesModelLabInputError(
            (
                "Expected Preparation revision must "
                "be a non-negative integer."
            )
        )


    if expected_revision < 0:

        raise DLTimeSeriesModelLabInputError(
            (
                "Expected Preparation revision must "
                "be a non-negative integer."
            )
        )


    if (
        int(
            actual_revision
        )
        !=
        expected_revision
    ):

        raise DLTimeSeriesModelLabInputError(
            (
                "Time-series Model Lab execution refused "
                "because the validated Preparation revision "
                "changed before training."
            )
        )


# ============================================================
# METRIC PROJECTION
# ============================================================


def _forecast_metrics(
    metrics,
) -> dict[
    str,
    float,
]:

    projected = {
        "mae":
            float(
                metrics.mae
            ),

        "mse":
            float(
                metrics.mse
            ),

        "rmse":
            float(
                metrics.rmse
            ),

        "mean_error":
            float(
                metrics.mean_error
            ),
    }


    if not all(
        math.isfinite(
            value
        )
        for value
        in projected.values()
    ):

        raise DLTimeSeriesModelLabExecutionError(
            (
                "Forecast evaluation produced "
                "non-finite metrics."
            )
        )


    return projected


# ============================================================
# ESTIMATOR DISPATCH
# ============================================================


def _execute_selected_estimator(
    *,
    windows,
    contract: MLTimeSeriesModelTrainingContract,
    execution_device: str | None,
):

    estimator_key = (
        contract.estimator_key
    )


    hyperparameters = (
        contract
        .effective_estimator_hyperparameters
    )


    try:

        if (
            estimator_key
            ==
            "time_series_mlp_regressor"
        ):

            return (
                execute_time_series_mlp(
                    windows=
                        windows,

                    hyperparameters=
                        hyperparameters,

                    execution_device=
                        execution_device,
                )
            )


        if (
            estimator_key
            ==
            "time_series_rnn_regressor"
        ):

            return (
                execute_time_series_rnn(
                    windows=
                        windows,

                    hyperparameters=
                        hyperparameters,

                    execution_device=
                        execution_device,
                )
            )


        if (
            estimator_key
            ==
            "time_series_lstm_regressor"
        ):

            return (
                execute_time_series_lstm(
                    windows=
                        windows,

                    hyperparameters=
                        hyperparameters,

                    execution_device=
                        execution_device,
                )
            )

    except (
        DLTimeSeriesMLPExecutorError,
        DLTimeSeriesRNNExecutorError,
        DLTimeSeriesLSTMExecutorError,
    ) as error:

        raise DLTimeSeriesModelLabEstimatorError(
            (
                "Selected forecasting estimator "
                "execution failed."
            )
        ) from error


    raise DLTimeSeriesModelLabEstimatorError(
        (
            "Unsupported forecasting estimator. "
            f"estimator_key={estimator_key!r}"
        )
    )


# ============================================================
# PUBLIC MODEL LAB EXECUTION
# ============================================================


def execute_time_series_model_lab(
    *,
    training_contract: MLTimeSeriesModelTrainingContract,
    expected_preparation_session_revision: int | None = None,
    execution_device: str | None = None,
) -> DLTimeSeriesModelLabExecutionResult:
    """
    Execute one server-owned forecasting Model Lab request.

    Authority flow:

        Forecast Model Training Contract
                    |
        validated Preparation handoff
                    |
        A2 time-series input authority
                    |
        A3 chronological holdout
                    |
        A4 supervised windows
                    |
        A5 naive baseline
                    |
        ONE selected neural estimator
                    |
        A13-P1 Artifact / Provenance registration
                    |
        privacy-minimal Model Lab result

    This function does not execute the four-model comparison.
    A10 remains the controlled experiment authority.
    """

    try:

        contract = (
            MLTimeSeriesModelTrainingContract
            .model_validate(
                training_contract
            )
        )

    except Exception as error:

        raise DLTimeSeriesModelLabInputError(
            (
                "Time-series Model Lab requires "
                "a valid Forecast Model Training Contract."
            )
        ) from error


    # ========================================================
    # SERVER-OWNED PREPARATION HANDOFF
    # ========================================================

    try:

        (
            dataframe,
            preparation_session_revision,
        ) = (
            load_authorized_ml_dataframe(
                contract=
                    contract,

                execution_label=
                    "Time-series Model Lab",
            )
        )

    except MLTrainingInputError as error:

        raise DLTimeSeriesModelLabInputError(
            str(
                error
            )
        ) from error


    _validate_preparation_revision_pin(
        actual_revision=
            preparation_session_revision,

        expected_revision=
            expected_preparation_session_revision,
    )


    # ========================================================
    # A2 / A3 / A4
    # ========================================================

    try:

        series = (
            validate_and_extract_time_series(
                dataframe=
                    dataframe,

                contract=
                    contract.task_contract,
            )
        )


        partition = (
            resolve_time_series_holdout(
                series=
                    series,

                contract=
                    contract.task_contract,
            )
        )


        windows = (
            build_time_series_supervised_windows(
                series=
                    series,

                partition=
                    partition,

                contract=
                    contract.task_contract,
            )
        )

    except Exception as error:

        raise DLTimeSeriesModelLabInputError(
            (
                "Time-series Model Lab could not build "
                "the validated forecasting population."
            )
        ) from error


    # ========================================================
    # A5 NAIVE BASELINE
    # ========================================================

    try:

        baseline = (
            evaluate_last_value_baseline(
                windows=
                    windows
            )
        )

    except MLTimeSeriesNaiveBaselineError as error:

        raise DLTimeSeriesModelLabExecutionError(
            (
                "Time-series naive baseline "
                "evaluation failed."
            )
        ) from error


    # ========================================================
    # SELECTED MODEL - TRAIN EXACTLY ONCE
    # ========================================================

    execution = (
        _execute_selected_estimator(
            windows=
                windows,

            contract=
                contract,

            execution_device=
                execution_device,
        )
    )


    model_metrics = (
        _forecast_metrics(
            execution.metrics
        )
    )


    baseline_metrics = (
        _forecast_metrics(
            baseline.metrics
        )
    )


    rmse_delta_vs_naive = (
        baseline_metrics[
            "rmse"
        ]
        -
        model_metrics[
            "rmse"
        ]
    )


    beats_naive_baseline = (
        model_metrics[
            "rmse"
        ]
        <
        baseline_metrics[
            "rmse"
        ]
    )


    # ========================================================
    # A13-P1 ARTIFACT + PROVENANCE
    # ========================================================

    try:

        model_artifact = (
            register_time_series_neural_execution_artifact(
                windows=
                    windows,

                training_contract=
                    contract,

                execution=
                    execution,

                preparation_session_revision=
                    preparation_session_revision,
            )
        )

    except DLTimeSeriesArtifactRegistrationError as error:

        raise DLTimeSeriesModelLabArtifactError(
            (
                "Forecast model trained successfully "
                "but Artifact registration failed."
            )
        ) from error


    experiment_provenance = (
        model_artifact
        .experiment_provenance
    )


    if experiment_provenance is None:

        raise DLTimeSeriesModelLabArtifactError(
            (
                "Forecast Model Artifact did not "
                "persist Experiment Provenance."
            )
        )


    # ========================================================
    # PRIVACY-MINIMAL RESULT
    # ========================================================

    return (
        DLTimeSeriesModelLabExecutionResult(
            workflow_id=
                contract.workflow_id,

            dataset_id=
                contract.dataset_id,

            preparation_session_revision=
                int(
                    preparation_session_revision
                ),

            estimator_key=
                contract.estimator_key,

            train_rows=
                windows.train.sample_count,

            test_rows=
                windows.test.sample_count,

            metrics=
                model_metrics,

            naive_baseline_metrics=
                baseline_metrics,

            beats_naive_baseline=
                beats_naive_baseline,

            rmse_delta_vs_naive=
                float(
                    rmse_delta_vs_naive
                ),

            experiment_provenance=
                experiment_provenance,

            model_artifact=
                model_artifact,
        )
    )
