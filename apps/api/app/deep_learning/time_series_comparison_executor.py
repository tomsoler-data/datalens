from __future__ import annotations


from dataclasses import (
    dataclass,
)


from app.deep_learning.time_series_contracts import (
    DLTimeSeriesLSTMRegressorHyperparameters,
    DLTimeSeriesMLPRegressorHyperparameters,
    DLTimeSeriesRNNRegressorHyperparameters,
)


from app.deep_learning.time_series_lstm_executor import (
    DLTimeSeriesLSTMExecutionResult,
    DLTimeSeriesLSTMExecutorError,
    execute_time_series_lstm,
)


from app.deep_learning.time_series_mlp_executor import (
    DLTimeSeriesMLPExecutionResult,
    DLTimeSeriesMLPExecutorError,
    execute_time_series_mlp,
)


from app.deep_learning.time_series_rnn_executor import (
    DLTimeSeriesRNNExecutionResult,
    DLTimeSeriesRNNExecutorError,
    execute_time_series_rnn,
)


from app.ml.time_series_baseline import (
    MLTimeSeriesNaiveBaselineError,
    MLTimeSeriesNaiveBaselineEvaluation,
    evaluate_last_value_baseline,
)


from app.ml.time_series_comparison import (
    ML_TIME_SERIES_NAIVE_ESTIMATOR_KEY,
    MLTimeSeriesComparisonError,
    MLTimeSeriesForecastCandidate,
    MLTimeSeriesForecastComparison,
    build_time_series_forecast_candidate,
    compare_time_series_forecast_candidates,
)


from app.ml.time_series_windows import (
    MLTimeSeriesSupervisedWindows,
)


# ============================================================
# VERSION
# ============================================================


DL_TIME_SERIES_COMPARISON_EXECUTOR_RULE_VERSION = (
    "dl_time_series_comparison_executor_v0.1"
)


# ============================================================
# ERRORS
# ============================================================


class DLTimeSeriesComparisonExecutorError(
    RuntimeError
):
    pass


class DLTimeSeriesCandidateExecutionError(
    DLTimeSeriesComparisonExecutorError
):
    pass


class DLTimeSeriesCandidateProjectionError(
    DLTimeSeriesComparisonExecutorError
):
    pass


# ============================================================
# RESULT
# ============================================================


@dataclass(
    frozen=True
)
class DLTimeSeriesComparisonExecutionResult:
    """
    Complete controlled DL-5 forecasting experiment.

    Execution-specific trained objects remain attached to their
    original neural execution results for later Artifact work.

    The ranking itself remains owned by the framework-neutral
    app.ml.time_series_comparison authority.
    """

    baseline: MLTimeSeriesNaiveBaselineEvaluation

    mlp: DLTimeSeriesMLPExecutionResult

    rnn: DLTimeSeriesRNNExecutionResult

    lstm: DLTimeSeriesLSTMExecutionResult

    comparison: MLTimeSeriesForecastComparison

    rule_version: str = (
        DL_TIME_SERIES_COMPARISON_EXECUTOR_RULE_VERSION
    )


    @property
    def winner(
        self,
    ) -> MLTimeSeriesForecastCandidate:

        return self.comparison.winner


    @property
    def ranked_estimator_keys(
        self,
    ) -> tuple[
        str,
        ...
    ]:

        return tuple(
            candidate.estimator_key
            for candidate
            in self.comparison.ranked_candidates
        )


# ============================================================
# CANDIDATE PROJECTION
# ============================================================


def _project_execution_candidate(
    *,
    estimator_key: str,
    predictions,
    targets,
    target_positions,
    metrics,
) -> MLTimeSeriesForecastCandidate:

    try:

        return (
            build_time_series_forecast_candidate(
                estimator_key=
                    estimator_key,

                predictions=
                    predictions,

                targets=
                    targets,

                target_positions=
                    target_positions,

                metrics=
                    metrics,
            )
        )

    except MLTimeSeriesComparisonError as error:

        raise DLTimeSeriesCandidateProjectionError(
            (
                "Forecast execution result could not be "
                "projected into the shared comparison plane. "
                f"estimator_key={estimator_key}"
            )
        ) from error


# ============================================================
# PUBLIC EXECUTION
# ============================================================


def execute_time_series_model_comparison(
    *,
    windows: MLTimeSeriesSupervisedWindows,
    mlp_hyperparameters: DLTimeSeriesMLPRegressorHyperparameters,
    rnn_hyperparameters: DLTimeSeriesRNNRegressorHyperparameters,
    lstm_hyperparameters: DLTimeSeriesLSTMRegressorHyperparameters,
    execution_device: str | None = None,
) -> DLTimeSeriesComparisonExecutionResult:
    """
    Execute the complete controlled DL-5 v0.1 experiment.

    Every candidate receives the exact same A4 supervised
    forecasting population.

    Candidate order:

        1. deterministic last-value baseline;
        2. temporal MLP;
        3. simple RNN;
        4. LSTM.

    All four are then projected into A9's framework-neutral
    comparison authority.

    No artifact persistence and no promotion occurs here.
    """

    if not isinstance(
        windows,
        MLTimeSeriesSupervisedWindows,
    ):

        raise DLTimeSeriesComparisonExecutorError(
            (
                "Forecast comparison execution requires "
                "validated supervised forecasting windows."
            )
        )


    mlp_parameters = (
        DLTimeSeriesMLPRegressorHyperparameters
        .model_validate(
            mlp_hyperparameters
        )
    )


    rnn_parameters = (
        DLTimeSeriesRNNRegressorHyperparameters
        .model_validate(
            rnn_hyperparameters
        )
    )


    lstm_parameters = (
        DLTimeSeriesLSTMRegressorHyperparameters
        .model_validate(
            lstm_hyperparameters
        )
    )


    # ========================================================
    # BASELINE
    # ========================================================

    try:

        baseline = (
            evaluate_last_value_baseline(
                windows=
                    windows
            )
        )

    except MLTimeSeriesNaiveBaselineError as error:

        raise DLTimeSeriesCandidateExecutionError(
            "Naive forecasting baseline execution failed."
        ) from error


    # ========================================================
    # TEMPORAL MLP
    # ========================================================

    try:

        mlp = (
            execute_time_series_mlp(
                windows=
                    windows,

                hyperparameters=
                    mlp_parameters,

                execution_device=
                    execution_device,
            )
        )

    except DLTimeSeriesMLPExecutorError as error:

        raise DLTimeSeriesCandidateExecutionError(
            "Temporal MLP candidate execution failed."
        ) from error


    # ========================================================
    # SIMPLE RNN
    # ========================================================

    try:

        rnn = (
            execute_time_series_rnn(
                windows=
                    windows,

                hyperparameters=
                    rnn_parameters,

                execution_device=
                    execution_device,
            )
        )

    except DLTimeSeriesRNNExecutorError as error:

        raise DLTimeSeriesCandidateExecutionError(
            "Simple RNN candidate execution failed."
        ) from error


    # ========================================================
    # LSTM
    # ========================================================

    try:

        lstm = (
            execute_time_series_lstm(
                windows=
                    windows,

                hyperparameters=
                    lstm_parameters,

                execution_device=
                    execution_device,
            )
        )

    except DLTimeSeriesLSTMExecutorError as error:

        raise DLTimeSeriesCandidateExecutionError(
            "LSTM candidate execution failed."
        ) from error


    # ========================================================
    # FRAMEWORK-NEUTRAL PROJECTIONS
    # ========================================================

    baseline_candidate = (
        _project_execution_candidate(
            estimator_key=
                ML_TIME_SERIES_NAIVE_ESTIMATOR_KEY,

            predictions=
                baseline.predictions,

            targets=
                baseline.targets,

            target_positions=
                baseline.target_positions,

            metrics=
                baseline.metrics,
        )
    )


    mlp_candidate = (
        _project_execution_candidate(
            estimator_key=
                mlp.estimator_key,

            predictions=
                mlp.predictions,

            targets=
                mlp.targets,

            target_positions=
                mlp.target_positions,

            metrics=
                mlp.metrics,
        )
    )


    rnn_candidate = (
        _project_execution_candidate(
            estimator_key=
                rnn.estimator_key,

            predictions=
                rnn.predictions,

            targets=
                rnn.targets,

            target_positions=
                rnn.target_positions,

            metrics=
                rnn.metrics,
        )
    )


    lstm_candidate = (
        _project_execution_candidate(
            estimator_key=
                lstm.estimator_key,

            predictions=
                lstm.predictions,

            targets=
                lstm.targets,

            target_positions=
                lstm.target_positions,

            metrics=
                lstm.metrics,
        )
    )


    # ========================================================
    # SHARED A9 COMPARISON
    # ========================================================

    try:

        comparison = (
            compare_time_series_forecast_candidates(
                candidates=
                    (
                        baseline_candidate,
                        mlp_candidate,
                        rnn_candidate,
                        lstm_candidate,
                    )
            )
        )

    except MLTimeSeriesComparisonError as error:

        raise DLTimeSeriesComparisonExecutorError(
            (
                "Framework-neutral forecasting comparison "
                "rejected the executed candidate family."
            )
        ) from error


    return (
        DLTimeSeriesComparisonExecutionResult(
            baseline=
                baseline,

            mlp=
                mlp,

            rnn=
                rnn,

            lstm=
                lstm,

            comparison=
                comparison,
        )
    )
