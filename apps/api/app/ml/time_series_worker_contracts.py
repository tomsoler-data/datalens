from __future__ import annotations


import math


from typing import (
    Literal,
    Union,
)


from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)


from app.ml.experiment_provenance import (
    MLExperimentProvenanceRecord,
)


from app.ml.model_artifacts import (
    MLModelArtifactRecord,
)


from app.ml.time_series_model_training_contracts import (
    MLTimeSeriesModelTrainingContract,
)


# ============================================================
# VERSION
# ============================================================


ML_TIME_SERIES_WORKER_PROTOCOL_RULE_VERSION = (
    "ml_time_series_worker_protocol_v0.1"
)


ML_TIME_SERIES_WORKER_REQUEST_RULE_VERSION = (
    "ml_time_series_worker_request_v0.1"
)


ML_TIME_SERIES_WORKER_RESULT_RULE_VERSION = (
    "ml_time_series_worker_result_v0.1"
)


# ============================================================
# TYPES
# ============================================================


MLTimeSeriesWorkerExecutionDevice = Literal[
    "cpu",
    "cuda",
]


MLTimeSeriesWorkerEstimatorKey = Literal[
    "time_series_mlp_regressor",
    "time_series_rnn_regressor",
    "time_series_lstm_regressor",
]


# ============================================================
# METRICS
# ============================================================


class MLTimeSeriesWorkerMetrics(
    BaseModel
):
    """
    Framework-neutral forecasting metric projection.

    No predictions or targets cross the worker boundary.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )


    mae: float = Field(
        allow_inf_nan=False,
    )


    mse: float = Field(
        ge=0.0,
        allow_inf_nan=False,
    )


    rmse: float = Field(
        ge=0.0,
        allow_inf_nan=False,
    )


    mean_error: float = Field(
        allow_inf_nan=False,
    )


# ============================================================
# REQUEST
# ============================================================


class MLTimeSeriesWorkerRequest(
    BaseModel
):
    """
    JSON-safe request sent from the torch-free FastAPI process
    to the isolated Deep Learning runtime.

    Deliberately absent:

    - dataframe;
    - raw rows;
    - windows;
    - tensors;
    - model state;
    - predictions;
    - targets.

    The worker resolves all data server-side from the
    workflow/dataset identities inside the Training Contract.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )


    training_contract: (
        MLTimeSeriesModelTrainingContract
    )


    expected_preparation_session_revision: int = Field(
        ge=0,
        strict=True,
    )


    execution_device: (
        MLTimeSeriesWorkerExecutionDevice
    ) = "cpu"


    rule_version: Literal[
        "ml_time_series_worker_request_v0.1"
    ] = (
        ML_TIME_SERIES_WORKER_REQUEST_RULE_VERSION
    )


# ============================================================
# SUCCESS RESULT
# ============================================================


class MLTimeSeriesWorkerResult(
    BaseModel
):
    """
    Privacy-minimal result returned by the DL worker.

    Learned state remains in the Model Artifact Store.
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
        MLTimeSeriesWorkerEstimatorKey
    )


    train_rows: int = Field(
        gt=0,
        strict=True,
    )


    test_rows: int = Field(
        gt=0,
        strict=True,
    )


    metrics: (
        MLTimeSeriesWorkerMetrics
    )


    naive_baseline_metrics: (
        MLTimeSeriesWorkerMetrics
    )


    beats_naive_baseline: bool


    rmse_delta_vs_naive: float = Field(
        allow_inf_nan=False,
    )


    experiment_provenance: (
        MLExperimentProvenanceRecord
    )


    model_artifact: (
        MLModelArtifactRecord
    )


    rule_version: Literal[
        "ml_time_series_worker_result_v0.1"
    ] = (
        ML_TIME_SERIES_WORKER_RESULT_RULE_VERSION
    )


    @model_validator(
        mode="after"
    )
    def validate_result_authority(
        self,
    ) -> "MLTimeSeriesWorkerResult":

        if (
            self.model_artifact.workflow_id
            !=
            self.workflow_id
        ):

            raise ValueError(
                (
                    "Worker result workflow_id must match "
                    "the persisted Model Artifact."
                )
            )


        if (
            self.model_artifact.dataset_id
            !=
            self.dataset_id
        ):

            raise ValueError(
                (
                    "Worker result dataset_id must match "
                    "the persisted Model Artifact."
                )
            )


        if (
            self.model_artifact.training_contract
            .estimator_key
            !=
            self.estimator_key
        ):

            raise ValueError(
                (
                    "Worker result estimator_key must "
                    "match the persisted Training Contract."
                )
            )


        persisted_provenance = (
            self.model_artifact
            .experiment_provenance
        )


        if persisted_provenance is None:

            raise ValueError(
                (
                    "Worker success requires persisted "
                    "Experiment Provenance."
                )
            )


        if (
            persisted_provenance
            !=
            self.experiment_provenance
        ):

            raise ValueError(
                (
                    "Worker result provenance must equal "
                    "the persisted Artifact provenance."
                )
            )


        expected_beats_naive = (
            self.metrics.rmse
            <
            self.naive_baseline_metrics.rmse
        )


        if (
            self.beats_naive_baseline
            !=
            expected_beats_naive
        ):

            raise ValueError(
                (
                    "beats_naive_baseline does not match "
                    "the RMSE evidence."
                )
            )


        expected_delta = (
            self.naive_baseline_metrics.rmse
            -
            self.metrics.rmse
        )


        if not math.isclose(
            self.rmse_delta_vs_naive,
            expected_delta,
            rel_tol=0.0,
            abs_tol=0.0,
        ):

            raise ValueError(
                (
                    "rmse_delta_vs_naive does not match "
                    "the RMSE evidence."
                )
            )


        return self


# ============================================================
# SUCCESS ENVELOPE
# ============================================================


class MLTimeSeriesWorkerSuccess(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )


    ok: Literal[
        True
    ] = True


    result: (
        MLTimeSeriesWorkerResult
    )


    protocol_version: Literal[
        "ml_time_series_worker_protocol_v0.1"
    ] = (
        ML_TIME_SERIES_WORKER_PROTOCOL_RULE_VERSION
    )


# ============================================================
# FAILURE ENVELOPE
# ============================================================


class MLTimeSeriesWorkerFailure(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )


    ok: Literal[
        False
    ] = False


    error: str = Field(
        min_length=1,
    )


    message: str = Field(
        min_length=1,
    )


    retryable: bool = False


    protocol_version: Literal[
        "ml_time_series_worker_protocol_v0.1"
    ] = (
        ML_TIME_SERIES_WORKER_PROTOCOL_RULE_VERSION
    )


# ============================================================
# RESPONSE FAMILY
# ============================================================


MLTimeSeriesWorkerResponse = Union[
    MLTimeSeriesWorkerSuccess,
    MLTimeSeriesWorkerFailure,
]


def validate_time_series_worker_response(
    value: object,
) -> MLTimeSeriesWorkerResponse:
    """
    Strictly decode one JSON-safe worker response.
    """

    if isinstance(
        value,
        MLTimeSeriesWorkerSuccess,
    ):

        return (
            MLTimeSeriesWorkerSuccess
            .model_validate(
                value
            )
        )


    if isinstance(
        value,
        MLTimeSeriesWorkerFailure,
    ):

        return (
            MLTimeSeriesWorkerFailure
            .model_validate(
                value
            )
        )


    if not isinstance(
        value,
        dict,
    ):

        raise ValueError(
            (
                "Worker response must be a supported "
                "response object or JSON-style mapping."
            )
        )


    ok = value.get(
        "ok"
    )


    if ok is True:

        return (
            MLTimeSeriesWorkerSuccess
            .model_validate(
                value
            )
        )


    if ok is False:

        return (
            MLTimeSeriesWorkerFailure
            .model_validate(
                value
            )
        )


    raise ValueError(
        (
            "Worker response must contain "
            "an explicit boolean ok field."
        )
    )
