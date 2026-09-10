from __future__ import annotations


import ast
from pathlib import Path
import sys


import app.api.model_training_service as service


from app.api.model_training_contracts import (
    ModelTrainingForecastDetail,
    ModelTrainingRequest,
)


from app.api.model_training_service import (
    ModelTrainingEstimatorError,
    ModelTrainingExecutionError,
    ModelTrainingInputError,
    train_model,
)


from app.deep_learning.time_series_contracts import (
    DLTimeSeriesMLPRegressorHyperparameters,
)


from app.ml.contracts import (
    MLTimeHoldoutSplitContract,
)


from app.ml.experiment_provenance import (
    build_ml_model_experiment_provenance,
)


from app.ml.model_artifacts import (
    MLModelArtifactRecord,
)


from app.ml.time_series_contracts import (
    MLTimeSeriesForecastingContract,
)


from app.ml.time_series_model_training_contracts import (
    MLTimeSeriesModelTrainingContract,
)


from app.ml.time_series_worker_bridge import (
    MLTimeSeriesWorkerExecutionError,
    MLTimeSeriesWorkerProtocolError,
    MLTimeSeriesWorkerUnavailableError,
)


from app.ml.time_series_worker_contracts import (
    MLTimeSeriesWorkerMetrics,
    MLTimeSeriesWorkerRequest,
    MLTimeSeriesWorkerResult,
)


assert (
    "torch"
    not in
    sys.modules
)


print(
    "[PASS] forecasting service acceptance starts torch-free"
)


WORKFLOW_ID = (
    "workflow:forecast-api-service"
)


DATASET_ID = (
    "dataset:forecast-api-service"
)


task = (
    MLTimeSeriesForecastingContract(
        workflow_id=
            WORKFLOW_ID,

        dataset_id=
            DATASET_ID,

        target_column=
            "revenue",

        lookback=
            4,

        split=
            MLTimeHoldoutSplitContract(
                time_column=
                    "order_date",

                test_size=
                    0.25,
            ),
    )
)


training_contract = (
    MLTimeSeriesModelTrainingContract(
        task_contract=
            task,

        estimator_hyperparameters=
            DLTimeSeriesMLPRegressorHyperparameters(
                hidden_features=
                    8,

                epochs=
                    2,

                batch_size=
                    8,

                learning_rate=
                    0.01,
            ),
    )
)


request = (
    ModelTrainingRequest(
        training=
            training_contract,

        expected_preparation_session_revision=
            7,

        execution_device=
            "cpu",
    )
)


assert isinstance(
    request.training,
    MLTimeSeriesModelTrainingContract,
)


print(
    "[PASS] ModelTrainingRequest accepts forecasting contract"
)


metrics = (
    MLTimeSeriesWorkerMetrics(
        mae=
            1.25,

        mse=
            2.25,

        rmse=
            1.5,

        mean_error=
            -0.25,
    )
)


baseline = (
    MLTimeSeriesWorkerMetrics(
        mae=
            2.0,

        mse=
            4.0,

        rmse=
            2.0,

        mean_error=
            -1.0,
    )
)


provenance = (
    build_ml_model_experiment_provenance(
        training_contract=
            training_contract,

        preparation_session_revision=
            7,

        model_id=
            "model:forecast-api-service",

        train_rows=
            26,

        test_rows=
            10,

        metrics=
            metrics.model_dump(
                mode="python"
            ),
    )
)


artifact = (
    MLModelArtifactRecord(
        model_id=
            "model:forecast-api-service",

        workflow_id=
            WORKFLOW_ID,

        dataset_id=
            DATASET_ID,

        training_contract=
            training_contract,

        experiment_provenance=
            provenance,

        metrics=
            metrics.model_dump(
                mode="python"
            ),

        train_rows=
            26,

        test_rows=
            10,

        created_at_utc=
            "2026-09-09T12:00:00+00:00",

        serialization_format=
            "pytorch_bundle",

        model_path=
            "models/model-forecast-api-service.ptbundle",

        model_file_bytes=
            456,

        model_sha256=
            (
                "a"
                *
                64
            ),
    )
)


worker_result = (
    MLTimeSeriesWorkerResult(
        workflow_id=
            WORKFLOW_ID,

        dataset_id=
            DATASET_ID,

        preparation_session_revision=
            7,

        estimator_key=
            "time_series_mlp_regressor",

        train_rows=
            26,

        test_rows=
            10,

        metrics=
            metrics,

        naive_baseline_metrics=
            baseline,

        beats_naive_baseline=
            True,

        rmse_delta_vs_naive=
            0.5,

        experiment_provenance=
            provenance,

        model_artifact=
            artifact,
    )
)


original_worker = (
    service.execute_time_series_worker
)


original_classical = (
    service.execute_classical_ml
)


captured = {}


def fake_worker(
    *,
    request,
):

    assert isinstance(
        request,
        MLTimeSeriesWorkerRequest,
    )

    captured[
        "request"
    ] = request

    return worker_result


def forbidden_classical(
    **kwargs,
):

    raise AssertionError(
        "Forecast request entered classical executor."
    )


service.execute_time_series_worker = (
    fake_worker
)

service.execute_classical_ml = (
    forbidden_classical
)


try:

    detail = (
        train_model(
            request
        )
    )

finally:

    service.execute_time_series_worker = (
        original_worker
    )

    service.execute_classical_ml = (
        original_classical
    )


assert isinstance(
    detail,
    ModelTrainingForecastDetail,
)


assert (
    captured[
        "request"
    ].training_contract
    ==
    training_contract
)


assert (
    captured[
        "request"
    ].expected_preparation_session_revision
    ==
    7
)


assert (
    detail.target_column
    ==
    "revenue"
)


assert (
    detail.time_column
    ==
    "order_date"
)


assert (
    detail.lookback
    ==
    4
)


assert (
    detail.forecast_horizon
    ==
    1
)


assert (
    detail.estimator_key
    ==
    "time_series_mlp_regressor"
)


print(
    "[PASS] forecast dispatch uses worker and exposes temporal semantics"
)


payload = (
    detail.model_dump(
        mode="json"
    )
)


for forbidden_field in (
    "feature_columns",
    "categorical_feature_columns",
    "preprocessing",
    "model_path",
    "model_file_bytes",
    "model_sha256",
    "predictions",
    "targets",
    "standardizer",
    "model",
    "epoch_losses",
):

    assert (
        forbidden_field
        not in
        payload
    )


print(
    "[PASS] forecast API detail remains privacy-minimal"
)


def expect_service_error(
    worker_error_code: str,
    expected_exception,
) -> None:

    original = (
        service.execute_time_series_worker
    )


    def fail_worker(
        *,
        request,
    ):

        raise MLTimeSeriesWorkerExecutionError(
            "safe worker message",
            error_code=
                worker_error_code,
            retryable=
                False,
        )


    service.execute_time_series_worker = (
        fail_worker
    )


    try:

        try:

            train_model(
                request
            )

        except expected_exception:

            return


        raise AssertionError(
            (
                "Expected service error "
                f"{expected_exception.__name__}"
            )
        )

    finally:

        service.execute_time_series_worker = (
            original
        )


expect_service_error(
    "forecast_input_invalid",
    ModelTrainingInputError,
)


expect_service_error(
    "forecast_estimator_failed",
    ModelTrainingEstimatorError,
)


expect_service_error(
    "forecast_artifact_failed",
    ModelTrainingExecutionError,
)


expect_service_error(
    "worker_internal_error",
    ModelTrainingExecutionError,
)


print(
    "[PASS] worker application failures map to service errors"
)


for failure_type in (
    MLTimeSeriesWorkerUnavailableError,
    MLTimeSeriesWorkerProtocolError,
):

    original = (
        service.execute_time_series_worker
    )


    def transport_failure(
        *,
        request,
        failure_type=
            failure_type,
    ):

        raise failure_type(
            "controlled transport failure"
        )


    service.execute_time_series_worker = (
        transport_failure
    )


    try:

        try:

            train_model(
                request
            )

        except ModelTrainingExecutionError:

            pass

        else:

            raise AssertionError(
                "Transport failure must fail closed."
            )

    finally:

        service.execute_time_series_worker = (
            original
        )


print(
    "[PASS] worker transport failures fail closed"
)


service_path = Path(
    "app/api/model_training_service.py"
)


source = service_path.read_text(
    encoding="utf-8"
)


tree = ast.parse(
    source
)


imports = []


for node in ast.walk(
    tree
):

    if isinstance(
        node,
        ast.Import,
    ):

        imports.extend(
            alias.name
            for alias
            in node.names
        )


    elif isinstance(
        node,
        ast.ImportFrom,
    ):

        if node.module:

            imports.append(
                node.module
            )


assert not any(
    name == "torch"
    or name.startswith("torch.")
    or name.startswith("app.deep_learning")
    for name in imports
)


assert (
    "app.ml.time_series_worker_bridge"
    in
    imports
)


assert (
    "execute_classical_ml"
    in
    source
)


assert (
    "execute_time_series_worker"
    in
    source
)


assert (
    "torch"
    not in
    sys.modules
)


print(
    "[PASS] dual service dispatch remains torch-free"
)


print()
print("=" * 80)
print("DL-5-A13-P3-C1 FINAL VERDICT")
print("=" * 80)
print()

print("Forecast request contract accepted           PASS")
print("Classical executor retained                  PASS")
print("Forecast worker dispatch                     PASS")
print("Temporal response semantics                  PASS")
print("Privacy-minimal forecast detail              PASS")
print("Worker error mapping                         PASS")
print("Transport failure isolation                  PASS")
print("FastAPI runtime torch-free                   PASS")

print()
print(
    "DL-5-A13-P3-C1 - FORECAST API CONTRACTS + SERVICE DISPATCH: PASS"
)
