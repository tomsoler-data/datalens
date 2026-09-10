from __future__ import annotations


import ast
from pathlib import Path


from app.deep_learning.time_series_contracts import (
    DLTimeSeriesMLPRegressorHyperparameters,
)


from app.deep_learning.time_series_model_lab_executor import (
    DLTimeSeriesModelLabExecutionResult,
    DLTimeSeriesModelLabInputError,
)


from app.deep_learning.time_series_worker import (
    DL_TIME_SERIES_WORKER_RULE_VERSION,
    execute_time_series_worker_request,
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


from app.ml.time_series_worker_contracts import (
    MLTimeSeriesWorkerFailure,
    MLTimeSeriesWorkerRequest,
    MLTimeSeriesWorkerSuccess,
)


# ============================================================
# FIXTURE
# ============================================================


task = (
    MLTimeSeriesForecastingContract(
        workflow_id=
            "workflow:worker-p3-b",

        dataset_id=
            "dataset:worker-p3-b",

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


metrics = {
    "mae":
        1.25,

    "mse":
        2.25,

    "rmse":
        1.5,

    "mean_error":
        -0.25,
}


baseline_metrics = {
    "mae":
        2.0,

    "mse":
        4.0,

    "rmse":
        2.0,

    "mean_error":
        -1.0,
}


provenance = (
    build_ml_model_experiment_provenance(
        training_contract=
            training_contract,

        preparation_session_revision=
            7,

        model_id=
            "model:worker-p3-b",

        train_rows=
            26,

        test_rows=
            10,

        metrics=
            metrics,
    )
)


artifact = (
    MLModelArtifactRecord(
        model_id=
            "model:worker-p3-b",

        workflow_id=
            task.workflow_id,

        dataset_id=
            task.dataset_id,

        training_contract=
            training_contract,

        experiment_provenance=
            provenance,

        metrics=
            metrics,

        train_rows=
            26,

        test_rows=
            10,

        created_at_utc=
            "2026-09-09T12:00:00+00:00",

        serialization_format=
            "pytorch_bundle",

        model_path=
            "models/model-worker-p3-b.ptbundle",

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


execution = (
    DLTimeSeriesModelLabExecutionResult(
        workflow_id=
            task.workflow_id,

        dataset_id=
            task.dataset_id,

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
            baseline_metrics,

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


request = (
    MLTimeSeriesWorkerRequest(
        training_contract=
            training_contract,

        expected_preparation_session_revision=
            7,

        execution_device=
            "cpu",
    )
)


print(
    "[PASS] controlled worker fixture"
)


# ============================================================
# 1. SUCCESS PROJECTION
# ============================================================


captured = {}


def successful_executor(
    *,
    training_contract,
    expected_preparation_session_revision,
    execution_device,
):

    captured[
        "training_contract"
    ] = training_contract

    captured[
        "revision"
    ] = (
        expected_preparation_session_revision
    )

    captured[
        "device"
    ] = execution_device


    return execution


success = (
    execute_time_series_worker_request(
        request,
        executor=
            successful_executor,
    )
)


assert isinstance(
    success,
    MLTimeSeriesWorkerSuccess,
)


assert (
    success.result.workflow_id
    ==
    task.workflow_id
)


assert (
    success.result.dataset_id
    ==
    task.dataset_id
)


assert (
    success.result.estimator_key
    ==
    "time_series_mlp_regressor"
)


assert (
    success.result.model_artifact
    ==
    artifact
)


assert (
    success.result.experiment_provenance
    ==
    provenance
)


assert (
    captured[
        "training_contract"
    ]
    ==
    training_contract
)


assert (
    captured[
        "revision"
    ]
    ==
    7
)


assert (
    captured[
        "device"
    ]
    ==
    "cpu"
)


print(
    "[PASS] worker maps A13-P2 execution into privacy-minimal success"
)


# ============================================================
# 2. INPUT FAILURE MAPPING
# ============================================================


def input_failure_executor(
    **kwargs,
):

    raise DLTimeSeriesModelLabInputError(
        "controlled internal detail"
    )


failure = (
    execute_time_series_worker_request(
        request,
        executor=
            input_failure_executor,
    )
)


assert isinstance(
    failure,
    MLTimeSeriesWorkerFailure,
)


assert (
    failure.error
    ==
    "forecast_input_invalid"
)


assert (
    "controlled internal detail"
    not in
    failure.message
)


print(
    "[PASS] worker converts input failure to stable safe envelope"
)


# ============================================================
# 3. UNKNOWN FAILURE DOES NOT LEAK INTERNAL DETAIL
# ============================================================


def unknown_failure_executor(
    **kwargs,
):

    raise RuntimeError(
        "secret-internal-worker-detail"
    )


internal_failure = (
    execute_time_series_worker_request(
        request,
        executor=
            unknown_failure_executor,
    )
)


assert (
    internal_failure.error
    ==
    "worker_internal_error"
)


assert (
    "secret-internal-worker-detail"
    not in
    internal_failure.message
)


print(
    "[PASS] worker internal exception detail does not cross process boundary"
)


# ============================================================
# 4. INVALID REQUEST FAILS CLOSED
# ============================================================


invalid = (
    execute_time_series_worker_request(
        {
            "unexpected":
                True
        },
        executor=
            successful_executor,
    )
)


assert isinstance(
    invalid,
    MLTimeSeriesWorkerFailure,
)


assert (
    invalid.error
    ==
    "worker_invalid_request"
)


print(
    "[PASS] invalid worker request fails closed before execution"
)


# ============================================================
# 5. STATIC WORKER BOUNDARY
# ============================================================


source_path = Path(
    "app/deep_learning/"
    "time_series_worker.py"
)


source = source_path.read_text(
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
    name == "fastapi"
    or name.startswith(
        "fastapi."
    )
    or name.startswith(
        "app.api"
    )
    for name in imports
)


assert (
    "subprocess"
    not in
    imports
)


assert (
    "redirect_stdout"
    in
    source
)


assert (
    "execute_time_series_model_lab"
    in
    source
)


print(
    "[PASS] DL worker is FastAPI-free and owns no subprocess spawning"
)


print()
print("=" * 80)
print("DL-5-A13-P3-B WORKER VERDICT")
print("=" * 80)
print()

print("A13-P2 execution projection                 PASS")
print("Stable success envelope                     PASS")
print("Stable failure envelopes                    PASS")
print("Internal-detail isolation                   PASS")
print("Invalid-request guard                       PASS")
print("stdout reserved for JSON protocol           PASS")
print("FastAPI-free worker                         PASS")

print()
print(
    "DL-5-A13-P3-B - DL WORKER CORE: PASS"
)
