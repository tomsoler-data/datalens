from __future__ import annotations


import ast
import json
import os
from pathlib import Path
import subprocess
import sys


import app.ml.time_series_worker_bridge as bridge_module


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
    execute_time_series_worker,
    resolve_time_series_dl_python_executable,
)


from app.ml.time_series_worker_contracts import (
    MLTimeSeriesWorkerMetrics,
    MLTimeSeriesWorkerRequest,
    MLTimeSeriesWorkerResult,
    MLTimeSeriesWorkerSuccess,
)


# ============================================================
# 0. TORCH-FREE PARENT RUNTIME
# ============================================================


assert (
    "torch"
    not in
    sys.modules
)


print(
    "[PASS] bridge test starts in torch-free FastAPI runtime"
)


# ============================================================
# FIXTURE
# ============================================================


task = (
    MLTimeSeriesForecastingContract(
        workflow_id=
            "workflow:worker-bridge-p3-b",

        dataset_id=
            "dataset:worker-bridge-p3-b",

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
    MLTimeSeriesWorkerRequest(
        training_contract=
            training_contract,

        expected_preparation_session_revision=
            7,

        execution_device=
            "cpu",
    )
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
            "model:bridge-p3-b",

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
            "model:bridge-p3-b",

        workflow_id=
            task.workflow_id,

        dataset_id=
            task.dataset_id,

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
            "models/model-bridge-p3-b.ptbundle",

        model_file_bytes=
            456,

        model_sha256=
            (
                "b"
                *
                64
            ),
    )
)


worker_result = (
    MLTimeSeriesWorkerResult(
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


success = (
    MLTimeSeriesWorkerSuccess(
        result=
            worker_result
    )
)


success_json = json.dumps(
    success.model_dump(
        mode="json"
    ),
    ensure_ascii=False,
    sort_keys=True,
    separators=(",", ":"),
    allow_nan=False,
)


# ============================================================
# 1. DL PYTHON RESOLUTION
# ============================================================


dl_python = (
    resolve_time_series_dl_python_executable()
)


assert (
    dl_python.exists()
)


assert (
    dl_python.is_file()
)


print(
    "[PASS] bridge resolves isolated .venv-dl Python"
)


# ============================================================
# 2. CONTROLLED SUCCESS PROCESS
# ============================================================


original_run = (
    bridge_module
    .subprocess
    .run
)


captured = {}


def fake_success_run(
    command,
    **kwargs,
):

    captured[
        "command"
    ] = tuple(
        command
    )

    captured[
        "kwargs"
    ] = kwargs


    return (
        subprocess.CompletedProcess(
            args=
                command,

            returncode=
                0,

            stdout=
                (
                    success_json
                    +
                    "\n"
                ),

            stderr=
                "",
        )
    )


bridge_module.subprocess.run = (
    fake_success_run
)


try:

    result = (
        execute_time_series_worker(
            request=
                request
        )
    )

finally:

    bridge_module.subprocess.run = (
        original_run
    )


assert (
    result
    ==
    worker_result
)


assert (
    captured[
        "command"
    ][
        0
    ]
    ==
    str(
        dl_python
    )
)


assert (
    captured[
        "command"
    ][
        1:
    ]
    ==
    (
        "-m",
        (
            "app.deep_learning."
            "time_series_worker"
        ),
    )
)


kwargs = captured[
    "kwargs"
]


assert (
    kwargs[
        "check"
    ]
    is False
)


assert (
    kwargs[
        "capture_output"
    ]
    is True
)


assert (
    kwargs[
        "text"
    ]
    is True
)


assert (
    kwargs[
        "encoding"
    ]
    ==
    "utf-8"
)


assert (
    kwargs[
        "env"
    ][
        "PYTHONDONTWRITEBYTECODE"
    ]
    ==
    "1"
)


assert (
    kwargs[
        "env"
    ][
        "PYTHONUTF8"
    ]
    ==
    "1"
)


assert (
    str(
        Path(
            "app"
        )
        .resolve()
        .parent
    )
    in
    kwargs[
        "env"
    ][
        "PYTHONPATH"
    ]
)


request_payload = json.loads(
    kwargs[
        "input"
    ]
)


assert (
    "dataframe"
    not in
    kwargs[
        "input"
    ]
)


assert (
    request_payload[
        "training_contract"
    ][
        "task_contract"
    ][
        "workflow_id"
    ]
    ==
    task.workflow_id
)


print(
    "[PASS] bridge launches exact module with identity-only JSON request"
)


# ============================================================
# 3. ACTUAL CHILD PROCESS FAILURE PATH
# ============================================================


try:

    execute_time_series_worker(
        request=
            request,

        timeout_seconds=
            120.0,
    )

except MLTimeSeriesWorkerExecutionError as error:

    assert (
        error.error_code
        ==
        "forecast_input_invalid"
    )


    assert (
        error.retryable
        is False
    )

else:

    raise AssertionError(
        (
            "Expected missing Preparation workflow "
            "to fail through real DL subprocess."
        )
    )


assert (
    "torch"
    not in
    sys.modules
)


print(
    "[PASS] real .venv-dl subprocess executes without loading torch in parent"
)


# ============================================================
# 4. MALFORMED WORKER STDOUT FAILS CLOSED
# ============================================================


def fake_invalid_json_run(
    command,
    **kwargs,
):

    return (
        subprocess.CompletedProcess(
            args=
                command,

            returncode=
                0,

            stdout=
                "not-json",

            stderr=
                "",
        )
    )


bridge_module.subprocess.run = (
    fake_invalid_json_run
)


try:

    try:

        execute_time_series_worker(
            request=
                request
        )

    except MLTimeSeriesWorkerProtocolError:

        pass

    else:

        raise AssertionError(
            "Malformed worker JSON must fail closed."
        )

finally:

    bridge_module.subprocess.run = (
        original_run
    )


print(
    "[PASS] malformed worker stdout fails closed"
)


# ============================================================
# 5. NONZERO CHILD EXIT FAILS CLOSED
# ============================================================


def fake_nonzero_run(
    command,
    **kwargs,
):

    return (
        subprocess.CompletedProcess(
            args=
                command,

            returncode=
                2,

            stdout=
                "",

            stderr=
                "controlled worker crash",
        )
    )


bridge_module.subprocess.run = (
    fake_nonzero_run
)


try:

    try:

        execute_time_series_worker(
            request=
                request
        )

    except MLTimeSeriesWorkerUnavailableError as error:

        assert (
            "controlled worker crash"
            not in
            str(
                error
            )
        )

    else:

        raise AssertionError(
            "Nonzero worker exit must fail closed."
        )

finally:

    bridge_module.subprocess.run = (
        original_run
    )


print(
    "[PASS] child process failure does not leak stderr into service error"
)


# ============================================================
# 6. ENVIRONMENT PROPAGATION
# ============================================================


old_value = os.environ.get(
    "DATALENS_ML_MODEL_ARTIFACT_STORE_PATH"
)


os.environ[
    "DATALENS_ML_MODEL_ARTIFACT_STORE_PATH"
] = (
    "C:\\controlled\\model-artifacts.json"
)


captured_environment = {}


def fake_environment_run(
    command,
    **kwargs,
):

    captured_environment.update(
        kwargs[
            "env"
        ]
    )


    return (
        subprocess.CompletedProcess(
            args=
                command,

            returncode=
                0,

            stdout=
                (
                    success_json
                    +
                    "\n"
                ),

            stderr=
                "",
        )
    )


bridge_module.subprocess.run = (
    fake_environment_run
)


try:

    execute_time_series_worker(
        request=
            request
    )

finally:

    bridge_module.subprocess.run = (
        original_run
    )


    if old_value is None:

        os.environ.pop(
            "DATALENS_ML_MODEL_ARTIFACT_STORE_PATH",
            None,
        )

    else:

        os.environ[
            "DATALENS_ML_MODEL_ARTIFACT_STORE_PATH"
        ] = old_value


assert (
    captured_environment[
        "DATALENS_ML_MODEL_ARTIFACT_STORE_PATH"
    ]
    ==
    "C:\\controlled\\model-artifacts.json"
)


print(
    "[PASS] persistence environment propagates to DL child"
)


# ============================================================
# 7. STATIC TORCH / FASTAPI / DL IMPORT BOUNDARY
# ============================================================


source_path = Path(
    "app/ml/"
    "time_series_worker_bridge.py"
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
    name == "torch"
    or name.startswith(
        "torch."
    )
    or name == "fastapi"
    or name.startswith(
        "fastapi."
    )
    or name.startswith(
        "app.deep_learning"
    )
    for name in imports
)


assert (
    "subprocess"
    in
    imports
)


assert (
    "shell=True"
    not in
    source
)


string_constants = {
    node.value
    for node
    in ast.walk(
        tree
    )
    if (
        isinstance(
            node,
            ast.Constant,
        )
        and
        isinstance(
            node.value,
            str,
        )
    )
}


assert (
    "app.deep_learning.time_series_worker"
    in
    string_constants
)


assert (
    "torch"
    not in
    sys.modules
)


print(
    "[PASS] FastAPI-side bridge stays torch-free and DL-import-free"
)


print()
print("=" * 80)
print("DL-5-A13-P3-B BRIDGE VERDICT")
print("=" * 80)
print()

print("Isolated DL Python resolution                PASS")
print("Exact worker module invocation               PASS")
print("Identity-only JSON request                   PASS")
print("Strict worker response validation            PASS")
print("Real subprocess boundary                     PASS")
print("Application failure envelope                 PASS")
print("Process failure isolation                    PASS")
print("Persistence environment propagation          PASS")
print("No shell execution                           PASS")
print("Parent runtime remains torch-free             PASS")

print()
print(
    "DL-5-A13-P3-B - TORCH-FREE SUBPROCESS BRIDGE: PASS"
)
