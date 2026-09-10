from __future__ import annotations


import ast
import json
from pathlib import Path


from pydantic import (
    ValidationError,
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


from app.ml.time_series_worker_contracts import (
    ML_TIME_SERIES_WORKER_PROTOCOL_RULE_VERSION,
    MLTimeSeriesWorkerFailure,
    MLTimeSeriesWorkerMetrics,
    MLTimeSeriesWorkerRequest,
    MLTimeSeriesWorkerResult,
    MLTimeSeriesWorkerSuccess,
    validate_time_series_worker_response,
)


# ============================================================
# HELPERS
# ============================================================


def require_validation_error(
    callback,
    *,
    label: str,
) -> None:

    try:

        callback()

    except (
        ValidationError,
        ValueError,
    ):

        return


    raise AssertionError(
        (
            "Expected worker-contract failure: "
            f"{label}"
        )
    )


# ============================================================
# 1. TRAINING CONTRACT
# ============================================================


task = (
    MLTimeSeriesForecastingContract(
        workflow_id=
            "workflow:worker-contract",

        dataset_id=
            "dataset:worker-contract",

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


print(
    "[PASS] controlled forecasting Training Contract"
)


# ============================================================
# 2. REQUEST IS IDENTITY-ONLY / DATA-FREE
# ============================================================


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


request_payload = (
    request.model_dump(
        mode="json"
    )
)


assert set(
    request_payload
) == {
    "training_contract",
    "expected_preparation_session_revision",
    "execution_device",
    "rule_version",
}


request_text = json.dumps(
    request_payload,
    ensure_ascii=False,
    sort_keys=True,
    separators=(",", ":"),
    allow_nan=False,
)


for forbidden in (
    "dataframe",
    "raw_rows",
    "windows",
    "predictions",
    "targets",
    "model_state",
    "tensor",
):

    assert (
        forbidden
        not in
        request_text
    )


restored_request = (
    MLTimeSeriesWorkerRequest
    .model_validate_json(
        request_text
    )
)


assert (
    restored_request
    ==
    request
)


print(
    "[PASS] worker request is deterministic JSON and contains no data payload"
)


# ============================================================
# 3. METRICS / PROVENANCE / ARTIFACT
# ============================================================


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


baseline_metrics = (
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
            "model:worker-contract",

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
            "model:worker-contract",

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
            "models/model-worker-contract.ptbundle",

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


print(
    "[PASS] generic provenance and Artifact metadata fit worker response"
)


# ============================================================
# 4. SUCCESS RESULT
# ============================================================


result = (
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


assert (
    result.model_artifact
    .training_contract
    .estimator_key
    ==
    result.estimator_key
)


print(
    "[PASS] privacy-minimal worker result binds Artifact and Provenance"
)


# ============================================================
# 5. SUCCESS JSON ROUND-TRIP
# ============================================================


success = (
    MLTimeSeriesWorkerSuccess(
        result=
            result
    )
)


success_payload = (
    success.model_dump(
        mode="json"
    )
)


success_encoded = json.dumps(
    success_payload,
    ensure_ascii=False,
    sort_keys=True,
    separators=(",", ":"),
    allow_nan=False,
)


success_decoded = json.loads(
    success_encoded
)


restored_success = (
    validate_time_series_worker_response(
        success_decoded
    )
)


assert isinstance(
    restored_success,
    MLTimeSeriesWorkerSuccess,
)


assert (
    restored_success
    ==
    success
)


assert (
    restored_success.protocol_version
    ==
    ML_TIME_SERIES_WORKER_PROTOCOL_RULE_VERSION
)


print(
    "[PASS] worker success envelope round-trips deterministically"
)


# ============================================================
# 6. FAILURE JSON ROUND-TRIP
# ============================================================


failure = (
    MLTimeSeriesWorkerFailure(
        error=
            "forecast_execution_failed",

        message=
            "Controlled worker failure.",

        retryable=
            False,
    )
)


failure_payload = (
    failure.model_dump(
        mode="json"
    )
)


restored_failure = (
    validate_time_series_worker_response(
        failure_payload
    )
)


assert isinstance(
    restored_failure,
    MLTimeSeriesWorkerFailure,
)


assert (
    restored_failure
    ==
    failure
)


print(
    "[PASS] worker failure envelope round-trips deterministically"
)


# ============================================================
# 7. INCONSISTENT RESULT FAILS CLOSED
# ============================================================


require_validation_error(
    lambda:
        MLTimeSeriesWorkerResult(
            workflow_id=
                task.workflow_id,

            dataset_id=
                task.dataset_id,

            preparation_session_revision=
                7,

            estimator_key=
                "time_series_rnn_regressor",

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
        ),
    label=
        "result estimator differs from Artifact Training Contract",
)


require_validation_error(
    lambda:
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
                baseline_metrics,

            beats_naive_baseline=
                False,

            rmse_delta_vs_naive=
                0.5,

            experiment_provenance=
                provenance,

            model_artifact=
                artifact,
        ),
    label=
        "baseline decision contradicts RMSE",
)


print(
    "[PASS] inconsistent worker success evidence fails closed"
)


# ============================================================
# 8. NO RAW / LEARNED STATE IN RESULT
# ============================================================


result_payload_text = json.dumps(
    success_payload,
    ensure_ascii=False,
    sort_keys=True,
)


for forbidden in (
    "predictions",
    "targets",
    "dataframe",
    "windows",
    "model_state",
    "standardizer",
    "epoch_losses",
    "optimizer_state",
):

    assert (
        forbidden
        not in
        result_payload_text
    )


print(
    "[PASS] worker response contains no raw or learned execution state"
)


# ============================================================
# 9. STATIC TORCH / FASTAPI ISOLATION
# ============================================================


source_path = Path(
    "app/ml/"
    "time_series_worker_contracts.py"
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
    for name in imports
)


assert (
    "subprocess"
    not in
    imports
)


print(
    "[PASS] worker transport contracts are torch-free, FastAPI-free and process-free"
)


print()
print("=" * 80)
print("DL-5-A13-P3-A FINAL VERDICT")
print("=" * 80)
print()

print("JSON-safe worker request                     PASS")
print("No raw data crosses request boundary         PASS")
print("Privacy-minimal worker result                PASS")
print("Artifact identity binding                    PASS")
print("Experiment Provenance binding                PASS")
print("Naive-baseline evidence binding              PASS")
print("Success envelope                             PASS")
print("Failure envelope                             PASS")
print("Deterministic JSON round-trip                 PASS")
print("Torch-free transport contract                PASS")
print("FastAPI-free shared contract                 PASS")
print("No subprocess authority in contracts         PASS")

print()
print(
    "DL-5-A13-P3-A - TORCH-FREE WORKER TRANSPORT CONTRACTS: PASS"
)
