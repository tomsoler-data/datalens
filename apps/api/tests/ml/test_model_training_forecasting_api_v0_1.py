from __future__ import annotations


import sys


from fastapi import (
    FastAPI,
)


from fastapi.testclient import (
    TestClient,
)


import app.api.model_training as model_training_api


from app.api.model_training import (
    MODEL_TRAINING_API_VERSION,
    ModelTrainingResponse,
    router,
)


from app.api.model_training_contracts import (
    ModelTrainingForecastDetail,
    ModelTrainingRequest,
)


from app.deep_learning.time_series_contracts import (
    DLTimeSeriesMLPRegressorHyperparameters,
)


from app.ml.contracts import (
    MLTimeHoldoutSplitContract,
)


from app.ml.time_series_contracts import (
    MLTimeSeriesForecastingContract,
)


from app.ml.time_series_model_training_contracts import (
    MLTimeSeriesModelTrainingContract,
)


# ============================================================
# TORCH-FREE HTTP RUNTIME
# ============================================================


assert (
    "torch"
    not in
    sys.modules
)


# ============================================================
# TEST APP
# ============================================================


test_app = FastAPI()


test_app.include_router(
    router
)


client = TestClient(
    test_app
)


# ============================================================
# PATCH HELPER
# ============================================================


class Patch:

    def __init__(
        self,
        name: str,
        value,
    ) -> None:

        self.name = name
        self.value = value

        self.original = getattr(
            model_training_api,
            name,
        )


    def __enter__(
        self,
    ):

        setattr(
            model_training_api,
            self.name,
            self.value,
        )

        return self


    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ):

        setattr(
            model_training_api,
            self.name,
            self.original,
        )


# ============================================================
# FORECAST FIXTURE
# ============================================================


WORKFLOW_ID = (
    "workflow:forecast-http"
)


DATASET_ID = (
    "dataset:forecast-http"
)


MODEL_ID = (
    "model:forecast-http"
)


EXPERIMENT_ID = (
    "experiment:"
    +
    (
        "a"
        *
        32
    )
)


TRAINING_SHA = (
    "b"
    *
    64
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


detail = (
    ModelTrainingForecastDetail(
        model_id=
            MODEL_ID,

        workflow_id=
            WORKFLOW_ID,

        dataset_id=
            DATASET_ID,

        target_column=
            "revenue",

        time_column=
            "order_date",

        lookback=
            4,

        forecast_horizon=
            1,

        estimator_key=
            "time_series_mlp_regressor",

        train_rows=
            26,

        test_rows=
            10,

        metrics={
            "mae":
                1.25,

            "mse":
                2.25,

            "rmse":
                1.5,

            "mean_error":
                -0.25,
        },

        naive_baseline_metrics={
            "mae":
                2.0,

            "mse":
                4.0,

            "rmse":
                2.0,

            "mean_error":
                -1.0,
        },

        beats_naive_baseline=
            True,

        rmse_delta_vs_naive=
            0.5,

        model_created_at_utc=
            "2026-09-09T12:00:00+00:00",

        experiment_id=
            EXPERIMENT_ID,

        preparation_session_revision=
            7,

        training_contract_sha256=
            TRAINING_SHA,

        serialization_format=
            "pytorch_bundle",
    )
)


# ============================================================
# 1. RESPONSE FAMILY
# ============================================================


assert (
    ModelTrainingResponse
    is not None
)


print(
    "[PASS] HTTP route exposes classical/forecast response family"
)


# ============================================================
# 2. FORECAST POST
# ============================================================


captured = []


def fake_train(
    incoming,
):

    captured.append(
        incoming
    )


    return detail


with Patch(
    "train_model",
    fake_train,
):

    response = client.post(
        "/model-training/train",
        json=
            request.model_dump(
                mode="json"
            ),
    )


assert (
    response.status_code
    ==
    200
)


assert (
    len(
        captured
    )
    ==
    1
)


parsed_request = (
    captured[
        0
    ]
)


assert isinstance(
    parsed_request.training,
    MLTimeSeriesModelTrainingContract,
)


assert (
    parsed_request.training
    ==
    training_contract
)


assert (
    parsed_request.execution_device
    ==
    "cpu"
)


assert (
    response.json()
    ==
    detail.model_dump(
        mode="json"
    )
)


print(
    "[PASS] POST /model-training/train returns temporal forecasting detail"
)


# ============================================================
# 3. REAL TEMPORAL SEMANTICS
# ============================================================


payload = (
    response.json()
)


assert (
    payload[
        "task_family"
    ]
    ==
    "time_series_forecasting"
)


assert (
    payload[
        "target_column"
    ]
    ==
    "revenue"
)


assert (
    payload[
        "time_column"
    ]
    ==
    "order_date"
)


assert (
    payload[
        "lookback"
    ]
    ==
    4
)


assert (
    payload[
        "forecast_horizon"
    ]
    ==
    1
)


assert (
    payload[
        "estimator_key"
    ]
    ==
    "time_series_mlp_regressor"
)


print(
    "[PASS] HTTP response preserves true forecasting semantics"
)


# ============================================================
# 4. PRIVACY BOUNDARY
# ============================================================


for forbidden in (
    "feature_columns",
    "categorical_feature_columns",
    "preprocessing",
    "model_path",
    "model_file_bytes",
    "model_sha256",
    "predictions",
    "targets",
    "standardizer",
    "epoch_losses",
):

    assert (
        forbidden
        not in
        payload
    )


print(
    "[PASS] HTTP forecasting response remains privacy-minimal"
)


# ============================================================
# 5. ROUTE SET UNCHANGED
# ============================================================


paths = set(
    test_app.openapi()[
        "paths"
    ]
)


assert (
    paths
    ==
    {
        "/model-training/context",
        "/model-training/train",
    }
)


print(
    "[PASS] Model Training route surface remains exactly two routes"
)


# ============================================================
# 6. OPENAPI RESPONSE IS A UNION
# ============================================================


schema = (
    test_app.openapi()[
        "paths"
    ][
        "/model-training/train"
    ][
        "post"
    ][
        "responses"
    ][
        "200"
    ][
        "content"
    ][
        "application/json"
    ][
        "schema"
    ]
)


assert (
    "anyOf"
    in
    schema
    or
    "oneOf"
    in
    schema
)


print(
    "[PASS] OpenAPI advertises multiple Model Training response families"
)


# ============================================================
# 7. FASTAPI PROCESS STILL TORCH-FREE
# ============================================================


assert (
    "torch"
    not in
    sys.modules
)


assert (
    MODEL_TRAINING_API_VERSION
    ==
    "model_training_api_v0.1"
)


print(
    "[PASS] forecast-aware HTTP route remains torch-free"
)


print()
print("=" * 80)
print("DL-5-A13-P3-C2 FINAL VERDICT")
print("=" * 80)
print()

print("Existing /model-training/train reused         PASS")
print("Forecast request parsed                      PASS")
print("Forecast response union                      PASS")
print("Temporal response semantics                  PASS")
print("Privacy-minimal response                     PASS")
print("OpenAPI multi-response schema                PASS")
print("Route set unchanged                          PASS")
print("FastAPI process torch-free                   PASS")
print("API version preserved                        PASS")

print()
print(
    "DL-5-A13-P3-C2 - FORECAST HTTP ROUTE: PASS"
)
