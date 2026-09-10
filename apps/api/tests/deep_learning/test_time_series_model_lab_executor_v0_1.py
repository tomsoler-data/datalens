from __future__ import annotations


import ast
from pathlib import Path
from types import SimpleNamespace


import numpy as np
import pandas as pd


import app.deep_learning.time_series_model_lab_executor as executor_module


from app.deep_learning.time_series_bundle import (
    deserialize_trusted_time_series_neural_bundle,
)


from app.deep_learning.time_series_bundle_inference import (
    predict_trusted_time_series_bundle,
)


from app.deep_learning.time_series_contracts import (
    DLTimeSeriesLSTMRegressorHyperparameters,
    DLTimeSeriesMLPRegressorHyperparameters,
    DLTimeSeriesRNNRegressorHyperparameters,
)


from app.deep_learning.time_series_model_lab_executor import (
    DL_TIME_SERIES_MODEL_LAB_EXECUTOR_RULE_VERSION,
    DLTimeSeriesModelLabExecutionResult,
    DLTimeSeriesModelLabInputError,
    execute_time_series_model_lab,
)


from app.ml.contracts import (
    MLTimeHoldoutSplitContract,
)


from app.ml.model_artifact_store import (
    list_ml_model_artifacts,
    load_ml_model_artifact_binary,
)


from app.ml.time_series_contracts import (
    MLTimeSeriesForecastingContract,
)


from app.ml.time_series_model_training_contracts import (
    MLTimeSeriesModelTrainingContract,
)


from app.ml.training_input import (
    load_authorized_ml_dataframe,
)


from tests.ml.test_ml_model_artifact_store_v0_1 import (
    isolated_environment,
    seed_preparation_authority,
)


# ============================================================
# FIXTURE
# ============================================================


WORKFLOW_ID = (
    "prep:forecast-model-lab"
)


DATASET_ID = (
    "dataset:forecast-model-lab"
)


PREPARATION_REVISION = (
    0
)


frame = pd.DataFrame(
    {
        "order_date":
            pd.date_range(
                "2026-01-01",
                periods=40,
                freq="D",
            ),

        "revenue":
            [
                float(
                    50
                    +
                    index
                )
                for index
                in range(
                    40
                )
            ],
    }
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


mlp_parameters = (
    DLTimeSeriesMLPRegressorHyperparameters(
        hidden_features=
            8,

        epochs=
            2,

        batch_size=
            16,

        learning_rate=
            0.02,
    )
)


rnn_parameters = (
    DLTimeSeriesRNNRegressorHyperparameters(
        hidden_size=
            8,

        epochs=
            2,

        batch_size=
            16,

        learning_rate=
            0.02,
    )
)


lstm_parameters = (
    DLTimeSeriesLSTMRegressorHyperparameters(
        hidden_size=
            8,

        epochs=
            2,

        batch_size=
            16,

        learning_rate=
            0.02,
    )
)


contracts = (
    MLTimeSeriesModelTrainingContract(
        task_contract=
            task,

        estimator_hyperparameters=
            mlp_parameters,
    ),
    MLTimeSeriesModelTrainingContract(
        task_contract=
            task,

        estimator_hyperparameters=
            rnn_parameters,
    ),
    MLTimeSeriesModelTrainingContract(
        task_contract=
            task,

        estimator_hyperparameters=
            lstm_parameters,
    ),
)


# ============================================================
# 1. SHARED PREPARATION HANDOFF AUTHORITY IS STRUCTURALLY REUSABLE
# ============================================================


handoff = SimpleNamespace(
    workflow_id=
        WORKFLOW_ID,

    dataset_ids=(
        DATASET_ID,
    ),

    dataset_records=(
        {
            "dataset_id":
                DATASET_ID,

            "dataframe":
                frame,
        },
    ),

    session_revision=
        PREPARATION_REVISION,
)


loaded_frame, loaded_revision = (
    load_authorized_ml_dataframe(
        contract=
            contracts[
                0
            ],

        handoff_loader=
            lambda workflow_id:
                handoff,

        execution_label=
            "Forecast compatibility test",
    )
)


assert (
    loaded_revision
    ==
    PREPARATION_REVISION
)


assert (
    loaded_frame
    is not
    frame
)


pd.testing.assert_frame_equal(
    loaded_frame,
    frame,
)


print(
    "[PASS] forecasting reuses the existing server-owned Preparation handoff"
)


# ============================================================
# 2. PUBLIC EXECUTOR HANDOFF PATCH
# ============================================================


original_loader = (
    executor_module
    .load_authorized_ml_dataframe
)


loader_calls = []


def fake_authorized_loader(
    *,
    contract,
    execution_label,
):

    loader_calls.append(
        (
            contract.workflow_id,
            contract.dataset_id,
            execution_label,
        )
    )


    return (
        frame.copy(
            deep=True
        ),
        PREPARATION_REVISION,
    )


executor_module.load_authorized_ml_dataframe = (
    fake_authorized_loader
)


try:

    with isolated_environment() as (
        _,
        _,
    ):

        seed_preparation_authority(
            workflow_id=
                WORKFLOW_ID,

            dataset_id=
                DATASET_ID,
        )


        results = []


        # ====================================================
        # 3. ONE MODEL LAB REQUEST = ONE SELECTED ESTIMATOR
        # ====================================================


        for contract in contracts:

            result = (
                execute_time_series_model_lab(
                    training_contract=
                        contract,

                    expected_preparation_session_revision=
                        PREPARATION_REVISION,

                    execution_device=
                        "cpu",
                )
            )


            assert isinstance(
                result,
                DLTimeSeriesModelLabExecutionResult,
            )


            assert (
                result.rule_version
                ==
                DL_TIME_SERIES_MODEL_LAB_EXECUTOR_RULE_VERSION
            )


            assert (
                result.workflow_id
                ==
                WORKFLOW_ID
            )


            assert (
                result.dataset_id
                ==
                DATASET_ID
            )


            assert (
                result.preparation_session_revision
                ==
                PREPARATION_REVISION
            )


            assert (
                result.task_family
                ==
                "time_series_forecasting"
            )


            assert (
                result.problem_type
                ==
                "regression"
            )


            assert (
                result.estimator_key
                ==
                contract.estimator_key
            )


            assert (
                result.train_rows
                ==
                26
            )


            assert (
                result.test_rows
                ==
                10
            )


            assert set(
                result.metrics
            ) == {
                "mae",
                "mse",
                "rmse",
                "mean_error",
            }


            assert set(
                result.naive_baseline_metrics
            ) == {
                "mae",
                "mse",
                "rmse",
                "mean_error",
            }


            assert (
                result.beats_naive_baseline
                ==
                (
                    result.metrics[
                        "rmse"
                    ]
                    <
                    result.naive_baseline_metrics[
                        "rmse"
                    ]
                )
            )


            np.testing.assert_allclose(
                result.rmse_delta_vs_naive,
                (
                    result.naive_baseline_metrics[
                        "rmse"
                    ]
                    -
                    result.metrics[
                        "rmse"
                    ]
                ),
                rtol=0.0,
                atol=0.0,
            )


            assert (
                result.model_artifact
                .training_contract
                ==
                contract
            )


            assert (
                result.model_artifact
                .serialization_format
                ==
                "pytorch_bundle"
            )


            assert (
                result.experiment_provenance
                ==
                result.model_artifact
                .experiment_provenance
            )


            results.append(
                result
            )


        assert (
            {
                result.estimator_key
                for result
                in results
            }
            ==
            {
                "time_series_mlp_regressor",
                "time_series_rnn_regressor",
                "time_series_lstm_regressor",
            }
        )


        print(
            "[PASS] Model Lab dispatches MLP/RNN/LSTM from exact Training Contract"
        )


        # ====================================================
        # 4. ONE ARTIFACT PER MODEL LAB EXECUTION
        # ====================================================


        listed = (
            list_ml_model_artifacts(
                workflow_id=
                    WORKFLOW_ID
            )
        )


        assert (
            len(
                listed
            )
            ==
            3
        )


        assert (
            {
                artifact.training_contract
                .estimator_key
                for artifact
                in listed
            }
            ==
            {
                "time_series_mlp_regressor",
                "time_series_rnn_regressor",
                "time_series_lstm_regressor",
            }
        )


        print(
            "[PASS] each forecasting Model Lab execution persists one Artifact"
        )


        # ====================================================
        # 5. STORED MODEL REMAINS EXECUTABLE
        # ====================================================


        for result in results:

            stored_bytes = (
                load_ml_model_artifact_binary(
                    model_id=
                        result.model_artifact
                        .model_id,

                    workflow_id=
                        WORKFLOW_ID,
                )
            )


            restored = (
                deserialize_trusted_time_series_neural_bundle(
                    trusted_bundle_bytes=
                        stored_bytes,

                    training_contract=
                        result.model_artifact
                        .training_contract,
                )
            )


            # Use the same final 10 rolling-origin input
            # windows produced by this controlled linear series.
            test_inputs = np.array(
                [
                    [
                        float(
                            50
                            +
                            position
                            -
                            offset
                        )
                        for offset
                        in (
                            4,
                            3,
                            2,
                            1,
                        )
                    ]
                    for position
                    in range(
                        30,
                        40,
                    )
                ],
                dtype=np.float64,
            )


            inference = (
                predict_trusted_time_series_bundle(
                    components=
                        restored,

                    inputs=
                        test_inputs,
                )
            )


            assert (
                inference.estimator_key
                ==
                result.estimator_key
            )


            assert (
                inference.sample_count
                ==
                10
            )


            assert np.isfinite(
                inference.predictions
            ).all()


        print(
            "[PASS] Model Lab persisted forecasting Artifacts remain inferable"
        )


        # ====================================================
        # 6. PRIVACY-MINIMAL RESULT
        # ====================================================


        forbidden_result_fields = (
            "predictions",
            "targets",
            "model",
            "standardizer",
            "epoch_losses",
            "windows",
            "dataframe",
            "train_positions",
            "test_positions",
        )


        for result in results:

            for field in forbidden_result_fields:

                assert not hasattr(
                    result,
                    field,
                )


        print(
            "[PASS] production Model Lab result remains privacy-minimal"
        )


        # ====================================================
        # 7. PREPARATION REVISION PIN FAILS BEFORE TRAINING
        # ====================================================


        artifacts_before = len(
            list_ml_model_artifacts(
                workflow_id=
                    WORKFLOW_ID
            )
        )


        original_mlp_executor = (
            executor_module
            .execute_time_series_mlp
        )


        training_called = {
            "value":
                False
        }


        def forbidden_training(
            **kwargs,
        ):

            training_called[
                "value"
            ] = True


            raise AssertionError(
                "Training must not execute after revision mismatch."
            )


        executor_module.execute_time_series_mlp = (
            forbidden_training
        )


        try:

            try:

                execute_time_series_model_lab(
                    training_contract=
                        contracts[
                            0
                        ],

                    expected_preparation_session_revision=
                        1,

                    execution_device=
                        "cpu",
                )

            except DLTimeSeriesModelLabInputError:

                pass

            else:

                raise AssertionError(
                    (
                        "Expected Preparation revision "
                        "mismatch failure."
                    )
                )

        finally:

            executor_module.execute_time_series_mlp = (
                original_mlp_executor
            )


        assert (
            training_called[
                "value"
            ]
            is False
        )


        artifacts_after = len(
            list_ml_model_artifacts(
                workflow_id=
                    WORKFLOW_ID
            )
        )


        assert (
            artifacts_after
            ==
            artifacts_before
        )


        print(
            "[PASS] stale Preparation revision fails before training or Artifact write"
        )


finally:

    executor_module.load_authorized_ml_dataframe = (
        original_loader
    )


# ============================================================
# 8. EXACT HANDOFF CALL COUNT
# ============================================================


assert (
    len(
        loader_calls
    )
    ==
    4
)


assert all(
    workflow_id
    ==
    WORKFLOW_ID
    and
    dataset_id
    ==
    DATASET_ID
    and
    execution_label
    ==
    "Time-series Model Lab"

    for (
        workflow_id,
        dataset_id,
        execution_label,
    )
    in loader_calls
)


print(
    "[PASS] Model Lab execution always resolves server-owned workflow/dataset authority"
)


# ============================================================
# 9. STATIC EXECUTION BOUNDARIES
# ============================================================


source_path = Path(
    "app/deep_learning/"
    "time_series_model_lab_executor.py"
)


source_text = source_path.read_text(
    encoding="utf-8"
)


tree = ast.parse(
    source_text
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


assert (
    "app.deep_learning.time_series_comparison_executor"
    not in
    imports
)


assert (
    "serialize_time_series_neural_bundle"
    not in
    source_text
)


assert (
    "register_ml_model_artifact"
    not in
    source_text
)


assert (
    "register_time_series_neural_execution_artifact"
    in
    source_text
)


assert (
    "load_authorized_ml_dataframe"
    in
    source_text
)


assert (
    "expected_preparation_session_revision"
    in
    source_text
)


print(
    "[PASS] Model Lab executor delegates Artifact lifecycle and keeps A10 separate"
)


print()
print("=" * 80)
print("DL-5-A13-P2 FINAL VERDICT")
print("=" * 80)
print()

print("Server-owned Preparation handoff             PASS")
print("Preparation revision pin                    PASS")
print("A2 forecasting input authority              PASS")
print("A3 chronological holdout                    PASS")
print("A4 supervised windows                       PASS")
print("A5 naive baseline                           PASS")
print("Single selected estimator dispatch           PASS")
print("MLP Model Lab execution                     PASS")
print("RNN Model Lab execution                     PASS")
print("LSTM Model Lab execution                    PASS")
print("A13-P1 Artifact delegation                  PASS")
print("Generic Experiment Provenance               PASS")
print("Privacy-minimal production result            PASS")
print("Persisted Artifact inference                 PASS")
print("No four-model retraining per API request     PASS")

print()
print(
    "DL-5-A13-P2 - MODEL LAB FORECASTING EXECUTOR: PASS"
)
