from __future__ import annotations

import os


import tempfile


from pathlib import (
    Path,
)


from app.persistence.sqlite_database import (
    sqlite_connection,
)


from contextlib import (
    contextmanager,
)


from dataclasses import (
    dataclass,
)


import math
import numpy as np
import pandas as pd
import torch


import app.deep_learning.anomaly_model_lab_executor as executor_module


from app.deep_learning.anomaly_model_lab_executor import (
    DL_TABULAR_AUTOENCODER_EXECUTOR_RULE_VERSION,
    TabularAutoencoderExecutionInputError,
    TabularAutoencoderExecutionResult,
    execute_tabular_autoencoder,
)


from app.deep_learning.autoencoder_executor import (
    DL_TABULAR_AUTOENCODER_CORE_EXECUTOR_RULE_VERSION,
    execute_tabular_autoencoder_core,
)


from app.ml.anomaly_contracts import (
    MLAnomalyTrainingContract,
)


from app.ml.anomaly_training_input import (
    validate_and_extract_anomaly_x,
)


# ============================================================
# FAKE PREPARATION HANDOFF
# ============================================================


@dataclass(
    frozen=True,
)
class FakeAnalysisInputHandoff:

    workflow_id: str

    session_revision: int

    dataset_ids: tuple[
        str,
        ...
    ]

    dataset_records: tuple[
        dict,
        ...
    ]


@contextmanager
def patched_handoff(
    *,
    dataframe: pd.DataFrame,
    contract: MLAnomalyTrainingContract,
    revision: int = 31,
):

    original = (
        executor_module
        .load_validated_analysis_input
    )


    previous_sqlite = os.environ.get(
        "DATALENS_SQLITE_PATH"
    )


    previous_store = os.environ.get(
        "DATALENS_ML_MODEL_ARTIFACT_STORE_PATH"
    )


    with tempfile.TemporaryDirectory(
        prefix=
            "datalens-autoencoder-a7-"
    ) as root:

        root_path = Path(
            root
        )


        os.environ[
            "DATALENS_SQLITE_PATH"
        ] = str(
            root_path
            /
            "datalens.sqlite3"
        )


        os.environ[
            "DATALENS_ML_MODEL_ARTIFACT_STORE_PATH"
        ] = str(
            root_path
            /
            "ml"
            /
            "model_artifacts.json"
        )


        def fake_load_validated_analysis_input(
            *,
            workflow_id: str,
        ):

            assert (
                workflow_id
                ==
                contract.workflow_id
            )


            return (
                FakeAnalysisInputHandoff(
                    workflow_id=
                        workflow_id,
                    session_revision=
                        revision,
                    dataset_ids=(
                        contract.dataset_id,
                    ),
                    dataset_records=(
                        {
                            "dataset_id":
                                contract.dataset_id,
                            "dataframe":
                                dataframe.copy(
                                    deep=True
                                ),
                        },
                    ),
                )
            )


        executor_module.load_validated_analysis_input = (
            fake_load_validated_analysis_input
        )


        try:

            with sqlite_connection(
                write=True
            ) as connection:

                connection.execute(
                    """
                    INSERT INTO preparation_sessions (
                        workflow_id,
                        revision,
                        payload_json,
                        created_at,
                        updated_at
                    )
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        contract.workflow_id,
                        revision,
                        "{}",
                        "2026-09-07T15:00:00+00:00",
                        "2026-09-07T15:00:00+00:00",
                    ),
                )


                connection.execute(
                    """
                    INSERT INTO preparation_artifacts (
                        store_root,
                        workflow_id,
                        dataset_id,
                        dataset_filename,
                        stage,
                        rows,
                        columns,
                        parent_dataset_ids_json,
                        evidence_refs_json,
                        datetime_dtypes_json,
                        data_path
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        "test-preparation-root",
                        contract.workflow_id,
                        contract.dataset_id,
                        "validated.csv",
                        "source",
                        int(
                            len(
                                dataframe
                            )
                        ),
                        int(
                            len(
                                dataframe.columns
                            )
                        ),
                        "[]",
                        "[]",
                        "[]",
                        "data/validated.json.gz",
                    ),
                )


            yield


        finally:

            executor_module.load_validated_analysis_input = (
                original
            )


            if previous_sqlite is None:

                os.environ.pop(
                    "DATALENS_SQLITE_PATH",
                    None,
                )

            else:

                os.environ[
                    "DATALENS_SQLITE_PATH"
                ] = previous_sqlite


            if previous_store is None:

                os.environ.pop(
                    "DATALENS_ML_MODEL_ARTIFACT_STORE_PATH",
                    None,
                )

            else:

                os.environ[
                    "DATALENS_ML_MODEL_ARTIFACT_STORE_PATH"
                ] = previous_store


# ============================================================
# FIXTURE
# ============================================================


def build_dataframe(
    *,
    test_shift: float = 0.0,
) -> pd.DataFrame:

    rows = 80


    phase = (
        np.arange(
            rows,
            dtype=
                np.float64,
        )
        %
        20
    ) / 19.0


    feature_a = (
        phase.copy()
    )


    feature_b = (
        2.0
        *
        phase
        +
        0.25
    )


    feature_c = (
        0.5
        *
        phase
        -
        0.10
    )


    # Strong future anomalies are restricted to TEST because
    # the fixed holdout below uses the first 60 rows as TRAIN.
    feature_a[
        75:
    ] += 8.0

    feature_b[
        75:
    ] += 16.0

    feature_c[
        75:
    ] += 6.0


    if test_shift != 0.0:

        feature_a[
            60:
        ] += test_shift

        feature_b[
            60:
        ] += (
            2.0
            *
            test_shift
        )

        feature_c[
            60:
        ] += (
            0.5
            *
            test_shift
        )


    return (
        pd.DataFrame(
            {
                "feature_a":
                    feature_a,
                "feature_b":
                    feature_b,
                "feature_c":
                    feature_c,
            },
            index=[
                70_000
                +
                index
                *
                13

                for index
                in range(
                    rows
                )
            ],
        )
    )


def build_contract(
    *,
    epochs: int = 80,
) -> MLAnomalyTrainingContract:

    return (
        MLAnomalyTrainingContract(
            workflow_id=
                "prep:autoencoder-production",
            dataset_id=
                "dataset:autoencoder-validated",
            feature_columns=[
                "feature_a",
                "feature_b",
                "feature_c",
            ],
            estimator_hyperparameters={
                "kind":
                    "tabular_autoencoder",
                "hidden_features":
                    12,
                "latent_features":
                    2,
                "epochs":
                    epochs,
                "batch_size":
                    16,
                "learning_rate":
                    0.03,
            },
            preprocessing={
                "numeric_imputation":
                    "error",
                "categorical_imputation":
                    "error",
                "categorical_encoding":
                    "one_hot",
                "handle_unknown_categories":
                    "ignore",
                "scale_numeric":
                    True,
            },
            split={
                "strategy":
                    "holdout",
                "test_size":
                    0.25,
                "random_seed":
                    31,
                "shuffle":
                    False,
                "stratify":
                    False,
            },
            threshold_quantile=
                0.95,
        )
    )


# ============================================================
# PRODUCTION EXECUTION
# ============================================================


def test_production_execution(
) -> None:

    dataframe = (
        build_dataframe()
    )


    contract = (
        build_contract()
    )


    with patched_handoff(
        dataframe=
            dataframe,
        contract=
            contract,
        revision=
            31,
    ):

        result = (
            execute_tabular_autoencoder(
                training_contract=
                    contract,
                expected_preparation_session_revision=
                    31,
                execution_device=
                    "cpu",
            )
        )


    assert isinstance(
        result,
        TabularAutoencoderExecutionResult,
    )


    assert (
        result.workflow_id
        ==
        contract.workflow_id
    )


    assert (
        result.dataset_id
        ==
        contract.dataset_id
    )


    assert (
        result.preparation_session_revision
        ==
        31
    )


    assert (
        result.problem_type
        ==
        "anomaly_detection"
    )


    assert (
        result.estimator_key
        ==
        "tabular_autoencoder"
    )


    assert (
        result.train_rows
        ==
        60
    )


    assert (
        result.test_rows
        ==
        20
    )


    assert (
        result.purged_rows
        ==
        0
    )


    assert math.isfinite(
        result.train_reconstruction_mse
    )


    assert math.isfinite(
        result.test_reconstruction_mse
    )


    assert math.isfinite(
        result.anomaly_threshold
    )


    assert (
        result.threshold_quantile
        ==
        0.95
    )


    assert (
        0
        <=
        result.train_anomaly_count
        <=
        result.train_rows
    )


    assert (
        0
        <=
        result.test_anomaly_count
        <=
        result.test_rows
    )


    assert (
        result.test_anomaly_count
        >=
        1
    )


    assert math.isclose(
        result.train_anomaly_rate,
        (
            result.train_anomaly_count
            /
            result.train_rows
        ),
        rel_tol=
            1e-12,
        abs_tol=
            1e-12,
    )


    assert math.isclose(
        result.test_anomaly_rate,
        (
            result.test_anomaly_count
            /
            result.test_rows
        ),
        rel_tol=
            1e-12,
        abs_tol=
            1e-12,
    )


# ============================================================
# PRIVACY-MINIMAL RESULT
# ============================================================


def test_privacy_minimal_result_surface(
) -> None:

    fields = set(
        TabularAutoencoderExecutionResult
        .model_fields
    )


    # Lifecycle metadata is intentionally public.

    required_lifecycle_fields = {
        "model_artifact",
        "experiment_provenance",
    }


    assert (
        required_lifecycle_fields
        .issubset(
            fields
        )
    )


    # Learned runtime state and row-level evidence remain private.

    for forbidden in (
        "row_positions",
        "train_positions",
        "test_positions",
        "reconstruction_errors",
        "train_errors",
        "test_errors",
        "anomaly_flags",
        "train_flags",
        "test_flags",
        "predictions",
        "targets",
        "target_column",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "model",
        "preprocessor",
        "threshold",
        "epoch_losses",
    ):

        assert (
            forbidden
            not in
            fields
        )


# ============================================================
# TRAIN-ONLY PREPROCESSING + THRESHOLD
# ============================================================


def test_test_population_cannot_change_training_authorities(
) -> None:

    contract = (
        build_contract(
            epochs=
                50,
        )
    )


    first_dataframe = (
        build_dataframe(
            test_shift=
                0.0,
        )
    )


    second_dataframe = (
        build_dataframe(
            test_shift=
                100.0,
        )
    )


    first_x = (
        validate_and_extract_anomaly_x(
            dataframe=
                first_dataframe,
            contract=
                contract,
        )
    )


    second_x = (
        validate_and_extract_anomaly_x(
            dataframe=
                second_dataframe,
            contract=
                contract,
        )
    )


    first = (
        execute_tabular_autoencoder_core(
            x=
                first_x,
            contract=
                contract,
            dataframe=
                first_dataframe,
            execution_device=
                "cpu",
        )
    )


    second = (
        execute_tabular_autoencoder_core(
            x=
                second_x,
            contract=
                contract,
            dataframe=
                second_dataframe,
            execution_device=
                "cpu",
        )
    )


    # TEST values changed drastically, but TRAIN is identical.
    # Training, fitted preprocessing and TRAIN threshold must
    # therefore remain unchanged.
    assert math.isclose(
        first.threshold.threshold,
        second.threshold.threshold,
        rel_tol=
            1e-12,
        abs_tol=
            1e-12,
    )


    assert torch.equal(
        first
        .train_evaluation
        .reconstruction_errors,
        second
        .train_evaluation
        .reconstruction_errors,
    )


    scaler = (
        first
        .preprocessor
        .named_transformers_[
            "numeric"
        ]
        .named_steps[
            "scaler"
        ]
    )


    expected_train_mean = (
        first_x.iloc[
            :60
        ]
        .mean()
        .to_numpy(
            dtype=
                np.float64
        )
    )


    assert np.allclose(
        scaler.mean_,
        expected_train_mean,
        rtol=
            0.0,
        atol=
            1e-12,
    )


    expected_threshold = float(
        torch.quantile(
            first
            .train_evaluation
            .reconstruction_errors,
            q=
                contract.threshold_quantile,
            interpolation=
                "linear",
        )
        .item()
    )


    assert math.isclose(
        first.threshold.threshold,
        expected_threshold,
        rel_tol=
            0.0,
        abs_tol=
            0.0,
    )


# ============================================================
# PREPARATION REVISION PIN
# ============================================================


def test_revision_mismatch_fails_before_training(
) -> None:

    dataframe = (
        build_dataframe()
    )


    contract = (
        build_contract(
            epochs=
                10,
        )
    )


    original_core = (
        executor_module
        .execute_tabular_autoencoder_core
    )


    core_called = False


    def forbidden_core(
        **kwargs,
    ):

        nonlocal core_called

        core_called = True

        raise AssertionError(
            (
                "Autoencoder core must not run "
                "after revision mismatch."
            )
        )


    executor_module.execute_tabular_autoencoder_core = (
        forbidden_core
    )


    try:

        with patched_handoff(
            dataframe=
                dataframe,
            contract=
                contract,
            revision=
                31,
        ):

            try:

                execute_tabular_autoencoder(
                    training_contract=
                        contract,
                    expected_preparation_session_revision=
                        30,
                    execution_device=
                        "cpu",
                )

            except TabularAutoencoderExecutionInputError:
                pass

            else:

                raise AssertionError(
                    (
                        "Preparation revision mismatch "
                        "must fail closed."
                    )
                )

    finally:

        executor_module.execute_tabular_autoencoder_core = (
            original_core
        )


    assert (
        core_called
        is False
    )


# ============================================================
# CUDA CORE SMOKE
# ============================================================


def test_cuda_core_execution(
) -> None:

    if not torch.cuda.is_available():

        raise RuntimeError(
            (
                "DL-4 acceptance requires the "
                "validated CUDA environment."
            )
        )


    dataframe = (
        build_dataframe()
    )


    contract = (
        build_contract(
            epochs=
                10,
        )
    )


    x = (
        validate_and_extract_anomaly_x(
            dataframe=
                dataframe,
            contract=
                contract,
        )
    )


    result = (
        execute_tabular_autoencoder_core(
            x=
                x,
            contract=
                contract,
            dataframe=
                dataframe,
            execution_device=
                "cuda",
        )
    )


    assert (
        result.execution_device
        .startswith(
            "cuda"
        )
    )


    assert (
        result.train_rows
        ==
        60
    )


    assert (
        result.test_rows
        ==
        20
    )


    assert bool(
        torch.isfinite(
            result
            .train_evaluation
            .reconstruction_errors
        )
        .all()
    )


    assert bool(
        torch.isfinite(
            result
            .test_evaluation
            .reconstruction_errors
        )
        .all()
    )


# ============================================================
# RULE VERSIONS
# ============================================================


def test_rule_versions(
) -> None:

    assert (
        DL_TABULAR_AUTOENCODER_CORE_EXECUTOR_RULE_VERSION
        ==
        "dl_tabular_autoencoder_core_executor_v0.1"
    )


    assert (
        DL_TABULAR_AUTOENCODER_EXECUTOR_RULE_VERSION
        ==
        "dl_tabular_autoencoder_executor_v0.1"
    )


# ============================================================
# DIRECT ACCEPTANCE
# ============================================================


def main(
) -> None:

    print(
        "=== DATALENS TABULAR AUTOENCODER MODEL LAB EXECUTOR v0.1 ==="
    )

    print()


    test_production_execution()

    print(
        "Preparation-to-anomaly execution: PASS"
    )


    test_privacy_minimal_result_surface()

    print(
        "Privacy-minimal result surface: PASS"
    )


    test_test_population_cannot_change_training_authorities()

    print(
        "TRAIN-only preprocessing: PASS"
    )

    print(
        "TRAIN-only threshold fit: PASS"
    )

    print(
        "TEST leakage guard: PASS"
    )


    test_revision_mismatch_fails_before_training()

    print(
        "Preparation revision pin: PASS"
    )


    test_cuda_core_execution()

    print(
        "CUDA core execution: PASS"
    )


    test_rule_versions()

    print(
        "Executor rule versions: PASS"
    )


    print()

    print(
        "PASS - DataLens Tabular Autoencoder Model Lab Executor v0.1"
    )


if __name__ == "__main__":

    main()
