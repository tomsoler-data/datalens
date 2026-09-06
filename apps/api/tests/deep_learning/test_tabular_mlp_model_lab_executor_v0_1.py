from __future__ import annotations


from contextlib import (
    contextmanager,
)


import os
import tempfile


from dataclasses import (
    dataclass,
)


from pathlib import (
    Path,
)


import numpy as np
import pandas as pd


import app.deep_learning.model_lab_executor as executor_module


from app.deep_learning.model_loader import (
    load_trusted_tabular_mlp_model,
)


from app.ml.model_artifact_store import (
    get_ml_model_artifact,
)


from app.persistence.sqlite_database import (
    sqlite_connection,
)


from app.deep_learning.model_lab_executor import (
    DL_TABULAR_MLP_EXECUTOR_RULE_VERSION,
    TabularMLPExecutionInputError,
    TabularMLPExecutionResult,
    execute_tabular_mlp,
)


from app.ml.baseline import (
    build_ml_baseline_evaluation,
    build_ml_baseline_predictions,
    compare_model_to_baseline,
)


from app.ml.contracts import (
    MLTrainingContract,
)


from app.ml.model_metrics import (
    compute_ml_regression_metrics,
    project_ml_baseline_metrics_v0_1,
)


from app.ml.splitting import (
    resolve_ml_holdout_partition,
)


from app.ml.training_input import (
    validate_and_extract_ml_xy,
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
    contract: MLTrainingContract,
    revision: int = 23,
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
            "datalens-mlp-production-"
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
                        "2026-09-06T14:00:00+00:00",
                        "2026-09-06T14:00:00+00:00",
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
) -> pd.DataFrame:

    rows = 96


    marketing_spend = np.linspace(
        20.0,
        120.0,
        rows,
        dtype=np.float64,
    )


    store_area = np.linspace(
        40.0,
        140.0,
        rows,
        dtype=np.float64,
    )


    monthly_revenue = (
        4.5
        *
        marketing_spend
        +
        2.0
        *
        store_area
        +
        30.0
    )


    return (
        pd.DataFrame(
            {
                "marketing_spend":
                    marketing_spend,

                "store_area":
                    store_area,

                "monthly_revenue":
                    monthly_revenue,
            },
            index=[
                50_000
                +
                index
                *
                11

                for index
                in range(
                    rows
                )
            ],
        )
    )


def build_contract(
) -> MLTrainingContract:

    return (
        MLTrainingContract(
            workflow_id=
                "prep:mlp-production",

            dataset_id=
                "dataset:validated",

            problem_type=
                "regression",

            target_column=
                "monthly_revenue",

            feature_columns=[
                "marketing_spend",
                "store_area",
            ],

            estimator_key=
                "tabular_mlp_regressor",

            estimator_hyperparameters={
                "kind":
                    "tabular_mlp_regressor",

                "hidden_features":
                    16,

                "epochs":
                    100,

                "batch_size":
                    24,

                "learning_rate":
                    0.0005,
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
                    True,

                "stratify":
                    False,
            },
        )
    )


# ============================================================
# PRODUCTION END-TO-END
# ============================================================


def test_production_mlp_executes_from_preparation_handoff(
) -> None:

    dataframe = (
        build_dataframe()
    )


    contract = (
        build_contract()
    )


    with patched_handoff(
        dataframe=dataframe,
        contract=contract,
        revision=23,
    ):

        result = (
            execute_tabular_mlp(
                training_contract=
                    contract,

                expected_preparation_session_revision=
                    23,

                execution_device=
                    "cpu",
            )
        )


        # ----------------------------------------------------
        # Persisted lifecycle checks must run while the
        # isolated SQLite database and Model Artifact Store
        # owned by patched_handoff are still alive.
        # ----------------------------------------------------


        restored_artifact = (
            get_ml_model_artifact(
                model_id=
                    result.model_artifact.model_id,

                workflow_id=
                    result.workflow_id,
            )
        )


        assert (
            restored_artifact
            ==
            result.model_artifact
        )


        loaded = (
            load_trusted_tabular_mlp_model(
                workflow_id=
                    result.workflow_id,

                model_id=
                    result.model_artifact.model_id,
            )
        )


        inference_features = (
            dataframe.loc[
                :,
                contract.feature_columns,
            ]
            .copy(
                deep=True
            )
        )


        restored_predictions = (
            loaded.predict(
                inference_features
            )
        )


        assert (
            restored_predictions.shape
            ==
            (
                len(
                    dataframe
                ),
            )
        )


        assert (
            np.isfinite(
                restored_predictions
            )
            .all()
        )


    assert isinstance(
        result,
        TabularMLPExecutionResult,
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
        23
    )


    assert (
        result.estimator_key
        ==
        "tabular_mlp_regressor"
    )


    assert (
        result.problem_type
        ==
        "regression"
    )


    assert (
        result.train_rows
        ==
        72
    )


    assert (
        result.test_rows
        ==
        24
    )


    assert (
        result.purged_rows
        ==
        0
    )


    assert (
        result.baseline.train_rows
        ==
        result.train_rows
    )


    assert (
        result.baseline.test_rows
        ==
        result.test_rows
    )


    assert (
        result.baseline.primary_metric
        ==
        "rmse"
    )


    assert (
        result.baseline_comparison.primary_metric
        ==
        "rmse"
    )


    assert (
        result.baseline_comparison.beats_baseline
    )


    assert (
        result.model_artifact.serialization_format
        ==
        "pytorch_bundle"
    )


    assert (
        result.model_artifact.model_path
        .endswith(
            ".ptbundle"
        )
    )


    assert (
        result.model_artifact.training_contract
        ==
        contract
    )


    assert (
        result.experiment_provenance
        ==
        result
        .model_artifact
        .experiment_provenance
    )


    assert (
        result.experiment_provenance.model_id
        ==
        result.model_artifact.model_id
    )


# ============================================================
# EXACT BASELINE POPULATION
# ============================================================


def test_baseline_reuses_exact_mlp_holdout_population(
) -> None:

    dataframe = (
        build_dataframe()
    )


    contract = (
        build_contract()
    )


    (
        x,
        y,
    ) = (
        validate_and_extract_ml_xy(
            dataframe=dataframe,
            contract=contract,
        )
    )


    partition = (
        resolve_ml_holdout_partition(
            x=x,
            y=y,
            contract=contract,
            dataframe=dataframe,
        )
    )


    y_train = (
        y.iloc[
            list(
                partition
                .train_positions
            )
        ]
    )


    y_test = (
        y.iloc[
            list(
                partition
                .test_positions
            )
        ]
    )


    prediction_bundle = (
        build_ml_baseline_predictions(
            problem_type=
                "regression",

            y_train=
                y_train,

            test_rows=
                len(
                    y_test
                ),
        )
    )


    full_metrics = (
        compute_ml_regression_metrics(
            y_true=
                y_test,

            predictions=
                prediction_bundle.predictions,
        )
    )


    projected_metrics = (
        project_ml_baseline_metrics_v0_1(
            problem_type=
                "regression",

            metrics=
                full_metrics,
        )
    )


    expected_baseline = (
        build_ml_baseline_evaluation(
            problem_type=
                "regression",

            strategy=
                prediction_bundle.strategy,

            metrics=
                projected_metrics,

            train_rows=
                len(
                    y_train
                ),

            test_rows=
                len(
                    y_test
                ),
        )
    )


    with patched_handoff(
        dataframe=dataframe,
        contract=contract,
    ):

        result = (
            execute_tabular_mlp(
                training_contract=
                    contract,

                execution_device=
                    "cpu",
            )
        )


    assert (
        result.baseline
        ==
        expected_baseline
    )


    expected_comparison = (
        compare_model_to_baseline(
            problem_type=
                "regression",

            model_metrics=
                result.metrics,

            baseline_metrics=
                expected_baseline.metrics,
        )
    )


    assert (
        result.baseline_comparison
        ==
        expected_comparison
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
        build_contract()
    )


    original_core = (
        executor_module
        .execute_tabular_mlp_core
    )


    core_called = False


    def forbidden_core(
        **kwargs,
    ):

        nonlocal core_called

        core_called = True

        raise AssertionError(
            "MLP core must not run after revision mismatch."
        )


    executor_module.execute_tabular_mlp_core = (
        forbidden_core
    )


    try:

        with patched_handoff(
            dataframe=dataframe,
            contract=contract,
            revision=23,
        ):

            try:

                execute_tabular_mlp(
                    training_contract=
                        contract,

                    expected_preparation_session_revision=
                        22,

                    execution_device=
                        "cpu",
                )

            except TabularMLPExecutionInputError:
                pass

            else:
                raise AssertionError(
                    (
                        "Preparation revision mismatch "
                        "must fail closed."
                    )
                )

    finally:

        executor_module.execute_tabular_mlp_core = (
            original_core
        )


    assert (
        core_called
        is False
    )


# ============================================================
# DETERMINISM
# ============================================================


def test_production_result_is_deterministic_on_cpu(
) -> None:

    dataframe = (
        build_dataframe()
    )


    contract = (
        build_contract()
    )


    with patched_handoff(
        dataframe=dataframe,
        contract=contract,
    ):

        first = (
            execute_tabular_mlp(
                training_contract=
                    contract,

                execution_device=
                    "cpu",
            )
        )


        second = (
            execute_tabular_mlp(
                training_contract=
                    contract,

                execution_device=
                    "cpu",
            )
        )


    assert (
        first.workflow_id
        ==
        second.workflow_id
    )


    assert (
        first.dataset_id
        ==
        second.dataset_id
    )


    assert (
        first.problem_type
        ==
        second.problem_type
    )


    assert (
        first.estimator_key
        ==
        second.estimator_key
    )


    assert (
        first.train_rows
        ==
        second.train_rows
    )


    assert (
        first.test_rows
        ==
        second.test_rows
    )


    assert (
        first.purged_rows
        ==
        second.purged_rows
    )


    assert (
        first.metrics
        ==
        second.metrics
    )


    assert (
        first.baseline
        ==
        second.baseline
    )


    assert (
        first.baseline_comparison
        ==
        second.baseline_comparison
    )


    assert (
        first.model_artifact.model_sha256
        ==
        second.model_artifact.model_sha256
    )


    assert (
        first
        .experiment_provenance
        .training_contract_sha256
        ==
        second
        .experiment_provenance
        .training_contract_sha256
    )


# ============================================================
# PRIVACY-MINIMAL RESULT
# ============================================================


def test_result_exposes_no_runtime_model_data(
) -> None:

    dataframe = (
        build_dataframe()
    )


    contract = (
        build_contract()
    )


    with patched_handoff(
        dataframe=dataframe,
        contract=contract,
    ):

        result = (
            execute_tabular_mlp(
                training_contract=
                    contract,

                execution_device=
                    "cpu",
            )
        )


    payload = (
        result.model_dump(
            mode="json"
        )
    )


    forbidden = {
        "predictions",
        "model",
        "preprocessor",
        "partition",
        "train_positions",
        "test_positions",
        "purged_positions",
        "epoch_losses",
        "test_loss",
    }


    assert (
        forbidden
        .isdisjoint(
            payload
        )
    )


    assert (
        "model_artifact"
        in
        payload
    )


    assert (
        "experiment_provenance"
        in
        payload
    )


    artifact_payload = (
        payload[
            "model_artifact"
        ]
    )


    assert (
        "model_path"
        in
        artifact_payload
    )


    assert (
        "model_sha256"
        in
        artifact_payload
    )


    for forbidden_runtime_field in (
        "model",
        "preprocessor",
        "predictions",
        "state_dict",
        "model_bytes",
    ):

        assert (
            forbidden_runtime_field
            not in
            artifact_payload
        )


# ============================================================
# SOURCE BOUNDARY
# ============================================================


def test_production_executor_lives_on_dl_side(
) -> None:

    root = (
        Path(__file__)
        .parents[
            2
        ]
    )


    dl_source = (
        root
        /
        "app"
        /
        "deep_learning"
        /
        "model_lab_executor.py"
    ).read_text(
        encoding="utf-8"
    )


    assert (
        "execute_tabular_mlp_core"
        in
        dl_source
    )


    for relative_path in (
        "app/ml/training_input.py",
        "app/ml/splitting.py",
        "app/ml/contracts.py",
        "app/ml/training_estimator_contracts.py",
    ):

        source = (
            root
            /
            relative_path
        ).read_text(
            encoding="utf-8"
        )


        for token in (
            "import torch",
            "from torch",
            "torch.",
        ):

            assert (
                token
                not in
                source
            )


# ============================================================
# RULE VERSION
# ============================================================


def test_rule_version(
) -> None:

    assert (
        DL_TABULAR_MLP_EXECUTOR_RULE_VERSION
        ==
        "dl_tabular_mlp_executor_v0.1"
    )


# ============================================================
# MAIN
# ============================================================


def main(
) -> None:

    print(
        "=== DATALENS TABULAR MLP MODEL LAB EXECUTOR v0.1 ==="
    )

    print()


    test_production_mlp_executes_from_preparation_handoff()

    print(
        "Preparation -> MLP -> persisted artifact: PASS"
    )


    test_baseline_reuses_exact_mlp_holdout_population()

    print(
        "Exact holdout baseline population: PASS"
    )


    test_revision_mismatch_fails_before_training()

    print(
        "Preparation revision pin: PASS"
    )


    test_production_result_is_deterministic_on_cpu()

    print(
        "Deterministic production result: PASS"
    )


    test_result_exposes_no_runtime_model_data()

    print(
        "Privacy-minimal production result: PASS"
    )


    test_production_executor_lives_on_dl_side()

    print(
        "Runtime / Deep Learning boundary: PASS"
    )


    test_rule_version()

    print(
        "Production executor rule version: PASS"
    )


    print()

    print(
        "PASS - DataLens Tabular MLP Model Lab Executor v0.1"
    )


if __name__ == "__main__":
    main()
