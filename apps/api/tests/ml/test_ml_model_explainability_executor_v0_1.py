from __future__ import annotations


from types import (
    SimpleNamespace,
)


import numpy as np
import pandas as pd


from app.ml.classical_executor import (
    _build_estimator,
    _split_dataset,
    _validate_and_extract_xy,
)


from app.ml.contracts import (
    MLTrainingContract,
)


from app.ml.experiment_provenance import (
    build_ml_experiment_provenance,
    ml_training_contract_sha256,
)


from app.ml.model_artifacts import (
    MLModelArtifactRecord,
)


from app.ml.model_explainability import (
    MLModelExplainabilityContract,
)


import app.ml.model_explainability_executor as explainability_executor


from app.ml.model_explainability_executor import (
    ML_MODEL_EXPLAINABILITY_EXECUTOR_RULE_VERSION,
    MLModelExplainabilityArtifactError,
    MLModelExplainabilityInputError,
    execute_ml_model_explainability,
)


from app.ml.model_loader import (
    LoadedMLModel,
)


# ============================================================
# DATA
# ============================================================


def regression_dataframe(
) -> pd.DataFrame:

    rows = []


    for index in range(
        40
    ):
        age = (
            20
            +
            index
        )


        tenure = (
            1
            +
            (
                index
                %
                12
            )
        )


        segment = (
            "premium"
            if (
                index
                %
                2
                ==
                0
            )
            else
            "standard"
        )


        revenue = (
            50.0
            +
            (
                2.0
                *
                age
            )
            +
            (
                3.0
                *
                tenure
            )
            +
            (
                20.0
                if segment
                ==
                "premium"
                else 0.0
            )
        )


        rows.append(
            {
                "age":
                    age,

                "tenure":
                    tenure,

                "segment":
                    segment,

                "revenue":
                    revenue,
            }
        )


    return pd.DataFrame(
        rows
    )


def classification_dataframe(
) -> pd.DataFrame:

    rows = []


    for index in range(
        60
    ):
        x1 = float(
            index
        )


        x2 = float(
            (
                index
                %
                7
            )
        )


        label = (
            "high"
            if index
            >=
            30
            else
            "low"
        )


        rows.append(
            {
                "x1":
                    x1,

                "x2":
                    x2,

                "label":
                    label,
            }
        )


    return pd.DataFrame(
        rows
    )


# ============================================================
# CONTRACTS
# ============================================================


def regression_contract(
) -> MLTrainingContract:

    return (
        MLTrainingContract(
            workflow_id=
                "prep:explainability-regression",

            dataset_id=
                "dataset:regression",

            problem_type=
                "regression",

            target_column=
                "revenue",

            feature_columns=[
                "age",
                "tenure",
                "segment",
            ],

            categorical_feature_columns=[
                "segment",
            ],

            estimator_key=
                "linear_regression",
        )
    )


def classification_contract(
) -> MLTrainingContract:

    return (
        MLTrainingContract(
            workflow_id=
                "prep:explainability-classification",

            dataset_id=
                "dataset:classification",

            problem_type=
                "classification",

            target_column=
                "label",

            feature_columns=[
                "x1",
                "x2",
            ],

            categorical_feature_columns=[],

            estimator_key=
                "logistic_regression",

            split={
                "strategy":
                    "holdout",

                "test_size":
                    0.20,

                "random_seed":
                    42,

                "shuffle":
                    True,

                "stratify":
                    True,
            },
        )
    )


# ============================================================
# TRUSTED MODEL FIXTURE
# ============================================================


def build_loaded_model(
    *,
    dataframe: pd.DataFrame,
    contract: MLTrainingContract,
    preparation_revision: int = 7,
    with_provenance: bool = True,
) -> LoadedMLModel:

    (
        x,
        y,
    ) = (
        _validate_and_extract_xy(
            dataframe=
                dataframe,

            contract=
                contract,
        )
    )


    (
        x_train,
        x_test,
        y_train,
        y_test,
    ) = (
        _split_dataset(
            x=
                x,

            y=
                y,

            contract=
                contract,
        )
    )


    estimator = (
        _build_estimator(
            contract=
                contract
        )
    )


    estimator.fit(
        x_train,
        y_train,
    )


    # If the explainability executor ever tries to refit the
    # trusted estimator, this test must fail immediately.
    def forbidden_fit(
        *args,
        **kwargs,
    ):
        raise AssertionError(
            (
                "Model Explainability must never "
                "refit the trusted estimator."
            )
        )


    estimator.fit = (
        forbidden_fit
    )


    model_id = (
        "model:explainability-test"
    )


    metrics = (
        {
            "rmse":
                1.0,
        }

        if contract.problem_type
        ==
        "regression"

        else {
            "f1_macro":
                1.0,
        }
    )


    provenance = (
        build_ml_experiment_provenance(
            training_contract=
                contract,

            preparation_session_revision=
                preparation_revision,

            model_id=
                model_id,

            train_rows=
                int(
                    len(
                        x_train
                    )
                ),

            test_rows=
                int(
                    len(
                        x_test
                    )
                ),

            metrics=
                metrics,
        )

        if with_provenance

        else None
    )


    artifact = (
        MLModelArtifactRecord(
            model_id=
                model_id,

            workflow_id=
                contract.workflow_id,

            dataset_id=
                contract.dataset_id,

            training_contract=
                contract,

            experiment_provenance=
                provenance,

            metrics=
                metrics,

            train_rows=
                int(
                    len(
                        x_train
                    )
                ),

            test_rows=
                int(
                    len(
                        x_test
                    )
                ),

            created_at_utc=
                "2026-08-29T00:00:00+00:00",

            serialization_format=
                "joblib",

            model_path=
                "models/explainability-test.joblib",

            model_file_bytes=
                100,

            model_sha256=
                (
                    "a"
                    *
                    64
                ),
        )
    )


    return (
        LoadedMLModel(
            artifact=
                artifact,

            estimator=
                estimator,
        )
    )


# ============================================================
# PATCH CONTEXT
# ============================================================


class ExecutorContext:

    def __init__(
        self,
        *,
        dataframe: pd.DataFrame,
        contract: MLTrainingContract,
        preparation_revision: int = 7,
        with_provenance: bool = True,
    ) -> None:

        self.dataframe = (
            dataframe.copy(
                deep=True
            )
        )


        self.contract = (
            contract
        )


        self.preparation_revision = (
            preparation_revision
        )


        self.loaded_model = (
            build_loaded_model(
                dataframe=
                    self.dataframe,

                contract=
                    contract,

                preparation_revision=
                    preparation_revision,

                with_provenance=
                    with_provenance,
            )
        )


        self.original_loader = (
            explainability_executor
            .load_trusted_ml_model
        )


        self.original_dataframe_loader = (
            explainability_executor
            ._load_authorized_dataframe
        )


        self.original_permutation = (
            explainability_executor
            .permutation_importance
        )


        self.permutation_test_indexes = (
            None
        )


    def __enter__(
        self,
    ):

        def fake_loader(
            *,
            workflow_id: str,
            model_id: str,
        ):

            assert (
                workflow_id
                ==
                self.contract
                .workflow_id
            )


            assert (
                model_id
                ==
                self.loaded_model
                .artifact
                .model_id
            )


            return (
                self.loaded_model
            )


        def fake_dataframe_loader(
            *,
            contract,
        ):

            assert (
                contract
                ==
                self.contract
            )


            return (
                self.dataframe.copy(
                    deep=True
                ),
                self.preparation_revision,
            )


        real_permutation = (
            self.original_permutation
        )


        def recording_permutation(
            estimator,
            x,
            y,
            **kwargs,
        ):

            self.permutation_test_indexes = (
                list(
                    x.index
                )
            )


            return (
                real_permutation(
                    estimator,
                    x,
                    y,
                    **kwargs,
                )
            )


        explainability_executor.load_trusted_ml_model = (
            fake_loader
        )


        explainability_executor._load_authorized_dataframe = (
            fake_dataframe_loader
        )


        explainability_executor.permutation_importance = (
            recording_permutation
        )


        return self


    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ) -> None:

        explainability_executor.load_trusted_ml_model = (
            self.original_loader
        )


        explainability_executor._load_authorized_dataframe = (
            self.original_dataframe_loader
        )


        explainability_executor.permutation_importance = (
            self.original_permutation
        )


# ============================================================
# REGRESSION
# ============================================================


def test_regression_explainability_uses_holdout_only(
) -> None:

    dataframe = (
        regression_dataframe()
    )


    contract = (
        regression_contract()
    )


    (
        x,
        y,
    ) = (
        _validate_and_extract_xy(
            dataframe=
                dataframe,

            contract=
                contract,
        )
    )


    (
        _x_train,
        expected_x_test,
        _y_train,
        _y_test,
    ) = (
        _split_dataset(
            x=
                x,

            y=
                y,

            contract=
                contract,
        )
    )


    with ExecutorContext(
        dataframe=
            dataframe,

        contract=
            contract,
    ) as context:

        result = (
            execute_ml_model_explainability(
                workflow_id=
                    contract.workflow_id,

                model_id=
                    context.loaded_model
                    .artifact
                    .model_id,

                explainability_contract=(
                    MLModelExplainabilityContract(
                        n_repeats=
                            5,

                        random_seed=
                            123,
                    )
                ),
            )
        )


        assert (
            context
            .permutation_test_indexes
            ==
            list(
                expected_x_test.index
            )
        )


    assert (
        result.problem_type
        ==
        "regression"
    )


    assert (
        result.scoring
        ==
        "neg_root_mean_squared_error"
    )


    assert (
        result.evaluation_rows
        ==
        len(
            expected_x_test
        )
    )


    assert (
        {
            item.feature_name

            for item
            in result.feature_importances
        }
        ==
        {
            "age",
            "tenure",
            "segment",
        }
    )


    assert (
        [
            item.rank

            for item
            in result.feature_importances
        ]
        ==
        [
            1,
            2,
            3,
        ]
    )


    for item in (
        result.feature_importances
    ):

        assert np.isfinite(
            item.importance_mean
        )


        assert np.isfinite(
            item.importance_std
        )


        assert (
            item.importance_std
            >=
            0.0
        )


# ============================================================
# DETERMINISM
# ============================================================


def test_same_seed_is_deterministic(
) -> None:

    dataframe = (
        regression_dataframe()
    )


    contract = (
        regression_contract()
    )


    config = (
        MLModelExplainabilityContract(
            n_repeats=
                7,

            random_seed=
                88,
        )
    )


    with ExecutorContext(
        dataframe=
            dataframe,

        contract=
            contract,
    ) as context:

        first = (
            execute_ml_model_explainability(
                workflow_id=
                    contract.workflow_id,

                model_id=
                    context.loaded_model
                    .artifact
                    .model_id,

                explainability_contract=
                    config,
            )
        )


        second = (
            execute_ml_model_explainability(
                workflow_id=
                    contract.workflow_id,

                model_id=
                    context.loaded_model
                    .artifact
                    .model_id,

                explainability_contract=
                    config,
            )
        )


    assert (
        first.model_dump(
            mode="json"
        )
        ==
        second.model_dump(
            mode="json"
        )
    )


# ============================================================
# CLASSIFICATION
# ============================================================


def test_classification_uses_f1_macro(
) -> None:

    dataframe = (
        classification_dataframe()
    )


    contract = (
        classification_contract()
    )


    with ExecutorContext(
        dataframe=
            dataframe,

        contract=
            contract,
    ) as context:

        result = (
            execute_ml_model_explainability(
                workflow_id=
                    contract.workflow_id,

                model_id=
                    context.loaded_model
                    .artifact
                    .model_id,

                explainability_contract=(
                    MLModelExplainabilityContract(
                        n_repeats=
                            5,

                        random_seed=
                            91,
                    )
                ),
            )
        )


    assert (
        result.problem_type
        ==
        "classification"
    )


    assert (
        result.scoring
        ==
        "f1_macro"
    )


    assert (
        {
            item.feature_name

            for item
            in result.feature_importances
        }
        ==
        {
            "x1",
            "x2",
        }
    )


# ============================================================
# PROVENANCE
# ============================================================


def test_result_is_bound_to_artifact_provenance(
) -> None:

    dataframe = (
        regression_dataframe()
    )


    contract = (
        regression_contract()
    )


    with ExecutorContext(
        dataframe=
            dataframe,

        contract=
            contract,
    ) as context:

        result = (
            execute_ml_model_explainability(
                workflow_id=
                    contract.workflow_id,

                model_id=
                    context.loaded_model
                    .artifact
                    .model_id,

                explainability_contract=(
                    MLModelExplainabilityContract()
                ),
            )
        )


        artifact = (
            context.loaded_model
            .artifact
        )


    provenance = (
        artifact
        .experiment_provenance
    )


    assert (
        provenance
        is not None
    )


    assert (
        result.model_id
        ==
        artifact.model_id
    )


    assert (
        result.experiment_id
        ==
        provenance.experiment_id
    )


    assert (
        result.preparation_session_revision
        ==
        provenance.preparation_session_revision
    )


    assert (
        result.training_contract_sha256
        ==
        ml_training_contract_sha256(
            contract
        )
    )


# ============================================================
# REVISION FAIL-CLOSED
# ============================================================


def test_preparation_revision_mismatch_fails_closed(
) -> None:

    dataframe = (
        regression_dataframe()
    )


    contract = (
        regression_contract()
    )


    with ExecutorContext(
        dataframe=
            dataframe,

        contract=
            contract,

        preparation_revision=
            7,
    ) as context:

        context.preparation_revision = (
            8
        )


        try:
            execute_ml_model_explainability(
                workflow_id=
                    contract.workflow_id,

                model_id=
                    context.loaded_model
                    .artifact
                    .model_id,

                explainability_contract=(
                    MLModelExplainabilityContract()
                ),
            )

        except MLModelExplainabilityInputError:
            pass

        else:
            raise AssertionError(
                (
                    "Preparation revision mismatch "
                    "must fail closed."
                )
            )


# ============================================================
# PROVENANCE REQUIRED
# ============================================================


def test_missing_experiment_provenance_fails_closed(
) -> None:

    dataframe = (
        regression_dataframe()
    )


    contract = (
        regression_contract()
    )


    with ExecutorContext(
        dataframe=
            dataframe,

        contract=
            contract,

        with_provenance=
            False,
    ) as context:

        try:
            execute_ml_model_explainability(
                workflow_id=
                    contract.workflow_id,

                model_id=
                    context.loaded_model
                    .artifact
                    .model_id,

                explainability_contract=(
                    MLModelExplainabilityContract()
                ),
            )

        except MLModelExplainabilityArtifactError:
            pass

        else:
            raise AssertionError(
                (
                    "Model Explainability v0.1 must "
                    "require Experiment Provenance."
                )
            )


# ============================================================
# PRIVACY-MINIMAL RESULT
# ============================================================


def test_result_contains_no_raw_rows_or_predictions(
) -> None:

    dataframe = (
        regression_dataframe()
    )


    contract = (
        regression_contract()
    )


    with ExecutorContext(
        dataframe=
            dataframe,

        contract=
            contract,
    ) as context:

        result = (
            execute_ml_model_explainability(
                workflow_id=
                    contract.workflow_id,

                model_id=
                    context.loaded_model
                    .artifact
                    .model_id,

                explainability_contract=(
                    MLModelExplainabilityContract()
                ),
            )
        )


    payload = (
        result.model_dump(
            mode="json"
        )
    )


    forbidden = {
        "raw_rows",
        "predictions",
        "permuted_rows",
        "x_train",
        "x_test",
        "y_train",
        "y_test",
        "estimator",
        "model_path",
        "model_bytes",
    }


    assert (
        forbidden
        .isdisjoint(
            payload
        )
    )


# ============================================================
# VERSION
# ============================================================


def test_rule_version(
) -> None:

    assert (
        ML_MODEL_EXPLAINABILITY_EXECUTOR_RULE_VERSION
        ==
        "ml_model_explainability_executor_v0.1"
    )


# ============================================================
# MAIN
# ============================================================


def main(
) -> None:

    print(
        "=== DATALENS ML MODEL EXPLAINABILITY EXECUTOR v0.1 ==="
    )


    tests = [
        (
            "Regression holdout-only permutation importance",
            test_regression_explainability_uses_holdout_only,
        ),
        (
            "Same seed is deterministic",
            test_same_seed_is_deterministic,
        ),
        (
            "Classification uses f1_macro",
            test_classification_uses_f1_macro,
        ),
        (
            "Artifact provenance binding",
            test_result_is_bound_to_artifact_provenance,
        ),
        (
            "Preparation revision mismatch fail-closed",
            test_preparation_revision_mismatch_fails_closed,
        ),
        (
            "Experiment Provenance required",
            test_missing_experiment_provenance_fails_closed,
        ),
        (
            "Privacy-minimal result",
            test_result_contains_no_raw_rows_or_predictions,
        ),
        (
            "Rule version",
            test_rule_version,
        ),
    ]


    for (
        label,
        test,
    ) in tests:

        test()

        print(
            f"[PASS] {label}"
        )


    print()
    print(
        "PASS - ML Model Explainability Executor v0.1"
    )


if __name__ == "__main__":
    main()
