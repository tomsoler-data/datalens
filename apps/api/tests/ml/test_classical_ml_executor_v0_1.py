from __future__ import annotations


import io
import os
import tempfile


from contextlib import (
    contextmanager,
)


from dataclasses import (
    dataclass,
)


from pathlib import (
    Path,
)


import joblib
import numpy as np
import pandas as pd


import app.ml.classical_executor as executor_module


from app.ml.classical_executor import (
    CLASSICAL_ML_EXECUTOR_RULE_VERSION,
    ClassicalMLEstimatorError,
    ClassicalMLInputError,
    execute_classical_ml,
)


from app.ml.contracts import (
    MLSplitContract,
    MLTrainingContract,
)


from app.ml.model_artifact_store import (
    list_ml_model_artifacts,
    load_ml_model_artifact_binary,
)


from app.persistence.sqlite_database import (
    sqlite_connection,
)


from app.preparation.analysis_input_handoff import (
    AnalysisInputHandoffError,
)


# ============================================================
# FAKE HANDOFF READ MODEL
#
# Only the already-tested handoff boundary itself is replaced
# in these executor tests.
#
# Model Artifact persistence remains real.
# ============================================================


@dataclass(
    frozen=True
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


# ============================================================
# ISOLATION
# ============================================================


@contextmanager
def isolated_environment(
):
    previous_sqlite = (
        os.environ.get(
            "DATALENS_SQLITE_PATH"
        )
    )


    previous_model_store = (
        os.environ.get(
            "DATALENS_ML_MODEL_ARTIFACT_STORE_PATH"
        )
    )


    with tempfile.TemporaryDirectory(
        prefix=
            "datalens-classical-ml-"
    ) as temporary_directory:

        root = Path(
            temporary_directory
        )


        os.environ[
            "DATALENS_SQLITE_PATH"
        ] = str(
            root
            /
            "datalens.sqlite3"
        )


        os.environ[
            "DATALENS_ML_MODEL_ARTIFACT_STORE_PATH"
        ] = str(
            root
            /
            "ml"
            /
            "model_artifacts.json"
        )


        try:
            yield root

        finally:
            if previous_sqlite is None:
                os.environ.pop(
                    "DATALENS_SQLITE_PATH",
                    None,
                )

            else:
                os.environ[
                    "DATALENS_SQLITE_PATH"
                ] = (
                    previous_sqlite
                )


            if (
                previous_model_store
                is None
            ):
                os.environ.pop(
                    "DATALENS_ML_MODEL_ARTIFACT_STORE_PATH",
                    None,
                )

            else:
                os.environ[
                    "DATALENS_ML_MODEL_ARTIFACT_STORE_PATH"
                ] = (
                    previous_model_store
                )


# ============================================================
# PREPARATION AUTHORITY
# ============================================================


def seed_preparation_authority(
    *,
    workflow_id: str,
    dataset_id: str,
) -> None:

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
            VALUES (
                ?,
                ?,
                ?,
                ?,
                ?
            )
            """,
            (
                workflow_id,
                0,
                "{}",
                "2026-08-29T00:00:00+00:00",
                "2026-08-29T00:00:00+00:00",
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
            VALUES (
                ?,
                ?,
                ?,
                ?,
                ?,
                ?,
                ?,
                ?,
                ?,
                ?,
                ?
            )
            """,
            (
                "test-preparation-root",
                workflow_id,
                dataset_id,
                "validated.csv",
                "source",
                100,
                3,
                "[]",
                "[]",
                "[]",
                "data/validated.json.gz",
            ),
        )


# ============================================================
# HANDOFF PATCH
# ============================================================


@contextmanager
def patched_handoff(
    *,
    dataframe: pd.DataFrame,
    workflow_id: str,
    dataset_id: str,
    authorized_dataset_ids: (
        tuple[
            str,
            ...
        ]
        |
        None
    ) = None,
):

    original = (
        executor_module
        .load_validated_analysis_input
    )


    effective_ids = (
        authorized_dataset_ids

        if authorized_dataset_ids
        is not None

        else (
            dataset_id,
        )
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
                    0,

                dataset_ids=
                    effective_ids,

                dataset_records=(
                    {
                        "dataset_id":
                            dataset_id,

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
        yield

    finally:
        executor_module.load_validated_analysis_input = (
            original
        )


# ============================================================
# DATA
# ============================================================


def regression_dataframe(
) -> pd.DataFrame:

    x1 = np.arange(
        1,
        31,
        dtype=float,
    )


    x2 = np.array(
        [
            float(
                (
                    index
                    *
                    3
                )
                %
                11
            )

            for index
            in range(
                30
            )
        ],
        dtype=float,
    )


    revenue = (
        4.0
        *
        x1
        -
        1.5
        *
        x2
        +
        7.0
    )


    return (
        pd.DataFrame(
            {
                "age":
                    x1,

                "tenure":
                    x2,

                "revenue":
                    revenue,
            }
        )
    )


def classification_dataframe(
) -> pd.DataFrame:

    negative = np.arange(
        -20,
        0,
        dtype=float,
    )


    positive = np.arange(
        1,
        21,
        dtype=float,
    )


    x1 = np.concatenate(
        [
            negative,
            positive,
        ]
    )


    x2 = np.array(
        [
            float(
                (
                    index
                    %
                    5
                )
                -
                2
            )

            for index
            in range(
                40
            )
        ],
        dtype=float,
    )


    labels = (
        [
            "no"
        ]
        *
        20
        +
        [
            "yes"
        ]
        *
        20
    )


    return (
        pd.DataFrame(
            {
                "signal":
                    x1,

                "aux":
                    x2,

                "churned":
                    labels,
            }
        )
    )


# ============================================================
# CONTRACTS
# ============================================================


def regression_contract(
    *,
    estimator_key: str = (
        "linear_regression"
    ),
    dataset_id: str = (
        "dataset:validated"
    ),
) -> MLTrainingContract:

    return (
        MLTrainingContract(
            workflow_id=
                "prep:ml-executor",

            dataset_id=
                dataset_id,

            problem_type=
                "regression",

            target_column=
                "revenue",

            feature_columns=[
                "age",
                "tenure",
            ],

            estimator_key=
                estimator_key,

            split=
                MLSplitContract(
                    test_size=
                        0.20,

                    random_seed=
                        42,

                    shuffle=
                        True,

                    stratify=
                        False,
                ),
        )
    )


def classification_contract(
    *,
    estimator_key: str = (
        "logistic_regression"
    ),
) -> MLTrainingContract:

    return (
        MLTrainingContract(
            workflow_id=
                "prep:ml-executor",

            dataset_id=
                "dataset:validated",

            problem_type=
                "classification",

            target_column=
                "churned",

            feature_columns=[
                "signal",
                "aux",
            ],

            estimator_key=
                estimator_key,

            split=
                MLSplitContract(
                    test_size=
                        0.25,

                    random_seed=
                        42,

                    shuffle=
                        True,

                    stratify=
                        True,
                ),
        )
    )


# ============================================================
# REGRESSION GOLDEN PATH
# ============================================================


def test_regression_training_and_artifact(
) -> None:

    with isolated_environment():

        seed_preparation_authority(
            workflow_id=
                "prep:ml-executor",

            dataset_id=
                "dataset:validated",
        )


        dataframe = (
            regression_dataframe()
        )


        with patched_handoff(
            dataframe=
                dataframe,

            workflow_id=
                "prep:ml-executor",

            dataset_id=
                "dataset:validated",
        ):
            result = (
                execute_classical_ml(
                    training_contract=
                        regression_contract()
                )
            )


        assert (
            result.problem_type
            ==
            "regression"
        )


        assert (
            result.estimator_key
            ==
            "linear_regression"
        )


        assert (
            result.train_rows
            ==
            24
        )


        assert (
            result.test_rows
            ==
            6
        )


        assert (
            result.metrics[
                "mae"
            ]
            <
            1e-9
        )


        assert (
            result.metrics[
                "rmse"
            ]
            <
            1e-9
        )


        assert (
            abs(
                result.metrics[
                    "r2"
                ]
                -
                1.0
            )
            <
            1e-12
        )


        assert (
            result.model_artifact
            .model_id
            .startswith(
                "model:"
            )
        )


        assert (
            result.model_artifact
            .metrics
            ==
            result.metrics
        )


        stored = (
            list_ml_model_artifacts(
                workflow_id=
                    "prep:ml-executor"
            )
        )


        assert (
            len(
                stored
            )
            ==
            1
        )


        assert (
            stored[
                0
            ].model_id
            ==
            result
            .model_artifact
            .model_id
        )


        model_bytes = (
            load_ml_model_artifact_binary(
                model_id=
                    result
                    .model_artifact
                    .model_id,

                workflow_id=
                    "prep:ml-executor",
            )
        )


        assert (
            len(
                model_bytes
            )
            >
            0
        )


        # Trusted reload:
        #
        # These bytes were just generated, persisted and
        # SHA-verified by DataLens itself. No user-provided
        # joblib file is involved.
        restored_pipeline = (
            joblib.load(
                io.BytesIO(
                    model_bytes
                )
            )
        )


        restored_prediction = (
            restored_pipeline.predict(
                pd.DataFrame(
                    {
                        "age":
                            [
                                40.0
                            ],

                        "tenure":
                            [
                                2.0
                            ],
                    }
                )
            )
        )


        assert (
            len(
                restored_prediction
            )
            ==
            1
        )


# ============================================================
# DETERMINISTIC REGRESSION
# ============================================================


def test_regression_metrics_are_deterministic(
) -> None:

    with isolated_environment():

        seed_preparation_authority(
            workflow_id=
                "prep:ml-executor",

            dataset_id=
                "dataset:validated",
        )


        dataframe = (
            regression_dataframe()
        )


        with patched_handoff(
            dataframe=
                dataframe,

            workflow_id=
                "prep:ml-executor",

            dataset_id=
                "dataset:validated",
        ):

            first = (
                execute_classical_ml(
                    training_contract=
                        regression_contract()
                )
            )


            second = (
                execute_classical_ml(
                    training_contract=
                        regression_contract()
                )
            )


        assert (
            first.metrics
            ==
            second.metrics
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
            first
            .model_artifact
            .model_id
            !=
            second
            .model_artifact
            .model_id
        )


# ============================================================
# CLASSIFICATION GOLDEN PATH
# ============================================================


def test_classification_training_and_artifact(
) -> None:

    with isolated_environment():

        seed_preparation_authority(
            workflow_id=
                "prep:ml-executor",

            dataset_id=
                "dataset:validated",
        )


        dataframe = (
            classification_dataframe()
        )


        with patched_handoff(
            dataframe=
                dataframe,

            workflow_id=
                "prep:ml-executor",

            dataset_id=
                "dataset:validated",
        ):

            result = (
                execute_classical_ml(
                    training_contract=
                        classification_contract()
                )
            )


        assert (
            result.problem_type
            ==
            "classification"
        )


        assert (
            result.estimator_key
            ==
            "logistic_regression"
        )


        assert (
            result.train_rows
            ==
            30
        )


        assert (
            result.test_rows
            ==
            10
        )


        assert (
            result.metrics[
                "accuracy"
            ]
            >=
            0.90
        )


        assert (
            result.metrics[
                "f1_macro"
            ]
            >=
            0.90
        )


        assert (
            result.model_artifact
            .training_contract
            .split
            .stratify
            is True
        )


# ============================================================
# SERVER-OWNED DATASET SCOPE
# ============================================================


def test_dataset_outside_handoff_scope_is_blocked(
) -> None:

    with isolated_environment():

        seed_preparation_authority(
            workflow_id=
                "prep:ml-executor",

            dataset_id=
                "dataset:validated",
        )


        with patched_handoff(
            dataframe=
                regression_dataframe(),

            workflow_id=
                "prep:ml-executor",

            dataset_id=
                "dataset:other",

            authorized_dataset_ids=(
                "dataset:other",
            ),
        ):

            try:
                execute_classical_ml(
                    training_contract=
                        regression_contract()
                )

            except ClassicalMLInputError:
                pass

            else:
                raise AssertionError(
                    (
                        "Dataset outside validated "
                        "handoff scope should have "
                        "been rejected."
                    )
                )


        assert (
            list_ml_model_artifacts(
                workflow_id=
                    "prep:ml-executor"
            )
            ==
            []
        )


# ============================================================
# MISSING FEATURE
# ============================================================


def test_missing_feature_is_blocked(
) -> None:

    with isolated_environment():

        seed_preparation_authority(
            workflow_id=
                "prep:ml-executor",

            dataset_id=
                "dataset:validated",
        )


        dataframe = (
            regression_dataframe()
            .drop(
                columns=[
                    "tenure",
                ]
            )
        )


        with patched_handoff(
            dataframe=
                dataframe,

            workflow_id=
                "prep:ml-executor",

            dataset_id=
                "dataset:validated",
        ):

            try:
                execute_classical_ml(
                    training_contract=
                        regression_contract()
                )

            except ClassicalMLInputError:
                pass

            else:
                raise AssertionError(
                    (
                        "Missing contract feature "
                        "should have been rejected."
                    )
                )


# ============================================================
# MISSING VALUES
# ============================================================


def test_missing_values_are_blocked(
) -> None:

    with isolated_environment():

        seed_preparation_authority(
            workflow_id=
                "prep:ml-executor",

            dataset_id=
                "dataset:validated",
        )


        dataframe = (
            regression_dataframe()
        )


        dataframe.loc[
            0,
            "age",
        ] = np.nan


        with patched_handoff(
            dataframe=
                dataframe,

            workflow_id=
                "prep:ml-executor",

            dataset_id=
                "dataset:validated",
        ):

            try:
                execute_classical_ml(
                    training_contract=
                        regression_contract()
                )

            except ClassicalMLInputError:
                pass

            else:
                raise AssertionError(
                    (
                        "Missing feature values "
                        "should have been rejected."
                    )
                )


# ============================================================
# NON-NUMERIC FEATURE
# ============================================================


def test_non_numeric_feature_is_blocked(
) -> None:

    with isolated_environment():

        seed_preparation_authority(
            workflow_id=
                "prep:ml-executor",

            dataset_id=
                "dataset:validated",
        )


        dataframe = (
            regression_dataframe()
        )


        dataframe[
            "tenure"
        ] = [
            (
                "segment-a"
                if (
                    index
                    %
                    2
                    ==
                    0
                )
                else
                "segment-b"
            )

            for index
            in range(
                len(
                    dataframe
                )
            )
        ]


        with patched_handoff(
            dataframe=
                dataframe,

            workflow_id=
                "prep:ml-executor",

            dataset_id=
                "dataset:validated",
        ):

            try:
                execute_classical_ml(
                    training_contract=
                        regression_contract()
                )

            except ClassicalMLInputError:
                pass

            else:
                raise AssertionError(
                    (
                        "Non-numeric feature "
                        "should have been rejected."
                    )
                )


# ============================================================
# ESTIMATOR / PROBLEM TYPE CONTRACT
# ============================================================


def test_wrong_regression_estimator_is_blocked(
) -> None:

    with isolated_environment():

        seed_preparation_authority(
            workflow_id=
                "prep:ml-executor",

            dataset_id=
                "dataset:validated",
        )


        with patched_handoff(
            dataframe=
                regression_dataframe(),

            workflow_id=
                "prep:ml-executor",

            dataset_id=
                "dataset:validated",
        ):

            try:
                execute_classical_ml(
                    training_contract=
                        regression_contract(
                            estimator_key=
                                "logistic_regression"
                        )
                )

            except ClassicalMLEstimatorError:
                pass

            else:
                raise AssertionError(
                    (
                        "Estimator/problem mismatch "
                        "should have been rejected."
                    )
                )


# ============================================================
# HANDOFF FAILURE
# ============================================================


def test_handoff_failure_is_mapped_to_ml_input_error(
) -> None:

    with isolated_environment():

        original = (
            executor_module
            .load_validated_analysis_input
        )


        def fail_handoff(
            *,
            workflow_id: str,
        ):
            raise (
                AnalysisInputHandoffError(
                    (
                        "injected handoff failure "
                        f"{workflow_id}"
                    )
                )
            )


        executor_module.load_validated_analysis_input = (
            fail_handoff
        )


        try:
            try:
                execute_classical_ml(
                    training_contract=
                        regression_contract()
                )

            except ClassicalMLInputError:
                pass

            else:
                raise AssertionError(
                    (
                        "Preparation handoff failure "
                        "should have been mapped to "
                        "ClassicalMLInputError."
                    )
                )

        finally:
            executor_module.load_validated_analysis_input = (
                original
            )


# ============================================================
# RULE VERSION
# ============================================================


def test_executor_rule_version(
) -> None:

    assert (
        CLASSICAL_ML_EXECUTOR_RULE_VERSION
        ==
        "classical_ml_executor_v0.1"
    )


# ============================================================
# MAIN
# ============================================================


def main(
) -> None:

    print(
        (
            "=== DATALENS CLASSICAL ML "
            "EXECUTOR v0.1 ==="
        )
    )

    print()


    test_regression_training_and_artifact()

    print(
        (
            "Linear regression -> metrics -> "
            "server-owned Model Artifact: PASS"
        )
    )


    test_regression_metrics_are_deterministic()

    print(
        "Deterministic regression metrics: PASS"
    )


    test_classification_training_and_artifact()

    print(
        (
            "Logistic regression -> metrics -> "
            "server-owned Model Artifact: PASS"
        )
    )


    test_dataset_outside_handoff_scope_is_blocked()

    print(
        "Validated handoff dataset scope enforced: PASS"
    )


    test_missing_feature_is_blocked()

    print(
        "Missing contract feature is blocked: PASS"
    )


    test_missing_values_are_blocked()

    print(
        "Missing ML values are blocked: PASS"
    )


    test_non_numeric_feature_is_blocked()

    print(
        "Non-numeric v0.1 feature is blocked: PASS"
    )


    test_wrong_regression_estimator_is_blocked()

    print(
        "Estimator/problem mismatch is blocked: PASS"
    )


    test_handoff_failure_is_mapped_to_ml_input_error()

    print(
        "Preparation handoff failure is fail-closed: PASS"
    )


    test_executor_rule_version()

    print(
        "Classical ML Executor rule version: PASS"
    )


    print()

    print(
        "Classical ML Executor v0.1: PASS"
    )


if __name__ == "__main__":
    main()