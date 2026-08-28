from __future__ import annotations


import math
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


import numpy as np
import pandas as pd


from sklearn.model_selection import (
    train_test_split,
)


import app.ml.classical_executor as executor_module


from app.ml.classical_executor import (
    ClassicalMLInputError,
    execute_classical_ml,
)


from app.ml.contracts import (
    MLPreprocessingContract,
    MLSplitContract,
    MLTrainingContract,
)


from app.ml.model_loader import (
    load_trusted_ml_model,
)


from app.ml.preprocessing import (
    ML_PREPROCESSING_RUNTIME_RULE_VERSION,
)


from app.persistence.sqlite_database import (
    sqlite_connection,
)


# ============================================================
# FAKE HANDOFF READ MODEL
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
def isolated_environment():
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
        prefix="datalens-ml-preprocessing-"
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
            if (
                previous_sqlite
                is None
            ):
                os.environ.pop(
                    "DATALENS_SQLITE_PATH",
                    None,
                )

            else:
                os.environ[
                    "DATALENS_SQLITE_PATH"
                ] = previous_sqlite


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
                ] = previous_model_store


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
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                workflow_id,
                0,
                "{}",
                "2026-08-29T02:00:00+00:00",
                "2026-08-29T02:00:00+00:00",
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
                workflow_id,
                dataset_id,
                "validated.csv",
                "source",
                20,
                4,
                "[]",
                "[]",
                "[]",
                "data/validated.json.gz",
            ),
        )


# ============================================================
# HANDOFF
# ============================================================


@contextmanager
def patched_handoff(
    *,
    dataframe: pd.DataFrame,
    workflow_id: str,
    dataset_id: str,
):
    original = (
        executor_module
        .load_validated_analysis_input
    )


    def fake_load_validated_analysis_input(
        *,
        workflow_id: str,
    ):
        return (
            FakeAnalysisInputHandoff(
                workflow_id=workflow_id,
                session_revision=0,
                dataset_ids=(
                    dataset_id,
                ),
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


def mixed_dataframe(
) -> pd.DataFrame:
    row_index = np.arange(
        20,
        dtype=int,
    )


    age = np.arange(
        20,
        dtype=float,
    )


    # These four rows belong to the deterministic test split
    # for random_seed=42 and test_size=0.20.
    #
    # Their extreme values deliberately make the full-dataset
    # median differ from the training-only median.
    age[
        0
    ] = 1000.0

    age[
        1
    ] = 1100.0

    age[
        15
    ] = 1200.0

    age[
        17
    ] = 1300.0


    # Missing numeric value is in the training split.
    age[
        9
    ] = np.nan


    tenure = (
        (
            row_index
            %
            5
        )
        +
        1
    ).astype(
        float
    )


    test_indices = {
        0,
        1,
        15,
        17,
    }


    additional_train_b = {
        3,
        5,
        8,
        11,
        13,
        16,
        18,
    }


    b_indices = (
        test_indices
        |
        additional_train_b
    )


    segment = np.asarray(
        [
            (
                "B"
                if index in b_indices
                else "A"
            )

            for index
            in row_index
        ],
        dtype=object,
    )


    # Missing categorical value is in the training split.
    segment[
        2
    ] = np.nan


    underlying_segment_effect = np.asarray(
        [
            (
                25.0
                if index in b_indices
                else 0.0
            )

            for index
            in row_index
        ],
        dtype=float,
    )


    revenue = (
        100.0
        +
        (
            row_index.astype(
                float
            )
            *
            4.0
        )
        +
        (
            tenure
            *
            3.0
        )
        +
        underlying_segment_effect
    )


    return (
        pd.DataFrame(
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
    )


# ============================================================
# CONTRACT
# ============================================================


def mixed_contract(
) -> MLTrainingContract:
    return (
        MLTrainingContract(
            workflow_id=
                "prep:ml-preprocessing",

            dataset_id=
                "dataset:validated",

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

            preprocessing=(
                MLPreprocessingContract(
                    numeric_imputation=
                        "median",

                    categorical_imputation=
                        "most_frequent",

                    categorical_encoding=
                        "one_hot",

                    handle_unknown_categories=
                        "ignore",

                    scale_numeric=
                        True,
                )
            ),

            split=(
                MLSplitContract(
                    test_size=
                        0.20,

                    random_seed=
                        42,

                    shuffle=
                        True,

                    stratify=
                        False,
                )
            ),
        )
    )


def fail_closed_contract(
) -> MLTrainingContract:
    return (
        MLTrainingContract(
            workflow_id=
                "prep:ml-preprocessing",

            dataset_id=
                "dataset:validated",

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

            preprocessing=(
                MLPreprocessingContract(
                    numeric_imputation=
                        "error",

                    categorical_imputation=
                        "error",
                )
            ),
        )
    )


# ============================================================
# EXECUTION HELPER
# ============================================================


def execute_mixed_model():
    dataframe = (
        mixed_dataframe()
    )


    contract = (
        mixed_contract()
    )


    result = (
        execute_classical_ml(
            training_contract=
                contract
        )
    )


    return (
        dataframe,
        contract,
        result,
    )


# ============================================================
# MIXED TRAINING
# ============================================================


def test_mixed_features_and_imputation_train_successfully(
) -> None:
    with isolated_environment():
        seed_preparation_authority(
            workflow_id=
                "prep:ml-preprocessing",

            dataset_id=
                "dataset:validated",
        )


        dataframe = (
            mixed_dataframe()
        )


        with patched_handoff(
            dataframe=
                dataframe,

            workflow_id=
                "prep:ml-preprocessing",

            dataset_id=
                "dataset:validated",
        ):
            (
                _,
                contract,
                result,
            ) = (
                execute_mixed_model()
            )


        assert (
            result.problem_type
            ==
            "regression"
        )


        assert (
            result.train_rows
            ==
            16
        )


        assert (
            result.test_rows
            ==
            4
        )


        assert (
            result.model_artifact
            .training_contract
            ==
            contract
        )


        assert (
            result.model_artifact
            .training_contract
            .categorical_feature_columns
            ==
            [
                "segment",
            ]
        )


        for value in (
            result.metrics.values()
        ):
            assert (
                math.isfinite(
                    float(
                        value
                    )
                )
            )


# ============================================================
# TRAIN-ONLY LEARNED STATISTICS
# ============================================================


def test_imputation_statistics_are_learned_from_train_only(
) -> None:
    with isolated_environment():
        seed_preparation_authority(
            workflow_id=
                "prep:ml-preprocessing",

            dataset_id=
                "dataset:validated",
        )


        dataframe = (
            mixed_dataframe()
        )


        contract = (
            mixed_contract()
        )


        with patched_handoff(
            dataframe=
                dataframe,

            workflow_id=
                contract.workflow_id,

            dataset_id=
                contract.dataset_id,
        ):
            result = (
                execute_classical_ml(
                    training_contract=
                        contract
                )
            )


        loaded = (
            load_trusted_ml_model(
                workflow_id=
                    contract.workflow_id,

                model_id=
                    result
                    .model_artifact
                    .model_id,
            )
        )


        fitted_pipeline = (
            loaded.estimator
        )


        preprocessor = (
            fitted_pipeline
            .named_steps[
                "preprocessor"
            ]
        )


        numeric_transformer = (
            preprocessor
            .named_transformers_[
                "numeric"
            ]
        )


        categorical_transformer = (
            preprocessor
            .named_transformers_[
                "categorical"
            ]
        )


        x = (
            dataframe.loc[
                :,
                contract.feature_columns,
            ]
        )


        y = (
            dataframe[
                contract.target_column
            ]
        )


        (
            x_train,
            _,
            _,
            _,
        ) = (
            train_test_split(
                x,
                y,
                test_size=
                    contract
                    .split
                    .test_size,

                random_state=
                    contract
                    .split
                    .random_seed,

                shuffle=
                    contract
                    .split
                    .shuffle,
            )
        )


        expected_train_age_median = float(
            x_train[
                "age"
            ]
            .median()
        )


        full_dataset_age_median = float(
            dataframe[
                "age"
            ]
            .median()
        )


        assert (
            expected_train_age_median
            !=
            full_dataset_age_median
        )


        learned_numeric_statistics = (
            numeric_transformer
            .named_steps[
                "imputer"
            ]
            .statistics_
        )


        age_index = (
            contract
            .numeric_feature_columns
            .index(
                "age"
            )
        )


        learned_age_median = float(
            learned_numeric_statistics[
                age_index
            ]
        )


        assert (
            learned_age_median
            ==
            expected_train_age_median
        )


        assert (
            learned_age_median
            !=
            full_dataset_age_median
        )


        expected_train_segment_mode = str(
            x_train[
                "segment"
            ]
            .mode(
                dropna=True
            )
            .iloc[
                0
            ]
        )


        full_dataset_segment_mode = str(
            dataframe[
                "segment"
            ]
            .mode(
                dropna=True
            )
            .iloc[
                0
            ]
        )


        assert (
            expected_train_segment_mode
            !=
            full_dataset_segment_mode
        )


        learned_categorical_statistics = (
            categorical_transformer
            .named_steps[
                "imputer"
            ]
            .statistics_
        )


        learned_segment_mode = str(
            learned_categorical_statistics[
                0
            ]
        )


        assert (
            learned_segment_mode
            ==
            expected_train_segment_mode
        )


        assert (
            learned_segment_mode
            !=
            full_dataset_segment_mode
        )


# ============================================================
# PERSISTED PIPELINE / UNSEEN CATEGORY
# ============================================================


def test_persisted_pipeline_handles_unseen_category(
) -> None:
    with isolated_environment():
        seed_preparation_authority(
            workflow_id=
                "prep:ml-preprocessing",

            dataset_id=
                "dataset:validated",
        )


        dataframe = (
            mixed_dataframe()
        )


        contract = (
            mixed_contract()
        )


        with patched_handoff(
            dataframe=
                dataframe,

            workflow_id=
                contract.workflow_id,

            dataset_id=
                contract.dataset_id,
        ):
            result = (
                execute_classical_ml(
                    training_contract=
                        contract
                )
            )


        loaded = (
            load_trusted_ml_model(
                workflow_id=
                    contract.workflow_id,

                model_id=
                    result
                    .model_artifact
                    .model_id,
            )
        )


        prediction_input = (
            pd.DataFrame(
                {
                    "age": [
                        32.0,
                        np.nan,
                    ],

                    "tenure": [
                        4.0,
                        3.0,
                    ],

                    "segment": [
                        "NEVER_SEEN",
                        np.nan,
                    ],
                }
            )
        )


        predictions = (
            loaded.predict(
                prediction_input
            )
        )


        assert (
            len(
                predictions
            )
            ==
            2
        )


        assert (
            np.isfinite(
                np.asarray(
                    predictions,
                    dtype=np.float64,
                )
            )
            .all()
        )


# ============================================================
# ERROR POLICY REMAINS FAIL-CLOSED
# ============================================================


def test_error_imputation_policy_blocks_missing_values(
) -> None:
    with isolated_environment():
        seed_preparation_authority(
            workflow_id=
                "prep:ml-preprocessing",

            dataset_id=
                "dataset:validated",
        )


        dataframe = (
            mixed_dataframe()
        )


        contract = (
            fail_closed_contract()
        )


        with patched_handoff(
            dataframe=
                dataframe,

            workflow_id=
                contract.workflow_id,

            dataset_id=
                contract.dataset_id,
        ):
            try:
                execute_classical_ml(
                    training_contract=
                        contract
                )

            except ClassicalMLInputError:
                return


        raise AssertionError(
            (
                "Missing values should remain "
                "fail-closed when the preprocessing "
                "contract requests error policies."
            )
        )


# ============================================================
# RULE VERSION
# ============================================================


def test_preprocessing_runtime_rule_version(
) -> None:
    assert (
        ML_PREPROCESSING_RUNTIME_RULE_VERSION
        ==
        "ml_preprocessing_runtime_v0.1"
    )


# ============================================================
# MAIN
# ============================================================


def main(
) -> None:
    print(
        (
            "=== DATALENS ML PREPROCESSING "
            "RUNTIME v0.1 ==="
        )
    )

    print()


    test_mixed_features_and_imputation_train_successfully()

    print(
        (
            "Mixed numeric/categorical training "
            "with imputation: PASS"
        )
    )


    test_imputation_statistics_are_learned_from_train_only()

    print(
        (
            "Numeric median + categorical mode "
            "are learned from train only: PASS"
        )
    )


    test_persisted_pipeline_handles_unseen_category()

    print(
        (
            "Persisted ColumnTransformer handles "
            "missing + unseen category: PASS"
        )
    )


    test_error_imputation_policy_blocks_missing_values()

    print(
        (
            "Error imputation policies remain "
            "fail-closed: PASS"
        )
    )


    test_preprocessing_runtime_rule_version()

    print(
        "ML Preprocessing Runtime rule version: PASS"
    )


    print()

    print(
        "ML Preprocessing Runtime v0.1: PASS"
    )


if __name__ == "__main__":
    main()