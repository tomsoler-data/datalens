from __future__ import annotations


import math
import os
import tempfile


from pathlib import (
    Path,
)


# ============================================================
# ISOLATED MIXED ML PRODUCT ENVIRONMENT
# ============================================================


_TEMP_DIRECTORY = tempfile.TemporaryDirectory(
    prefix="datalens-ml-preprocessing-e2e-v0-1-"
)


_ROOT = Path(
    _TEMP_DIRECTORY.name
)


_DATABASE_PATH = (
    _ROOT
    /
    "datalens.sqlite3"
)


_PREPARATION_ARTIFACT_ROOT = (
    _ROOT
    /
    "preparation_artifacts"
)


_LEGACY_PREPARATION_SESSION_PATH = (
    _ROOT
    /
    "preparation_sessions.json"
)


_ML_MODEL_ARTIFACT_PATH = (
    _ROOT
    /
    "ml"
    /
    "model_artifacts.json"
)


_RUNTIME_TRACE_PATH = (
    _ROOT
    /
    "runtime_requests.jsonl"
)


os.environ[
    "DATALENS_SQLITE_PATH"
] = str(
    _DATABASE_PATH
)


os.environ[
    "DATALENS_PREPARATION_ARTIFACT_STORE_PATH"
] = str(
    _PREPARATION_ARTIFACT_ROOT
)


os.environ[
    "DATALENS_PREPARATION_SESSION_STORE_PATH"
] = str(
    _LEGACY_PREPARATION_SESSION_PATH
)


os.environ[
    "DATALENS_ML_MODEL_ARTIFACT_STORE_PATH"
] = str(
    _ML_MODEL_ARTIFACT_PATH
)


os.environ[
    "DATALENS_RUNTIME_TRACE_ENABLED"
] = "0"


os.environ[
    "DATALENS_RUNTIME_TRACE_PATH"
] = str(
    _RUNTIME_TRACE_PATH
)


# ============================================================
# APPLICATION IMPORTS
# ============================================================


import numpy as np
import pandas as pd


from fastapi.testclient import (
    TestClient,
)


from sklearn.compose import (
    ColumnTransformer,
)


from sklearn.pipeline import (
    Pipeline,
)


from app.main import (
    app,
)


from app.ml.classical_executor import (
    execute_classical_ml,
)


from app.ml.contracts import (
    MLPreprocessingContract,
    MLTrainingContract,
)


from app.ml.model_loader import (
    load_trusted_ml_model,
)


from app.preparation.analysis_input_handoff import (
    load_validated_analysis_input,
)


from app.preparation.preparation_artifact_store import (
    PreparationArtifactStore,
    reset_preparation_artifact_store_for_tests,
)


from app.preparation.preparation_session import (
    PreparationSessionStore,
    reset_preparation_session_store_for_tests,
)


# ============================================================
# CONTRACT
# ============================================================


ML_PREPROCESSING_GOLDEN_PATH_RULE_VERSION = (
    "ml_preprocessing_golden_path_v0.1"
)


WORKFLOW_ROOT_DATASET_ID = (
    "dataset:0001"
)


# ============================================================
# GOLDEN MIXED DATASET
# ============================================================
#
# Deliberately clean at Preparation level:
#
# - no missing values;
# - no duplicate rows;
# - numeric age / tenure;
# - categorical segment;
# - numeric regression target;
#
# revenue =
#     50
#     + 2 * age
#     + 3 * tenure
#     + 20 when segment == premium
#
# This allows Preparation to remain responsible for data quality
# while the ML layer proves real mixed-type preprocessing.
# ============================================================


_ROWS: list[
    tuple[
        int,
        int,
        str,
    ]
] = [
    (21, 1, "standard"),
    (22, 2, "premium"),
    (23, 4, "standard"),
    (24, 3, "premium"),
    (25, 5, "standard"),
    (26, 2, "premium"),
    (27, 6, "standard"),
    (28, 4, "premium"),
    (29, 7, "standard"),
    (30, 3, "premium"),
    (31, 8, "standard"),
    (32, 5, "premium"),
    (33, 9, "standard"),
    (34, 6, "premium"),
    (35, 10, "standard"),
    (36, 7, "premium"),
    (37, 11, "standard"),
    (38, 8, "premium"),
    (39, 12, "standard"),
    (40, 9, "premium"),
    (41, 13, "standard"),
    (42, 10, "premium"),
    (43, 14, "standard"),
    (44, 11, "premium"),
    (45, 15, "standard"),
    (46, 12, "premium"),
    (47, 16, "standard"),
    (48, 13, "premium"),
    (49, 17, "standard"),
    (50, 14, "premium"),
]


def expected_revenue(
    *,
    age: float,
    tenure: float,
    segment: str,
) -> float:
    segment_effect = (
        20.0
        if segment
        ==
        "premium"
        else 0.0
    )


    return (
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
        segment_effect
    )


CSV_CONTENT = (
    "age,tenure,segment,revenue\n"
    +
    "\n".join(
        (
            f"{age},"
            f"{tenure},"
            f"{segment},"
            f"{expected_revenue(age=age, tenure=tenure, segment=segment)}"
        )

        for (
            age,
            tenure,
            segment,
        )
        in _ROWS
    )
    +
    "\n"
)


# ============================================================
# RESET
# ============================================================


def reset_product_state(
) -> None:
    reset_preparation_session_store_for_tests()

    reset_preparation_artifact_store_for_tests()


# ============================================================
# SESSION
# ============================================================


def create_preparation_session(
    client: TestClient,
) -> str:
    response = (
        client.post(
            "/preparation/sessions",
            json={
                "selected_analysis_dataset_ids": [
                    WORKFLOW_ROOT_DATASET_ID
                ]
            },
        )
    )


    assert (
        response.status_code
        ==
        201
    ), response.text


    body = (
        response.json()
    )


    workflow_id = str(
        body[
            "workflow_id"
        ]
    )


    assert (
        workflow_id.startswith(
            "prep:"
        )
    )


    assert (
        body[
            "snapshot"
        ][
            "ready_for_analysis"
        ]
        is False
    )


    print(
        (
            "[PASS] server-owned Preparation "
            f"session created: {workflow_id}"
        )
    )


    return workflow_id


# ============================================================
# REAL QUALITY
# ============================================================


def run_real_quality(
    client: TestClient,
    *,
    workflow_id: str,
) -> None:
    response = (
        client.post(
            "/preparation/quality",
            data={
                "workflow_id":
                    workflow_id,
            },
            files=[
                (
                    "dataset_files",
                    (
                        "mixed_ml.csv",
                        CSV_CONTENT,
                        "text/csv",
                    ),
                )
            ],
        )
    )


    assert (
        response.status_code
        ==
        200
    ), response.text


    print(
        (
            "[PASS] mixed CSV crossed real "
            "IMPORT / UNDERSTAND / QUALITY"
        )
    )


# ============================================================
# CLEAN SHOULD BE SKIPPED
# ============================================================


def run_real_cleaning_plan(
    client: TestClient,
    *,
    workflow_id: str,
) -> None:
    response = (
        client.post(
            "/preparation/cleaning-plan",
            data={
                "workflow_id":
                    workflow_id,
            },
            files=[
                (
                    "dataset_files",
                    (
                        "mixed_ml.csv",
                        CSV_CONTENT,
                        "text/csv",
                    ),
                )
            ],
        )
    )


    assert (
        response.status_code
        ==
        200
    ), response.text


    body = (
        response.json()
    )


    assert (
        body[
            "action_count"
        ]
        ==
        0
    ), body


    assert (
        body[
            "protected_issue_count"
        ]
        ==
        0
    ), body


    session = (
        client.get(
            (
                "/preparation/sessions/"
                f"{workflow_id}"
            )
        )
        .json()
    )


    assert (
        session[
            "snapshot"
        ][
            "next_stage"
        ]
        ==
        "validate"
    )


    print(
        (
            "[PASS] CLEAN skipped for clean "
            "mixed-type ML dataset"
        )
    )


# ============================================================
# SELECT OUTPUT
# ============================================================


def select_analysis_output(
    client: TestClient,
    *,
    workflow_id: str,
) -> None:
    candidates_response = (
        client.get(
            (
                "/preparation/sessions/"
                f"{workflow_id}"
                "/analysis-output-candidates"
            )
        )
    )


    assert (
        candidates_response.status_code
        ==
        200
    ), candidates_response.text


    candidates = (
        candidates_response
        .json()[
            "candidates"
        ]
    )


    matching = [
        candidate

        for candidate
        in candidates

        if (
            candidate[
                "dataset_id"
            ]
            ==
            WORKFLOW_ROOT_DATASET_ID
        )
    ]


    assert (
        len(
            matching
        )
        ==
        1
    ), matching


    response = (
        client.post(
            "/preparation/analysis-output",
            json={
                "workflow_id":
                    workflow_id,

                "dataset_ids": [
                    WORKFLOW_ROOT_DATASET_ID
                ],
            },
        )
    )


    assert (
        response.status_code
        ==
        200
    ), response.text


    body = (
        response.json()
    )


    assert (
        body[
            "analysis_output_dataset_ids"
        ]
        ==
        [
            WORKFLOW_ROOT_DATASET_ID
        ]
    )


    print(
        "[PASS] mixed server-owned ML input artifact selected"
    )


# ============================================================
# FINAL VALIDATION
# ============================================================


def validate_preparation(
    client: TestClient,
    *,
    workflow_id: str,
) -> None:
    response = (
        client.post(
            "/preparation/validate",
            json={
                "workflow_id":
                    workflow_id
            },
        )
    )


    assert (
        response.status_code
        ==
        200
    ), response.text


    body = (
        response.json()
    )


    assert (
        body[
            "snapshot"
        ][
            "ready_for_analysis"
        ]
        is True
    )


    assert (
        body[
            "snapshot"
        ][
            "validated_analysis_dataset_ids"
        ]
        ==
        [
            WORKFLOW_ROOT_DATASET_ID
        ]
    )


    assert (
        body[
            "snapshot"
        ][
            "blocking_reasons"
        ]
        ==
        []
    )


    print(
        "[PASS] Final Preparation Validation crossed"
    )


# ============================================================
# DURABLE PREPARATION
# ============================================================


def verify_preparation_persistence(
    *,
    workflow_id: str,
) -> None:
    fresh_session_store = (
        PreparationSessionStore()
    )


    restored_session = (
        fresh_session_store.get(
            workflow_id
        )
    )


    assert (
        restored_session
        .validate_stage
        .passed
        is True
    )


    fresh_artifact_store = (
        PreparationArtifactStore()
    )


    restored_dataframe = (
        fresh_artifact_store
        .get_dataframe(
            workflow_id=
                workflow_id,

            dataset_id=
                WORKFLOW_ROOT_DATASET_ID,
        )
    )


    assert (
        list(
            restored_dataframe.columns
        )
        ==
        [
            "age",
            "tenure",
            "segment",
            "revenue",
        ]
    )


    assert (
        len(
            restored_dataframe
        )
        ==
        len(
            _ROWS
        )
    )


    assert (
        restored_dataframe[
            "segment"
        ]
        .isna()
        .sum()
        ==
        0
    )


    print(
        (
            "[PASS] durable Preparation restores "
            "clean mixed-type dataset"
        )
    )


# ============================================================
# REAL HANDOFF
# ============================================================


def verify_real_handoff(
    *,
    workflow_id: str,
) -> None:
    handoff = (
        load_validated_analysis_input(
            workflow_id=
                workflow_id
        )
    )


    assert (
        handoff.workflow_id
        ==
        workflow_id
    )


    assert (
        tuple(
            handoff.dataset_ids
        )
        ==
        (
            WORKFLOW_ROOT_DATASET_ID,
        )
    )


    matching = [
        record

        for record
        in handoff.dataset_records

        if (
            record[
                "dataset_id"
            ]
            ==
            WORKFLOW_ROOT_DATASET_ID
        )
    ]


    assert (
        len(
            matching
        )
        ==
        1
    )


    dataframe = (
        matching[
            0
        ][
            "dataframe"
        ]
    )


    assert (
        isinstance(
            dataframe,
            pd.DataFrame,
        )
    )


    assert (
        dataframe[
            "segment"
        ]
        .nunique()
        ==
        2
    )


    print(
        (
            "[PASS] validated Preparation handoff "
            "preserves categorical feature"
        )
    )


# ============================================================
# CONTRACT
# ============================================================


def build_training_contract(
    *,
    workflow_id: str,
) -> MLTrainingContract:
    return (
        MLTrainingContract(
            workflow_id=
                workflow_id,

            dataset_id=
                WORKFLOW_ROOT_DATASET_ID,

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
        )
    )


# ============================================================
# REAL MIXED ML TRAINING
# ============================================================


def train_real_mixed_model(
    *,
    workflow_id: str,
):
    contract = (
        build_training_contract(
            workflow_id=
                workflow_id
        )
    )


    result = (
        execute_classical_ml(
            training_contract=
                contract
        )
    )


    assert (
        result.workflow_id
        ==
        workflow_id
    )


    assert (
        result.dataset_id
        ==
        WORKFLOW_ROOT_DATASET_ID
    )


    assert (
        result.train_rows
        +
        result.test_rows
        ==
        len(
            _ROWS
        )
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
            "segment"
        ]
    )


    for metric_value in (
        result.metrics.values()
    ):
        assert (
            math.isfinite(
                float(
                    metric_value
                )
            )
        )


    assert (
        result.metrics[
            "mae"
        ]
        <
        1e-8
    )


    assert (
        result.metrics[
            "rmse"
        ]
        <
        1e-8
    )


    assert (
        result.metrics[
            "r2"
        ]
        >
        0.999999
    )


    print(
        (
            "[PASS] real mixed-type LinearRegression "
            "training produced deterministic metrics"
        )
    )


    return (
        contract,
        result,
    )


# ============================================================
# TRUSTED RELOAD
# ============================================================


def verify_trusted_reload(
    *,
    workflow_id: str,
    contract: MLTrainingContract,
    execution_result,
) -> None:
    loaded = (
        load_trusted_ml_model(
            workflow_id=
                workflow_id,

            model_id=
                execution_result
                .model_artifact
                .model_id,
        )
    )


    assert (
        isinstance(
            loaded.estimator,
            Pipeline,
        )
    )


    preprocessor = (
        loaded
        .estimator
        .named_steps[
            "preprocessor"
        ]
    )


    assert (
        isinstance(
            preprocessor,
            ColumnTransformer,
        )
    )


    assert (
        "numeric"
        in
        preprocessor.named_transformers_
    )


    assert (
        "categorical"
        in
        preprocessor.named_transformers_
    )


    feature_names = [
        str(
            value
        )

        for value
        in preprocessor.get_feature_names_out()
    ]


    assert (
        any(
            (
                "segment"
                in
                feature_name
            )

            for feature_name
            in feature_names
        )
    )


    assert (
        loaded.artifact
        .training_contract
        ==
        contract
    )


    print(
        (
            "[PASS] persisted Model Artifact restores "
            "fitted ColumnTransformer"
        )
    )


# ============================================================
# KNOWN CATEGORY PREDICTION
# ============================================================


def verify_known_category_prediction(
    *,
    workflow_id: str,
    execution_result,
) -> None:
    loaded = (
        load_trusted_ml_model(
            workflow_id=
                workflow_id,

            model_id=
                execution_result
                .model_artifact
                .model_id,
        )
    )


    prediction_input = (
        pd.DataFrame(
            {
                "age": [
                    33.0,
                    44.0,
                ],

                "tenure": [
                    6.0,
                    11.0,
                ],

                "segment": [
                    "standard",
                    "premium",
                ],
            }
        )
    )


    predictions = (
        loaded.predict(
            prediction_input
        )
    )


    expected = (
        np.asarray(
            [
                expected_revenue(
                    age=
                        33.0,

                    tenure=
                        6.0,

                    segment=
                        "standard",
                ),

                expected_revenue(
                    age=
                        44.0,

                    tenure=
                        11.0,

                    segment=
                        "premium",
                ),
            ],
            dtype=np.float64,
        )
    )


    assert (
        np.allclose(
            predictions,
            expected,
            rtol=1e-9,
            atol=1e-7,
        )
    ), (
        predictions,
        expected,
    )


    print(
        (
            "[PASS] trusted reload predicts known "
            "categorical values correctly"
        )
    )


# ============================================================
# UNSEEN CATEGORY
# ============================================================


def verify_unseen_category_prediction(
    *,
    workflow_id: str,
    execution_result,
) -> None:
    loaded = (
        load_trusted_ml_model(
            workflow_id=
                workflow_id,

            model_id=
                execution_result
                .model_artifact
                .model_id,
        )
    )


    prediction_input = (
        pd.DataFrame(
            {
                "age": [
                    36.0,
                ],

                "tenure": [
                    8.0,
                ],

                "segment": [
                    "enterprise",
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
        1
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


    print(
        (
            "[PASS] trusted persisted pipeline handles "
            "unseen category without refit"
        )
    )


# ============================================================
# RULE VERSION
# ============================================================


def verify_rule_version(
) -> None:
    assert (
        ML_PREPROCESSING_GOLDEN_PATH_RULE_VERSION
        ==
        "ml_preprocessing_golden_path_v0.1"
    )


    print(
        "[PASS] ML Preprocessing Golden Path rule version"
    )


# ============================================================
# GOLDEN PATH
# ============================================================


def test_ml_preprocessing_golden_path_v0_1(
) -> None:
    reset_product_state()


    with TestClient(
        app
    ) as client:
        workflow_id = (
            create_preparation_session(
                client
            )
        )


        run_real_quality(
            client,
            workflow_id=
                workflow_id,
        )


        run_real_cleaning_plan(
            client,
            workflow_id=
                workflow_id,
        )


        select_analysis_output(
            client,
            workflow_id=
                workflow_id,
        )


        validate_preparation(
            client,
            workflow_id=
                workflow_id,
        )


        verify_preparation_persistence(
            workflow_id=
                workflow_id,
        )


        verify_real_handoff(
            workflow_id=
                workflow_id,
        )


        (
            contract,
            execution_result,
        ) = (
            train_real_mixed_model(
                workflow_id=
                    workflow_id
            )
        )


        verify_trusted_reload(
            workflow_id=
                workflow_id,

            contract=
                contract,

            execution_result=
                execution_result,
        )


        verify_known_category_prediction(
            workflow_id=
                workflow_id,

            execution_result=
                execution_result,
        )


        verify_unseen_category_prediction(
            workflow_id=
                workflow_id,

            execution_result=
                execution_result,
        )


        verify_rule_version()


# ============================================================
# MAIN
# ============================================================


def main(
) -> None:
    print()

    print(
        "=" * 78
    )


    print(
        "DATALENS ML PREPROCESSING GOLDEN PATH E2E v0.1"
    )


    print(
        "=" * 78
    )


    print(
        "Preparation : real clean mixed-type CSV"
    )


    print(
        "Features    : numeric + categorical"
    )


    print(
        "Pipeline    : real leakage-safe ColumnTransformer"
    )


    print(
        "Encoding    : persisted OneHotEncoder"
    )


    print(
        "Training    : real LinearRegression"
    )


    print(
        "Reload      : trusted SHA-verified Model Artifact"
    )


    print(
        "Prediction  : known + unseen category"
    )


    print()


    test_ml_preprocessing_golden_path_v0_1()


    print()

    print(
        "=" * 78
    )


    print(
        (
            "PASS - Mixed CSV → Preparation → VALIDATE → "
            "ColumnTransformer → Train → Model Artifact → "
            "Trusted Reload → Known/Unseen Category Prediction"
        )
    )


    print(
        "=" * 78
    )


if __name__ == "__main__":
    main()