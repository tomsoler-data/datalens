from __future__ import annotations


import math
import os
import tempfile


from pathlib import (
    Path,
)


# ============================================================
# ISOLATED ML PRODUCT ENVIRONMENT
# ============================================================
#
# Environment variables MUST be configured before importing
# DataLens persistence/application modules.
#
# This E2E test must never touch the developer's normal stores.
# ============================================================


_TEMP_DIRECTORY = tempfile.TemporaryDirectory(
    prefix="datalens-ml-golden-path-v0-1-"
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


from app.main import (
    app,
)


from app.ml.classical_executor import (
    ClassicalMLInputError,
    execute_classical_ml,
)


from app.ml.contracts import (
    MLTrainingContract,
)


from app.ml.model_artifact_store import (
    get_ml_model_artifact,
    load_ml_model_artifact_binary,
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


ML_GOLDEN_PATH_RULE_VERSION = (
    "ml_golden_path_v0.1"
)


WORKFLOW_ROOT_DATASET_ID = (
    "dataset:0001"
)


# ============================================================
# GOLDEN DATASET
# ============================================================
#
# Clean supervised regression dataset:
#
# revenue =
#     2.5 * customer_age
#     + 8.0 * tenure_months
#     + 15.0
#
# - numeric features only;
# - numeric target;
# - no missing values;
# - no duplicates;
# - enough rows for deterministic holdout;
# - no client-created model identity.
# ============================================================


_ROWS = [
    (21, 1),
    (22, 3),
    (23, 2),
    (24, 5),
    (25, 4),
    (26, 7),
    (27, 3),
    (28, 8),
    (29, 6),
    (30, 10),
    (31, 5),
    (32, 11),
    (33, 7),
    (34, 12),
    (35, 9),
    (36, 14),
    (37, 8),
    (38, 15),
    (39, 11),
    (40, 16),
    (41, 10),
    (42, 18),
    (43, 13),
    (44, 19),
    (45, 15),
    (46, 20),
    (47, 17),
    (48, 22),
    (49, 18),
    (50, 24),
]


def expected_revenue(
    *,
    customer_age: float,
    tenure_months: float,
) -> float:
    return (
        2.5
        *
        customer_age
        +
        8.0
        *
        tenure_months
        +
        15.0
    )


CSV_CONTENT = (
    "customer_age,tenure_months,revenue\n"
    +
    "\n".join(
        (
            f"{customer_age},"
            f"{tenure_months},"
            f"{expected_revenue(customer_age=customer_age, tenure_months=tenure_months)}"
        )
        for (
            customer_age,
            tenure_months,
        )
        in _ROWS
    )
    +
    "\n"
)


# ============================================================
# RESET
# ============================================================


def reset_ml_product_state() -> None:
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
# REAL CSV -> QUALITY
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
                        "ml_training.csv",
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
        "[PASS] real CSV crossed IMPORT / UNDERSTAND / QUALITY"
    )


# ============================================================
# CLEAN
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
                        "ml_training.csv",
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

    print(
        "[PASS] CLEAN deterministically skipped for clean ML dataset"
    )


# ============================================================
# ML MUST FAIL BEFORE VALIDATION
# ============================================================


def require_ml_blocked_before_validation(
    *,
    workflow_id: str,
) -> None:
    contract = (
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
                "customer_age",
                "tenure_months",
            ],

            estimator_key=
                "linear_regression",
        )
    )

    try:
        execute_classical_ml(
            training_contract=
                contract
        )

    except ClassicalMLInputError:
        print(
            "[PASS] ML gate blocks unvalidated Preparation"
        )

        return

    raise AssertionError(
        (
            "Classical ML execution should fail "
            "before Final Preparation Validation."
        )
    )


# ============================================================
# SELECT SERVER-OWNED ANALYSIS / ML OUTPUT
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
        "[PASS] real server-owned ML input artifact selected"
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
    assert (
        _DATABASE_PATH.exists()
    )

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
        .analysis_output_dataset_ids
        ==
        [
            WORKFLOW_ROOT_DATASET_ID
        ]
    )

    assert (
        restored_session
        .validate_stage
        .completed
        is True
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
            "customer_age",
            "tenure_months",
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

    print(
        "[PASS] fresh Preparation stores restore validated ML dataset"
    )


# ============================================================
# REAL SERVER-OWNED HANDOFF
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
        len(
            dataframe
        )
        ==
        len(
            _ROWS
        )
    )

    print(
        "[PASS] real validated Preparation -> ML handoff restored"
    )


# ============================================================
# REAL CLASSICAL ML TRAINING
# ============================================================


def train_real_model(
    *,
    workflow_id: str,
):
    contract = (
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
                "customer_age",
                "tenure_months",
            ],

            estimator_key=
                "linear_regression",
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
        +
        result.test_rows
        ==
        len(
            _ROWS
        )
    )

    assert (
        set(
            result.metrics
        )
        ==
        {
            "mae",
            "rmse",
            "r2",
        }
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

    assert (
        result.model_artifact
        .model_id
        .startswith(
            "model:"
        )
    )

    assert (
        result.model_artifact
        .workflow_id
        ==
        workflow_id
    )

    assert (
        result.model_artifact
        .dataset_id
        ==
        WORKFLOW_ROOT_DATASET_ID
    )

    print(
        (
            "[PASS] real deterministic Classical ML training "
            "produced regression metrics"
        )
    )

    return result


# ============================================================
# SERVER-OWNED MODEL ARTIFACT
# ============================================================


def verify_model_artifact(
    *,
    workflow_id: str,
    execution_result,
) -> None:
    model_id = (
        execution_result
        .model_artifact
        .model_id
    )

    restored = (
        get_ml_model_artifact(
            model_id=
                model_id,

            workflow_id=
                workflow_id,
        )
    )

    assert (
        restored
        ==
        execution_result
        .model_artifact
    )

    verified_binary = (
        load_ml_model_artifact_binary(
            model_id=
                model_id,

            workflow_id=
                workflow_id,
        )
    )

    assert (
        isinstance(
            verified_binary,
            bytes,
        )
    )

    assert (
        len(
            verified_binary
        )
        ==
        restored.model_file_bytes
    )

    assert (
        len(
            verified_binary
        )
        >
        0
    )

    print(
        (
            "[PASS] server-owned Model Artifact metadata "
            "and SHA-verified binary restored"
        )
    )


# ============================================================
# TRUSTED RELOAD + PREDICTION
# ============================================================


def verify_reload_prediction(
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

    prediction_features = (
        pd.DataFrame(
            {
                "customer_age": [
                    27.0,
                    34.0,
                    43.0,
                ],

                "tenure_months": [
                    4.0,
                    13.0,
                    16.0,
                ],
            }
        )
    )

    predictions = (
        loaded.predict(
            prediction_features
        )
    )

    expected = (
        np.asarray(
            [
                expected_revenue(
                    customer_age=
                        27.0,

                    tenure_months=
                        4.0,
                ),

                expected_revenue(
                    customer_age=
                        34.0,

                    tenure_months=
                        13.0,
                ),

                expected_revenue(
                    customer_age=
                        43.0,

                    tenure_months=
                        16.0,
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

    assert (
        loaded.artifact.model_id
        ==
        execution_result
        .model_artifact
        .model_id
    )

    print(
        (
            "[PASS] trusted Model Artifact reload "
            "produces correct predictions"
        )
    )


# ============================================================
# RULE VERSION
# ============================================================


def verify_rule_version() -> None:
    assert (
        ML_GOLDEN_PATH_RULE_VERSION
        ==
        "ml_golden_path_v0.1"
    )

    print(
        "[PASS] ML Golden Path rule version"
    )


# ============================================================
# GOLDEN PATH
# ============================================================


def test_ml_golden_path_v0_1() -> None:
    reset_ml_product_state()

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

        require_ml_blocked_before_validation(
            workflow_id=
                workflow_id
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

        execution_result = (
            train_real_model(
                workflow_id=
                    workflow_id
            )
        )

        verify_model_artifact(
            workflow_id=
                workflow_id,

            execution_result=
                execution_result,
        )

        verify_reload_prediction(
            workflow_id=
                workflow_id,

            execution_result=
                execution_result,
        )

        verify_rule_version()


# ============================================================
# MAIN
# ============================================================


def main() -> None:
    print()
    print(
        "=" * 78
    )

    print(
        "DATALENS ML GOLDEN PATH E2E v0.1"
    )

    print(
        "=" * 78
    )

    print(
        "Preparation : real FastAPI + SQLite + Artifact Store"
    )

    print(
        "ML handoff   : real validated server-owned handoff"
    )

    print(
        "Training     : real scikit-learn LinearRegression"
    )

    print(
        "Evaluation   : real deterministic holdout metrics"
    )

    print(
        "Persistence  : real Model Artifact SQLite + filesystem"
    )

    print(
        "Reload       : trusted SHA-verified joblib loader"
    )

    print()

    test_ml_golden_path_v0_1()

    print()
    print(
        "=" * 78
    )

    print(
        (
            "PASS - CSV → Preparation → VALIDATE → "
            "ML Contract → Train → Evaluate → Model Artifact "
            "→ Trusted Reload → Prediction"
        )
    )

    print(
        "=" * 78
    )


if __name__ == "__main__":
    main()