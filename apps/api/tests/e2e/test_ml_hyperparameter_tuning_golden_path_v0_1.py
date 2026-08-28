from __future__ import annotations


import math


from fastapi.testclient import (
    TestClient,
)


# ============================================================
# REAL ISOLATED PRODUCT ENVIRONMENT
# ============================================================
#
# The Model Comparison Golden Path ultimately reuses the
# isolated Preparation / preprocessing E2E environment.
#
# This gives this test:
# - isolated SQLite;
# - isolated Preparation Artifact Store;
# - isolated Model Artifact Store;
# - real FastAPI Preparation flow;
# - real mixed-type validated dataset.
# ============================================================


from tests.e2e.test_ml_model_comparison_golden_path_v0_1 import (
    WORKFLOW_ROOT_DATASET_ID,
    app,
    build_candidate,
    create_preparation_session,
    reset_product_state,
    run_real_cleaning_plan,
    run_real_quality,
    select_analysis_output,
    validate_preparation,
    verify_preparation_persistence,
    verify_real_handoff,
)


from app.ml.experiment_provenance import (
    ml_training_contract_sha256,
)


from app.ml.hyperparameter_tuning import (
    ML_HYPERPARAMETER_TUNING_RULE_VERSION,
    MLHyperparameterSearchContract,
    expected_hyperparameter_metric_names,
    server_owned_hyperparameter_candidates,
)


from app.ml.hyperparameter_tuning_executor import (
    ML_HYPERPARAMETER_TUNING_EXECUTOR_RULE_VERSION,
    execute_ml_hyperparameter_tuning,
)


from app.persistence.sqlite_database import (
    sqlite_connection,
)


from app.preparation.analysis_input_handoff import (
    load_validated_analysis_input,
)


from app.preparation.analysis_readiness_gate import (
    require_analysis_readiness,
)


# ============================================================
# VERSION
# ============================================================


ML_HYPERPARAMETER_TUNING_GOLDEN_PATH_RULE_VERSION = (
    "ml_hyperparameter_tuning_golden_path_v0.1"
)


# ============================================================
# MODEL ARTIFACT COUNT
# ============================================================


def ml_model_artifact_count(
    *,
    workflow_id: str,
) -> int:
    """
    Hyperparameter Tuning v0.1 must never persist candidate
    models.

    Experiment Provenance v0.1 is attached to persisted
    Model Artifacts, so zero Model Artifacts also proves that
    tuning did not create persisted experiment provenance.
    """

    with sqlite_connection(
        write=False
    ) as connection:

        row = (
            connection.execute(
                """
                SELECT COUNT(*) AS count
                FROM ml_model_artifacts

                WHERE
                    workflow_id = ?
                """,
                (
                    workflow_id,
                ),
            )
            .fetchone()
        )


    assert (
        row
        is not None
    )


    return int(
        row[
            "count"
        ]
    )


# ============================================================
# REAL TRAINING CONTRACT
# ============================================================


def build_real_tuning_training_contract(
    *,
    workflow_id: str,
):
    """
    Build one real mixed-type Ridge Regression contract.

    Dataset:
    - age: numeric
    - tenure: numeric
    - segment: categorical
    - revenue: regression target

    Preprocessing and outer holdout are inherited from the
    existing production Golden Path helper.
    """

    return (
        build_candidate(
            workflow_id=
                workflow_id,

            estimator_key=
                "ridge_regression",
        )
    )


# ============================================================
# REAL HANDOFF
# ============================================================


def verify_real_tuning_handoff(
    *,
    workflow_id: str,
) -> int:

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


    records = [
        record

        for record
        in handoff.dataset_records

        if (
            isinstance(
                record,
                dict,
            )
            and
            record.get(
                "dataset_id"
            )
            ==
            WORKFLOW_ROOT_DATASET_ID
        )
    ]


    assert (
        len(
            records
        )
        ==
        1
    )


    dataframe = (
        records[
            0
        ]
        .get(
            "dataframe"
        )
    )


    assert (
        dataframe
        is not None
    )


    assert (
        list(
            dataframe.columns
        )
        ==
        [
            "age",
            "tenure",
            "segment",
            "revenue",
        ]
    )


    row_count = int(
        len(
            dataframe
        )
    )


    assert (
        row_count
        ==
        30
    )


    assert (
        dataframe[
            "segment"
        ]
        .nunique(
            dropna=False
        )
        >=
        2
    )


    print(
        (
            "[PASS] Hyperparameter Tuning resolved "
            "the real 30-row mixed-type "
            "Preparation artifact"
        )
    )


    return row_count


# ============================================================
# REAL TUNING
# ============================================================


def run_real_hyperparameter_tuning(
    *,
    workflow_id: str,
    training_contract,
    search_contract,
):
    """
    Execute the real tuning engine and prove that:

    - Preparation revision does not change;
    - no Model Artifact exists before tuning;
    - no Model Artifact exists after tuning;
    - the result remains pinned to the same Preparation
      revision.
    """

    readiness_before = (
        require_analysis_readiness(
            workflow_id=
                workflow_id
        )
    )


    artifacts_before = (
        ml_model_artifact_count(
            workflow_id=
                workflow_id
        )
    )


    assert (
        artifacts_before
        ==
        0
    )


    base_contract_before = (
        training_contract.model_dump(
            mode="json"
        )
    )


    base_sha_before = (
        ml_training_contract_sha256(
            training_contract
        )
    )


    result = (
        execute_ml_hyperparameter_tuning(
            training_contract=
                training_contract,

            search_contract=
                search_contract,
        )
    )


    readiness_after = (
        require_analysis_readiness(
            workflow_id=
                workflow_id
        )
    )


    artifacts_after = (
        ml_model_artifact_count(
            workflow_id=
                workflow_id
        )
    )


    base_contract_after = (
        training_contract.model_dump(
            mode="json"
        )
    )


    base_sha_after = (
        ml_training_contract_sha256(
            training_contract
        )
    )


    assert (
        readiness_before
        .session_revision
        ==
        readiness_after
        .session_revision
    )


    assert (
        result
        .preparation_session_revision
        ==
        readiness_before
        .session_revision
    )


    assert (
        artifacts_after
        ==
        artifacts_before
        ==
        0
    )


    assert (
        base_contract_before
        ==
        base_contract_after
    )


    assert (
        base_sha_before
        ==
        base_sha_after
    )


    assert (
        result
        .base_training_contract_sha256
        ==
        base_sha_before
    )


    print(
        (
            "[PASS] complete Hyperparameter Tuning "
            "remained pinned to one Preparation revision"
        )
    )


    print(
        (
            "[PASS] base Training Contract and SHA-256 "
            "remained unchanged"
        )
    )


    print(
        (
            "[PASS] Hyperparameter Tuning remained "
            "evaluation-only with zero Model Artifacts"
        )
    )


    print(
        (
            "[PASS] zero Model Artifacts means zero "
            "persisted Experiment Provenance records"
        )
    )


    return result


# ============================================================
# OUTER HOLDOUT
# ============================================================


def verify_real_outer_holdout(
    *,
    result,
    row_count: int,
) -> None:
    """
    The real production split contract is 80/20.

    30 rows therefore produce:
    - 24 OUTER training rows;
    - 6 untouched holdout rows.

    The executor's unit test already proves by index recording
    that every fit and predict operation is a subset of the
    OUTER train indexes.

    This Golden Path proves the same boundary using the real
    Preparation artifact and production split implementation.
    """

    assert (
        row_count
        ==
        30
    )


    assert (
        result.outer_train_rows
        ==
        24
    )


    assert (
        result.holdout_test_rows
        ==
        6
    )


    assert (
        result.outer_train_rows
        +
        result.holdout_test_rows
        ==
        row_count
    )


    print(
        (
            "[PASS] real deterministic OUTER holdout "
            "produced 24 tuning rows + 6 untouched test rows"
        )
    )


# ============================================================
# COMPLETE SERVER-OWNED GRID
# ============================================================


def verify_real_server_owned_grid(
    *,
    result,
) -> None:

    expected_grid = (
        server_owned_hyperparameter_candidates(
            estimator_key=
                "ridge_regression"
        )
    )


    assert (
        len(
            expected_grid
        )
        ==
        3
    )


    assert (
        result.candidate_count
        ==
        3
    )


    assert (
        len(
            result.candidate_results
        )
        ==
        3
    )


    assert (
        {
            candidate.candidate_index

            for candidate
            in result.candidate_results
        }
        ==
        {
            1,
            2,
            3,
        }
    )


    result_by_index = {
        candidate.candidate_index:
            candidate

        for candidate
        in result.candidate_results
    }


    for (
        candidate_index,
        expected_hyperparameters,
    ) in enumerate(
        expected_grid,
        start=1,
    ):

        actual = (
            result_by_index[
                candidate_index
            ]
        )


        assert (
            actual
            .hyperparameters
            .model_dump(
                mode="json"
            )
            ==
            expected_hyperparameters
            .model_dump(
                mode="json"
            )
        )


    assert (
        [
            float(
                candidate.alpha
            )

            for candidate
            in expected_grid
        ]
        ==
        [
            0.1,
            1.0,
            10.0,
        ]
    )


    print(
        (
            "[PASS] real tuning exhaustively evaluated "
            "the complete Ridge alpha grid: 0.1 / 1.0 / 10.0"
        )
    )


# ============================================================
# REAL INNER CV METRICS
# ============================================================


def verify_real_candidate_metrics(
    *,
    result,
) -> None:

    expected_metrics = set(
        expected_hyperparameter_metric_names(
            problem_type=
                "regression"
        )
    )


    assert (
        expected_metrics
        ==
        {
            "mae",
            "rmse",
            "r2",
            "median_absolute_error",
            "explained_variance",
        }
    )


    candidate_contract_hashes = set()


    for candidate in (
        result.candidate_results
    ):

        assert (
            set(
                candidate.metric_summary
            )
            ==
            expected_metrics
        )


        for summary in (
            candidate
            .metric_summary
            .values()
        ):

            assert math.isfinite(
                float(
                    summary.mean
                )
            )


            assert math.isfinite(
                float(
                    summary.std
                )
            )


            assert (
                summary.std
                >=
                0.0
            )


        candidate_contract_hashes.add(
            candidate
            .training_contract_sha256
        )


    assert (
        len(
            candidate_contract_hashes
        )
        ==
        3
    )


    print(
        (
            "[PASS] every real candidate exposes "
            "five finite richer regression metrics"
        )
    )


    print(
        (
            "[PASS] three hyperparameter candidates "
            "produced three distinct Training Contract SHA-256s"
        )
    )


# ============================================================
# RANKING
# ============================================================


def verify_real_ranking(
    *,
    result,
) -> None:

    assert (
        result.primary_metric
        ==
        "rmse"
    )


    assert (
        result.metric_direction
        ==
        "minimize"
    )


    assert (
        result.validation_strategy
        ==
        "k_fold"
    )


    assert (
        [
            candidate.rank

            for candidate
            in result.candidate_results
        ]
        ==
        [
            1,
            2,
            3,
        ]
    )


    assert (
        result.best_candidate_index
        ==
        result.candidate_results[
            0
        ].candidate_index
    )


    for (
        left,
        right,
    ) in zip(
        result.candidate_results,
        result.candidate_results[
            1:
        ],
    ):

        left_summary = (
            left.metric_summary[
                "rmse"
            ]
        )


        right_summary = (
            right.metric_summary[
                "rmse"
            ]
        )


        left_key = (
            left_summary.mean,
            left_summary.std,
            left.candidate_index,
        )


        right_key = (
            right_summary.mean,
            right_summary.std,
            right.candidate_index,
        )


        assert (
            left_key
            <=
            right_key
        )


    print(
        (
            "[PASS] real candidate ranking minimizes "
            "RMSE with std + candidate-index tie-breakers"
        )
    )


# ============================================================
# RESULT AUTHORITY
# ============================================================


def verify_real_result_authority(
    *,
    result,
    workflow_id: str,
    training_contract,
    search_contract,
) -> None:

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
        "ridge_regression"
    )


    assert (
        result.search_strategy
        ==
        "server_owned_grid"
    )


    assert (
        result.validation_strategy
        ==
        "k_fold"
    )


    assert (
        result.primary_metric
        ==
        "rmse"
    )


    assert (
        result.metric_direction
        ==
        "minimize"
    )


    assert (
        result.folds
        ==
        search_contract.folds
        ==
        5
    )


    assert (
        result.shuffle
        is True
    )


    assert (
        result.random_seed
        ==
        search_contract.random_seed
        ==
        73
    )


    assert (
        result.base_training_contract_sha256
        ==
        ml_training_contract_sha256(
            training_contract
        )
    )


    print(
        (
            "[PASS] result preserves workflow / dataset / "
            "Preparation revision / Training Contract authority"
        )
    )


# ============================================================
# DETERMINISM + NO PERSISTENCE
# ============================================================


def verify_real_repeat_is_deterministic(
    *,
    workflow_id: str,
    training_contract,
    search_contract,
    first_result,
) -> None:

    artifacts_before = (
        ml_model_artifact_count(
            workflow_id=
                workflow_id
        )
    )


    assert (
        artifacts_before
        ==
        0
    )


    second_result = (
        execute_ml_hyperparameter_tuning(
            training_contract=
                training_contract,

            search_contract=
                search_contract,
        )
    )


    artifacts_after = (
        ml_model_artifact_count(
            workflow_id=
                workflow_id
        )
    )


    assert (
        first_result.model_dump(
            mode="json"
        )
        ==
        second_result.model_dump(
            mode="json"
        )
    )


    assert (
        artifacts_after
        ==
        artifacts_before
        ==
        0
    )


    print(
        (
            "[PASS] repeated real tuning with the same "
            "seed is exactly deterministic"
        )
    )


    print(
        (
            "[PASS] deterministic repeat still persisted "
            "zero Model Artifacts / Experiments"
        )
    )


# ============================================================
# PRIVACY
# ============================================================


def verify_real_result_is_privacy_minimal(
    *,
    result,
) -> None:

    payload = (
        result.model_dump(
            mode="json"
        )
    )


    forbidden = {
        "raw_rows",
        "predictions",
        "fold_predictions",
        "holdout_predictions",
        "holdout_metrics",
        "x_train",
        "x_test",
        "y_train",
        "y_test",
        "model_id",
        "experiment_id",
        "experiment_provenance",
        "model_artifact",
        "model_bytes",
        "model_path",
        "estimator",
    }


    assert (
        forbidden
        .isdisjoint(
            payload
        )
    )


    for candidate in (
        result.candidate_results
    ):

        candidate_payload = (
            candidate.model_dump(
                mode="json"
            )
        )


        assert (
            forbidden
            .isdisjoint(
                candidate_payload
            )
        )


    print(
        (
            "[PASS] tuning result remains privacy-minimal "
            "with no holdout score, prediction or model identity"
        )
    )


# ============================================================
# RULE VERSIONS
# ============================================================


def verify_rule_versions(
) -> None:

    assert (
        ML_HYPERPARAMETER_TUNING_RULE_VERSION
        ==
        "ml_hyperparameter_tuning_v0.1"
    )


    assert (
        ML_HYPERPARAMETER_TUNING_EXECUTOR_RULE_VERSION
        ==
        "ml_hyperparameter_tuning_executor_v0.1"
    )


    assert (
        ML_HYPERPARAMETER_TUNING_GOLDEN_PATH_RULE_VERSION
        ==
        "ml_hyperparameter_tuning_golden_path_v0.1"
    )


    print(
        "[PASS] Hyperparameter Tuning rule versions"
    )


# ============================================================
# GOLDEN PATH
# ============================================================


def test_ml_hyperparameter_tuning_golden_path_v0_1(
) -> None:

    reset_product_state()


    with TestClient(
        app
    ) as client:

        # ----------------------------------------------------
        # REAL PREPARATION
        # ----------------------------------------------------

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


        row_count = (
            verify_real_tuning_handoff(
                workflow_id=
                    workflow_id
            )
        )


        # ----------------------------------------------------
        # REAL TRAINING CONTRACT
        # ----------------------------------------------------

        training_contract = (
            build_real_tuning_training_contract(
                workflow_id=
                    workflow_id
            )
        )


        search_contract = (
            MLHyperparameterSearchContract(
                folds=
                    5,

                shuffle=
                    True,

                random_seed=
                    73,
            )
        )


        # ----------------------------------------------------
        # REAL TUNING ? INNER TRAIN ONLY
        # ----------------------------------------------------

        result = (
            run_real_hyperparameter_tuning(
                workflow_id=
                    workflow_id,

                training_contract=
                    training_contract,

                search_contract=
                    search_contract,
            )
        )


        # ----------------------------------------------------
        # ASSERT AUTHORITY + LEAKAGE BOUNDARY
        # ----------------------------------------------------

        verify_real_outer_holdout(
            result=
                result,

            row_count=
                row_count,
        )


        verify_real_server_owned_grid(
            result=
                result
        )


        verify_real_candidate_metrics(
            result=
                result
        )


        verify_real_ranking(
            result=
                result
        )


        verify_real_result_authority(
            result=
                result,

            workflow_id=
                workflow_id,

            training_contract=
                training_contract,

            search_contract=
                search_contract,
        )


        verify_real_result_is_privacy_minimal(
            result=
                result
        )


        # ----------------------------------------------------
        # EXACT DETERMINISTIC REPEAT
        # ----------------------------------------------------

        verify_real_repeat_is_deterministic(
            workflow_id=
                workflow_id,

            training_contract=
                training_contract,

            search_contract=
                search_contract,

            first_result=
                result,
        )


        verify_rule_versions()


# ============================================================
# MAIN
# ============================================================


def main(
) -> None:

    print()

    print(
        "="
        *
        78
    )


    print(
        "DATALENS ML HYPERPARAMETER TUNING GOLDEN PATH E2E v0.1"
    )


    print(
        "="
        *
        78
    )


    print(
        "Preparation : real validated mixed-type CSV"
    )


    print(
        "Outer split : real deterministic 24/6 holdout"
    )


    print(
        "Tuning      : OUTER train only"
    )


    print(
        "Inner CV    : deterministic 5-fold KFold"
    )


    print(
        "Estimator   : RidgeRegression"
    )


    print(
        "Grid        : alpha 0.1 / 1.0 / 10.0"
    )


    print(
        "Preprocess  : refitted inside every inner fold"
    )


    print(
        "Metrics     : five richer regression metrics"
    )


    print(
        "Ranking     : RMSE mean / std / candidate index"
    )


    print(
        "Holdout     : untouched and unscored"
    )


    print(
        "Persistence : zero Model Artifact / Experiment"
    )


    print()


    test_ml_hyperparameter_tuning_golden_path_v0_1()


    print()

    print(
        "="
        *
        78
    )


    print(
        (
            "PASS - Preparation -> OUTER Holdout -> "
            "Train-only Hyperparameter Tuning -> "
            "Inner 5-Fold CV -> 3 Ridge Candidates -> "
            "5 Metrics -> Deterministic Ranking -> "
            "Untouched Holdout -> No Model Artifact / Experiment"
        )
    )


    print(
        "="
        *
        78
    )


if __name__ == "__main__":
    main()
