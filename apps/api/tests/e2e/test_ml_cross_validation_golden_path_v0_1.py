from __future__ import annotations


import math
import statistics


from fastapi.testclient import (
    TestClient,
)


from app.ml.cross_validation import (
    ML_CROSS_VALIDATION_RULE_VERSION,
    MLCrossValidationContract,
)


from app.ml.cross_validation_executor import (
    execute_ml_cross_validation,
)


from app.ml.experiment_provenance import (
    ml_training_contract_sha256,
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


# ============================================================
# VERSION
# ============================================================


ML_CROSS_VALIDATION_GOLDEN_PATH_RULE_VERSION = (
    "ml_cross_validation_golden_path_v0.1"
)


# ============================================================
# EXPECTED METRICS
# ============================================================


REGRESSION_METRIC_NAMES = {
    "mae",
    "rmse",
    "r2",
    "median_absolute_error",
    "explained_variance",
}


# ============================================================
# MODEL ARTIFACT COUNT
# ============================================================


def ml_model_artifact_count(
    *,
    workflow_id: str,
) -> int:

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


def build_real_training_contract(
    *,
    workflow_id: str,
):

    return (
        build_candidate(
            workflow_id=
                workflow_id,

            estimator_key=
                "linear_regression",
        )
    )


# ============================================================
# REAL HANDOFF ROW COUNT
# ============================================================


def real_handoff_row_count(
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


    print(
        (
            "[PASS] Cross-Validation resolved the "
            "real 30-row server-owned Preparation artifact"
        )
    )


    return row_count


# ============================================================
# REAL CROSS-VALIDATION
# ============================================================


def run_real_cross_validation(
    *,
    workflow_id: str,
    training_contract,
    cross_validation_contract,
):

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


    result = (
        execute_ml_cross_validation(
            training_contract=
                training_contract,

            cross_validation_contract=
                cross_validation_contract,
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
    )


    print(
        (
            "[PASS] complete Cross-Validation remained "
            "pinned to one Preparation revision"
        )
    )


    print(
        (
            "[PASS] Cross-Validation remained "
            "evaluation-only and persisted no Model Artifact"
        )
    )


    return result


# ============================================================
# CROSS-VALIDATION RESULT
# ============================================================


def verify_real_cross_validation_result(
    *,
    result,
    training_contract,
    cross_validation_contract,
    row_count: int,
) -> None:

    assert (
        result.workflow_id
        ==
        training_contract.workflow_id
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
        result.strategy
        ==
        "k_fold"
    )


    assert (
        result.folds
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
        cross_validation_contract
        .random_seed
    )


    assert (
        result.training_contract_sha256
        ==
        ml_training_contract_sha256(
            training_contract
        )
    )


    assert (
        len(
            result.fold_results
        )
        ==
        5
    )


    assert (
        [
            fold.fold_index

            for fold
            in result.fold_results
        ]
        ==
        [
            1,
            2,
            3,
            4,
            5,
        ]
    )


    total_validation_rows = 0


    for fold in (
        result.fold_results
    ):

        assert (
            fold.train_rows
            +
            fold.validation_rows
            ==
            row_count
        )


        assert (
            fold.train_rows
            ==
            24
        )


        assert (
            fold.validation_rows
            ==
            6
        )


        total_validation_rows += (
            fold.validation_rows
        )


        assert (
            set(
                fold.metrics
            )
            ==
            REGRESSION_METRIC_NAMES
        )


        for value in (
            fold.metrics.values()
        ):
            assert math.isfinite(
                float(
                    value
                )
            )


    assert (
        total_validation_rows
        ==
        row_count
    )


    assert (
        set(
            result.metric_summary
        )
        ==
        REGRESSION_METRIC_NAMES
    )


    print(
        (
            "[PASS] real KFold produced five disjoint "
            "24/6 train-validation folds"
        )
    )


    print(
        (
            "[PASS] every real fold exposes exactly "
            "five finite richer regression metrics"
        )
    )


# ============================================================
# AGGREGATION
# ============================================================


def verify_metric_aggregation(
    *,
    result,
) -> None:

    for metric_name in (
        REGRESSION_METRIC_NAMES
    ):

        fold_values = [
            float(
                fold.metrics[
                    metric_name
                ]
            )

            for fold
            in result.fold_results
        ]


        expected_mean = float(
            statistics.fmean(
                fold_values
            )
        )


        expected_std = float(
            statistics.pstdev(
                fold_values
            )
        )


        actual = (
            result.metric_summary[
                metric_name
            ]
        )


        assert math.isclose(
            actual.mean,
            expected_mean,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )


        assert math.isclose(
            actual.std,
            expected_std,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )


        assert (
            actual.std
            >=
            0.0
        )


    print(
        (
            "[PASS] metric summaries equal the real "
            "fold population mean + standard deviation"
        )
    )


# ============================================================
# PRIVACY / EVALUATION-ONLY SURFACE
# ============================================================


def verify_privacy_minimal_result(
    *,
    result,
) -> None:

    payload = (
        result.model_dump(
            mode="json"
        )
    )


    assert (
        "predictions"
        not in
        payload
    )


    assert (
        "model_artifact"
        not in
        payload
    )


    assert (
        "model_id"
        not in
        payload
    )


    assert (
        "experiment_id"
        not in
        payload
    )


    assert (
        "raw_rows"
        not in
        payload
    )


    print(
        (
            "[PASS] Cross-Validation result remains "
            "privacy-minimal and evaluation-only"
        )
    )


# ============================================================
# DETERMINISM
# ============================================================


def verify_real_cross_validation_is_deterministic(
    *,
    training_contract,
    cross_validation_contract,
    first_result,
) -> None:

    second_result = (
        execute_ml_cross_validation(
            training_contract=
                training_contract,

            cross_validation_contract=
                cross_validation_contract,
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


    print(
        (
            "[PASS] repeated real Cross-Validation with "
            "the same seed is exactly deterministic"
        )
    )


# ============================================================
# RULE VERSIONS
# ============================================================


def verify_rule_versions(
) -> None:

    assert (
        ML_CROSS_VALIDATION_RULE_VERSION
        ==
        "ml_cross_validation_v0.1"
    )


    assert (
        ML_CROSS_VALIDATION_GOLDEN_PATH_RULE_VERSION
        ==
        "ml_cross_validation_golden_path_v0.1"
    )


    print(
        "[PASS] Cross-Validation rule versions"
    )


# ============================================================
# GOLDEN PATH
# ============================================================


def test_ml_cross_validation_golden_path_v0_1(
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
            real_handoff_row_count(
                workflow_id=
                    workflow_id
            )
        )


        # ----------------------------------------------------
        # REAL ML TRAINING AUTHORITY
        # ----------------------------------------------------

        training_contract = (
            build_real_training_contract(
                workflow_id=
                    workflow_id
            )
        )


        training_sha_before = (
            ml_training_contract_sha256(
                training_contract
            )
        )


        cross_validation_contract = (
            MLCrossValidationContract(
                folds=
                    5,

                shuffle=
                    True,

                random_seed=
                    73,
            )
        )


        # ----------------------------------------------------
        # REAL CROSS-VALIDATION
        # ----------------------------------------------------

        result = (
            run_real_cross_validation(
                workflow_id=
                    workflow_id,

                training_contract=
                    training_contract,

                cross_validation_contract=
                    cross_validation_contract,
            )
        )


        verify_real_cross_validation_result(
            result=
                result,

            training_contract=
                training_contract,

            cross_validation_contract=
                cross_validation_contract,

            row_count=
                row_count,
        )


        verify_metric_aggregation(
            result=
                result
        )


        verify_privacy_minimal_result(
            result=
                result
        )


        # ----------------------------------------------------
        # TRAINING CONTRACT PROVENANCE IS UNCHANGED
        # ----------------------------------------------------

        training_sha_after = (
            ml_training_contract_sha256(
                training_contract
            )
        )


        assert (
            training_sha_before
            ==
            training_sha_after
        )


        assert (
            result.training_contract_sha256
            ==
            training_sha_before
        )


        print(
            (
                "[PASS] separate CV configuration did "
                "not mutate Training Contract provenance"
            )
        )


        # ----------------------------------------------------
        # REAL DETERMINISTIC REPEAT
        # ----------------------------------------------------

        verify_real_cross_validation_is_deterministic(
            training_contract=
                training_contract,

            cross_validation_contract=
                cross_validation_contract,

            first_result=
                result,
        )


        assert (
            ml_model_artifact_count(
                workflow_id=
                    workflow_id
            )
            ==
            0
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
        "DATALENS ML CROSS-VALIDATION GOLDEN PATH E2E v0.1"
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
        "Input       : real server-owned Analysis handoff"
    )


    print(
        "Estimator   : LinearRegression"
    )


    print(
        "Strategy    : deterministic 5-fold KFold"
    )


    print(
        "Preprocess  : refitted inside every fold"
    )


    print(
        "Metrics     : five richer metrics per fold"
    )


    print(
        "Summary     : mean + population std"
    )


    print(
        "Persistence : evaluation-only, no Model Artifact"
    )


    print()


    test_ml_cross_validation_golden_path_v0_1()


    print()

    print(
        "="
        *
        78
    )


    print(
        (
            "PASS - Preparation -> Validated Handoff -> "
            "5-Fold Cross-Validation -> 5 Richer Metrics -> "
            "Mean/Std -> Deterministic Repeat -> "
            "No Model Artifact Persistence"
        )
    )


    print(
        "="
        *
        78
    )


if __name__ == "__main__":
    main()
