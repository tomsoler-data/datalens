from __future__ import annotations


import json


from app.ml.model_artifact_index import (
    ml_model_artifact_store_scope,
)


from app.ml.model_artifact_store import (
    resolve_ml_model_artifact_store_path,
)


from app.ml.performance_evaluation import (
    MLPerformanceEvaluationRecord,
)


from app.ml.performance_evaluation_store import (
    ML_PERFORMANCE_EVALUATION_STORE_RULE_VERSION,
    MLPerformanceEvaluationAlreadyExistsError,
    MLPerformanceEvaluationAuthorityError,
    MLPerformanceEvaluationNotFoundError,
    MLPerformanceEvaluationWorkflowMismatchError,
    get_ml_performance_evaluation,
    list_ml_performance_evaluations_for_model,
    list_ml_performance_evaluations_for_workflow,
    register_ml_performance_evaluation,
)


from app.ml.performance_evaluator import (
    evaluate_ml_performance,
)


from app.persistence.sqlite_database import (
    SQLITE_SCHEMA_VERSION,
    sqlite_connection,
    sqlite_schema_version,
)


from app.preparation.preparation_session import (
    record_validation_stage_signal,
)


from tests.ml.test_ml_monitoring_service_real_handoff_v0_1 import (
    build_ready_preparation_workflow,
    isolated_real_handoff_environment,
)


from tests.ml.test_ml_performance_evaluator_v0_1 import (
    classification_model,
    observed_classification_frame,
)


# ============================================================
# HELPERS
# ============================================================


def expect_error(
    error_type,
    factory,
) -> None:

    try:
        factory()

    except error_type:
        return


    raise AssertionError(
        (
            "Expected "
            f"{error_type.__name__}."
        )
    )


def persisted_candidate(
):

    (
        session,
        dataset_id,
        _,
    ) = (
        build_ready_preparation_workflow()
    )


    (
        trusted_model,
        _,
    ) = (
        classification_model(
            workflow_id=
                session.workflow_id,

            dataset_id=
                dataset_id,

            revision=
                session.revision,

            predictions=[
                0,
                1,
                0,
                1,
            ],
        )
    )


    evaluation = (
        evaluate_ml_performance(
            observed_dataframe=
                observed_classification_frame(),

            observed_dataset_id=
                "dataset:performance-observed",

            observed_preparation_session_revision=
                session.revision,

            trusted_model=
                trusted_model,
        )
    )


    return (
        session,
        dataset_id,
        trusted_model,
        evaluation,
    )


# ============================================================
# SCHEMA v13
# ============================================================


def test_sqlite_schema_v13(
) -> None:

    with isolated_real_handoff_environment():

        assert (
            SQLITE_SCHEMA_VERSION
            ==
            13
        )


        assert (
            sqlite_schema_version()
            ==
            13
        )


        with sqlite_connection(
            write=False
        ) as connection:

            columns = {
                str(
                    row[
                        "name"
                    ]
                )

                for row
                in connection.execute(
                    """
                    PRAGMA table_info(
                        ml_performance_evaluations
                    )
                    """
                ).fetchall()
            }


        assert (
            "performance_evaluation_id"
            in
            columns
        )


        assert (
            "observed_preparation_session_revision"
            in
            columns
        )


        assert (
            "payload_json"
            in
            columns
        )


# ============================================================
# ROUNDTRIP
# ============================================================


def test_register_get_roundtrip(
) -> None:

    with isolated_real_handoff_environment():

        (
            session,
            _,
            _,
            evaluation,
        ) = (
            persisted_candidate()
        )


        registered = (
            register_ml_performance_evaluation(
                evaluation=
                    evaluation
            )
        )


        assert (
            registered
            ==
            evaluation
        )


        restored = (
            get_ml_performance_evaluation(
                performance_evaluation_id=(
                    evaluation
                    .performance_evaluation_id
                ),

                workflow_id=
                    session.workflow_id,
            )
        )


        assert (
            restored
            ==
            evaluation
        )


# ============================================================
# MULTIPLE HISTORY
# ============================================================


def test_multiple_evaluations_per_model(
) -> None:

    with isolated_real_handoff_environment():

        (
            session,
            _,
            trusted_model,
            first,
        ) = (
            persisted_candidate()
        )


        second = (
            evaluate_ml_performance(
                observed_dataframe=
                    observed_classification_frame(),

                observed_dataset_id=
                    "dataset:performance-observed-2",

                observed_preparation_session_revision=
                    session.revision,

                trusted_model=
                    trusted_model,
            )
        )


        register_ml_performance_evaluation(
            evaluation=
                first
        )


        register_ml_performance_evaluation(
            evaluation=
                second
        )


        history = (
            list_ml_performance_evaluations_for_model(
                model_id=
                    trusted_model
                    .artifact
                    .model_id,

                workflow_id=
                    session.workflow_id,
            )
        )


        assert (
            len(
                history
            )
            ==
            2
        )


        assert {
            item.performance_evaluation_id

            for item
            in history
        } == {
            first.performance_evaluation_id,
            second.performance_evaluation_id,
        }


# ============================================================
# DUPLICATE IDENTITY
# ============================================================


def test_duplicate_identity_blocked(
) -> None:

    with isolated_real_handoff_environment():

        (
            _,
            _,
            _,
            evaluation,
        ) = (
            persisted_candidate()
        )


        register_ml_performance_evaluation(
            evaluation=
                evaluation
        )


        expect_error(
            MLPerformanceEvaluationAlreadyExistsError,

            lambda:
                register_ml_performance_evaluation(
                    evaluation=
                        evaluation
                ),
        )


# ============================================================
# MODEL AUTHORITY
# ============================================================


def test_missing_model_authority_blocked(
) -> None:

    with isolated_real_handoff_environment():

        (
            _,
            _,
            trusted_model,
            evaluation,
        ) = (
            persisted_candidate()
        )


        store_root = (
            ml_model_artifact_store_scope(
                resolve_ml_model_artifact_store_path()
            )
        )


        with sqlite_connection(
            write=True
        ) as connection:

            connection.execute(
                """
                DELETE FROM ml_model_artifacts

                WHERE
                    store_root = ?
                    AND
                    model_id = ?
                """,
                (
                    store_root,
                    trusted_model
                    .artifact
                    .model_id,
                ),
            )


        expect_error(
            MLPerformanceEvaluationAuthorityError,

            lambda:
                register_ml_performance_evaluation(
                    evaluation=
                        evaluation
                ),
        )


# ============================================================
# REFERENCE METRIC AUTHORITY
# ============================================================


def test_reference_metric_tampering_blocked(
) -> None:

    with isolated_real_handoff_environment():

        (
            _,
            _,
            _,
            evaluation,
        ) = (
            persisted_candidate()
        )


        payload = (
            evaluation.model_dump(
                mode="python"
            )
        )


        result = (
            payload[
                "metric_results"
            ][
                0
            ]
        )


        result[
            "reference_value"
        ] = 0.79


        result[
            "delta"
        ] = (
            result[
                "observed_value"
            ]
            -
            result[
                "reference_value"
            ]
        )


        result[
            "degradation_amount"
        ] = max(
            0.0,
            result[
                "reference_value"
            ]
            -
            result[
                "observed_value"
            ],
        )


        tampered = (
            MLPerformanceEvaluationRecord
            .model_validate(
                payload
            )
        )


        expect_error(
            MLPerformanceEvaluationAuthorityError,

            lambda:
                register_ml_performance_evaluation(
                    evaluation=
                        tampered
                ),
        )


# ============================================================
# TRAINING FINGERPRINT AUTHORITY
# ============================================================


def test_training_fingerprint_authority(
) -> None:

    with isolated_real_handoff_environment():

        (
            _,
            _,
            _,
            evaluation,
        ) = (
            persisted_candidate()
        )


        payload = (
            evaluation.model_dump(
                mode="python"
            )
        )


        payload[
            "training_contract_sha256"
        ] = (
            "e" * 64
        )


        tampered = (
            MLPerformanceEvaluationRecord
            .model_validate(
                payload
            )
        )


        expect_error(
            MLPerformanceEvaluationAuthorityError,

            lambda:
                register_ml_performance_evaluation(
                    evaluation=
                        tampered
                ),
        )


# ============================================================
# OBSERVED SNAPSHOT RACE
# ============================================================


def test_late_observed_revision_race_blocked(
) -> None:

    with isolated_real_handoff_environment():

        (
            session,
            dataset_id,
            _,
            evaluation,
        ) = (
            persisted_candidate()
        )


        changed = (
            record_validation_stage_signal(
                workflow_id=
                    session.workflow_id,

                completed=
                    True,

                passed=
                    False,

                dataset_ids=[
                    dataset_id
                ],

                evidence_refs=[
                    "test:performance-race"
                ],

                blocking_reasons=[
                    "test:revision-changed"
                ],

                expected_revision=
                    session.revision,
            )
        )


        assert (
            changed.revision
            >
            session.revision
        )


        expect_error(
            MLPerformanceEvaluationAuthorityError,

            lambda:
                register_ml_performance_evaluation(
                    evaluation=
                        evaluation
                ),
        )


        history = (
            list_ml_performance_evaluations_for_workflow(
                workflow_id=
                    session.workflow_id
            )
        )


        assert (
            history
            ==
            []
        )


# ============================================================
# WORKFLOW ISOLATION
# ============================================================


def test_read_workflow_mismatch_blocked(
) -> None:

    with isolated_real_handoff_environment():

        (
            _,
            _,
            _,
            evaluation,
        ) = (
            persisted_candidate()
        )


        register_ml_performance_evaluation(
            evaluation=
                evaluation
        )


        expect_error(
            MLPerformanceEvaluationWorkflowMismatchError,

            lambda:
                get_ml_performance_evaluation(
                    performance_evaluation_id=(
                        evaluation
                        .performance_evaluation_id
                    ),

                    workflow_id=
                        "prep:other-workflow",
                ),
        )


# ============================================================
# MISSING
# ============================================================


def test_missing_evaluation(
) -> None:

    with isolated_real_handoff_environment():

        expect_error(
            MLPerformanceEvaluationNotFoundError,

            lambda:
                get_ml_performance_evaluation(
                    performance_evaluation_id=(
                        "performance-evaluation:"
                        +
                        "f" * 32
                    )
                ),
        )


# ============================================================
# PRIVACY
# ============================================================


def test_aggregate_only_persistence(
) -> None:

    with isolated_real_handoff_environment():

        (
            _,
            _,
            _,
            evaluation,
        ) = (
            persisted_candidate()
        )


        register_ml_performance_evaluation(
            evaluation=
                evaluation
        )


        store_root = (
            ml_model_artifact_store_scope(
                resolve_ml_model_artifact_store_path()
            )
        )


        with sqlite_connection(
            write=False
        ) as connection:

            row = (
                connection.execute(
                    """
                    SELECT payload_json
                    FROM ml_performance_evaluations

                    WHERE
                        store_root = ?
                        AND
                        performance_evaluation_id = ?
                    """,
                    (
                        store_root,
                        evaluation
                        .performance_evaluation_id,
                    ),
                )
                .fetchone()
            )


        assert (
            row
            is not None
        )


        payload = json.loads(
            str(
                row[
                    "payload_json"
                ]
            )
        )


        serialized = str(
            payload
        )


        assert (
            payload[
                "privacy_scope"
            ]
            ==
            "aggregate_only"
        )


        assert (
            "raw-private-value"
            not in
            serialized
        )


        assert (
            "business_note"
            not in
            serialized
        )


        assert (
            "y_true"
            not in
            serialized
        )


        assert (
            "predictions"
            not in
            serialized
        )


# ============================================================
# MODEL DELETE CASCADE
# ============================================================


def test_model_deletion_cascades_history(
) -> None:

    with isolated_real_handoff_environment():

        (
            session,
            _,
            trusted_model,
            evaluation,
        ) = (
            persisted_candidate()
        )


        register_ml_performance_evaluation(
            evaluation=
                evaluation
        )


        assert (
            len(
                list_ml_performance_evaluations_for_model(
                    model_id=
                        trusted_model
                        .artifact
                        .model_id,

                    workflow_id=
                        session.workflow_id,
                )
            )
            ==
            1
        )


        store_root = (
            ml_model_artifact_store_scope(
                resolve_ml_model_artifact_store_path()
            )
        )


        with sqlite_connection(
            write=True
        ) as connection:

            connection.execute(
                """
                DELETE FROM ml_model_artifacts

                WHERE
                    store_root = ?
                    AND
                    model_id = ?
                """,
                (
                    store_root,
                    trusted_model
                    .artifact
                    .model_id,
                ),
            )


        assert (
            list_ml_performance_evaluations_for_model(
                model_id=
                    trusted_model
                    .artifact
                    .model_id,

                workflow_id=
                    session.workflow_id,
            )
            ==
            []
        )


# ============================================================
# VERSION
# ============================================================


def test_rule_version(
) -> None:

    assert (
        ML_PERFORMANCE_EVALUATION_STORE_RULE_VERSION
        ==
        "ml_performance_evaluation_store_v0.1"
    )


# ============================================================
# MAIN
# ============================================================


def main(
) -> None:

    print(
        (
            "=== DATALENS ML PERFORMANCE "
            "EVALUATION STORE v0.1 ==="
        )
    )

    print()


    tests = [
        (
            "SQLite schema v13",
            test_sqlite_schema_v13,
        ),
        (
            "Register / get roundtrip",
            test_register_get_roundtrip,
        ),
        (
            "Multiple evaluations per Model Artifact",
            test_multiple_evaluations_per_model,
        ),
        (
            "Duplicate performance identity blocked",
            test_duplicate_identity_blocked,
        ),
        (
            "Missing Model Artifact authority blocked",
            test_missing_model_authority_blocked,
        ),
        (
            "Reference metric authority binding",
            test_reference_metric_tampering_blocked,
        ),
        (
            "Training fingerprint authority binding",
            test_training_fingerprint_authority,
        ),
        (
            "Late observed Preparation revision race blocked",
            test_late_observed_revision_race_blocked,
        ),
        (
            "Read workflow mismatch blocked",
            test_read_workflow_mismatch_blocked,
        ),
        (
            "Explicit missing evaluation",
            test_missing_evaluation,
        ),
        (
            "Aggregate-only persistence",
            test_aggregate_only_persistence,
        ),
        (
            "Model deletion cascades Performance history",
            test_model_deletion_cascades_history,
        ),
        (
            "Performance Evaluation Store rule version",
            test_rule_version,
        ),
    ]


    for (
        label,
        callback,
    ) in tests:

        callback()

        print(
            f"[PASS] {label}"
        )


    print()

    print(
        (
            "PASS - ML Performance "
            "Evaluation Store v0.1"
        )
    )


if __name__ == "__main__":
    main()
