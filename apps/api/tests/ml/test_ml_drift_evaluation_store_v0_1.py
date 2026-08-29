from __future__ import annotations


import json


from app.ml.drift_evaluation import (
    MLDriftEvaluationRecord,
)


from app.ml.drift_evaluation_store import (
    ML_DRIFT_EVALUATION_STORE_RULE_VERSION,
    MLDriftEvaluationAlreadyExistsError,
    MLDriftEvaluationAuthorityError,
    MLDriftEvaluationNotFoundError,
    MLDriftEvaluationWorkflowMismatchError,
    get_ml_drift_evaluation,
    list_ml_drift_evaluations_for_model,
    list_ml_drift_evaluations_for_workflow,
    register_ml_drift_evaluation,
)


from app.ml.drift_evaluator import (
    evaluate_ml_drift,
)


from app.ml.model_artifact_store import (
    resolve_ml_model_artifact_store_path,
)


from app.ml.monitoring_profile_store import (
    register_ml_monitoring_profile,
)


from app.persistence.sqlite_database import (
    SQLITE_SCHEMA_VERSION,
    sqlite_connection,
    sqlite_schema_version,
)


from tests.ml.test_ml_monitoring_profile_store_v0_1 import (
    isolated_environment,
    persisted_artifact_and_profile,
    training_frame,
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


def persisted_authority(
):
    (
        artifact,
        profile,
    ) = (
        persisted_artifact_and_profile()
    )


    register_ml_monitoring_profile(
        profile=profile
    )


    return (
        artifact,
        profile,
    )


def evaluation(
    *,
    artifact,
    profile,
    observed_dataset_id: str = "dataset:observed",
):

    return (
        evaluate_ml_drift(
            observed_features=(
                training_frame()
            ),

            observed_dataset_id=(
                observed_dataset_id
            ),

            monitoring_profile=profile,

            model_artifact=artifact,
        )
    )


# ============================================================
# SQLITE V11
# ============================================================


def test_sqlite_schema_v11(
) -> None:

    with isolated_environment():

        assert (
            SQLITE_SCHEMA_VERSION
            ==
            11
        )


        assert (
            sqlite_schema_version()
            ==
            SQLITE_SCHEMA_VERSION
        )


        with sqlite_connection(
            write=False
        ) as connection:

            migration = (
                connection.execute(
                    """
                    SELECT
                        version,
                        name

                    FROM schema_migrations

                    WHERE
                        version = 11
                    """
                )
                .fetchone()
            )


            assert (
                migration
                is not None
            )


            assert (
                int(
                    migration[
                        "version"
                    ]
                )
                ==
                11
            )


            assert (
                str(
                    migration[
                        "name"
                    ]
                )
                ==
                "ml_drift_evaluation_metadata"
            )


            table = (
                connection.execute(
                    """
                    SELECT name
                    FROM sqlite_master

                    WHERE
                        type = 'table'
                        AND
                        name = 'ml_drift_evaluations'
                    """
                )
                .fetchone()
            )


            assert (
                table
                is not None
            )


# ============================================================
# ROUNDTRIP
# ============================================================


def test_register_get_roundtrip(
) -> None:

    with isolated_environment():

        (
            artifact,
            profile,
        ) = persisted_authority()


        drift = (
            evaluation(
                artifact=artifact,
                profile=profile,
            )
        )


        persisted = (
            register_ml_drift_evaluation(
                evaluation=drift
            )
        )


        restored = (
            get_ml_drift_evaluation(
                evaluation_id=(
                    drift.evaluation_id
                ),

                workflow_id=(
                    drift.workflow_id
                ),
            )
        )


        assert (
            persisted
            ==
            drift
        )


        assert (
            restored
            ==
            drift
        )


# ============================================================
# MANY EVALUATIONS / HISTORY
# ============================================================


def test_multiple_evaluations_per_profile(
) -> None:

    with isolated_environment():

        (
            artifact,
            profile,
        ) = persisted_authority()


        first = (
            evaluation(
                artifact=artifact,
                profile=profile,
                observed_dataset_id=(
                    "dataset:observed-1"
                ),
            )
        )


        second = (
            evaluation(
                artifact=artifact,
                profile=profile,
                observed_dataset_id=(
                    "dataset:observed-2"
                ),
            )
        )


        register_ml_drift_evaluation(
            evaluation=first
        )

        register_ml_drift_evaluation(
            evaluation=second
        )


        evaluations = (
            list_ml_drift_evaluations_for_model(
                model_id=artifact.model_id,
                workflow_id=artifact.workflow_id,
            )
        )


        assert (
            len(
                evaluations
            )
            ==
            2
        )


        assert {
            item.evaluation_id
            for item
            in evaluations
        } == {
            first.evaluation_id,
            second.evaluation_id,
        }


# ============================================================
# DUPLICATE ID
# ============================================================


def test_duplicate_evaluation_identity_blocked(
) -> None:

    with isolated_environment():

        (
            artifact,
            profile,
        ) = persisted_authority()


        drift = (
            evaluation(
                artifact=artifact,
                profile=profile,
            )
        )


        register_ml_drift_evaluation(
            evaluation=drift
        )


        expect_error(
            MLDriftEvaluationAlreadyExistsError,

            lambda:
                register_ml_drift_evaluation(
                    evaluation=drift
                ),
        )


# ============================================================
# AUTHORITY
# ============================================================


def test_unregistered_monitoring_profile_blocked(
) -> None:

    with isolated_environment():

        (
            artifact,
            profile,
        ) = (
            persisted_artifact_and_profile()
        )


        drift = (
            evaluation(
                artifact=artifact,
                profile=profile,
            )
        )


        expect_error(
            MLDriftEvaluationAuthorityError,

            lambda:
                register_ml_drift_evaluation(
                    evaluation=drift
                ),
        )


def test_wrong_profile_identity_blocked(
) -> None:

    with isolated_environment():

        (
            artifact,
            profile,
        ) = persisted_authority()


        drift = (
            evaluation(
                artifact=artifact,
                profile=profile,
            )
        )


        payload = (
            drift.model_dump(
                mode="json"
            )
        )


        payload[
            "profile_id"
        ] = (
            "monitoring-profile:"
            +
            (
                "f"
                *
                32
            )
        )


        tampered = (
            MLDriftEvaluationRecord
            .model_validate(
                payload
            )
        )


        expect_error(
            MLDriftEvaluationAuthorityError,

            lambda:
                register_ml_drift_evaluation(
                    evaluation=tampered
                ),
        )


def test_wrong_reference_dataset_binding_blocked(
) -> None:

    with isolated_environment():

        (
            artifact,
            profile,
        ) = persisted_authority()


        drift = (
            evaluation(
                artifact=artifact,
                profile=profile,
            )
        )


        payload = (
            drift.model_dump(
                mode="json"
            )
        )


        payload[
            "reference_dataset_id"
        ] = (
            "dataset:invented-reference"
        )


        tampered = (
            MLDriftEvaluationRecord
            .model_validate(
                payload
            )
        )


        expect_error(
            MLDriftEvaluationAuthorityError,

            lambda:
                register_ml_drift_evaluation(
                    evaluation=tampered
                ),
        )


def test_training_fingerprint_binding_blocked(
) -> None:

    with isolated_environment():

        (
            artifact,
            profile,
        ) = persisted_authority()


        drift = (
            evaluation(
                artifact=artifact,
                profile=profile,
            )
        )


        payload = (
            drift.model_dump(
                mode="json"
            )
        )


        payload[
            "training_contract_sha256"
        ] = (
            "b"
            *
            64
        )


        tampered = (
            MLDriftEvaluationRecord
            .model_validate(
                payload
            )
        )


        expect_error(
            MLDriftEvaluationAuthorityError,

            lambda:
                register_ml_drift_evaluation(
                    evaluation=tampered
                ),
        )


# ============================================================
# HISTORICAL SNAPSHOT
# ============================================================


def test_current_preparation_revision_does_not_rewrite_history(
) -> None:

    with isolated_environment():

        (
            artifact,
            profile,
        ) = persisted_authority()


        with sqlite_connection(
            write=True
        ) as connection:

            connection.execute(
                """
                UPDATE preparation_sessions

                SET revision = ?

                WHERE
                    workflow_id = ?
                """,
                (
                    99,
                    profile.workflow_id,
                ),
            )


        drift = (
            evaluation(
                artifact=artifact,
                profile=profile,
            )
        )


        persisted = (
            register_ml_drift_evaluation(
                evaluation=drift
            )
        )


        assert (
            persisted
            .preparation_session_revision
            ==
            profile
            .preparation_session_revision
        )


# ============================================================
# LIST / ISOLATION
# ============================================================


def test_workflow_listing_and_isolation(
) -> None:

    with isolated_environment():

        (
            artifact,
            profile,
        ) = persisted_authority()


        drift = (
            evaluation(
                artifact=artifact,
                profile=profile,
            )
        )


        register_ml_drift_evaluation(
            evaluation=drift
        )


        matching = (
            list_ml_drift_evaluations_for_workflow(
                workflow_id=(
                    profile.workflow_id
                )
            )
        )


        other = (
            list_ml_drift_evaluations_for_workflow(
                workflow_id="prep:other"
            )
        )


        assert (
            matching
            ==
            [
                drift
            ]
        )


        assert (
            other
            ==
            []
        )


def test_get_workflow_mismatch_blocked(
) -> None:

    with isolated_environment():

        (
            artifact,
            profile,
        ) = persisted_authority()


        drift = (
            evaluation(
                artifact=artifact,
                profile=profile,
            )
        )


        register_ml_drift_evaluation(
            evaluation=drift
        )


        expect_error(
            MLDriftEvaluationWorkflowMismatchError,

            lambda:
                get_ml_drift_evaluation(
                    evaluation_id=(
                        drift.evaluation_id
                    ),

                    workflow_id="prep:other",
                ),
        )


def test_missing_evaluation_explicit(
) -> None:

    with isolated_environment():

        _ = (
            sqlite_schema_version()
        )


        expect_error(
            MLDriftEvaluationNotFoundError,

            lambda:
                get_ml_drift_evaluation(
                    evaluation_id=(
                        "drift-evaluation:"
                        +
                        (
                            "0"
                            *
                            32
                        )
                    )
                ),
        )


# ============================================================
# PRIVACY
# ============================================================


def test_persisted_payload_is_aggregate_only(
) -> None:

    with isolated_environment():

        (
            artifact,
            profile,
        ) = persisted_authority()


        observed = (
            training_frame()
        )


        secret = (
            "secret-drift-category"
        )


        observed[
            "segment"
        ] = [
            secret
            for _
            in range(
                len(
                    observed
                )
            )
        ]


        drift = (
            evaluate_ml_drift(
                observed_features=observed,

                observed_dataset_id=(
                    "dataset:privacy-store"
                ),

                monitoring_profile=profile,

                model_artifact=artifact,
            )
        )


        register_ml_drift_evaluation(
            evaluation=drift
        )


        store_root = str(
            resolve_ml_model_artifact_store_path()
            .expanduser()
            .resolve()
        )


        with sqlite_connection(
            write=False
        ) as connection:

            row = (
                connection.execute(
                    """
                    SELECT payload_json
                    FROM ml_drift_evaluations

                    WHERE
                        store_root = ?
                        AND
                        evaluation_id = ?
                    """,
                    (
                        store_root,
                        drift.evaluation_id,
                    ),
                )
                .fetchone()
            )


        assert (
            row
            is not None
        )


        raw_payload = str(
            row[
                "payload_json"
            ]
        )


        payload = json.loads(
            raw_payload
        )


        assert (
            secret
            not in
            raw_payload
        )


        forbidden = {
            "raw_rows",
            "raw_values",
            "predictions",
            "model_bytes",
            "model_path",
            "x_train",
            "x_test",
            "y_train",
            "y_test",
        }


        assert (
            forbidden
            .isdisjoint(
                payload
            )
        )


        assert (
            payload[
                "privacy_scope"
            ]
            ==
            "aggregate_only"
        )


# ============================================================
# CASCADE
# ============================================================


def test_model_delete_cascades_drift_history(
) -> None:

    with isolated_environment():

        (
            artifact,
            profile,
        ) = persisted_authority()


        drift = (
            evaluation(
                artifact=artifact,
                profile=profile,
            )
        )


        register_ml_drift_evaluation(
            evaluation=drift
        )


        store_root = str(
            resolve_ml_model_artifact_store_path()
            .expanduser()
            .resolve()
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
                    artifact.model_id,
                ),
            )


        expect_error(
            MLDriftEvaluationNotFoundError,

            lambda:
                get_ml_drift_evaluation(
                    evaluation_id=(
                        drift.evaluation_id
                    )
                ),
        )


        with sqlite_connection(
            write=False
        ) as connection:

            profile_row = (
                connection.execute(
                    """
                    SELECT profile_id
                    FROM ml_monitoring_profiles

                    WHERE
                        store_root = ?
                        AND
                        profile_id = ?
                    """,
                    (
                        store_root,
                        profile.profile_id,
                    ),
                )
                .fetchone()
            )


            drift_row = (
                connection.execute(
                    """
                    SELECT evaluation_id
                    FROM ml_drift_evaluations

                    WHERE
                        store_root = ?
                        AND
                        evaluation_id = ?
                    """,
                    (
                        store_root,
                        drift.evaluation_id,
                    ),
                )
                .fetchone()
            )


        assert (
            profile_row
            is None
        )


        assert (
            drift_row
            is None
        )


# ============================================================
# VERSION
# ============================================================


def test_store_rule_version(
) -> None:

    assert (
        ML_DRIFT_EVALUATION_STORE_RULE_VERSION
        ==
        "ml_drift_evaluation_store_v0.1"
    )


# ============================================================
# MAIN
# ============================================================


def main(
) -> None:

    print(
        "=== DATALENS ML DRIFT EVALUATION STORE v0.1 ==="
    )

    print()


    tests = [
        (
            "SQLite schema v11",
            test_sqlite_schema_v11,
        ),
        (
            "Register / get roundtrip",
            test_register_get_roundtrip,
        ),
        (
            "Multiple evaluations per Monitoring Profile",
            test_multiple_evaluations_per_profile,
        ),
        (
            "Duplicate evaluation identity blocked",
            test_duplicate_evaluation_identity_blocked,
        ),
        (
            "Unregistered Monitoring Profile blocked",
            test_unregistered_monitoring_profile_blocked,
        ),
        (
            "Monitoring Profile identity binding",
            test_wrong_profile_identity_blocked,
        ),
        (
            "Reference dataset authority binding",
            test_wrong_reference_dataset_binding_blocked,
        ),
        (
            "Training fingerprint authority binding",
            test_training_fingerprint_binding_blocked,
        ),
        (
            "Historical Preparation snapshot preserved",
            test_current_preparation_revision_does_not_rewrite_history,
        ),
        (
            "Workflow drift history isolation",
            test_workflow_listing_and_isolation,
        ),
        (
            "Read workflow mismatch blocked",
            test_get_workflow_mismatch_blocked,
        ),
        (
            "Explicit missing evaluation",
            test_missing_evaluation_explicit,
        ),
        (
            "Aggregate-only persistence",
            test_persisted_payload_is_aggregate_only,
        ),
        (
            "Model deletion cascades Drift history",
            test_model_delete_cascades_drift_history,
        ),
        (
            "Drift Evaluation Store rule version",
            test_store_rule_version,
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
        "PASS - ML Drift Evaluation Store v0.1"
    )


if __name__ == "__main__":
    main()
