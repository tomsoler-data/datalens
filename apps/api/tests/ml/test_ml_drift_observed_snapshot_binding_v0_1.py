from __future__ import annotations


from app.ml.drift_evaluation_store import (
    MLDriftEvaluationAuthorityError,
    get_ml_drift_evaluation,
    register_ml_drift_evaluation,
)


from app.ml.drift_evaluator import (
    evaluate_ml_drift,
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
# VERSION
# ============================================================


ML_DRIFT_OBSERVED_SNAPSHOT_BINDING_RULE_VERSION = (
    "ml_drift_observed_snapshot_binding_v0.1"
)


# ============================================================
# HELPERS
# ============================================================


def expect_authority_error(
    factory,
) -> None:

    try:
        factory()

    except MLDriftEvaluationAuthorityError:
        return


    raise AssertionError(
        (
            "Expected "
            "MLDriftEvaluationAuthorityError."
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


# ============================================================
# SCHEMA V12
# ============================================================


def test_sqlite_schema_v12(
) -> None:

    with isolated_environment():

        assert (
            SQLITE_SCHEMA_VERSION
            >=
            12
        )


        assert (
            sqlite_schema_version()
            ==
            SQLITE_SCHEMA_VERSION
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
                        ml_drift_evaluations
                    )
                    """
                ).fetchall()
            }


            migration = (
                connection.execute(
                    """
                    SELECT name
                    FROM schema_migrations

                    WHERE
                        version = 12
                    """
                )
                .fetchone()
            )


        assert (
            "observed_preparation_session_revision"
            in
            columns
        )


        assert (
            migration
            is not None
        )


        assert (
            str(
                migration[
                    "name"
                ]
            )
            ==
            "ml_drift_observed_snapshot_binding"
        )


# ============================================================
# ROUNDTRIP
# ============================================================


def test_observed_revision_roundtrip(
) -> None:

    with isolated_environment():

        (
            artifact,
            profile,
        ) = persisted_authority()


        drift = (
            evaluate_ml_drift(
                observed_features=
                    training_frame(),

                observed_dataset_id=
                    "dataset:observed",

                observed_preparation_session_revision=
                    7,

                monitoring_profile=
                    profile,

                model_artifact=
                    artifact,
            )
        )


        assert (
            drift
            .observed_preparation_session_revision
            ==
            7
        )


        register_ml_drift_evaluation(
            evaluation=drift
        )


        restored = (
            get_ml_drift_evaluation(
                evaluation_id=
                    drift.evaluation_id,

                workflow_id=
                    drift.workflow_id,
            )
        )


        assert (
            restored
            .observed_preparation_session_revision
            ==
            7
        )


        assert (
            restored
            ==
            drift
        )


# ============================================================
# LATE REVISION RACE
# ============================================================


def test_late_observed_revision_race_is_fail_closed(
) -> None:

    with isolated_environment():

        (
            artifact,
            profile,
        ) = persisted_authority()


        drift = (
            evaluate_ml_drift(
                observed_features=
                    training_frame(),

                observed_dataset_id=
                    "dataset:observed",

                observed_preparation_session_revision=
                    7,

                monitoring_profile=
                    profile,

                model_artifact=
                    artifact,
            )
        )


        # Simulate Preparation changing after the observed
        # DataFrame was loaded/evaluated but before persistence.
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
                    8,
                    artifact.workflow_id,
                ),
            )


        expect_authority_error(
            lambda:
                register_ml_drift_evaluation(
                    evaluation=drift
                )
        )


        with sqlite_connection(
            write=False
        ) as connection:

            count = int(
                connection.execute(
                    """
                    SELECT COUNT(*)
                    FROM ml_drift_evaluations

                    WHERE
                        evaluation_id = ?
                    """,
                    (
                        drift.evaluation_id,
                    ),
                )
                .fetchone()[0]
            )


        assert (
            count
            ==
            0
        )


# ============================================================
# LEGACY CONTRACT COMPATIBILITY
# ============================================================


def test_pre_v12_payload_can_still_be_read_as_unbound(
) -> None:

    with isolated_environment():

        (
            artifact,
            profile,
        ) = persisted_authority()


        drift = (
            evaluate_ml_drift(
                observed_features=
                    training_frame(),

                observed_dataset_id=
                    "dataset:legacy",

                observed_preparation_session_revision=
                    7,

                monitoring_profile=
                    profile,

                model_artifact=
                    artifact,
            )
        )


        payload = (
            drift.model_dump(
                mode="json"
            )
        )


        payload.pop(
            "observed_preparation_session_revision"
        )


        from app.ml.drift_evaluation import (
            MLDriftEvaluationRecord,
        )


        restored_legacy = (
            MLDriftEvaluationRecord
            .model_validate(
                payload
            )
        )


        assert (
            restored_legacy
            .observed_preparation_session_revision
            is None
        )


# ============================================================
# VERSION
# ============================================================


def test_rule_version(
) -> None:

    assert (
        ML_DRIFT_OBSERVED_SNAPSHOT_BINDING_RULE_VERSION
        ==
        "ml_drift_observed_snapshot_binding_v0.1"
    )


# ============================================================
# MAIN
# ============================================================


def main(
) -> None:

    print(
        (
            "=== DATALENS ML DRIFT OBSERVED "
            "SNAPSHOT BINDING v0.1 ==="
        )
    )

    print()


    test_sqlite_schema_v12()

    print(
        "[PASS] SQLite schema v12 observed snapshot binding"
    )


    test_observed_revision_roundtrip()

    print(
        "[PASS] Observed Preparation revision roundtrip"
    )


    test_late_observed_revision_race_is_fail_closed()

    print(
        "[PASS] Late observed Preparation revision race blocked"
    )


    test_pre_v12_payload_can_still_be_read_as_unbound()

    print(
        "[PASS] Pre-v12 payload compatibility"
    )


    test_rule_version()

    print(
        "[PASS] Observed snapshot binding rule version"
    )


    print()

    print(
        (
            "PASS - ML Drift Observed "
            "Snapshot Binding v0.1"
        )
    )


if __name__ == "__main__":
    main()
