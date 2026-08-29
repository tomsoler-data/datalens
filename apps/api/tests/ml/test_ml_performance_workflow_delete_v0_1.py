from __future__ import annotations


import os


from contextlib import (
    contextmanager,
)


from app.ml.performance_evaluation_store import (
    get_ml_performance_evaluation,
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
    archive_preparation_session,
)


from app.preparation.preparation_workflow_delete import (
    delete_preparation_workflow,
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
# ENVIRONMENT
# ============================================================


@contextmanager
def isolated_environment(
):

    previous_quarantine = (
        os.environ.get(
            "DATALENS_WORKFLOW_DELETE_QUARANTINE_ROOT"
        )
    )


    with isolated_real_handoff_environment() as root:

        quarantine_root = (
            root
            /
            "workflow-delete-quarantine"
        )


        os.environ[
            "DATALENS_WORKFLOW_DELETE_QUARANTINE_ROOT"
        ] = str(
            quarantine_root
        )


        try:
            yield root

        finally:

            if previous_quarantine is None:
                os.environ.pop(
                    "DATALENS_WORKFLOW_DELETE_QUARANTINE_ROOT",
                    None,
                )

            else:
                os.environ[
                    "DATALENS_WORKFLOW_DELETE_QUARANTINE_ROOT"
                ] = previous_quarantine


# ============================================================
# SEED
# ============================================================


def seed_performance_workflow(
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


    performance = (
        evaluate_ml_performance(
            observed_dataframe=
                observed_classification_frame(),

            observed_dataset_id=
                dataset_id,

            observed_preparation_session_revision=
                session.revision,

            trusted_model=
                trusted_model,
        )
    )


    register_ml_performance_evaluation(
        evaluation=
            performance
    )


    archived = (
        archive_preparation_session(
            session.workflow_id
        )
    )


    assert (
        archived
        .session
        .workflow_id
        ==
        session.workflow_id
    )


    assert (
        archived
        .session
        .revision
        ==
        session.revision
    )


    assert (
        archived.archived
        is True
    )


    assert (
        archived.display_name
        ==
        "Real monitoring handoff"
    )


    return (
        archived.session,
        trusted_model.artifact,
        performance,
    )


# ============================================================
# COUNT
# ============================================================


def workflow_counts(
    *,
    workflow_id: str,
) -> dict[
    str,
    int,
]:

    with sqlite_connection(
        write=False
    ) as connection:

        return {
            table:
                int(
                    connection.execute(
                        (
                            "SELECT COUNT(*) "
                            f"FROM {table} "
                            "WHERE workflow_id = ?"
                        ),
                        (
                            workflow_id,
                        ),
                    )
                    .fetchone()[0]
                )

            for table
            in [
                "ml_model_artifacts",
                "ml_performance_evaluations",
            ]
        }


# ============================================================
# SUCCESSFUL CASCADE
# ============================================================


def test_permanent_workflow_delete_cascades_performance_history(
) -> None:

    with isolated_environment():

        (
            session,
            _,
            performance,
        ) = (
            seed_performance_workflow()
        )


        before = (
            workflow_counts(
                workflow_id=
                    session.workflow_id
            )
        )


        assert before == {
            "ml_model_artifacts":
                1,

            "ml_performance_evaluations":
                1,
        }


        result = (
            delete_preparation_workflow(
                workflow_id=
                    session.workflow_id,

                confirmation_workflow_id=
                    session.workflow_id,

                confirmation_display_name=
                    "Real monitoring handoff",

                expected_revision=
                    session.revision,
            )
        )


        assert (
            result.ml_model_artifacts_deleted
            ==
            1
        )


        after = (
            workflow_counts(
                workflow_id=
                    session.workflow_id
            )
        )


        assert after == {
            "ml_model_artifacts":
                0,

            "ml_performance_evaluations":
                0,
        }


        # The exact Performance record must have disappeared
        # through the Model Artifact foreign-key cascade.
        assert (
            performance
            .performance_evaluation_id
        )


# ============================================================
# TRANSACTION ROLLBACK
# ============================================================


def test_workflow_delete_rollback_restores_performance_history(
) -> None:

    with isolated_environment():

        (
            session,
            artifact,
            performance,
        ) = (
            seed_performance_workflow()
        )


        def injected_failure(
            stage: str,
        ) -> None:

            if (
                stage
                ==
                "before_commit"
            ):
                raise RuntimeError(
                    "injected performance rollback"
                )


        try:
            delete_preparation_workflow(
                workflow_id=
                    session.workflow_id,

                confirmation_workflow_id=
                    session.workflow_id,

                confirmation_display_name=
                    "Real monitoring handoff",

                expected_revision=
                    session.revision,

                _failure_hook=
                    injected_failure,
            )

        except RuntimeError as error:

            assert (
                "injected performance rollback"
                in
                str(
                    error
                )
            )

        else:
            raise AssertionError(
                (
                    "Injected workflow-delete "
                    "failure should propagate."
                )
            )


        restored = (
            get_ml_performance_evaluation(
                performance_evaluation_id=(
                    performance
                    .performance_evaluation_id
                ),

                workflow_id=
                    session.workflow_id,
            )
        )


        assert (
            restored
            ==
            performance
        )


        assert (
            restored.model_id
            ==
            artifact.model_id
        )


        counts = (
            workflow_counts(
                workflow_id=
                    session.workflow_id
            )
        )


        assert counts == {
            "ml_model_artifacts":
                1,

            "ml_performance_evaluations":
                1,
        }


# ============================================================
# FOREIGN KEY
# ============================================================


def test_performance_history_is_model_subordinate(
) -> None:

    with isolated_environment():

        (
            session,
            artifact,
            _,
        ) = (
            seed_performance_workflow()
        )


        with sqlite_connection(
            write=True
        ) as connection:

            connection.execute(
                """
                DELETE FROM ml_model_artifacts

                WHERE
                    model_id = ?
                    AND
                    workflow_id = ?
                """,
                (
                    artifact.model_id,
                    session.workflow_id,
                ),
            )


        counts = (
            workflow_counts(
                workflow_id=
                    session.workflow_id
            )
        )


        assert counts == {
            "ml_model_artifacts":
                0,

            "ml_performance_evaluations":
                0,
        }


# ============================================================
# SCHEMA
# ============================================================


def test_current_schema_supports_performance_history(
) -> None:

    with isolated_environment():

        assert (
            SQLITE_SCHEMA_VERSION
            ==
            13
        )


        assert (
            sqlite_schema_version()
            ==
            SQLITE_SCHEMA_VERSION
        )


        with sqlite_connection(
            write=False
        ) as connection:

            foreign_keys = (
                connection.execute(
                    """
                    PRAGMA foreign_key_list(
                        ml_performance_evaluations
                    )
                    """
                )
                .fetchall()
            )


        assert any(
            (
                str(
                    row[
                        "table"
                    ]
                )
                ==
                "ml_model_artifacts"
                and
                str(
                    row[
                        "on_delete"
                    ]
                )
                .upper()
                ==
                "CASCADE"
            )

            for row
            in foreign_keys
        )


# ============================================================
# MAIN
# ============================================================


def main(
) -> None:

    print(
        (
            "=== DATALENS ML PERFORMANCE "
            "WORKFLOW DELETE v0.1 ==="
        )
    )

    print()


    tests = [
        (
            "Current SQLite schema supports Performance history",
            test_current_schema_supports_performance_history,
        ),
        (
            "Permanent workflow deletion cascades Performance history",
            test_permanent_workflow_delete_cascades_performance_history,
        ),
        (
            "Workflow deletion rollback restores Performance history",
            test_workflow_delete_rollback_restores_performance_history,
        ),
        (
            "Performance history is Model Artifact subordinate",
            test_performance_history_is_model_subordinate,
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
            "Workflow Delete v0.1"
        )
    )


if __name__ == "__main__":
    main()
