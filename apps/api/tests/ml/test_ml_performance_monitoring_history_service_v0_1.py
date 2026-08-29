from __future__ import annotations


from contextlib import (
    contextmanager,
)


import app.ml.performance_monitoring_history_service as history_module


from app.ml.performance_evaluation_store import (
    MLPerformanceEvaluationStoreError,
    register_ml_performance_evaluation,
)


from app.ml.performance_evaluator import (
    evaluate_ml_performance,
)


from app.ml.performance_monitoring_history_service import (
    ML_PERFORMANCE_MONITORING_HISTORY_SERVICE_RULE_VERSION,
    MLPerformanceMonitoringHistoryInputError,
    MLPerformanceMonitoringHistoryNotFoundError,
    MLPerformanceMonitoringHistoryStorageError,
    get_ml_performance_monitoring_evaluation,
    list_ml_performance_monitoring_model_history,
    list_ml_performance_monitoring_workflow_history,
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


def seed_performance_history(
):

    (
        session,
        dataset_id,
        dataframe,
    ) = (
        build_ready_preparation_workflow()
    )


    (
        trusted_model,
        estimator,
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
                0,
                1,
            ],
        )
    )


    evaluation = (
        evaluate_ml_performance(
            observed_dataframe=
                dataframe,

            observed_dataset_id=
                dataset_id,

            observed_preparation_session_revision=
                session.revision,

            trusted_model=
                trusted_model,
        )
    )


    persisted = (
        register_ml_performance_evaluation(
            evaluation=
                evaluation
        )
    )


    assert (
        estimator.call_count
        ==
        1
    )


    return (
        session,
        dataset_id,
        trusted_model.artifact,
        persisted,
    )


@contextmanager
def patched_store_failure(
    *,
    function_name: str,
):

    original = getattr(
        history_module,
        function_name,
    )


    def failing_store(
        **kwargs,
    ):

        raise MLPerformanceEvaluationStoreError(
            "injected history storage failure"
        )


    setattr(
        history_module,
        function_name,
        failing_store,
    )


    try:
        yield

    finally:
        setattr(
            history_module,
            function_name,
            original,
        )


# ============================================================
# INPUT
# ============================================================


def test_empty_history_identities_blocked(
) -> None:

    expect_error(
        MLPerformanceMonitoringHistoryInputError,

        lambda:
            get_ml_performance_monitoring_evaluation(
                workflow_id=
                    "   ",

                performance_evaluation_id=
                    "performance-evaluation:any",
            ),
    )


    expect_error(
        MLPerformanceMonitoringHistoryInputError,

        lambda:
            get_ml_performance_monitoring_evaluation(
                workflow_id=
                    "prep:any",

                performance_evaluation_id=
                    "   ",
            ),
    )


    expect_error(
        MLPerformanceMonitoringHistoryInputError,

        lambda:
            list_ml_performance_monitoring_model_history(
                workflow_id=
                    "prep:any",

                model_id=
                    "   ",
            ),
    )


    expect_error(
        MLPerformanceMonitoringHistoryInputError,

        lambda:
            list_ml_performance_monitoring_workflow_history(
                workflow_id=
                    "   "
            ),
    )


# ============================================================
# DETAIL
# ============================================================


def test_persisted_evaluation_detail_roundtrip(
) -> None:

    with isolated_real_handoff_environment():

        (
            session,
            _,
            _,
            evaluation,
        ) = (
            seed_performance_history()
        )


        restored = (
            get_ml_performance_monitoring_evaluation(
                workflow_id=
                    session.workflow_id,

                performance_evaluation_id=(
                    evaluation
                    .performance_evaluation_id
                ),
            )
        )


        assert (
            restored
            ==
            evaluation
        )


# ============================================================
# MODEL HISTORY
# ============================================================


def test_persisted_model_history_roundtrip(
) -> None:

    with isolated_real_handoff_environment():

        (
            session,
            _,
            artifact,
            evaluation,
        ) = (
            seed_performance_history()
        )


        history = (
            list_ml_performance_monitoring_model_history(
                workflow_id=
                    session.workflow_id,

                model_id=
                    artifact.model_id,
            )
        )


        assert history == [
            evaluation
        ]


# ============================================================
# WORKFLOW HISTORY
# ============================================================


def test_persisted_workflow_history_roundtrip(
) -> None:

    with isolated_real_handoff_environment():

        (
            session,
            _,
            _,
            evaluation,
        ) = (
            seed_performance_history()
        )


        history = (
            list_ml_performance_monitoring_workflow_history(
                workflow_id=
                    session.workflow_id
            )
        )


        assert history == [
            evaluation
        ]


# ============================================================
# READINESS INDEPENDENCE
# ============================================================


def test_history_remains_readable_after_workflow_becomes_unready(
) -> None:

    with isolated_real_handoff_environment():

        (
            session,
            dataset_id,
            artifact,
            evaluation,
        ) = (
            seed_performance_history()
        )


        # ----------------------------------------------------
        # Advance Preparation after Performance persistence.
        #
        # The persisted Performance Evaluation now refers to an
        # older observed revision. This is expected historical
        # evidence and must remain readable.
        # ----------------------------------------------------


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
                    "test:history-readiness-independent"
                ],

                blocking_reasons=[
                    "test:workflow-no-longer-ready"
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


        assert (
            changed
            .snapshot
            .ready_for_analysis
            is False
        )


        detail = (
            get_ml_performance_monitoring_evaluation(
                workflow_id=
                    session.workflow_id,

                performance_evaluation_id=(
                    evaluation
                    .performance_evaluation_id
                ),
            )
        )


        model_history = (
            list_ml_performance_monitoring_model_history(
                workflow_id=
                    session.workflow_id,

                model_id=
                    artifact.model_id,
            )
        )


        workflow_history = (
            list_ml_performance_monitoring_workflow_history(
                workflow_id=
                    session.workflow_id
            )
        )


        assert (
            detail
            ==
            evaluation
        )


        assert model_history == [
            evaluation
        ]


        assert workflow_history == [
            evaluation
        ]


        assert (
            detail
            .observed_preparation_session_revision
            ==
            session.revision
        )


# ============================================================
# NON-ENUMERATION
# ============================================================


def test_cross_workflow_evaluation_is_non_enumerating(
) -> None:

    with isolated_real_handoff_environment():

        (
            first_session,
            _,
            _,
            evaluation,
        ) = (
            seed_performance_history()
        )


        (
            second_session,
            _,
            _,
        ) = (
            build_ready_preparation_workflow()
        )


        assert (
            first_session.workflow_id
            !=
            second_session.workflow_id
        )


        expect_error(
            MLPerformanceMonitoringHistoryNotFoundError,

            lambda:
                get_ml_performance_monitoring_evaluation(
                    workflow_id=
                        second_session.workflow_id,

                    performance_evaluation_id=(
                        evaluation
                        .performance_evaluation_id
                    ),
                ),
        )


def test_cross_workflow_model_is_non_enumerating(
) -> None:

    with isolated_real_handoff_environment():

        (
            first_session,
            _,
            artifact,
            _,
        ) = (
            seed_performance_history()
        )


        (
            second_session,
            _,
            _,
        ) = (
            build_ready_preparation_workflow()
        )


        assert (
            first_session.workflow_id
            !=
            second_session.workflow_id
        )


        expect_error(
            MLPerformanceMonitoringHistoryNotFoundError,

            lambda:
                list_ml_performance_monitoring_model_history(
                    workflow_id=
                        second_session.workflow_id,

                    model_id=
                        artifact.model_id,
                ),
        )


# ============================================================
# MISSING RESOURCES
# ============================================================


def test_missing_workflow_is_not_found(
) -> None:

    with isolated_real_handoff_environment():

        expect_error(
            MLPerformanceMonitoringHistoryNotFoundError,

            lambda:
                list_ml_performance_monitoring_workflow_history(
                    workflow_id=
                        "prep:missing-workflow"
                ),
        )


def test_missing_evaluation_is_not_found(
) -> None:

    with isolated_real_handoff_environment():

        (
            session,
            _,
            _,
        ) = (
            build_ready_preparation_workflow()
        )


        expect_error(
            MLPerformanceMonitoringHistoryNotFoundError,

            lambda:
                get_ml_performance_monitoring_evaluation(
                    workflow_id=
                        session.workflow_id,

                    performance_evaluation_id=(
                        "performance-evaluation:"
                        +
                        "f" * 32
                    ),
                ),
        )


# ============================================================
# STORAGE TRANSLATION
# ============================================================


def test_detail_store_failure_is_storage_error(
) -> None:

    with isolated_real_handoff_environment():

        (
            session,
            _,
            _,
        ) = (
            build_ready_preparation_workflow()
        )


        with patched_store_failure(
            function_name=
                "get_ml_performance_evaluation"
        ):

            expect_error(
                MLPerformanceMonitoringHistoryStorageError,

                lambda:
                    get_ml_performance_monitoring_evaluation(
                        workflow_id=
                            session.workflow_id,

                        performance_evaluation_id=(
                            "performance-evaluation:"
                            +
                            "a" * 32
                        ),
                    ),
            )


def test_model_history_store_failure_is_storage_error(
) -> None:

    with isolated_real_handoff_environment():

        (
            session,
            _,
            artifact,
            _,
        ) = (
            seed_performance_history()
        )


        with patched_store_failure(
            function_name=(
                "list_ml_performance_evaluations_for_model"
            )
        ):

            expect_error(
                MLPerformanceMonitoringHistoryStorageError,

                lambda:
                    list_ml_performance_monitoring_model_history(
                        workflow_id=
                            session.workflow_id,

                        model_id=
                            artifact.model_id,
                    ),
            )


def test_workflow_history_store_failure_is_storage_error(
) -> None:

    with isolated_real_handoff_environment():

        (
            session,
            _,
            _,
        ) = (
            build_ready_preparation_workflow()
        )


        with patched_store_failure(
            function_name=(
                "list_ml_performance_evaluations_for_workflow"
            )
        ):

            expect_error(
                MLPerformanceMonitoringHistoryStorageError,

                lambda:
                    list_ml_performance_monitoring_workflow_history(
                        workflow_id=
                            session.workflow_id
                    ),
            )


# ============================================================
# RULE VERSION
# ============================================================


def test_rule_version(
) -> None:

    assert (
        ML_PERFORMANCE_MONITORING_HISTORY_SERVICE_RULE_VERSION
        ==
        "ml_performance_monitoring_history_service_v0.1"
    )


# ============================================================
# MAIN
# ============================================================


def main(
) -> None:

    print(
        (
            "=== DATALENS ML PERFORMANCE MONITORING "
            "HISTORY SERVICE v0.1 ==="
        )
    )

    print()


    tests = [
        (
            "Empty history identities blocked",
            test_empty_history_identities_blocked,
        ),
        (
            "Persisted evaluation detail roundtrip",
            test_persisted_evaluation_detail_roundtrip,
        ),
        (
            "Persisted model history roundtrip",
            test_persisted_model_history_roundtrip,
        ),
        (
            "Persisted workflow history roundtrip",
            test_persisted_workflow_history_roundtrip,
        ),
        (
            "History remains readable after workflow becomes unready",
            test_history_remains_readable_after_workflow_becomes_unready,
        ),
        (
            "Cross-workflow evaluation remains non-enumerating",
            test_cross_workflow_evaluation_is_non_enumerating,
        ),
        (
            "Cross-workflow model remains non-enumerating",
            test_cross_workflow_model_is_non_enumerating,
        ),
        (
            "Missing workflow maps to not-found",
            test_missing_workflow_is_not_found,
        ),
        (
            "Missing evaluation maps to not-found",
            test_missing_evaluation_is_not_found,
        ),
        (
            "Detail Store failure maps to storage error",
            test_detail_store_failure_is_storage_error,
        ),
        (
            "Model history Store failure maps to storage error",
            test_model_history_store_failure_is_storage_error,
        ),
        (
            "Workflow history Store failure maps to storage error",
            test_workflow_history_store_failure_is_storage_error,
        ),
        (
            "Performance History Service rule version",
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
            "PASS - ML Performance Monitoring "
            "History Service v0.1"
        )
    )


if __name__ == "__main__":
    main()
