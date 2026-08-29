from __future__ import annotations


import inspect


from app.ml.monitoring_history_service import (
    ML_MONITORING_HISTORY_SERVICE_RULE_VERSION,
    MLMonitoringHistoryInputError,
    MLMonitoringHistoryNotFoundError,
    get_ml_monitoring_evaluation,
    list_ml_monitoring_model_history,
    list_ml_monitoring_workflow_history,
)


from app.ml.monitoring_service import (
    run_ml_monitoring,
)


from app.preparation.preparation_session import (
    record_validation_stage_signal,
)


from tests.ml.test_ml_monitoring_service_real_handoff_v0_1 import (
    build_ready_preparation_workflow,
    isolated_real_handoff_environment,
    persist_model_and_profile,
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


def persisted_history(
    *,
    evaluation_count: int = 1,
):

    (
        session,
        dataset_id,
        dataframe,
    ) = (
        build_ready_preparation_workflow()
    )


    (
        artifact,
        profile,
    ) = (
        persist_model_and_profile(
            workflow_id=
                session.workflow_id,

            dataset_id=
                dataset_id,

            preparation_revision=
                session.revision,

            dataframe=
                dataframe,
        )
    )


    evaluations = [
        run_ml_monitoring(
            workflow_id=
                session.workflow_id,

            model_id=
                artifact.model_id,

            observed_dataset_id=
                dataset_id,
        )

        for _
        in range(
            evaluation_count
        )
    ]


    return (
        session,
        dataset_id,
        dataframe,
        artifact,
        profile,
        evaluations,
    )


# ============================================================
# PUBLIC SURFACE
# ============================================================


def test_public_read_surface_accepts_identities_only(
) -> None:

    detail_signature = (
        inspect.signature(
            get_ml_monitoring_evaluation
        )
    )


    assert (
        set(
            detail_signature.parameters
        )
        ==
        {
            "workflow_id",
            "evaluation_id",
        }
    )


    model_signature = (
        inspect.signature(
            list_ml_monitoring_model_history
        )
    )


    assert (
        set(
            model_signature.parameters
        )
        ==
        {
            "workflow_id",
            "model_id",
        }
    )


    workflow_signature = (
        inspect.signature(
            list_ml_monitoring_workflow_history
        )
    )


    assert (
        set(
            workflow_signature.parameters
        )
        ==
        {
            "workflow_id",
        }
    )


# ============================================================
# INPUT
# ============================================================


def test_empty_identities_are_blocked(
) -> None:

    expect_error(
        MLMonitoringHistoryInputError,

        lambda:
            list_ml_monitoring_workflow_history(
                workflow_id="   "
            ),
    )


    expect_error(
        MLMonitoringHistoryInputError,

        lambda:
            list_ml_monitoring_model_history(
                workflow_id="prep:any",
                model_id="   ",
            ),
    )


    expect_error(
        MLMonitoringHistoryInputError,

        lambda:
            get_ml_monitoring_evaluation(
                workflow_id="prep:any",
                evaluation_id="   ",
            ),
    )


# ============================================================
# DETAIL
# ============================================================


def test_evaluation_detail_roundtrip(
) -> None:

    with isolated_real_handoff_environment():

        (
            session,
            _,
            _,
            _,
            _,
            evaluations,
        ) = (
            persisted_history()
        )


        expected = (
            evaluations[
                0
            ]
        )


        restored = (
            get_ml_monitoring_evaluation(
                workflow_id=
                    session.workflow_id,

                evaluation_id=
                    expected.evaluation_id,
            )
        )


        assert (
            restored
            ==
            expected
        )


# ============================================================
# MODEL HISTORY
# ============================================================


def test_model_history_roundtrip(
) -> None:

    with isolated_real_handoff_environment():

        (
            session,
            _,
            _,
            artifact,
            _,
            evaluations,
        ) = (
            persisted_history(
                evaluation_count=2
            )
        )


        history = (
            list_ml_monitoring_model_history(
                workflow_id=
                    session.workflow_id,

                model_id=
                    artifact.model_id,
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
            item.evaluation_id

            for item
            in history
        } == {
            item.evaluation_id

            for item
            in evaluations
        }


        assert all(
            item.model_id
            ==
            artifact.model_id

            for item
            in history
        )


# ============================================================
# WORKFLOW HISTORY
# ============================================================


def test_workflow_history_roundtrip(
) -> None:

    with isolated_real_handoff_environment():

        (
            session,
            _,
            _,
            artifact,
            _,
            evaluations,
        ) = (
            persisted_history(
                evaluation_count=2
            )
        )


        history = (
            list_ml_monitoring_workflow_history(
                workflow_id=
                    session.workflow_id
            )
        )


        assert (
            len(
                history
            )
            ==
            2
        )


        assert all(
            item.workflow_id
            ==
            session.workflow_id

            for item
            in history
        )


        assert all(
            item.model_id
            ==
            artifact.model_id

            for item
            in history
        )


        assert {
            item.evaluation_id

            for item
            in history
        } == {
            item.evaluation_id

            for item
            in evaluations
        }


# ============================================================
# EMPTY HISTORY
# ============================================================


def test_existing_workflow_can_have_empty_history(
) -> None:

    with isolated_real_handoff_environment():

        (
            session,
            _,
            _,
        ) = (
            build_ready_preparation_workflow()
        )


        history = (
            list_ml_monitoring_workflow_history(
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
# NON-ENUMERATION ? EVALUATION
# ============================================================


def test_cross_workflow_evaluation_is_not_enumerable(
) -> None:

    with isolated_real_handoff_environment():

        (
            session,
            _,
            _,
            _,
            _,
            evaluations,
        ) = (
            persisted_history()
        )


        (
            other_session,
            _,
            _,
        ) = (
            build_ready_preparation_workflow()
        )


        assert (
            other_session.workflow_id
            !=
            session.workflow_id
        )


        expect_error(
            MLMonitoringHistoryNotFoundError,

            lambda:
                get_ml_monitoring_evaluation(
                    workflow_id=
                        other_session.workflow_id,

                    evaluation_id=
                        evaluations[
                            0
                        ]
                        .evaluation_id,
                ),
        )


# ============================================================
# NON-ENUMERATION ? MODEL
# ============================================================


def test_cross_workflow_model_is_not_enumerable(
) -> None:

    with isolated_real_handoff_environment():

        (
            session,
            _,
            _,
            artifact,
            _,
            _,
        ) = (
            persisted_history()
        )


        (
            other_session,
            _,
            _,
        ) = (
            build_ready_preparation_workflow()
        )


        assert (
            other_session.workflow_id
            !=
            session.workflow_id
        )


        expect_error(
            MLMonitoringHistoryNotFoundError,

            lambda:
                list_ml_monitoring_model_history(
                    workflow_id=
                        other_session.workflow_id,

                    model_id=
                        artifact.model_id,
                ),
        )


# ============================================================
# MISSING WORKFLOW
# ============================================================


def test_missing_workflow_is_not_found(
) -> None:

    with isolated_real_handoff_environment():

        expect_error(
            MLMonitoringHistoryNotFoundError,

            lambda:
                list_ml_monitoring_workflow_history(
                    workflow_id=
                        "prep:does-not-exist"
                ),
        )


# ============================================================
# HISTORICAL READ DOES NOT REQUIRE CURRENT READINESS
# ============================================================


def test_history_remains_readable_after_workflow_not_ready(
) -> None:

    with isolated_real_handoff_environment():

        (
            session,
            dataset_id,
            _,
            artifact,
            _,
            evaluations,
        ) = (
            persisted_history()
        )


        # ----------------------------------------------------
        # The Drift Evaluation is historical evidence from the
        # previously READY snapshot.
        #
        # Preparation now advances to a new state where final
        # validation is no longer passed.
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
                    "test:validation-reopened"
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


        history = (
            list_ml_monitoring_model_history(
                workflow_id=
                    session.workflow_id,

                model_id=
                    artifact.model_id,
            )
        )


        assert (
            history
            ==
            evaluations
        )


        restored = (
            get_ml_monitoring_evaluation(
                workflow_id=
                    session.workflow_id,

                evaluation_id=
                    evaluations[
                        0
                    ]
                    .evaluation_id,
            )
        )


        assert (
            restored
            ==
            evaluations[
                0
            ]
        )


# ============================================================
# PRIVACY
# ============================================================


def test_history_remains_aggregate_only(
) -> None:

    with isolated_real_handoff_environment():

        (
            session,
            _,
            _,
            artifact,
            _,
            _,
        ) = (
            persisted_history()
        )


        history = (
            list_ml_monitoring_model_history(
                workflow_id=
                    session.workflow_id,

                model_id=
                    artifact.model_id,
            )
        )


        serialized = str(
            [
                item.model_dump(
                    mode="json"
                )

                for item
                in history
            ]
        )


        assert (
            "business_note"
            not in
            serialized
        )


        assert (
            "not-monitored"
            not in
            serialized
        )


        assert (
            "standard"
            not in
            serialized
        )


        assert (
            "premium"
            not in
            serialized
        )


# ============================================================
# VERSION
# ============================================================


def test_rule_version(
) -> None:

    assert (
        ML_MONITORING_HISTORY_SERVICE_RULE_VERSION
        ==
        "ml_monitoring_history_service_v0.1"
    )


# ============================================================
# MAIN
# ============================================================


def main(
) -> None:

    print(
        (
            "=== DATALENS ML MONITORING "
            "HISTORY SERVICE v0.1 ==="
        )
    )

    print()


    tests = [
        (
            "Public history surface accepts identities only",
            test_public_read_surface_accepts_identities_only,
        ),
        (
            "Empty history identities blocked",
            test_empty_identities_are_blocked,
        ),
        (
            "Evaluation detail roundtrip",
            test_evaluation_detail_roundtrip,
        ),
        (
            "Model history roundtrip",
            test_model_history_roundtrip,
        ),
        (
            "Workflow history roundtrip",
            test_workflow_history_roundtrip,
        ),
        (
            "Existing workflow may have empty history",
            test_existing_workflow_can_have_empty_history,
        ),
        (
            "Cross-workflow evaluation non-enumeration",
            test_cross_workflow_evaluation_is_not_enumerable,
        ),
        (
            "Cross-workflow model non-enumeration",
            test_cross_workflow_model_is_not_enumerable,
        ),
        (
            "Missing workflow -> not found",
            test_missing_workflow_is_not_found,
        ),
        (
            "History readable after workflow leaves READY",
            test_history_remains_readable_after_workflow_not_ready,
        ),
        (
            "History remains aggregate-only",
            test_history_remains_aggregate_only,
        ),
        (
            "Monitoring History Service rule version",
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
            "PASS - ML Monitoring "
            "History Service v0.1"
        )
    )


if __name__ == "__main__":
    main()
