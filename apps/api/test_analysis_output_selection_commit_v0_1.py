from __future__ import annotations

import pandas as pd

import app.preparation.preparation_session as session_module

from app.preparation.analysis_output_selection import (
    AnalysisOutputSelectionBlockedError,
)

from app.preparation.analysis_output_selection_commit import (
    ANALYSIS_OUTPUT_SELECTION_COMMIT_VERSION,
    commit_analysis_output_selection,
)

from app.preparation.preparation_artifact_store import (
    put_preparation_artifact,
    reset_preparation_artifact_store_for_tests,
)

from app.preparation.preparation_session import (
    PREPARATION_SESSION_RULE_VERSION,
    PreparationSessionRevisionConflictError,
    create_preparation_session,
    get_preparation_session,
    record_analysis_output_selection,
    record_optional_stage_signal,
    record_validation_stage_signal,
    reset_preparation_session_store_for_tests,
)

from app.preparation.preparation_workflow import (
    PreparationStage,
)


# ============================================================
# DATA
# ============================================================


def sales_frame() -> pd.DataFrame:
    return (
        pd.DataFrame(
            {
                "sale_id": [
                    1,
                    2,
                    3,
                ],

                "customer_id": [
                    "c1",
                    "c1",
                    "c2",
                ],

                "amount": [
                    10.0,
                    20.0,
                    30.0,
                ],
            }
        )
    )


def customers_frame() -> pd.DataFrame:
    return (
        pd.DataFrame(
            {
                "customer_id": [
                    "c1",
                    "c2",
                ],

                "segment": [
                    "A",
                    "B",
                ],
            }
        )
    )


def joined_frame() -> pd.DataFrame:
    return (
        pd.DataFrame(
            {
                "sale_id": [
                    1,
                    2,
                    3,
                ],

                "customer_id": [
                    "c1",
                    "c1",
                    "c2",
                ],

                "amount": [
                    10.0,
                    20.0,
                    30.0,
                ],

                "segment": [
                    "A",
                    "A",
                    "B",
                ],
            }
        )
    )


# ============================================================
# FIXTURE
# ============================================================


def build_ready_session():
    reset_preparation_session_store_for_tests()

    reset_preparation_artifact_store_for_tests()


    session = (
        create_preparation_session(
            selected_analysis_dataset_ids=[
                "sales",
                "customers",
            ]
        )
    )


    workflow_id = (
        session.workflow_id
    )


    # ========================================================
    # MATERIALIZED ROOT ARTIFACTS
    # ========================================================


    put_preparation_artifact(
        workflow_id=
            workflow_id,

        dataset_id=
            "sales",

        dataset_filename=
            "sales.csv",

        stage=
            "clean",

        dataframe=
            sales_frame(),

        parent_dataset_ids=[
            "sales",
        ],

        evidence_refs=[
            "cleaning:sales",
        ],
    )


    put_preparation_artifact(
        workflow_id=
            workflow_id,

        dataset_id=
            "customers",

        dataset_filename=
            "customers.csv",

        stage=
            "clean",

        dataframe=
            customers_frame(),

        parent_dataset_ids=[
            "customers",
        ],

        evidence_refs=[
            "cleaning:customers",
        ],
    )


    # ========================================================
    # MATERIALIZED COMBINE OUTPUT
    # ========================================================


    put_preparation_artifact(
        workflow_id=
            workflow_id,

        dataset_id=
            "sales_customers",

        dataset_filename=
            "sales_customers.csv",

        stage=
            "combine",

        dataframe=
            joined_frame(),

        parent_dataset_ids=[
            "sales",
            "customers",
        ],

        evidence_refs=[
            "join:validated",
        ],
    )


    # ========================================================
    # RESOLVE COMBINE
    #
    # analysis_output_selection_v0.1 requires COMBINE to be
    # PASSED or SKIPPED.
    # ========================================================


    session = (
        record_optional_stage_signal(
            workflow_id=
                workflow_id,

            stage=
                PreparationStage.COMBINE,

            required=
                True,

            completed=
                True,

            review_required=
                False,

            blocked=
                False,

            dataset_ids=[
                "sales_customers",
            ],

            evidence_refs=[
                "join_materialization:test",
            ],

            blocking_reasons=[],
        )
    )


    return (
        session
    )


# ============================================================
# 1. VERSION
# ============================================================


def test_versions() -> None:
    assert (
        PREPARATION_SESSION_RULE_VERSION
        ==
        "preparation_session_v0.2"
    )


    assert (
        ANALYSIS_OUTPUT_SELECTION_COMMIT_VERSION
        ==
        "analysis_output_selection_commit_v0.1"
    )


    print(
        "Preparation Session v0.2 + commit version: PASS"
    )


# ============================================================
# 2. INITIAL OUTPUT SCOPE EMPTY
# ============================================================


def test_initial_output_scope_empty() -> None:
    reset_preparation_session_store_for_tests()

    session = (
        create_preparation_session(
            selected_analysis_dataset_ids=[
                "sales",
                "customers",
            ]
        )
    )


    assert (
        session.selected_analysis_dataset_ids
        ==
        [
            "sales",
            "customers",
        ]
    )


    assert (
        session.analysis_output_dataset_ids
        ==
        []
    )


    print(
        "Initial analytical output scope is empty: PASS"
    )


# ============================================================
# 3. VALID OUTPUT COMMITTED
# ============================================================


def test_valid_output_committed() -> None:
    session = (
        build_ready_session()
    )


    previous_revision = (
        session.revision
    )


    result = (
        commit_analysis_output_selection(
            workflow_id=
                session.workflow_id,

            requested_dataset_ids=[
                "sales_customers",
            ],
        )
    )


    assert (
        result.previous_revision
        ==
        previous_revision
    )


    assert (
        result.committed_revision
        ==
        previous_revision
        +
        1
    )


    assert (
        result.analysis_output_dataset_ids
        ==
        [
            "sales_customers",
        ]
    )


    current = (
        get_preparation_session(
            session.workflow_id
        )
    )


    assert (
        current.selected_analysis_dataset_ids
        ==
        [
            "sales",
            "customers",
        ]
    )


    assert (
        current.analysis_output_dataset_ids
        ==
        [
            "sales_customers",
        ]
    )


    print(
        "Validated analytical output committed: PASS"
    )


# ============================================================
# 4. INVENTED OUTPUT NEVER COMMITTED
# ============================================================


def test_invented_output_rejected() -> None:
    session = (
        build_ready_session()
    )


    revision_before = (
        session.revision
    )


    try:
        commit_analysis_output_selection(
            workflow_id=
                session.workflow_id,

            requested_dataset_ids=[
                "invented_dataset",
            ],
        )


    except AnalysisOutputSelectionBlockedError:
        pass


    else:
        raise AssertionError(
            (
                "Invented analytical output must never "
                "be committed."
            )
        )


    current = (
        get_preparation_session(
            session.workflow_id
        )
    )


    assert (
        current.revision
        ==
        revision_before
    )


    assert (
        current.analysis_output_dataset_ids
        ==
        []
    )


    print(
        "Invented analytical output cannot reach session: PASS"
    )


# ============================================================
# 5. REVISION CONFLICT
# ============================================================


def test_revision_conflict_rejected() -> None:
    session = (
        build_ready_session()
    )


    stale_revision = (
        session.revision
    )


    # Simulate another legitimate backend state update.
    newer = (
        record_optional_stage_signal(
            workflow_id=
                session.workflow_id,

            stage=
                PreparationStage.TRANSFORM,

            required=
                False,

            completed=
                False,

            review_required=
                False,

            blocked=
                False,

            dataset_ids=[
                "sales",
                "customers",
            ],

            evidence_refs=[
                "transformation_plan:test",
            ],

            blocking_reasons=[],
        )
    )


    assert (
        newer.revision
        ==
        stale_revision
        +
        1
    )


    try:
        record_analysis_output_selection(
            workflow_id=
                session.workflow_id,

            analysis_output_dataset_ids=[
                "sales_customers",
            ],

            expected_revision=
                stale_revision,
        )


    except PreparationSessionRevisionConflictError:
        pass


    else:
        raise AssertionError(
            (
                "Stale analytical output commit must be "
                "rejected."
            )
        )


    current = (
        get_preparation_session(
            session.workflow_id
        )
    )


    assert (
        current.analysis_output_dataset_ids
        ==
        []
    )


    print(
        "Stale analysis output selection rejected: PASS"
    )


# ============================================================
# 6. STAGE UPDATE CANNOT CHANGE OUTPUT SCOPE
# ============================================================


def test_stage_update_cannot_change_output_scope() -> None:
    session = (
        build_ready_session()
    )


    result = (
        commit_analysis_output_selection(
            workflow_id=
                session.workflow_id,

            requested_dataset_ids=[
                "sales_customers",
            ],
        )
    )


    committed_revision = (
        result.committed_revision
    )


    def malicious_stage_updater(
        state,
    ):
        return (
            state.model_copy(
                update={
                    "analysis_output_dataset_ids":
                        [
                            "invented_dataset",
                        ]
                }
            )
        )


    try:
        session_module._SESSION_STORE.update(
            session.workflow_id,
            malicious_stage_updater,
        )


    except ValueError:
        pass


    else:
        raise AssertionError(
            (
                "Generic stage update must never alter "
                "analysis_output_dataset_ids."
            )
        )


    current = (
        get_preparation_session(
            session.workflow_id
        )
    )


    assert (
        current.revision
        ==
        committed_revision
    )


    assert (
        current.analysis_output_dataset_ids
        ==
        [
            "sales_customers",
        ]
    )


    print(
        "Stage update cannot mutate analytical outputs: PASS"
    )


# ============================================================
# 7. ROOT SCOPE REMAINS IMMUTABLE
# ============================================================


def test_root_scope_remains_immutable() -> None:
    session = (
        build_ready_session()
    )


    def malicious_root_updater(
        state,
    ):
        return (
            state.model_copy(
                update={
                    "selected_analysis_dataset_ids":
                        [
                            "sales_customers",
                        ]
                }
            )
        )


    try:
        session_module._SESSION_STORE.update(
            session.workflow_id,
            malicious_root_updater,
        )


    except ValueError:
        pass


    else:
        raise AssertionError(
            (
                "Preparation root scope must remain "
                "immutable."
            )
        )


    current = (
        get_preparation_session(
            session.workflow_id
        )
    )


    assert (
        current.selected_analysis_dataset_ids
        ==
        [
            "sales",
            "customers",
        ]
    )


    print(
        "Preparation root scope remains immutable: PASS"
    )


# ============================================================
# 8. FAILED VALIDATE IS INVALIDATED BY NEW SELECTION
# ============================================================


def test_failed_validate_is_reset_by_selection() -> None:
    session = (
        build_ready_session()
    )


    failed = (
        record_validation_stage_signal(
            workflow_id=
                session.workflow_id,

            completed=
                True,

            passed=
                False,

            dataset_ids=[
                "sales",
            ],

            evidence_refs=[
                "final_validation:failed-test",
            ],

            blocking_reasons=[
                "Synthetic failed validation.",
            ],
        )
    )


    assert (
        failed.snapshot.ready_for_analysis
        is False
    )


    result = (
        commit_analysis_output_selection(
            workflow_id=
                session.workflow_id,

            requested_dataset_ids=[
                "sales_customers",
            ],
        )
    )


    current_state = (
        session_module
        ._SESSION_STORE
        .get(
            session.workflow_id
        )
    )


    assert (
        current_state
        .validate_stage
        .completed
        is False
    )


    assert (
        current_state
        .validate_stage
        .passed
        is False
    )


    assert (
        current_state
        .validate_stage
        .dataset_ids
        ==
        []
    )


    assert (
        result.analysis_output_dataset_ids
        ==
        [
            "sales_customers",
        ]
    )


    print(
        "New output selection invalidates failed VALIDATE: PASS"
    )


# ============================================================
# 9. PASSED VALIDATE LOCKS SELECTION
# ============================================================


def test_passed_validate_locks_selection() -> None:
    session = (
        build_ready_session()
    )


    committed = (
        commit_analysis_output_selection(
            workflow_id=
                session.workflow_id,

            requested_dataset_ids=[
                "sales_customers",
            ],
        )
    )


    # The current Final Validation contract still validates
    # the root scope. We simulate a PASSED validation here only
    # to test the session-level locking rule.
    validated = (
        record_validation_stage_signal(
            workflow_id=
                session.workflow_id,

            completed=
                True,

            passed=
                True,

            dataset_ids=[
                "sales",
                "customers",
            ],

            evidence_refs=[
                "final_validation:passed-test",
            ],

            blocking_reasons=[],
        )
    )


    try:
        record_analysis_output_selection(
            workflow_id=
                session.workflow_id,

            analysis_output_dataset_ids=[
                "sales",
            ],

            expected_revision=
                validated.revision,
        )


    except ValueError:
        pass


    else:
        raise AssertionError(
            (
                "A PASSED VALIDATE stage must lock final "
                "analysis output selection."
            )
        )


    current = (
        get_preparation_session(
            session.workflow_id
        )
    )


    assert (
        current.analysis_output_dataset_ids
        ==
        committed.analysis_output_dataset_ids
    )


    print(
        "PASSED VALIDATE locks analytical output scope: PASS"
    )


# ============================================================
# MAIN
# ============================================================


def main() -> None:
    print(
        (
            "=== DATALENS ANALYSIS OUTPUT "
            "SELECTION COMMIT v0.1 ==="
        )
    )

    print()


    test_versions()

    test_initial_output_scope_empty()

    test_valid_output_committed()

    test_invented_output_rejected()

    test_revision_conflict_rejected()

    test_stage_update_cannot_change_output_scope()

    test_root_scope_remains_immutable()

    test_failed_validate_is_reset_by_selection()

    test_passed_validate_locks_selection()


    print()

    print(
        (
            "Analysis Output Selection "
            "Commit v0.1: PASS"
        )
    )


if __name__ == "__main__":
    main()