from __future__ import annotations

from contextlib import (
    contextmanager,
)

from types import (
    SimpleNamespace,
)

import pandas as pd

from fastapi import (
    HTTPException,
)

import app.api.preparation_semantic as semantic_api


# ============================================================
# CONSTANTS
# ============================================================

WORKFLOW_ID = (
    "workflow-semantic-wiring-test"
)


# ============================================================
# PATCH HELPER
# ============================================================

@contextmanager
def patched_globals(
    **replacements,
):

    originals = {}


    try:

        for (
            name,
            replacement,
        ) in replacements.items():

            originals[
                name
            ] = getattr(
                semantic_api,
                name,
            )


            setattr(
                semantic_api,
                name,
                replacement,
            )


        yield


    finally:

        for (
            name,
            original,
        ) in originals.items():

            setattr(
                semantic_api,
                name,
                original,
            )


# ============================================================
# FIXTURES
# ============================================================

def deterministic_frame() -> pd.DataFrame:

    return (
        pd.DataFrame(
            {
                "customer_id": [
                    "c1",
                    "c2",
                ],

                "city": [
                    "New York",
                    "NY",
                ],
            }
        )
    )


def semantic_frame() -> pd.DataFrame:

    return (
        pd.DataFrame(
            {
                "customer_id": [
                    "c1",
                    "c2",
                ],

                "city": [
                    "New York",
                    "New York",
                ],
            }
        )
    )


def semantic_plan():

    return (
        SimpleNamespace(
            rule_version=(
                "semantic_plan_test_v0.1"
            )
        )
    )


def semantic_execution(
    *,
    changed_cell_count: int = 1,
):

    return (
        SimpleNamespace(
            rule_version=(
                "semantic_execution_test_v0.1"
            ),

            changed_cell_count=(
                changed_cell_count
            ),

            applied_action_count=(
                1
                if changed_cell_count > 0
                else 0
            ),

            skipped_action_count=(
                0
            ),

            dataset_count=(
                1
            ),

            provenance=[],
        )
    )


def confirmation_report():

    return (
        SimpleNamespace(
            rule_version=(
                "semantic_confirmation_test_v0.1"
            ),

            confirmed=True,
        )
    )


def fake_response(
    **kwargs,
):

    return (
        SimpleNamespace(
            **kwargs
        )
    )


# ============================================================
# COMMON PATCHES
# ============================================================

def base_replacements(
    *,
    events: list[
        str
    ],
    execution=None,
):

    deterministic = (
        deterministic_frame()
    )


    derived = (
        semantic_frame()
    )


    current_plan = (
        semantic_plan()
    )


    current_execution = (
        execution
        or
        semantic_execution()
    )


    def fake_precondition(
        *,
        workflow_id,
    ):

        events.append(
            "precondition"
        )


    def fake_rebuild(
        **kwargs,
    ):

        events.append(
            "rebuild"
        )


        return (
            {
                "sales":
                    deterministic,
            },

            [
                SimpleNamespace(
                    issue_id=(
                        "issue-1"
                    )
                )
            ],

            current_plan,
        )


    def fake_session(
        workflow_id,
    ):

        return (
            SimpleNamespace(
                selected_analysis_dataset_ids=[
                    "sales",
                ]
            )
        )


    def fake_execute(
        **kwargs,
    ):

        events.append(
            "execute"
        )


        return (
            {
                "sales":
                    derived,
            },

            current_execution,
        )


    return {
        "_require_semantic_review_precondition":
            fake_precondition,

        "_rebuild_semantic_cleaning_context":
            fake_rebuild,

        "get_preparation_session":
            fake_session,

        "_parse_semantic_choices":
            lambda raw_value:
                [],

        "execute_semantic_cleaning_plan":
            fake_execute,

        "_parse_confirmed_issue_ids":
            lambda raw_value:
                {
                    "issue-1",
                },

        "_parse_manual_resolutions":
            lambda raw_value:
                [],

        "SemanticReviewConfirmationResponse":
            fake_response,
    }


# ============================================================
# 1. SUCCESS ORDER
# ============================================================

def test_confirmation_materializes_before_clean_passed(
) -> None:

    events: list[
        str
    ] = []


    replacements = (
        base_replacements(
            events=(
                events
            )
        )
    )


    def fake_confirmation(
        **kwargs,
    ):

        assert (
            events[
                -1
            ]
            ==
            "execute"
        )


        events.append(
            "confirm"
        )


        return (
            confirmation_report()
        )


    def fake_materialize(
        **kwargs,
    ):

        assert (
            events[
                -1
            ]
            ==
            "confirm"
        )


        assert (
            kwargs[
                "deterministic_frames"
            ][
                "sales"
            ]
            .equals(
                deterministic_frame()
            )
        )


        assert (
            kwargs[
                "derived_frames"
            ][
                "sales"
            ]
            .equals(
                semantic_frame()
            )
        )


        events.append(
            "materialize"
        )


        return (
            SimpleNamespace(
                materialization_kind=(
                    "semantic_cleaned"
                )
            )
        )


    def fake_record_passed(
        **kwargs,
    ):

        assert (
            events[
                -1
            ]
            ==
            "materialize"
        )


        events.append(
            "record-passed"
        )


    replacements.update(
        {
            "require_semantic_confirmation":
                fake_confirmation,

            "materialize_semantic_cleaning_artifacts":
                fake_materialize,

            "_record_semantic_confirmation_passed":
                fake_record_passed,
        }
    )


    with patched_globals(
        **replacements
    ):

        result = (
            semantic_api
            .confirm_uploaded_semantic_review(
                dataset_files=[],

                workflow_id=(
                    WORKFLOW_ID
                ),

                semantic_decisions_json=(
                    "{}"
                ),

                confirmed_issue_ids_json=(
                    '["issue-1"]'
                ),

                approved_semantic_choices_json=(
                    "[]"
                ),

                manual_resolutions_json=(
                    None
                ),

                approved_action_ids_json=(
                    None
                ),
            )
        )


    assert (
        result.status
        ==
        "confirmed"
    )


    assert (
        events
        ==
        [
            "precondition",
            "rebuild",
            "execute",
            "confirm",
            "materialize",
            "record-passed",
        ]
    )


    print(
        "Semantic confirmation persists artifact before "
        "CLEAN PASSED: PASS"
    )


# ============================================================
# 2. BLOCKED CONFIRMATION DOES NOT MATERIALIZE
# ============================================================

def test_blocked_confirmation_does_not_materialize(
) -> None:

    events: list[
        str
    ] = []


    class FakeSemanticConfirmationBlockedError(
        Exception,
    ):

        def __init__(
            self,
        ) -> None:

            self.report = (
                SimpleNamespace(
                    rule_version=(
                        "semantic_confirmation_test_v0.1"
                    ),

                    model_dump=(
                        lambda mode=None:
                            {
                                "status":
                                    "blocked",
                            }
                    ),
                )
            )


            super().__init__(
                "Synthetic blocked semantic confirmation."
            )


    replacements = (
        base_replacements(
            events=(
                events
            )
        )
    )


    def fake_confirmation(
        **kwargs,
    ):

        events.append(
            "confirm-blocked"
        )


        raise (
            FakeSemanticConfirmationBlockedError()
        )


    def fake_record_blocked(
        **kwargs,
    ):

        events.append(
            "record-blocked"
        )


    def forbidden_materialize(
        **kwargs,
    ):

        raise AssertionError(
            "Blocked semantic confirmation must not "
            "materialize a final artifact."
        )


    def forbidden_passed(
        **kwargs,
    ):

        raise AssertionError(
            "Blocked semantic confirmation must not mark "
            "CLEAN as passed."
        )


    replacements.update(
        {
            "SemanticConfirmationBlockedError":
                FakeSemanticConfirmationBlockedError,

            "require_semantic_confirmation":
                fake_confirmation,

            "_record_semantic_confirmation_blocked":
                fake_record_blocked,

            "materialize_semantic_cleaning_artifacts":
                forbidden_materialize,

            "_record_semantic_confirmation_passed":
                forbidden_passed,
        }
    )


    with patched_globals(
        **replacements
    ):

        try:

            semantic_api.confirm_uploaded_semantic_review(
                dataset_files=[],

                workflow_id=(
                    WORKFLOW_ID
                ),

                semantic_decisions_json=(
                    "{}"
                ),

                confirmed_issue_ids_json=(
                    '["issue-1"]'
                ),

                approved_semantic_choices_json=(
                    "[]"
                ),

                manual_resolutions_json=(
                    None
                ),

                approved_action_ids_json=(
                    None
                ),
            )


        except HTTPException as error:

            assert (
                error.status_code
                ==
                409
            )


        else:

            raise AssertionError(
                "Blocked semantic confirmation must return "
                "HTTP 409."
            )


    assert (
        "record-blocked"
        in events
    )


    assert (
        "materialize"
        not in events
    )


    print(
        "Blocked semantic confirmation does not persist "
        "artifact: PASS"
    )


# ============================================================
# 3. ARTIFACT FAILURE PREVENTS CLEAN PASSED
# ============================================================

def test_artifact_failure_prevents_clean_passed(
) -> None:

    events: list[
        str
    ] = []


    replacements = (
        base_replacements(
            events=(
                events
            )
        )
    )


    def fake_confirmation(
        **kwargs,
    ):

        events.append(
            "confirm"
        )


        return (
            confirmation_report()
        )


    def failing_materialize(
        **kwargs,
    ):

        events.append(
            "materialize-failed"
        )


        raise RuntimeError(
            "Synthetic Semantic Artifact Store failure."
        )


    def forbidden_record_passed(
        **kwargs,
    ):

        events.append(
            "record-passed"
        )


        raise AssertionError(
            "CLEAN must not become PASSED after semantic "
            "artifact persistence failed."
        )


    replacements.update(
        {
            "require_semantic_confirmation":
                fake_confirmation,

            "materialize_semantic_cleaning_artifacts":
                failing_materialize,

            "_record_semantic_confirmation_passed":
                forbidden_record_passed,
        }
    )


    with patched_globals(
        **replacements
    ):

        try:

            semantic_api.confirm_uploaded_semantic_review(
                dataset_files=[],

                workflow_id=(
                    WORKFLOW_ID
                ),

                semantic_decisions_json=(
                    "{}"
                ),

                confirmed_issue_ids_json=(
                    '["issue-1"]'
                ),

                approved_semantic_choices_json=(
                    "[]"
                ),

                manual_resolutions_json=(
                    None
                ),

                approved_action_ids_json=(
                    None
                ),
            )


        except HTTPException as error:

            assert (
                error.status_code
                ==
                500
            )


        else:

            raise AssertionError(
                "Semantic artifact failure must propagate "
                "as HTTP 500."
            )


    assert (
        "materialize-failed"
        in events
    )


    assert (
        "record-passed"
        not in events
    )


    print(
        "Semantic artifact failure prevents CLEAN PASSED: PASS"
    )


# ============================================================
# 4. NON-MUTATING CONFIRMATION CAN STILL PASS
# ============================================================

def test_non_mutating_confirmation_can_pass(
) -> None:

    events: list[
        str
    ] = []


    no_change_execution = (
        semantic_execution(
            changed_cell_count=(
                0
            )
        )
    )


    replacements = (
        base_replacements(
            events=(
                events
            ),

            execution=(
                no_change_execution
            ),
        )
    )


    def fake_confirmation(
        **kwargs,
    ):

        events.append(
            "confirm"
        )


        return (
            confirmation_report()
        )


    def fake_materialize(
        **kwargs,
    ):

        assert (
            kwargs[
                "execution"
            ]
            .changed_cell_count
            ==
            0
        )


        events.append(
            "materialize-no-change"
        )


        return (
            SimpleNamespace(
                materialization_kind=(
                    "no_change"
                )
            )
        )


    def fake_record_passed(
        **kwargs,
    ):

        assert (
            events[
                -1
            ]
            ==
            "materialize-no-change"
        )


        events.append(
            "record-passed"
        )


    replacements.update(
        {
            "require_semantic_confirmation":
                fake_confirmation,

            "materialize_semantic_cleaning_artifacts":
                fake_materialize,

            "_record_semantic_confirmation_passed":
                fake_record_passed,
        }
    )


    with patched_globals(
        **replacements
    ):

        result = (
            semantic_api
            .confirm_uploaded_semantic_review(
                dataset_files=[],

                workflow_id=(
                    WORKFLOW_ID
                ),

                semantic_decisions_json=(
                    "{}"
                ),

                confirmed_issue_ids_json=(
                    '["issue-1"]'
                ),

                approved_semantic_choices_json=(
                    "[]"
                ),

                manual_resolutions_json=(
                    None
                ),

                approved_action_ids_json=(
                    None
                ),
            )
        )


    assert (
        result.status
        ==
        "confirmed"
    )


    assert (
        events[
            -2:
        ]
        ==
        [
            "materialize-no-change",
            "record-passed",
        ]
    )


    print(
        "Non-mutating semantic confirmation can still "
        "complete CLEAN: PASS"
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print(
        "=== DATALENS SEMANTIC → ARTIFACT STORE WIRING v0.1 ==="
    )

    print()


    test_confirmation_materializes_before_clean_passed()

    test_blocked_confirmation_does_not_materialize()

    test_artifact_failure_prevents_clean_passed()

    test_non_mutating_confirmation_can_pass()


    print()


    print(
        "Semantic → Artifact Store Wiring v0.1: PASS"
    )


if __name__ == "__main__":
    main()