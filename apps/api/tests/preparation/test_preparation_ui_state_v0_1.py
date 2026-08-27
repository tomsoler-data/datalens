from __future__ import annotations

import os
import tempfile
from pathlib import Path


# Keep the test isolated from the user's real Preparation
# session store.
_TEMP_DIRECTORY = tempfile.TemporaryDirectory(
    prefix="datalens-preparation-ui-state-"
)

_TEST_ROOT = Path(
    _TEMP_DIRECTORY.name
)

os.environ[
    "DATALENS_PREPARATION_SESSION_STORE_PATH"
] = str(
    _TEST_ROOT
    /
    "preparation_sessions.json"
)


from fastapi import (
    FastAPI,
)

from fastapi.testclient import (
    TestClient,
)


from app.api.preparation_session import (
    router,
)

from app.preparation.preparation_session import (
    create_preparation_session,
    reset_preparation_session_store_for_tests,
)

from app.preparation.preparation_ui_state import (
    PREPARATION_UI_STATE_RULE_VERSION,
    get_preparation_ui_state,
    reset_preparation_ui_state_store_for_tests,
    update_preparation_ui_state,
)


app = FastAPI()

app.include_router(
    router
)

client = TestClient(
    app
)


def reset_state() -> None:
    reset_preparation_session_store_for_tests()

    reset_preparation_ui_state_store_for_tests()


def test_empty_state() -> None:
    reset_state()

    state = (
        get_preparation_ui_state(
            "prep:ui-state-empty"
        )
    )

    assert state.workflow_id == (
        "prep:ui-state-empty"
    )

    assert state.revision == 0

    assert state.quality_report is None

    assert state.cleaning_plan is None

    assert state.cleaning_execution is None

    assert state.semantic_review is None

    assert state.semantic_cleaning_plan is None

    assert state.semantic_cleaning_execution is None

    assert state.semantic_confirmation is None

    assert state.applied_semantic_choices == []

    assert state.confirmed_semantic_issue_ids == []

    assert state.semantic_manual_resolutions == []

    assert state.storage == "sqlite"

    assert state.persistent is True

    assert (
        state.rule_version
        ==
        PREPARATION_UI_STATE_RULE_VERSION
    )

    print(
        "[PASS] empty Preparation UI state"
    )


def test_update_and_invalidation() -> None:
    reset_state()

    workflow_id = (
        "prep:ui-state-update"
    )

    quality_payload = {
        "status":
            "ready",

        "issues": [
            {
                "issue_id":
                    "quality:1",
            },
        ],
    }

    first = (
        update_preparation_ui_state(
            workflow_id=
                workflow_id,

            quality_report=
                quality_payload,
        )
    )

    assert first.revision == 1

    assert (
        first.quality_report
        ==
        quality_payload
    )


    # External input mutation must not mutate the store.
    quality_payload[
        "issues"
    ][
        0
    ][
        "issue_id"
    ] = "mutated-outside"

    stored = (
        get_preparation_ui_state(
            workflow_id
        )
    )

    assert (
        stored
        .quality_report[
            "issues"
        ][
            0
        ][
            "issue_id"
        ]
        ==
        "quality:1"
    )


    second = (
        update_preparation_ui_state(
            workflow_id=
                workflow_id,

            cleaning_plan={
                "actions": [
                    {
                        "action_id":
                            "clean:1",
                    },
                ],
            },

            semantic_review={
                "decisions": [
                    {
                        "issue_id":
                            "semantic:1",
                    },
                ],
            },
        )
    )

    assert second.revision == 2

    assert (
        second.quality_report
        is not None
    )

    assert (
        second.cleaning_plan
        is not None
    )

    assert (
        second.semantic_review
        is not None
    )


    third = (
        update_preparation_ui_state(
            workflow_id=
                workflow_id,

            cleaning_plan=None,

            semantic_review=None,
        )
    )

    assert third.revision == 3

    assert (
        third.quality_report
        is not None
    )

    assert (
        third.cleaning_plan
        is None
    )

    assert (
        third.semantic_review
        is None
    )

    print(
        "[PASS] update + explicit invalidation semantics"
    )


def test_deep_copy_read_isolation() -> None:
    reset_state()

    workflow_id = (
        "prep:ui-state-copy"
    )

    update_preparation_ui_state(
        workflow_id=
            workflow_id,

        semantic_confirmation={
            "confirmed":
                True,

            "unresolved_issue_ids":
                [],
        },

        applied_semantic_choices=[
            {
                "action_id":
                    "semantic:merge:1",

                "canonical_value":
                    "Spain",
            },
        ],

        confirmed_semantic_issue_ids=[
            "semantic:1",
        ],

        semantic_manual_resolutions=[
            {
                "issue_id":
                    "semantic:2",

                "note":
                    "Validated by analyst",
            },
        ],
    )

    first = (
        get_preparation_ui_state(
            workflow_id
        )
    )

    first.applied_semantic_choices[
        0
    ][
        "canonical_value"
    ] = "MUTATED"

    first.confirmed_semantic_issue_ids.append(
        "semantic:999"
    )

    second = (
        get_preparation_ui_state(
            workflow_id
        )
    )

    assert (
        second
        .applied_semantic_choices[
            0
        ][
            "canonical_value"
        ]
        ==
        "Spain"
    )

    assert (
        second.confirmed_semantic_issue_ids
        ==
        [
            "semantic:1",
        ]
    )

    print(
        "[PASS] deep-copy isolation"
    )


def test_read_endpoint() -> None:
    reset_state()

    session = (
        create_preparation_session(
            selected_analysis_dataset_ids=[
                "dataset:orders",
            ]
        )
    )

    update_preparation_ui_state(
        workflow_id=
            session.workflow_id,

        quality_report={
            "status":
                "ready",

            "rule_version":
                "test-quality-v0.1",
        },
    )

    response = client.get(
        (
            "/preparation/sessions/"
            f"{session.workflow_id}"
            "/ui-state"
        )
    )

    assert response.status_code == 200

    payload = response.json()

    assert (
        payload[
            "workflow_id"
        ]
        ==
        session.workflow_id
    )

    assert (
        payload[
            "quality_report"
        ][
            "rule_version"
        ]
        ==
        "test-quality-v0.1"
    )

    assert (
        payload[
            "persistent"
        ]
        is True
    )


    missing = client.get(
        (
            "/preparation/sessions/"
            "prep:does-not-exist"
            "/ui-state"
        )
    )

    assert missing.status_code == 404

    print(
        "[PASS] Preparation UI-state read endpoint"
    )


def main() -> None:
    print()
    print(
        "=== DATALENS PREPARATION UI STATE v0.1 ==="
    )
    print()

    test_empty_state()

    test_update_and_invalidation()

    test_deep_copy_read_isolation()

    test_read_endpoint()

    print()
    print(
        "PASS - Preparation UI state v0.1"
    )


if __name__ == "__main__":
    main()
