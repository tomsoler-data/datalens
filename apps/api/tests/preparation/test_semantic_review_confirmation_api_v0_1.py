from __future__ import annotations

import json

import pandas as pd

from fastapi import FastAPI

from fastapi.testclient import (
    TestClient,
)


import app.api.preparation_semantic as semantic_api


from app.preparation.data_quality import (
    QualityIssueKind,
)

from app.preparation.preparation_artifact_store import (
    put_preparation_artifact,
    reset_preparation_artifact_store_for_tests,
)

from app.preparation.preparation_session import (
    create_preparation_session,
    get_preparation_session,
    record_optional_stage_signal,
    record_required_stage_signal,
    reset_preparation_session_store_for_tests,
)

from app.preparation.preparation_workflow import (
    PreparationStage,
)

from app.preparation.semantic_cleaning import (
    SemanticCleaningAction,
    SemanticCleaningPlan,
)

from app.preparation.semantic_review import (
    SemanticVerdict,
    ValidatedSemanticDecision,
)


# ============================================================
# TEST APP
# ============================================================


app = FastAPI()

app.include_router(
    semantic_api.router
)


client = TestClient(
    app
)


# ============================================================
# SESSION
# ============================================================


def create_semantic_reviewed_session(
    *,
    artifact_frame: pd.DataFrame,
):
    session = (
        create_preparation_session(
            selected_analysis_dataset_ids=[
                "dataset:0001"
            ]
        )
    )


    put_preparation_artifact(
        workflow_id=
            session.workflow_id,

        dataset_id=
            "dataset:0001",

        dataset_filename=
            "orders.csv",

        stage=
            "source",

        dataframe=
            artifact_frame,

        parent_dataset_ids=[],

        evidence_refs=[
            "source:test",
        ],
    )


    for stage in [
        PreparationStage.IMPORT,
        PreparationStage.UNDERSTAND,
        PreparationStage.QUALITY,
    ]:
        record_required_stage_signal(
            workflow_id=
                session.workflow_id,

            stage=
                stage,

            completed=
                True,

            dataset_ids=[
                "dataset:0001"
            ],

            evidence_refs=[
                f"test:{stage.value}"
            ],

            blocking_reasons=[],
        )


    record_optional_stage_signal(
        workflow_id=
            session.workflow_id,

        stage=
            PreparationStage.CLEAN,

        required=
            True,

        completed=
            False,

        review_required=
            True,

        blocked=
            False,

        dataset_ids=[
            "dataset:0001"
        ],

        evidence_refs=[
            "cleaning_plan:test",
            "cleaning_execution:test",
            "semantic_review:semantic_review_v0.3",
        ],

        blocking_reasons=[
            "Semantic confirmation required."
        ],
    )


    return (
        get_preparation_session(
            session.workflow_id
        )
    )


def clean_stage(
    workflow_id: str,
):
    session = (
        get_preparation_session(
            workflow_id
        )
    )


    return next(
        stage

        for stage
        in session.snapshot.stages

        if (
            stage.stage
            ==
            PreparationStage.CLEAN
        )
    )


# ============================================================
# SYNTHETIC CONTEXT
# ============================================================


def no_change_frame() -> pd.DataFrame:
    return (
        pd.DataFrame(
            {
                "category": [
                    "Premium",
                    "Standard",
                ]
            }
        )
    )


def merge_frame() -> pd.DataFrame:
    return (
        pd.DataFrame(
            {
                "category": [
                    "Premium",
                    "PREMIUM",
                    "Standard",
                ]
            }
        )
    )


def no_change_decision():
    return (
        ValidatedSemanticDecision(
            issue_id=
                "issue:no-change",

            dataset_id=
                "dataset:0001",

            dataset_filename=
                "orders.csv",

            column=
                "category",

            kind=
                QualityIssueKind
                .POSSIBLE_SEMANTIC_ALIASES,

            verdict=
                SemanticVerdict.NO_CHANGE,

            confidence=
                0.95,

            rationale=
                "No semantic merge is needed.",

            source_values=[],

            canonical_value=
                None,

            user_message=
                "Keep current values.",

            python_validated=
                True,

            executable=
                False,

            requires_user_confirmation=
                True,

            validation_notes=[],
        )
    )


def merge_decision():
    return (
        ValidatedSemanticDecision(
            issue_id=
                "issue:merge",

            dataset_id=
                "dataset:0001",

            dataset_filename=
                "orders.csv",

            column=
                "category",

            kind=
                QualityIssueKind
                .POSSIBLE_SEMANTIC_ALIASES,

            verdict=
                SemanticVerdict.MERGE_VALUES,

            confidence=
                0.98,

            rationale=
                "Equivalent aliases.",

            source_values=[
                "Premium",
                "PREMIUM",
            ],

            canonical_value=
                "Premium",

            user_message=
                "Merge aliases.",

            python_validated=
                True,

            executable=
                False,

            requires_user_confirmation=
                True,

            validation_notes=[],
        )
    )


def empty_plan():
    return (
        SemanticCleaningPlan(
            status=
                "ready",

            action_count=
                0,

            actions=[],

            notes=[],
        )
    )


def merge_plan():
    return (
        SemanticCleaningPlan(
            status=
                "ready",

            action_count=
                1,

            actions=[
                SemanticCleaningAction(
                    action_id=
                        "semantic:dataset:0001:test",

                    issue_id=
                        "issue:merge",

                    dataset_id=
                        "dataset:0001",

                    dataset_filename=
                        "orders.csv",

                    column=
                        "category",

                    source_values=[
                        "Premium",
                        "PREMIUM",
                    ],

                    suggested_canonical_value=
                        "Premium",

                    allowed_canonical_values=[
                        "Premium",
                        "PREMIUM",
                    ],

                    confidence=
                        0.98,

                    rationale=
                        "Equivalent aliases.",

                    requires_user_confirmation=
                        True,

                    python_validated=
                        True,
                )
            ],

            notes=[],
        )
    )


def fake_no_change_context(
    **_,
):
    return (
        {
            "dataset:0001":
                no_change_frame()
        },

        [
            no_change_decision()
        ],

        empty_plan(),
    )


def fake_merge_context(
    **_,
):
    return (
        {
            "dataset:0001":
                merge_frame()
        },

        [
            merge_decision()
        ],

        merge_plan(),
    )


# ============================================================
# CONFIRM NO CHANGE
# ============================================================


def test_no_change_confirmation_passes_clean():
    session = (
        create_semantic_reviewed_session(
            artifact_frame=
                no_change_frame()
        )
    )


    original = (
        semantic_api
        ._rebuild_semantic_cleaning_context
    )


    semantic_api._rebuild_semantic_cleaning_context = (
        fake_no_change_context
    )


    try:
        response = (
            client.post(
                "/preparation/semantic-review-confirm",

                data={
                    "workflow_id":
                        session.workflow_id,

                    "semantic_decisions_json":
                        json.dumps(
                            [
                                {
                                    "issue_id":
                                        "ignored-by-test"
                                }
                            ]
                        ),

                    "confirmed_issue_ids_json":
                        json.dumps(
                            [
                                "issue:no-change"
                            ]
                        ),

                    "approved_semantic_choices_json":
                        "[]",
                },

                files=[
                    (
                        "dataset_files",
                        (
                            "orders.csv",
                            (
                                "category\n"
                                "Premium\n"
                            ),
                            "text/csv",
                        ),
                    )
                ],
            )
        )


    finally:
        semantic_api._rebuild_semantic_cleaning_context = (
            original
        )


    print(
        "\n=== NO CHANGE CONFIRMATION ==="
    )

    print(
        f"Status: "
        f"{response.status_code}"
    )


    assert (
        response.status_code
        ==
        200
    )


    body = (
        response.json()
    )


    print(
        (
            "Confirmed: "
            f"{body['confirmation']['confirmed']}"
        )
    )


    updated = (
        get_preparation_session(
            session.workflow_id
        )
    )


    clean = (
        clean_stage(
            session.workflow_id
        )
    )


    print(
        f"Revision: "
        f"{updated.revision}"
    )

    print(
        f"Clean: "
        f"{clean.status.value}"
    )

    print(
        f"Next stage: "
        f"{updated.snapshot.next_stage}"
    )


    assert (
        body[
            "confirmation"
        ][
            "confirmed"
        ]
        is True
    )

    assert (
        updated.revision
        ==
        5
    )

    assert (
        clean.status.value
        ==
        "passed"
    )

    assert (
        updated.snapshot.next_stage
        ==
        PreparationStage.VALIDATE
    )


# ============================================================
# MERGE MUST ACTUALLY BE APPLIED
# ============================================================


def test_merge_without_choice_remains_review_required():
    session = (
        create_semantic_reviewed_session(
            artifact_frame=
                merge_frame()
        )
    )


    original = (
        semantic_api
        ._rebuild_semantic_cleaning_context
    )


    semantic_api._rebuild_semantic_cleaning_context = (
        fake_merge_context
    )


    try:
        response = (
            client.post(
                "/preparation/semantic-review-confirm",

                data={
                    "workflow_id":
                        session.workflow_id,

                    "semantic_decisions_json":
                        "[]",

                    "confirmed_issue_ids_json":
                        json.dumps(
                            [
                                "issue:merge"
                            ]
                        ),

                    "approved_semantic_choices_json":
                        "[]",
                },

                files=[
                    (
                        "dataset_files",
                        (
                            "orders.csv",
                            (
                                "category\n"
                                "Premium\n"
                                "PREMIUM\n"
                            ),
                            "text/csv",
                        ),
                    )
                ],
            )
        )


    finally:
        semantic_api._rebuild_semantic_cleaning_context = (
            original
        )


    print(
        "\n=== MERGE WITHOUT CHOICE ==="
    )

    print(
        f"Status: "
        f"{response.status_code}"
    )


    assert (
        response.status_code
        ==
        409
    )


    body = (
        response.json()
    )


    assert (
        body[
            "detail"
        ][
            "error"
        ]
        ==
        "semantic_confirmation_incomplete"
    )


    clean = (
        clean_stage(
            session.workflow_id
        )
    )


    print(
        f"Clean: "
        f"{clean.status.value}"
    )


    assert (
        clean.status.value
        ==
        "review_required"
    )


# ============================================================
# MERGE WITH CHOICE
# ============================================================


def test_merge_with_choice_passes_clean():
    session = (
        create_semantic_reviewed_session(
            artifact_frame=
                merge_frame()
        )
    )


    original = (
        semantic_api
        ._rebuild_semantic_cleaning_context
    )


    semantic_api._rebuild_semantic_cleaning_context = (
        fake_merge_context
    )


    try:
        response = (
            client.post(
                "/preparation/semantic-review-confirm",

                data={
                    "workflow_id":
                        session.workflow_id,

                    "semantic_decisions_json":
                        "[]",

                    "confirmed_issue_ids_json":
                        json.dumps(
                            [
                                "issue:merge"
                            ]
                        ),

                    "approved_semantic_choices_json":
                        json.dumps(
                            [
                                {
                                    "action_id":
                                        (
                                            "semantic:"
                                            "dataset:0001:"
                                            "test"
                                        ),

                                    "canonical_value":
                                        "Premium",
                                }
                            ]
                        ),
                },

                files=[
                    (
                        "dataset_files",
                        (
                            "orders.csv",
                            (
                                "category\n"
                                "Premium\n"
                                "PREMIUM\n"
                            ),
                            "text/csv",
                        ),
                    )
                ],
            )
        )


    finally:
        semantic_api._rebuild_semantic_cleaning_context = (
            original
        )


    print(
        "\n=== MERGE WITH CHOICE ==="
    )

    print(
        f"Status: "
        f"{response.status_code}"
    )


    assert (
        response.status_code
        ==
        200
    )


    body = (
        response.json()
    )


    print(
        (
            "Applied semantic actions: "
            f"{body['execution']['applied_action_count']}"
        )
    )


    assert (
        body[
            "execution"
        ][
            "applied_action_count"
        ]
        ==
        1
    )


    clean = (
        clean_stage(
            session.workflow_id
        )
    )


    print(
        f"Clean: "
        f"{clean.status.value}"
    )


    assert (
        clean.status.value
        ==
        "passed"
    )


# ============================================================
# SEMANTIC REVIEW REQUIRED
# ============================================================


def test_confirmation_requires_semantic_review():
    session = (
        create_preparation_session(
            selected_analysis_dataset_ids=[
                "dataset:0001"
            ]
        )
    )


    response = (
        client.post(
            "/preparation/semantic-review-confirm",

            data={
                "workflow_id":
                    session.workflow_id,

                "semantic_decisions_json":
                    "[]",

                "confirmed_issue_ids_json":
                    "[]",

                "approved_semantic_choices_json":
                    "[]",
            },

            files=[
                (
                    "dataset_files",
                    (
                        "orders.csv",
                        "category\nPremium\n",
                        "text/csv",
                    ),
                )
            ],
        )
    )


    print(
        "\n=== REVIEW PRECONDITION ==="
    )

    print(
        f"Status: "
        f"{response.status_code}"
    )


    assert (
        response.status_code
        ==
        409
    )


    assert (
        response.json()[
            "detail"
        ][
            "error"
        ]
        ==
        "semantic_review_required"
    )


# ============================================================
# UNKNOWN SESSION
# ============================================================


def test_unknown_session_returns_404():
    response = (
        client.post(
            "/preparation/semantic-review-confirm",

            data={
                "workflow_id":
                    "prep:does-not-exist",

                "semantic_decisions_json":
                    "[]",

                "confirmed_issue_ids_json":
                    "[]",

                "approved_semantic_choices_json":
                    "[]",
            },

            files=[
                (
                    "dataset_files",
                    (
                        "orders.csv",
                        "category\nPremium\n",
                        "text/csv",
                    ),
                )
            ],
        )
    )


    print(
        "\n=== UNKNOWN SESSION ==="
    )

    print(
        f"Status: "
        f"{response.status_code}"
    )


    assert (
        response.status_code
        ==
        404
    )


# ============================================================
# OPENAPI
# ============================================================


def test_confirmation_route_registered():
    paths = (
        app.openapi()[
            "paths"
        ]
    )


    path = (
        "/preparation/"
        "semantic-review-confirm"
    )


    print(
        "\n=== CONFIRMATION ROUTE ==="
    )

    print(
        f"{path}: "
        f"{path in paths}"
    )


    assert (
        path
        in
        paths
    )


# ============================================================
# MAIN
# ============================================================


def main():
    reset_preparation_session_store_for_tests()

    reset_preparation_artifact_store_for_tests()


    print(
        "\n========================================"
    )

    print(
        (
            "DataLens Semantic Review "
            "Confirmation API v0.1"
        )
    )

    print(
        "========================================"
    )


    test_no_change_confirmation_passes_clean()

    test_merge_without_choice_remains_review_required()

    test_merge_with_choice_passes_clean()

    test_confirmation_requires_semantic_review()

    test_unknown_session_returns_404()

    test_confirmation_route_registered()


    print(
        "\n========================================"
    )

    print(
        (
            "PASS - semantic review "
            "confirmation API v0.1"
        )
    )

    print(
        "========================================"
    )


if __name__ == "__main__":
    main()