from __future__ import annotations

import json


from fastapi import (
    FastAPI,
)

from fastapi.testclient import (
    TestClient,
)


from app.api.preparation_cleaning import (
    router,
)

from app.preparation.preparation_session import (
    create_preparation_session,
    get_preparation_session,
    record_required_stage_signal,
    reset_preparation_session_store_for_tests,
)

from app.preparation.preparation_workflow import (
    PreparationStage,
)


# ============================================================
# TEST APP
# ============================================================


app = FastAPI()

app.include_router(
    router
)


client = TestClient(
    app
)


# ============================================================
# FIXTURES
# ============================================================


CSV_WITH_DUPLICATE = (
    "value,category\n"
    "10,A\n"
    "20,B\n"
    "20,B\n"
    "30,C\n"
)


CSV_CLEAN = (
    "value,category\n"
    "10,A\n"
    "20,B\n"
    "30,C\n"
    "40,D\n"
)


# ============================================================
# SESSION HELPERS
# ============================================================


def create_quality_ready_session():
    session = (
        create_preparation_session(
            selected_analysis_dataset_ids=[
                "dataset:0001"
            ]
        )
    )


    for stage, evidence in [
        (
            PreparationStage.IMPORT,
            "csv_ingestion",
        ),
        (
            PreparationStage.UNDERSTAND,
            "dataset_profile",
        ),
        (
            PreparationStage.QUALITY,
            "data_quality_engine",
        ),
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
                evidence
            ],

            blocking_reasons=[],
        )


    return (
        get_preparation_session(
            session.workflow_id
        )
    )


def find_stage(
    session,
    stage_name: str,
):
    return next(
        stage

        for stage
        in session.snapshot.stages

        if (
            stage.stage.value
            ==
            stage_name
        )
    )


# ============================================================
# PLAN -> REVIEW REQUIRED
# ============================================================


def test_cleaning_plan_requires_review():
    session = (
        create_quality_ready_session()
    )


    response = (
        client.post(
            "/preparation/cleaning-plan",

            data={
                "workflow_id":
                    session.workflow_id,
            },

            files=[
                (
                    "dataset_files",
                    (
                        "orders.csv",
                        CSV_WITH_DUPLICATE,
                        "text/csv",
                    ),
                )
            ],
        )
    )


    print(
        "\n=== CLEANING PLAN ==="
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


    plan = (
        response.json()
    )


    print(
        f"Actions: "
        f"{plan['action_count']}"
    )

    print(
        (
            "Blocked issues: "
            f"{plan['protected_issue_count']}"
        )
    )


    assert (
        plan[
            "action_count"
        ]
        >
        0
    )


    updated = (
        get_preparation_session(
            session.workflow_id
        )
    )


    clean_stage = (
        find_stage(
            updated,
            "clean",
        )
    )


    print(
        f"Revision: "
        f"{updated.revision}"
    )

    print(
        f"Clean: "
        f"{clean_stage.status.value}"
    )

    print(
        f"Next stage: "
        f"{updated.snapshot.next_stage}"
    )


    assert (
        updated.revision
        ==
        4
    )

    assert (
        clean_stage.status.value
        ==
        "review_required"
    )

    assert (
        updated.snapshot.next_stage
        ==
        PreparationStage.CLEAN
    )


# ============================================================
# APPLY -> PASSED WHEN FULLY RESOLVED
# ============================================================


def test_cleaning_apply_can_pass_clean_stage():
    session = (
        create_quality_ready_session()
    )


    plan_response = (
        client.post(
            "/preparation/cleaning-plan",

            data={
                "workflow_id":
                    session.workflow_id,
            },

            files=[
                (
                    "dataset_files",
                    (
                        "orders.csv",
                        CSV_WITH_DUPLICATE,
                        "text/csv",
                    ),
                )
            ],
        )
    )


    assert (
        plan_response.status_code
        ==
        200
    )


    plan = (
        plan_response.json()
    )


    assert (
        plan[
            "action_count"
        ]
        >
        0
    )


    assert (
        plan[
            "protected_issue_count"
        ]
        ==
        0
    )


    approved_ids = [
        action[
            "action_id"
        ]

        for action
        in plan[
            "actions"
        ]

        if (
            action[
                "safe_candidate"
            ]
        )
    ]


    assert (
        len(
            approved_ids
        )
        >
        0
    )


    response = (
        client.post(
            "/preparation/cleaning-apply",

            data={
                "workflow_id":
                    session.workflow_id,

                "approved_action_ids_json":
                    json.dumps(
                        approved_ids
                    ),
            },

            files=[
                (
                    "dataset_files",
                    (
                        "orders.csv",
                        CSV_WITH_DUPLICATE,
                        "text/csv",
                    ),
                )
            ],
        )
    )


    print(
        "\n=== CLEANING APPLY ==="
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
            "Applied actions: "
            f"{body['execution']['applied_action_count']}"
        )
    )

    print(
        (
            "Skipped actions: "
            f"{body['execution']['skipped_action_count']}"
        )
    )

    print(
        (
            "Blocked actions: "
            f"{body['execution']['blocked_action_count']}"
        )
    )


    updated = (
        get_preparation_session(
            session.workflow_id
        )
    )


    clean_stage = (
        find_stage(
            updated,
            "clean",
        )
    )


    print(
        f"Revision: "
        f"{updated.revision}"
    )

    print(
        f"Clean: "
        f"{clean_stage.status.value}"
    )

    print(
        f"Next stage: "
        f"{updated.snapshot.next_stage}"
    )


    assert (
        updated.revision
        ==
        5
    )

    assert (
        clean_stage.status.value
        ==
        "passed"
    )

    assert (
        updated.snapshot.ready_for_analysis
        is False
    )

    assert (
        updated.snapshot.next_stage
        ==
        PreparationStage.VALIDATE
    )


# ============================================================
# NO CLEANING -> SKIPPED
# ============================================================


def test_cleaning_plan_can_skip_clean_stage():
    session = (
        create_quality_ready_session()
    )


    response = (
        client.post(
            "/preparation/cleaning-plan",

            data={
                "workflow_id":
                    session.workflow_id,
            },

            files=[
                (
                    "dataset_files",
                    (
                        "clean.csv",
                        CSV_CLEAN,
                        "text/csv",
                    ),
                )
            ],
        )
    )


    print(
        "\n=== NO CLEANING REQUIRED ==="
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


    plan = (
        response.json()
    )


    print(
        f"Actions: "
        f"{plan['action_count']}"
    )

    print(
        (
            "Blocked issues: "
            f"{plan['protected_issue_count']}"
        )
    )


    assert (
        plan[
            "action_count"
        ]
        ==
        0
    )

    assert (
        plan[
            "protected_issue_count"
        ]
        ==
        0
    )


    updated = (
        get_preparation_session(
            session.workflow_id
        )
    )


    clean_stage = (
        find_stage(
            updated,
            "clean",
        )
    )


    print(
        f"Clean: "
        f"{clean_stage.status.value}"
    )

    print(
        f"Next stage: "
        f"{updated.snapshot.next_stage}"
    )


    assert (
        clean_stage.status.value
        ==
        "skipped"
    )

    assert (
        updated.snapshot.next_stage
        ==
        PreparationStage.VALIDATE
    )


# ============================================================
# UNKNOWN SESSION
# ============================================================


def test_unknown_session_returns_404():
    response = (
        client.post(
            "/preparation/cleaning-plan",

            data={
                "workflow_id":
                    "prep:does-not-exist",
            },

            files=[
                (
                    "dataset_files",
                    (
                        "orders.csv",
                        CSV_WITH_DUPLICATE,
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
# SCOPE MISMATCH
# ============================================================


def test_dataset_scope_mismatch_returns_409():
    session = (
        create_preparation_session(
            selected_analysis_dataset_ids=[
                "dataset:0001",
                "dataset:0002",
            ]
        )
    )


    response = (
        client.post(
            "/preparation/cleaning-plan",

            data={
                "workflow_id":
                    session.workflow_id,
            },

            files=[
                (
                    "dataset_files",
                    (
                        "orders.csv",
                        CSV_WITH_DUPLICATE,
                        "text/csv",
                    ),
                )
            ],
        )
    )


    print(
        "\n=== DATASET SCOPE MISMATCH ==="
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
        "preparation_dataset_scope_mismatch"
    )


# ============================================================
# WORKFLOW ID REQUIRED
# ============================================================


def test_workflow_id_required():
    response = (
        client.post(
            "/preparation/cleaning-plan",

            files=[
                (
                    "dataset_files",
                    (
                        "orders.csv",
                        CSV_WITH_DUPLICATE,
                        "text/csv",
                    ),
                )
            ],
        )
    )


    print(
        "\n=== WORKFLOW ID REQUIRED ==="
    )

    print(
        f"Status: "
        f"{response.status_code}"
    )


    assert (
        response.status_code
        ==
        422
    )


# ============================================================
# EMPTY APPROVAL CANNOT RESOLVE A REAL PLAN
# ============================================================


def test_empty_approval_rejected():
    session = (
        create_quality_ready_session()
    )


    response = (
        client.post(
            "/preparation/cleaning-apply",

            data={
                "workflow_id":
                    session.workflow_id,

                "approved_action_ids_json":
                    "[]",
            },

            files=[
                (
                    "dataset_files",
                    (
                        "orders.csv",
                        CSV_WITH_DUPLICATE,
                        "text/csv",
                    ),
                )
            ],
        )
    )


    print(
        "\n=== EMPTY APPROVAL ==="
    )

    print(
        f"Status: "
        f"{response.status_code}"
    )


    assert (
        response.status_code
        ==
        422
    )


# ============================================================
# MAIN
# ============================================================


def main():
    reset_preparation_session_store_for_tests()


    print(
        "\n========================================"
    )

    print(
        (
            "DataLens Preparation Cleaning "
            "Session Sync v0.1"
        )
    )

    print(
        "========================================"
    )


    test_cleaning_plan_requires_review()

    test_cleaning_apply_can_pass_clean_stage()

    test_cleaning_plan_can_skip_clean_stage()

    test_unknown_session_returns_404()

    test_dataset_scope_mismatch_returns_409()

    test_workflow_id_required()

    test_empty_approval_rejected()


    print(
        "\n========================================"
    )

    print(
        (
            "PASS - preparation cleaning "
            "session sync v0.1"
        )
    )

    print(
        "========================================"
    )


if __name__ == "__main__":
    main()
