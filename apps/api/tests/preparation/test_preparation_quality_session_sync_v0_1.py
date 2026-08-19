from __future__ import annotations


from fastapi import (
    FastAPI,
)

from fastapi.testclient import (
    TestClient,
)


from app.api.preparation_quality import (
    router,
)

from app.preparation.preparation_session import (
    create_preparation_session,
    get_preparation_session,
    reset_preparation_session_store_for_tests,
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
# CSV
# ============================================================


CSV_CONTENT = (
    "order_id,amount,segment\n"
    "O001,100,Premium\n"
    "O002,75,Standard\n"
    "O003,120,Premium\n"
)


# ============================================================
# HELPERS
# ============================================================


def create_single_dataset_session():
    return (
        create_preparation_session(
            selected_analysis_dataset_ids=[
                "dataset:0001"
            ]
        )
    )


def find_stage(
    snapshot,
    stage_name: str,
):
    return next(
        stage

        for stage
        in snapshot.stages

        if (
            stage.stage.value
            ==
            stage_name
        )
    )


# ============================================================
# SESSION SYNCHRONIZATION
# ============================================================


def test_quality_advances_required_stages():
    created = (
        create_single_dataset_session()
    )


    response = (
        client.post(
            "/preparation/quality",

            data={
                "workflow_id":
                    created.workflow_id,
            },

            files=[
                (
                    "dataset_files",
                    (
                        "orders.csv",
                        CSV_CONTENT,
                        "text/csv",
                    ),
                )
            ],
        )
    )


    print(
        "\n=== QUALITY SESSION SYNC ==="
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


    session = (
        get_preparation_session(
            created.workflow_id
        )
    )


    snapshot = (
        session.snapshot
    )


    import_stage = (
        find_stage(
            snapshot,
            "import",
        )
    )

    understand_stage = (
        find_stage(
            snapshot,
            "understand",
        )
    )

    quality_stage = (
        find_stage(
            snapshot,
            "quality",
        )
    )


    print(
        f"Revision: "
        f"{session.revision}"
    )

    print(
        f"Import: "
        f"{import_stage.status.value}"
    )

    print(
        f"Understand: "
        f"{understand_stage.status.value}"
    )

    print(
        f"Quality: "
        f"{quality_stage.status.value}"
    )

    print(
        f"Next stage: "
        f"{snapshot.next_stage}"
    )

    print(
        (
            "Ready for analysis: "
            f"{snapshot.ready_for_analysis}"
        )
    )


    assert (
        session.revision
        ==
        3
    )

    assert (
        import_stage.status.value
        ==
        "passed"
    )

    assert (
        understand_stage.status.value
        ==
        "passed"
    )

    assert (
        quality_stage.status.value
        ==
        "passed"
    )

    assert (
        snapshot.ready_for_analysis
        is False
    )


# ============================================================
# UNKNOWN SESSION
# ============================================================


def test_unknown_session_returns_404():
    response = (
        client.post(
            "/preparation/quality",

            data={
                "workflow_id":
                    "prep:does-not-exist",
            },

            files=[
                (
                    "dataset_files",
                    (
                        "orders.csv",
                        CSV_CONTENT,
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
# DATASET SCOPE MISMATCH
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
            "/preparation/quality",

            data={
                "workflow_id":
                    session.workflow_id,
            },

            files=[
                (
                    "dataset_files",
                    (
                        "orders.csv",
                        CSV_CONTENT,
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
            "/preparation/quality",

            files=[
                (
                    "dataset_files",
                    (
                        "orders.csv",
                        CSV_CONTENT,
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
# MAIN
# ============================================================


def main():
    reset_preparation_session_store_for_tests()


    print(
        "\n========================================"
    )

    print(
        (
            "DataLens Preparation Quality "
            "Session Sync v0.1"
        )
    )

    print(
        "========================================"
    )


    test_quality_advances_required_stages()

    test_unknown_session_returns_404()

    test_dataset_scope_mismatch_returns_409()

    test_workflow_id_required()


    print(
        "\n========================================"
    )

    print(
        (
            "PASS - preparation quality "
            "session sync v0.1"
        )
    )

    print(
        "========================================"
    )


if __name__ == "__main__":
    main()