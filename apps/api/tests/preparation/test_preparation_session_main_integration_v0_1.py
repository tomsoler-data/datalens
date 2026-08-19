from __future__ import annotations

from fastapi.testclient import (
    TestClient,
)

from main import (
    app,
)

from app.preparation.preparation_session import (
    reset_preparation_session_store_for_tests,
)


# ============================================================
# CLIENT
# ============================================================


client = TestClient(
    app
)


# ============================================================
# CREATE SESSION
# ============================================================


def test_create_session_from_real_app():
    response = (
        client.post(
            "/preparation/sessions",

            json={
                "selected_analysis_dataset_ids": [
                    "dataset:orders"
                ]
            },
        )
    )

    print(
        "\n=== MAIN APP CREATE SESSION ==="
    )

    print(
        f"Status: "
        f"{response.status_code}"
    )

    assert (
        response.status_code
        ==
        201
    )

    body = (
        response.json()
    )

    print(
        f"Workflow ID: "
        f"{body['workflow_id']}"
    )

    print(
        f"Revision: "
        f"{body['revision']}"
    )

    print(
        (
            "Ready for analysis: "
            f"{body['snapshot']['ready_for_analysis']}"
        )
    )

    print(
        (
            "Next stage: "
            f"{body['snapshot']['next_stage']}"
        )
    )

    assert (
        body[
            "workflow_id"
        ].startswith(
            "prep:"
        )
    )

    assert (
        body[
            "revision"
        ]
        ==
        0
    )

    assert (
        body[
            "snapshot"
        ][
            "ready_for_analysis"
        ]
        is False
    )

    assert (
        body[
            "snapshot"
        ][
            "next_stage"
        ]
        ==
        "import"
    )

    return (
        body[
            "workflow_id"
        ]
    )


# ============================================================
# READ SESSION
# ============================================================


def test_read_session_from_real_app(
    workflow_id: str,
):
    response = (
        client.get(
            (
                "/preparation/sessions/"
                f"{workflow_id}"
            )
        )
    )

    print(
        "\n=== MAIN APP READ SESSION ==="
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

    assert (
        body[
            "workflow_id"
        ]
        ==
        workflow_id
    )

    assert (
        body[
            "selected_analysis_dataset_ids"
        ]
        ==
        [
            "dataset:orders"
        ]
    )

    assert (
        body[
            "snapshot"
        ][
            "ready_for_analysis"
        ]
        is False
    )


# ============================================================
# CAPABILITIES
# ============================================================


def test_session_capabilities():
    response = (
        client.get(
            "/preparation/sessions/capabilities"
        )
    )

    print(
        "\n=== SESSION CAPABILITIES ==="
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
            "Client can update status: "
            f"{body['client_can_update_stage_status']}"
        )
    )

    print(
        (
            "Client can set ready: "
            f"{body['client_can_set_ready_for_analysis']}"
        )
    )

    assert (
        body[
            "client_can_update_stage_status"
        ]
        is False
    )

    assert (
        body[
            "client_can_set_ready_for_analysis"
        ]
        is False
    )

    assert (
        body[
            "client_can_set_workflow_id"
        ]
        is False
    )


# ============================================================
# WORKFLOW ID INJECTION
# ============================================================


def test_workflow_id_injection_rejected():
    response = (
        client.post(
            "/preparation/sessions",

            json={
                "workflow_id":
                    "prep:browser-controlled",

                "selected_analysis_dataset_ids": [
                    "dataset:orders"
                ],
            },
        )
    )

    print(
        "\n=== WORKFLOW ID INJECTION ==="
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

    body = (
        response.json()
    )

    errors = (
        body[
            "detail"
        ]
    )

    assert (
        isinstance(
            errors,
            list,
        )
    )

    assert any(
        (
            error.get(
                "type"
            )
            ==
            "extra_forbidden"
        )

        for error
        in errors
    )


# ============================================================
# PUBLIC MUTATION SURFACE
# ============================================================


def test_no_public_session_mutation():
    paths = (
        app.openapi()[
            "paths"
        ]
    )

    session_path = (
        "/preparation/sessions/{workflow_id}"
    )

    assert (
        session_path
        in
        paths
    )

    methods = (
        paths[
            session_path
        ]
    )

    print(
        "\n=== SESSION PUBLIC METHODS ==="
    )

    print(
        sorted(
            methods.keys()
        )
    )

    assert (
        "get"
        in
        methods
    )

    assert (
        "post"
        not in
        methods
    )

    assert (
        "put"
        not in
        methods
    )

    assert (
        "patch"
        not in
        methods
    )

    assert (
        "delete"
        not in
        methods
    )


# ============================================================
# ROUTES
# ============================================================


def test_preparation_routes_registered():
    paths = (
        app.openapi()[
            "paths"
        ]
    )

    expected_paths = [
        (
            "/preparation/"
            "sessions"
        ),

        (
            "/preparation/"
            "sessions/capabilities"
        ),

        (
            "/preparation/"
            "sessions/{workflow_id}"
        ),

        (
            "/preparation/"
            "workflow/capabilities"
        ),

        (
            "/preparation/"
            "workflow/evaluate"
        ),
    ]

    print(
        "\n=== PREPARATION ROUTES ==="
    )

    for path in (
        expected_paths
    ):
        present = (
            path
            in
            paths
        )

        print(
            f"{path}: "
            f"{present}"
        )

        assert (
            present
            is True
        )


# ============================================================
# EXISTING ANALYSIS ROUTES
# ============================================================


def test_existing_analysis_routes_preserved():
    paths = (
        app.openapi()[
            "paths"
        ]
    )

    print(
        "\n=== ANALYSIS ROUTES PRESERVED ==="
    )

    print(
        (
            "/analysis/run: "
            f"{'/analysis/run' in paths}"
        )
    )

    print(
        (
            "/analysis/run-contextualized: "
            f"{'/analysis/run-contextualized' in paths}"
        )
    )

    assert (
        "/analysis/run"
        in
        paths
    )

    assert (
        "/analysis/run-contextualized"
        in
        paths
    )


# ============================================================
# UNKNOWN SESSION
# ============================================================


def test_unknown_session_returns_404():
    response = (
        client.get(
            (
                "/preparation/sessions/"
                "prep:does-not-exist"
            )
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
        "preparation_session_not_found"
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
            "DataLens Preparation Session "
            "Main Integration v0.1"
        )
    )

    print(
        "========================================"
    )

    test_session_capabilities()

    workflow_id = (
        test_create_session_from_real_app()
    )

    test_read_session_from_real_app(
        workflow_id
    )

    test_workflow_id_injection_rejected()

    test_no_public_session_mutation()

    test_preparation_routes_registered()

    test_existing_analysis_routes_preserved()

    test_unknown_session_returns_404()

    print(
        "\n========================================"
    )

    print(
        (
            "PASS - preparation session "
            "main integration v0.1"
        )
    )

    print(
        "========================================"
    )


if __name__ == "__main__":
    main()