from __future__ import annotations

from fastapi import (
    FastAPI,
)

from fastapi.testclient import (
    TestClient,
)

from app.api.preparation_session import (
    PREPARATION_SESSION_API_VERSION,
    router,
)

from app.preparation.preparation_session import (
    PREPARATION_SESSION_RULE_VERSION,
    record_optional_stage_signal,
    record_required_stage_signal,
    record_validation_stage_signal,
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
# HELPERS
# ============================================================


def create_test_session(
    dataset_id: str = (
        "dataset:orders"
    ),
):
    response = (
        client.post(
            "/preparation/sessions",

            json={
                "selected_analysis_dataset_ids": [
                    dataset_id
                ]
            },
        )
    )

    assert (
        response.status_code
        ==
        201
    )

    return (
        response.json()
    )


def find_validation_error(
    body,
    field_name: str,
):
    details = (
        body.get(
            "detail",
            [],
        )
    )

    if not isinstance(
        details,
        list,
    ):
        return None

    for error in details:
        location = (
            error.get(
                "loc",
                [],
            )
        )

        if (
            location
            and
            location[
                -1
            ]
            ==
            field_name
        ):
            return (
                error
            )

    return None


# ============================================================
# CAPABILITIES
# ============================================================


def test_capabilities():
    response = (
        client.get(
            "/preparation/sessions/capabilities"
        )
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
        "\n=== SESSION CAPABILITIES ==="
    )

    print(
        f"API version: "
        f"{body['api_version']}"
    )

    print(
        f"Session version: "
        f"{body['session_version']}"
    )

    print(
        f"Storage: "
        f"{body['storage']}"
    )

    print(
        f"Persistent: "
        f"{body['persistent']}"
    )

    print(
        (
            "Client can update status: "
            f"{body['client_can_update_stage_status']}"
        )
    )

    assert (
        body[
            "api_version"
        ]
        ==
        PREPARATION_SESSION_API_VERSION
    )

    assert (
        body[
            "session_version"
        ]
        ==
        PREPARATION_SESSION_RULE_VERSION
    )

    assert (
        body[
            "storage"
        ]
        ==
        "sqlite"
    )

    assert (
        body[
            "persistent"
        ]
        is True
    )

    assert (
        body[
            "client_can_set_workflow_id"
        ]
        is False
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


# ============================================================
# CREATE
# ============================================================


def test_create_session():
    body = (
        create_test_session()
    )

    print(
        "\n=== CREATE SESSION ==="
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


# ============================================================
# READ
# ============================================================


def test_read_session():
    created = (
        create_test_session(
            "dataset:customers"
        )
    )

    workflow_id = (
        created[
            "workflow_id"
        ]
    )

    response = (
        client.get(
            (
                "/preparation/sessions/"
                f"{workflow_id}"
            )
        )
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
        "\n=== READ SESSION ==="
    )

    print(
        f"Workflow ID: "
        f"{body['workflow_id']}"
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
            "dataset:customers"
        ]
    )


# ============================================================
# CLIENT CANNOT CHOOSE WORKFLOW ID
# ============================================================


def test_client_workflow_id_injection_rejected():
    response = (
        client.post(
            "/preparation/sessions",

            json={
                "workflow_id":
                    "prep:client-controlled",

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

    error = (
        find_validation_error(
            body,
            "workflow_id",
        )
    )

    assert (
        error
        is not None
    )

    assert (
        error[
            "type"
        ]
        ==
        "extra_forbidden"
    )


# ============================================================
# NO PUBLIC STAGE UPDATE ROUTE
# ============================================================


def test_no_public_stage_mutation_endpoint():
    paths = (
        app.openapi()[
            "paths"
        ]
    )

    print(
        "\n=== PUBLIC SESSION SURFACE ==="
    )

    for path in sorted(
        paths.keys()
    ):
        print(
            path
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
# BACKEND-OWNED PROGRESSION
# ============================================================


def test_backend_owned_progression():
    dataset_id = (
        "dataset:orders"
    )

    created = (
        create_test_session(
            dataset_id
        )
    )

    workflow_id = (
        created[
            "workflow_id"
        ]
    )

    # ========================================================
    # IMPORT
    # ========================================================

    record_required_stage_signal(
        workflow_id=
            workflow_id,

        stage=
            PreparationStage.IMPORT,

        completed=
            True,

        dataset_ids=[
            dataset_id
        ],

        evidence_refs=[
            "csv_ingestion"
        ],

        blocking_reasons=[],
    )

    # ========================================================
    # UNDERSTAND
    # ========================================================

    record_required_stage_signal(
        workflow_id=
            workflow_id,

        stage=
            PreparationStage.UNDERSTAND,

        completed=
            True,

        dataset_ids=[
            dataset_id
        ],

        evidence_refs=[
            "dataset_profile"
        ],

        blocking_reasons=[],
    )

    # ========================================================
    # QUALITY
    # ========================================================

    record_required_stage_signal(
        workflow_id=
            workflow_id,

        stage=
            PreparationStage.QUALITY,

        completed=
            True,

        dataset_ids=[
            dataset_id
        ],

        evidence_refs=[
            "data_quality_engine_v0.2"
        ],

        blocking_reasons=[],
    )

    # ========================================================
    # VALIDATE
    # ========================================================

    record_validation_stage_signal(
        workflow_id=
            workflow_id,

        completed=
            True,

        passed=
            True,

        dataset_ids=[
            dataset_id
        ],

        evidence_refs=[
            "final_validation"
        ],

        blocking_reasons=[],
    )

    response = (
        client.get(
            (
                "/preparation/sessions/"
                f"{workflow_id}"
            )
        )
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
        "\n=== BACKEND-OWNED PROGRESSION ==="
    )

    print(
        f"Revision: "
        f"{body['revision']}"
    )

    print(
        (
            "Passed stages: "
            f"{body['snapshot']['passed_stage_count']}"
        )
    )

    print(
        (
            "Skipped stages: "
            f"{body['snapshot']['skipped_stage_count']}"
        )
    )

    print(
        (
            "Ready for analysis: "
            f"{body['snapshot']['ready_for_analysis']}"
        )
    )

    assert (
        body[
            "revision"
        ]
        ==
        4
    )

    assert (
        body[
            "snapshot"
        ][
            "passed_stage_count"
        ]
        ==
        4
    )

    assert (
        body[
            "snapshot"
        ][
            "skipped_stage_count"
        ]
        ==
        3
    )

    assert (
        body[
            "analysis_output_dataset_ids"
        ]
        ==
        []
    )

    assert (
        body[
            "snapshot"
        ][
            "analysis_output_dataset_ids"
        ]
        ==
        []
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
        "validate"
    )


# ============================================================
# BACKEND TRANSFORM REVIEW
# ============================================================


def test_backend_transform_review():
    dataset_id = (
        "dataset:orders-review"
    )

    created = (
        create_test_session(
            dataset_id
        )
    )

    workflow_id = (
        created[
            "workflow_id"
        ]
    )

    record_required_stage_signal(
        workflow_id=
            workflow_id,

        stage=
            PreparationStage.IMPORT,

        completed=
            True,

        dataset_ids=[
            dataset_id
        ],

        evidence_refs=[
            "csv_ingestion"
        ],

        blocking_reasons=[],
    )

    record_required_stage_signal(
        workflow_id=
            workflow_id,

        stage=
            PreparationStage.UNDERSTAND,

        completed=
            True,

        dataset_ids=[
            dataset_id
        ],

        evidence_refs=[
            "dataset_profile"
        ],

        blocking_reasons=[],
    )

    record_required_stage_signal(
        workflow_id=
            workflow_id,

        stage=
            PreparationStage.QUALITY,

        completed=
            True,

        dataset_ids=[
            dataset_id
        ],

        evidence_refs=[
            "data_quality_engine_v0.2"
        ],

        blocking_reasons=[],
    )

    record_optional_stage_signal(
        workflow_id=
            workflow_id,

        stage=
            PreparationStage.TRANSFORM,

        required=
            True,

        completed=
            False,

        review_required=
            True,

        blocked=
            False,

        dataset_ids=[
            dataset_id
        ],

        evidence_refs=[
            "transformation_planner_v0.1"
        ],

        blocking_reasons=[
            (
                "Age-band thresholds require "
                "analyst approval."
            )
        ],
    )

    response = (
        client.get(
            (
                "/preparation/sessions/"
                f"{workflow_id}"
            )
        )
    )

    assert (
        response.status_code
        ==
        200
    )

    body = (
        response.json()
    )

    snapshot = (
        body[
            "snapshot"
        ]
    )

    print(
        "\n=== BACKEND TRANSFORM REVIEW ==="
    )

    print(
        f"Next stage: "
        f"{snapshot['next_stage']}"
    )

    print(
        (
            "Ready for analysis: "
            f"{snapshot['ready_for_analysis']}"
        )
    )

    assert (
        snapshot[
            "next_stage"
        ]
        ==
        "transform"
    )

    assert (
        snapshot[
            "ready_for_analysis"
        ]
        is False
    )

    transform = next(
        stage

        for stage
        in snapshot[
            "stages"
        ]

        if (
            stage[
                "stage"
            ]
            ==
            "transform"
        )
    )

    assert (
        transform[
            "status"
        ]
        ==
        "review_required"
    )


# ============================================================
# NOT FOUND
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
        "DataLens Preparation Session v0.1"
    )

    print(
        "========================================"
    )

    test_capabilities()

    test_create_session()

    test_read_session()

    test_client_workflow_id_injection_rejected()

    test_no_public_stage_mutation_endpoint()

    test_backend_owned_progression()

    test_backend_transform_review()

    test_unknown_session_returns_404()

    print(
        "\n========================================"
    )

    print(
        "PASS - preparation session v0.1"
    )

    print(
        "========================================"
    )


if __name__ == "__main__":
    main()