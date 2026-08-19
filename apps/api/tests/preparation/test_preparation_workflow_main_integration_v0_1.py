from __future__ import annotations

from fastapi.testclient import (
    TestClient,
)

from main import app


# ============================================================
# CLIENT
# ============================================================


client = TestClient(
    app
)


# ============================================================
# PAYLOAD
# ============================================================


def build_ready_payload():
    dataset_id = (
        "dataset:orders"
    )

    return {
        "workflow_id": (
            "workflow:main-integration"
        ),

        "selected_analysis_dataset_ids": [
            dataset_id
        ],

        "import_stage": {
            "completed":
                True,

            "dataset_ids": [
                dataset_id
            ],

            "evidence_refs": [
                "csv_ingestion"
            ],

            "blocking_reasons": [],
        },

        "understand_stage": {
            "completed":
                True,

            "dataset_ids": [
                dataset_id
            ],

            "evidence_refs": [
                "dataset_profile"
            ],

            "blocking_reasons": [],
        },

        "quality_stage": {
            "completed":
                True,

            "dataset_ids": [
                dataset_id
            ],

            "evidence_refs": [
                "data_quality_engine_v0.2"
            ],

            "blocking_reasons": [],
        },

        "clean_stage": {
            "required":
                False,

            "completed":
                False,

            "review_required":
                False,

            "blocked":
                False,

            "dataset_ids": [
                dataset_id
            ],

            "evidence_refs": [
                "no_cleaning_required"
            ],

            "blocking_reasons": [],
        },

        "transform_stage": {
            "required":
                False,

            "completed":
                False,

            "review_required":
                False,

            "blocked":
                False,

            "dataset_ids": [
                dataset_id
            ],

            "evidence_refs": [],

            "blocking_reasons": [],
        },

        "combine_stage": {
            "required":
                False,

            "completed":
                False,

            "review_required":
                False,

            "blocked":
                False,

            "dataset_ids": [
                dataset_id
            ],

            "evidence_refs": [],

            "blocking_reasons": [],
        },

        "validate_stage": {
            "completed":
                True,

            "passed":
                True,

            "dataset_ids": [
                dataset_id
            ],

            "evidence_refs": [
                "final_validation"
            ],

            "blocking_reasons": [],
        },
    }


# ============================================================
# OPENAPI ROUTES
# ============================================================


def test_routes_registered():
    openapi = (
        app.openapi()
    )

    paths = (
        openapi[
            "paths"
        ]
    )

    print(
        "\n=== REGISTERED WORKFLOW ROUTES ==="
    )

    for path in [
        (
            "/preparation/"
            "workflow/capabilities"
        ),
        (
            "/preparation/"
            "workflow/evaluate"
        ),
    ]:
        print(
            f"{path}: "
            f"{path in paths}"
        )

    assert (
        "/preparation/workflow/capabilities"
        in
        paths
    )

    assert (
        "/preparation/workflow/evaluate"
        in
        paths
    )

    capabilities_methods = (
        paths[
            "/preparation/workflow/capabilities"
        ]
    )

    evaluate_methods = (
        paths[
            "/preparation/workflow/evaluate"
        ]
    )

    assert (
        "get"
        in
        capabilities_methods
    )

    assert (
        "post"
        in
        evaluate_methods
    )


# ============================================================
# CAPABILITIES FROM REAL APP
# ============================================================


def test_capabilities():
    response = (
        client.get(
            "/preparation/workflow/capabilities"
        )
    )

    print(
        "\n=== MAIN APP CAPABILITIES ==="
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
        f"API version: "
        f"{body['api_version']}"
    )

    print(
        (
            "Unknown fields rejected: "
            f"{body['unknown_fields_are_rejected']}"
        )
    )

    assert (
        body[
            "client_can_set_stage_status_directly"
        ]
        is False
    )

    assert (
        body[
            "ready_for_analysis_is_computed"
        ]
        is True
    )

    assert (
        body[
            "unknown_fields_are_rejected"
        ]
        is True
    )


# ============================================================
# WORKFLOW FROM REAL APP
# ============================================================


def test_workflow_evaluation():
    response = (
        client.post(
            "/preparation/workflow/evaluate",

            json=
                build_ready_payload(),
        )
    )

    print(
        "\n=== MAIN APP WORKFLOW EVALUATION ==="
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
            "Ready for analysis: "
            f"{body['ready_for_analysis']}"
        )
    )

    print(
        f"Next stage: "
        f"{body['next_stage']}"
    )

    assert (
        body[
            "ready_for_analysis"
        ]
        is True
    )

    assert (
        body[
            "next_stage"
        ]
        is None
    )


# ============================================================
# SECURITY FROM REAL APP
# ============================================================


def test_status_injection_rejected():
    payload = (
        build_ready_payload()
    )

    payload[
        "transform_stage"
    ][
        "status"
    ] = (
        "passed"
    )

    response = (
        client.post(
            "/preparation/workflow/evaluate",

            json=
                payload,
        )
    )

    print(
        "\n=== MAIN APP STATUS INJECTION ==="
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
# EXISTING ANALYSIS ROUTES STILL REGISTERED
# ============================================================


def test_existing_analysis_routes_preserved():
    paths = (
        app.openapi()[
            "paths"
        ]
    )

    print(
        "\n=== EXISTING ANALYSIS ROUTES ==="
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
# MAIN
# ============================================================


def main():
    print(
        "\n========================================"
    )

    print(
        "DataLens Preparation Workflow "
        "Main Integration v0.1"
    )

    print(
        "========================================"
    )

    test_routes_registered()

    test_capabilities()

    test_workflow_evaluation()

    test_status_injection_rejected()

    test_existing_analysis_routes_preserved()

    print(
        "\n========================================"
    )

    print(
        "PASS - preparation workflow "
        "main integration v0.1"
    )

    print(
        "========================================"
    )


if __name__ == "__main__":
    main()