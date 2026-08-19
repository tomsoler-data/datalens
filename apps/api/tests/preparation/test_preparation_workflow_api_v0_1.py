from __future__ import annotations

from fastapi import (
    FastAPI,
)

from fastapi.testclient import (
    TestClient,
)

from app.api.preparation_workflow import (
    PREPARATION_WORKFLOW_API_VERSION,
    router,
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
# PAYLOADS
# ============================================================


def build_simple_ready_payload():
    return {
        "workflow_id": (
            "workflow:api-simple"
        ),

        "selected_analysis_dataset_ids": [
            "dataset:orders"
        ],

        "import_stage": {
            "completed":
                True,

            "dataset_ids": [
                "dataset:orders"
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
                "dataset:orders"
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
                "dataset:orders"
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
                "dataset:orders"
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
                "dataset:orders"
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
                "dataset:orders"
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
                "dataset:orders"
            ],

            "evidence_refs": [
                "final_validation"
            ],

            "blocking_reasons": [],
        },
    }


def build_transform_review_payload():
    payload = (
        build_simple_ready_payload()
    )

    payload[
        "workflow_id"
    ] = (
        "workflow:api-transform-review"
    )

    payload[
        "transform_stage"
    ] = {
        "required":
            True,

        "completed":
            False,

        "review_required":
            True,

        "blocked":
            False,

        "dataset_ids": [
            "dataset:orders"
        ],

        "evidence_refs": [
            "transformation_planner_v0.1"
        ],

        "blocking_reasons": [
            (
                "Age-band thresholds "
                "require analyst approval."
            )
        ],
    }

    payload[
        "validate_stage"
    ] = {
        "completed":
            False,

        "passed":
            False,

        "dataset_ids": [],

        "evidence_refs": [],

        "blocking_reasons": [],
    }

    return (
        payload
    )


# ============================================================
# HELPERS
# ============================================================


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
            "/preparation/workflow/capabilities"
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
        "\n=== CAPABILITIES ==="
    )

    print(
        f"API version: "
        f"{body['api_version']}"
    )

    print(
        f"Stages: "
        f"{body['stages']}"
    )

    print(
        (
            "Client can set status directly: "
            f"{body['client_can_set_stage_status_directly']}"
        )
    )

    print(
        (
            "Ready for analysis computed: "
            f"{body['ready_for_analysis_is_computed']}"
        )
    )

    print(
        (
            "Unknown fields rejected: "
            f"{body['unknown_fields_are_rejected']}"
        )
    )

    assert (
        body[
            "api_version"
        ]
        ==
        "preparation_workflow_api_v0.1"
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

    assert (
        body[
            "stages"
        ]
        ==
        [
            "import",
            "understand",
            "quality",
            "clean",
            "transform",
            "combine",
            "validate",
        ]
    )


# ============================================================
# SIMPLE READY
# ============================================================


def test_simple_ready():
    response = (
        client.post(
            "/preparation/workflow/evaluate",

            json=(
                build_simple_ready_payload()
            ),
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
        "\n=== SIMPLE READY ==="
    )

    print(
        f"Ready for analysis: "
        f"{body['ready_for_analysis']}"
    )

    print(
        f"Passed stages: "
        f"{body['passed_stage_count']}"
    )

    print(
        f"Skipped stages: "
        f"{body['skipped_stage_count']}"
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
            "passed_stage_count"
        ]
        ==
        4
    )

    assert (
        body[
            "skipped_stage_count"
        ]
        ==
        3
    )

    assert (
        body[
            "next_stage"
        ]
        is None
    )

    assert (
        body[
            "blocking_reasons"
        ]
        ==
        []
    )


# ============================================================
# TRANSFORM REVIEW
# ============================================================


def test_transform_review():
    response = (
        client.post(
            "/preparation/workflow/evaluate",

            json=(
                build_transform_review_payload()
            ),
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
        "\n=== TRANSFORM REVIEW ==="
    )

    print(
        f"Ready for analysis: "
        f"{body['ready_for_analysis']}"
    )

    print(
        f"Next stage: "
        f"{body['next_stage']}"
    )

    print(
        "Blocking reasons:"
    )

    for reason in (
        body[
            "blocking_reasons"
        ]
    ):
        print(
            f"- {reason}"
        )

    assert (
        body[
            "ready_for_analysis"
        ]
        is False
    )

    assert (
        body[
            "next_stage"
        ]
        ==
        "transform"
    )

    transform = next(
        stage

        for stage
        in body[
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
# CONTRADICTORY SIGNAL
# ============================================================


def test_invalid_signal_returns_422():
    payload = (
        build_simple_ready_payload()
    )

    payload[
        "workflow_id"
    ] = (
        "workflow:api-invalid"
    )

    payload[
        "transform_stage"
    ] = {
        "required":
            True,

        "completed":
            True,

        "review_required":
            True,

        "blocked":
            False,

        "dataset_ids": [
            "dataset:orders"
        ],

        "evidence_refs": [],

        "blocking_reasons": [],
    }

    response = (
        client.post(
            "/preparation/workflow/evaluate",

            json=
                payload,
        )
    )

    print(
        "\n=== INVALID SIGNAL ==="
    )

    print(
        f"Status code: "
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

    assert (
        body[
            "detail"
        ][
            "error"
        ]
        ==
        "invalid_preparation_workflow"
    )


# ============================================================
# DIRECT STATUS MUST BE REJECTED
# ============================================================


def test_direct_status_rejected():
    payload = (
        build_transform_review_payload()
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
        "\n=== DIRECT STATUS ATTEMPT ==="
    )

    print(
        "Client requested: passed"
    )

    print(
        f"Status code: "
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
            "status",
        )
    )

    assert (
        error
        is not None
    )

    print(
        f"Validation type: "
        f"{error['type']}"
    )

    assert (
        error[
            "type"
        ]
        ==
        "extra_forbidden"
    )


# ============================================================
# READY FOR ANALYSIS INJECTION
# ============================================================


def test_ready_for_analysis_injection_rejected():
    payload = (
        build_simple_ready_payload()
    )

    payload[
        "ready_for_analysis"
    ] = (
        True
    )

    response = (
        client.post(
            "/preparation/workflow/evaluate",

            json=
                payload,
        )
    )

    print(
        "\n=== READY FOR ANALYSIS INJECTION ==="
    )

    print(
        "Client requested: ready_for_analysis=True"
    )

    print(
        f"Status code: "
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
            "ready_for_analysis",
        )
    )

    assert (
        error
        is not None
    )

    print(
        f"Validation type: "
        f"{error['type']}"
    )

    assert (
        error[
            "type"
        ]
        ==
        "extra_forbidden"
    )


# ============================================================
# REQUIRED STAGE STATUS INJECTION
# ============================================================


def test_required_stage_status_rejected():
    payload = (
        build_simple_ready_payload()
    )

    payload[
        "quality_stage"
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
        "\n=== QUALITY STATUS INJECTION ==="
    )

    print(
        f"Status code: "
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
            "status",
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
# UNKNOWN FIELD IN VALIDATION STAGE
# ============================================================


def test_validation_stage_unknown_field_rejected():
    payload = (
        build_simple_ready_payload()
    )

    payload[
        "validate_stage"
    ][
        "force_pass"
    ] = (
        True
    )

    response = (
        client.post(
            "/preparation/workflow/evaluate",

            json=
                payload,
        )
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
            "force_pass",
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
# MISSING REQUIRED PAYLOAD
# ============================================================


def test_missing_required_payload_returns_422():
    response = (
        client.post(
            "/preparation/workflow/evaluate",

            json={
                "workflow_id": (
                    "workflow:missing-fields"
                )
            },
        )
    )

    print(
        "\n=== MISSING REQUIRED PAYLOAD ==="
    )

    print(
        f"Status code: "
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
    print(
        "\n========================================"
    )

    print(
        "DataLens Preparation Workflow API v0.1"
    )

    print(
        "========================================"
    )

    test_capabilities()

    test_simple_ready()

    test_transform_review()

    test_invalid_signal_returns_422()

    test_direct_status_rejected()

    test_ready_for_analysis_injection_rejected()

    test_required_stage_status_rejected()

    test_validation_stage_unknown_field_rejected()

    test_missing_required_payload_returns_422()

    assert (
        PREPARATION_WORKFLOW_API_VERSION
        ==
        "preparation_workflow_api_v0.1"
    )

    print(
        "\n========================================"
    )

    print(
        "PASS - preparation workflow API v0.1"
    )

    print(
        "========================================"
    )


if __name__ == "__main__":
    main()