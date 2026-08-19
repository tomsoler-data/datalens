from __future__ import annotations


from fastapi.testclient import (
    TestClient,
)


from main import (
    app,
)

from app.preparation.analysis_readiness_gate import (
    require_analysis_readiness,
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


# ============================================================
# CLIENT
# ============================================================


client = TestClient(
    app
)


# ============================================================
# HELPERS
# ============================================================


def create_quality_ready_session():
    session = (
        create_preparation_session(
            selected_analysis_dataset_ids=[
                "dataset:0001"
            ]
        )
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
                (
                    "test:"
                    f"{stage.value}"
                )
            ],

            blocking_reasons=[],
        )


    return (
        get_preparation_session(
            session.workflow_id
        )
    )


def explicitly_skip_clean(
    workflow_id: str,
):
    return (
        record_optional_stage_signal(
            workflow_id=
                workflow_id,

            stage=
                PreparationStage.CLEAN,

            required=
                False,

            completed=
                False,

            review_required=
                False,

            blocked=
                False,

            dataset_ids=[
                "dataset:0001"
            ],

            evidence_refs=[
                "cleaning_plan:test"
            ],

            blocking_reasons=[],
        )
    )


def pass_clean(
    workflow_id: str,
):
    return (
        record_optional_stage_signal(
            workflow_id=
                workflow_id,

            stage=
                PreparationStage.CLEAN,

            required=
                True,

            completed=
                True,

            review_required=
                False,

            blocked=
                False,

            dataset_ids=[
                "dataset:0001"
            ],

            evidence_refs=[
                "cleaning_plan:test",
                "cleaning_execution:test",
                "semantic_review:test",
                "semantic_confirmation:test",
            ],

            blocking_reasons=[],
        )
    )


def review_clean(
    workflow_id: str,
):
    return (
        record_optional_stage_signal(
            workflow_id=
                workflow_id,

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
                "semantic_review:test",
            ],

            blocking_reasons=[
                "Analyst review remains."
            ],
        )
    )


# ============================================================
# DEFAULT SKIP CANNOT BYPASS CLEANING
# ============================================================


def test_default_clean_skip_cannot_bypass_validation():
    session = (
        create_quality_ready_session()
    )


    response = (
        client.post(
            "/preparation/validate",

            json={
                "workflow_id":
                    session.workflow_id
            },
        )
    )


    print(
        "\n=== DEFAULT CLEAN BYPASS ATTEMPT ==="
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


    validation = (
        body[
            "detail"
        ][
            "validation"
        ]
    )


    print(
        (
            "Failed checks: "
            f"{validation['failed_check_count']}"
        )
    )


    assert any(
        check[
            "code"
        ]
        ==
        "cleaning_plan_evaluated"

        and
        check[
            "passed"
        ]
        is False

        for check
        in validation[
            "checks"
        ]
    )


    updated = (
        get_preparation_session(
            session.workflow_id
        )
    )


    validate_stage = next(
        stage

        for stage
        in updated.snapshot.stages

        if (
            stage.stage
            ==
            PreparationStage.VALIDATE
        )
    )


    print(
        f"Validate: "
        f"{validate_stage.status.value}"
    )


    assert (
        validate_stage.status.value
        ==
        "blocked"
    )

    assert (
        updated.snapshot.ready_for_analysis
        is False
    )


# ============================================================
# EXPLICIT CLEAN SKIP -> READY
# ============================================================


def test_explicit_clean_skip_can_validate():
    session = (
        create_quality_ready_session()
    )


    explicitly_skip_clean(
        session.workflow_id
    )


    response = (
        client.post(
            "/preparation/validate",

            json={
                "workflow_id":
                    session.workflow_id
            },
        )
    )


    print(
        "\n=== EXPLICIT CLEAN SKIP ==="
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


    snapshot = (
        body[
            "snapshot"
        ]
    )


    print(
        (
            "Validate: "
            f"{
                next(
                    stage['status']
                    for stage
                    in snapshot['stages']
                    if stage['stage'] == 'validate'
                )
            }"
        )
    )

    print(
        (
            "Ready for analysis: "
            f"{snapshot['ready_for_analysis']}"
        )
    )


    assert (
        snapshot[
            "ready_for_analysis"
        ]
        is True
    )

    assert (
        snapshot[
            "validated_analysis_dataset_ids"
        ]
        ==
        [
            "dataset:0001"
        ]
    )


    decision = (
        require_analysis_readiness(
            workflow_id=
                session.workflow_id,

            requested_analysis_dataset_ids=[
                "dataset:0001"
            ],
        )
    )


    print(
        (
            "Readiness gate: "
            f"{decision.ready_for_analysis}"
        )
    )


    assert (
        decision.ready_for_analysis
        is True
    )


# ============================================================
# CLEAN PASSED -> READY
# ============================================================


def test_clean_passed_can_validate():
    session = (
        create_quality_ready_session()
    )


    pass_clean(
        session.workflow_id
    )


    response = (
        client.post(
            "/preparation/validate",

            json={
                "workflow_id":
                    session.workflow_id
            },
        )
    )


    print(
        "\n=== CLEAN PASSED ==="
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
            f"{body['snapshot']['ready_for_analysis']}"
        )
    )


    assert (
        body[
            "snapshot"
        ][
            "ready_for_analysis"
        ]
        is True
    )


# ============================================================
# CLEAN REVIEW -> BLOCKED
# ============================================================


def test_clean_review_blocks_validation():
    session = (
        create_quality_ready_session()
    )


    review_clean(
        session.workflow_id
    )


    response = (
        client.post(
            "/preparation/validate",

            json={
                "workflow_id":
                    session.workflow_id
            },
        )
    )


    print(
        "\n=== CLEAN REVIEW ==="
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


    validation = (
        response.json()[
            "detail"
        ][
            "validation"
        ]
    )


    assert any(
        check[
            "code"
        ]
        ==
        "clean_stage_resolved"

        and
        check[
            "passed"
        ]
        is False

        for check
        in validation[
            "checks"
        ]
    )


# ============================================================
# CLIENT CANNOT INJECT PASSED
# ============================================================


def test_client_cannot_inject_validation_status():
    session = (
        create_quality_ready_session()
    )


    explicitly_skip_clean(
        session.workflow_id
    )


    response = (
        client.post(
            "/preparation/validate",

            json={
                "workflow_id":
                    session.workflow_id,

                "passed":
                    True,

                "ready_for_analysis":
                    True,
            },
        )
    )


    print(
        "\n=== VALIDATION STATUS INJECTION ==="
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
# UNKNOWN SESSION
# ============================================================


def test_unknown_session_returns_404():
    response = (
        client.post(
            "/preparation/validate",

            json={
                "workflow_id":
                    "prep:does-not-exist"
            },
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
# ROUTE
# ============================================================


def test_validation_route_registered():
    paths = (
        app.openapi()[
            "paths"
        ]
    )


    path = (
        "/preparation/validate"
    )


    print(
        "\n=== VALIDATION ROUTE ==="
    )

    print(
        f"{path}: "
        f"{path in paths}"
    )


    assert (
        path
        in paths
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
            "DataLens Final Preparation "
            "Validation v0.1"
        )
    )

    print(
        "========================================"
    )


    test_default_clean_skip_cannot_bypass_validation()

    test_explicit_clean_skip_can_validate()

    test_clean_passed_can_validate()

    test_clean_review_blocks_validation()

    test_client_cannot_inject_validation_status()

    test_unknown_session_returns_404()

    test_validation_route_registered()


    print(
        "\n========================================"
    )

    print(
        (
            "PASS - final preparation "
            "validation v0.1"
        )
    )

    print(
        "========================================"
    )


if __name__ == "__main__":
    main()