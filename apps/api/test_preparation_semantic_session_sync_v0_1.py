from __future__ import annotations

from fastapi import (
    FastAPI,
)

from fastapi.testclient import (
    TestClient,
)


import app.api.preparation_semantic as semantic_api


from app.preparation.data_quality import (
    QualityIssueKind,
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

from app.preparation.semantic_review import (
    SemanticReviewReport,
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
# FIXTURE
# ============================================================


CSV_WITH_DUPLICATE = (
    "category,value\n"
    "Premium,10\n"
    "Standard,20\n"
    "Standard,20\n"
    "Premium,30\n"
)


# ============================================================
# SESSION HELPERS
# ============================================================


def create_session(
    *,
    cleaning_executed: bool,
):
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


    evidence_refs = [
        "cleaning_plan:cleaning_engine_test"
    ]


    if (
        cleaning_executed
    ):
        evidence_refs.append(
            "cleaning_execution:cleaning_engine_test"
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

        evidence_refs=
            evidence_refs,

        blocking_reasons=[
            (
                "Protected issues require "
                "semantic review."
            )
        ],
    )


    return (
        get_preparation_session(
            session.workflow_id
        )
    )


def find_clean_stage(
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
# FAKE SEMANTIC REVIEW
# ============================================================


def fake_semantic_review(
    **_,
) -> SemanticReviewReport:
    decision = (
        ValidatedSemanticDecision(
            issue_id=
                "semantic:test:0001",

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
                (
                    "The observed categories should "
                    "remain unchanged."
                ),

            source_values=[],

            canonical_value=
                None,

            user_message=
                (
                    "Conserver les catégories "
                    "actuelles."
                ),

            python_validated=
                True,

            executable=
                False,

            requires_user_confirmation=
                True,

            validation_notes=[],
        )
    )


    return (
        SemanticReviewReport(
            status=
                "ready",

            model=
                "test-model",

            candidate_count=
                1,

            decision_count=
                1,

            merge_proposal_count=
                0,

            abstention_count=
                0,

            decisions=[
                decision
            ],

            notes=[
                "Synthetic semantic-review test."
            ],
        )
    )


# ============================================================
# REVIEW SYNCHRONIZES CLEAN
# ============================================================


def test_semantic_review_keeps_clean_in_review():
    session = (
        create_session(
            cleaning_executed=
                True
        )
    )


    original = (
        semantic_api
        .review_quality_semantics
    )


    semantic_api.review_quality_semantics = (
        fake_semantic_review
    )


    try:
        response = (
            client.post(
                "/preparation/semantic-review",

                data={
                    "workflow_id":
                        session.workflow_id,

                    "model":
                        "test-model",
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


    finally:
        semantic_api.review_quality_semantics = (
            original
        )


    print(
        "\n=== SEMANTIC REVIEW ==="
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


    updated = (
        get_preparation_session(
            session.workflow_id
        )
    )


    clean = (
        find_clean_stage(
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

    print(
        (
            "Semantic evidence: "
            f"{clean.evidence_refs}"
        )
    )


    assert (
        updated.revision
        ==
        5
    )

    assert (
        clean.status.value
        ==
        "review_required"
    )

    assert (
        updated.snapshot.next_stage
        ==
        PreparationStage.CLEAN
    )

    assert any(
        evidence.startswith(
            "semantic_review:"
        )

        for evidence
        in clean.evidence_refs
    )


# ============================================================
# SEMANTIC REVIEW CANNOT BYPASS DETERMINISTIC CLEANING
# ============================================================


def test_semantic_review_requires_deterministic_cleaning():
    session = (
        create_session(
            cleaning_executed=
                False
        )
    )


    response = (
        client.post(
            "/preparation/semantic-review",

            data={
                "workflow_id":
                    session.workflow_id,

                "model":
                    "test-model",
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
        "\n=== DETERMINISTIC CLEANING PRECONDITION ==="
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
        "deterministic_cleaning_required"
    )


# ============================================================
# UNKNOWN SESSION
# ============================================================


def test_unknown_session_returns_404():
    response = (
        client.post(
            "/preparation/semantic-review",

            data={
                "workflow_id":
                    "prep:does-not-exist",

                "model":
                    "test-model",
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
# DATASET SCOPE
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
            "/preparation/semantic-review",

            data={
                "workflow_id":
                    session.workflow_id,

                "model":
                    "test-model",
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
# WORKFLOW ID REQUIRED ON ALL SEMANTIC ROUTES
# ============================================================


def test_workflow_id_required():
    openapi = (
        app.openapi()
    )


    paths = (
        openapi[
            "paths"
        ]
    )


    expected = [
        "/preparation/semantic-review",
        "/preparation/semantic-cleaning-plan",
        "/preparation/semantic-cleaning-apply",
    ]


    print(
        "\n=== SEMANTIC SESSION CONTRACT ==="
    )


    for path in expected:
        assert (
            path
            in paths
        )

        print(
            f"{path}: True"
        )


    response = (
        client.post(
            "/preparation/semantic-review",

            data={
                "model":
                    "test-model",
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
        (
            "Missing workflow_id "
            f"status: {response.status_code}"
        )
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
            "DataLens Preparation Semantic "
            "Session Sync v0.1"
        )
    )

    print(
        "========================================"
    )


    test_semantic_review_keeps_clean_in_review()

    test_semantic_review_requires_deterministic_cleaning()

    test_unknown_session_returns_404()

    test_dataset_scope_mismatch_returns_409()

    test_workflow_id_required()


    print(
        "\n========================================"
    )

    print(
        (
            "PASS - preparation semantic "
            "session sync v0.1"
        )
    )

    print(
        "========================================"
    )


if __name__ == "__main__":
    main()