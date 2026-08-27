from __future__ import annotations

from io import StringIO

import pandas as pd

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


def fixture_dataframe() -> pd.DataFrame:
    return (
        pd.read_csv(
            StringIO(
                CSV_WITH_DUPLICATE
            )
        )
    )


# ============================================================
# SERVER-OWNED ARTIFACT HELPERS
# ============================================================


def put_session_artifact(
    *,
    workflow_id: str,
    dataset_id: str,
    dataset_filename: str,
    cleaning_executed: bool,
) -> None:
    """
    Materialize the dataset exactly where the semantic API now
    expects to find it: the server-owned Preparation Artifact
    Store.

    The dataframe intentionally retains the duplicate row.

    This is useful for the precondition test because the
    deterministic cleaning plan continues to contain an
    executable action. Therefore:

    - with cleaning_execution evidence -> semantic review may
      proceed;
    - without cleaning_execution evidence -> semantic review
      must return 409.

    The semantic API must not reconstruct this state from the
    browser-uploaded multipart file.
    """

    put_preparation_artifact(
        workflow_id=
            workflow_id,

        dataset_id=
            dataset_id,

        dataset_filename=
            dataset_filename,

        stage=(
            "clean"
            if cleaning_executed
            else
            "source"
        ),

        dataframe=
            fixture_dataframe(),

        evidence_refs=(
            [
                "cleaning_execution:cleaning_engine_test"
            ]
            if cleaning_executed
            else
            [
                "csv_ingestion"
            ]
        ),
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


    # ========================================================
    # SERVER-OWNED MATERIALIZATION
    # ========================================================
    #
    # Previous versions of this test stopped at the
    # PreparationSession.
    #
    # Semantic Review is now server-owned and therefore also
    # requires the exact materialized Preparation artifact.
    # ========================================================

    put_session_artifact(
        workflow_id=
            session.workflow_id,

        dataset_id=
            "dataset:0001",

        dataset_filename=
            "orders.csv",

        cleaning_executed=
            cleaning_executed,
    )


    return (
        get_preparation_session(
            session.workflow_id
        )
    )


def create_two_dataset_session():
    dataset_ids = [
        "dataset:0001",
        "dataset:0002",
    ]


    session = (
        create_preparation_session(
            selected_analysis_dataset_ids=
                dataset_ids
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

            dataset_ids=
                dataset_ids,

            evidence_refs=[
                evidence
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

        dataset_ids=
            dataset_ids,

        evidence_refs=[
            "cleaning_plan:cleaning_engine_test",
            "cleaning_execution:cleaning_engine_test",
        ],

        blocking_reasons=[
            (
                "Protected issues require "
                "semantic review."
            )
        ],
    )


    put_session_artifact(
        workflow_id=
            session.workflow_id,

        dataset_id=
            "dataset:0001",

        dataset_filename=
            "orders.csv",

        cleaning_executed=
            True,
    )


    put_session_artifact(
        workflow_id=
            session.workflow_id,

        dataset_id=
            "dataset:0002",

        dataset_filename=
            "customers.csv",

        cleaning_executed=
            True,
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
# SERVER-OWNED DATASET SCOPE
# ============================================================


def test_dataset_scope_is_server_owned():
    """
    Browser multipart files no longer define the semantic
    dataset scope.

    The PreparationSession selects two datasets and the
    Preparation Artifact Store owns both DataFrames.

    We deliberately send only one bogus browser file.

    Semantic Review must still receive both server-owned
    datasets.
    """

    session = (
        create_two_dataset_session()
    )


    observed_dataset_ids: set[
        str
    ] = set()


    original = (
        semantic_api
        .review_quality_semantics
    )


    def fake_scope_review(
        *,
        quality_report,
        dataset_frames,
        model,
    ) -> SemanticReviewReport:
        del quality_report
        del model


        observed_dataset_ids.update(
            dataset_frames.keys()
        )


        return (
            fake_semantic_review()
        )


    semantic_api.review_quality_semantics = (
        fake_scope_review
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

                # Intentionally does NOT correspond to the
                # server-owned Preparation scope.
                #
                # The bytes must not become analytical input.
                files=[
                    (
                        "dataset_files",
                        (
                            "tampered-browser-copy.csv",
                            (
                                "this,is,not,the,"
                                "server,owned,data\n"
                            ),
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
        "\n=== SERVER-OWNED DATASET SCOPE ==="
    )

    print(
        f"Status: "
        f"{response.status_code}"
    )

    print(
        (
            "Semantic dataset ids: "
            f"{sorted(observed_dataset_ids)}"
        )
    )


    assert (
        response.status_code
        ==
        200
    )


    assert (
        observed_dataset_ids
        ==
        {
            "dataset:0001",
            "dataset:0002",
        }
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

    reset_preparation_artifact_store_for_tests()


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

    test_dataset_scope_is_server_owned()

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