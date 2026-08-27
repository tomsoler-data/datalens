from __future__ import annotations


from types import (
    SimpleNamespace,
)

from unittest.mock import (
    patch,
)


from fastapi.testclient import (
    TestClient,
)


from app.api import (
    analysis_run as analysis_run_module,
)

from app.document_summary import (
    DocumentSummaryCitation,
    VerifiedDocumentClaim,
)

from app.main import (
    app,
)

from app.planning.request_planner import (
    request_identifier,
)

from app.planning.schemas import (
    RequestedAnalysisPlan,
    RequestedAnalysisPlanReport,
)


# ============================================================
# CLIENT
# ============================================================

client = (
    TestClient(
        app
    )
)


# ============================================================
# ASSERT HELPERS
# ============================================================

def assert_equal(
    actual,
    expected,
    message: str,
) -> None:
    if (
        actual
        !=
        expected
    ):
        raise AssertionError(
            (
                f"{message}\n"
                f"Expected: {expected!r}\n"
                f"Actual:   {actual!r}"
            )
        )


def assert_true(
    value,
    message: str,
) -> None:
    if not value:
        raise AssertionError(
            message
        )


def pass_test(
    message: str,
) -> None:
    print(
        f"[PASS] {message}"
    )


# ============================================================
# CLAIM FIXTURE
# ============================================================

def make_claim(
    *,
    chunk_id: str,
    evidence_unit_id: int,
    statement: str,
) -> VerifiedDocumentClaim:
    return (
        VerifiedDocumentClaim(
            category=
                "analytical_request",

            statement=
                statement,

            evidence_quote=
                statement,

            evidence_unit_id=
                evidence_unit_id,

            context_quote=
                None,

            context_evidence_unit_id=
                None,

            citation=
                DocumentSummaryCitation(
                    chunk_id=
                        chunk_id,

                    document_id=
                        "document:http-test",

                    filename=
                        "brief.pdf",

                    source_locator=
                        "page 1",

                    page_number=
                        1,
                ),
        )
    )


# ============================================================
# PLAN FIXTURE
# ============================================================

def make_ready_plan(
    claim: VerifiedDocumentClaim,
) -> RequestedAnalysisPlan:
    return (
        RequestedAnalysisPlan(
            request_id=
                request_identifier(
                    claim
                ),

            request_text=
                claim.statement,

            context_text=
                claim.context_quote,

            evidence_quote=
                claim.evidence_quote,

            source_filename=
                claim.citation.filename,

            source_locator=
                claim.citation.source_locator,

            page_number=
                claim.citation.page_number,

            source_chunk_id=
                claim.citation.chunk_id,

            evidence_unit_id=
                claim.evidence_unit_id,

            kind=
                "transaction_count",

            status=
                "ready",

            target_family=
                "descriptive_metric",

            matched_columns=[],

            required_dataset_ids=[],

            required_dataset_filenames=[],

            required_operations=[
                (
                    "Compter les événements "
                    "transactionnels."
                )
            ],

            reasons=[],

            blockers=[],
        )
    )


# ============================================================
# FAIL-IF-REACHED
# ============================================================

def fail_if_preparation_reached(
    *args,
    **kwargs,
):
    raise AssertionError(
        (
            "prepare_analysis_datasets() was called "
            "even though request coverage was incomplete."
        )
    )


def fail_if_execution_reached(
    *args,
    **kwargs,
):
    raise AssertionError(
        (
            "execute_requested_analysis_plan() was called "
            "even though request coverage was incomplete."
        )
    )


# ============================================================
# TEST
# ============================================================

def test_incomplete_coverage_blocks_http_before_execution(
) -> None:
    """
    Simulate:

        2 verified documentary requests
                    ↓
        planner keeps only 1
                    ↓
        coverage = incomplete
                    ↓
        HTTP 409

    The test also proves that neither analytical dataset
    preparation nor Requested Analysis execution is reached.
    """

    first_claim = (
        make_claim(
            chunk_id=
                "chunk:http:001",

            evidence_unit_id=
                1,

            statement=
                "nombre de transactions",
        )
    )


    lost_claim = (
        make_claim(
            chunk_id=
                "chunk:http:001",

            evidence_unit_id=
                2,

            statement=
                "courbe de Lorenz",
        )
    )


    analytical_requests = [
        first_claim,
        lost_claim,
    ]


    incomplete_plan = (
        RequestedAnalysisPlanReport(
            request_count=
                1,

            ready_count=
                1,

            blocked_count=
                0,

            ambiguous_count=
                0,

            requests=[
                make_ready_plan(
                    first_claim
                )
            ],

            planner_notes=[],

            planner_rule_version=
                "analytical_request_planner_v0.2",
        )
    )


    fake_handoff = (
        SimpleNamespace(
            ingestion=
                object(),

            dataset_records=[],
        )
    )


    fake_document_summary = (
        SimpleNamespace(
            analytical_requests=
                analytical_requests,
        )
    )


    with (
        patch.object(
            analysis_run_module,
            "load_validated_analysis_input_for_http",
            return_value=
                fake_handoff,
        ),

        patch.object(
            analysis_run_module,
            "ingest_document_uploads",
            return_value=
                object(),
        ),

        patch.object(
            analysis_run_module,
            "summarize_document_ingestion",
            return_value=
                fake_document_summary,
        ),

        patch.object(
            analysis_run_module,
            "build_requested_analysis_plan",
            return_value=
                incomplete_plan,
        ),

        patch.object(
            analysis_run_module,
            "prepare_analysis_datasets",
            side_effect=
                fail_if_preparation_reached,
        ),

        patch.object(
            analysis_run_module,
            "execute_requested_analysis_plan",
            side_effect=
                fail_if_execution_reached,
        ),
    ):
        response = (
            client.post(
                "/analysis/run-contextualized",

                data={
                    "workflow_id":
                        "prep:http-coverage-test",
                },

                files=[
                    (
                        "dataset_files",

                        (
                            "ignored.csv",
                            b"value\n1\n",
                            "text/csv",
                        ),
                    ),

                    (
                        "document_files",

                        (
                            "brief.txt",
                            b"ignored",
                            "text/plain",
                        ),
                    ),
                ],
            )
        )


    # ========================================================
    # HTTP CONTRACT
    # ========================================================

    assert_equal(
        response.status_code,
        409,
        (
            "Incomplete request coverage must "
            "return HTTP 409."
        ),
    )


    payload = (
        response.json()
    )


    assert_true(
        "detail"
        in payload,
        (
            "HTTP 409 response must expose a "
            "structured detail object."
        ),
    )


    detail = (
        payload[
            "detail"
        ]
    )


    assert_equal(
        detail[
            "error"
        ],
        "analysis_request_coverage_incomplete",
        (
            "Unexpected HTTP request coverage "
            "error code."
        ),
    )


    # ========================================================
    # COVERAGE NUMBERS
    # ========================================================

    assert_equal(
        detail[
            "detected_count"
        ],
        2,
        (
            "Two verified documentary requests "
            "should have been detected."
        ),
    )


    assert_equal(
        detail[
            "planner_request_count"
        ],
        1,
        (
            "The simulated planner should expose "
            "one request."
        ),
    )


    assert_equal(
        detail[
            "planned_count"
        ],
        1,
        (
            "Exactly one request should have valid "
            "planner coverage."
        ),
    )


    assert_equal(
        detail[
            "lost_count"
        ],
        1,
        (
            "Exactly one request must be reported "
            "as lost."
        ),
    )


    assert_equal(
        detail[
            "coverage_rate"
        ],
        0.5,
        (
            "Coverage should be 50% in this "
            "controlled scenario."
        ),
    )


    assert_true(
        detail[
            "plan_accounting_valid"
        ],
        (
            "The planner report itself should be "
            "internally consistent in this test."
        ),
    )


    # ========================================================
    # LOST REQUEST ID
    # ========================================================

    expected_lost_request_id = (
        request_identifier(
            lost_claim
        )
    )


    assert_equal(
        detail[
            "lost_request_ids"
        ],
        [
            expected_lost_request_id
        ],
        (
            "The exact missing documentary request "
            "must be reported."
        ),
    )


    # ========================================================
    # NO OTHER INTEGRITY FAILURE
    # ========================================================

    assert_equal(
        detail[
            "provenance_mismatch_request_ids"
        ],
        [],
        (
            "This scenario should contain no "
            "provenance mismatch."
        ),
    )


    assert_equal(
        detail[
            "orphan_plan_request_ids"
        ],
        [],
        (
            "This scenario should contain no "
            "orphan planner request."
        ),
    )


    assert_equal(
        detail[
            "duplicate_detected_request_ids"
        ],
        [],
        (
            "This scenario should contain no "
            "duplicate detected request."
        ),
    )


    assert_equal(
        detail[
            "duplicate_planned_request_ids"
        ],
        [],
        (
            "This scenario should contain no "
            "duplicate planner request."
        ),
    )


    pass_test(
        (
            "incomplete request coverage returns "
            "HTTP 409 before analytical preparation"
        )
    )


    pass_test(
        (
            "Requested Analysis execution is not "
            "reached after coverage failure"
        )
    )


    pass_test(
        (
            "HTTP response exposes the exact lost "
            "documentary request"
        )
    )


# ============================================================
# ROUTE CONTRACT
# ============================================================

def test_contextualized_route_exposes_coverage_field(
) -> None:
    """
    Verify the FastAPI OpenAPI contract itself.

    This proves that successful contextualized responses now
    publicly expose request_coverage.
    """

    openapi = (
        app.openapi()
    )


    path = (
        openapi[
            "paths"
        ][
            "/analysis/run-contextualized"
        ][
            "post"
        ]
    )


    response_schema = (
        path[
            "responses"
        ][
            "200"
        ][
            "content"
        ][
            "application/json"
        ][
            "schema"
        ]
    )


    ref = (
        response_schema.get(
            "$ref"
        )
    )


    assert_true(
        isinstance(
            ref,
            str,
        ),
        (
            "Contextualized endpoint should expose "
            "a named response schema."
        ),
    )


    schema_name = (
        ref.rsplit(
            "/",
            1,
        )[
            -1
        ]
    )


    schema = (
        openapi[
            "components"
        ][
            "schemas"
        ][
            schema_name
        ]
    )


    properties = (
        schema.get(
            "properties",
            {}
        )
    )


    assert_true(
        "request_coverage"
        in properties,
        (
            "ContextualizedAnalysisResponse must "
            "expose request_coverage."
        ),
    )


    required = (
        schema.get(
            "required",
            []
        )
    )


    assert_true(
        "request_coverage"
        in required,
        (
            "request_coverage must be required in "
            "successful contextualized responses."
        ),
    )


    pass_test(
        (
            "contextualized HTTP contract exposes "
            "required request_coverage"
        )
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print(
        "=== DATALENS REQUEST COVERAGE HTTP GUARD v0.1 ==="
    )

    print()


    test_incomplete_coverage_blocks_http_before_execution()

    test_contextualized_route_exposes_coverage_field()


    print()

    print(
        "PASS - request coverage HTTP guard v0.1"
    )


if (
    __name__
    ==
    "__main__"
):
    main()