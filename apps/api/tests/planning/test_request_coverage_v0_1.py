from __future__ import annotations


from app.document_summary import (
    DocumentSummaryCitation,
    VerifiedDocumentClaim,
)

from app.planning.request_coverage import (
    REQUEST_COVERAGE_RULE_VERSION,
    AnalysisRequestCoverageError,
    build_analysis_request_coverage,
    require_complete_analysis_request_coverage,
)

from app.planning.request_planner import (
    request_identifier,
)

from app.planning.schemas import (
    RequestedAnalysisPlan,
    RequestedAnalysisPlanReport,
)


# ============================================================
# TEST HELPERS
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
# FIXTURES
# ============================================================

def make_claim(
    *,
    chunk_id: str,
    evidence_unit_id: int,
    statement: str,
    filename: str = "brief.pdf",
    source_locator: str = "page 1",
    page_number: int | None = 1,
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
                        "document:test",

                    filename=
                        filename,

                    source_locator=
                        source_locator,

                    page_number=
                        page_number,
                ),
        )
    )


def make_plan(
    *,
    claim: VerifiedDocumentClaim,
    status: str,
    kind: str = "transaction_count",
    target_family: str | None = "descriptive_metric",
    source_filename: str | None = None,
    request_id: str | None = None,
) -> RequestedAnalysisPlan:
    blockers = (
        []

        if (
            status
            ==
            "ready"
        )

        else
        [
            (
                "Demande conservée mais non "
                "exécutable automatiquement."
            )
        ]
    )


    return (
        RequestedAnalysisPlan(
            request_id=
                (
                    request_id

                    if (
                        request_id
                        is not None
                    )

                    else
                    request_identifier(
                        claim
                    )
                ),

            request_text=
                claim.statement,

            context_text=
                claim.context_quote,

            evidence_quote=
                claim.evidence_quote,

            source_filename=
                (
                    source_filename

                    if (
                        source_filename
                        is not None
                    )

                    else
                    claim.citation.filename
                ),

            source_locator=
                claim.citation.source_locator,

            page_number=
                claim.citation.page_number,

            source_chunk_id=
                claim.citation.chunk_id,

            evidence_unit_id=
                claim.evidence_unit_id,

            kind=
                kind,

            status=
                status,

            target_family=
                target_family,

            matched_columns=[],

            required_dataset_ids=[],

            required_dataset_filenames=[],

            required_operations=[],

            reasons=[],

            blockers=
                blockers,
        )
    )


def make_report(
    plans: list[
        RequestedAnalysisPlan
    ],
    *,
    override_request_count: int | None = None,
    override_ready_count: int | None = None,
    override_blocked_count: int | None = None,
    override_ambiguous_count: int | None = None,
) -> RequestedAnalysisPlanReport:
    ready_count = sum(
        1

        for plan
        in plans

        if (
            plan.status
            ==
            "ready"
        )
    )


    blocked_count = sum(
        1

        for plan
        in plans

        if (
            plan.status
            ==
            "blocked"
        )
    )


    ambiguous_count = sum(
        1

        for plan
        in plans

        if (
            plan.status
            ==
            "ambiguous"
        )
    )


    return (
        RequestedAnalysisPlanReport(
            request_count=
                (
                    len(
                        plans
                    )

                    if (
                        override_request_count
                        is None
                    )

                    else
                    override_request_count
                ),

            ready_count=
                (
                    ready_count

                    if (
                        override_ready_count
                        is None
                    )

                    else
                    override_ready_count
                ),

            blocked_count=
                (
                    blocked_count

                    if (
                        override_blocked_count
                        is None
                    )

                    else
                    override_blocked_count
                ),

            ambiguous_count=
                (
                    ambiguous_count

                    if (
                        override_ambiguous_count
                        is None
                    )

                    else
                    override_ambiguous_count
                ),

            requests=
                plans,

            planner_notes=[],

            planner_rule_version=
                "analytical_request_planner_v0.2",
        )
    )


# ============================================================
# TEST 1
# ============================================================

def test_complete_coverage() -> None:
    claims = [
        make_claim(
            chunk_id=
                "chunk:001",

            evidence_unit_id=1,

            statement=
                "nombre de transactions",
        ),

        make_claim(
            chunk_id=
                "chunk:001",

            evidence_unit_id=2,

            statement=
                "tops des références",
        ),

        make_claim(
            chunk_id=
                "chunk:002",

            evidence_unit_id=1,

            statement=
                "répartition du chiffre d'affaires BtoB",
        ),
    ]


    plans = [
        make_plan(
            claim=
                claims[
                    0
                ],

            status=
                "ready",
        ),

        make_plan(
            claim=
                claims[
                    1
                ],

            status=
                "ambiguous",

            kind=
                "top_products",

            target_family=
                "ranking",
        ),

        make_plan(
            claim=
                claims[
                    2
                ],

            status=
                "blocked",

            kind=
                "b2b_revenue_distribution",

            target_family=
                "aggregate_breakdown",
        ),
    ]


    report = (
        build_analysis_request_coverage(
            analytical_requests=
                claims,

            plan=
                make_report(
                    plans
                ),
        )
    )


    assert_equal(
        report.status,
        "complete",
        "Coverage should be complete.",
    )


    assert_equal(
        report.detected_count,
        3,
        "Three requests should be detected.",
    )


    assert_equal(
        report.planned_count,
        3,
        "All three requests should be planned.",
    )


    assert_equal(
        report.ready_count,
        1,
        "One request should be ready.",
    )


    assert_equal(
        report.blocked_count,
        1,
        "One request should be blocked.",
    )


    assert_equal(
        report.ambiguous_count,
        1,
        "One request should be ambiguous.",
    )


    assert_equal(
        report.lost_count,
        0,
        "No request should be lost.",
    )


    assert_equal(
        report.coverage_rate,
        1.0,
        "Coverage rate should be 100%.",
    )


    require_complete_analysis_request_coverage(
        report
    )


    pass_test(
        (
            "ready / blocked / ambiguous requests "
            "all count as preserved"
        )
    )


# ============================================================
# TEST 2
# ============================================================

def test_missing_request_is_detected() -> None:
    claims = [
        make_claim(
            chunk_id=
                "chunk:010",

            evidence_unit_id=1,

            statement=
                "nombre de transactions",
        ),

        make_claim(
            chunk_id=
                "chunk:010",

            evidence_unit_id=2,

            statement=
                "nombre de produits vendus",
        ),

        make_claim(
            chunk_id=
                "chunk:010",

            evidence_unit_id=3,

            statement=
                "courbe de Lorenz",
        ),
    ]


    plans = [
        make_plan(
            claim=
                claims[
                    0
                ],

            status=
                "ready",
        ),

        make_plan(
            claim=
                claims[
                    1
                ],

            status=
                "ready",

            kind=
                "products_sold_count",
        ),
    ]


    report = (
        build_analysis_request_coverage(
            analytical_requests=
                claims,

            plan=
                make_report(
                    plans
                ),
        )
    )


    expected_lost_id = (
        request_identifier(
            claims[
                2
            ]
        )
    )


    assert_equal(
        report.status,
        "incomplete",
        "Missing request must make coverage incomplete.",
    )


    assert_equal(
        report.detected_count,
        3,
        "Three requests should be detected.",
    )


    assert_equal(
        report.planned_count,
        2,
        "Only two requests should be covered.",
    )


    assert_equal(
        report.lost_count,
        1,
        "Exactly one request should be lost.",
    )


    assert_equal(
        report.lost_request_ids,
        [
            expected_lost_id
        ],
        "The missing request ID should be reported.",
    )


    try:
        require_complete_analysis_request_coverage(
            report
        )

    except AnalysisRequestCoverageError:
        pass


    else:
        raise AssertionError(
            (
                "Strict coverage guard should reject "
                "a lost request."
            )
        )


    pass_test(
        "a missing analytical request is detected"
    )


# ============================================================
# TEST 3
# ============================================================

def test_provenance_mismatch_is_detected() -> None:
    claim = (
        make_claim(
            chunk_id=
                "chunk:020",

            evidence_unit_id=1,

            statement=
                "nombre de transactions",

            filename=
                "brief.pdf",
        )
    )


    tampered_plan = (
        make_plan(
            claim=
                claim,

            status=
                "ready",

            source_filename=
                "tampered.pdf",
        )
    )


    report = (
        build_analysis_request_coverage(
            analytical_requests=[
                claim
            ],

            plan=
                make_report(
                    [
                        tampered_plan
                    ]
                ),
        )
    )


    expected_id = (
        request_identifier(
            claim
        )
    )


    assert_equal(
        report.status,
        "incomplete",
        (
            "A provenance mismatch must make "
            "coverage incomplete."
        ),
    )


    assert_equal(
        report.lost_count,
        1,
        (
            "A request with invalid provenance "
            "must not count as covered."
        ),
    )


    assert_equal(
        (
            report
            .provenance_mismatch_request_ids
        ),
        [
            expected_id
        ],
        (
            "The provenance mismatch must be "
            "explicitly reported."
        ),
    )


    pass_test(
        "tampered request provenance is rejected"
    )


# ============================================================
# TEST 4
# ============================================================

def test_orphan_plan_is_detected() -> None:
    source_claim = (
        make_claim(
            chunk_id=
                "chunk:030",

            evidence_unit_id=1,

            statement=
                "nombre de transactions",
        )
    )


    orphan_claim = (
        make_claim(
            chunk_id=
                "chunk:999",

            evidence_unit_id=99,

            statement=
                "demande inventée",
        )
    )


    source_plan = (
        make_plan(
            claim=
                source_claim,

            status=
                "ready",
        )
    )


    orphan_plan = (
        make_plan(
            claim=
                orphan_claim,

            status=
                "ambiguous",

            kind=
                "unknown",

            target_family=
                None,
        )
    )


    report = (
        build_analysis_request_coverage(
            analytical_requests=[
                source_claim
            ],

            plan=
                make_report(
                    [
                        source_plan,
                        orphan_plan,
                    ]
                ),
        )
    )


    assert_equal(
        report.status,
        "incomplete",
        (
            "An orphan planner request must make "
            "coverage incomplete."
        ),
    )


    assert_equal(
        report.lost_count,
        0,
        (
            "The genuine source request is still "
            "covered."
        ),
    )


    assert_equal(
        report.orphan_plan_request_ids,
        [
            request_identifier(
                orphan_claim
            )
        ],
        (
            "The invented/orphan planner request "
            "must be reported."
        ),
    )


    pass_test(
        "planner cannot introduce an orphan request"
    )


# ============================================================
# TEST 5
# ============================================================

def test_duplicate_plan_is_detected() -> None:
    claim = (
        make_claim(
            chunk_id=
                "chunk:040",

            evidence_unit_id=1,

            statement=
                "nombre de transactions",
        )
    )


    plan = (
        make_plan(
            claim=
                claim,

            status=
                "ready",
        )
    )


    duplicate = (
        plan.model_copy(
            deep=True
        )
    )


    report = (
        build_analysis_request_coverage(
            analytical_requests=[
                claim
            ],

            plan=
                make_report(
                    [
                        plan,
                        duplicate,
                    ]
                ),
        )
    )


    request_id = (
        request_identifier(
            claim
        )
    )


    assert_equal(
        report.status,
        "incomplete",
        (
            "Duplicate planner entries must make "
            "coverage incomplete."
        ),
    )


    assert_equal(
        report.planned_count,
        1,
        (
            "The source request is present, but "
            "its duplication is an integrity error."
        ),
    )


    assert_equal(
        report.duplicate_planned_request_ids,
        [
            request_id
        ],
        (
            "The duplicated request ID must be "
            "reported."
        ),
    )


    assert_equal(
        report.items[
            0
        ].coverage_state,
        "invalid",
        (
            "Duplicated coverage must not be shown "
            "as a clean planned state."
        ),
    )


    pass_test(
        "duplicate planner request is detected"
    )


# ============================================================
# TEST 6
# ============================================================

def test_planner_accounting_is_verified() -> None:
    claim = (
        make_claim(
            chunk_id=
                "chunk:050",

            evidence_unit_id=1,

            statement=
                "nombre de transactions",
        )
    )


    plan = (
        make_plan(
            claim=
                claim,

            status=
                "ready",
        )
    )


    malformed_report = (
        make_report(
            [
                plan
            ],

            override_request_count=2,
        )
    )


    report = (
        build_analysis_request_coverage(
            analytical_requests=[
                claim
            ],

            plan=
                malformed_report,
        )
    )


    assert_equal(
        report.status,
        "incomplete",
        (
            "Invalid planner counters must make "
            "coverage incomplete."
        ),
    )


    assert_true(
        not report.plan_accounting_valid,
        (
            "Planner accounting should be marked "
            "invalid."
        ),
    )


    assert_equal(
        report.lost_count,
        0,
        (
            "The request itself remains present; "
            "the failure is report integrity."
        ),
    )


    pass_test(
        "planner public counters are reconciled"
    )


# ============================================================
# VERSION
# ============================================================

def test_version() -> None:
    assert_equal(
        REQUEST_COVERAGE_RULE_VERSION,
        "analysis_request_coverage_v0.1",
        "Unexpected coverage rule version.",
    )


    pass_test(
        "request coverage rule version"
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print(
        "=== DATALENS ANALYSIS REQUEST COVERAGE v0.1 ==="
    )

    print()


    test_complete_coverage()

    test_missing_request_is_detected()

    test_provenance_mismatch_is_detected()

    test_orphan_plan_is_detected()

    test_duplicate_plan_is_detected()

    test_planner_accounting_is_verified()

    test_version()


    print()

    print(
        "PASS - analysis request coverage v0.1"
    )


if (
    __name__
    ==
    "__main__"
):
    main()