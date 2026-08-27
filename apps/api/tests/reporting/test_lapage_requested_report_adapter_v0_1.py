from __future__ import annotations


from collections import (
    Counter,
)


from app.reporting.requested_adapter import (
    REPORTABLE_REQUESTED_STATUSES,
    build_requested_report_findings,
)

from tests.execution.test_lapage_requested_real_execution_v0_1 import (
    READY_KINDS,
    NON_READY_KINDS,
    build_real_execution_report,
)


# ============================================================
# ASSERTIONS
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
# BUILD
# ============================================================

def build_lapage_requested_findings():
    (
        plan,
        _,
        execution_report,
    ) = build_real_execution_report()


    findings = (
        build_requested_report_findings(
            execution_report=
                execution_report,

            plan_report=
                plan,
        )
    )


    return (
        plan,
        execution_report,
        findings,
    )


# ============================================================
# TEST 1
# ============================================================

def test_report_adapter_produces_11_findings(
) -> None:
    (
        _,
        execution_report,
        findings,
    ) = build_lapage_requested_findings()


    assert_equal(
        execution_report.request_count,
        15,
        (
            "Execution report must contain "
            "15 requests before adaptation."
        ),
    )


    assert_equal(
        len(
            findings
        ),
        11,
        (
            "The report adapter should produce "
            "one finding for each complete or "
            "descriptive_only requested analysis."
        ),
    )


    pass_test(
        (
            "report adapter produces 11 reportable "
            "findings from 15 requests"
        )
    )


# ============================================================
# TEST 2
# ============================================================

def test_all_ready_kinds_become_findings(
) -> None:
    (
        _,
        _,
        findings,
    ) = build_lapage_requested_findings()


    finding_kinds = {
        finding.kind

        for finding
        in findings
    }


    assert_equal(
        finding_kinds,
        READY_KINDS,
        (
            "A ready Lapage analysis disappeared "
            "during requested-report adaptation."
        ),
    )


    pass_test(
        (
            "all 11 ready Lapage analysis kinds "
            "reach requested_findings"
        )
    )


# ============================================================
# TEST 3
# ============================================================

def test_reportable_statuses_only(
) -> None:
    (
        _,
        _,
        findings,
    ) = build_lapage_requested_findings()


    statuses = {
        finding.execution_status

        for finding
        in findings
    }


    assert_equal(
        statuses,
        REPORTABLE_REQUESTED_STATUSES,
        (
            "Unexpected execution status entered "
            "requested_findings."
        ),
    )


    counts = (
        Counter(
            finding.execution_status

            for finding
            in findings
        )
    )


    assert_equal(
        counts[
            "complete"
        ],
        9,
        (
            "Expected 9 complete requested "
            "findings."
        ),
    )


    assert_equal(
        counts[
            "descriptive_only"
        ],
        2,
        (
            "Expected 2 descriptive-only "
            "requested findings."
        ),
    )


    pass_test(
        (
            "requested findings preserve "
            "9 complete + 2 descriptive_only"
        )
    )


# ============================================================
# TEST 4
# ============================================================

def test_non_ready_requests_are_not_misrepresented_as_findings(
) -> None:
    (
        _,
        _,
        findings,
    ) = build_lapage_requested_findings()


    finding_kinds = {
        finding.kind

        for finding
        in findings
    }


    incorrectly_reported = (
        finding_kinds
        &
        NON_READY_KINDS
    )


    assert_equal(
        incorrectly_reported,
        set(),
        (
            "Blocked or ambiguous requests must "
            "not be presented as completed "
            "analytical findings."
        ),
    )


    pass_test(
        (
            "ambiguous and blocked requests are "
            "not misrepresented as findings"
        )
    )


# ============================================================
# TEST 5
# ============================================================

def test_exact_request_identity_is_preserved(
) -> None:
    (
        plan,
        _,
        findings,
    ) = build_lapage_requested_findings()


    plan_by_id = {
        request.request_id:
            request

        for request
        in plan.requests
    }


    finding_request_ids = [
        finding.request_id

        for finding
        in findings
    ]


    assert_equal(
        len(
            finding_request_ids
        ),
        len(
            set(
                finding_request_ids
            )
        ),
        (
            "Requested report findings must have "
            "unique request IDs."
        ),
    )


    for finding in findings:
        assert_true(
            finding.request_id
            in plan_by_id,
            (
                "Requested finding has no matching "
                "planner request."
            ),
        )


        request = (
            plan_by_id[
                finding.request_id
            ]
        )


        assert_equal(
            finding.title,
            request.request_text,
            (
                "Requested finding changed the "
                "original request text."
            ),
        )


        assert_equal(
            finding.source_filename,
            request.source_filename,
            (
                "Requested finding changed "
                "document provenance."
            ),
        )


        assert_equal(
            finding.source_locator,
            request.source_locator,
            (
                "Requested finding changed "
                "document locator provenance."
            ),
        )


        assert_equal(
            finding.page_number,
            request.page_number,
            (
                "Requested finding changed "
                "document page provenance."
            ),
        )


        assert_equal(
            finding.source_chunk_id,
            request.source_chunk_id,
            (
                "Requested finding changed "
                "source chunk provenance."
            ),
        )


        assert_equal(
            finding.evidence_unit_id,
            request.evidence_unit_id,
            (
                "Requested finding changed "
                "evidence unit provenance."
            ),
        )


        assert_equal(
            finding.evidence_quote,
            request.evidence_quote,
            (
                "Requested finding changed "
                "evidence text."
            ),
        )


    pass_test(
        (
            "all requested findings preserve "
            "exact planner identity and provenance"
        )
    )


# ============================================================
# TEST 6
# ============================================================

def test_document_distribution(
) -> None:
    (
        _,
        _,
        findings,
    ) = build_lapage_requested_findings()


    counts = (
        Counter(
            finding.source_filename

            for finding
            in findings
        )
    )


    assert_equal(
        counts[
            "conversation avec Julie.txt"
        ],
        5,
        (
            "All five executable Julie requests "
            "should reach requested_findings."
        ),
    )


    assert_equal(
        counts[
            "Brief de l'analyse.txt"
        ],
        6,
        (
            "Six executable brief requests "
            "should reach requested_findings."
        ),
    )


    pass_test(
        (
            "requested findings preserve "
            "5 Julie + 6 brief results"
        )
    )


# ============================================================
# TEST 7
# ============================================================

def test_analysis_ids_are_unique(
) -> None:
    (
        _,
        _,
        findings,
    ) = build_lapage_requested_findings()


    analysis_ids = [
        finding.analysis_id

        for finding
        in findings
    ]


    assert_equal(
        len(
            analysis_ids
        ),
        11,
        (
            "Expected eleven requested "
            "analysis IDs."
        ),
    )


    assert_equal(
        len(
            set(
                analysis_ids
            )
        ),
        11,
        (
            "Requested analysis IDs must be "
            "unique."
        ),
    )


    for analysis_id in analysis_ids:
        assert_true(
            analysis_id.startswith(
                "requested:"
            ),
            (
                "Requested finding analysis IDs "
                "must remain in requested namespace."
            ),
        )


    pass_test(
        (
            "all 11 requested findings have "
            "unique requested analysis IDs"
        )
    )


# ============================================================
# DIAGNOSTIC
# ============================================================

def print_findings() -> None:
    (
        _,
        execution_report,
        findings,
    ) = build_lapage_requested_findings()


    print()

    print(
        "===== LAPAGE REQUESTED REPORT FINDINGS ====="
    )

    print()


    for (
        index,
        finding,
    ) in enumerate(
        findings,
        start=1,
    ):
        print(
            (
                f"{index:02d}. "
                f"[{finding.execution_status.upper()}] "
                f"{finding.kind}"
            )
        )

        print(
            (
                f"    analysis_id: "
                f"{finding.analysis_id}"
            )
        )

        print(
            (
                f"    source: "
                f"{finding.source_filename}"
            )
        )

        print(
            (
                f"    dataset: "
                f"{finding.dataset_id}"
            )
        )

        print()


    print(
        "===== REPORT ADAPTER SUMMARY ====="
    )

    print()


    print(
        (
            "execution requests: "
            f"{execution_report.request_count}"
        )
    )

    print(
        (
            "execution attempted: "
            f"{execution_report.attempted_count}"
        )
    )

    print(
        (
            "reportable findings: "
            f"{len(findings)}"
        )
    )

    print(
        (
            "not represented as analytical findings: "
            f"{execution_report.not_executed_count}"
        )
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print(
        (
            "=== DATALENS LAPAGE REQUESTED "
            "REPORT ADAPTER v0.1 ==="
        )
    )

    print()


    test_report_adapter_produces_11_findings()

    test_all_ready_kinds_become_findings()

    test_reportable_statuses_only()

    test_non_ready_requests_are_not_misrepresented_as_findings()

    test_exact_request_identity_is_preserved()

    test_document_distribution()

    test_analysis_ids_are_unique()


    print_findings()


    print()

    print(
        (
            "PASS - Lapage requested "
            "report adapter v0.1"
        )
    )


if (
    __name__
    ==
    "__main__"
):
    main()