from __future__ import annotations


from unittest.mock import (
    patch,
)


from app.execution import (
    requested_executor as requested_executor_module,
)

from app.execution.requested_schemas import (
    RequestedAnalysisExecution,
)

from tests.planning.test_lapage_request_planning_coverage_v0_1 import (
    build_lapage_plan,
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
# CONTROLLED EXECUTION RESULT
# ============================================================

def controlled_execution(
    *,
    request,
    datasets,
) -> RequestedAnalysisExecution:
    """
    Preserve the planner decision while replacing the actual
    statistical execution.

    ready
        -> complete

    blocked / ambiguous
        -> not_executed

    This isolates RequestedAnalysisExecutionReport accounting
    from dataframe resolution and statistical execution.
    """

    if (
        request.status
        ==
        "ready"
    ):
        execution_status = (
            "complete"
        )

        limitations = []


    else:
        execution_status = (
            "not_executed"
        )

        limitations = list(
            request.blockers
        )


    return (
        RequestedAnalysisExecution(
            request_id=
                request.request_id,

            request_text=
                request.request_text,

            kind=
                request.kind,

            plan_status=
                request.status,

            execution_status=
                execution_status,

            inferential_status=
                "not_applicable",

            source_filename=
                request.source_filename,

            source_locator=
                request.source_locator,

            evidence_quote=
                request.evidence_quote,

            dataset_id=
                None,

            dataset_filename=
                None,

            analytical_grain=
                None,

            analysis_mode=
                None,

            variables={},

            descriptive_statistics={},

            result=
                None,

            warnings=[],

            limitations=
                limitations,
        )
    )


# ============================================================
# BUILD REPORT
# ============================================================

def build_controlled_execution_report():
    (
        _,
        _,
        plan,
    ) = build_lapage_plan()


    with patch.object(
        requested_executor_module,
        "execute_requested_analysis",
        side_effect=
            controlled_execution,
    ):
        report = (
            requested_executor_module
            .execute_requested_analysis_plan(
                plan=
                    plan,

                datasets=[],
            )
        )


    return (
        plan,
        report,
    )


# ============================================================
# TEST 1
# ============================================================

def test_report_preserves_all_15_requests(
) -> None:
    (
        plan,
        report,
    ) = build_controlled_execution_report()


    assert_equal(
        plan.request_count,
        15,
        (
            "Lapage plan should contain "
            "15 requests."
        ),
    )


    assert_equal(
        report.request_count,
        15,
        (
            "Execution report must account for "
            "all 15 planned requests."
        ),
    )


    assert_equal(
        len(
            report.results
        ),
        15,
        (
            "Execution report must contain one "
            "result record per planned request."
        ),
    )


    pass_test(
        (
            "execution report preserves "
            "all 15 planned requests"
        )
    )


# ============================================================
# TEST 2
# ============================================================

def test_request_ids_are_preserved_exactly(
) -> None:
    (
        plan,
        report,
    ) = build_controlled_execution_report()


    planned_ids = [
        request.request_id

        for request
        in plan.requests
    ]


    executed_ids = [
        result.request_id

        for result
        in report.results
    ]


    assert_equal(
        executed_ids,
        planned_ids,
        (
            "Execution report changed, reordered "
            "or dropped request IDs."
        ),
    )


    assert_equal(
        len(
            set(
                executed_ids
            )
        ),
        15,
        (
            "Execution report should contain "
            "15 unique request IDs."
        ),
    )


    pass_test(
        (
            "execution report preserves exact "
            "request identity and ordering"
        )
    )


# ============================================================
# TEST 3
# ============================================================

def test_ready_requests_are_attempted(
) -> None:
    (
        _,
        report,
    ) = build_controlled_execution_report()


    assert_equal(
        report.attempted_count,
        11,
        (
            "The eleven ready Lapage requests "
            "should count as attempted."
        ),
    )


    assert_equal(
        report.complete_count,
        11,
        (
            "Controlled execution should mark "
            "all eleven ready requests complete."
        ),
    )


    pass_test(
        (
            "all 11 ready requests remain visible "
            "as attempted execution outcomes"
        )
    )


# ============================================================
# TEST 4
# ============================================================

def test_non_ready_requests_are_not_dropped(
) -> None:
    (
        _,
        report,
    ) = build_controlled_execution_report()


    assert_equal(
        report.not_executed_count,
        4,
        (
            "Three ambiguous requests and one "
            "blocked request must remain visible."
        ),
    )


    non_executed = {
        result.kind:
            result.plan_status

        for result
        in report.results

        if (
            result.execution_status
            ==
            "not_executed"
        )
    }


    expected = {
        "revenue_moving_average":
            "ambiguous",

        "top_products":
            "ambiguous",

        "flop_products":
            "ambiguous",

        "b2b_revenue_distribution":
            "blocked",
    }


    assert_equal(
        non_executed,
        expected,
        (
            "Non-ready planner requests were "
            "not preserved correctly."
        ),
    )


    pass_test(
        (
            "ambiguous and blocked requests remain "
            "visible in the execution report"
        )
    )


# ============================================================
# TEST 5
# ============================================================

def test_execution_accounting_reconciles(
) -> None:
    (
        _,
        report,
    ) = build_controlled_execution_report()


    accounted = (
        report.complete_count
        +
        report.descriptive_only_count
        +
        report.needs_information_count
        +
        report.needs_specialized_method_count
        +
        report.skipped_count
        +
        report.failed_count
        +
        report.not_executed_count
        +
        report.not_supported_yet_count
    )


    assert_equal(
        accounted,
        15,
        (
            "Execution status accounting must "
            "reconcile exactly to request_count."
        ),
    )


    assert_equal(
        accounted,
        report.request_count,
        (
            "Execution report contains an "
            "unaccounted request."
        ),
    )


    pass_test(
        (
            "execution status accounting "
            "reconciles to 15/15"
        )
    )


# ============================================================
# TEST 6
# ============================================================

def test_no_controlled_result_is_lost(
) -> None:
    (
        plan,
        report,
    ) = build_controlled_execution_report()


    results_by_id = {
        result.request_id:
            result

        for result
        in report.results
    }


    for request in plan.requests:
        assert_true(
            request.request_id
            in results_by_id,
            (
                "A planned request disappeared "
                "from Requested Analysis execution."
            ),
        )


        result = (
            results_by_id[
                request.request_id
            ]
        )


        assert_equal(
            result.request_text,
            request.request_text,
            (
                "Execution result changed the "
                "documentary request text."
            ),
        )


        assert_equal(
            result.source_filename,
            request.source_filename,
            (
                "Execution result changed request "
                "source provenance."
            ),
        )


        assert_equal(
            result.evidence_quote,
            request.evidence_quote,
            (
                "Execution result changed the "
                "documentary evidence quote."
            ),
        )


    pass_test(
        (
            "execution report preserves request "
            "text and documentary provenance"
        )
    )


# ============================================================
# DIAGNOSTIC
# ============================================================

def print_report() -> None:
    (
        _,
        report,
    ) = build_controlled_execution_report()


    print()

    print(
        "===== LAPAGE EXECUTION REPORT ====="
    )

    print()


    for (
        index,
        result,
    ) in enumerate(
        report.results,
        start=1,
    ):
        print(
            (
                f"{index:02d}. "
                f"[{result.execution_status.upper()}] "
                f"{result.kind} "
                f"(plan={result.plan_status})"
            )
        )


    print()

    print(
        (
            "SUMMARY: "
            f"{report.request_count} requests · "
            f"{report.attempted_count} attempted · "
            f"{report.complete_count} complete · "
            f"{report.not_executed_count} not_executed"
        )
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print(
        (
            "=== DATALENS LAPAGE REQUESTED "
            "EXECUTION REPORT COVERAGE v0.1 ==="
        )
    )

    print()


    test_report_preserves_all_15_requests()

    test_request_ids_are_preserved_exactly()

    test_ready_requests_are_attempted()

    test_non_ready_requests_are_not_dropped()

    test_execution_accounting_reconciles()

    test_no_controlled_result_is_lost()


    print_report()


    print()

    print(
        (
            "PASS - Lapage requested execution "
            "report coverage v0.1"
        )
    )


if (
    __name__
    ==
    "__main__"
):
    main()