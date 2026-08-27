from __future__ import annotations


from typing import (
    TYPE_CHECKING,
)


from fastapi import (
    HTTPException,
)


from app.planning.request_coverage import (
    AnalysisRequestCoverageError,
    AnalysisRequestCoverageReport,
    build_analysis_request_coverage,
    require_complete_analysis_request_coverage,
)

from app.planning.schemas import (
    RequestedAnalysisPlanReport,
)


if TYPE_CHECKING:
    from app.document_summary import (
        VerifiedDocumentClaim,
    )


# ============================================================
# VERSION
# ============================================================

REQUEST_COVERAGE_HTTP_GUARD_VERSION = (
    "analysis_request_coverage_http_guard_v0.1"
)


# ============================================================
# HTTP GUARD
# ============================================================

def require_analysis_request_coverage_for_http(
    *,
    analytical_requests: list[
        VerifiedDocumentClaim
    ],
    plan: RequestedAnalysisPlanReport,
) -> AnalysisRequestCoverageReport:
    """
    Build and enforce deterministic analytical-request
    coverage at the HTTP orchestration boundary.

    IMPORTANT
    ---------

    The following planner states still count as preserved:

        ready
        blocked
        ambiguous

    DataLens fails closed only when a request disappears or
    when identity / provenance / report integrity cannot be
    proven.

    No raw dataset rows are involved in this guard.
    """

    report = (
        build_analysis_request_coverage(
            analytical_requests=
                analytical_requests,

            plan=
                plan,
        )
    )


    try:
        return (
            require_complete_analysis_request_coverage(
                report
            )
        )


    except AnalysisRequestCoverageError as error:
        raise HTTPException(
            status_code=409,

            detail={
                "error":
                    "analysis_request_coverage_incomplete",

                "message":
                    str(
                        error
                    ),

                "coverage_rule_version":
                    report.rule_version,

                "http_guard_version":
                    REQUEST_COVERAGE_HTTP_GUARD_VERSION,

                "detected_count":
                    report.detected_count,

                "planner_request_count":
                    report.planner_request_count,

                "planned_count":
                    report.planned_count,

                "ready_count":
                    report.ready_count,

                "blocked_count":
                    report.blocked_count,

                "ambiguous_count":
                    report.ambiguous_count,

                "lost_count":
                    report.lost_count,

                "coverage_rate":
                    report.coverage_rate,

                "plan_accounting_valid":
                    report.plan_accounting_valid,

                "lost_request_ids":
                    list(
                        report
                        .lost_request_ids
                    ),

                "provenance_mismatch_request_ids":
                    list(
                        report
                        .provenance_mismatch_request_ids
                    ),

                "orphan_plan_request_ids":
                    list(
                        report
                        .orphan_plan_request_ids
                    ),

                "duplicate_detected_request_ids":
                    list(
                        report
                        .duplicate_detected_request_ids
                    ),

                "duplicate_planned_request_ids":
                    list(
                        report
                        .duplicate_planned_request_ids
                    ),
            },
        ) from error