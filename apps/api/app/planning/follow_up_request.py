from __future__ import annotations


from app.document_summary import (
    DocumentSummaryCitation,
    VerifiedDocumentClaim,
)

from app.ingestion.schemas import (
    MultiDatasetIngestion,
)

from app.planning.request_planner import (
    build_requested_analysis_plan,
    classify_request,
)

from app.planning.schemas import (
    RequestedAnalysisPlanReport,
)


# ============================================================
# VERSION
# ============================================================

FOLLOW_UP_REQUEST_RULE_VERSION = (
    "follow_up_requested_analysis_v0.1"
)


# ============================================================
# SUPPORTED DETERMINISTIC FOLLOW-UP REQUESTS
# ============================================================

SUPPORTED_FOLLOW_UP_REQUEST_KINDS = (
    frozenset(
        {
            "revenue_moving_average",
        }
    )
)


# ============================================================
# SYNTHETIC SERVER-OWNED REQUEST CLAIM
# ============================================================

def _build_follow_up_claim(
    *,
    objective: str,
    request_key: str,
) -> VerifiedDocumentClaim:
    """
    Adapt an explicit workspace follow-up prompt to the
    existing Requested Analysis planner contract.

    This is not documentary evidence.

    The synthetic citation exists only because the historical
    Requested Analysis planner consumes VerifiedDocumentClaim.

    The final AnalysisArtifact source_type remains
    follow_up_prompt and therefore preserves the true
    provenance of the user request.
    """
    normalized_objective = str(
        objective
        or
        ""
    ).strip()

    normalized_request_key = str(
        request_key
        or
        ""
    ).strip()


    if not normalized_objective:
        raise ValueError(
            "Follow-up objective cannot be empty."
        )


    if not normalized_request_key:
        raise ValueError(
            "Follow-up request_key cannot be empty."
        )


    synthetic_id = (
        "follow-up:"
        +
        normalized_request_key
    )


    return (
        VerifiedDocumentClaim(
            category=
                "analytical_request",

            statement=
                normalized_objective,

            evidence_quote=
                normalized_objective,

            evidence_unit_id=
                1,

            context_quote=
                None,

            context_evidence_unit_id=
                None,

            citation=
                DocumentSummaryCitation(
                    chunk_id=
                        synthetic_id,

                    document_id=
                        synthetic_id,

                    filename=
                        "follow_up_prompt",

                    source_locator=
                        "workspace:analysis-follow-up",

                    page_number=
                        None,
                ),
        )
    )


# ============================================================
# PUBLIC ROUTER
# ============================================================

def plan_follow_up_requested_analysis(
    *,
    ingestion: MultiDatasetIngestion,
    objective: str,
    request_key: str,
) -> (
    RequestedAnalysisPlanReport
    | None
):
    """
    Give deterministic Requested Analysis rules priority over
    the generic AI-native planner for known follow-up intents.

    Return None when the prompt is not supported by this
    deterministic bridge. The caller may then use the existing
    AI-native fallback unchanged.

    v0.1 deliberately routes only revenue_moving_average.
    """
    claim = (
        _build_follow_up_claim(
            objective=
                objective,

            request_key=
                request_key,
        )
    )


    kind = (
        classify_request(
            claim
        )
    )


    if (
        kind
        not in
        SUPPORTED_FOLLOW_UP_REQUEST_KINDS
    ):
        return None


    report = (
        build_requested_analysis_plan(
            ingestion=
                ingestion,

            analytical_requests=[
                claim
            ],
        )
    )


    if (
        report.request_count
        !=
        1
        or
        len(
            report.requests
        )
        !=
        1
    ):
        raise RuntimeError(
            (
                "Deterministic follow-up routing must produce "
                "exactly one Requested Analysis plan."
            )
        )


    plan = (
        report.requests[
            0
        ]
    )


    if (
        plan.kind
        !=
        kind
    ):
        raise RuntimeError(
            (
                "Follow-up request classification changed "
                "during deterministic planning."
            )
        )


    return report
