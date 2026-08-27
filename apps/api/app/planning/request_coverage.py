from __future__ import annotations


from collections import (
    Counter,
    defaultdict,
)

from typing import (
    TYPE_CHECKING,
    Literal,
)


from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)


from app.planning.request_planner import (
    request_identifier,
)

from app.planning.schemas import (
    RequestedAnalysisPlan,
    RequestedAnalysisPlanReport,
)


if TYPE_CHECKING:
    from app.document_summary import (
        VerifiedDocumentClaim,
    )


# ============================================================
# VERSION
# ============================================================

REQUEST_COVERAGE_RULE_VERSION = (
    "analysis_request_coverage_v0.1"
)


# ============================================================
# TYPES
# ============================================================

RequestCoverageStatus = Literal[
    "complete",
    "incomplete",
]


RequestCoverageState = Literal[
    "planned",
    "missing",
    "invalid",
]


PlanningStatus = Literal[
    "ready",
    "blocked",
    "ambiguous",
]


# ============================================================
# ONE REQUEST
# ============================================================

class AnalysisRequestCoverageItem(
    BaseModel
):
    """
    Traceability record for one verified analytical request.

    This object answers:

    - what was detected;
    - where it came from;
    - whether the planner preserved it;
    - whether planner provenance still matches the source;
    - what planning status it received.

    No analytical decision is made here.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    request_id: str

    request_text: str

    source_filename: str

    source_locator: str

    page_number: (
        int
        | None
    ) = None

    source_chunk_id: str

    evidence_unit_id: int

    coverage_state: RequestCoverageState

    planned: bool

    plan_count: int

    provenance_valid: bool

    planning_status: (
        PlanningStatus
        | None
    ) = None

    kind: (
        str
        | None
    ) = None

    target_family: (
        str
        | None
    ) = None

    blockers: list[
        str
    ] = Field(
        default_factory=list
    )


# ============================================================
# COVERAGE REPORT
# ============================================================

class AnalysisRequestCoverageReport(
    BaseModel
):
    """
    Deterministic audit of the boundary:

        verified analytical requests
                    ↓
              request planner

    COMPLETE means that every detected analytical request has
    exactly traceable planner coverage and that the planner
    report itself is internally consistent.

    A blocked or ambiguous request is still covered.

    Therefore:

        blocked   != lost
        ambiguous != lost

    Only disappearance or provenance/integrity failure causes
    coverage to become incomplete.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    status: RequestCoverageStatus

    detected_count: int

    planner_request_count: int

    planned_count: int

    ready_count: int

    blocked_count: int

    ambiguous_count: int

    lost_count: int

    coverage_rate: float

    plan_accounting_valid: bool

    lost_request_ids: list[
        str
    ] = Field(
        default_factory=list
    )

    provenance_mismatch_request_ids: list[
        str
    ] = Field(
        default_factory=list
    )

    orphan_plan_request_ids: list[
        str
    ] = Field(
        default_factory=list
    )

    duplicate_detected_request_ids: list[
        str
    ] = Field(
        default_factory=list
    )

    duplicate_planned_request_ids: list[
        str
    ] = Field(
        default_factory=list
    )

    items: list[
        AnalysisRequestCoverageItem
    ] = Field(
        default_factory=list
    )

    notes: list[
        str
    ] = Field(
        default_factory=list
    )

    rule_version: str = (
        REQUEST_COVERAGE_RULE_VERSION
    )


# ============================================================
# ERROR
# ============================================================

class AnalysisRequestCoverageError(
    ValueError
):
    """
    Raised when an analytical request inventory cannot be
    proven complete across the planner boundary.
    """

    def __init__(
        self,
        report: AnalysisRequestCoverageReport,
    ) -> None:
        self.report = (
            report
        )


        details: list[
            str
        ] = []


        if report.lost_request_ids:
            details.append(
                (
                    "lost_request_ids="
                    f"{report.lost_request_ids}"
                )
            )


        if (
            report
            .provenance_mismatch_request_ids
        ):
            details.append(
                (
                    "provenance_mismatch_request_ids="
                    f"{report.provenance_mismatch_request_ids}"
                )
            )


        if report.orphan_plan_request_ids:
            details.append(
                (
                    "orphan_plan_request_ids="
                    f"{report.orphan_plan_request_ids}"
                )
            )


        if (
            report
            .duplicate_detected_request_ids
        ):
            details.append(
                (
                    "duplicate_detected_request_ids="
                    f"{report.duplicate_detected_request_ids}"
                )
            )


        if (
            report
            .duplicate_planned_request_ids
        ):
            details.append(
                (
                    "duplicate_planned_request_ids="
                    f"{report.duplicate_planned_request_ids}"
                )
            )


        if not report.plan_accounting_valid:
            details.append(
                "plan_accounting_valid=False"
            )


        suffix = (
            "; ".join(
                details
            )
        )


        if suffix:
            suffix = (
                " "
                +
                suffix
            )


        super().__init__(
            (
                "Analytical request coverage is incomplete."
                f"{suffix}"
            )
        )


# ============================================================
# FILTER
# ============================================================

def _analytical_requests_only(
    claims: list[
        VerifiedDocumentClaim
    ],
) -> list[
    VerifiedDocumentClaim
]:
    """
    Keep only verified analytical requests.

    This mirrors the same semantic boundary used by
    build_requested_analysis_plan().
    """

    return [
        claim

        for claim
        in claims

        if (
            claim.category
            ==
            "analytical_request"
        )
    ]


# ============================================================
# PROVENANCE
# ============================================================

def _plan_matches_claim(
    *,
    plan: RequestedAnalysisPlan,
    claim: VerifiedDocumentClaim,
) -> bool:
    """
    Verify that a planner entry still represents the exact
    verified documentary request that produced its request_id.

    request_id alone is not considered sufficient.

    This prevents a stale, malformed or tampered planner result
    from being counted as valid coverage merely because the
    identifier happens to match.
    """

    expected_request_id = (
        request_identifier(
            claim
        )
    )


    return (
        plan.request_id
        ==
        expected_request_id

        and

        plan.request_text
        ==
        claim.statement

        and

        plan.context_text
        ==
        claim.context_quote

        and

        plan.evidence_quote
        ==
        claim.evidence_quote

        and

        plan.source_filename
        ==
        claim.citation.filename

        and

        plan.source_locator
        ==
        claim.citation.source_locator

        and

        plan.page_number
        ==
        claim.citation.page_number

        and

        plan.source_chunk_id
        ==
        claim.citation.chunk_id

        and

        plan.evidence_unit_id
        ==
        claim.evidence_unit_id
    )


# ============================================================
# PLANNER ACCOUNTING
# ============================================================

def _plan_accounting_valid(
    plan: RequestedAnalysisPlanReport,
) -> bool:
    """
    Verify that the public counters of the planner report agree
    with its actual request list.

    This catches stale or malformed report envelopes.
    """

    actual_request_count = (
        len(
            plan.requests
        )
    )


    actual_ready_count = sum(
        1

        for request
        in plan.requests

        if (
            request.status
            ==
            "ready"
        )
    )


    actual_blocked_count = sum(
        1

        for request
        in plan.requests

        if (
            request.status
            ==
            "blocked"
        )
    )


    actual_ambiguous_count = sum(
        1

        for request
        in plan.requests

        if (
            request.status
            ==
            "ambiguous"
        )
    )


    return (
        plan.request_count
        ==
        actual_request_count

        and

        plan.ready_count
        ==
        actual_ready_count

        and

        plan.blocked_count
        ==
        actual_blocked_count

        and

        plan.ambiguous_count
        ==
        actual_ambiguous_count

        and

        (
            plan.ready_count
            +
            plan.blocked_count
            +
            plan.ambiguous_count
        )
        ==
        plan.request_count
    )


# ============================================================
# DUPLICATES
# ============================================================

def _duplicate_ids(
    values: list[
        str
    ],
) -> list[
    str
]:
    counts = (
        Counter(
            values
        )
    )


    return sorted(
        value

        for (
            value,
            count,
        )
        in counts.items()

        if (
            count
            >
            1
        )
    )


# ============================================================
# REPORT BUILDER
# ============================================================

def build_analysis_request_coverage(
    *,
    analytical_requests: list[
        VerifiedDocumentClaim
    ],
    plan: RequestedAnalysisPlanReport,
) -> AnalysisRequestCoverageReport:
    """
    Compare verified analytical requests with the exact output
    of Requested Analysis Planner.

    IMPORTANT TRUST RULE
    --------------------

    The following planner statuses all count as preserved:

        ready
        blocked
        ambiguous

    A request is lost only when no planner item with valid
    source provenance can be found.

    Structural integrity problems such as duplicate or orphan
    plan IDs also make the overall report incomplete.
    """

    detected_requests = (
        _analytical_requests_only(
            analytical_requests
        )
    )


    detected_request_ids = [
        request_identifier(
            claim
        )

        for claim
        in detected_requests
    ]


    planned_request_ids = [
        request.request_id

        for request
        in plan.requests
    ]


    detected_id_set = set(
        detected_request_ids
    )


    duplicate_detected_request_ids = (
        _duplicate_ids(
            detected_request_ids
        )
    )


    duplicate_planned_request_ids = (
        _duplicate_ids(
            planned_request_ids
        )
    )


    orphan_plan_request_ids = sorted(
        {
            request_id

            for request_id
            in planned_request_ids

            if (
                request_id
                not in
                detected_id_set
            )
        }
    )


    plans_by_request_id: dict[
        str,
        list[
            RequestedAnalysisPlan
        ],
    ] = defaultdict(
        list
    )


    for request in plan.requests:
        plans_by_request_id[
            request.request_id
        ].append(
            request
        )


    items: list[
        AnalysisRequestCoverageItem
    ] = []


    lost_request_ids: list[
        str
    ] = []


    provenance_mismatch_request_ids: list[
        str
    ] = []


    planned_count = 0

    ready_count = 0

    blocked_count = 0

    ambiguous_count = 0


    for claim in detected_requests:
        expected_request_id = (
            request_identifier(
                claim
            )
        )


        candidate_plans = (
            plans_by_request_id.get(
                expected_request_id,
                [],
            )
        )


        valid_plans = [
            candidate_plan

            for candidate_plan
            in candidate_plans

            if (
                _plan_matches_claim(
                    plan=
                        candidate_plan,

                    claim=
                        claim,
                )
            )
        ]


        # ====================================================
        # COMPLETELY MISSING
        # ====================================================

        if not candidate_plans:
            lost_request_ids.append(
                expected_request_id
            )


            items.append(
                AnalysisRequestCoverageItem(
                    request_id=
                        expected_request_id,

                    request_text=
                        claim.statement,

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

                    coverage_state=
                        "missing",

                    planned=False,

                    plan_count=0,

                    provenance_valid=False,

                    planning_status=None,

                    kind=None,

                    target_family=None,

                    blockers=[],
                )
            )


            continue


        # ====================================================
        # SAME ID, WRONG PROVENANCE
        # ====================================================

        if not valid_plans:
            lost_request_ids.append(
                expected_request_id
            )


            provenance_mismatch_request_ids.append(
                expected_request_id
            )


            representative = (
                candidate_plans[
                    0
                ]
            )


            items.append(
                AnalysisRequestCoverageItem(
                    request_id=
                        expected_request_id,

                    request_text=
                        claim.statement,

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

                    coverage_state=
                        "invalid",

                    planned=False,

                    plan_count=
                        len(
                            candidate_plans
                        ),

                    provenance_valid=False,

                    planning_status=
                        representative.status,

                    kind=
                        representative.kind,

                    target_family=
                        representative.target_family,

                    blockers=
                        list(
                            representative.blockers
                        ),
                )
            )


            continue


        # ====================================================
        # VALID COVERAGE
        # ====================================================

        representative = (
            valid_plans[
                0
            ]
        )


        planned_count += 1


        if (
            representative.status
            ==
            "ready"
        ):
            ready_count += 1


        elif (
            representative.status
            ==
            "blocked"
        ):
            blocked_count += 1


        elif (
            representative.status
            ==
            "ambiguous"
        ):
            ambiguous_count += 1


        coverage_state: RequestCoverageState = (
            "planned"

            if (
                len(
                    candidate_plans
                )
                ==
                1
            )

            else
            "invalid"
        )


        items.append(
            AnalysisRequestCoverageItem(
                request_id=
                    expected_request_id,

                request_text=
                    claim.statement,

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

                coverage_state=
                    coverage_state,

                planned=True,

                plan_count=
                    len(
                        candidate_plans
                    ),

                provenance_valid=True,

                planning_status=
                    representative.status,

                kind=
                    representative.kind,

                target_family=
                    representative.target_family,

                blockers=
                    list(
                        representative.blockers
                    ),
            )
        )


    detected_count = (
        len(
            detected_requests
        )
    )


    lost_count = (
        detected_count
        -
        planned_count
    )


    if (
        detected_count
        ==
        0
    ):
        coverage_rate = (
            1.0
        )


    else:
        coverage_rate = (
            planned_count
            /
            detected_count
        )


    accounting_valid = (
        _plan_accounting_valid(
            plan
        )
    )


    integrity_valid = (
        lost_count
        ==
        0

        and

        not provenance_mismatch_request_ids

        and

        not orphan_plan_request_ids

        and

        not duplicate_detected_request_ids

        and

        not duplicate_planned_request_ids

        and

        accounting_valid
    )


    status: RequestCoverageStatus = (
        "complete"

        if integrity_valid

        else
        "incomplete"
    )


    notes = [
        (
            "Une demande ready, blocked ou ambiguous est "
            "considérée comme conservée tant que son "
            "identifiant et sa provenance correspondent "
            "exactement à la demande documentaire vérifiée."
        ),
        (
            "Le statut blocked n'est pas une perte : la "
            "demande reste visible avec son motif de blocage."
        ),
        (
            "Le statut ambiguous n'est pas une perte : la "
            "demande reste visible jusqu'à résolution ou "
            "décision utilisateur."
        ),
        (
            "La couverture n'est complète que si aucune "
            "demande n'a disparu et si aucune incohérence "
            "d'identité, de provenance ou de comptage n'est "
            "détectée."
        ),
    ]


    return (
        AnalysisRequestCoverageReport(
            status=
                status,

            detected_count=
                detected_count,

            planner_request_count=
                len(
                    plan.requests
                ),

            planned_count=
                planned_count,

            ready_count=
                ready_count,

            blocked_count=
                blocked_count,

            ambiguous_count=
                ambiguous_count,

            lost_count=
                lost_count,

            coverage_rate=
                coverage_rate,

            plan_accounting_valid=
                accounting_valid,

            lost_request_ids=
                sorted(
                    set(
                        lost_request_ids
                    )
                ),

            provenance_mismatch_request_ids=
                sorted(
                    set(
                        provenance_mismatch_request_ids
                    )
                ),

            orphan_plan_request_ids=
                orphan_plan_request_ids,

            duplicate_detected_request_ids=
                duplicate_detected_request_ids,

            duplicate_planned_request_ids=
                duplicate_planned_request_ids,

            items=
                items,

            notes=
                notes,
        )
    )


# ============================================================
# STRICT GUARD
# ============================================================

def require_complete_analysis_request_coverage(
    report: AnalysisRequestCoverageReport,
) -> AnalysisRequestCoverageReport:
    """
    Fail closed when analytical request coverage cannot be
    proven complete.

    This function is intended for the future server-side
    orchestration boundary before Requested Analysis execution.
    """

    if (
        report.status
        !=
        "complete"
    ):
        raise (
            AnalysisRequestCoverageError(
                report
            )
        )


    return report