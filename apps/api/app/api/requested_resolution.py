from __future__ import annotations


from threading import (
    RLock,
)

from typing import (
    Any,
    Literal,
)

from uuid import (
    uuid4,
)


from fastapi import (
    APIRouter,
    HTTPException,
)

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)


from app.api.analysis_run import (
    load_validated_analysis_input_for_http,
    prepare_analysis_datasets,
)

from app.execution.requested_executor import (
    execute_requested_analysis,
    execute_requested_analysis_plan,
)

from app.planning.request_resolution import (
    reconfigure_requested_analysis,
    resolve_requested_analysis,
)

from app.planning.follow_up_request import (
    plan_follow_up_requested_analysis,
)

from app.planning.schemas import (
    RequestedAnalysisPlan,
    RequestedAnalysisResolution,
)

from app.reporting.analysis_artifact_store import (
    AnalysisArtifactNotFoundError,
    AnalysisArtifactRecord,
    get_analysis_artifact,
    list_analysis_artifacts,
)

from app.reporting.requested_adapter import (
    build_requested_report_finding,
)

from app.reporting.unified_report_artifacts import (
    REQUESTED_ANALYSIS_SOURCE_TYPES,
    register_requested_report_finding,
    register_unresolved_requested_analysis_artifacts,
)


# ============================================================
# VERSION
# ============================================================

REQUESTED_ANALYSIS_RESOLUTION_API_VERSION = (
    "requested_analysis_resolution_api_v0.1"
)


REQUESTED_ANALYSIS_RECONFIGURATION_API_VERSION = (
    "requested_analysis_reconfiguration_api_v0.1"
)


FOLLOW_UP_REQUESTED_ANALYSIS_API_VERSION = (
    "follow_up_requested_analysis_api_v0.1"
)


router = APIRouter()


_RESOLUTION_LOCK = (
    RLock()
)


# ============================================================
# HTTP MODELS
# ============================================================

class FollowUpRequestedAnalysisRouteRequest(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )


    workflow_id: str = Field(
        min_length=1
    )


    objective: str = Field(
        min_length=1
    )


class FollowUpRequestedAnalysisRouteResponse(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid"
    )


    workflow_id: str

    objective: str

    route_kind: Literal[
        "requested_analysis",
        "ai_native",
    ]

    analysis_id: (
        str
        | None
    ) = None

    request_id: (
        str
        | None
    ) = None

    kind: (
        str
        | None
    ) = None

    plan_status: (
        str
        | None
    ) = None

    source_type: (
        str
        | None
    ) = None

    api_version: str = (
        FOLLOW_UP_REQUESTED_ANALYSIS_API_VERSION
    )


class RequestedAnalysisResolutionRequest(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )


    workflow_id: str = Field(
        min_length=1
    )


    request_id: str = Field(
        min_length=1
    )


    resolution: RequestedAnalysisResolution


class RequestedAnalysisResolutionResponse(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid"
    )


    workflow_id: str

    request_id: str

    analysis_id: str

    plan_status: str

    execution_status: str

    executed: bool

    source_type: str

    resolution: RequestedAnalysisResolution

    api_version: str = (
        REQUESTED_ANALYSIS_RESOLUTION_API_VERSION
    )


class RequestedAnalysisReconfigurationRequest(
    RequestedAnalysisResolutionRequest
):
    """
    Explicit parameters for reconfiguring an already executed
    documentary time-series request.

    Dataset identity, column bindings, execution results and
    analytical output are never accepted from the browser.
    """

    pass


class RequestedAnalysisReconfigurationResponse(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid"
    )


    workflow_id: str

    request_id: str

    analysis_id: str

    plan_status: str

    execution_status: str

    executed: bool

    source_type: str

    resolution: RequestedAnalysisResolution

    api_version: str = (
        REQUESTED_ANALYSIS_RECONFIGURATION_API_VERSION
    )


# ============================================================
# REQUESTED SOURCE IDENTITY
# ============================================================

def _requested_source_analysis_id(
    request_id: str,
) -> str:
    """
    Reproduce the canonical requested finding identity for a
    non-empty request_id.

    Documentary requests always have a server-generated
    request_id, so the fallback identity paths used for legacy
    executions are intentionally not needed here.
    """
    normalized = (
        str(
            request_id
        )
        .strip()
    )


    if not (
        normalized
    ):
        raise ValueError(
            "request_id cannot be empty."
        )


    if normalized.startswith(
        "request:"
    ):
        suffix = normalized[
            len(
                "request:"
            ):
        ]


        if not (
            suffix
        ):
            raise ValueError(
                (
                    "request_id cannot contain an empty "
                    "request: suffix."
                )
            )


        return (
            "requested:"
            +
            suffix
        )


    return (
        "requested:"
        +
        normalized
    )


# ============================================================
# ARTIFACT SOURCE IDENTITY
# ============================================================

def _artifact_source_analysis_id(
    artifact:
        AnalysisArtifactRecord,
) -> (
    str
    | None
):
    payload = (
        artifact.pipeline_payload
    )


    report_context = (
        payload.get(
            "report_context"
        )
    )


    if isinstance(
        report_context,
        dict,
    ):
        source_analysis_id = str(
            report_context.get(
                "source_analysis_id",
                "",
            )
        ).strip()


        if source_analysis_id:
            return (
                source_analysis_id
            )


    lifecycle = (
        payload.get(
            "request_lifecycle"
        )
    )


    if isinstance(
        lifecycle,
        dict,
    ):
        request_id = str(
            lifecycle.get(
                "request_id",
                "",
            )
        ).strip()


        if request_id:
            return (
                _requested_source_analysis_id(
                    request_id
                )
            )


    requested_plan = (
        payload.get(
            "requested_plan"
        )
    )


    if isinstance(
        requested_plan,
        dict,
    ):
        request_id = str(
            requested_plan.get(
                "request_id",
                "",
            )
        ).strip()


        if request_id:
            return (
                _requested_source_analysis_id(
                    request_id
                )
            )


    return None


# ============================================================
# ARTIFACT LOOKUP
# ============================================================

def _find_requested_artifact(
    *,
    workflow_id: str,
    request_id: str,
) -> AnalysisArtifactRecord:
    expected_source_id = (
        _requested_source_analysis_id(
            request_id
        )
    )


    matches = [
        artifact

        for artifact
        in list_analysis_artifacts(
            workflow_id=
                workflow_id
        )

        if (
            artifact.source_type
            in
            REQUESTED_ANALYSIS_SOURCE_TYPES
            and
            _artifact_source_analysis_id(
                artifact
            )
            ==
            expected_source_id
        )
    ]


    if not (
        matches
    ):
        raise HTTPException(
            status_code=404,

            detail={
                "error":
                    "requested_analysis_not_found",

                "message":
                    (
                        "No server-owned documentary request "
                        "artifact matches this request_id."
                    ),

                "workflow_id":
                    workflow_id,

                "request_id":
                    request_id,

                "api_version":
                    REQUESTED_ANALYSIS_RESOLUTION_API_VERSION,
            },
        )


    if (
        len(
            matches
        )
        !=
        1
    ):
        raise HTTPException(
            status_code=409,

            detail={
                "error":
                    "requested_analysis_identity_conflict",

                "message":
                    (
                        "Multiple server-owned artifacts match "
                        "the same documentary request identity."
                    ),

                "workflow_id":
                    workflow_id,

                "request_id":
                    request_id,

                "matching_analysis_ids":
                    [
                        artifact.analysis_id

                        for artifact
                        in matches
                    ],

                "api_version":
                    REQUESTED_ANALYSIS_RESOLUTION_API_VERSION,
            },
        )


    return (
        matches[
            0
        ]
    )


# ============================================================
# LIFECYCLE PLAN RESTORE
# ============================================================

def _restore_ambiguous_plan(
    *,
    artifact:
        AnalysisArtifactRecord,

    request_id: str,
) -> RequestedAnalysisPlan:
    if (
        artifact.executed
    ):
        raise HTTPException(
            status_code=409,

            detail={
                "error":
                    "requested_analysis_already_executed",

                "message":
                    (
                        "This documentary request has already "
                        "produced an executable analysis."
                    ),

                "analysis_id":
                    artifact.analysis_id,

                "request_id":
                    request_id,

                "api_version":
                    REQUESTED_ANALYSIS_RESOLUTION_API_VERSION,
            },
        )


    payload = (
        artifact.pipeline_payload
    )


    if (
        payload.get(
            "artifact_kind"
        )
        !=
        "requested_analysis_lifecycle"
    ):
        raise HTTPException(
            status_code=409,

            detail={
                "error":
                    "requested_analysis_not_resolvable",

                "message":
                    (
                        "The server-owned artifact is not an "
                        "unresolved requested-analysis lifecycle "
                        "artifact."
                    ),

                "analysis_id":
                    artifact.analysis_id,

                "request_id":
                    request_id,

                "api_version":
                    REQUESTED_ANALYSIS_RESOLUTION_API_VERSION,
            },
        )


    raw_plan = (
        payload.get(
            "requested_plan"
        )
    )


    if not isinstance(
        raw_plan,
        dict,
    ):
        raise HTTPException(
            status_code=409,

            detail={
                "error":
                    "requested_analysis_plan_missing",

                "message":
                    (
                        "The lifecycle artifact does not contain "
                        "a valid server-owned requested plan."
                    ),

                "analysis_id":
                    artifact.analysis_id,

                "request_id":
                    request_id,

                "api_version":
                    REQUESTED_ANALYSIS_RESOLUTION_API_VERSION,
            },
        )


    try:
        plan = (
            RequestedAnalysisPlan
            .model_validate(
                raw_plan
            )
        )


    except Exception as error:
        raise HTTPException(
            status_code=409,

            detail={
                "error":
                    "requested_analysis_plan_invalid",

                "message":
                    (
                        "The persisted requested plan failed "
                        "schema validation."
                    ),

                "analysis_id":
                    artifact.analysis_id,

                "request_id":
                    request_id,

                "validation_error":
                    (
                        f"{type(error).__name__}: "
                        f"{error}"
                    ),

                "api_version":
                    REQUESTED_ANALYSIS_RESOLUTION_API_VERSION,
            },
        ) from error


    if (
        plan.request_id
        !=
        request_id
    ):
        raise HTTPException(
            status_code=409,

            detail={
                "error":
                    "requested_analysis_identity_mismatch",

                "message":
                    (
                        "The persisted plan request_id does not "
                        "match the requested lifecycle identity."
                    ),

                "analysis_id":
                    artifact.analysis_id,

                "request_id":
                    request_id,

                "persisted_request_id":
                    plan.request_id,

                "api_version":
                    REQUESTED_ANALYSIS_RESOLUTION_API_VERSION,
            },
        )


    if (
        plan.status
        !=
        "ambiguous"
    ):
        raise HTTPException(
            status_code=409,

            detail={
                "error":
                    "requested_analysis_not_ambiguous",

                "message":
                    (
                        "Only a server-owned ambiguous request "
                        "can accept a user clarification."
                    ),

                "analysis_id":
                    artifact.analysis_id,

                "request_id":
                    request_id,

                "plan_status":
                    plan.status,

                "api_version":
                    REQUESTED_ANALYSIS_RESOLUTION_API_VERSION,
            },
        )


    return plan


# ============================================================
# EXECUTED PLAN RESTORE
# ============================================================

def _restore_reconfigurable_plan(
    *,
    artifact:
        AnalysisArtifactRecord,

    request_id: str,
) -> RequestedAnalysisPlan:
    """
    Restore the exact server-owned plan that produced an
    already executed documentary request.

    This deliberately rejects unresolved lifecycle artifacts
    and legacy executed artifacts that do not contain the
    server-owned requested_plan / requested_finding pair.
    """

    if not (
        artifact.executed
    ):
        raise HTTPException(
            status_code=409,

            detail={
                "error":
                    "requested_analysis_not_executed",

                "message":
                    (
                        "Only an already executed documentary "
                        "request can be reconfigured."
                    ),

                "analysis_id":
                    artifact.analysis_id,

                "request_id":
                    request_id,

                "api_version":
                    REQUESTED_ANALYSIS_RECONFIGURATION_API_VERSION,
            },
        )


    payload = (
        artifact.pipeline_payload
    )


    raw_plan = (
        payload.get(
            "requested_plan"
        )
    )

    raw_finding = (
        payload.get(
            "requested_finding"
        )
    )


    if not isinstance(
        raw_plan,
        dict,
    ):
        raise HTTPException(
            status_code=409,

            detail={
                "error":
                    "requested_reconfiguration_plan_missing",

                "message":
                    (
                        "The executed artifact does not contain "
                        "a valid server-owned requested plan."
                    ),

                "analysis_id":
                    artifact.analysis_id,

                "request_id":
                    request_id,

                "api_version":
                    REQUESTED_ANALYSIS_RECONFIGURATION_API_VERSION,
            },
        )


    if not isinstance(
        raw_finding,
        dict,
    ):
        raise HTTPException(
            status_code=409,

            detail={
                "error":
                    "requested_reconfiguration_finding_missing",

                "message":
                    (
                        "The executed artifact does not contain "
                        "its server-owned requested finding."
                    ),

                "analysis_id":
                    artifact.analysis_id,

                "request_id":
                    request_id,

                "api_version":
                    REQUESTED_ANALYSIS_RECONFIGURATION_API_VERSION,
            },
        )


    try:
        plan = (
            RequestedAnalysisPlan
            .model_validate(
                raw_plan
            )
        )

    except Exception as error:
        raise HTTPException(
            status_code=409,

            detail={
                "error":
                    "requested_reconfiguration_plan_invalid",

                "message":
                    (
                        "The server-owned requested plan cannot "
                        "be restored safely."
                    ),

                "analysis_id":
                    artifact.analysis_id,

                "request_id":
                    request_id,

                "api_version":
                    REQUESTED_ANALYSIS_RECONFIGURATION_API_VERSION,
            },
        ) from error


    if (
        plan.request_id
        !=
        request_id
    ):
        raise HTTPException(
            status_code=409,

            detail={
                "error":
                    "requested_reconfiguration_identity_mismatch",

                "message":
                    (
                        "The requested plan identity does not "
                        "match the documentary artifact."
                    ),

                "analysis_id":
                    artifact.analysis_id,

                "request_id":
                    request_id,

                "plan_request_id":
                    plan.request_id,

                "api_version":
                    REQUESTED_ANALYSIS_RECONFIGURATION_API_VERSION,
            },
        )


    finding_request_id = str(
        raw_finding.get(
            "request_id",
            "",
        )
        or
        ""
    ).strip()


    if (
        finding_request_id
        !=
        request_id
    ):
        raise HTTPException(
            status_code=409,

            detail={
                "error":
                    "requested_reconfiguration_finding_identity_mismatch",

                "message":
                    (
                        "The stored requested finding identity "
                        "does not match its server-owned plan."
                    ),

                "analysis_id":
                    artifact.analysis_id,

                "request_id":
                    request_id,

                "finding_request_id":
                    finding_request_id,

                "api_version":
                    REQUESTED_ANALYSIS_RECONFIGURATION_API_VERSION,
            },
        )


    expected_source_analysis_id = (
        _requested_source_analysis_id(
            request_id
        )
    )


    finding_analysis_id = str(
        raw_finding.get(
            "analysis_id",
            "",
        )
        or
        ""
    ).strip()


    if (
        finding_analysis_id
        !=
        expected_source_analysis_id
    ):
        raise HTTPException(
            status_code=409,

            detail={
                "error":
                    "requested_reconfiguration_source_identity_mismatch",

                "message":
                    (
                        "The stored requested finding no longer "
                        "matches the canonical documentary "
                        "analysis identity."
                    ),

                "analysis_id":
                    artifact.analysis_id,

                "request_id":
                    request_id,

                "expected_source_analysis_id":
                    expected_source_analysis_id,

                "finding_analysis_id":
                    finding_analysis_id,

                "api_version":
                    REQUESTED_ANALYSIS_RECONFIGURATION_API_VERSION,
            },
        )


    if (
        plan.status
        !=
        "ready"
    ):
        raise HTTPException(
            status_code=409,

            detail={
                "error":
                    "requested_analysis_not_reconfigurable",

                "message":
                    (
                        "Only a ready server-owned requested "
                        "analysis can be reconfigured."
                    ),

                "analysis_id":
                    artifact.analysis_id,

                "request_id":
                    request_id,

                "plan_status":
                    plan.status,

                "api_version":
                    REQUESTED_ANALYSIS_RECONFIGURATION_API_VERSION,
            },
        )


    if (
        plan.kind
        !=
        "revenue_moving_average"
    ):
        raise HTTPException(
            status_code=409,

            detail={
                "error":
                    "requested_analysis_kind_not_reconfigurable",

                "message":
                    (
                        "Reconfiguration is currently supported "
                        "only for revenue_moving_average."
                    ),

                "analysis_id":
                    artifact.analysis_id,

                "request_id":
                    request_id,

                "kind":
                    plan.kind,

                "api_version":
                    REQUESTED_ANALYSIS_RECONFIGURATION_API_VERSION,
            },
        )


    if (
        plan.resolution
        is None
        or
        plan.resolution.resolution_type
        !=
        "time_series_parameters"
    ):
        raise HTTPException(
            status_code=409,

            detail={
                "error":
                    "requested_analysis_parameters_missing",

                "message":
                    (
                        "The executed request does not contain "
                        "reconfigurable time-series parameters."
                    ),

                "analysis_id":
                    artifact.analysis_id,

                "request_id":
                    request_id,

                "api_version":
                    REQUESTED_ANALYSIS_RECONFIGURATION_API_VERSION,
            },
        )


    return plan



# ============================================================
# FOLLOW-UP REQUEST ROUTER
# ============================================================

@router.post(
    "/analysis/requested/route-follow-up",

    response_model=
        FollowUpRequestedAnalysisRouteResponse,
)
def route_follow_up_requested_analysis_http(
    request:
        FollowUpRequestedAnalysisRouteRequest,
) -> FollowUpRequestedAnalysisRouteResponse:
    """
    Give known deterministic Requested Analysis intents
    priority for workspace follow-up prompts.

    Browser trust boundary:

        browser
            -> workflow_id
            -> objective

        server
            -> validated Preparation handoff
            -> deterministic request classification
            -> deterministic Requested Analysis plan
            -> unresolved lifecycle artifact

    Unknown / unsupported follow-up prompts are NOT rejected.
    They return route_kind=ai_native so the frontend can use
    the existing local-AI pipeline unchanged.

    No browser-supplied dataset id, column binding, plan,
    calculation, metric or chart payload is accepted here.
    """
    workflow_id = str(
        request.workflow_id
    ).strip()

    objective = str(
        request.objective
    ).strip()


    if not workflow_id:
        raise HTTPException(
            status_code=422,

            detail={
                "error":
                    "invalid_follow_up_workflow_id",

                "message":
                    "workflow_id cannot be empty.",

                "api_version":
                    FOLLOW_UP_REQUESTED_ANALYSIS_API_VERSION,
            },
        )


    if not objective:
        raise HTTPException(
            status_code=422,

            detail={
                "error":
                    "invalid_follow_up_objective",

                "message":
                    "objective cannot be empty.",

                "workflow_id":
                    workflow_id,

                "api_version":
                    FOLLOW_UP_REQUESTED_ANALYSIS_API_VERSION,
            },
        )


    # --------------------------------------------------------
    # Server-owned validated Preparation source of truth.
    # --------------------------------------------------------

    handoff = (
        load_validated_analysis_input_for_http(
            workflow_id=
                workflow_id
        )
    )


    plan_report = (
        plan_follow_up_requested_analysis(
            ingestion=
                handoff.ingestion,

            objective=
                objective,

            request_key=
                uuid4().hex,
        )
    )


    # --------------------------------------------------------
    # Unsupported deterministic intent -> existing AI-native
    # fallback. No artifact is created by this route.
    # --------------------------------------------------------

    if plan_report is None:
        return (
            FollowUpRequestedAnalysisRouteResponse(
                workflow_id=
                    workflow_id,

                objective=
                    objective,

                route_kind=
                    "ai_native",

                analysis_id=
                    None,

                request_id=
                    None,

                kind=
                    None,

                plan_status=
                    None,

                source_type=
                    None,
            )
        )


    if (
        plan_report.request_count
        !=
        1
        or
        len(
            plan_report.requests
        )
        !=
        1
    ):
        raise HTTPException(
            status_code=500,

            detail={
                "error":
                    "follow_up_requested_plan_cardinality_error",

                "message":
                    (
                        "Deterministic follow-up routing did not "
                        "produce exactly one requested plan."
                    ),

                "workflow_id":
                    workflow_id,

                "api_version":
                    FOLLOW_UP_REQUESTED_ANALYSIS_API_VERSION,
            },
        )


    plan = (
        plan_report.requests[
            0
        ]
    )


    # v0.1 routes revenue_moving_average specifically because
    # it requires a deliberate human parameter choice.
    #
    # A future planner regression that unexpectedly returns
    # ready must fail closed instead of silently executing with
    # implicit defaults.
    if (
        plan.status
        not in {
            "ambiguous",
            "blocked",
        }
    ):
        raise HTTPException(
            status_code=500,

            detail={
                "error":
                    "follow_up_requested_plan_status_error",

                "message":
                    (
                        "The deterministic follow-up route "
                        "returned an unexpected executable "
                        "plan without explicit user resolution."
                    ),

                "workflow_id":
                    workflow_id,

                "request_id":
                    plan.request_id,

                "kind":
                    plan.kind,

                "plan_status":
                    plan.status,

                "api_version":
                    FOLLOW_UP_REQUESTED_ANALYSIS_API_VERSION,
            },
        )


    source_dataset_records = [
        dict(
            record
        )

        for record
        in handoff.dataset_records
    ]


    # Requested-only analytical views are allowed here because
    # they are still derived from the current validated
    # Preparation output and remain server-owned.
    (
        _,
        analysis_datasets,
    ) = (
        prepare_analysis_datasets(
            source_datasets=
                source_dataset_records,

            objective=
                objective,

            include_requested_context=
                True,
        )
    )


    execution_report = (
        execute_requested_analysis_plan(
            plan=
                plan_report,

            datasets=
                analysis_datasets,
        )
    )


    registered = (
        register_unresolved_requested_analysis_artifacts(
            workflow_id=
                workflow_id,

            execution_report=
                execution_report,

            plan_report=
                plan_report,

            source_type=
                "follow_up_prompt",
        )
    )


    if (
        len(
            registered
        )
        !=
        1
    ):
        raise HTTPException(
            status_code=500,

            detail={
                "error":
                    "follow_up_lifecycle_registration_error",

                "message":
                    (
                        "The deterministic follow-up request "
                        "did not produce exactly one unresolved "
                        "server-owned lifecycle artifact."
                    ),

                "workflow_id":
                    workflow_id,

                "request_id":
                    plan.request_id,

                "kind":
                    plan.kind,

                "plan_status":
                    plan.status,

                "api_version":
                    FOLLOW_UP_REQUESTED_ANALYSIS_API_VERSION,
            },
        )


    artifact = (
        registered[
            0
        ]
    )


    if (
        artifact.source_type
        !=
        "follow_up_prompt"
        or
        artifact.executed
    ):
        raise HTTPException(
            status_code=500,

            detail={
                "error":
                    "follow_up_lifecycle_integrity_error",

                "message":
                    (
                        "The persisted follow-up lifecycle "
                        "artifact failed its source/execution "
                        "integrity check."
                    ),

                "workflow_id":
                    workflow_id,

                "analysis_id":
                    artifact.analysis_id,

                "request_id":
                    plan.request_id,

                "source_type":
                    artifact.source_type,

                "executed":
                    artifact.executed,

                "api_version":
                    FOLLOW_UP_REQUESTED_ANALYSIS_API_VERSION,
            },
        )


    return (
        FollowUpRequestedAnalysisRouteResponse(
            workflow_id=
                workflow_id,

            objective=
                objective,

            route_kind=
                "requested_analysis",

            analysis_id=
                artifact.analysis_id,

            request_id=
                plan.request_id,

            kind=
                plan.kind,

            plan_status=
                plan.status,

            source_type=
                artifact.source_type,
        )
    )


# ============================================================
# PUBLIC ENDPOINT
# ============================================================

@router.post(
    "/analysis/requested/resolve",

    response_model=
        RequestedAnalysisResolutionResponse,
)
def resolve_requested_analysis_http(
    request:
        RequestedAnalysisResolutionRequest,
) -> RequestedAnalysisResolutionResponse:
    """
    Apply one human clarification to an existing server-owned
    documentary request.

    Trust boundary:

        browser
            -> workflow_id
            -> request_id
            -> explicit resolution only

        server
            -> lifecycle artifact
            -> original RequestedAnalysisPlan
            -> deterministic resolution validation
            -> validated Preparation handoff
            -> deterministic analytical views
            -> single requested execution
            -> requested finding adapter
            -> same report artifact identity

    The browser cannot submit datasets, columns, plan status,
    execution status, chart data, metrics or analytical output.
    """
    workflow_id = (
        request.workflow_id
    )

    request_id = (
        request.request_id
    )


    with _RESOLUTION_LOCK:
        artifact = (
            _find_requested_artifact(
                workflow_id=
                    workflow_id,

                request_id=
                    request_id,
            )
        )


        plan = (
            _restore_ambiguous_plan(
                artifact=
                    artifact,

                request_id=
                    request_id,
            )
        )


        try:
            resolved_plan = (
                resolve_requested_analysis(
                    plan=
                        plan,

                    resolution=
                        request.resolution,
                )
            )


        except ValueError as error:
            raise HTTPException(
                status_code=409,

                detail={
                    "error":
                        "requested_resolution_rejected",

                    "message":
                        str(
                            error
                        ),

                    "workflow_id":
                        workflow_id,

                    "request_id":
                        request_id,

                    "analysis_id":
                        artifact.analysis_id,

                    "api_version":
                        REQUESTED_ANALYSIS_RESOLUTION_API_VERSION,
                },
            ) from error


        if (
            resolved_plan.status
            !=
            "ready"
        ):
            raise HTTPException(
                status_code=409,

                detail={
                    "error":
                        "requested_resolution_not_executable",

                    "message":
                        (
                            "The selected clarification does not "
                            "produce an executable requested plan."
                        ),

                    "workflow_id":
                        workflow_id,

                    "request_id":
                        request_id,

                    "analysis_id":
                        artifact.analysis_id,

                    "plan_status":
                        resolved_plan.status,

                    "blockers":
                        list(
                            resolved_plan.blockers
                        ),

                    "resolution":
                        request
                        .resolution
                        .model_dump(
                            mode="json"
                        ),

                    "api_version":
                        REQUESTED_ANALYSIS_RESOLUTION_API_VERSION,
                },
            )


        # ----------------------------------------------------
        # Reload the exact currently authorized Preparation
        # outputs. No dataset identifier comes from the client.
        # ----------------------------------------------------

        handoff = (
            load_validated_analysis_input_for_http(
                workflow_id=
                    workflow_id
            )
        )


        source_dataset_records = [
            dict(
                record
            )

            for record
            in handoff.dataset_records
        ]


        # ----------------------------------------------------
        # Rebuild the deterministic analytical context required
        # by requested analyses, including requested-only views.
        # ----------------------------------------------------

        (
            _,
            analysis_datasets,
        ) = (
            prepare_analysis_datasets(
                source_datasets=
                    source_dataset_records,

                objective=
                    artifact.objective,

                include_requested_context=
                    True,
            )
        )


        # ----------------------------------------------------
        # Execute this request only.
        # ----------------------------------------------------

        try:
            execution = (
                execute_requested_analysis(
                    request=
                        resolved_plan,

                    datasets=
                        analysis_datasets,
                )
            )


        except Exception as error:
            raise HTTPException(
                status_code=500,

                detail={
                    "error":
                        "requested_resolution_execution_failed",

                    "message":
                        (
                            "Deterministic execution failed after "
                            "the clarification was validated."
                        ),

                    "workflow_id":
                        workflow_id,

                    "request_id":
                        request_id,

                    "analysis_id":
                        artifact.analysis_id,

                    "execution_error":
                        (
                            f"{type(error).__name__}: "
                            f"{error}"
                        ),

                    "api_version":
                        REQUESTED_ANALYSIS_RESOLUTION_API_VERSION,
                },
            ) from error


        if (
            execution.request_id
            !=
            request_id
        ):
            raise HTTPException(
                status_code=500,

                detail={
                    "error":
                        "requested_resolution_execution_identity_mismatch",

                    "message":
                        (
                            "Requested execution changed the "
                            "server-owned request identity."
                        ),

                    "workflow_id":
                        workflow_id,

                    "request_id":
                        request_id,

                    "execution_request_id":
                        execution.request_id,

                    "api_version":
                        REQUESTED_ANALYSIS_RESOLUTION_API_VERSION,
                },
            )


        # ----------------------------------------------------
        # Reuse the canonical requested finding adapter.
        # Non-reportable executions remain fail-closed.
        # ----------------------------------------------------

        finding = (
            build_requested_report_finding(
                execution,

                plan=
                    resolved_plan,
            )
        )


        if (
            finding
            is None
        ):
            raise HTTPException(
                status_code=409,

                detail={
                    "error":
                        "requested_resolution_not_reportable",

                    "message":
                        (
                            "The clarified request did not produce "
                            "a reportable deterministic result."
                        ),

                    "workflow_id":
                        workflow_id,

                    "request_id":
                        request_id,

                    "analysis_id":
                        artifact.analysis_id,

                    "plan_status":
                        resolved_plan.status,

                    "execution_status":
                        execution.execution_status,

                    "warnings":
                        list(
                            execution.warnings
                        ),

                    "limitations":
                        list(
                            execution.limitations
                        ),

                    "api_version":
                        REQUESTED_ANALYSIS_RESOLUTION_API_VERSION,
                },
            )


        expected_source_analysis_id = (
            _requested_source_analysis_id(
                request_id
            )
        )


        if (
            finding.analysis_id
            !=
            expected_source_analysis_id
        ):
            raise HTTPException(
                status_code=500,

                detail={
                    "error":
                        "requested_resolution_finding_identity_mismatch",

                    "message":
                        (
                            "Requested finding identity differs "
                            "from the lifecycle request identity."
                        ),

                    "workflow_id":
                        workflow_id,

                    "request_id":
                        request_id,

                    "expected_source_analysis_id":
                        expected_source_analysis_id,

                    "finding_analysis_id":
                        finding.analysis_id,

                    "api_version":
                        REQUESTED_ANALYSIS_RESOLUTION_API_VERSION,
                },
            )


        # ----------------------------------------------------
        # Re-read immediately before the lifecycle promotion.
        # This catches stale state before the write.
        # ----------------------------------------------------

        try:
            current_artifact = (
                get_analysis_artifact(
                    workflow_id=
                        workflow_id,

                    analysis_id=
                        artifact.analysis_id,
                )
            )


        except AnalysisArtifactNotFoundError as error:
            raise HTTPException(
                status_code=409,

                detail={
                    "error":
                        "requested_analysis_changed_during_resolution",

                    "message":
                        (
                            "The lifecycle artifact disappeared "
                            "while the request was being resolved."
                        ),

                    "workflow_id":
                        workflow_id,

                    "request_id":
                        request_id,

                    "analysis_id":
                        artifact.analysis_id,

                    "api_version":
                        REQUESTED_ANALYSIS_RESOLUTION_API_VERSION,
                },
            ) from error


        if (
            current_artifact.executed
        ):
            raise HTTPException(
                status_code=409,

                detail={
                    "error":
                        "requested_analysis_changed_during_resolution",

                    "message":
                        (
                            "The lifecycle artifact became executable "
                            "while the request was being resolved."
                        ),

                    "workflow_id":
                        workflow_id,

                    "request_id":
                        request_id,

                    "analysis_id":
                        artifact.analysis_id,

                    "api_version":
                        REQUESTED_ANALYSIS_RESOLUTION_API_VERSION,
                },
            )


        # ----------------------------------------------------
        # Promote the same lifecycle artifact.
        #
        # expected_analysis_id is checked BEFORE the write by
        # register_requested_report_finding().
        # ----------------------------------------------------

        try:
            registered = (
                register_requested_report_finding(
                    workflow_id=
                        workflow_id,

                    finding=
                        finding,

                    requested_plan=
                        resolved_plan,

                    expected_analysis_id=
                        artifact.analysis_id,

                    source_type=
                        artifact.source_type,

                    select_by_default=
                        True,
                )
            )


        except ValueError as error:
            raise HTTPException(
                status_code=409,

                detail={
                    "error":
                        "requested_analysis_promotion_rejected",

                    "message":
                        str(
                            error
                        ),

                    "workflow_id":
                        workflow_id,

                    "request_id":
                        request_id,

                    "analysis_id":
                        artifact.analysis_id,

                    "api_version":
                        REQUESTED_ANALYSIS_RESOLUTION_API_VERSION,
                },
            ) from error


        if (
            registered.analysis_id
            !=
            artifact.analysis_id
            or
            registered.source_type
            !=
            artifact.source_type
            or
            not registered.executed
        ):
            raise HTTPException(
                status_code=500,

                detail={
                    "error":
                        "requested_analysis_promotion_integrity_error",

                    "message":
                        (
                            "The promoted requested artifact failed "
                            "its post-write integrity check."
                        ),

                    "workflow_id":
                        workflow_id,

                    "request_id":
                        request_id,

                    "before_analysis_id":
                        artifact.analysis_id,

                    "after_analysis_id":
                        registered.analysis_id,

                    "executed":
                        registered.executed,

                    "source_type":
                        registered.source_type,

                    "api_version":
                        REQUESTED_ANALYSIS_RESOLUTION_API_VERSION,
                },
            )


        return (
            RequestedAnalysisResolutionResponse(
                workflow_id=
                    workflow_id,

                request_id=
                    request_id,

                analysis_id=
                    registered.analysis_id,

                plan_status=
                    resolved_plan.status,

                execution_status=
                    execution.execution_status,

                executed=
                    registered.executed,

                source_type=
                    registered.source_type,

                resolution=
                    request.resolution,

                api_version=
                    REQUESTED_ANALYSIS_RESOLUTION_API_VERSION,
            )
        )


# ============================================================
# REQUESTED ANALYSIS RECONFIGURATION
# ============================================================

@router.post(
    "/analysis/requested/reconfigure",

    response_model=
        RequestedAnalysisReconfigurationResponse,
)
def reconfigure_requested_analysis_http(
    request:
        RequestedAnalysisReconfigurationRequest,
) -> RequestedAnalysisReconfigurationResponse:
    """
    Reconfigure an already executed documentary time series.

    Trust boundary:

        browser
            -> workflow_id
            -> request_id
            -> explicit time-series parameters only

        server
            -> existing executed document-request artifact
            -> persisted RequestedAnalysisPlan
            -> deterministic reconfiguration validation
            -> current validated Preparation handoff
            -> deterministic requested execution
            -> requested finding adapter
            -> same artifact identity

    The browser cannot submit datasets, columns, chart data,
    calculated metrics, plan status or analytical output.
    """

    workflow_id = (
        request.workflow_id
    )

    request_id = (
        request.request_id
    )


    with _RESOLUTION_LOCK:
        artifact = (
            _find_requested_artifact(
                workflow_id=
                    workflow_id,

                request_id=
                    request_id,
            )
        )


        plan = (
            _restore_reconfigurable_plan(
                artifact=
                    artifact,

                request_id=
                    request_id,
            )
        )


        try:
            reconfigured_plan = (
                reconfigure_requested_analysis(
                    plan=
                        plan,

                    resolution=
                        request.resolution,
                )
            )

        except ValueError as error:
            raise HTTPException(
                status_code=409,

                detail={
                    "error":
                        "requested_reconfiguration_rejected",

                    "message":
                        str(
                            error
                        ),

                    "workflow_id":
                        workflow_id,

                    "request_id":
                        request_id,

                    "analysis_id":
                        artifact.analysis_id,

                    "api_version":
                        REQUESTED_ANALYSIS_RECONFIGURATION_API_VERSION,
                },
            ) from error


        if (
            reconfigured_plan.status
            !=
            "ready"
            or
            reconfigured_plan.resolution
            is None
        ):
            raise HTTPException(
                status_code=409,

                detail={
                    "error":
                        "requested_reconfiguration_not_executable",

                    "message":
                        (
                            "The selected parameters do not "
                            "produce an executable requested plan."
                        ),

                    "workflow_id":
                        workflow_id,

                    "request_id":
                        request_id,

                    "analysis_id":
                        artifact.analysis_id,

                    "plan_status":
                        reconfigured_plan.status,

                    "blockers":
                        list(
                            reconfigured_plan.blockers
                        ),

                    "api_version":
                        REQUESTED_ANALYSIS_RECONFIGURATION_API_VERSION,
                },
            )


        # ----------------------------------------------------
        # Current validated Preparation output only.
        # ----------------------------------------------------

        handoff = (
            load_validated_analysis_input_for_http(
                workflow_id=
                    workflow_id
            )
        )


        source_dataset_records = [
            dict(
                record
            )

            for record
            in handoff.dataset_records
        ]


        (
            _,
            analysis_datasets,
        ) = (
            prepare_analysis_datasets(
                source_datasets=
                    source_dataset_records,

                objective=
                    artifact.objective,

                include_requested_context=
                    True,
            )
        )


        # ----------------------------------------------------
        # Deterministic execution.
        # ----------------------------------------------------

        try:
            execution = (
                execute_requested_analysis(
                    request=
                        reconfigured_plan,

                    datasets=
                        analysis_datasets,
                )
            )

        except Exception as error:
            raise HTTPException(
                status_code=500,

                detail={
                    "error":
                        "requested_reconfiguration_execution_failed",

                    "message":
                        (
                            "Deterministic execution failed after "
                            "the reconfiguration was validated."
                        ),

                    "workflow_id":
                        workflow_id,

                    "request_id":
                        request_id,

                    "analysis_id":
                        artifact.analysis_id,

                    "execution_error":
                        (
                            f"{type(error).__name__}: "
                            f"{error}"
                        ),

                    "api_version":
                        REQUESTED_ANALYSIS_RECONFIGURATION_API_VERSION,
                },
            ) from error


        if (
            execution.request_id
            !=
            request_id
        ):
            raise HTTPException(
                status_code=500,

                detail={
                    "error":
                        "requested_reconfiguration_execution_identity_mismatch",

                    "message":
                        (
                            "Requested execution changed the "
                            "server-owned request identity."
                        ),

                    "workflow_id":
                        workflow_id,

                    "request_id":
                        request_id,

                    "execution_request_id":
                        execution.request_id,

                    "api_version":
                        REQUESTED_ANALYSIS_RECONFIGURATION_API_VERSION,
                },
            )


        # ----------------------------------------------------
        # Canonical finding adapter.
        # ----------------------------------------------------

        finding = (
            build_requested_report_finding(
                execution,

                plan=
                    reconfigured_plan,
            )
        )


        if (
            finding
            is None
        ):
            raise HTTPException(
                status_code=409,

                detail={
                    "error":
                        "requested_reconfiguration_not_reportable",

                    "message":
                        (
                            "The reconfigured request did not "
                            "produce a reportable deterministic "
                            "result."
                        ),

                    "workflow_id":
                        workflow_id,

                    "request_id":
                        request_id,

                    "analysis_id":
                        artifact.analysis_id,

                    "plan_status":
                        reconfigured_plan.status,

                    "execution_status":
                        execution.execution_status,

                    "warnings":
                        list(
                            execution.warnings
                        ),

                    "limitations":
                        list(
                            execution.limitations
                        ),

                    "api_version":
                        REQUESTED_ANALYSIS_RECONFIGURATION_API_VERSION,
                },
            )


        expected_source_analysis_id = (
            _requested_source_analysis_id(
                request_id
            )
        )


        if (
            finding.analysis_id
            !=
            expected_source_analysis_id
        ):
            raise HTTPException(
                status_code=500,

                detail={
                    "error":
                        "requested_reconfiguration_finding_identity_mismatch",

                    "message":
                        (
                            "The reconfigured finding changed "
                            "the documentary analysis identity."
                        ),

                    "workflow_id":
                        workflow_id,

                    "request_id":
                        request_id,

                    "analysis_id":
                        artifact.analysis_id,

                    "expected_source_analysis_id":
                        expected_source_analysis_id,

                    "finding_analysis_id":
                        finding.analysis_id,

                    "api_version":
                        REQUESTED_ANALYSIS_RECONFIGURATION_API_VERSION,
                },
            )


        # ----------------------------------------------------
        # Stale-state guard immediately before refresh.
        # ----------------------------------------------------

        try:
            current_artifact = (
                get_analysis_artifact(
                    workflow_id=
                        workflow_id,

                    analysis_id=
                        artifact.analysis_id,
                )
            )

        except AnalysisArtifactNotFoundError as error:
            raise HTTPException(
                status_code=409,

                detail={
                    "error":
                        "requested_analysis_changed_during_reconfiguration",

                    "message":
                        (
                            "The requested-analysis artifact "
                            "disappeared during reconfiguration."
                        ),

                    "workflow_id":
                        workflow_id,

                    "request_id":
                        request_id,

                    "analysis_id":
                        artifact.analysis_id,

                    "api_version":
                        REQUESTED_ANALYSIS_RECONFIGURATION_API_VERSION,
                },
            ) from error


        if (
            not current_artifact.executed
            or
            current_artifact.source_type
            !=
            artifact.source_type
            or
            current_artifact.created_at_utc
            !=
            artifact.created_at_utc
            or
            current_artifact.pipeline_payload.get(
                "requested_plan"
            )
            !=
            artifact.pipeline_payload.get(
                "requested_plan"
            )
        ):
            raise HTTPException(
                status_code=409,

                detail={
                    "error":
                        "requested_analysis_changed_during_reconfiguration",

                    "message":
                        (
                            "The server-owned requested analysis "
                            "changed while the new parameters "
                            "were being evaluated."
                        ),

                    "workflow_id":
                        workflow_id,

                    "request_id":
                        request_id,

                    "analysis_id":
                        artifact.analysis_id,

                    "api_version":
                        REQUESTED_ANALYSIS_RECONFIGURATION_API_VERSION,
                },
            )


        # ----------------------------------------------------
        # Refresh the SAME artifact.
        #
        # select_by_default=False is intentional:
        # true -> true refresh must preserve the user's current
        # report-selection decision.
        # ----------------------------------------------------

        try:
            registered = (
                register_requested_report_finding(
                    workflow_id=
                        workflow_id,

                    finding=
                        finding,

                    requested_plan=
                        reconfigured_plan,

                    expected_analysis_id=
                        artifact.analysis_id,

                    source_type=
                        artifact.source_type,

                    select_by_default=
                        False,
                )
            )

        except ValueError as error:
            raise HTTPException(
                status_code=409,

                detail={
                    "error":
                        "requested_reconfiguration_refresh_rejected",

                    "message":
                        str(
                            error
                        ),

                    "workflow_id":
                        workflow_id,

                    "request_id":
                        request_id,

                    "analysis_id":
                        artifact.analysis_id,

                    "api_version":
                        REQUESTED_ANALYSIS_RECONFIGURATION_API_VERSION,
                },
            ) from error


        if (
            registered.analysis_id
            !=
            artifact.analysis_id
            or
            registered.source_type
            !=
            artifact.source_type
            or
            not registered.executed
            or
            registered.created_at_utc
            !=
            artifact.created_at_utc
        ):
            raise HTTPException(
                status_code=500,

                detail={
                    "error":
                        "requested_reconfiguration_artifact_invariant_failed",

                    "message":
                        (
                            "The reconfigured analysis did not "
                            "preserve its server-owned artifact "
                            "identity."
                        ),

                    "workflow_id":
                        workflow_id,

                    "request_id":
                        request_id,

                    "analysis_id":
                        artifact.analysis_id,

                    "registered_analysis_id":
                        registered.analysis_id,

                    "api_version":
                        REQUESTED_ANALYSIS_RECONFIGURATION_API_VERSION,
                },
            )


        stored_plan = (
            registered
            .pipeline_payload
            .get(
                "requested_plan"
            )
        )

        stored_finding = (
            registered
            .pipeline_payload
            .get(
                "requested_finding"
            )
        )


        if (
            not isinstance(
                stored_plan,
                dict,
            )
            or
            not isinstance(
                stored_finding,
                dict,
            )
        ):
            raise HTTPException(
                status_code=500,

                detail={
                    "error":
                        "requested_reconfiguration_payload_missing",

                    "message":
                        (
                            "The refreshed artifact does not "
                            "contain both its requested plan and "
                            "requested finding."
                        ),

                    "workflow_id":
                        workflow_id,

                    "request_id":
                        request_id,

                    "analysis_id":
                        artifact.analysis_id,

                    "api_version":
                        REQUESTED_ANALYSIS_RECONFIGURATION_API_VERSION,
                },
            )


        return (
            RequestedAnalysisReconfigurationResponse(
                workflow_id=
                    workflow_id,

                request_id=
                    request_id,

                analysis_id=
                    registered.analysis_id,

                plan_status=
                    reconfigured_plan.status,

                execution_status=
                    execution.execution_status,

                executed=
                    registered.executed,

                source_type=
                    registered.source_type,

                resolution=
                    reconfigured_plan.resolution,

                api_version=
                    REQUESTED_ANALYSIS_RECONFIGURATION_API_VERSION,
            )
        )
