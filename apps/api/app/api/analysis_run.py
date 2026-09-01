from __future__ import annotations

from fastapi import Request as FastAPIRequest


import json


from time import (
    perf_counter,
)


from typing import (
    Any,
)


from fastapi import (
    APIRouter,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
)

from fastapi.responses import (
    Response,
)

from pydantic import (
    BaseModel,
)


from app.observability.runtime_trace import (
    stamp_validated_runtime_workflow_id,
)

from app.analysis.analytical_views import (
    ANALYTICAL_VIEW_RULE_VERSION,
    build_analytical_views,
)

from app.analysis.entity_outlier_requests import (
    resolve_entity_outlier_intent,
)

from app.analysis.derived_policy import (
    apply_aggregate_breakdown_ranking_policy,
    apply_derived_discovery_policy,
    inject_aggregate_breakdown_execution,
    normalize_report_aggregate_families,
)

from app.analysis.feature_lineage import (
    apply_feature_lineage_policy,
    apply_feature_lineage_ranking_policy,
)

from app.api.document_ingestion import (
    ingest_document_uploads,
)

from app.api.routes import (
    load_uploaded_dataset_bundle,
)

from app.api.request_coverage_guard import (
    require_analysis_request_coverage_for_http,
)

from app.discovery import (
    discover_analyses,
)

from app.discovery.schemas import (
    AnalysisDiscoveryReport,
)

from app.analysis_prioritization import (
    ANALYSIS_PRIORITIZATION_RULE_VERSION,
    ANALYTICAL_VALUE_GUARD_RULE_VERSION,
    AnalysisPrioritizationAudit,
    build_analysis_prioritization_audit,
    build_prioritized_execution_discovery,
    prioritize_analysis_discovery,
)

from app.document_summary import (
    DocumentSummaryReport,
    summarize_document_ingestion,
)

from app.execution import (
    RequestedAnalysisExecutionReport,
    execute_cross_dataset_discovery,
    execute_requested_analysis_plan,
    execute_single_dataset_discovery,
)

from app.planning import (
    RequestedAnalysisPlanReport,
    build_requested_analysis_plan,
)

from app.planning.request_coverage import (
    AnalysisRequestCoverageReport,
)

from app.planning.objective_coverage import (
    require_objective_coverage,
)

from app.planning.ai_analytical_planner import (
    AIPlannerReport,
    DEFAULT_AI_PLANNER_MODEL,
    PlannerCatalog,
)

from app.planning.analytical_request_router import (
    route_analytical_request,
)

from app.planning.intent_routed_planner import (
    plan_analyses_with_intent_routing,
)

from app.planning.planner_catalog import (
    planner_catalog_from_dataset_records,
)

from app.ai.tool_orchestrator import (
    AIToolOrchestrationReport,
    execute_ai_planner_report,
)

from app.ai.ai_native_pipeline import (
    AINativePipelineReport,
    execute_native_ai_pipeline,
)

from app.ai.native_tool_calling import (
    DEFAULT_NATIVE_TOOL_MODEL,
)

from app.observability import (
    build_ai_trace,
    new_ai_trace_id,
    write_ai_trace,
)

from app.observability.ai_trace import (
    AITraceRecord,
)

from app.observability.trace_store import (
    AITraceListResponse,
    AITraceMetricsResponse,
    get_ai_trace,
    get_ai_trace_metrics,
    get_latest_ai_trace,
    list_ai_traces,
)

from app.observability.trace_explorer import (
    TraceExplorerReadError,
    TraceExplorerResponse,
    get_request_trace_explorer,
)

from app.rag_context import (
    RagContextReport,
    retrieve_context_for_report,
)

from app.rag_retrieval import (
    DEFAULT_EMBEDDING_MODEL,
)

from app.ranking import (
    rank_unified_analysis,
)

from app.reporting.entity_outlier_adapter import (
    EntityOutlierFinding,
    adapt_entity_outlier_request_to_finding,
)

from app.reporting.pdf_export import (
    build_analysis_pdf,
    build_export_filename,
)

from app.reporting.requested_adapter import (
    attach_requested_findings,
    build_requested_first_rag_report_view,
)

from app.reporting.unified_composer import (
    compose_unified_report,
)

from app.reporting.unified_schemas import (
    UnifiedAnalysisReport,
)

from app.reporting.unified_report_artifacts import (
    register_unresolved_requested_analysis_artifacts,
    register_unified_report_artifacts,
    register_contextualized_report_artifacts_atomic,
)

from app.preparation.cleaning_engine import (
    CleaningExecutionResult,
    CleaningPlan,
    build_cleaning_plan,
    execute_cleaning_plan,
)

from app.preparation.data_quality import (
    DataQualityReport,
    build_data_quality_report,
)

from app.preparation.semantic_review import (
    RawSemanticDecision,
    RawSemanticReviewResponse,
    build_semantic_review_candidates,
    validate_semantic_review_response,
)

from app.preparation.semantic_cleaning import (
    SemanticCleaningChoice,
    SemanticCleaningExecutionResult,
    SemanticCleaningPlan,
    build_semantic_cleaning_plan,
    execute_semantic_cleaning_plan,
)

from app.preparation.analysis_readiness_gate import (
    AnalysisDatasetNotAuthorizedError,
    AnalysisNotReadyError,
    require_analysis_readiness,
)

from app.preparation.analysis_input_handoff import (
    AnalysisInputHandoffError,
    AnalysisPreparedArtifactUnavailableError,
    load_validated_analysis_input,
)

from app.preparation.preparation_session import (
    PreparationSessionNotFoundError,
)


router = APIRouter()


# ============================================================
# ROUTED UNIFIED RESPONSE
# ============================================================

class RoutedUnifiedAnalysisReport(
    UnifiedAnalysisReport
):
    """
    Backward-compatible extension of the canonical unified
    report.

    Existing UnifiedAnalysisReport fields remain unchanged.

    The optional entity_outlier_finding is populated only when
    the explicit user objective resolves to an entity-level
    outlier request such as:

        "Détecte les clients atypiques."

    Generic requests such as:

        "Détecte les outliers."

    continue through the existing analytical planner path and
    leave this field empty.
    """

    entity_outlier_finding: (
        EntityOutlierFinding
        | None
    ) = None

    prioritization_audit: (
        AnalysisPrioritizationAudit
        | None
    ) = None


# ============================================================
# CONTEXTUALIZED RESPONSE
# ============================================================

class ContextualizedAnalysisResponse(
    BaseModel
):
    analysis: RoutedUnifiedAnalysisReport

    document_summary: DocumentSummaryReport

    requested_analysis_plan: (
        RequestedAnalysisPlanReport
    )

    request_coverage: (
        AnalysisRequestCoverageReport
    )

    requested_analysis_execution: (
        RequestedAnalysisExecutionReport
    )

    rag: RagContextReport


# ============================================================
# PDF EXPORT REQUEST
# ============================================================

class PdfExportRequest(
    BaseModel
):
    analysis: UnifiedAnalysisReport

    objective: (
        str
        | None
    ) = None

    document_summary: (
        DocumentSummaryReport
        | None
    ) = None

    requested_analysis_plan: (
        RequestedAnalysisPlanReport
        | None
    ) = None

    quality_report: (
        DataQualityReport
        | None
    ) = None


# ============================================================
# OBJECTIVE NORMALIZATION
# ============================================================

def normalize_objective(
    objective: (
        str
        | None
    ),
) -> (
    str
    | None
):
    if objective is None:
        return None


    normalized = (
        objective
        .strip()
    )


    if not normalized:
        return None


    return normalized


# ============================================================
# EXPLICIT ENTITY OUTLIER FINDING
# ============================================================

def build_entity_outlier_finding_if_requested(
    *,
    objective: (
        str
        | None
    ),

    source_dataset_records: list[
        dict[
            str,
            Any,
        ]
    ],
) -> (
    EntityOutlierFinding
    | None
):
    """
    Execute the explicit entity-outlier branch only when the
    user objective actually matches that intent.

    The pre-check is important.

    route_analytical_request() also supports generic and AI
    planner fallbacks. The standard unified-analysis endpoint
    must not unexpectedly invoke those planner paths simply to
    decide whether an entity finding exists.

    Therefore:

        explicit entity request
            -> global router
            -> entity result
            -> user-facing finding

        every other request
            -> no entity finding here
            -> existing DataLens analysis flow remains intact
    """

    normalized_objective = (
        normalize_objective(
            objective
        )
    )


    if (
        normalized_objective
        is None
    ):
        return None


    resolution = (
        resolve_entity_outlier_intent(
            normalized_objective
        )
    )


    if (
        resolution.status
        !=
        "matched"
    ):
        return None


    catalog = (
        planner_catalog_from_dataset_records(
            source_dataset_records
        )
    )


    routed = (
        route_analytical_request(
            objective=
                normalized_objective,

            source_dataset_records=
                source_dataset_records,

            catalog=
                catalog,
        )
    )


    if (
        routed.route_kind
        !=
        "entity_outlier"

        or

        routed.entity_outlier_report
        is None
    ):
        raise RuntimeError(
            (
                "The explicit entity-outlier intent "
                "was matched, but the analytical "
                "request router did not return the "
                "entity-outlier branch."
            )
        )


    return (
        adapt_entity_outlier_request_to_finding(
            routed.entity_outlier_report
        )
    )


def build_routed_unified_analysis_report(
    *,
    report: UnifiedAnalysisReport,

    entity_outlier_finding: (
        EntityOutlierFinding
        | None
    ),

    prioritization_audit: (
        AnalysisPrioritizationAudit
        | None
    ) = None,
) -> RoutedUnifiedAnalysisReport:
    """
    Preserve the existing UnifiedAnalysisReport wire contract
    and add the optional routed entity finding.

    This deliberately avoids mutating the canonical report
    schema at this stage.
    """

    payload = (
        report.model_dump(
            mode="python"
        )
    )


    payload[
        "entity_outlier_finding"
    ] = (
        entity_outlier_finding
    )


    payload[
        "prioritization_audit"
    ] = (
        prioritization_audit
    )


    return (
        RoutedUnifiedAnalysisReport
        .model_validate(
            payload
        )
    )


# ============================================================
# ANALYSIS READINESS — HTTP GATE
# ============================================================

def require_analysis_readiness_for_records(
    *,
    workflow_id: str,

    source_dataset_records: list[
        dict[
            str,
            Any,
        ]
    ],
) -> None:
    """
    Enforce the server-owned preparation gate before an
    HTTP analysis endpoint may execute.

    The requested analysis dataset scope is derived from the
    dataset records created by DataLens ingestion. The browser
    cannot submit arbitrary dataset IDs to this helper.
    """

    requested_dataset_ids = [
        str(
            record[
                "dataset_id"
            ]
        )

        for record
        in source_dataset_records
    ]


    try:
        require_analysis_readiness(
            workflow_id=
                workflow_id,

            requested_analysis_dataset_ids=
                requested_dataset_ids,
        )


    except PreparationSessionNotFoundError as error:
        raise HTTPException(
            status_code=404,

            detail={
                "error":
                    "preparation_session_not_found",

                "message":
                    str(
                        error
                    ),

                "workflow_id":
                    workflow_id,
            },
        ) from error


    except AnalysisDatasetNotAuthorizedError as error:
        decision = (
            error.decision
        )


        raise HTTPException(
            status_code=403,

            detail={
                "error":
                    "analysis_dataset_not_authorized",

                "message":
                    str(
                        error
                    ),

                "workflow_id":
                    decision.workflow_id,

                "requested_analysis_dataset_ids":
                    list(
                        decision
                        .requested_analysis_dataset_ids
                    ),

                "unauthorized_dataset_ids":
                    list(
                        decision
                        .unauthorized_dataset_ids
                    ),

                "selected_analysis_dataset_ids":
                    list(
                        decision
                        .selected_analysis_dataset_ids
                    ),
            },
        ) from error


    except AnalysisNotReadyError as error:
        decision = (
            error.decision
        )


        raise HTTPException(
            status_code=409,

            detail={
                "error":
                    "analysis_not_ready",

                "message":
                    str(
                        error
                    ),

                "workflow_id":
                    decision.workflow_id,

                "session_revision":
                    decision.session_revision,

                "ready_for_analysis":
                    decision.ready_for_analysis,

                "next_stage":
                    (
                        decision
                        .next_stage
                        .value

                        if (
                            decision.next_stage
                            is not None
                        )

                        else
                        None
                    ),

                "blocking_reasons":
                    list(
                        decision
                        .blocking_reasons
                    ),

                "unvalidated_dataset_ids":
                    list(
                        decision
                        .unvalidated_dataset_ids
                    ),
            },
        ) from error


    except ValueError as error:
        raise HTTPException(
            status_code=422,

            detail={
                "error":
                    "invalid_analysis_readiness_request",

                "message":
                    str(
                        error
                    ),

                "workflow_id":
                    workflow_id,
            },
        ) from error



# ============================================================
# ANALYSIS INPUT HANDOFF — HTTP
# ============================================================

def load_validated_analysis_input_for_http(
    *,
    workflow_id: str,
):
    """
    Resolve the exact server-owned Preparation output that may
    cross the HTTP boundary into Analysis.

    The browser-provided dataset upload is intentionally not
    consulted here.

    HTTP mapping:

    - unknown Preparation session -> 404;
    - unauthorized scope -> 403;
    - Preparation not ready -> 409;
    - validated artifact missing -> 409;
    - handoff consistency failure -> 409;
    - malformed identifier -> 422.
    """

    try:
        return (
            load_validated_analysis_input(
                workflow_id=
                    workflow_id
            )
        )


    except PreparationSessionNotFoundError as error:
        raise HTTPException(
            status_code=404,

            detail={
                "error":
                    "preparation_session_not_found",

                "message":
                    str(
                        error
                    ),

                "workflow_id":
                    workflow_id,
            },
        ) from error


    except AnalysisDatasetNotAuthorizedError as error:
        decision = (
            error.decision
        )


        raise HTTPException(
            status_code=403,

            detail={
                "error":
                    "analysis_dataset_not_authorized",

                "message":
                    str(
                        error
                    ),

                "workflow_id":
                    decision.workflow_id,

                "requested_analysis_dataset_ids":
                    list(
                        decision
                        .requested_analysis_dataset_ids
                    ),

                "unauthorized_dataset_ids":
                    list(
                        decision
                        .unauthorized_dataset_ids
                    ),

                "selected_analysis_dataset_ids":
                    list(
                        decision
                        .selected_analysis_dataset_ids
                    ),

                "analysis_output_dataset_ids":
                    list(
                        decision
                        .analysis_output_dataset_ids
                    ),
            },
        ) from error


    except AnalysisNotReadyError as error:
        decision = (
            error.decision
        )


        raise HTTPException(
            status_code=409,

            detail={
                "error":
                    "analysis_not_ready",

                "message":
                    str(
                        error
                    ),

                "workflow_id":
                    decision.workflow_id,

                "session_revision":
                    decision.session_revision,

                "ready_for_analysis":
                    decision.ready_for_analysis,

                "next_stage":
                    (
                        decision
                        .next_stage
                        .value

                        if (
                            decision.next_stage
                            is not None
                        )

                        else
                        None
                    ),

                "blocking_reasons":
                    list(
                        decision
                        .blocking_reasons
                    ),

                "unvalidated_dataset_ids":
                    list(
                        decision
                        .unvalidated_dataset_ids
                    ),
            },
        ) from error


    except AnalysisPreparedArtifactUnavailableError as error:
        raise HTTPException(
            status_code=409,

            detail={
                "error":
                    "validated_analysis_artifact_unavailable",

                "message":
                    str(
                        error
                    ),

                "workflow_id":
                    error.workflow_id,

                "dataset_id":
                    error.dataset_id,
            },
        ) from error


    except AnalysisInputHandoffError as error:
        raise HTTPException(
            status_code=409,

            detail={
                "error":
                    "analysis_input_handoff_failed",

                "message":
                    str(
                        error
                    ),

                "workflow_id":
                    workflow_id,
            },
        ) from error


    except ValueError as error:
        raise HTTPException(
            status_code=422,

            detail={
                "error":
                    "invalid_analysis_readiness_request",

                "message":
                    str(
                        error
                    ),

                "workflow_id":
                    workflow_id,
            },
        ) from error


# ============================================================
# POST-VALIDATION PREPARATION OVERRIDE GUARD
# ============================================================

def reject_post_validation_preparation_overrides(
    *,
    approved_action_ids_json: (
        str
        | None
    ),

    semantic_decisions_json: (
        str
        | None
    ),

    approved_semantic_choices_json: (
        str
        | None
    ),
) -> None:
    """
    Prevent Analysis from mutating a dataset after Preparation
    has crossed VALIDATE.

    Empty legacy multipart values remain accepted so the
    current frontend contract can migrate gradually.

    Any substantive deterministic or semantic cleaning request
    must be completed inside Preparation before final
    validation.
    """

    approved_action_ids = (
        parse_approved_cleaning_action_ids(
            approved_action_ids_json
        )
    )


    semantic_decisions = (
        parse_semantic_decisions_json(
            semantic_decisions_json
        )
    )


    semantic_choices = (
        parse_approved_semantic_choices_json(
            approved_semantic_choices_json
        )
    )


    supplied_fields: list[
        str
    ] = []


    if (
        approved_action_ids
    ):
        supplied_fields.append(
            "approved_action_ids_json"
        )


    if (
        semantic_decisions is not None
        and
        semantic_decisions.decisions
    ):
        supplied_fields.append(
            "semantic_decisions_json"
        )


    if (
        semantic_choices
    ):
        supplied_fields.append(
            "approved_semantic_choices_json"
        )


    if (
        supplied_fields
    ):
        raise ValueError(
            (
                "Analysis cannot apply Preparation mutations "
                "after VALIDATE. Complete cleaning, semantic "
                "cleaning, transformation and combination "
                "inside the Preparation workflow before "
                "selecting the final analysis output. "
                "Unsupported post-validation field(s): "
                f"{sorted(supplied_fields)}"
            )
        )


# ============================================================
# CONTROLLED CLEANING — ANALYSIS INPUT
# ============================================================

def parse_approved_cleaning_action_ids(
    raw_value: (
        str
        | None
    ),
) -> set[
    str
]:
    if raw_value is None:
        return set()


    normalized = (
        raw_value.strip()
    )


    if not normalized:
        return set()


    try:
        parsed = json.loads(
            normalized
        )


    except json.JSONDecodeError as error:
        raise ValueError(
            "approved_action_ids_json must "
            "be a valid JSON array."
        ) from error


    if not isinstance(
        parsed,
        list,
    ):
        raise ValueError(
            "approved_action_ids_json must "
            "contain a JSON array."
        )


    action_ids: set[
        str
    ] = set()


    for value in parsed:
        if not isinstance(
            value,
            str,
        ):
            raise ValueError(
                "Every approved cleaning "
                "action id must be a string."
            )


        action_id = (
            value.strip()
        )


        if not action_id:
            raise ValueError(
                "Approved cleaning action "
                "ids cannot be empty."
            )


        action_ids.add(
            action_id
        )


    return action_ids


def apply_approved_cleaning_to_records(
    *,
    source_dataset_records: list[
        dict[
            str,
            Any,
        ]
    ],

    approved_action_ids: set[
        str
    ],
) -> tuple[
    list[
        dict[
            str,
            Any,
        ]
    ],

    DataQualityReport
    | None,

    CleaningPlan
    | None,

    CleaningExecutionResult
    | None,
]:
    if not approved_action_ids:
        return (
            source_dataset_records,
            None,
            None,
            None,
        )


    quality_report = (
        build_data_quality_report(
            source_dataset_records
        )
    )


    dataset_frames = {
        str(
            record[
                "dataset_id"
            ]
        ):
            record[
                "dataframe"
            ]

        for record
        in source_dataset_records
    }


    cleaning_plan = (
        build_cleaning_plan(
            quality_report,
            dataset_frames,
        )
    )


    (
        derived_frames,
        cleaning_execution,
    ) = execute_cleaning_plan(
        plan=
            cleaning_plan,

        dataset_frames=
            dataset_frames,

        approved_action_ids=
            approved_action_ids,
    )


    provenance_by_dataset = {
        item.dataset_id:
            item

        for item
        in cleaning_execution.provenance
    }


    prepared_records: list[
        dict[
            str,
            Any,
        ]
    ] = []


    for record in (
        source_dataset_records
    ):
        dataset_id = str(
            record[
                "dataset_id"
            ]
        )


        derived_dataframe = (
            derived_frames.get(
                dataset_id
            )
        )


        if derived_dataframe is None:
            raise RuntimeError(
                "Controlled cleaning did not "
                "return a derived DataFrame for "
                f"{dataset_id}."
            )


        provenance = (
            provenance_by_dataset.get(
                dataset_id
            )
        )


        if provenance is None:
            raise RuntimeError(
                "Controlled cleaning did not "
                "return provenance for "
                f"{dataset_id}."
            )


        prepared_record = dict(
            record
        )


        prepared_record[
            "dataframe"
        ] = derived_dataframe


        prepared_record[
            "controlled_cleaning"
        ] = {
            "rule_version":
                cleaning_execution
                .rule_version,

            "rows_before":
                provenance
                .rows_before,

            "rows_after":
                provenance
                .rows_after,

            "columns_before":
                provenance
                .columns_before,

            "columns_after":
                provenance
                .columns_after,

            "source_fingerprint":
                provenance
                .source_fingerprint,

            "derived_fingerprint":
                provenance
                .derived_fingerprint,

            "applied_action_ids":
                list(
                    provenance
                    .applied_action_ids
                ),

            "skipped_action_ids":
                list(
                    provenance
                    .skipped_action_ids
                ),
        }


        prepared_records.append(
            prepared_record
        )


    return (
        prepared_records,
        quality_report,
        cleaning_plan,
        cleaning_execution,
    )


def append_controlled_cleaning_note(
    *,
    report: UnifiedAnalysisReport,

    cleaning_execution: (
        CleaningExecutionResult
        | None
    ),
) -> UnifiedAnalysisReport:
    if cleaning_execution is None:
        return report


    rows_before = sum(
        item.rows_before

        for item
        in cleaning_execution.provenance
    )


    rows_after = sum(
        item.rows_after

        for item
        in cleaning_execution.provenance
    )


    note = (
        "Nettoyage contrôlé avant analyse : "
        f"{cleaning_execution.applied_action_count} "
        "action(s) déterministe(s) appliquée(s), "
        f"{rows_before} ligne(s) avant préparation, "
        f"{rows_after} ligne(s) analysée(s). "
        "Les DataFrames source n'ont pas été modifiés. "
        "Moteur : "
        f"{cleaning_execution.rule_version}."
    )


    if (
        note
        not in report.methodology_notes
    ):
        report.methodology_notes.append(
            note
        )


    return report


# ============================================================
# CONTROLLED SEMANTIC CLEANING — ANALYSIS INPUT
# ============================================================

def parse_semantic_decisions_json(
    raw_value: (
        str
        | None
    ),
) -> (
    RawSemanticReviewResponse
    | None
):
    if raw_value is None:
        return None

    normalized = raw_value.strip()

    if not normalized:
        return None

    try:
        parsed = json.loads(
            normalized
        )
    except json.JSONDecodeError as error:
        raise ValueError(
            "semantic_decisions_json must be valid JSON."
        ) from error

    if isinstance(
        parsed,
        dict,
    ):
        parsed = parsed.get(
            "decisions"
        )

    if not isinstance(
        parsed,
        list,
    ):
        raise ValueError(
            "semantic_decisions_json must contain a decisions array."
        )

    decisions: list[
        RawSemanticDecision
    ] = []

    for item in parsed:
        if not isinstance(
            item,
            dict,
        ):
            raise ValueError(
                "Every semantic decision must be a JSON object."
            )

        decisions.append(
            RawSemanticDecision.model_validate(
                item
            )
        )

    return RawSemanticReviewResponse(
        decisions=decisions
    )


def parse_approved_semantic_choices_json(
    raw_value: (
        str
        | None
    ),
) -> list[
    SemanticCleaningChoice
]:
    if raw_value is None:
        return []

    normalized = raw_value.strip()

    if not normalized:
        return []

    try:
        parsed = json.loads(
            normalized
        )
    except json.JSONDecodeError as error:
        raise ValueError(
            "approved_semantic_choices_json must be valid JSON."
        ) from error

    if not isinstance(
        parsed,
        list,
    ):
        raise ValueError(
            "approved_semantic_choices_json must contain a JSON array."
        )

    choices: list[
        SemanticCleaningChoice
    ] = []

    for item in parsed:
        if not isinstance(
            item,
            dict,
        ):
            raise ValueError(
                "Every approved semantic choice must be a JSON object."
            )

        choices.append(
            SemanticCleaningChoice.model_validate(
                item
            )
        )

    return choices


def apply_approved_semantic_cleaning_to_records(
    *,
    prepared_dataset_records: list[
        dict[
            str,
            Any,
        ]
    ],
    semantic_decisions: (
        RawSemanticReviewResponse
        | None
    ),
    approved_semantic_choices: list[
        SemanticCleaningChoice
    ],
) -> tuple[
    list[
        dict[
            str,
            Any,
        ]
    ],
    SemanticCleaningPlan
    | None,
    SemanticCleaningExecutionResult
    | None,
]:
    if (
        semantic_decisions is None
        and
        not approved_semantic_choices
    ):
        return (
            prepared_dataset_records,
            None,
            None,
        )

    if semantic_decisions is None:
        raise ValueError(
            "approved semantic choices require semantic_decisions_json."
        )

    dataset_frames = {
        str(
            record[
                "dataset_id"
            ]
        ):
            record[
                "dataframe"
            ]
        for record in prepared_dataset_records
    }

    quality_report = build_data_quality_report(
        prepared_dataset_records
    )

    candidates = build_semantic_review_candidates(
        quality_report=quality_report,
        dataset_frames=dataset_frames,
    )

    validated_decisions = validate_semantic_review_response(
        raw_response=semantic_decisions,
        candidates=candidates,
        dataset_frames=dataset_frames,
    )

    semantic_plan = build_semantic_cleaning_plan(
        validated_decisions
    )

    (
        derived_frames,
        semantic_execution,
    ) = execute_semantic_cleaning_plan(
        plan=semantic_plan,
        dataset_frames=dataset_frames,
        approved_choices=approved_semantic_choices,
    )

    provenance_by_dataset = {
        item.dataset_id:
            item
        for item in semantic_execution.provenance
    }

    semantic_records: list[
        dict[
            str,
            Any,
        ]
    ] = []

    for record in prepared_dataset_records:
        dataset_id = str(
            record[
                "dataset_id"
            ]
        )

        dataframe = derived_frames.get(
            dataset_id
        )

        if dataframe is None:
            raise RuntimeError(
                "Semantic cleaning did not return a derived DataFrame for "
                f"{dataset_id}."
            )

        provenance = provenance_by_dataset.get(
            dataset_id
        )

        if provenance is None:
            raise RuntimeError(
                "Semantic cleaning did not return provenance for "
                f"{dataset_id}."
            )

        semantic_record = dict(
            record
        )

        semantic_record[
            "dataframe"
        ] = dataframe

        semantic_record[
            "semantic_cleaning"
        ] = {
            "rule_version":
                semantic_execution.rule_version,
            "rows_before":
                provenance.rows_before,
            "rows_after":
                provenance.rows_after,
            "source_fingerprint":
                provenance.source_fingerprint,
            "derived_fingerprint":
                provenance.derived_fingerprint,
            "applied_action_ids":
                list(
                    provenance.applied_action_ids
                ),
            "changed_cell_count":
                provenance.changed_cell_count,
        }

        semantic_records.append(
            semantic_record
        )

    return (
        semantic_records,
        semantic_plan,
        semantic_execution,
    )


def append_semantic_cleaning_note(
    *,
    report: UnifiedAnalysisReport,
    semantic_execution: (
        SemanticCleaningExecutionResult
        | None
    ),
) -> UnifiedAnalysisReport:
    if semantic_execution is None:
        return report

    note = (
        "Nettoyage sémantique avant analyse : "
        f"{semantic_execution.applied_action_count} "
        "fusion(s) confirmée(s) par l'utilisateur et exécutée(s) par Python, "
        f"{semantic_execution.changed_cell_count} "
        "cellule(s) modifiée(s). "
        "Les DataFrames précédents n'ont pas été mutés. "
        "Moteur : "
        f"{semantic_execution.rule_version}."
    )

    if note not in report.methodology_notes:
        report.methodology_notes.append(
            note
        )

    return report


# ============================================================
# DERIVED-DATASET DISCOVERY MERGE
# ============================================================

def merge_derived_single_dataset_discovery(
    *,
    source_discovery: AnalysisDiscoveryReport,

    derived_discovery: AnalysisDiscoveryReport,

    derived_datasets: list[
        dict[
            str,
            Any,
        ]
    ],
) -> AnalysisDiscoveryReport:
    """
    Merge safe single-dataset discoveries from
    analytical views into the source discovery.

    Derived cross-dataset relationships are never
    reinjected into the main relationship graph.
    """

    derived_map = {
        str(
            dataset[
                "dataset_id"
            ]
        ):
            dataset

        for dataset
        in derived_datasets
    }


    derived_candidates = []


    for candidate in (
        derived_discovery.candidates
    ):
        if (
            candidate.scope
            !=
            "single_dataset"
        ):
            continue


        if (
            len(
                candidate.dataset_ids
            )
            !=
            1
        ):
            continue


        dataset_id = str(
            candidate.dataset_ids[
                0
            ]
        )


        dataset_record = (
            derived_map.get(
                dataset_id
            )
        )


        if dataset_record is None:
            continue


        candidate.observed_signals[
            "analytical_view"
        ] = {
            "is_derived":
                True,

            "derivation_type":
                dataset_record.get(
                    "derivation_type"
                ),

            "derivation_depth":
                dataset_record.get(
                    "derivation_depth"
                ),

            "source_dataset_ids":
                list(
                    dataset_record.get(
                        "source_dataset_ids",
                        [],
                    )
                ),

            "provenance":
                dataset_record.get(
                    "provenance",
                    {},
                ),

            "rule_version":
                dataset_record.get(
                    "analytical_view_rule_version",
                    ANALYTICAL_VIEW_RULE_VERSION,
                ),
        }


        derived_candidates.append(
            candidate
        )


    selected = {}


    for candidate in [
        *source_discovery.candidates,
        *derived_candidates,
    ]:
        key = (
            candidate.redundancy_key
        )


        existing = (
            selected.get(
                key
            )
        )


        if (
            existing is None
            or
            candidate.priority_score
            >
            existing.priority_score
        ):
            selected[
                key
            ] = candidate


    combined_candidates = list(
        selected.values()
    )


    combined_candidates.sort(
        key=lambda candidate: (
            candidate.priority_score,

            1
            if (
                candidate.scope
                ==
                "cross_dataset"
            )
            else
            0,
        ),
        reverse=True,
    )


    source_discovery.candidates = (
        combined_candidates
    )


    source_discovery.candidate_count = (
        len(
            combined_candidates
        )
    )


    source_discovery.single_dataset_candidate_count = sum(
        1

        for candidate
        in combined_candidates

        if (
            candidate.scope
            ==
            "single_dataset"
        )
    )


    source_discovery.cross_dataset_candidate_count = sum(
        1

        for candidate
        in combined_candidates

        if (
            candidate.scope
            ==
            "cross_dataset"
        )
    )


    source_discovery.discovery_notes.append(
        (
            f"{len(derived_datasets)} "
            "internal analytical view(s) were "
            "materialized before secondary "
            "discovery."
        )
    )


    source_discovery.discovery_notes.append(
        (
            f"{len(derived_candidates)} "
            "single-dataset candidate(s) from "
            "analytical views were retained "
            "after the derived-view and feature-"
            "lineage policies."
        )
    )


    source_discovery.discovery_notes.append(
        (
            "Cross-dataset candidates produced "
            "from derived views were excluded."
        )
    )


    source_discovery.discovery_notes.append(
        (
            "Analytical View Builder version: "
            f"{ANALYTICAL_VIEW_RULE_VERSION}."
        )
    )


    return source_discovery


# ============================================================
# ANALYTICAL VIEW PREPARATION
# ============================================================

def prepare_analysis_datasets(
    *,
    source_datasets: list[
        dict[
            str,
            Any,
        ]
    ],

    objective: (
        str
        | None
    ),

    include_requested_context: bool = False,
) -> tuple[
    AnalysisDiscoveryReport,

    list[
        dict[
            str,
            Any,
        ]
    ],
]:
    # ========================================================
    # 1. SOURCE DISCOVERY
    # ========================================================

    source_discovery = (
        discover_analyses(
            datasets=
                source_datasets,

            objective=
                objective,
        )
    )


    # ========================================================
    # 2. ANALYTICAL VIEW BUILDING
    # ========================================================

    try:
        view_build = (
            build_analytical_views(
                source_datasets,

                include_requested_context=
                    include_requested_context,
            )
        )


    except Exception:
        source_discovery.discovery_notes.append(
            (
                "Analytical View Builder was "
                "disabled because its controlled "
                "construction failed."
            )
        )


        return (
            source_discovery,
            source_datasets,
        )


    source_discovery.discovery_notes.extend(
        [
            (
                "Analytical View Builder — "
                + note
            )

            for note
            in view_build.notes
        ]
    )


    derived_datasets = (
        view_build.derived_datasets
    )


    if not derived_datasets:
        source_discovery.discovery_notes.append(
            (
                "No safe analytical view was "
                "materialized."
            )
        )


        return (
            source_discovery,
            source_datasets,
        )


    discoverable_derived_datasets = [
        dataset

        for dataset
        in derived_datasets

        if bool(
            dataset.get(
                "discoverable",
                True,
            )
        )
    ]


    requested_only_count = (
        len(
            derived_datasets
        )
        -
        len(
            discoverable_derived_datasets
        )
    )


    if requested_only_count:
        source_discovery.discovery_notes.append(
            (
                f"{requested_only_count} requested-"
                "only analytical context view(s) "
                "were excluded from exploratory "
                "Discovery."
            )
        )


    if not discoverable_derived_datasets:
        analysis_datasets = [
            *source_datasets,
            *derived_datasets,
        ]


        return (
            source_discovery,
            analysis_datasets,
        )


    # ========================================================
    # 3. SECONDARY DERIVED DISCOVERY
    # ========================================================

    derived_discovery = (
        discover_analyses(
            datasets=
                discoverable_derived_datasets,

            objective=
                objective,
        )
    )


    # ========================================================
    # 4. DERIVED DISCOVERY POLICY
    # ========================================================

    derived_discovery = (
        apply_derived_discovery_policy(
            derived_discovery,

            derived_datasets=
                discoverable_derived_datasets,

            objective=
                objective,
        )
    )


    # ========================================================
    # 5. FEATURE LINEAGE POLICY
    # ========================================================

    derived_discovery = (
        apply_feature_lineage_policy(
            derived_discovery,

            derived_datasets=
                discoverable_derived_datasets,
        )
    )


    # ========================================================
    # 6. MERGE
    # ========================================================

    discovery = (
        merge_derived_single_dataset_discovery(
            source_discovery=
                source_discovery,

            derived_discovery=
                derived_discovery,

            derived_datasets=
                discoverable_derived_datasets,
        )
    )


    # ========================================================
    # 7. INTERNAL EXECUTION DATASETS
    # ========================================================

    analysis_datasets = [
        *source_datasets,
        *derived_datasets,
    ]


    return (
        discovery,
        analysis_datasets,
    )


# ============================================================
# AI PLANNER ANALYTICAL DATASET UNIVERSE
# ============================================================

def prepare_ai_planner_dataset_universe(
    *,
    source_dataset_records: list[
        dict[
            str,
            Any,
        ]
    ],

    objective: (
        str
        | None
    ),
) -> tuple[
    list[
        dict[
            str,
            Any,
        ]
    ],

    PlannerCatalog,
]:
    """
    Build the exact internal dataset universe shared by the AI
    planner and its deterministic executor.

    Architecture invariant:

        validated Preparation output
            -> source_dataset_records
            -> prepare_analysis_datasets(...)
            -> analysis_datasets
            -> planner catalog
            -> validated AI contract
            -> deterministic execution on analysis_datasets

    The browser never supplies authoritative analytical views.

    Analytical views remain server-owned, deterministic internal
    derivatives of the validated Preparation output.

    Requested-only document context is deliberately excluded
    from this direct AI planning path. The contextualized
    document workflow prepares that context separately with
    include_requested_context=True.

    Returning the same analysis_datasets used to build the
    catalog is important: a validated contract may target a
    derived analytical view, so the executor must receive that
    exact same allowed dataset universe.
    """

    normalized_objective = (
        normalize_objective(
            objective
        )
    )


    (
        _,
        analysis_datasets,
    ) = prepare_analysis_datasets(
        source_datasets=
            source_dataset_records,

        objective=
            normalized_objective,

        include_requested_context=
            False,
    )


    catalog = (
        planner_catalog_from_dataset_records(
            analysis_datasets
        )
    )


    return (
        analysis_datasets,
        catalog,
    )


# ============================================================
# UNIFIED ANALYSIS FROM PREPARED DATASETS
# ============================================================

def run_unified_analysis_from_prepared_records(
    *,
    source_dataset_records: list[
        dict[
            str,
            Any,
        ]
    ],

    discovery: AnalysisDiscoveryReport,

    analysis_datasets: list[
        dict[
            str,
            Any,
        ]
    ],
) -> UnifiedAnalysisReport:
    """
    Execute the deterministic unified-analysis pipeline from
    already prepared datasets.

    Discovery remains broad and auditable.

    Only the deterministic prioritization shortlist is sent to
    exploratory executors and ranking.

    Explicit Requested Analysis is intentionally outside this
    function and therefore remains unaffected by exploratory
    prioritization.
    """

    # ========================================================
    # 1. EXPLORATORY PRIORITIZATION
    # ========================================================

    prioritization = (
        prioritize_analysis_discovery(
            discovery,

            datasets=
                analysis_datasets,
        )
    )


    execution_discovery = (
        build_prioritized_execution_discovery(
            source_discovery=
                discovery,

            prioritization=
                prioritization,
        )
    )


    # ========================================================
    # 2. SINGLE EXECUTION — SHORTLIST ONLY
    # ========================================================

    single_execution = (
        execute_single_dataset_discovery(
            discovery=
                execution_discovery,

            datasets=
                analysis_datasets,
        )
    )


    # ========================================================
    # 3. AGGREGATE BREAKDOWN EXECUTION — SHORTLIST ONLY
    # ========================================================

    single_execution = (
        inject_aggregate_breakdown_execution(
            single_execution,

            discovery=
                execution_discovery,

            datasets=
                analysis_datasets,
        )
    )


    # ========================================================
    # 4. CROSS EXECUTION — SHORTLIST ONLY
    # ========================================================

    cross_execution = (
        execute_cross_dataset_discovery(
            discovery=
                execution_discovery,

            datasets=
                analysis_datasets,
        )
    )


    # ========================================================
    # 5. NORMAL UNIFIED RANKING — SHORTLIST ONLY
    # ========================================================

    ranking = (
        rank_unified_analysis(
            discovery=
                execution_discovery,

            single_execution=
                single_execution,

            cross_execution=
                cross_execution,

            datasets=
                analysis_datasets,
        )
    )


    # ========================================================
    # 6. AGGREGATE BREAKDOWN RANKING — SHORTLIST ONLY
    # ========================================================

    ranking = (
        apply_aggregate_breakdown_ranking_policy(
            ranking,

            discovery=
                execution_discovery,
        )
    )


    # ========================================================
    # 7. FEATURE LINEAGE RANKING — SHORTLIST ONLY
    # ========================================================

    ranking = (
        apply_feature_lineage_ranking_policy(
            ranking,

            discovery=
                execution_discovery,
        )
    )


    # ========================================================
    # 8. USER-FACING REPORT — EXECUTED SHORTLIST
    # ========================================================
    #
    # Deferred candidates must not be represented as execution
    # failures merely because they were not selected by the
    # exploratory execution budget.
    # ========================================================

    report = (
        compose_unified_report(
            discovery=
                execution_discovery,

            single_execution=
                single_execution,

            cross_execution=
                cross_execution,

            ranking=
                ranking,

            datasets=
                source_dataset_records,

            title=
                "Analyse DataLens",
        )
    )


    # ========================================================
    # 9. RESTORE BROAD DISCOVERY INVENTORY
    # ========================================================

    report.inventory.discovered_analysis_count = (
        discovery.candidate_count
    )


    # ========================================================
    # 10. FINAL FAMILY NORMALIZATION — SHORTLIST ONLY
    # ========================================================

    report = (
        normalize_report_aggregate_families(
            report,

            discovery=
                execution_discovery,
        )
    )


    # ========================================================
    # 11. PRIORITIZATION TRACEABILITY
    # ========================================================

    prioritization_note = (
        "Priorisation exploratoire avant exécution : "
        f"{prioritization.selected_count} analyse(s) retenue(s) "
        f"sur {prioritization.discovered_count} découverte(s), "
        f"{prioritization.deferred_count} différée(s) et "
        f"{prioritization.rejected_count} rejetée(s). "
        "La Discovery complète reste comptabilisée dans "
        "l'inventaire public. Les analyses explicitement "
        "demandées sont exécutées séparément par le flux "
        "Requested Analysis et ne sont pas limitées par ce "
        "budget exploratoire. "
        "Règles : "
        f"{ANALYSIS_PRIORITIZATION_RULE_VERSION} + "
        f"{ANALYTICAL_VALUE_GUARD_RULE_VERSION}."
    )


    if (
        prioritization_note
        not in report.methodology_notes
    ):
        report.methodology_notes.append(
            prioritization_note
        )


    # ========================================================
    # 12. INTERNAL VIEW TRACEABILITY
    # ========================================================

    internal_view_count = (
        len(
            analysis_datasets
        )
        -
        len(
            source_dataset_records
        )
    )


    note = (
        f"{internal_view_count} vue(s) analytique(s) "
        "interne(s) ont été utilisées pour "
        "l'analyse sans être comptées comme "
        "datasets source."
    )


    if (
        note
        not in report.methodology_notes
    ):
        report.methodology_notes.append(
            note
        )


    return report


# ============================================================
# UNIFIED ANALYSIS FROM LOADED DATASETS
# ============================================================

def run_unified_analysis_from_records(
    *,
    source_dataset_records: list[
        dict[
            str,
            Any,
        ]
    ],

    objective: (
        str
        | None
    ),
) -> UnifiedAnalysisReport:
    """
    Public deterministic analysis entry point.

    Analytical datasets are prepared once and
    delegated to the prepared-data execution
    pipeline.
    """

    normalized_objective = (
        normalize_objective(
            objective
        )
    )


    (
        discovery,
        analysis_datasets,
    ) = prepare_analysis_datasets(
        source_datasets=
            source_dataset_records,

        objective=
            normalized_objective,
    )


    return (
        run_unified_analysis_from_prepared_records(
            source_dataset_records=
                source_dataset_records,

            discovery=
                discovery,

            analysis_datasets=
                analysis_datasets,
        )
    )


def run_unified_analysis_from_records_with_prioritization_audit(
    *,
    source_dataset_records: list[
        dict[
            str,
            Any,
        ]
    ],

    objective: (
        str
        | None
    ),
) -> tuple[
    UnifiedAnalysisReport,
    AnalysisPrioritizationAudit,
]:
    """
    Execute the deterministic analysis and expose the exact
    prioritization decisions as a compact public audit view.

    The audit recomputes only the deterministic prioritization
    decision layer from the same Discovery + prepared datasets.
    Statistical execution is not repeated.
    """

    normalized_objective = (
        normalize_objective(
            objective
        )
    )


    (
        discovery,
        analysis_datasets,
    ) = prepare_analysis_datasets(
        source_datasets=
            source_dataset_records,

        objective=
            normalized_objective,
    )


    report = (
        run_unified_analysis_from_prepared_records(
            source_dataset_records=
                source_dataset_records,

            discovery=
                discovery,

            analysis_datasets=
                analysis_datasets,
        )
    )


    prioritization = (
        prioritize_analysis_discovery(
            discovery,

            datasets=
                analysis_datasets,
        )
    )


    audit = (
        build_analysis_prioritization_audit(
            prioritization
        )
    )


    return (
        report,
        audit,
    )


# ============================================================
# UNIFIED ANALYSIS FROM UPLOADS
# ============================================================

def run_unified_analysis_from_uploads(
    *,
    dataset_files: list[
        UploadFile
    ],

    objective: (
        str
        | None
    ),

    workflow_id: (
        str
        | None
    ) = None,

    approved_action_ids_json: (
        str
        | None
    ) = None,

    semantic_decisions_json: (
        str
        | None
    ) = None,

    approved_semantic_choices_json: (
        str
        | None
    ) = None,
) -> RoutedUnifiedAnalysisReport:
    """
    Execute the standard unified analysis.

    Two execution modes intentionally coexist during the
    migration:

    WORKFLOW-BACKED MODE
        When workflow_id is supplied, Analysis ignores the
        browser-uploaded dataset content and loads the exact
        validated analysis_output_dataset_ids from the
        Preparation Artifact Store.

        No cleaning or semantic-cleaning mutation is allowed
        after VALIDATE.

    LEGACY INTERNAL MODE
        When workflow_id is omitted, the historical upload
        pipeline remains available for internal callers and
        experimental endpoints that have not yet migrated to
        Preparation-backed execution.
    """

    if (
        workflow_id
        is not None
    ):
        # ====================================================
        # SERVER-OWNED PREPARATION HANDOFF
        # ====================================================

        handoff = (
            load_validated_analysis_input_for_http(
                workflow_id=
                    workflow_id
            )
        )


        reject_post_validation_preparation_overrides(
            approved_action_ids_json=
                approved_action_ids_json,

            semantic_decisions_json=
                semantic_decisions_json,

            approved_semantic_choices_json=
                approved_semantic_choices_json,
        )


        analysis_source_records = [
            dict(
                record
            )

            for record
            in handoff.dataset_records
        ]


        # The multipart dataset_files field is retained only
        # for HTTP contract compatibility during the frontend
        # migration.
        #
        # Its bytes are deliberately never read in this mode.
        _ = (
            dataset_files
        )


        cleaning_execution = (
            None
        )


        semantic_execution = (
            None
        )


    else:
        # ====================================================
        # LEGACY UPLOAD-BACKED INTERNAL MODE
        # ====================================================

        (
            _,
            source_dataset_records,
        ) = load_uploaded_dataset_bundle(
            dataset_files
        )


        approved_action_ids = (
            parse_approved_cleaning_action_ids(
                approved_action_ids_json
            )
        )


        (
            analysis_source_records,
            _,
            _,
            cleaning_execution,
        ) = apply_approved_cleaning_to_records(
            source_dataset_records=
                source_dataset_records,

            approved_action_ids=
                approved_action_ids,
        )


        semantic_decisions = (
            parse_semantic_decisions_json(
                semantic_decisions_json
            )
        )


        semantic_choices = (
            parse_approved_semantic_choices_json(
                approved_semantic_choices_json
            )
        )


        (
            analysis_source_records,
            _,
            semantic_execution,
        ) = apply_approved_semantic_cleaning_to_records(
            prepared_dataset_records=
                analysis_source_records,

            semantic_decisions=
                semantic_decisions,

            approved_semantic_choices=
                semantic_choices,
        )


    entity_outlier_finding = (
        build_entity_outlier_finding_if_requested(
            objective=
                objective,

            source_dataset_records=
                analysis_source_records,
        )
    )


    (
        report,
        prioritization_audit,
    ) = (
        run_unified_analysis_from_records_with_prioritization_audit(
            source_dataset_records=
                analysis_source_records,

            objective=
                objective,
        )
    )


    report = (
        append_controlled_cleaning_note(
            report=
                report,

            cleaning_execution=
                cleaning_execution,
        )
    )


    report = (
        append_semantic_cleaning_note(
            report=
                report,

            semantic_execution=
                semantic_execution,
        )
    )


    return (
        build_routed_unified_analysis_report(
            report=
                report,

            entity_outlier_finding=
                entity_outlier_finding,

            prioritization_audit=
                prioritization_audit,
        )
    )



# ============================================================
# AI ANALYTICAL PLANNING PREVIEW
# ============================================================

@router.post(
    "/planning/ai-preview",
    response_model=
        AIPlannerReport,
)
def preview_ai_analytical_plan(
    dataset_files: (
        list[
            UploadFile
        ]
        | None
    ) = File(
        default=None,
    ),

    workflow_id: str = Form(
        ...,
        min_length=1,
    ),

    objective: str = Form(
        ...,
        min_length=1,
    ),

    planner_model: str = Form(
        default=
            DEFAULT_AI_PLANNER_MODEL,

        min_length=1,
    ),
) -> AIPlannerReport:
    """
    Route the analytical request through DataLens from the
    exact server-owned Preparation output authorized for
    Analysis.

    Generic supported intents may be expanded deterministically
    by Python from the centrally typed dataset catalog.
    Other requests fall back to the local LLM planner and remain
    deterministically validated by Python.

    This endpoint is PREVIEW-ONLY.

    It does not execute an analysis and it does not allow the
    LLM to modify data, invent joins or create derived variables.

    The multipart dataset_files field remains optional for
    backward compatibility. Its bytes are never read after
    VALIDATE; workflow_id resolves the server-owned input.
    """

    normalized_objective = (
        normalize_objective(
            objective
        )
    )


    if (
        normalized_objective
        is None
    ):
        raise HTTPException(
            status_code=422,
            detail=(
                "L'objectif utilisateur "
                "ne peut pas être vide."
            ),
        )


    try:
        # ====================================================
        # SERVER-OWNED PREPARATION HANDOFF
        # ====================================================

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


        # Optional multipart compatibility only. Browser
        # dataset bytes are not authoritative after
        # Preparation VALIDATE.
        _ = (
            dataset_files
        )


        (
            analysis_datasets,
            catalog,
        ) = prepare_ai_planner_dataset_universe(
            source_dataset_records=
                source_dataset_records,

            objective=
                normalized_objective,
        )


        # Preview does not execute, but the catalog must be
        # built from the same analytical universe that execution
        # would receive.
        _ = (
            analysis_datasets
        )


        return (
            plan_analyses_with_intent_routing(
                objective=
                    normalized_objective,

                catalog=
                    catalog,

                model=
                    planner_model,
            )
        )


    except ValueError as error:
        raise HTTPException(
            status_code=422,
            detail=str(
                error
            ),
        ) from error


    except RuntimeError as error:
        raise HTTPException(
            status_code=503,
            detail=(
                "Local AI processing is unavailable or returned an invalid response."
            ),
        ) from error


# ============================================================
# AI TOOL ORCHESTRATION — EXPLICIT EXECUTION
# ============================================================

@router.post(
    "/planning/ai-tool-run",
    response_model=
        AIToolOrchestrationReport,
)
def run_ai_analytical_tool(
    dataset_files: (
        list[
            UploadFile
        ]
        | None
    ) = File(
        default=None,
    ),

    workflow_id: str = Form(
        ...,
        min_length=1,
    ),

    objective: str = Form(
        ...,
        min_length=1,
    ),

    planner_model: str = Form(
        default=
            DEFAULT_AI_PLANNER_MODEL,

        min_length=1,
    ),

    approved_action_ids_json: (
        str
        | None
    ) = Form(
        default=None,
    ),

    semantic_decisions_json: (
        str
        | None
    ) = Form(
        default=None,
    ),

    approved_semantic_choices_json: (
        str
        | None
    ) = Form(
        default=None,
    ),
) -> AIToolOrchestrationReport:
    """
    Experimental AI-tool orchestration endpoint operating on
    the exact validated Preparation output.

    Flow:

    1. load the server-owned validated analysis artifact(s);
    2. build the centrally typed planner catalog from the exact
       DataFrames that will be analyzed;
    3. route generic intents through deterministic Python or
       fall back to the local LLM planner;
    4. validate every resulting contract with Python;
    5. map ONLY validated contracts to whitelisted
       deterministic analysis tools;
    6. execute the selected Python tool.

    The LLM never receives arbitrary Python execution rights.

    Cleaning, semantic cleaning, transformation and combination
    must already have been completed inside Preparation before
    VALIDATE. Post-validation Preparation overrides are refused.

    The multipart dataset_files field remains optional for
    backward compatibility. Its bytes are never read after
    VALIDATE; workflow_id resolves the server-owned input.
    """

    normalized_objective = (
        normalize_objective(
            objective
        )
    )


    if (
        normalized_objective
        is None
    ):
        raise HTTPException(
            status_code=422,
            detail=(
                "L'objectif utilisateur "
                "ne peut pas être vide."
            ),
        )


    try:
        # ====================================================
        # SERVER-OWNED PREPARATION HANDOFF
        # ====================================================

        handoff = (
            load_validated_analysis_input_for_http(
                workflow_id=
                    workflow_id
            )
        )


        reject_post_validation_preparation_overrides(
            approved_action_ids_json=
                approved_action_ids_json,

            semantic_decisions_json=
                semantic_decisions_json,

            approved_semantic_choices_json=
                approved_semantic_choices_json,
        )


        dataset_ingestion = (
            handoff.ingestion
        )


        source_dataset_records = [
            dict(
                record
            )

            for record
            in handoff.dataset_records
        ]


        # The ingestion read model is intentionally retained
        # because it belongs to the authoritative handoff.
        # Planning below uses the deterministic analytical
        # universe derived from dataset_records.
        _ = (
            dataset_ingestion
        )


        # Optional multipart compatibility only. Browser
        # dataset bytes are not authoritative after
        # Preparation VALIDATE.
        _ = (
            dataset_files
        )


        cleaning_execution = (
            None
        )


        semantic_execution = (
            None
        )


        (
            analysis_datasets,
            catalog,
        ) = prepare_ai_planner_dataset_universe(
            source_dataset_records=
                source_dataset_records,

            objective=
                normalized_objective,
        )


        planner_report = (
            plan_analyses_with_intent_routing(
                objective=
                    normalized_objective,

                catalog=
                    catalog,

                model=
                    planner_model,
            )
        )


        # ====================================================
        # OBJECTIVE COVERAGE EXECUTION GATE
        # DATALENS_OBJECTIVE_COVERAGE_EXECUTION_GATE_V0_1
        #
        # A Python-valid contract is not sufficient on its own.
        # Before any deterministic analytical tool may execute,
        # the union of validated contracts must preserve every
        # conservative metric / dimension requirement extracted
        # from the user objective.
        #
        # ObjectiveCoverageIncompleteError inherits ValueError
        # and is therefore mapped by the existing endpoint
        # boundary to HTTP 422.
        # ====================================================

        require_objective_coverage(
            objective=
                normalized_objective,

            catalog=
                catalog,

            planner_report=
                planner_report,
        )


        orchestration_report = (
            execute_ai_planner_report(
                planner_report=
                    planner_report,

                datasets=
                    analysis_datasets,
            )
        )


        if (
            cleaning_execution
            is not None
        ):
            orchestration_report.notes.append(
                (
                    "Controlled cleaning before tool execution: "
                    f"{cleaning_execution.applied_action_count} "
                    "deterministic action(s) applied. "
                    f"Engine: {cleaning_execution.rule_version}."
                )
            )


        if (
            semantic_execution
            is not None
        ):
            orchestration_report.notes.append(
                (
                    "Controlled semantic cleaning before tool "
                    "execution was applied inside this endpoint."
                )
            )


        return (
            orchestration_report
        )


    except ValueError as error:
        raise HTTPException(
            status_code=422,
            detail=str(
                error
            ),
        ) from error


    except RuntimeError as error:
        raise HTTPException(
            status_code=503,
            detail=(
                "Local AI processing is unavailable or returned an invalid response."
            ),
        ) from error


# ============================================================
# AI NATIVE PIPELINE — INTENT ROUTER → QWEN TOOL CALL → PYTHON
# ============================================================

@router.post(
    "/planning/ai-native-run",
    response_model=
        AINativePipelineReport,
)
def run_ai_native_pipeline(
    request: FastAPIRequest = None,

    dataset_files: (
        list[
            UploadFile
        ]
        | None
    ) = File(
        default=None,
    ),

    workflow_id: str = Form(
        ...,
        min_length=1,
    ),

    objective: str = Form(
        ...,
        min_length=1,
    ),

    planner_model: str = Form(
        default=
            DEFAULT_AI_PLANNER_MODEL,

        min_length=1,
    ),

    tool_model: str = Form(
        default=
            DEFAULT_NATIVE_TOOL_MODEL,

        min_length=1,
    ),

    approved_action_ids_json: (
        str
        | None
    ) = Form(
        default=None,
    ),

    semantic_decisions_json: (
        str
        | None
    ) = Form(
        default=None,
    ),

    approved_semantic_choices_json: (
        str
        | None
    ) = Form(
        default=None,
    ),
) -> AINativePipelineReport:
    """
    Full experimental local AI execution pipeline operating on
    the exact validated Preparation output.

    1. DataLens loads the server-owned validated analysis
       artifact(s).
    2. DataLens builds a centrally typed planner catalog from
       the exact DataFrames that will be analyzed.
    3. Generic supported intents are expanded by Python;
       other requests fall back to the local LLM planner.
    4. Python validates the resulting analytical contracts.
    5. Qwen requests the whitelisted function through
       Ollama native function calling.
    6. Python validates that tool request against the
       already validated contract.
    7. The deterministic DataLens executor computes
       the statistical result.

    Cleaning, semantic cleaning, transformation and combination
    must already have been completed inside Preparation before
    VALIDATE. Post-validation Preparation overrides are refused.

    The native analytical family registry is validated
    deterministically before any execution.

    Observability v0.3 records a local JSONL decision trace
    containing schema metadata, planner decisions, tool calls
    and stage latencies. Raw dataset rows are not persisted
    in the trace.

    The multipart dataset_files field remains optional for
    backward compatibility. Its bytes are never read after
    VALIDATE; workflow_id resolves the server-owned input.
    """

    normalized_objective = (
        normalize_objective(
            objective
        )
    )


    if (
        normalized_objective
        is None
    ):
        raise HTTPException(
            status_code=422,
            detail=(
                "L'objectif utilisateur "
                "ne peut pas être vide."
            ),
        )


    trace_id = (
        new_ai_trace_id()
    )


    total_started_at = (
        perf_counter()
    )


    # Failure-safe observability state.
    #
    # These values intentionally exist before any runtime stage
    # starts so an early failure can still produce a valid
    # AITraceRecord without masking the original exception.
    catalog = None
    planner_report = None
    pipeline_report = None

    ingestion_ms = 0.0
    planner_ms = 0.0
    native_pipeline_ms = 0.0

    ingestion_started_at = None
    planner_started_at = None
    native_started_at = None

    failure_stage = (
        "preparation_handoff"
    )


    def write_failed_ai_trace_best_effort(
        error: Exception,
    ) -> None:
        """
        Persist a structured failed AI trace without ever
        changing the functional outcome of the endpoint.

        Observability is strictly best-effort:
        any failure while building or writing the trace is
        swallowed so the original application exception keeps
        its HTTP/error semantics.
        """

        try:
            failure_observed_at = (
                perf_counter()
            )


            failed_ingestion_ms = (
                ingestion_ms
            )

            failed_planner_ms = (
                planner_ms
            )

            failed_native_pipeline_ms = (
                native_pipeline_ms
            )


            # Preserve useful partial latency when a stage fails
            # before its normal timing assignment executes.
            if (
                failure_stage
                in {
                    "preparation_handoff",
                    "catalog",
                }
                and
                ingestion_started_at
                is not None
                and
                failed_ingestion_ms
                <= 0.0
            ):
                failed_ingestion_ms = (
                    (
                        failure_observed_at
                        -
                        ingestion_started_at
                    )
                    *
                    1000.0
                )


            if (
                failure_stage
                ==
                "planner"
                and
                planner_started_at
                is not None
                and
                failed_planner_ms
                <= 0.0
            ):
                failed_planner_ms = (
                    (
                        failure_observed_at
                        -
                        planner_started_at
                    )
                    *
                    1000.0
                )


            if (
                failure_stage
                ==
                "native_pipeline"
                and
                native_started_at
                is not None
                and
                failed_native_pipeline_ms
                <= 0.0
            ):
                failed_native_pipeline_ms = (
                    (
                        failure_observed_at
                        -
                        native_started_at
                    )
                    *
                    1000.0
                )


            failed_trace = (
                build_ai_trace(
                    trace_id=
                        trace_id,

                    workflow_id=
                        workflow_id,

                    objective=
                        normalized_objective,

                    catalog=
                        catalog,

                    planner_report=
                        planner_report,

                    pipeline_report=
                        pipeline_report,

                    ingestion_ms=
                        failed_ingestion_ms,

                    planner_ms=
                        failed_planner_ms,

                    native_pipeline_ms=
                        failed_native_pipeline_ms,

                    total_ms=(
                        (
                            failure_observed_at
                            -
                            total_started_at
                        )
                        *
                        1000.0
                    ),

                    run_status=
                        "failed",

                    failure={
                        "stage":
                            failure_stage,

                        "error_type":
                            type(
                                error
                            ).__name__,

                        "message_safe":
                            (
                                "DataLens AI execution "
                                "failed before completion."
                            ),
                    },
                )
            )


            # write_ai_trace() already reports persistence
            # failures through AITraceWriteResult. For a failed
            # application run there is deliberately no attempt
            # to alter the original response based on that
            # secondary observability result.
            write_ai_trace(
                failed_trace
            )


        except Exception:
            # Observability must never mask or replace the
            # original application error.
            return


    try:
        ingestion_started_at = (
            perf_counter()
        )


        # ====================================================
        # SERVER-OWNED PREPARATION HANDOFF
        # ====================================================

        handoff = (
            load_validated_analysis_input_for_http(
                workflow_id=
                    workflow_id
            )
        )


        # Publish runtime correlation only after the
        # authoritative server-owned Preparation handoff.
        #
        # request=None intentionally preserves historical
        # direct Python invocations used in deterministic tests.
        if request is not None:
            stamp_validated_runtime_workflow_id(
                scope=
                    request.scope,

                workflow_id=
                    workflow_id,
            )


        reject_post_validation_preparation_overrides(
            approved_action_ids_json=
                approved_action_ids_json,

            semantic_decisions_json=
                semantic_decisions_json,

            approved_semantic_choices_json=
                approved_semantic_choices_json,
        )


        dataset_ingestion = (
            handoff.ingestion
        )


        source_dataset_records = [
            dict(
                record
            )

            for record
            in handoff.dataset_records
        ]


        # The ingestion read model remains part of the exact
        # authoritative handoff. Internal analytical views are
        # derived deterministically from dataset_records below.
        _ = (
            dataset_ingestion
        )


        # Optional multipart compatibility only. Browser
        # dataset bytes are not authoritative after
        # Preparation VALIDATE.
        _ = (
            dataset_files
        )


        cleaning_execution = (
            None
        )


        semantic_execution = (
            None
        )


        failure_stage = (
            "catalog"
        )


        (
            analysis_datasets,
            catalog,
        ) = prepare_ai_planner_dataset_universe(
            source_dataset_records=
                source_dataset_records,

            objective=
                normalized_objective,
        )


        ingestion_ms = (
            (
                perf_counter()
                -
                ingestion_started_at
            )
            *
            1000.0
        )


        planner_started_at = (
            perf_counter()
        )


        failure_stage = (
            "planner"
        )


        planner_report = (
            plan_analyses_with_intent_routing(
                objective=
                    normalized_objective,

                catalog=
                    catalog,

                model=
                    planner_model,
            )
        )


        # ====================================================
        # OBJECTIVE COVERAGE EXECUTION GATE
        # DATALENS_OBJECTIVE_COVERAGE_EXECUTION_GATE_V0_1
        #
        # Keep the failure stage as "planner" until semantic
        # objective coverage has also passed. Qwen tool calling
        # must never receive an incomplete analytical plan.
        # ====================================================

        require_objective_coverage(
            objective=
                normalized_objective,

            catalog=
                catalog,

            planner_report=
                planner_report,
        )


        planner_ms = (
            (
                perf_counter()
                -
                planner_started_at
            )
            *
            1000.0
        )


        native_started_at = (
            perf_counter()
        )


        failure_stage = (
            "native_pipeline"
        )


        pipeline_report = (
            execute_native_ai_pipeline(
                planner_report=
                    planner_report,

                datasets=
                    analysis_datasets,

                tool_model=
                    tool_model,

                trace_id=
                    trace_id,
            )
        )


        native_pipeline_ms = (
            (
                perf_counter()
                -
                native_started_at
            )
            *
            1000.0
        )


        failure_stage = (
            "post_pipeline"
        )


        if (
            cleaning_execution
            is not None
        ):
            pipeline_report.notes.append(
                (
                    "Controlled cleaning before native AI execution: "
                    f"{cleaning_execution.applied_action_count} "
                    "deterministic action(s) applied. "
                    f"Engine: {cleaning_execution.rule_version}."
                )
            )


        if (
            semantic_execution
            is not None
        ):
            pipeline_report.notes.append(
                (
                    "Controlled semantic cleaning before native "
                    "AI execution was applied inside this endpoint."
                )
            )


        total_ms = (
            (
                perf_counter()
                -
                total_started_at
            )
            *
            1000.0
        )


        failure_stage = (
            "observability"
        )


        trace = (
            build_ai_trace(
                trace_id=
                    trace_id,

                workflow_id=
                    workflow_id,

                objective=
                    normalized_objective,

                catalog=
                    catalog,

                planner_report=
                    planner_report,

                pipeline_report=
                    pipeline_report,

                ingestion_ms=
                    ingestion_ms,

                planner_ms=
                    planner_ms,

                native_pipeline_ms=
                    native_pipeline_ms,

                total_ms=
                    total_ms,
            )
        )


        trace_write = (
            write_ai_trace(
                trace
            )
        )


        if (
            trace_write.enabled
            and
            not trace_write.written
        ):
            pipeline_report.notes.append(
                (
                    "Local AI observability trace "
                    "could not be persisted."
                )
            )


        return (
            pipeline_report
        )


    except HTTPException as error:
        write_failed_ai_trace_best_effort(
            error
        )

        # Preserve the exact existing HTTPException, including
        # status code and structured detail.
        raise


    except ValueError as error:
        write_failed_ai_trace_best_effort(
            error
        )

        raise HTTPException(
            status_code=422,
            detail=str(
                error
            ),
        ) from error


    except RuntimeError as error:
        write_failed_ai_trace_best_effort(
            error
        )

        raise HTTPException(
            status_code=503,
            detail=(
                "Local AI processing is unavailable or returned an invalid response."
            ),
        ) from error


    except Exception as error:
        write_failed_ai_trace_best_effort(
            error
        )

        # Unexpected errors keep their original server behavior.
        raise


# ============================================================
# AI OBSERVABILITY — LOCAL TRACE READ API
# ============================================================

@router.get(
    "/observability/metrics",
    response_model=
        AITraceMetricsResponse,
)
def read_local_ai_trace_metrics(
    response: Response,

    limit: int = Query(
        default=200,
        ge=1,
        le=5000,
    ),
) -> AITraceMetricsResponse:
    """
    Return aggregate local AI observability metrics.

    Metrics are computed from valid local traces only.
    No raw uploaded dataset rows are read or returned.
    """

    response.headers[
        "Cache-Control"
    ] = "no-store"


    try:
        return (
            get_ai_trace_metrics(
                limit=
                    limit
            )
        )


    except ValueError as error:
        raise HTTPException(
            status_code=422,
            detail=str(
                error
            ),
        ) from error


@router.get(
    "/observability/traces",
    response_model=
        AITraceListResponse,
)
def list_local_ai_traces(
    response: Response,

    limit: int = Query(
        default=20,
        ge=1,
        le=200,
    ),
) -> AITraceListResponse:
    """
    Return recent local AI decision-trace summaries.

    The newest trace is returned first.

    The trace store contains analytical metadata only.
    Raw uploaded dataset rows are not persisted by the
    observability writer.
    """

    response.headers[
        "Cache-Control"
    ] = "no-store"


    try:
        return (
            list_ai_traces(
                limit=
                    limit
            )
        )


    except ValueError as error:
        raise HTTPException(
            status_code=422,
            detail=str(
                error
            ),
        ) from error


@router.get(
    "/observability/traces/latest",
    response_model=
        AITraceRecord,
)
def read_latest_local_ai_trace(
    response: Response,
) -> AITraceRecord:
    """
    Return the latest valid local AI trace.
    """

    response.headers[
        "Cache-Control"
    ] = "no-store"


    trace = (
        get_latest_ai_trace()
    )


    if trace is None:
        raise HTTPException(
            status_code=404,
            detail=(
                "Aucune trace IA locale "
                "n'est disponible."
            ),
        )


    return trace


@router.get(
    "/observability/traces/{trace_id}",
    response_model=
        AITraceRecord,
)
def read_local_ai_trace(
    trace_id: str,
    response: Response,
) -> AITraceRecord:
    """
    Return one local AI trace by its stable trace_id.
    """

    response.headers[
        "Cache-Control"
    ] = "no-store"


    trace = (
        get_ai_trace(
            trace_id
        )
    )


    if trace is None:
        raise HTTPException(
            status_code=404,
            detail=(
                "Trace IA locale introuvable : "
                f"{trace_id}"
            ),
        )


    return trace


# ============================================================
# REQUEST TRACE EXPLORER
# ============================================================

@router.get(
    "/observability/requests/{request_id}",
    response_model=
        TraceExplorerResponse,
)
def read_local_request_trace(
    request_id: str,
    response: Response,
) -> TraceExplorerResponse:
    """
    Correlate one server-owned HTTP runtime request with any
    local AI traces created during the same request.

    The response intentionally contains only diagnostic
    metadata. It never exposes raw request/response bodies,
    incoming headers, client IPs, raw dataset rows, uploaded
    document contents, filesystem paths or raw model output.
    """

    response.headers[
        "Cache-Control"
    ] = "no-store"

    try:
        explorer = (
            get_request_trace_explorer(
                request_id
            )
        )

    except ValueError as error:
        raise HTTPException(
            status_code=422,
            detail=(
                "Invalid DataLens server-owned "
                "request identifier."
            ),
        ) from error

    except TraceExplorerReadError as error:
        raise HTTPException(
            status_code=503,
            detail=(
                "Local observability data "
                "is unavailable."
            ),
        ) from error

    if explorer is None:
        raise HTTPException(
            status_code=404,
            detail=(
                "Local request observability "
                "correlation was not found."
            ),
        )

    return explorer


# ============================================================
# PDF EXPORT
# ============================================================

@router.post(
    "/analysis/export-pdf",
)
def export_analysis_pdf(
    payload: PdfExportRequest,
) -> Response:
    try:
        pdf_bytes = (
            build_analysis_pdf(
                report=
                    payload.analysis,

                objective=
                    normalize_objective(
                        payload.objective
                    ),

                document_summary=
                    payload.document_summary,

                requested_analysis_plan=
                    payload.requested_analysis_plan,

                quality_report=
                    payload.quality_report,
            )
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=(
                "La génération locale du PDF a échoué."
            ),
        ) from error


    filename = (
        build_export_filename(
            payload.analysis.title
        )
    )


    return Response(
        content=
            pdf_bytes,

        media_type=
            "application/pdf",

        headers={
            "Content-Disposition":
                f'attachment; filename="{filename}"',

            "Cache-Control":
                "no-store",
        },
    )


# ============================================================
# STANDARD ANALYSIS
# ============================================================

@router.post(
    "/analysis/run",
    response_model=
        RoutedUnifiedAnalysisReport,
)
def run_dataset_analysis(
    dataset_files: (
        list[
            UploadFile
        ]
        | None
    ) = File(
        default=None,
    ),

    workflow_id: str = Form(
        ...,
        min_length=1,
    ),

    objective: str | None = Form(
        default=None,
    ),

    approved_action_ids_json: (
        str
        | None
    ) = Form(
        default=None,
    ),

    semantic_decisions_json: (
        str
        | None
    ) = Form(
        default=None,
    ),

    approved_semantic_choices_json: (
        str
        | None
    ) = Form(
        default=None,
    ),
) -> RoutedUnifiedAnalysisReport:
    try:
        report = (
            run_unified_analysis_from_uploads(
                dataset_files=
                    dataset_files or [],

                objective=
                    objective,

                workflow_id=
                    workflow_id,

                approved_action_ids_json=
                    approved_action_ids_json,

                semantic_decisions_json=
                    semantic_decisions_json,

                approved_semantic_choices_json=
                    approved_semantic_choices_json,
            )
        )


        register_unified_report_artifacts(
            workflow_id=
                workflow_id,

            report=
                report,
        )


        return report


    except ValueError as error:
        raise HTTPException(
            status_code=422,
            detail=str(
                error
            ),
        ) from error


# ============================================================
# CONTEXTUALIZED ANALYSIS
# ============================================================

@router.post(
    "/analysis/run-contextualized",
    response_model=
        ContextualizedAnalysisResponse,
)
def run_contextualized_dataset_analysis(
    dataset_files: (
        list[
            UploadFile
        ]
        | None
    ) = File(
        default=None,
    ),

    workflow_id: str = Form(
        ...,
        min_length=1,
    ),

    document_files: list[
        UploadFile
    ] = File(
        ...,
    ),

    objective: str | None = Form(
        default=None,
    ),

    rag_top_k: int = Form(
        default=3,
        ge=1,
        le=20,
    ),

    embedding_model: str = Form(
        default=
            DEFAULT_EMBEDDING_MODEL,

        min_length=1,
    ),

    approved_action_ids_json: (
        str
        | None
    ) = Form(
        default=None,
    ),

    semantic_decisions_json: (
        str
        | None
    ) = Form(
        default=None,
    ),

    approved_semantic_choices_json: (
        str
        | None
    ) = Form(
        default=None,
    ),
) -> ContextualizedAnalysisResponse:
    normalized_objective = (
        normalize_objective(
            objective
        )
    )


    # ========================================================
    # 1. SERVER-OWNED PREPARATION HANDOFF — ONCE
    # ========================================================

    handoff = (
        load_validated_analysis_input_for_http(
            workflow_id=
                workflow_id
        )
    )


    try:
        reject_post_validation_preparation_overrides(
            approved_action_ids_json=
                approved_action_ids_json,

            semantic_decisions_json=
                semantic_decisions_json,

            approved_semantic_choices_json=
                approved_semantic_choices_json,
        )


    except ValueError as error:
        raise HTTPException(
            status_code=422,

            detail=str(
                error
            ),
        ) from error


    dataset_ingestion = (
        handoff.ingestion
    )


    source_dataset_records = [
        dict(
            record
        )

        for record
        in handoff.dataset_records
    ]


    # dataset_files remains an optional multipart compatibility
    # field. The uploaded bytes are intentionally ignored after
    # VALIDATE; server-owned Preparation artifacts are the only
    # analytical source of truth.
    _ = (
        dataset_files
    )


    cleaning_execution = (
        None
    )


    semantic_execution = (
        None
    )


    entity_outlier_finding = (
        build_entity_outlier_finding_if_requested(
            objective=
                normalized_objective,

            source_dataset_records=
                source_dataset_records,
        )
    )


    # ========================================================
    # 2. DOCUMENT INGESTION
    # ========================================================

    document_ingestion = (
        ingest_document_uploads(
            document_files
        )
    )


    # ========================================================
    # 3. VERIFIED DOCUMENT SUMMARY
    # ========================================================

    try:
        document_summary = (
            summarize_document_ingestion(
                ingestion=
                    document_ingestion,

                objective=
                    normalized_objective,
            )
        )


    except ValueError as error:
        raise HTTPException(
            status_code=422,
            detail=str(
                error
            ),
        ) from error


    except RuntimeError as error:
        raise HTTPException(
            status_code=503,
            detail=(
                "Local AI processing is unavailable or returned an invalid response."
            ),
        ) from error


    # ========================================================
    # 4. REQUEST PLANNING
    # ========================================================

    requested_analysis_plan = (
        build_requested_analysis_plan(
            ingestion=
                dataset_ingestion,

            analytical_requests=
                document_summary
                .analytical_requests,
        )
    )


    # ========================================================
    # 5. ANALYTICAL REQUEST COVERAGE
    # ========================================================
    #
    # Every verified documentary analytical request must
    # remain accounted for before Requested Analysis execution.
    #
    # ready / blocked / ambiguous are all preserved states.
    #
    # Missing, duplicated, orphaned or provenance-inconsistent
    # requests fail closed.
    # ========================================================

    request_coverage = (
        require_analysis_request_coverage_for_http(
            analytical_requests=
                document_summary
                .analytical_requests,

            plan=
                requested_analysis_plan,
        )
    )


    # ========================================================
    # 5. ANALYTICAL DATASET PREPARATION — ONCE
    # ========================================================

    (
        discovery,
        analysis_datasets,
    ) = prepare_analysis_datasets(
        source_datasets=
            source_dataset_records,

        objective=
            normalized_objective,

        include_requested_context=
            True,
    )


    # ========================================================
    # 6. REQUESTED ANALYSIS EXECUTION
    # ========================================================

    requested_analysis_execution = (
        execute_requested_analysis_plan(
            plan=
                requested_analysis_plan,

            datasets=
                analysis_datasets,
        )
    )


    # ========================================================
    # 7. EXPLORATORY / UNIFIED ANALYSIS
    # ========================================================

    analysis = (
        run_unified_analysis_from_prepared_records(
            source_dataset_records=
                source_dataset_records,

            discovery=
                discovery,

            analysis_datasets=
                analysis_datasets,
        )
    )


    prioritization_audit = (
        build_analysis_prioritization_audit(
            prioritize_analysis_discovery(
                discovery,

                datasets=
                    analysis_datasets,
            )
        )
    )


    # ========================================================
    # 8. ATTACH REQUESTED FINDINGS
    # ========================================================

    analysis = (
        attach_requested_findings(
            report=
                analysis,

            execution_report=
                requested_analysis_execution,

            plan_report=
                requested_analysis_plan,
        )
    )


    analysis = (
        append_controlled_cleaning_note(
            report=
                analysis,

            cleaning_execution=
                cleaning_execution,
        )
    )


    analysis = (
        append_semantic_cleaning_note(
            report=
                analysis,

            semantic_execution=
                semantic_execution,
        )
    )


    analysis = (
        build_routed_unified_analysis_report(
            report=
                analysis,

            entity_outlier_finding=
                entity_outlier_finding,

            prioritization_audit=
                prioritization_audit,
        )
    )


    # ========================================================
    # 9. REQUESTED-FIRST RAG INPUT
    # ========================================================

    rag_report_view = (
        build_requested_first_rag_report_view(
            report=
                analysis,
        )
    )


    # ========================================================
    # 10. RAG CONTEXT
    # ========================================================

    try:
        rag = (
            retrieve_context_for_report(
                report=
                    rag_report_view,

                ingestion=
                    document_ingestion,

                objective=
                    normalized_objective,

                top_k=
                    rag_top_k,

                model=
                    embedding_model,
            )
        )


    except ValueError as error:
        raise HTTPException(
            status_code=422,
            detail=str(
                error
            ),
        ) from error


    except RuntimeError as error:
        raise HTTPException(
            status_code=503,
            detail=(
                "Local AI processing is unavailable or returned an invalid response."
            ),
        ) from error


    # ========================================================
    # 11. CONTEXTUALIZED RESPONSE ? BUILD BEFORE PERSISTENCE
    # ========================================================
    #
    # The complete response contract is validated only after
    # Requested Analysis and grounded RAG both succeeded.
    #
    # No report artifact is persisted before this point.
    # ========================================================

    contextualized_response = (
        ContextualizedAnalysisResponse(
            analysis=
                analysis,

            document_summary=
                document_summary,

            requested_analysis_plan=
                requested_analysis_plan,

            request_coverage=
                request_coverage,

            requested_analysis_execution=
                requested_analysis_execution,

            rag=
                rag,
        )
    )


    # ========================================================
    # 12. SERVER-OWNED REPORT ARTIFACTS ? SUCCESS ONLY
    # ========================================================
    #
    # Persistence deliberately occurs after successful RAG and
    # successful contextualized-response construction.
    #
    # A RAG / response-contract failure therefore cannot leave
    # a report artifact for an HTTP response that never existed.
    # ========================================================

    try:
        register_contextualized_report_artifacts_atomic(
            workflow_id=
                workflow_id,

            execution_report=
                requested_analysis_execution,

            plan_report=
                requested_analysis_plan,

            report=
                analysis,
        )


    except ValueError as error:
        raise HTTPException(
            status_code=
                422,

            detail={
                "error":
                    "invalid_contextualized_artifact_persistence",

                "message":
                    (
                        "Contextualized analysis artifact "
                        "persistence request is invalid."
                    ),

                "workflow_id":
                    workflow_id,
            },
        ) from error


    except RuntimeError as error:
        raise HTTPException(
            status_code=
                503,

            detail={
                "error":
                    "contextualized_artifact_persistence_failed",

                "message":
                    (
                        "Contextualized analysis artifacts "
                        "could not be persisted."
                    ),

                "workflow_id":
                    workflow_id,
            },
        ) from error


    # ========================================================
    # 13. RETURN VALIDATED CONTEXTUALIZED RESPONSE
    # ========================================================

    return (
        contextualized_response
    )
