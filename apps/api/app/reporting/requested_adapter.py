from __future__ import annotations


from typing import (
    Any,
)


from app.execution.requested_schemas import (
    RequestedAnalysisExecution,
    RequestedAnalysisExecutionReport,
)

from app.planning.schemas import (
    RequestedAnalysisPlan,
    RequestedAnalysisPlanReport,
)

from app.reporting.unified_schemas import (
    ReportRequestedFinding,
    UnifiedAnalysisReport,
)


# ============================================================
# VERSION
# ============================================================

REQUESTED_REPORT_ADAPTER_RULE_VERSION = (
    "requested_report_adapter_v0.2"
)


# ============================================================
# ELIGIBLE STATUSES
# ============================================================

REPORTABLE_REQUESTED_STATUSES = {
    "complete",
    "descriptive_only",
}


# ============================================================
# RAG REPORT VIEW
# ============================================================

class RequestedFirstRagReportView:
    """
    Lightweight view consumed only by the existing
    RAG contextualization layer.

    The RAG currently consumes report.main_findings.

    Requested findings are exposed first, followed
    by exploratory main findings.

    The actual UnifiedAnalysisReport is not
    mutated.
    """

    def __init__(
        self,
        *,
        report: UnifiedAnalysisReport,
    ) -> None:
        self.main_findings = [
            *report.requested_findings,
            *report.main_findings,
        ]


# ============================================================
# GENERIC HELPERS
# ============================================================

def deduplicate_strings(
    values: list[
        str
    ],
) -> list[
    str
]:
    result: list[
        str
    ] = []

    seen: set[
        str
    ] = set()


    for value in values:
        normalized = (
            str(
                value
            )
            .strip()
        )


        if not normalized:
            continue


        if normalized in seen:
            continue


        seen.add(
            normalized
        )

        result.append(
            normalized
        )


    return result


def format_coefficient(
    value: Any,
) -> (
    str
    | None
):
    if value is None:
        return None


    try:
        numeric = float(
            value
        )


    except (
        TypeError,
        ValueError,
    ):
        return None


    return f"{numeric:.3f}"


def requested_analysis_id(
    execution: RequestedAnalysisExecution,
) -> str:
    request_id = (
        str(
            execution.request_id
        )
        .strip()
    )


    if request_id.startswith(
        "request:"
    ):
        suffix = request_id[
            len(
                "request:"
            ):
        ]


        if suffix:
            return (
                "requested:"
                +
                suffix
            )


    if request_id:
        return (
            "requested:"
            +
            request_id
        )


    technical = (
        execution.result
    )


    if technical is not None:
        technical_id = (
            str(
                technical.analysis_id
            )
            .strip()
        )


        if technical_id:
            return (
                "requested:"
                +
                technical_id
            )


    return "requested:unknown"


def requested_sample_size(
    execution: RequestedAnalysisExecution,
) -> int:
    descriptive = (
        execution.descriptive_statistics
        or
        {}
    )


    raw_n = (
        descriptive.get(
            "n"
        )
    )


    if raw_n is not None:
        try:
            return max(
                0,
                int(
                    raw_n
                ),
            )

        except (
            TypeError,
            ValueError,
        ):
            pass


    technical = (
        execution.result
    )


    if technical is not None:
        metrics = (
            technical.metrics
            or
            {}
        )


        for key in (
            "valid_pairs",
            "n",
            "sample_size",
            "valid_observations",
        ):
            raw_value = (
                metrics.get(
                    key
                )
            )


            if raw_value is None:
                continue


            try:
                return max(
                    0,
                    int(
                        raw_value
                    ),
                )

            except (
                TypeError,
                ValueError,
            ):
                continue


    return 0


# ============================================================
# PLAN LOOKUP
# ============================================================

def build_request_plan_map(
    plan_report: RequestedAnalysisPlanReport,
) -> dict[
    str,
    RequestedAnalysisPlan,
]:
    """
    Build deterministic request_id -> plan mapping.

    request_id is the stable join key between the
    Request Planner and Requested Analysis Executor.
    """

    result: dict[
        str,
        RequestedAnalysisPlan,
    ] = {}


    for request in (
        plan_report.requests
    ):
        request_id = (
            str(
                request.request_id
            )
            .strip()
        )


        if not request_id:
            continue


        if request_id in result:
            raise ValueError(
                (
                    "Duplicate request_id detected "
                    "while adapting requested "
                    "analysis results: "
                    f"{request_id}"
                )
            )


        result[
            request_id
        ] = request


    return result


# ============================================================
# PRODUCT-FACING METRICS
# ============================================================

def build_requested_metrics(
    execution: RequestedAnalysisExecution,
) -> dict[
    str,
    Any,
]:
    """
    Build structured metrics for the unified
    product report.

    x_column and y_column intentionally use the
    naming convention consumed by the existing
    deterministic RAG analytical-contract builder.
    """

    metrics: dict[
        str,
        Any,
    ] = {}


    technical = (
        execution.result
    )


    if technical is not None:
        technical_metrics = (
            technical.metrics
            or
            {}
        )


        metrics.update(
            dict(
                technical_metrics
            )
        )


    variables = (
        execution.variables
        or
        {}
    )


    x_column = (
        variables.get(
            "x"
        )
    )


    y_column = (
        variables.get(
            "y"
        )
    )


    if x_column:
        metrics[
            "x_column"
        ] = x_column


    if y_column:
        metrics[
            "y_column"
        ] = y_column


    descriptive = (
        execution.descriptive_statistics
        or
        {}
    )


    if descriptive:
        metrics.update(
            dict(
                descriptive
            )
        )


    # Reassert deterministic identifiers after
    # merging descriptive statistics.

    if x_column:
        metrics[
            "x_column"
        ] = x_column


    if y_column:
        metrics[
            "y_column"
        ] = y_column


    metrics[
        "execution_status"
    ] = (
        execution.execution_status
    )


    metrics[
        "inferential_status"
    ] = (
        execution.inferential_status
    )


    metrics[
        "analysis_mode"
    ] = (
        execution.analysis_mode
    )


    return metrics


# ============================================================
# PRODUCT-FACING SUMMARY
# ============================================================

def build_requested_summary(
    execution: RequestedAnalysisExecution,
) -> list[
    str
]:
    summary: list[
        str
    ] = []


    descriptive = (
        execution.descriptive_statistics
        or
        {}
    )


    if (
        execution.execution_status
        ==
        "descriptive_only"
        and
        descriptive
    ):
        n = (
            descriptive.get(
                "n"
            )
        )


        if n is not None:
            summary.append(
                (
                    f"{n} observation(s) complète(s) "
                    "ont été utilisées pour décrire "
                    "l'association."
                )
            )


        pearson = (
            format_coefficient(
                descriptive.get(
                    "pearson_r"
                )
            )
        )


        if pearson is not None:
            summary.append(
                (
                    "Coefficient descriptif de "
                    f"Pearson : r = {pearson}."
                )
            )


        spearman = (
            format_coefficient(
                descriptive.get(
                    "spearman_rho"
                )
            )
        )


        if spearman is not None:
            summary.append(
                (
                    "Coefficient descriptif de "
                    f"Spearman : ρ = {spearman}."
                )
            )


        summary.append(
            (
                "Aucun test inférentiel n'a été "
                "sélectionné automatiquement ; "
                "aucune p-value n'est interprétée "
                "pour ce résultat descriptif."
            )
        )


    technical = (
        execution.result
    )


    if (
        technical is not None
        and
        execution.execution_status
        ==
        "complete"
    ):
        summary.extend(
            list(
                technical.summary
                or
                []
            )
        )


    return (
        deduplicate_strings(
            summary
        )
    )


# ============================================================
# DOCUMENTARY PROVENANCE
# ============================================================

def validate_request_provenance(
    *,
    execution: RequestedAnalysisExecution,

    plan: RequestedAnalysisPlan,
) -> list[
    str
]:
    """
    Check that the documentary fields copied into
    the executor still agree with the authoritative
    Request Planner record.

    The planner remains the source of truth for
    page/chunk/evidence-unit provenance.
    """

    warnings: list[
        str
    ] = []


    if (
        execution.source_filename
        !=
        plan.source_filename
    ):
        warnings.append(
            (
                "La source documentaire conservée "
                "par l'exécution diffère de celle "
                "du plan vérifié."
            )
        )


    if (
        execution.source_locator
        !=
        plan.source_locator
    ):
        warnings.append(
            (
                "Le locator documentaire conservé "
                "par l'exécution diffère de celui "
                "du plan vérifié."
            )
        )


    if (
        execution.evidence_quote
        !=
        plan.evidence_quote
    ):
        warnings.append(
            (
                "La preuve documentaire conservée "
                "par l'exécution diffère de celle "
                "du plan vérifié."
            )
        )


    return warnings


# ============================================================
# SINGLE REQUESTED FINDING
# ============================================================

def build_requested_report_finding(
    execution: RequestedAnalysisExecution,
    *,
    plan: RequestedAnalysisPlan,
) -> (
    ReportRequestedFinding
    | None
):
    """
    Join one deterministic execution result with
    its verified Request Planner provenance.

    Statistical truth comes from execution.

    Documentary provenance comes from the verified
    Request Planner request identified by the same
    request_id.
    """

    if (
        execution.execution_status
        not in
        REPORTABLE_REQUESTED_STATUSES
    ):
        return None


    technical = (
        execution.result
    )


    if technical is None:
        return None


    if (
        execution.request_id
        !=
        plan.request_id
    ):
        raise ValueError(
            (
                "Requested result / plan request_id "
                "mismatch: "
                f"{execution.request_id} != "
                f"{plan.request_id}"
            )
        )


    dataset_names: list[
        str
    ] = []


    if execution.dataset_filename:
        dataset_names.append(
            str(
                execution.dataset_filename
            )
        )


    provenance_warnings = (
        validate_request_provenance(
            execution=
                execution,

            plan=
                plan,
        )
    )


    reasons = [
        (
            "Cette analyse répond à une demande "
            "explicite vérifiée dans la "
            "documentation fournie."
        ),

        (
            "La provenance documentaire directe "
            "est issue du Request Planner vérifié "
            "et n'a pas besoin d'être redécouverte "
            "par le RAG."
        ),
    ]


    warnings = (
        execution.warnings
        or
        []
    )


    limitations = (
        execution.limitations
        or
        []
    )


    caveats = (
        deduplicate_strings(
            [
                *warnings,
                *limitations,
                *provenance_warnings,
            ]
        )
    )


    chart_data = (
        technical.chart_data
        or
        []
    )


    return (
        ReportRequestedFinding(
            request_id=
                execution.request_id,

            analysis_id=
                requested_analysis_id(
                    execution
                ),

            title=
                execution.request_text,

            kind=
                execution.kind,

            scope=
                "single_dataset",

            family=
                technical.family,

            execution_status=
                execution.execution_status,

            inferential_status=
                execution.inferential_status,

            analysis_mode=
                execution.analysis_mode,

            dataset_id=
                execution.dataset_id,

            datasets=
                dataset_names,

            analytical_grain=
                execution.analytical_grain,

            variables=
                dict(
                    execution.variables
                    or
                    {}
                ),

            sample_size=
                requested_sample_size(
                    execution
                ),

            summary=
                build_requested_summary(
                    execution
                ),

            reasons=
                reasons,

            caveats=
                caveats,

            chart_type=
                technical.chart_type,

            chart_data=
                list(
                    chart_data
                ),

            metrics=
                build_requested_metrics(
                    execution
                ),

            # =================================================
            # AUTHORITATIVE VERIFIED DOCUMENTARY PROVENANCE
            # =================================================

            source_filename=
                plan.source_filename,

            source_locator=
                plan.source_locator,

            page_number=
                plan.page_number,

            source_chunk_id=
                plan.source_chunk_id,

            evidence_unit_id=
                plan.evidence_unit_id,

            evidence_quote=
                plan.evidence_quote,

            adapter_rule_version=
                REQUESTED_REPORT_ADAPTER_RULE_VERSION,
        )
    )


# ============================================================
# REQUESTED FINDING COLLECTION
# ============================================================

def build_requested_report_findings(
    *,
    execution_report: (
        RequestedAnalysisExecutionReport
    ),

    plan_report: (
        RequestedAnalysisPlanReport
    ),
) -> list[
    ReportRequestedFinding
]:
    findings: list[
        ReportRequestedFinding
    ] = []


    seen_analysis_ids: set[
        str
    ] = set()


    plan_map = (
        build_request_plan_map(
            plan_report
        )
    )


    for execution in (
        execution_report.results
    ):
        if (
            execution.execution_status
            not in
            REPORTABLE_REQUESTED_STATUSES
        ):
            continue


        plan = (
            plan_map.get(
                execution.request_id
            )
        )


        if plan is None:
            raise ValueError(
                (
                    "No Request Planner record was "
                    "found for reportable requested "
                    "execution "
                    f"{execution.request_id}."
                )
            )


        finding = (
            build_requested_report_finding(
                execution,

                plan=
                    plan,
            )
        )


        if finding is None:
            continue


        if (
            finding.analysis_id
            in
            seen_analysis_ids
        ):
            raise ValueError(
                (
                    "Duplicate requested finding "
                    "analysis_id detected: "
                    f"{finding.analysis_id}"
                )
            )


        seen_analysis_ids.add(
            finding.analysis_id
        )


        findings.append(
            finding
        )


    return findings


# ============================================================
# ATTACH TO UNIFIED PRODUCT REPORT
# ============================================================

def attach_requested_findings(
    *,
    report: UnifiedAnalysisReport,

    execution_report: (
        RequestedAnalysisExecutionReport
    ),

    plan_report: (
        RequestedAnalysisPlanReport
    ),
) -> UnifiedAnalysisReport:
    """
    Attach explicitly requested analytical results
    to a unified product report.

    Statistical outputs come from the Requested
    Analysis Executor.

    Documentary provenance comes directly from the
    verified Request Planner.

    Requested findings remain outside the
    exploratory ranking.
    """

    adapted_report = (
        report.model_copy(
            deep=True
        )
    )


    requested_findings = (
        build_requested_report_findings(
            execution_report=
                execution_report,

            plan_report=
                plan_report,
        )
    )


    adapted_report.requested_findings = (
        requested_findings
    )


    adapted_report.inventory.requested_finding_count = (
        len(
            requested_findings
        )
    )


    if requested_findings:
        count = len(
            requested_findings
        )


        if count == 1:
            requested_summary = (
                "1 analyse explicitement demandée "
                "dans la documentation a produit "
                "un résultat exploitable."
            )

        else:
            requested_summary = (
                f"{count} analyses explicitement "
                "demandées dans la documentation "
                "ont produit un résultat exploitable."
            )


        adapted_report.executive_summary = (
            deduplicate_strings(
                [
                    requested_summary,

                    *adapted_report
                    .executive_summary,
                ]
            )
        )


    adapted_report.methodology_notes = (
        deduplicate_strings(
            [
                *adapted_report
                .methodology_notes,

                (
                    "Les analyses explicitement "
                    "demandées dans la documentation "
                    "sont présentées séparément des "
                    "analyses découvertes "
                    "automatiquement."
                ),

                (
                    "Les requested findings ne "
                    "participent pas au ranking "
                    "exploratoire et ne reçoivent "
                    "aucun score artificiel "
                    "d'intéressance."
                ),

                (
                    "La provenance documentaire "
                    "primaire des requested findings "
                    "est reprise directement du "
                    "Request Planner vérifié."
                ),

                (
                    "Le RAG peut enrichir un "
                    "requested finding avec du "
                    "contexte supplémentaire, mais "
                    "son abstention n'annule pas la "
                    "preuve documentaire directe qui "
                    "a déclenché l'analyse."
                ),

                (
                    "Une demande bloquée, ambiguë, "
                    "non encore prise en charge ou "
                    "en échec n'est jamais "
                    "transformée en finding observé."
                ),

                (
                    "Requested Report Adapter "
                    "version : "
                    f"{REQUESTED_REPORT_ADAPTER_RULE_VERSION}."
                ),
            ]
        )
    )


    return adapted_report


# ============================================================
# RAG INPUT VIEW
# ============================================================

def build_requested_first_rag_report_view(
    *,
    report: UnifiedAnalysisReport,
) -> RequestedFirstRagReportView:
    """
    Build the lightweight input consumed by the
    existing RAG layer.

    Ordering is intentional:

    1. explicitly requested findings;
    2. exploratory main findings.

    The RAG remains free to abstain on supplementary
    documentary context.

    The direct documentary provenance of requested
    findings is already guaranteed independently.
    """

    return (
        RequestedFirstRagReportView(
            report=
                report,
        )
    )