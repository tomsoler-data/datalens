from __future__ import annotations

from collections import (
    defaultdict,
)

from typing import (
    TYPE_CHECKING,
    Any,
)

import pandas as pd

from app.discovery.schemas import (
    AnalysisDiscoveryReport,
)

from app.execution.cross_schemas import (
    CrossDatasetExecutedAnalysis,
    CrossDatasetExecutionReport,
)

from app.execution.single_schemas import (
    SingleDatasetExecutedAnalysis,
    SingleDatasetExecutionReport,
)

from app.reporting.unified_schemas import (
    ReportBlockedAnalysis,
    ReportDatasetSummary,
    ReportFinding,
    ReportInventory,
    ReportQualityItem,
    UnifiedAnalysisReport,
)


# ============================================================
# TYPE-ONLY IMPORTS
# ============================================================

if TYPE_CHECKING:
    from app.ranking.unified_schemas import (
        UnifiedRankedAnalysis,
        UnifiedRankingReport,
    )


# ============================================================
# REPORT POLICY
# ============================================================

MAIN_FINDING_LIMIT = 6

MAX_MAIN_FINDINGS_PER_FAMILY = 2


PRIMARY_FINDING_FAMILIES = {
    "quantitative_association",
    "categorical_association",
    "time_series",
    "derived_gap",
    "group_comparison",
}


DIAGNOSTIC_FAMILIES = {
    "distribution",
}


QUALITY_FAMILIES = {
    "data_quality",
}


CONTEXT_FAMILIES = {
    "geographic_comparison",
}


BLOCKED_TIERS = {
    "blocked",
}


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
    return list(
        dict.fromkeys(
            value
            for value
            in values
            if value
        )
    )


# ============================================================
# EXECUTION LOOKUP
# ============================================================

def build_execution_map(
    *,
    single_execution: SingleDatasetExecutionReport,
    cross_execution: CrossDatasetExecutionReport,
) -> dict[
    str,
    SingleDatasetExecutedAnalysis
    |
    CrossDatasetExecutedAnalysis,
]:
    result: dict[
        str,
        SingleDatasetExecutedAnalysis
        |
        CrossDatasetExecutedAnalysis,
    ] = {}


    for execution_result in (
        single_execution.results
    ):
        result[
            execution_result.analysis_id
        ] = execution_result


    for execution_result in (
        cross_execution.results
    ):
        result[
            execution_result.analysis_id
        ] = execution_result


    return result


# ============================================================
# DATASETS
# ============================================================

def build_dataset_summaries(
    datasets: list[
        dict[
            str,
            Any,
        ]
    ],
) -> list[
    ReportDatasetSummary
]:
    summaries: list[
        ReportDatasetSummary
    ] = []


    for dataset in datasets:
        dataframe = dataset.get(
            "dataframe"
        )


        if not isinstance(
            dataframe,
            pd.DataFrame,
        ):
            continue


        summaries.append(
            ReportDatasetSummary(
                dataset_id=
                    str(
                        dataset[
                            "dataset_id"
                        ]
                    ),

                filename=
                    str(
                        dataset[
                            "filename"
                        ]
                    ),

                row_count=
                    int(
                        len(
                            dataframe
                        )
                    ),

                column_count=
                    int(
                        len(
                            dataframe.columns
                        )
                    ),

                columns=[
                    str(
                        column
                    )
                    for column
                    in dataframe.columns
                ],
            )
        )


    return summaries


# ============================================================
# EXECUTION CONTENT
# ============================================================

def execution_summary(
    execution_result: (
        SingleDatasetExecutedAnalysis
        |
        CrossDatasetExecutedAnalysis
        |
        None
    ),
) -> list[
    str
]:
    if execution_result is None:
        return []


    summary = getattr(
        execution_result,
        "summary",
        [],
    )


    return list(
        summary
        or
        []
    )


def execution_chart_type(
    execution_result: (
        SingleDatasetExecutedAnalysis
        |
        CrossDatasetExecutedAnalysis
        |
        None
    ),
) -> str | None:
    if execution_result is None:
        return None


    return getattr(
        execution_result,
        "chart_type",
        None,
    )


def execution_chart_data(
    execution_result: (
        SingleDatasetExecutedAnalysis
        |
        CrossDatasetExecutedAnalysis
        |
        None
    ),
) -> list[
    dict[
        str,
        Any,
    ]
]:
    if execution_result is None:
        return []


    chart_data = getattr(
        execution_result,
        "chart_data",
        [],
    )


    return list(
        chart_data
        or
        []
    )


def execution_metrics(
    execution_result: (
        SingleDatasetExecutedAnalysis
        |
        CrossDatasetExecutedAnalysis
        |
        None
    ),
) -> dict[
    str,
    Any,
]:
    if execution_result is None:
        return {}


    metrics = getattr(
        execution_result,
        "metrics",
        {},
    )


    return dict(
        metrics
        or
        {}
    )


# ============================================================
# FINDING BUILDER
# ============================================================

def build_report_finding(
    finding: UnifiedRankedAnalysis,
    *,
    role: str,
    execution_result: (
        SingleDatasetExecutedAnalysis
        |
        CrossDatasetExecutedAnalysis
        |
        None
    ),
) -> ReportFinding:
    return ReportFinding(
        analysis_id=
            finding.analysis_id,

        title=
            finding.title,

        role=
            role,

        scope=
            finding.scope,

        family=
            finding.family,

        tier=
            finding.tier,

        execution_status=
            finding.execution_status,

        interestingness_score=
            finding.interestingness_score,

        signal_score=
            finding.signal_score,

        coverage_score=
            finding.coverage_score,

        consistency_score=
            finding.consistency_score,

        direction=
            finding.direction,

        strength=
            finding.strength,

        sample_size=
            finding.sample_size,

        period_count=
            finding.period_count,

        datasets=list(
            finding.datasets
        ),

        summary=(
            execution_summary(
                execution_result
            )
        ),

        reasons=list(
            finding.reasons
        ),

        caveats=list(
            finding.caveats
        ),

        chart_type=(
            execution_chart_type(
                execution_result
            )
        ),

        chart_data=(
            execution_chart_data(
                execution_result
            )
        ),

        metrics=(
            execution_metrics(
                execution_result
            )
        ),
    )


# ============================================================
# MAIN FINDING SELECTION
# ============================================================

def select_main_findings(
    ranking: UnifiedRankingReport,
) -> list[
    UnifiedRankedAnalysis
]:
    """
    Select a small and diverse set of analyses
    for the executive part of the report.

    This deliberately avoids simply copying the
    first N global scores.

    Distributions, quality audits and geographic
    rankings are presented in dedicated sections.
    """

    candidates = [
        finding
        for finding
        in ranking.findings
        if (
            finding.tier
            not in BLOCKED_TIERS
            and
            finding.family
            in PRIMARY_FINDING_FAMILIES
        )
    ]


    selected: list[
        UnifiedRankedAnalysis
    ] = []


    family_counts: dict[
        str,
        int,
    ] = defaultdict(
        int
    )


    # ========================================================
    # FIRST PASS
    #
    # One finding per analytical family to maximize
    # diversity in the executive summary.
    # ========================================================

    used_families: set[
        str
    ] = set()


    for finding in candidates:
        if (
            finding.family
            in used_families
        ):
            continue


        selected.append(
            finding
        )


        used_families.add(
            finding.family
        )


        family_counts[
            finding.family
        ] += 1


        if (
            len(
                selected
            )
            >=
            MAIN_FINDING_LIMIT
        ):
            return selected


    # ========================================================
    # SECOND PASS
    #
    # Fill remaining slots while limiting a family
    # to two findings maximum.
    # ========================================================

    selected_ids = {
        finding.analysis_id
        for finding
        in selected
    }


    for finding in candidates:
        if (
            finding.analysis_id
            in selected_ids
        ):
            continue


        if (
            family_counts[
                finding.family
            ]
            >=
            MAX_MAIN_FINDINGS_PER_FAMILY
        ):
            continue


        selected.append(
            finding
        )


        selected_ids.add(
            finding.analysis_id
        )


        family_counts[
            finding.family
        ] += 1


        if (
            len(
                selected
            )
            >=
            MAIN_FINDING_LIMIT
        ):
            break


    return selected


# ============================================================
# QUALITY SECTION
# ============================================================

def build_quality_items(
    single_execution: SingleDatasetExecutionReport,
) -> list[
    ReportQualityItem
]:
    items: list[
        ReportQualityItem
    ] = []


    for result in (
        single_execution.results
    ):
        if (
            result.family
            !=
            "data_quality"
        ):
            continue


        metrics = (
            result.metrics
        )


        items.append(
            ReportQualityItem(
                analysis_id=
                    result.analysis_id,

                dataset=
                    result.dataset,

                row_count=
                    int(
                        metrics.get(
                            "row_count",
                            result.valid_observations,
                        )
                        or
                        0
                    ),

                column_count=
                    int(
                        metrics.get(
                            "column_count",
                            0,
                        )
                        or
                        0
                    ),

                missing_cells=
                    int(
                        metrics.get(
                            "missing_cells",
                            0,
                        )
                        or
                        0
                    ),

                missing_ratio=
                    float(
                        metrics.get(
                            "missing_ratio",
                            0.0,
                        )
                        or
                        0.0
                    ),

                duplicate_rows=
                    int(
                        metrics.get(
                            "duplicate_rows",
                            0,
                        )
                        or
                        0
                    ),

                duplicate_ratio=
                    float(
                        metrics.get(
                            "duplicate_ratio",
                            0.0,
                        )
                        or
                        0.0
                    ),

                completely_missing_columns=list(
                    metrics.get(
                        "completely_missing_columns",
                        [],
                    )
                    or
                    []
                ),

                constant_columns=list(
                    metrics.get(
                        "constant_columns",
                        [],
                    )
                    or
                    []
                ),

                summary=list(
                    result.summary
                ),
            )
        )


    return items


# ============================================================
# BLOCKED SECTION
# ============================================================

def build_blocked_analyses(
    ranking: UnifiedRankingReport,
) -> list[
    ReportBlockedAnalysis
]:
    blocked: list[
        ReportBlockedAnalysis
    ] = []


    for finding in (
        ranking.findings
    ):
        if (
            finding.tier
            !=
            "blocked"
        ):
            continue


        reason = (
            finding.reasons[
                0
            ]
            if finding.reasons
            else
            (
                "L'analyse n'a pas pu être "
                "exécutée de manière suffisamment "
                "sûre."
            )
        )


        blocked.append(
            ReportBlockedAnalysis(
                analysis_id=
                    finding.analysis_id,

                title=
                    finding.title,

                family=
                    finding.family,

                datasets=list(
                    finding.datasets
                ),

                reason=
                    reason,

                caveats=list(
                    finding.caveats
                ),

                discovery_priority_score=
                    finding.discovery_priority_score,
            )
        )


    return blocked


# ============================================================
# EXECUTED COUNT
# ============================================================

def count_executed_results(
    *,
    single_execution: SingleDatasetExecutionReport,
    cross_execution: CrossDatasetExecutionReport,
) -> int:
    blocked_statuses = {
        "requires_alignment",
        "needs_specialized_method",
        "skipped",
        "failed",
    }


    count = 0


    for result in (
        single_execution.results
    ):
        if (
            result.execution_status
            not in blocked_statuses
        ):
            count += 1


    for result in (
        cross_execution.results
    ):
        if (
            result.execution_status
            not in blocked_statuses
        ):
            count += 1


    return count


# ============================================================
# MAIN COMPOSER
# ============================================================

def compose_unified_report(
    *,
    discovery: AnalysisDiscoveryReport,
    single_execution: SingleDatasetExecutionReport,
    cross_execution: CrossDatasetExecutionReport,
    ranking: UnifiedRankingReport,
    datasets: list[
        dict[
            str,
            Any,
        ]
    ],
    title: str = "Analyse des données",
) -> UnifiedAnalysisReport:
    execution_map = (
        build_execution_map(
            single_execution=
                single_execution,

            cross_execution=
                cross_execution,
        )
    )


    dataset_summaries = (
        build_dataset_summaries(
            datasets
        )
    )


    selected_main = (
        select_main_findings(
            ranking
        )
    )


    selected_main_ids = {
        finding.analysis_id
        for finding
        in selected_main
    }


    main_findings = [
        build_report_finding(
            finding,

            role=
                "main_finding",

            execution_result=
                execution_map.get(
                    finding.analysis_id
                ),
        )
        for finding
        in selected_main
    ]


    additional_findings = [
        build_report_finding(
            finding,

            role=
                "additional_finding",

            execution_result=
                execution_map.get(
                    finding.analysis_id
                ),
        )
        for finding
        in ranking.findings
        if (
            finding.tier
            !=
            "blocked"
            and
            finding.family
            in PRIMARY_FINDING_FAMILIES
            and
            finding.analysis_id
            not in selected_main_ids
        )
    ]


    diagnostics = [
        build_report_finding(
            finding,

            role=
                "diagnostic",

            execution_result=
                execution_map.get(
                    finding.analysis_id
                ),
        )
        for finding
        in ranking.findings
        if (
            finding.tier
            !=
            "blocked"
            and
            finding.family
            in DIAGNOSTIC_FAMILIES
        )
    ]


    context_analyses = [
        build_report_finding(
            finding,

            role=
                "context",

            execution_result=
                execution_map.get(
                    finding.analysis_id
                ),
        )
        for finding
        in ranking.findings
        if (
            finding.tier
            !=
            "blocked"
            and
            finding.family
            in CONTEXT_FAMILIES
        )
    ]


    quality = (
        build_quality_items(
            single_execution
        )
    )


    blocked_analyses = (
        build_blocked_analyses(
            ranking
        )
    )


    executed_count = (
        count_executed_results(
            single_execution=
                single_execution,

            cross_execution=
                cross_execution,
        )
    )


    inventory = (
        ReportInventory(
            dataset_count=
                len(
                    dataset_summaries
                ),

            discovered_analysis_count=
                discovery.candidate_count,

            executed_analysis_count=
                executed_count,

            main_finding_count=
                len(
                    main_findings
                ),

            additional_finding_count=
                len(
                    additional_findings
                ),

            diagnostic_count=
                len(
                    diagnostics
                ),

            quality_check_count=
                len(
                    quality
                ),

            context_analysis_count=
                len(
                    context_analyses
                ),

            blocked_analysis_count=
                len(
                    blocked_analyses
                ),
        )
    )


    executive_summary = [
        (
            f"{inventory.dataset_count} dataset(s) "
            "ont été analysés."
        ),

        (
            f"{inventory.discovered_analysis_count} "
            "analyses candidates ont été "
            "identifiées."
        ),

        (
            f"{inventory.executed_analysis_count} "
            "analyses ont pu être exécutées de "
            "manière suffisamment sûre."
        ),

        (
            f"{inventory.main_finding_count} "
            "analyses prioritaires et diversifiées "
            "ont été retenues pour la synthèse."
        ),
    ]


    if (
        inventory.blocked_analysis_count
        >
        0
    ):
        executive_summary.append(
            (
                f"{inventory.blocked_analysis_count} "
                "analyses supplémentaires restent "
                "bloquées car leur grain ou leur "
                "alignement ne permet pas encore "
                "une exécution sûre."
            )
        )


    methodology_notes = (
        deduplicate_strings(
            [
                (
                    "Les calculs numériques et les "
                    "contraintes statistiques sont "
                    "déterminés par le moteur "
                    "Python."
                ),

                (
                    "Le rapport distingue les "
                    "enseignements analytiques, "
                    "les diagnostics de distribution, "
                    "la qualité des données et le "
                    "contexte géographique."
                ),

                (
                    "Les principaux enseignements "
                    "sont sélectionnés avec une "
                    "contrainte de diversité afin "
                    "qu'une seule famille analytique "
                    "ne monopolise pas la synthèse."
                ),

                (
                    "Les analyses bloquées ne sont "
                    "jamais présentées comme des "
                    "résultats observés."
                ),

                *single_execution.executor_notes,

                *cross_execution.executor_notes,

                *ranking.ranking_notes,
            ]
        )
    )


    return UnifiedAnalysisReport(
        title=
            title,

        executive_summary=
            executive_summary,

        inventory=
            inventory,

        datasets=
            dataset_summaries,

        main_findings=
            main_findings,

        additional_findings=
            additional_findings,

        diagnostics=
            diagnostics,

        quality=
            quality,

        context_analyses=
            context_analyses,

        blocked_analyses=
            blocked_analyses,

        methodology_notes=
            methodology_notes,

        report_rule_version=(
            "unified_report_composer_v0.2"
        ),
    )