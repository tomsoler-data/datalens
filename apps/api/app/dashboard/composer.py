from typing import Any

from app.statistics.schemas import (
    CorrelationExecution,
    CorrelationTestDecision,
)

from app.visualization.schemas import (
    VisualizationDecision,
)

from app.dashboard.schemas import (
    DashboardChart,
    DashboardDecisionExplanation,
    DashboardEvidenceReferences,
    DashboardKPI,
    DashboardSpec,
    DashboardStatisticalResult,
)


# ============================================================
# DASHBOARD ERROR
# ============================================================

class DashboardCompositionError(
    ValueError
):
    """
    Raised when incompatible statistical and
    visualization objects are passed to the
    dashboard composer.
    """

    pass


# ============================================================
# CANONICAL VALUE
# ============================================================

def canonicalize_dashboard_value(
    value: Any,
) -> str:
    """
    Preserve deterministic values exactly.

    This follows the same philosophy as the
    DataLens evidence layer.
    """

    if value is None:
        return "null"

    if isinstance(
        value,
        bool,
    ):
        return (
            "true"
            if value
            else "false"
        )

    if isinstance(
        value,
        float,
    ):
        return repr(
            value
        )

    return str(
        value
    )


# ============================================================
# UNIQUE WARNINGS
# ============================================================

def deduplicate_messages(
    messages: list[str],
) -> list[str]:
    """
    Preserve message order while removing
    duplicates.
    """

    result = []

    seen = set()

    for message in messages:
        if message in seen:
            continue

        seen.add(
            message
        )

        result.append(
            message
        )

    return result


# ============================================================
# CONSISTENCY VALIDATION
# ============================================================

def validate_dashboard_inputs(
    decision: CorrelationTestDecision,
    visualization: VisualizationDecision,
    execution: CorrelationExecution | None,
) -> None:
    """
    Prevent the dashboard from combining
    decisions and results produced for
    different variables or tests.
    """

    if (
        visualization.x_column
        != decision.x_column
    ):
        raise DashboardCompositionError(
            (
                "Visualization x_column does not "
                "match the statistical decision."
            )
        )

    if (
        visualization.y_column
        != decision.y_column
    ):
        raise DashboardCompositionError(
            (
                "Visualization y_column does not "
                "match the statistical decision."
            )
        )

    if execution is None:
        return

    if (
        execution.x_column
        != decision.x_column
    ):
        raise DashboardCompositionError(
            (
                "Execution x_column does not "
                "match the statistical decision."
            )
        )

    if (
        execution.y_column
        != decision.y_column
    ):
        raise DashboardCompositionError(
            (
                "Execution y_column does not "
                "match the statistical decision."
            )
        )

    if (
        decision.status
        != "selected"
    ):
        raise DashboardCompositionError(
            (
                "A statistical execution cannot "
                "be attached to a decision that "
                "did not select a test."
            )
        )

    if (
        execution.selected_test
        != decision.selected_test
    ):
        raise DashboardCompositionError(
            (
                "Executed test does not match "
                "the selected statistical test."
            )
        )

    if (
        execution.result.test
        != execution.selected_test
    ):
        raise DashboardCompositionError(
            (
                "Statistical result does not "
                "match the executed test."
            )
        )


# ============================================================
# DASHBOARD STATUS
# ============================================================

def determine_dashboard_status(
    decision: CorrelationTestDecision,
    execution: CorrelationExecution | None,
) -> str:
    """
    Determine the dashboard lifecycle state.
    """

    if (
        decision.status
        == "needs_information"
    ):
        return (
            "needs_information"
        )

    if (
        decision.status
        == "not_applicable"
    ):
        return (
            "not_applicable"
        )

    if (
        decision.status
        == "selected"
        and execution is None
    ):
        return (
            "ready_for_execution"
        )

    return "complete"


# ============================================================
# TITLE
# ============================================================

def build_dashboard_title(
    decision: CorrelationTestDecision,
) -> str:
    """
    Build a deterministic dashboard title.
    """

    return (
        f"{decision.x_column} × "
        f"{decision.y_column}"
    )


# ============================================================
# SUBTITLE
# ============================================================

def build_dashboard_subtitle(
    decision: CorrelationTestDecision,
) -> str:
    """
    Human-readable analytical context.
    """

    goal_labels = {
        "linear_association":
            "Linear association",

        "monotonic_association":
            "Monotonic association",

        "general_association":
            "Association analysis",
    }

    mode_labels = {
        "confirmatory":
            "Confirmatory",

        "exploratory":
            "Exploratory",
    }

    goal = goal_labels.get(
        decision.analysis_goal,
        decision.analysis_goal,
    )

    mode = mode_labels.get(
        decision.analysis_mode,
        decision.analysis_mode,
    )

    return (
        f"{goal} · {mode}"
    )


# ============================================================
# SUMMARY
# ============================================================

def build_dashboard_summary(
    decision: CorrelationTestDecision,
    execution: CorrelationExecution | None,
) -> str:
    """
    Build a deterministic summary.

    The LLM will later be able to provide a
    richer explanation, but the factual base
    remains Python-generated.
    """

    if (
        decision.status
        == "needs_information"
    ):
        return (
            "Additional information is required "
            "before DataLens can safely execute "
            "a statistical test."
        )

    if (
        decision.status
        == "not_applicable"
    ):
        return (
            "The current correlation analysis "
            "branch is not applicable to this "
            "request."
        )

    if execution is None:
        return (
            f"{decision.selected_test.capitalize()} "
            "has been selected, but the "
            "statistical test has not yet been "
            "executed."
        )

    result = execution.result

    significance = (
        "statistically significant"
        if result.statistically_significant
        else "not statistically significant"
    )

    return (
        f"{result.test.capitalize()} was executed "
        f"for {decision.x_column} and "
        f"{decision.y_column}. "
        f"The result is {significance}."
    )


# ============================================================
# KPI BUILDER
# ============================================================

def build_dashboard_kpis(
    decision: CorrelationTestDecision,
    execution: CorrelationExecution | None,
) -> list[
    DashboardKPI
]:
    """
    Build deterministic KPI cards.
    """

    kpis = [
        DashboardKPI(
            key=
                "n_valid",

            label=
                "Valid observations",

            kind=
                "sample_size",

            value=
                canonicalize_dashboard_value(
                    decision
                    .diagnostics
                    .n_valid
                ),

            source_reference=
                "decision:0001",

            source_field=
                "diagnostics.n_valid",
        ),

        DashboardKPI(
            key=
                "n_excluded",

            label=
                "Excluded rows",

            kind=
                "excluded_rows",

            value=
                canonicalize_dashboard_value(
                    decision
                    .diagnostics
                    .n_excluded
                ),

            source_reference=
                "decision:0001",

            source_field=
                "diagnostics.n_excluded",
        ),
    ]

    if execution is None:
        if (
            decision.selected_test
            is not None
        ):
            kpis.append(
                DashboardKPI(
                    key=
                        "selected_test",

                    label=
                        "Selected test",

                    kind=
                        "test",

                    value=
                        decision.selected_test,

                    source_reference=
                        "decision:0001",

                    source_field=
                        "selected_test",
                )
            )

        return kpis

    result = (
        execution.result
    )

    kpis.extend(
        [
            DashboardKPI(
                key=
                    "test",

                label=
                    "Statistical test",

                kind=
                    "test",

                value=
                    result.test,

                source_reference=
                    "statistic:0001",

                source_field=
                    "test",
            ),

            DashboardKPI(
                key=
                    "coefficient",

                label=
                    result.coefficient_name,

                kind=
                    "coefficient",

                value=
                    canonicalize_dashboard_value(
                        result.coefficient
                    ),

                source_reference=
                    "statistic:0001",

                source_field=
                    "coefficient",
            ),

            DashboardKPI(
                key=
                    "p_value",

                label=
                    "p-value",

                kind=
                    "p_value",

                value=
                    canonicalize_dashboard_value(
                        result.p_value
                    ),

                source_reference=
                    "statistic:0001",

                source_field=
                    "p_value",
            ),

            DashboardKPI(
                key=
                    "significance",

                label=
                    "Statistically significant",

                kind=
                    "significance",

                value=
                    canonicalize_dashboard_value(
                        result.statistically_significant
                    ),

                source_reference=
                    "statistic:0001",

                source_field=
                    "statistically_significant",
            ),
        ]
    )

    return kpis


# ============================================================
# CHART BUILDER
# ============================================================

def build_dashboard_chart(
    visualization: VisualizationDecision,
) -> DashboardChart | None:
    """
    Convert a visualization decision into
    frontend-ready chart configuration.
    """

    if (
        visualization.status
        != "selected"
        or visualization.chart_type
        is None
    ):
        return None

    return DashboardChart(
        visualization_reference=
            visualization.visualization_id,

        chart_type=
            visualization.chart_type,

        purpose=
            visualization.purpose,

        x_column=
            visualization.x_column,

        y_column=
            visualization.y_column,

        aggregation=
            visualization.aggregation,

        trend=
            visualization.trend,

        show_raw_points=
            visualization.show_raw_points,

        show_missing_summary=
            visualization.show_missing_summary,

        reasons=
            list(
                visualization.reasons
            ),
    )


# ============================================================
# STATISTICAL RESULT BUILDER
# ============================================================

def build_statistical_result(
    execution: CorrelationExecution | None,
) -> DashboardStatisticalResult | None:
    """
    Convert the executed result into a compact
    dashboard result block.
    """

    if execution is None:
        return None

    result = (
        execution.result
    )

    return DashboardStatisticalResult(
        statistic_reference=
            "statistic:0001",

        test=
            result.test,

        relationship_type=
            result.relationship_type,

        coefficient_name=
            result.coefficient_name,

        coefficient=
            canonicalize_dashboard_value(
                result.coefficient
            ),

        p_value=
            canonicalize_dashboard_value(
                result.p_value
            ),

        alpha=
            canonicalize_dashboard_value(
                result.alpha
            ),

        statistically_significant=
            result.statistically_significant,

        n=
            result.n,

        inference_method=
            execution.inference_method_used,

        permutation_mode=
            execution.permutation_mode,
    )


# ============================================================
# DECISION BLOCK
# ============================================================

def build_decision_explanation(
    decision: CorrelationTestDecision,
) -> DashboardDecisionExplanation:
    """
    Build the 'Why this test?' dashboard block.
    """

    return DashboardDecisionExplanation(
        decision_reference=
            "decision:0001",

        status=
            decision.status,

        analysis_goal=
            decision.analysis_goal,

        analysis_mode=
            decision.analysis_mode,

        selected_test=
            decision.selected_test,

        selection_is_data_driven=
            decision.selection_is_data_driven,

        reasons=
            list(
                decision.reasons
            ),

        missing_information=
            list(
                decision.missing_information
            ),
    )


# ============================================================
# WARNINGS
# ============================================================

def build_dashboard_warnings(
    decision: CorrelationTestDecision,
    visualization: VisualizationDecision,
    execution: CorrelationExecution | None,
) -> list[str]:
    """
    Combine warnings from every deterministic
    layer without repeating them.
    """

    warnings = []

    warnings.extend(
        decision.warnings
    )

    warnings.extend(
        decision
        .diagnostics
        .warnings
    )

    warnings.extend(
        visualization.warnings
    )

    if execution is not None:
        warnings.extend(
            execution.warnings
        )

    return (
        deduplicate_messages(
            warnings
        )
    )


# ============================================================
# MAIN DASHBOARD COMPOSER
# ============================================================

def compose_correlation_dashboard(
    decision: CorrelationTestDecision,
    visualization: VisualizationDecision,
    execution: CorrelationExecution | None = None,
) -> DashboardSpec:
    """
    Compose a deterministic dashboard
    specification.

    No statistical calculation occurs here.

    No visualization choice occurs here.

    No LLM is involved.
    """

    validate_dashboard_inputs(
        decision=
            decision,

        visualization=
            visualization,

        execution=
            execution,
    )

    status = (
        determine_dashboard_status(
            decision=
                decision,

            execution=
                execution,
        )
    )

    statistic_reference = (
        "statistic:0001"
        if execution is not None
        else None
    )

    return DashboardSpec(
        status=
            status,

        title=
            build_dashboard_title(
                decision
            ),

        subtitle=
            build_dashboard_subtitle(
                decision
            ),

        summary=
            build_dashboard_summary(
                decision=
                    decision,

                execution=
                    execution,
            ),

        x_column=
            decision.x_column,

        y_column=
            decision.y_column,

        kpis=
            build_dashboard_kpis(
                decision=
                    decision,

                execution=
                    execution,
            ),

        chart=
            build_dashboard_chart(
                visualization
            ),

        statistical_result=
            build_statistical_result(
                execution
            ),

        decision=
            build_decision_explanation(
                decision
            ),

        action_required=
            list(
                decision.missing_information
            ),

        warnings=
            build_dashboard_warnings(
                decision=
                    decision,

                visualization=
                    visualization,

                execution=
                    execution,
            ),

        evidence=
            DashboardEvidenceReferences(
                decision=
                    "decision:0001",

                statistic=
                    statistic_reference,

                visualization=
                    visualization
                    .visualization_id,
            ),
    )