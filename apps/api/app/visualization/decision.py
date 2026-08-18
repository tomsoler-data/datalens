from app.statistics.schemas import (
    CorrelationTestDecision,
)

from app.visualization.schemas import (
    VisualizationDecision,
)


# ============================================================
# VISUALIZATION RULES
# ============================================================
#
# These thresholds are DataLens rendering
# heuristics.
#
# They are deliberately explicit and versioned.
# ============================================================

HEXBIN_THRESHOLD = 3000

MAX_ORDINAL_LEVELS_FOR_BOXPLOT = 20

HIGH_TIE_FRACTION = 0.50


# ============================================================
# CONTINUOUS × CONTINUOUS
# ============================================================

def decide_continuous_continuous(
    decision: CorrelationTestDecision,
) -> VisualizationDecision:
    """
    Choose a relationship visualization for
    two continuous variables.
    """

    diagnostics = (
        decision.diagnostics
    )

    reasons = [
        (
            "Both variables are continuous."
        ),
        (
            "The analysis concerns the "
            "relationship between two "
            "quantitative variables."
        ),
    ]

    warnings = []

    # ========================================================
    # DENSE DATASET
    # ========================================================

    if (
        diagnostics.n_valid
        > HEXBIN_THRESHOLD
    ):
        reasons.append(
            (
                "The number of valid observations "
                "is high enough that a standard "
                "scatter plot could suffer from "
                "overplotting."
            )
        )

        reasons.append(
            (
                "A hexbin chart is selected to "
                "preserve the density structure "
                "of the relationship."
            )
        )

        return VisualizationDecision(
            status=
                "selected",

            purpose=
                "relationship",

            chart_type=
                "hexbin",

            x_column=
                decision.x_column,

            y_column=
                decision.y_column,

            aggregation=
                "count",

            trend=
                "none",

            show_raw_points=
                False,

            show_missing_summary=
                True,

            selection_is_data_driven=
                True,

            reasons=
                reasons,

            warnings=
                warnings,

            compatible_alternatives=[
                "scatter",
            ],
        )

    # ========================================================
    # STANDARD SCATTER
    # ========================================================

    reasons.append(
        (
            "A scatter plot preserves the "
            "individual paired observations."
        )
    )

    trend = "none"

    # --------------------------------------------------------
    # Pearson
    # --------------------------------------------------------

    if (
        decision.status
        == "selected"
        and decision.selected_test
        == "pearson"
    ):
        trend = "linear"

        reasons.append(
            (
                "Pearson was selected, so a "
                "linear trend may be displayed "
                "to match the linear association "
                "being analysed."
            )
        )

    # --------------------------------------------------------
    # Spearman
    # --------------------------------------------------------

    elif (
        decision.status
        == "selected"
        and decision.selected_test
        == "spearman"
    ):
        reasons.append(
            (
                "Spearman was selected, so "
                "DataLens does not impose a "
                "linear trend line on the "
                "visualization."
            )
        )

    # --------------------------------------------------------
    # No statistical test yet
    # --------------------------------------------------------

    else:
        reasons.append(
            (
                "No correlation test has been "
                "selected, so the scatter plot "
                "is treated as a diagnostic "
                "visualization."
            )
        )

    purpose = (
        "relationship"
        if decision.status
        == "selected"
        else "diagnostic"
    )

    return VisualizationDecision(
        status=
            "selected",

        purpose=
            purpose,

        chart_type=
            "scatter",

        x_column=
            decision.x_column,

        y_column=
            decision.y_column,

        aggregation=
            "none",

        trend=
            trend,

        show_raw_points=
            True,

        show_missing_summary=
            True,

        selection_is_data_driven=
            False,

        reasons=
            reasons,

        warnings=
            warnings,

        compatible_alternatives=[
            "hexbin",
        ],
    )


# ============================================================
# ORDINAL × ORDINAL
# ============================================================

def decide_ordinal_ordinal(
    decision: CorrelationTestDecision,
) -> VisualizationDecision:
    """
    Use a frequency heatmap for two ordinal
    variables.

    This avoids severe overlap when many rows
    share the same ranks.
    """

    diagnostics = (
        decision.diagnostics
    )

    reasons = [
        (
            "Both variables are ordinal."
        ),
        (
            "Ordinal variables contain a "
            "limited ordered set of levels."
        ),
        (
            "A frequency heatmap shows how "
            "observations are distributed across "
            "pairs of ordinal levels."
        ),
    ]

    warnings = []

    if (
        diagnostics
        .x_tied_observation_fraction
        >= HIGH_TIE_FRACTION
        or diagnostics
        .y_tied_observation_fraction
        >= HIGH_TIE_FRACTION
    ):
        reasons.append(
            (
                "A high proportion of observations "
                "share repeated ranks, so a raw "
                "scatter plot would produce heavy "
                "point overlap."
            )
        )

    return VisualizationDecision(
        status=
            "selected",

        purpose=
            "relationship",

        chart_type=
            "ordinal_heatmap",

        x_column=
            decision.x_column,

        y_column=
            decision.y_column,

        aggregation=
            "count",

        trend=
            "none",

        show_raw_points=
            False,

        show_missing_summary=
            True,

        selection_is_data_driven=
            False,

        reasons=
            reasons,

        warnings=
            warnings,

        compatible_alternatives=[
            "scatter",
        ],
    )


# ============================================================
# ORDINAL × CONTINUOUS
# ============================================================

def decide_ordinal_continuous(
    decision: CorrelationTestDecision,
) -> VisualizationDecision:
    """
    Compare the distribution of a continuous
    measure across ordered categories.
    """

    diagnostics = (
        decision.diagnostics
    )

    x_is_ordinal = (
        diagnostics.x_kind
        == "ordinal"
    )

    ordinal_column = (
        decision.x_column
        if x_is_ordinal
        else decision.y_column
    )

    continuous_column = (
        decision.y_column
        if x_is_ordinal
        else decision.x_column
    )

    ordinal_unique = (
        diagnostics.x_unique
        if x_is_ordinal
        else diagnostics.y_unique
    )

    reasons = [
        (
            "One variable is ordinal and the "
            "other is continuous."
        )
    ]

    warnings = []

    # ========================================================
    # MANAGEABLE NUMBER OF LEVELS
    # ========================================================

    if (
        ordinal_unique
        <= MAX_ORDINAL_LEVELS_FOR_BOXPLOT
    ):
        reasons.append(
            (
                "The ordinal variable has a "
                "manageable number of levels."
            )
        )

        reasons.append(
            (
                "A boxplot is selected to compare "
                "the continuous distribution "
                "across the ordered levels."
            )
        )

        return VisualizationDecision(
            status=
                "selected",

            purpose=
                "relationship",

            chart_type=
                "boxplot",

            x_column=
                ordinal_column,

            y_column=
                continuous_column,

            aggregation=
                "none",

            trend=
                "none",

            show_raw_points=
                False,

            show_missing_summary=
                True,

            selection_is_data_driven=
                False,

            reasons=
                reasons,

            warnings=
                warnings,

            compatible_alternatives=[
                "scatter",
            ],
        )

    # ========================================================
    # TOO MANY ORDINAL LEVELS
    # ========================================================

    warnings.append(
        (
            "The ordinal variable has many "
            "distinct levels. A grouped boxplot "
            "would become difficult to read."
        )
    )

    reasons.append(
        (
            "A scatter plot is selected as the "
            "fallback relationship view because "
            "the ordinal cardinality is high."
        )
    )

    return VisualizationDecision(
        status=
            "selected",

        purpose=
            "relationship",

        chart_type=
            "scatter",

        x_column=
            decision.x_column,

        y_column=
            decision.y_column,

        aggregation=
            "none",

        trend=
            "none",

        show_raw_points=
            True,

        show_missing_summary=
            True,

        selection_is_data_driven=
            True,

        reasons=
            reasons,

        warnings=
            warnings,

        compatible_alternatives=[
            "boxplot",
        ],
    )


# ============================================================
# MAIN VISUALIZATION DECISION ENGINE
# ============================================================

def decide_correlation_visualization(
    decision: CorrelationTestDecision,
) -> VisualizationDecision:
    """
    Select a visualization for a correlation /
    association decision.

    A statistical test does not need to have
    been selected.

    When DataLens requires more information,
    the chart may still be useful as a
    diagnostic view.
    """

    diagnostics = (
        decision.diagnostics
    )

    x_kind = (
        diagnostics.x_kind
    )

    y_kind = (
        diagnostics.y_kind
    )

    # ========================================================
    # CONTINUOUS × CONTINUOUS
    # ========================================================

    if (
        x_kind
        == "continuous"
        and y_kind
        == "continuous"
    ):
        return (
            decide_continuous_continuous(
                decision
            )
        )

    # ========================================================
    # ORDINAL × ORDINAL
    # ========================================================

    if (
        x_kind
        == "ordinal"
        and y_kind
        == "ordinal"
    ):
        return (
            decide_ordinal_ordinal(
                decision
            )
        )

    # ========================================================
    # ORDINAL × CONTINUOUS
    # ========================================================

    if (
        {
            x_kind,
            y_kind,
        }
        == {
            "ordinal",
            "continuous",
        }
    ):
        return (
            decide_ordinal_continuous(
                decision
            )
        )

    # ========================================================
    # UNSUPPORTED
    # ========================================================

    return VisualizationDecision(
        status=
            "not_applicable",

        purpose=
            "diagnostic",

        chart_type=
            None,

        x_column=
            decision.x_column,

        y_column=
            decision.y_column,

        aggregation=
            "none",

        trend=
            "none",

        show_raw_points=
            False,

        show_missing_summary=
            True,

        selection_is_data_driven=
            False,

        reasons=[
            (
                "The current association "
                "visualization engine supports "
                "continuous and ordinal variable "
                "combinations only."
            )
        ],

        warnings=[],

        compatible_alternatives=[],
    )