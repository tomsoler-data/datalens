import pandas as pd

from app.statistics.diagnostics import (
    build_correlation_diagnostics,
)

from app.statistics.schemas import (
    AnalysisGoal,
    AnalysisMode,
    CorrelationTestDecision,
    CorrelationTestName,
    InferenceMethod,
    VariableKind,
)


# ============================================================
# DATALENS DECISION RULES
# ============================================================
#
# These are explicit project heuristics.
#
# They are versioned and must not be presented
# as universal statistical laws.
# ============================================================

MATERIAL_OUTLIER_FRACTION = 0.05

MINIMUM_MATERIAL_OUTLIER_COUNT = 2

PEARSON_SMALL_SAMPLE_HEURISTIC = 30

SPEARMAN_ASYMPTOTIC_SAMPLE_THRESHOLD = 500


# ============================================================
# OUTLIER SIGNAL
# ============================================================

def has_material_outlier_signal(
    outlier_count: int,
    outlier_fraction: float,
) -> bool:
    """
    Require both a minimum number and a minimum
    fraction of IQR flags before considering the
    signal material.

    One isolated flagged value does not silently
    change the selected statistical test.
    """

    return (
        outlier_count
        >= MINIMUM_MATERIAL_OUTLIER_COUNT
        and outlier_fraction
        >= MATERIAL_OUTLIER_FRACTION
    )


# ============================================================
# INFERENCE METHOD
# ============================================================

def choose_inference_method(
    selected_test: CorrelationTestName,
    n_valid: int,
    material_outlier_signal: bool,
) -> tuple[
    InferenceMethod,
    list[str],
]:
    """
    Decide separately how inference should
    eventually be performed.

    Test choice and inference-method choice are
    deliberately different decisions.
    """

    warnings = []

    # --------------------------------------------------------
    # Spearman
    # --------------------------------------------------------

    if (
        selected_test
        == "spearman"
    ):
        if (
            n_valid
            <= SPEARMAN_ASYMPTOTIC_SAMPLE_THRESHOLD
        ):
            warnings.append(
                (
                    "Permutation-based inference "
                    "is recommended for Spearman "
                    "because the sample does not "
                    "exceed the threshold used by "
                    "the current DataLens rule."
                )
            )

            return (
                "permutation_recommended",
                warnings,
            )

        return (
            "standard",
            warnings,
        )

    # --------------------------------------------------------
    # Pearson
    # --------------------------------------------------------

    if (
        n_valid
        < PEARSON_SMALL_SAMPLE_HEURISTIC
        or material_outlier_signal
    ):
        warnings.append(
            (
                "DataLens recommends considering "
                "permutation-based inference for "
                "this Pearson analysis because "
                "the sample is small under the "
                "current project heuristic or a "
                "material potential-outlier "
                "signal is present."
            )
        )

        return (
            "permutation_recommended",
            warnings,
        )

    return (
        "standard",
        warnings,
    )


# ============================================================
# SELECTED DECISION BUILDER
# ============================================================

def build_selected_decision(
    *,
    analysis_goal: AnalysisGoal,
    analysis_mode: AnalysisMode,
    x_column: str,
    y_column: str,
    selected_test: CorrelationTestName,
    reasons: list[str],
    warnings: list[str],
    diagnostics,
    selection_is_data_driven: bool,
) -> CorrelationTestDecision:
    """
    Build a selected decision and determine
    its inference strategy separately.
    """

    x_outlier_signal = (
        has_material_outlier_signal(
            outlier_count=
                diagnostics.x_outlier_count,

            outlier_fraction=
                diagnostics.x_outlier_fraction,
        )
    )

    y_outlier_signal = (
        has_material_outlier_signal(
            outlier_count=
                diagnostics.y_outlier_count,

            outlier_fraction=
                diagnostics.y_outlier_fraction,
        )
    )

    material_outlier_signal = (
        x_outlier_signal
        or y_outlier_signal
    )

    (
        inference_method,
        inference_warnings,
    ) = choose_inference_method(
        selected_test=
            selected_test,

        n_valid=
            diagnostics.n_valid,

        material_outlier_signal=
            material_outlier_signal,
    )

    return CorrelationTestDecision(
        status=
            "selected",

        analysis_goal=
            analysis_goal,

        analysis_mode=
            analysis_mode,

        x_column=
            x_column,

        y_column=
            y_column,

        selected_test=
            selected_test,

        inference_method=
            inference_method,

        selection_is_data_driven=
            selection_is_data_driven,

        reasons=
            reasons,

        missing_information=[],

        warnings=(
            warnings
            + inference_warnings
        ),

        diagnostics=
            diagnostics,
    )


# ============================================================
# MAIN DECISION ENGINE
# ============================================================

def decide_correlation_test(
    dataframe: pd.DataFrame,
    x_column: str,
    y_column: str,
    analysis_goal: AnalysisGoal,
    analysis_mode: AnalysisMode,
    x_kind: VariableKind = "unknown",
    y_kind: VariableKind = "unknown",
    observations_independent: (
        bool | None
    ) = None,
) -> CorrelationTestDecision:
    """
    Select a correlation test deterministically.

    Design principles:

    CONFIRMATORY:
        Do not choose Pearson or Spearman by
        comparing the observed coefficients.

    EXPLORATORY:
        Observed relationship-shape diagnostics
        may help select a candidate test, but the
        decision is explicitly marked data-driven.

    The function may refuse to select a test.
    """

    # ========================================================
    # SHOULD OUTCOME SHAPE BE INSPECTED?
    # ========================================================

    assess_shape = (
        analysis_mode
        == "exploratory"
        and analysis_goal
        == "general_association"
        and x_kind
        == "continuous"
        and y_kind
        == "continuous"
    )

    diagnostics = (
        build_correlation_diagnostics(
            dataframe=
                dataframe,

            x_column=
                x_column,

            y_column=
                y_column,

            x_kind=
                x_kind,

            y_kind=
                y_kind,

            assess_shape=
                assess_shape,
        )
    )

    warnings = list(
        diagnostics.warnings
    )

    # ========================================================
    # VARIABLE TYPE INFORMATION
    # ========================================================

    missing_information = []

    if (
        x_kind
        == "unknown"
    ):
        missing_information.append(
            (
                f"Measurement type of "
                f"{x_column!r}."
            )
        )

    if (
        y_kind
        == "unknown"
    ):
        missing_information.append(
            (
                f"Measurement type of "
                f"{y_column!r}."
            )
        )

    if missing_information:
        return CorrelationTestDecision(
            status=
                "needs_information",

            analysis_goal=
                analysis_goal,

            analysis_mode=
                analysis_mode,

            x_column=
                x_column,

            y_column=
                y_column,

            selected_test=None,

            inference_method=None,

            selection_is_data_driven=
                False,

            reasons=[
                (
                    "The measurement type of "
                    "each variable is required "
                    "before selecting a "
                    "correlation test."
                )
            ],

            missing_information=
                missing_information,

            warnings=
                warnings,

            diagnostics=
                diagnostics,
        )

    supported_kinds = {
        "continuous",
        "ordinal",
    }

    if (
        x_kind
        not in supported_kinds
        or y_kind
        not in supported_kinds
    ):
        return CorrelationTestDecision(
            status=
                "not_applicable",

            analysis_goal=
                analysis_goal,

            analysis_mode=
                analysis_mode,

            x_column=
                x_column,

            y_column=
                y_column,

            selected_test=None,

            inference_method=None,

            selection_is_data_driven=
                False,

            reasons=[
                (
                    "The quantitative-association "
                    "decision branch currently "
                    "supports only continuous and "
                    "ordinal variables."
                ),
                (
                    f"Received "
                    f"{x_column}={x_kind}, "
                    f"{y_column}={y_kind}."
                ),
            ],

            missing_information=[],

            warnings=
                warnings,

            diagnostics=
                diagnostics,
        )

    # ========================================================
    # OBSERVATION STRUCTURE
    # ========================================================

    if (
        observations_independent
        is None
    ):
        return CorrelationTestDecision(
            status=
                "needs_information",

            analysis_goal=
                analysis_goal,

            analysis_mode=
                analysis_mode,

            x_column=
                x_column,

            y_column=
                y_column,

            selected_test=None,

            inference_method=None,

            selection_is_data_driven=
                False,

            reasons=[
                (
                    "The independence structure "
                    "of observations cannot be "
                    "inferred safely from the "
                    "numeric values alone."
                )
            ],

            missing_information=[
                (
                    "Whether rows represent "
                    "independent observational "
                    "units or repeated/clustered "
                    "measurements."
                )
            ],

            warnings=
                warnings,

            diagnostics=
                diagnostics,
        )

    if (
        observations_independent
        is False
    ):
        return CorrelationTestDecision(
            status=
                "not_applicable",

            analysis_goal=
                analysis_goal,

            analysis_mode=
                analysis_mode,

            x_column=
                x_column,

            y_column=
                y_column,

            selected_test=None,

            inference_method=None,

            selection_is_data_driven=
                False,

            reasons=[
                (
                    "The current simple "
                    "Pearson/Spearman branch is "
                    "not intended for repeated "
                    "or clustered observations."
                ),
                (
                    "A method accounting for the "
                    "dependence structure should "
                    "be selected instead."
                ),
            ],

            missing_information=[],

            warnings=
                warnings,

            diagnostics=
                diagnostics,
        )

    base_reasons = [
        (
            "Rows are treated as independent "
            "observational units."
        )
    ]

    # ========================================================
    # ORDINAL VARIABLES
    # ========================================================

    if (
        x_kind == "ordinal"
        or y_kind == "ordinal"
    ):
        if (
            analysis_goal
            == "linear_association"
        ):
            return CorrelationTestDecision(
                status=
                    "not_applicable",

                analysis_goal=
                    analysis_goal,

                analysis_mode=
                    analysis_mode,

                x_column=
                    x_column,

                y_column=
                    y_column,

                selected_test=None,

                inference_method=None,

                selection_is_data_driven=
                    False,

                reasons=(
                    base_reasons
                    + [
                        (
                            "At least one variable "
                            "is ordinal, while the "
                            "requested goal is "
                            "specifically linear "
                            "association."
                        ),
                        (
                            "The current DataLens "
                            "branch will not treat "
                            "ordinal ranks as a "
                            "continuous linear "
                            "measurement scale."
                        ),
                    ]
                ),

                missing_information=[],

                warnings=
                    warnings,

                diagnostics=
                    diagnostics,
            )

        reasons = (
            base_reasons
            + [
                (
                    "At least one variable is "
                    "ordinal."
                ),
                (
                    "Rank-based monotonic "
                    "association is therefore "
                    "selected."
                ),
            ]
        )

        if (
            diagnostics
            .x_tied_observation_fraction
            > 0
            or diagnostics
            .y_tied_observation_fraction
            > 0
        ):
            warnings.append(
                (
                    "Repeated ranks / tied values "
                    "are present and are recorded "
                    "as part of the diagnostics."
                )
            )

        return build_selected_decision(
            analysis_goal=
                analysis_goal,

            analysis_mode=
                analysis_mode,

            x_column=
                x_column,

            y_column=
                y_column,

            selected_test=
                "spearman",

            reasons=
                reasons,

            warnings=
                warnings,

            diagnostics=
                diagnostics,

            selection_is_data_driven=
                False,
        )

    # ========================================================
    # BOTH VARIABLES CONTINUOUS
    # ========================================================

    base_reasons.append(
        (
            "Both variables are continuous."
        )
    )

    x_outlier_signal = (
        has_material_outlier_signal(
            outlier_count=
                diagnostics.x_outlier_count,

            outlier_fraction=
                diagnostics.x_outlier_fraction,
        )
    )

    y_outlier_signal = (
        has_material_outlier_signal(
            outlier_count=
                diagnostics.y_outlier_count,

            outlier_fraction=
                diagnostics.y_outlier_fraction,
        )
    )

    material_outlier_signal = (
        x_outlier_signal
        or y_outlier_signal
    )

    # ========================================================
    # EXPLICIT LINEAR GOAL
    # ========================================================

    if (
        analysis_goal
        == "linear_association"
    ):
        reasons = (
            base_reasons
            + [
                (
                    "The analytical goal was "
                    "specified as linear "
                    "association before the "
                    "correlation result was used "
                    "for test selection."
                ),
                (
                    "Pearson is selected because "
                    "it targets linear "
                    "association between "
                    "continuous variables."
                ),
            ]
        )

        if material_outlier_signal:
            warnings.append(
                (
                    "A material potential-outlier "
                    "signal is present. DataLens "
                    "does not silently switch the "
                    "requested linear estimand to "
                    "Spearman; the issue is "
                    "surfaced explicitly."
                )
            )

        return build_selected_decision(
            analysis_goal=
                analysis_goal,

            analysis_mode=
                analysis_mode,

            x_column=
                x_column,

            y_column=
                y_column,

            selected_test=
                "pearson",

            reasons=
                reasons,

            warnings=
                warnings,

            diagnostics=
                diagnostics,

            selection_is_data_driven=
                False,
        )

    # ========================================================
    # EXPLICIT MONOTONIC GOAL
    # ========================================================

    if (
        analysis_goal
        == "monotonic_association"
    ):
        reasons = (
            base_reasons
            + [
                (
                    "The analytical goal was "
                    "specified as monotonic "
                    "association before the "
                    "correlation result was used "
                    "for test selection."
                ),
                (
                    "Spearman is selected because "
                    "it targets monotonic "
                    "rank association."
                ),
            ]
        )

        return build_selected_decision(
            analysis_goal=
                analysis_goal,

            analysis_mode=
                analysis_mode,

            x_column=
                x_column,

            y_column=
                y_column,

            selected_test=
                "spearman",

            reasons=
                reasons,

            warnings=
                warnings,

            diagnostics=
                diagnostics,

            selection_is_data_driven=
                False,
        )

    # ========================================================
    # GENERAL ASSOCIATION + CONFIRMATORY
    # ========================================================

    if (
        analysis_mode
        == "confirmatory"
    ):
        return CorrelationTestDecision(
            status=
                "needs_information",

            analysis_goal=
                analysis_goal,

            analysis_mode=
                analysis_mode,

            x_column=
                x_column,

            y_column=
                y_column,

            selected_test=None,

            inference_method=None,

            selection_is_data_driven=
                False,

            reasons=(
                base_reasons
                + [
                    (
                        "The analysis is "
                        "confirmatory, but the "
                        "requested goal is only "
                        "general association."
                    ),
                    (
                        "DataLens will not inspect "
                        "the observed Pearson and "
                        "Spearman results and then "
                        "choose whichever appears "
                        "more favorable."
                    ),
                ]
            ),

            missing_information=[
                (
                    "Whether the confirmatory "
                    "target is specifically a "
                    "linear association or a "
                    "monotonic association."
                )
            ],

            warnings=
                warnings,

            diagnostics=
                diagnostics,
        )

    # ========================================================
    # GENERAL ASSOCIATION + EXPLORATORY
    # ========================================================
    #
    # From here the selection is explicitly
    # data-driven.
    # ========================================================

    warnings.append(
        (
            "The test recommendation is based "
            "on exploratory inspection of the "
            "observed relationship shape. "
            "This is data-driven selection, not "
            "a pre-specified confirmatory test."
        )
    )

    # --------------------------------------------------------
    # Small sample
    # --------------------------------------------------------

    if (
        diagnostics.reliability
        == "limited"
    ):
        return CorrelationTestDecision(
            status=
                "needs_information",

            analysis_goal=
                analysis_goal,

            analysis_mode=
                analysis_mode,

            x_column=
                x_column,

            y_column=
                y_column,

            selected_test=None,

            inference_method=None,

            selection_is_data_driven=
                True,

            reasons=(
                base_reasons
                + [
                    (
                        "The valid sample is too "
                        "small for the current "
                        "automatic exploratory "
                        "shape heuristic."
                    )
                ]
            ),

            missing_information=[
                (
                    "Manual scatter-plot or "
                    "domain-based assessment of "
                    "whether the relationship of "
                    "interest is linear or "
                    "monotonic."
                )
            ],

            warnings=
                warnings,

            diagnostics=
                diagnostics,
        )

    # --------------------------------------------------------
    # Material outliers
    # --------------------------------------------------------

    if material_outlier_signal:
        return CorrelationTestDecision(
            status=
                "needs_information",

            analysis_goal=
                analysis_goal,

            analysis_mode=
                analysis_mode,

            x_column=
                x_column,

            y_column=
                y_column,

            selected_test=None,

            inference_method=None,

            selection_is_data_driven=
                True,

            reasons=(
                base_reasons
                + [
                    (
                        "A material potential-"
                        "outlier signal is present."
                    ),
                    (
                        "DataLens does not "
                        "automatically change from "
                        "Pearson to Spearman merely "
                        "because outliers were "
                        "flagged."
                    ),
                ]
            ),

            missing_information=[
                (
                    "Review whether the flagged "
                    "observations are data errors, "
                    "legitimate extreme values, "
                    "or influential observations, "
                    "and inspect the relationship "
                    "visually."
                )
            ],

            warnings=
                warnings,

            diagnostics=
                diagnostics,
        )

    # --------------------------------------------------------
    # Linear candidate
    # --------------------------------------------------------

    if (
        diagnostics.shape_signal
        == "linear_candidate"
    ):
        return build_selected_decision(
            analysis_goal=
                analysis_goal,

            analysis_mode=
                analysis_mode,

            x_column=
                x_column,

            y_column=
                y_column,

            selected_test=
                "pearson",

            reasons=(
                base_reasons
                + [
                    (
                        "Exploratory diagnostics "
                        "identify an approximately "
                        "linear candidate pattern "
                        "under the current DataLens "
                        "heuristic."
                    )
                ]
            ),

            warnings=
                warnings,

            diagnostics=
                diagnostics,

            selection_is_data_driven=
                True,
        )

    # --------------------------------------------------------
    # Monotonic non-linear candidate
    # --------------------------------------------------------

    if (
        diagnostics.shape_signal
        == "monotonic_non_linear_candidate"
    ):
        return build_selected_decision(
            analysis_goal=
                analysis_goal,

            analysis_mode=
                analysis_mode,

            x_column=
                x_column,

            y_column=
                y_column,

            selected_test=
                "spearman",

            reasons=(
                base_reasons
                + [
                    (
                        "Exploratory diagnostics "
                        "identify a monotonic but "
                        "potentially non-linear "
                        "candidate pattern under "
                        "the current DataLens "
                        "heuristic."
                    )
                ]
            ),

            warnings=
                warnings,

            diagnostics=
                diagnostics,

            selection_is_data_driven=
                True,
        )

    # --------------------------------------------------------
    # Non-monotonic curve
    # --------------------------------------------------------

    if (
        diagnostics.shape_signal
        == "nonlinear_nonmonotonic_candidate"
    ):
        return CorrelationTestDecision(
            status=
                "needs_information",

            analysis_goal=
                analysis_goal,

            analysis_mode=
                analysis_mode,

            x_column=
                x_column,

            y_column=
                y_column,

            selected_test=None,

            inference_method=None,

            selection_is_data_driven=
                True,

            reasons=(
                base_reasons
                + [
                    (
                        "The exploratory "
                        "diagnostics suggest a "
                        "non-linear relationship "
                        "that is not clearly "
                        "monotonic."
                    ),
                    (
                        "Neither Pearson nor "
                        "Spearman is automatically "
                        "selected as the sole "
                        "summary."
                    ),
                ]
            ),

            missing_information=[
                (
                    "Visual inspection or a "
                    "model designed for the "
                    "observed non-linear "
                    "relationship."
                )
            ],

            warnings=
                warnings,

            diagnostics=
                diagnostics,
        )

    # --------------------------------------------------------
    # Conflicting signals
    # --------------------------------------------------------

    if (
        diagnostics.shape_signal
        == "conflicting"
    ):
        return CorrelationTestDecision(
            status=
                "needs_information",

            analysis_goal=
                analysis_goal,

            analysis_mode=
                analysis_mode,

            x_column=
                x_column,

            y_column=
                y_column,

            selected_test=None,

            inference_method=None,

            selection_is_data_driven=
                True,

            reasons=(
                base_reasons
                + [
                    (
                        "Exploratory association "
                        "signals conflict."
                    )
                ]
            ),

            missing_information=[
                (
                    "Manual inspection of the "
                    "relationship shape and "
                    "potential influential "
                    "observations."
                )
            ],

            warnings=
                warnings,

            diagnostics=
                diagnostics,
        )

    # --------------------------------------------------------
    # No clear pattern / insufficient
    # --------------------------------------------------------

    return CorrelationTestDecision(
        status=
            "needs_information",

        analysis_goal=
            analysis_goal,

        analysis_mode=
            analysis_mode,

        x_column=
            x_column,

        y_column=
            y_column,

        selected_test=None,

        inference_method=None,

        selection_is_data_driven=
            True,

        reasons=(
            base_reasons
            + [
                (
                    "The exploratory diagnostics "
                    "do not establish a clear "
                    "enough linear or monotonic "
                    "candidate pattern to choose "
                    "Pearson or Spearman "
                    "automatically."
                )
            ]
        ),

        missing_information=[
            (
                "Manual visual inspection or "
                "an explicitly specified "
                "analytical target."
            )
        ],

        warnings=
            warnings,

        diagnostics=
            diagnostics,
    )