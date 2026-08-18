from typing import Literal

from pydantic import (
    BaseModel,
    Field,
)


# ============================================================
# CORRELATION TEST TYPES
# ============================================================

CorrelationTestName = Literal[
    "pearson",
    "spearman",
]


RelationshipType = Literal[
    "linear",
    "monotonic",
]


CoefficientName = Literal[
    "r",
    "rho",
]


AlternativeHypothesis = Literal[
    "two-sided",
]


# ============================================================
# VARIABLE TYPES
# ============================================================

VariableKind = Literal[
    "continuous",
    "ordinal",
    "binary",
    "nominal",
    "temporal",
    "unknown",
]


# ============================================================
# ANALYSIS INTENT
# ============================================================

AnalysisGoal = Literal[
    "linear_association",
    "monotonic_association",
    "general_association",
]


AnalysisMode = Literal[
    "confirmatory",
    "exploratory",
]


# ============================================================
# DIAGNOSTIC TYPES
# ============================================================

ShapeSignal = Literal[
    "not_evaluated",
    "linear_candidate",
    "monotonic_non_linear_candidate",
    "nonlinear_nonmonotonic_candidate",
    "no_clear_pattern",
    "conflicting",
    "insufficient_for_shape",
]


DiagnosticReliability = Literal[
    "limited",
    "standard",
]


# ============================================================
# DECISION TYPES
# ============================================================

DecisionStatus = Literal[
    "selected",
    "needs_information",
    "not_applicable",
]


InferenceMethod = Literal[
    "standard",
    "permutation_recommended",
]


# ============================================================
# EXECUTION TYPES
# ============================================================

InferenceMethodUsed = Literal[
    "standard",
    "permutation",
]


PermutationMode = Literal[
    "exact",
    "randomized",
]


# ============================================================
# CORRELATION RESULT
# ============================================================

class CorrelationResult(
    BaseModel
):
    """
    Deterministic result of one
    correlation test.
    """

    test: CorrelationTestName

    relationship_type: (
        RelationshipType
    )

    coefficient_name: (
        CoefficientName
    )

    coefficient: float = Field(
        ge=-1.0,
        le=1.0,
    )

    p_value: float = Field(
        ge=0.0,
        le=1.0,
    )

    alternative: (
        AlternativeHypothesis
    ) = "two-sided"

    n: int = Field(
        ge=3,
    )

    alpha: float = Field(
        gt=0.0,
        lt=1.0,
    )

    statistically_significant: bool


# ============================================================
# COMPLETE LEGACY CORRELATION ANALYSIS
# ============================================================

class CorrelationAnalysis(
    BaseModel
):
    """
    Complete deterministic analysis of
    two quantitative variables.

    This model is retained for the existing
    DataLens validation pipeline.

    The new decision/execution pipeline does
    not need to run both tests.
    """

    x_column: str = Field(
        min_length=1,
    )

    y_column: str = Field(
        min_length=1,
    )

    n_total: int = Field(
        ge=0,
    )

    n_valid: int = Field(
        ge=3,
    )

    n_excluded: int = Field(
        ge=0,
    )

    x_unique: int = Field(
        ge=2,
    )

    y_unique: int = Field(
        ge=2,
    )

    pearson: CorrelationResult

    spearman: CorrelationResult

    warnings: list[
        str
    ] = Field(
        default_factory=list,
    )


# ============================================================
# CORRELATION DIAGNOSTICS
# ============================================================

class CorrelationDiagnostics(
    BaseModel
):
    """
    Deterministic diagnostics used by the
    statistical decision engine.

    Outcome-shape diagnostics may be disabled
    for confirmatory analyses.
    """

    x_column: str = Field(
        min_length=1,
    )

    y_column: str = Field(
        min_length=1,
    )

    x_kind: VariableKind

    y_kind: VariableKind

    n_total: int = Field(
        ge=0,
    )

    n_valid: int = Field(
        ge=3,
    )

    n_excluded: int = Field(
        ge=0,
    )

    excluded_fraction: float = Field(
        ge=0.0,
        le=1.0,
    )

    x_unique: int = Field(
        ge=2,
    )

    y_unique: int = Field(
        ge=2,
    )

    x_tied_observation_count: int = Field(
        ge=0,
    )

    y_tied_observation_count: int = Field(
        ge=0,
    )

    x_tied_observation_fraction: float = Field(
        ge=0.0,
        le=1.0,
    )

    y_tied_observation_fraction: float = Field(
        ge=0.0,
        le=1.0,
    )

    x_outlier_count: int = Field(
        ge=0,
    )

    y_outlier_count: int = Field(
        ge=0,
    )

    x_outlier_fraction: float = Field(
        ge=0.0,
        le=1.0,
    )

    y_outlier_fraction: float = Field(
        ge=0.0,
        le=1.0,
    )

    pearson_coefficient: (
        float | None
    ) = Field(
        default=None,
        ge=-1.0,
        le=1.0,
    )

    spearman_coefficient: (
        float | None
    ) = Field(
        default=None,
        ge=-1.0,
        le=1.0,
    )

    coefficient_gap: (
        float | None
    ) = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    linear_r_squared: (
        float | None
    ) = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    quadratic_r_squared: (
        float | None
    ) = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    quadratic_r_squared_gain: (
        float | None
    ) = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    same_direction: (
        bool | None
    ) = None

    shape_signal: ShapeSignal

    reliability: (
        DiagnosticReliability
    )

    data_driven_shape_assessment: bool

    warnings: list[
        str
    ] = Field(
        default_factory=list,
    )


# ============================================================
# STATISTICAL TEST DECISION
# ============================================================

class CorrelationTestDecision(
    BaseModel
):
    """
    Traceable deterministic decision about
    which correlation test should be used.

    A decision may deliberately refuse to
    select a test.
    """

    status: DecisionStatus

    analysis_goal: AnalysisGoal

    analysis_mode: AnalysisMode

    x_column: str = Field(
        min_length=1,
    )

    y_column: str = Field(
        min_length=1,
    )

    selected_test: (
        CorrelationTestName | None
    ) = None

    inference_method: (
        InferenceMethod | None
    ) = None

    selection_is_data_driven: bool = False

    reasons: list[
        str
    ] = Field(
        default_factory=list,
    )

    missing_information: list[
        str
    ] = Field(
        default_factory=list,
    )

    warnings: list[
        str
    ] = Field(
        default_factory=list,
    )

    diagnostics: CorrelationDiagnostics

    decision_rule_version: str = (
        "quantitative_association_v0.2"
    )


# ============================================================
# SELECTED TEST EXECUTION
# ============================================================

class CorrelationExecution(
    BaseModel
):
    """
    Result of executing one selected
    correlation test.

    The decision and the numerical result are
    deliberately stored separately.

    decision:
        why this test was chosen

    result:
        what the selected test calculated
    """

    x_column: str = Field(
        min_length=1,
    )

    y_column: str = Field(
        min_length=1,
    )

    selected_test: CorrelationTestName

    inference_method_used: (
        InferenceMethodUsed
    )

    permutation_mode: (
        PermutationMode | None
    ) = None

    permutation_resamples_requested: (
        int | None
    ) = Field(
        default=None,
        ge=1,
    )

    random_seed: (
        int | None
    ) = None

    n_total: int = Field(
        ge=0,
    )

    n_valid: int = Field(
        ge=3,
    )

    n_excluded: int = Field(
        ge=0,
    )

    result: CorrelationResult

    decision: CorrelationTestDecision

    warnings: list[
        str
    ] = Field(
        default_factory=list,
    )

    execution_rule_version: str = (
        "correlation_executor_v0.1"
    )