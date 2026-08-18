from __future__ import annotations

from typing import (
    Annotated,
    Literal,
)

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)


# ============================================================
# VERSION
# ============================================================

ANALYTICAL_PLANNER_CONTRACT_VERSION = (
    "analytical_planner_contract_v0.9"
)


# ============================================================
# COMMON TYPES
# ============================================================

NonEmptyString = Annotated[
    str,
    Field(
        min_length=1,
    ),
]


# ============================================================
# ANALYTICAL INTENTS
#
# We deliberately reuse the stable vocabulary already used in
# DataLens evaluations.
# ============================================================

AnalyticalIntent = Literal[
    "aggregate_metric",
    "compare_groups",
    "measure_relationship",
    "time_series_analysis",
    "distribution_analysis",
    "entity_anomaly_analysis",
    "data_quality_analysis",
]


# ============================================================
# ANALYTICAL FAMILIES
# ============================================================

AnalyticalFamily = Literal[
    "aggregation",
    "group_comparison",
    "association",
    "time_series",
    "distribution",
    "entity_outlier",
    "data_quality",
]


# ============================================================
# TOOL — AGGREGATE
# ============================================================

class AggregateToolCall(
    BaseModel
):
    """
    Aggregate one or more analytical variables.

    `metrics` contains planner-visible analytical column names.

    `group_by`:
        None for a global aggregation.

        A list of planner-visible columns when grouped
        aggregation is required.

    The deterministic planner validator will later verify that
    every referenced column exists in the corresponding
    AnalyticalPlannerInput.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    name: Literal[
        "aggregate"
    ]

    metrics: list[
        NonEmptyString
    ] = Field(
        min_length=1,
    )

    group_by: (
        list[
            NonEmptyString
        ]
        | None
    )


# ============================================================
# TOOL — BUILD ENTITY VIEW
# ============================================================

class BuildEntityViewToolCall(
    BaseModel
):
    """
    Request an entity-level analytical view.

    This is an analytical grain transformation, not a dataset
    join.

    Structural dataset composition remains outside the LLM
    planner.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    name: Literal[
        "build_entity_view"
    ]

    entity: NonEmptyString


# ============================================================
# TOOL — DERIVE METRIC
# ============================================================

class DeriveMetricToolCall(
    BaseModel
):
    """
    Define a derived analytical metric.

    inputs:
        Existing planner-visible columns or outputs from
        previous derive_metric steps.

    output:
        New local analytical metric name.

    formula:
        Declarative formula to be validated by Python before
        execution.

    The contract permits the proposal.

    Python will later determine whether the formula and
    dependencies are actually valid.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    name: Literal[
        "derive_metric"
    ]

    inputs: list[
        NonEmptyString
    ] = Field(
        min_length=1,
    )

    output: NonEmptyString

    formula: NonEmptyString


# ============================================================
# TOOL — ANALYZE DISTRIBUTION
# ============================================================

class AnalyzeDistributionToolCall(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid",
    )

    name: Literal[
        "analyze_distribution"
    ]

    target: NonEmptyString


# ============================================================
# TOOL — DETECT OUTLIERS
# ============================================================

class DetectOutliersToolCall(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid",
    )

    name: Literal[
        "detect_outliers"
    ]

    target: NonEmptyString


# ============================================================
# TOOL — DETECT ENTITY OUTLIERS
# ============================================================

class DetectEntityOutliersToolCall(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid",
    )

    name: Literal[
        "detect_entity_outliers"
    ]

    entity: NonEmptyString

    metrics: list[
        NonEmptyString
    ] = Field(
        min_length=1,
    )


# ============================================================
# TOOL — COMPARE GROUPS
# ============================================================

class CompareGroupsToolCall(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid",
    )

    name: Literal[
        "compare_groups"
    ]

    target: NonEmptyString

    group_by: NonEmptyString


# ============================================================
# TOOL — MEASURE ASSOCIATION
# ============================================================

class MeasureAssociationToolCall(
    BaseModel
):
    """
    `target` and `value` are intentionally symmetric at the
    analytical level.

    The deterministic scorer/validator should therefore not
    treat argument order as meaningfully different when the
    underlying association operation is symmetric.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    name: Literal[
        "measure_association"
    ]

    target: NonEmptyString

    value: NonEmptyString


# ============================================================
# TOOL — ANALYZE TIME SERIES
# ============================================================

class AnalyzeTimeSeriesToolCall(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid",
    )

    name: Literal[
        "analyze_time_series"
    ]

    date: NonEmptyString

    target: NonEmptyString


# ============================================================
# DISCRIMINATED TOOL UNION
#
# IMPORTANT:
#
# There is deliberately NO join_datasets variant.
#
# Structural composition has already been validated and
# resolved by Python before this planner is invoked.
# ============================================================

AnalyticalToolCall = Annotated[
    (
        AggregateToolCall
        | BuildEntityViewToolCall
        | DeriveMetricToolCall
        | AnalyzeDistributionToolCall
        | DetectOutliersToolCall
        | DetectEntityOutliersToolCall
        | CompareGroupsToolCall
        | MeasureAssociationToolCall
        | AnalyzeTimeSeriesToolCall
    ),
    Field(
        discriminator="name",
    ),
]


# ============================================================
# PLAN STEP
# ============================================================

class AnalyticalPlanStep(
    BaseModel
):
    """
    One ordered analytical action.

    step_id exists so future orchestration can attach:

    - execution state;
    - observations;
    - evidence;
    - errors;
    - retries or replanning decisions.

    That will later support the target agent loop:

        action
        -> observation
        -> decision
        -> next action
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    step_id: NonEmptyString

    action: AnalyticalToolCall


# ============================================================
# ONE REQUIREMENT PLAN
# ============================================================

class AnalyticalRequirementPlan(
    BaseModel
):
    """
    Analytical plan for exactly one requirement from
    AnalyticalPlannerInput.

    requirement_id must eventually match an input requirement
    exactly.

    That cross-object validation belongs to the deterministic
    planner validator, not this pure schema contract.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    requirement_id: NonEmptyString

    intent: AnalyticalIntent

    family: AnalyticalFamily

    target_grain: NonEmptyString

    steps: list[
        AnalyticalPlanStep
    ] = Field(
        min_length=1,
    )


    @model_validator(
        mode="after",
    )
    def validate_steps(
        self,
    ) -> "AnalyticalRequirementPlan":

        step_ids = [
            step.step_id

            for step
            in self.steps
        ]


        if (
            len(
                step_ids
            )
            != len(
                set(
                    step_ids
                )
            )
        ):
            raise ValueError(
                "step_id values must be unique "
                "inside one analytical requirement."
            )


        return self


# ============================================================
# COMPLETE PLANNER CANDIDATE
# ============================================================

class AnalyticalPlannerCandidate(
    BaseModel
):
    """
    Structured output produced by the analytical planning
    model.

    The contract itself only guarantees shape and vocabulary.

    It does NOT yet guarantee that:

    - requirement IDs exist in planner input;
    - every input requirement is covered;
    - tools are allowed for that requirement;
    - columns are planner-visible;
    - bridge-only columns remain unused;
    - derived metrics exist before use;
    - target grains are valid;
    - tool/family combinations are coherent.

    Those properties will be enforced by the deterministic
    Analytical Planner Validator v0.9.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    plans: list[
        AnalyticalRequirementPlan
    ] = Field(
        min_length=1,
    )


    @model_validator(
        mode="after",
    )
    def validate_plans(
        self,
    ) -> "AnalyticalPlannerCandidate":

        requirement_ids = [
            plan.requirement_id

            for plan
            in self.plans
        ]


        if (
            len(
                requirement_ids
            )
            != len(
                set(
                    requirement_ids
                )
            )
        ):
            raise ValueError(
                "requirement_id values must be unique "
                "inside an analytical planner candidate."
            )


        return self