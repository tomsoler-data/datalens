from __future__ import annotations

from typing import (
    Literal,
)

from pydantic import (
    BaseModel,
    ConfigDict,
)

from app.evals.analytical_planner_contract_v0_9 import (
    AggregateToolCall,
    AnalyticalPlannerCandidate,
    AnalyzeDistributionToolCall,
    AnalyzeTimeSeriesToolCall,
    BuildEntityViewToolCall,
    CompareGroupsToolCall,
    DeriveMetricToolCall,
    DetectEntityOutliersToolCall,
    DetectOutliersToolCall,
    MeasureAssociationToolCall,
)

from app.evals.analytical_planner_input_v0_9 import (
    AnalyticalPlannerInput,
    AnalyticalPlannerRequirementInput,
)


# ============================================================
# VERSION
# ============================================================

ANALYTICAL_PLANNER_VALIDATOR_VERSION = (
    "analytical_planner_validator_v0.9"
)


# ============================================================
# ISSUE CODES
# ============================================================

ValidationIssueCode = Literal[
    "unknown_requirement",
    "missing_requirement",
    "intent_family_mismatch",
    "invalid_target_grain",
    "tool_not_allowed",
    "unknown_analytical_reference",
    "bridge_column_reference",
    "derived_metric_used_before_definition",
    "derived_metric_collision",
    "duplicate_derived_metric",
    "invalid_entity_reference",
    "entity_view_required",
    "entity_view_mismatch",
    "temporal_column_required",
    "association_requires_distinct_variables",
    "missing_family_anchor",
]


# ============================================================
# INTENT / FAMILY COHERENCE
# ============================================================

INTENT_FAMILY_MAP = {
    "aggregate_metric":
        "aggregation",

    "compare_groups":
        "group_comparison",

    "measure_relationship":
        "association",

    "time_series_analysis":
        "time_series",

    "distribution_analysis":
        "distribution",

    "entity_anomaly_analysis":
        "entity_outlier",

    "data_quality_analysis":
        "data_quality",
}


# ============================================================
# FAMILY ANCHOR TOOLS
#
# A plan may contain preparation steps such as derive_metric,
# but it must ultimately contain an analytical action coherent
# with its declared family.
# ============================================================

FAMILY_ANCHOR_TOOLS = {
    "aggregation": {
        "aggregate",
    },

    "group_comparison": {
        "compare_groups",
    },

    "association": {
        "measure_association",
    },

    "time_series": {
        "analyze_time_series",
    },

    "distribution": {
        "analyze_distribution",
        "detect_outliers",
    },

    "entity_outlier": {
        "detect_entity_outliers",
    },

    "data_quality": {
        "analyze_distribution",
        "detect_outliers",
    },
}


# ============================================================
# DERIVED TEMPORAL GRAINS
#
# If a requirement exposes at least one temporal analytical
# column, the planner may target one of these standard temporal
# aggregation grains.
# ============================================================

STANDARD_TEMPORAL_GRAINS = {
    "hour",
    "day",
    "week",
    "month",
    "quarter",
    "year",
}


# ============================================================
# VALIDATION ISSUE
# ============================================================

class AnalyticalPlannerValidationIssue(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid",
    )

    code: ValidationIssueCode

    requirement_id: (
        str
        | None
    )

    step_id: (
        str
        | None
    )

    message: str


# ============================================================
# VALIDATION RESULT
# ============================================================

class AnalyticalPlannerValidationResult(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid",
    )

    validator_version: str

    valid: bool

    validated_requirement_ids: list[
        str
    ]

    issues: list[
        AnalyticalPlannerValidationIssue
    ]


# ============================================================
# INPUT INDEX
# ============================================================

def _index_input_requirements(
    planner_input: AnalyticalPlannerInput,
) -> dict[
    str,
    AnalyticalPlannerRequirementInput,
]:

    result: dict[
        str,
        AnalyticalPlannerRequirementInput
    ] = {}


    for requirement in (
        planner_input.requirements
    ):

        if (
            requirement.requirement_id
            in result
        ):
            raise ValueError(
                "AnalyticalPlannerInput contains duplicate "
                "requirement_id: "
                f"{requirement.requirement_id}"
            )


        result[
            requirement.requirement_id
        ] = requirement


    return result


# ============================================================
# ANALYTICAL COLUMN CATALOG
# ============================================================

def _analytical_column_catalog(
    requirement: AnalyticalPlannerRequirementInput,
) -> dict[
    str,
    dict,
]:

    catalog: dict[
        str,
        dict,
    ] = {}


    for column in (
        requirement.analytical_columns
    ):

        qualified_name = (
            column[
                "qualified_name"
            ]
        )


        catalog[
            qualified_name
        ] = column


    return catalog


# ============================================================
# BRIDGE DATASETS
# ============================================================

def _bridge_dataset_ids(
    requirement: AnalyticalPlannerRequirementInput,
) -> set[str]:

    return {
        dataset.dataset_id

        for dataset
        in requirement.datasets

        if (
            dataset.role
            == "bridge"
        )
    }


# ============================================================
# SEMANTIC ENTITY REFERENCES
# ============================================================

def _semantic_entity_references(
    requirement: AnalyticalPlannerRequirementInput,
) -> set[str]:

    result: set[
        str
    ] = set()


    for dataset in requirement.datasets:

        if (
            dataset.role
            != "semantic"
        ):
            continue


        for entity_column in (
            dataset.entity_columns
        ):

            result.add(
                (
                    f"{dataset.dataset_id}"
                    f".{entity_column}"
                )
            )


    return result


# ============================================================
# TARGET GRAINS
# ============================================================

def _allowed_target_grains(
    requirement: AnalyticalPlannerRequirementInput,
) -> set[str]:

    allowed = {
        "global",
    }


    # ========================================================
    # SEMANTIC DATASET GRAINS
    # ========================================================

    for dataset in requirement.datasets:

        if (
            dataset.role
            != "semantic"
        ):
            continue


        allowed.add(
            dataset.grain
        )


        # ====================================================
        # ENTITY-GRAIN CANDIDATES
        #
        # customer_id -> customer
        # store_id    -> store
        #
        # We also retain the literal entity column name.
        # ====================================================

        for entity_column in (
            dataset.entity_columns
        ):

            allowed.add(
                entity_column
            )


            if (
                entity_column.endswith(
                    "_id"
                )
                and len(
                    entity_column
                )
                > 3
            ):

                allowed.add(
                    entity_column[
                        :-3
                    ]
                )


    # ========================================================
    # STANDARD TEMPORAL GRAINS
    # ========================================================

    has_temporal_column = any(
        column.get(
            "analytical_type"
        )
        == "temporal"

        for column
        in requirement.analytical_columns
    )


    if has_temporal_column:

        allowed.update(
            STANDARD_TEMPORAL_GRAINS
        )


    return allowed


# ============================================================
# ISSUE HELPER
# ============================================================

def _issue(
    issues: list[
        AnalyticalPlannerValidationIssue
    ],
    *,
    code: ValidationIssueCode,
    requirement_id: str | None,
    step_id: str | None,
    message: str,
) -> None:

    issues.append(
        AnalyticalPlannerValidationIssue(
            code=code,
            requirement_id=(
                requirement_id
            ),
            step_id=(
                step_id
            ),
            message=(
                message
            ),
        )
    )


# ============================================================
# REFERENCE VALIDATION
# ============================================================

def _validate_reference(
    *,
    reference: str,
    requirement_id: str,
    step_id: str,
    base_columns: set[str],
    bridge_dataset_ids: set[str],
    all_derived_outputs: set[str],
    defined_derived_outputs: set[str],
    issues: list[
        AnalyticalPlannerValidationIssue
    ],
) -> bool:
    """
    Validate one analytical reference.

    A reference may point to:

    - a planner-visible qualified base column;
    - a derived metric produced by an earlier step.

    It may NOT point to:

    - bridge-only analytical columns;
    - future derived metrics;
    - invented columns.
    """

    if (
        reference
        in base_columns
    ):
        return True


    if (
        reference
        in defined_derived_outputs
    ):
        return True


    # ========================================================
    # FUTURE DERIVED METRIC
    # ========================================================

    if (
        reference
        in all_derived_outputs
    ):

        _issue(
            issues,
            code=(
                "derived_metric_used_before_definition"
            ),
            requirement_id=(
                requirement_id
            ),
            step_id=(
                step_id
            ),
            message=(
                "Derived metric is referenced before it "
                f"is defined: {reference}"
            ),
        )


        return False


    # ========================================================
    # BRIDGE COLUMN LEAK
    # ========================================================

    if "." in reference:

        dataset_id = (
            reference.split(
                ".",
                1,
            )[
                0
            ]
        )


        if (
            dataset_id
            in bridge_dataset_ids
        ):

            _issue(
                issues,
                code=(
                    "bridge_column_reference"
                ),
                requirement_id=(
                    requirement_id
                ),
                step_id=(
                    step_id
                ),
                message=(
                    "Bridge-only dataset column cannot be "
                    "used as an analytical variable: "
                    f"{reference}"
                ),
            )


            return False


    # ========================================================
    # UNKNOWN REFERENCE
    # ========================================================

    _issue(
        issues,
        code=(
            "unknown_analytical_reference"
        ),
        requirement_id=(
            requirement_id
        ),
        step_id=(
            step_id
        ),
        message=(
            "Unknown analytical reference: "
            f"{reference}"
        ),
    )


    return False


# ============================================================
# ENTITY VALIDATION
# ============================================================

def _validate_entity_reference(
    *,
    entity: str,
    semantic_entities: set[str],
    requirement_id: str,
    step_id: str,
    issues: list[
        AnalyticalPlannerValidationIssue
    ],
) -> bool:

    if (
        entity
        in semantic_entities
    ):
        return True


    _issue(
        issues,
        code=(
            "invalid_entity_reference"
        ),
        requirement_id=(
            requirement_id
        ),
        step_id=(
            step_id
        ),
        message=(
            "Entity reference is not an entity column from "
            "a semantic dataset: "
            f"{entity}"
        ),
    )


    return False


# ============================================================
# REQUIREMENT PLAN VALIDATION
# ============================================================

def _validate_requirement_plan(
    *,
    plan,
    requirement: AnalyticalPlannerRequirementInput,
    issues: list[
        AnalyticalPlannerValidationIssue
    ],
) -> None:

    requirement_id = (
        plan.requirement_id
    )


    # ========================================================
    # INTENT ↔ FAMILY
    # ========================================================

    expected_family = (
        INTENT_FAMILY_MAP[
            plan.intent
        ]
    )


    if (
        plan.family
        != expected_family
    ):

        _issue(
            issues,
            code=(
                "intent_family_mismatch"
            ),
            requirement_id=(
                requirement_id
            ),
            step_id=None,
            message=(
                "Intent/family mismatch: "
                f"{plan.intent} expects "
                f"{expected_family}, received "
                f"{plan.family}."
            ),
        )


    # ========================================================
    # TARGET GRAIN
    # ========================================================

    allowed_grains = (
        _allowed_target_grains(
            requirement
        )
    )


    if (
        plan.target_grain
        not in allowed_grains
    ):

        _issue(
            issues,
            code=(
                "invalid_target_grain"
            ),
            requirement_id=(
                requirement_id
            ),
            step_id=None,
            message=(
                "Target grain is not supported by the "
                "planner input: "
                f"{plan.target_grain}. "
                "Allowed grains: "
                f"{sorted(allowed_grains)}"
            ),
        )


    # ========================================================
    # CATALOGS
    # ========================================================

    analytical_catalog = (
        _analytical_column_catalog(
            requirement
        )
    )


    base_columns = set(
        analytical_catalog
    )


    bridge_datasets = (
        _bridge_dataset_ids(
            requirement
        )
    )


    semantic_entities = (
        _semantic_entity_references(
            requirement
        )
    )


    allowed_tools = set(
        requirement.allowed_analytical_tools
    )


    # ========================================================
    # ALL DERIVED OUTPUTS
    #
    # This allows us to distinguish:
    #
    # invented_metric
    #
    # from:
    #
    # valid_metric_but_used_too_early
    # ========================================================

    all_derived_outputs = {
        step.action.output

        for step
        in plan.steps

        if isinstance(
            step.action,
            DeriveMetricToolCall,
        )
    }


    defined_derived_outputs: set[
        str
    ] = set()


    entity_views: set[
        str
    ] = set()


    used_action_names: set[
        str
    ] = set()


    # ========================================================
    # ORDERED STEP VALIDATION
    # ========================================================

    for step in plan.steps:

        step_id = (
            step.step_id
        )


        action = (
            step.action
        )


        action_name = (
            action.name
        )


        used_action_names.add(
            action_name
        )


        # ====================================================
        # TOOL PERMISSION
        # ====================================================

        if (
            action_name
            not in allowed_tools
        ):

            _issue(
                issues,
                code=(
                    "tool_not_allowed"
                ),
                requirement_id=(
                    requirement_id
                ),
                step_id=(
                    step_id
                ),
                message=(
                    "Analytical tool is not allowed for "
                    "this requirement: "
                    f"{action_name}"
                ),
            )


        # ====================================================
        # AGGREGATE
        # ====================================================

        if isinstance(
            action,
            AggregateToolCall,
        ):

            for metric in (
                action.metrics
            ):

                _validate_reference(
                    reference=(
                        metric
                    ),
                    requirement_id=(
                        requirement_id
                    ),
                    step_id=(
                        step_id
                    ),
                    base_columns=(
                        base_columns
                    ),
                    bridge_dataset_ids=(
                        bridge_datasets
                    ),
                    all_derived_outputs=(
                        all_derived_outputs
                    ),
                    defined_derived_outputs=(
                        defined_derived_outputs
                    ),
                    issues=(
                        issues
                    ),
                )


            if (
                action.group_by
                is not None
            ):

                for group_reference in (
                    action.group_by
                ):

                    _validate_reference(
                        reference=(
                            group_reference
                        ),
                        requirement_id=(
                            requirement_id
                        ),
                        step_id=(
                            step_id
                        ),
                        base_columns=(
                            base_columns
                        ),
                        bridge_dataset_ids=(
                            bridge_datasets
                        ),
                        all_derived_outputs=(
                            all_derived_outputs
                        ),
                        defined_derived_outputs=(
                            defined_derived_outputs
                        ),
                        issues=(
                            issues
                        ),
                    )


        # ====================================================
        # BUILD ENTITY VIEW
        # ====================================================

        elif isinstance(
            action,
            BuildEntityViewToolCall,
        ):

            entity_valid = (
                _validate_entity_reference(
                    entity=(
                        action.entity
                    ),
                    semantic_entities=(
                        semantic_entities
                    ),
                    requirement_id=(
                        requirement_id
                    ),
                    step_id=(
                        step_id
                    ),
                    issues=(
                        issues
                    ),
                )
            )


            if entity_valid:

                entity_views.add(
                    action.entity
                )


        # ====================================================
        # DERIVE METRIC
        # ====================================================

        elif isinstance(
            action,
            DeriveMetricToolCall,
        ):

            for input_reference in (
                action.inputs
            ):

                _validate_reference(
                    reference=(
                        input_reference
                    ),
                    requirement_id=(
                        requirement_id
                    ),
                    step_id=(
                        step_id
                    ),
                    base_columns=(
                        base_columns
                    ),
                    bridge_dataset_ids=(
                        bridge_datasets
                    ),
                    all_derived_outputs=(
                        all_derived_outputs
                    ),
                    defined_derived_outputs=(
                        defined_derived_outputs
                    ),
                    issues=(
                        issues
                    ),
                )


            # =================================================
            # OUTPUT COLLISION
            # =================================================

            if (
                action.output
                in base_columns
            ):

                _issue(
                    issues,
                    code=(
                        "derived_metric_collision"
                    ),
                    requirement_id=(
                        requirement_id
                    ),
                    step_id=(
                        step_id
                    ),
                    message=(
                        "Derived metric output collides "
                        "with an existing analytical column: "
                        f"{action.output}"
                    ),
                )


            elif (
                action.output
                in defined_derived_outputs
            ):

                _issue(
                    issues,
                    code=(
                        "duplicate_derived_metric"
                    ),
                    requirement_id=(
                        requirement_id
                    ),
                    step_id=(
                        step_id
                    ),
                    message=(
                        "Derived metric is defined more "
                        "than once: "
                        f"{action.output}"
                    ),
                )


            else:

                # ---------------------------------------------
                # The formula itself is deliberately not
                # executed here.
                #
                # Formula AST / safe expression validation
                # belongs to the execution guardrail layer.
                # ---------------------------------------------

                defined_derived_outputs.add(
                    action.output
                )


        # ====================================================
        # DISTRIBUTION
        # ====================================================

        elif isinstance(
            action,
            AnalyzeDistributionToolCall,
        ):

            _validate_reference(
                reference=(
                    action.target
                ),
                requirement_id=(
                    requirement_id
                ),
                step_id=(
                    step_id
                ),
                base_columns=(
                    base_columns
                ),
                bridge_dataset_ids=(
                    bridge_datasets
                ),
                all_derived_outputs=(
                    all_derived_outputs
                ),
                defined_derived_outputs=(
                    defined_derived_outputs
                ),
                issues=(
                    issues
                ),
            )


        # ====================================================
        # VARIABLE OUTLIERS
        # ====================================================

        elif isinstance(
            action,
            DetectOutliersToolCall,
        ):

            _validate_reference(
                reference=(
                    action.target
                ),
                requirement_id=(
                    requirement_id
                ),
                step_id=(
                    step_id
                ),
                base_columns=(
                    base_columns
                ),
                bridge_dataset_ids=(
                    bridge_datasets
                ),
                all_derived_outputs=(
                    all_derived_outputs
                ),
                defined_derived_outputs=(
                    defined_derived_outputs
                ),
                issues=(
                    issues
                ),
            )


        # ====================================================
        # ENTITY OUTLIERS
        # ====================================================

        elif isinstance(
            action,
            DetectEntityOutliersToolCall,
        ):

            entity_valid = (
                _validate_entity_reference(
                    entity=(
                        action.entity
                    ),
                    semantic_entities=(
                        semantic_entities
                    ),
                    requirement_id=(
                        requirement_id
                    ),
                    step_id=(
                        step_id
                    ),
                    issues=(
                        issues
                    ),
                )
            )


            for metric in (
                action.metrics
            ):

                _validate_reference(
                    reference=(
                        metric
                    ),
                    requirement_id=(
                        requirement_id
                    ),
                    step_id=(
                        step_id
                    ),
                    base_columns=(
                        base_columns
                    ),
                    bridge_dataset_ids=(
                        bridge_datasets
                    ),
                    all_derived_outputs=(
                        all_derived_outputs
                    ),
                    defined_derived_outputs=(
                        defined_derived_outputs
                    ),
                    issues=(
                        issues
                    ),
                )


            if (
                entity_valid
                and action.entity
                not in entity_views
            ):

                _issue(
                    issues,
                    code=(
                        "entity_view_required"
                    ),
                    requirement_id=(
                        requirement_id
                    ),
                    step_id=(
                        step_id
                    ),
                    message=(
                        "detect_entity_outliers requires a "
                        "previous build_entity_view step for "
                        f"the same entity: {action.entity}"
                    ),
                )


        # ====================================================
        # GROUP COMPARISON
        # ====================================================

        elif isinstance(
            action,
            CompareGroupsToolCall,
        ):

            _validate_reference(
                reference=(
                    action.target
                ),
                requirement_id=(
                    requirement_id
                ),
                step_id=(
                    step_id
                ),
                base_columns=(
                    base_columns
                ),
                bridge_dataset_ids=(
                    bridge_datasets
                ),
                all_derived_outputs=(
                    all_derived_outputs
                ),
                defined_derived_outputs=(
                    defined_derived_outputs
                ),
                issues=(
                    issues
                ),
            )


            _validate_reference(
                reference=(
                    action.group_by
                ),
                requirement_id=(
                    requirement_id
                ),
                step_id=(
                    step_id
                ),
                base_columns=(
                    base_columns
                ),
                bridge_dataset_ids=(
                    bridge_datasets
                ),
                all_derived_outputs=(
                    all_derived_outputs
                ),
                defined_derived_outputs=(
                    defined_derived_outputs
                ),
                issues=(
                    issues
                ),
            )


        # ====================================================
        # ASSOCIATION
        # ====================================================

        elif isinstance(
            action,
            MeasureAssociationToolCall,
        ):

            _validate_reference(
                reference=(
                    action.target
                ),
                requirement_id=(
                    requirement_id
                ),
                step_id=(
                    step_id
                ),
                base_columns=(
                    base_columns
                ),
                bridge_dataset_ids=(
                    bridge_datasets
                ),
                all_derived_outputs=(
                    all_derived_outputs
                ),
                defined_derived_outputs=(
                    defined_derived_outputs
                ),
                issues=(
                    issues
                ),
            )


            _validate_reference(
                reference=(
                    action.value
                ),
                requirement_id=(
                    requirement_id
                ),
                step_id=(
                    step_id
                ),
                base_columns=(
                    base_columns
                ),
                bridge_dataset_ids=(
                    bridge_datasets
                ),
                all_derived_outputs=(
                    all_derived_outputs
                ),
                defined_derived_outputs=(
                    defined_derived_outputs
                ),
                issues=(
                    issues
                ),
            )


            if (
                action.target
                == action.value
            ):

                _issue(
                    issues,
                    code=(
                        "association_requires_distinct_variables"
                    ),
                    requirement_id=(
                        requirement_id
                    ),
                    step_id=(
                        step_id
                    ),
                    message=(
                        "Association requires two distinct "
                        "analytical variables."
                    ),
                )


        # ====================================================
        # TIME SERIES
        # ====================================================

        elif isinstance(
            action,
            AnalyzeTimeSeriesToolCall,
        ):

            date_valid = (
                _validate_reference(
                    reference=(
                        action.date
                    ),
                    requirement_id=(
                        requirement_id
                    ),
                    step_id=(
                        step_id
                    ),
                    base_columns=(
                        base_columns
                    ),
                    bridge_dataset_ids=(
                        bridge_datasets
                    ),
                    all_derived_outputs=(
                        all_derived_outputs
                    ),
                    defined_derived_outputs=(
                        defined_derived_outputs
                    ),
                    issues=(
                        issues
                    ),
                )
            )


            _validate_reference(
                reference=(
                    action.target
                ),
                requirement_id=(
                    requirement_id
                ),
                step_id=(
                    step_id
                ),
                base_columns=(
                    base_columns
                ),
                bridge_dataset_ids=(
                    bridge_datasets
                ),
                all_derived_outputs=(
                    all_derived_outputs
                ),
                defined_derived_outputs=(
                    defined_derived_outputs
                ),
                issues=(
                    issues
                ),
            )


            # =================================================
            # BASE DATE TYPE
            #
            # If the date reference is a real planner-visible
            # base column, its analytical type must be temporal.
            #
            # Derived temporal expressions are not typed yet
            # and are therefore not rejected here.
            # =================================================

            if (
                date_valid
                and action.date
                in analytical_catalog
            ):

                date_type = (
                    analytical_catalog[
                        action.date
                    ]
                    .get(
                        "analytical_type"
                    )
                )


                if (
                    date_type
                    != "temporal"
                ):

                    _issue(
                        issues,
                        code=(
                            "temporal_column_required"
                        ),
                        requirement_id=(
                            requirement_id
                        ),
                        step_id=(
                            step_id
                        ),
                        message=(
                            "analyze_time_series.date must "
                            "reference a temporal analytical "
                            "column: "
                            f"{action.date}"
                        ),
                    )


    # ========================================================
    # FAMILY ANCHOR
    # ========================================================

    expected_anchor_tools = (
        FAMILY_ANCHOR_TOOLS[
            plan.family
        ]
    )


    if not (
        used_action_names
        & expected_anchor_tools
    ):

        _issue(
            issues,
            code=(
                "missing_family_anchor"
            ),
            requirement_id=(
                requirement_id
            ),
            step_id=None,
            message=(
                "Analytical family does not contain an "
                "appropriate analytical action. "
                f"Family={plan.family}, expected one of "
                f"{sorted(expected_anchor_tools)}."
            ),
        )


# ============================================================
# PUBLIC VALIDATOR
# ============================================================

def validate_analytical_planner_candidate(
    *,
    candidate: AnalyticalPlannerCandidate,
    planner_input: AnalyticalPlannerInput,
) -> AnalyticalPlannerValidationResult:
    """
    Validate one model-produced analytical planner candidate
    against the exact trusted AnalyticalPlannerInput that was
    supplied to the model.

    This is the main deterministic guardrail between planner
    generation and analytical execution.
    """

    input_requirements = (
        _index_input_requirements(
            planner_input
        )
    )


    input_requirement_ids = set(
        input_requirements
    )


    candidate_requirement_ids = {
        plan.requirement_id

        for plan
        in candidate.plans
    }


    issues: list[
        AnalyticalPlannerValidationIssue
    ] = []


    # ========================================================
    # UNKNOWN REQUIREMENTS
    # ========================================================

    unknown_requirements = (
        candidate_requirement_ids
        - input_requirement_ids
    )


    for requirement_id in sorted(
        unknown_requirements
    ):

        _issue(
            issues,
            code=(
                "unknown_requirement"
            ),
            requirement_id=(
                requirement_id
            ),
            step_id=None,
            message=(
                "Planner produced a requirement that was "
                "not present in AnalyticalPlannerInput: "
                f"{requirement_id}"
            ),
        )


    # ========================================================
    # MISSING REQUIREMENTS
    # ========================================================

    missing_requirements = (
        input_requirement_ids
        - candidate_requirement_ids
    )


    for requirement_id in sorted(
        missing_requirements
    ):

        _issue(
            issues,
            code=(
                "missing_requirement"
            ),
            requirement_id=(
                requirement_id
            ),
            step_id=None,
            message=(
                "Planner failed to produce a plan for input "
                "requirement: "
                f"{requirement_id}"
            ),
        )


    # ========================================================
    # VALIDATE KNOWN PLANS
    # ========================================================

    validated_requirement_ids: list[
        str
    ] = []


    for plan in candidate.plans:

        requirement_id = (
            plan.requirement_id
        )


        if (
            requirement_id
            not in input_requirements
        ):
            continue


        issue_count_before = len(
            issues
        )


        _validate_requirement_plan(
            plan=plan,

            requirement=(
                input_requirements[
                    requirement_id
                ]
            ),

            issues=(
                issues
            ),
        )


        issue_count_after = len(
            issues
        )


        if (
            issue_count_after
            == issue_count_before
        ):

            validated_requirement_ids.append(
                requirement_id
            )


    return (
        AnalyticalPlannerValidationResult(
            validator_version=(
                ANALYTICAL_PLANNER_VALIDATOR_VERSION
            ),

            valid=(
                len(
                    issues
                )
                == 0
            ),

            validated_requirement_ids=(
                validated_requirement_ids
            ),

            issues=(
                issues
            ),
        )
    )


# ============================================================
# EXECUTION GUARD
# ============================================================

def require_valid_analytical_plan(
    *,
    candidate: AnalyticalPlannerCandidate,
    planner_input: AnalyticalPlannerInput,
) -> AnalyticalPlannerCandidate:
    """
    Final deterministic guard before an analytical planner
    candidate may proceed toward execution.

    Invalid plans are never silently repaired here.

    They are rejected so a future orchestration layer can:

    - ask the model to replan;
    - surface the error;
    - record an evaluation failure;
    - or safely stop execution.
    """

    result = (
        validate_analytical_planner_candidate(
            candidate=candidate,
            planner_input=planner_input,
        )
    )


    if not (
        result.valid
    ):

        issue_codes = [
            issue.code

            for issue
            in result.issues
        ]


        raise ValueError(
            "Analytical planner candidate failed "
            "deterministic validation. "
            f"Issues: {issue_codes}"
        )


    return candidate