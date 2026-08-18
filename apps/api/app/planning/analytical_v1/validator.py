from __future__ import annotations

from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
)

from app.planning.analytical_v1.contract import (
    AnalyticalPlannerCandidate,
    BuildEntityViewToolCall,
    CompareGroupsToolCall,
    DeriveMetricToolCall,
)

from app.planning.analytical_v1.input import (
    AnalyticalPlannerInput,
    AnalyticalPlannerRequirementInput,
)

from app.planning.analytical_v1.validator_base import (
    validate_analytical_planner_candidate as validate_v0_9,
)


# ============================================================
# VERSION
# ============================================================

ANALYTICAL_PLANNER_VALIDATOR_VERSION = (
    "analytical_planner_validator_v0.9.1"
)


# ============================================================
# ISSUE CODES
#
# v0.9 codes are preserved verbatim.
#
# v0.9.1 adds semantic consistency checks discovered from the
# first real Ministral/Qwen baseline.
# ============================================================

ValidationIssueCode = Literal[
    # --------------------------------------------------------
    # v0.9
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # v0.9.1
    # --------------------------------------------------------

    "compare_groups_requires_distinct_references",
    "invalid_compare_target_type",
    "invalid_compare_group_type",
    "entity_target_grain_mismatch",
]


# ============================================================
# TYPE FAMILIES
# ============================================================

QUANTITATIVE_TYPES = {
    "quantitative",
    "numeric",
    "continuous",
    "discrete",
}


GROUPING_TYPES = {
    "categorical",
    "nominal",
    "ordinal",
    "temporal",
    "identifier",
    "boolean",
}


# ============================================================
# ISSUE MODEL
# ============================================================

class AnalyticalPlannerValidationIssueV091(
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
# RESULT MODEL
# ============================================================

class AnalyticalPlannerValidationResultV091(
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
        AnalyticalPlannerValidationIssueV091
    ]


# ============================================================
# HELPERS
# ============================================================

def _issue(
    issues: list[
        AnalyticalPlannerValidationIssueV091
    ],
    *,
    code: ValidationIssueCode,
    requirement_id: str | None,
    step_id: str | None,
    message: str,
) -> None:

    issues.append(
        AnalyticalPlannerValidationIssueV091(
            code=code,
            requirement_id=requirement_id,
            step_id=step_id,
            message=message,
        )
    )


def _requirement_index(
    planner_input: AnalyticalPlannerInput,
) -> dict[
    str,
    AnalyticalPlannerRequirementInput,
]:

    return {
        requirement.requirement_id:
            requirement

        for requirement
        in planner_input.requirements
    }


def _analytical_type_catalog(
    requirement: AnalyticalPlannerRequirementInput,
) -> dict[
    str,
    str,
]:

    result: dict[
        str,
        str,
    ] = {}


    for column in (
        requirement.analytical_columns
    ):

        qualified_name = (
            column.get(
                "qualified_name"
            )
        )


        analytical_type = (
            column.get(
                "analytical_type"
            )
        )


        if (
            qualified_name is None
            or analytical_type is None
        ):
            continue


        result[
            str(
                qualified_name
            )
        ] = (
            str(
                analytical_type
            )
            .strip()
            .lower()
        )


    return result


def _semantic_entity_references(
    requirement: AnalyticalPlannerRequirementInput,
) -> set[str]:

    entities: set[
        str
    ] = set()


    for dataset in requirement.datasets:

        if (
            dataset.role
            != "semantic"
        ):
            continue


        for column_name in (
            dataset.entity_columns
        ):

            entities.add(
                (
                    f"{dataset.dataset_id}"
                    f".{column_name}"
                )
            )


    return entities


def _entity_grain_aliases(
    entity_reference: str,
) -> set[str]:
    """
    Convert an entity reference into valid entity-grain names.

    Example:

        sales.customer_id

    becomes:

        customer_id
        customer
    """

    column_name = (
        entity_reference
        .split(
            ".",
            1,
        )[
            -1
        ]
    )


    aliases = {
        column_name,
    }


    if (
        column_name.endswith(
            "_id"
        )
        and len(
            column_name
        )
        > 3
    ):

        aliases.add(
            column_name[
                :-3
            ]
        )


    return aliases


# ============================================================
# COMPARE GROUPS SEMANTICS
# ============================================================

def _validate_compare_groups_semantics(
    *,
    plan,
    requirement: AnalyticalPlannerRequirementInput,
    issues: list[
        AnalyticalPlannerValidationIssueV091
    ],
) -> None:

    type_catalog = (
        _analytical_type_catalog(
            requirement
        )
    )


    defined_derived_outputs: set[
        str
    ] = set()


    for step in plan.steps:

        action = (
            step.action
        )


        # ====================================================
        # DERIVED METRICS BECOME AVAILABLE AFTER DEFINITION
        # ====================================================

        if isinstance(
            action,
            DeriveMetricToolCall,
        ):

            defined_derived_outputs.add(
                action.output
            )


            continue


        if not isinstance(
            action,
            CompareGroupsToolCall,
        ):

            continue


        # ====================================================
        # TARGET AND GROUP CANNOT BE IDENTICAL
        #
        # Regression:
        #
        # target   = marketing.channel
        # group_by = marketing.channel
        # ====================================================

        if (
            action.target
            == action.group_by
        ):

            _issue(
                issues,
                code=(
                    "compare_groups_requires_distinct_references"
                ),
                requirement_id=(
                    plan.requirement_id
                ),
                step_id=(
                    step.step_id
                ),
                message=(
                    "compare_groups requires a target and "
                    "group_by that represent distinct "
                    "analytical references. Received the "
                    "same reference for both: "
                    f"{action.target}"
                ),
            )


        # ====================================================
        # TARGET TYPE
        #
        # A base target must be quantitative.
        #
        # A derived target is currently accepted because
        # derived metrics do not yet carry explicit type
        # metadata in the planner contract.
        # ====================================================

        target_type = (
            type_catalog.get(
                action.target
            )
        )


        if (
            target_type is not None
            and target_type
            not in QUANTITATIVE_TYPES
        ):

            _issue(
                issues,
                code=(
                    "invalid_compare_target_type"
                ),
                requirement_id=(
                    plan.requirement_id
                ),
                step_id=(
                    step.step_id
                ),
                message=(
                    "compare_groups.target must reference "
                    "a quantitative base column or an "
                    "already-defined derived metric. "
                    f"Reference={action.target}, "
                    f"type={target_type}."
                ),
            )


        # ====================================================
        # GROUP TYPE
        #
        # Base grouping columns should describe groups:
        #
        # categorical / temporal / identifier / ordinal...
        #
        # Derived grouping references remain conservatively
        # allowed until derived metrics become typed.
        # ====================================================

        group_type = (
            type_catalog.get(
                action.group_by
            )
        )


        if (
            group_type is not None
            and group_type
            not in GROUPING_TYPES
        ):

            _issue(
                issues,
                code=(
                    "invalid_compare_group_type"
                ),
                requirement_id=(
                    plan.requirement_id
                ),
                step_id=(
                    step.step_id
                ),
                message=(
                    "compare_groups.group_by must reference "
                    "a grouping-compatible base column. "
                    f"Reference={action.group_by}, "
                    f"type={group_type}."
                ),
            )


# ============================================================
# ENTITY GRAIN SEMANTICS
# ============================================================

def _validate_entity_grain_semantics(
    *,
    plan,
    requirement: AnalyticalPlannerRequirementInput,
    issues: list[
        AnalyticalPlannerValidationIssueV091
    ],
) -> None:
    """
    If an entity-level plan explicitly builds an entity view,
    target_grain must represent that entity.

    Example:

        build_entity_view(customer_id)

    should lead to:

        target_grain = customer
        or
        target_grain = customer_id

    not:

        target_grain = customer_order
    """

    if (
        plan.family
        != "entity_outlier"
    ):
        return


    semantic_entities = (
        _semantic_entity_references(
            requirement
        )
    )


    allowed_entity_grains: set[
        str
    ] = set()


    for step in plan.steps:

        action = (
            step.action
        )


        if not isinstance(
            action,
            BuildEntityViewToolCall,
        ):
            continue


        if (
            action.entity
            not in semantic_entities
        ):
            # v0.9 already reports invalid_entity_reference.
            continue


        allowed_entity_grains.update(
            _entity_grain_aliases(
                action.entity
            )
        )


    # ========================================================
    # No valid entity view:
    # v0.9 already owns that diagnostic.
    # ========================================================

    if not allowed_entity_grains:
        return


    if (
        plan.target_grain
        not in allowed_entity_grains
    ):

        _issue(
            issues,
            code=(
                "entity_target_grain_mismatch"
            ),
            requirement_id=(
                plan.requirement_id
            ),
            step_id=None,
            message=(
                "Entity-level analytical plan declares a "
                "target_grain inconsistent with the entity "
                "view it builds. "
                f"target_grain={plan.target_grain}, "
                f"allowed={sorted(allowed_entity_grains)}."
            ),
        )


# ============================================================
# PUBLIC VALIDATOR
# ============================================================

def validate_analytical_planner_candidate(
    *,
    candidate: AnalyticalPlannerCandidate,
    planner_input: AnalyticalPlannerInput,
) -> AnalyticalPlannerValidationResultV091:
    """
    v0.9.1 validation.

    Layer 1:
        execute the complete historical v0.9 validator.

    Layer 2:
        add targeted semantic consistency checks discovered
        from the first real model baseline.

    The historical v0.9 implementation is never modified.
    """

    base_result = (
        validate_v0_9(
            candidate=candidate,
            planner_input=planner_input,
        )
    )


    issues: list[
        AnalyticalPlannerValidationIssueV091
    ] = [
        AnalyticalPlannerValidationIssueV091(
            code=issue.code,
            requirement_id=(
                issue.requirement_id
            ),
            step_id=(
                issue.step_id
            ),
            message=(
                issue.message
            ),
        )

        for issue
        in base_result.issues
    ]


    requirements = (
        _requirement_index(
            planner_input
        )
    )


    # ========================================================
    # SEMANTIC CONSISTENCY CHECKS
    # ========================================================

    for plan in candidate.plans:

        requirement = (
            requirements.get(
                plan.requirement_id
            )
        )


        if requirement is None:
            # v0.9 already reports unknown_requirement.
            continue


        _validate_compare_groups_semantics(
            plan=plan,
            requirement=requirement,
            issues=issues,
        )


        _validate_entity_grain_semantics(
            plan=plan,
            requirement=requirement,
            issues=issues,
        )


    # ========================================================
    # REQUIREMENTS VALIDATED BY BOTH LAYERS
    # ========================================================

    invalid_requirement_ids = {
        issue.requirement_id

        for issue
        in issues

        if (
            issue.requirement_id
            is not None
        )
    }


    validated_requirement_ids = [
        requirement_id

        for requirement_id
        in base_result.validated_requirement_ids

        if (
            requirement_id
            not in invalid_requirement_ids
        )
    ]


    return (
        AnalyticalPlannerValidationResultV091(
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

    result = (
        validate_analytical_planner_candidate(
            candidate=candidate,
            planner_input=planner_input,
        )
    )


    if not result.valid:

        issue_codes = [
            issue.code

            for issue
            in result.issues
        ]


        raise ValueError(
            "Analytical planner candidate failed "
            "deterministic validation v0.9.1. "
            f"Issues: {issue_codes}"
        )


    return candidate