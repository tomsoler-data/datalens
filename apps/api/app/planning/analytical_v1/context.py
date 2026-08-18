from __future__ import annotations

from pydantic import (
    BaseModel,
    ConfigDict,
)

from app.planning.analytical_v1.dependency import (
    DatasetDependencyCandidate,
)

from app.planning.analytical_v1.structural_handoff import (
    PlannerRequirementStructuralContext,
    PlannerStructuralHandoff,
    build_planner_structural_handoff,
)

from app.planning.analytical_v1.relationship_paths import (
    RelationshipTraversalStep,
)

from app.planning.analytical_v1.relationships import (
    RoutingRelationshipContext,
)


# ============================================================
# VERSION
# ============================================================

ANALYTICAL_PLANNER_CONTEXT_VERSION = (
    "analytical_planner_context_v0.9"
)


# ============================================================
# STRUCTURAL TOOLS
#
# These operations are controlled by the deterministic
# structural layer and must not be independently selected by
# the analytical planner.
# ============================================================

STRUCTURAL_TOOLS = {
    "join_datasets",
}


# ============================================================
# QUALIFIED ANALYTICAL COLUMN
# ============================================================

class PlannerAnalyticalColumn(
    BaseModel
):
    """
    Column that the analytical planner is allowed to reason
    about as an analytical variable.

    Only columns belonging to semantic datasets selected by
    the dependency extractor are exposed here.

    Bridge-only dataset columns are deliberately excluded.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    qualified_name: str

    dataset_id: str

    column_name: str

    analytical_type: str

    semantic_role: (
        str
        | None
    )


# ============================================================
# STRUCTURAL KEY
# ============================================================

class PlannerStructuralKey(
    BaseModel
):
    """
    Structural column used by a deterministic relationship
    traversal.

    This does NOT make the column an analytical variable.

    Example:

        customers.customer_id

    may be necessary to connect sales to support while
    customers.segment remains invisible to the analytical
    planner.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    qualified_name: str

    dataset_id: str

    column_name: str


# ============================================================
# ONE REQUIREMENT CONTEXT
# ============================================================

class AnalyticalPlannerRequirementContext(
    BaseModel
):
    """
    Trusted input for one analytical requirement.

    semantic_dataset_ids:
        Datasets semantically required by the user request.

    bridge_dataset_ids:
        Datasets used only to satisfy structural connectivity.

    materialization_dataset_ids:
        Complete deterministic structural footprint.

    analytical_columns:
        Variables the planner may actually use analytically.

    structural_keys:
        Join/alignment columns needed by Python.

    relationship_ids / traversal_steps:
        Structural plan already selected and validated by
        Python.

    allowed_analytical_tools:
        Tools the planner may select after deterministic
        structural materialization.

        Structural tools such as join_datasets are removed.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    requirement_id: str

    semantic_dataset_ids: list[
        str
    ]

    bridge_dataset_ids: list[
        str
    ]

    materialization_dataset_ids: list[
        str
    ]

    analytical_columns: list[
        PlannerAnalyticalColumn
    ]

    structural_keys: list[
        PlannerStructuralKey
    ]

    relationship_ids: list[
        str
    ]

    traversal_steps: list[
        RelationshipTraversalStep
    ]

    allowed_analytical_tools: list[
        str
    ]


# ============================================================
# COMPLETE PLANNER CONTEXT
# ============================================================

class AnalyticalPlannerContext(
    BaseModel
):
    """
    Complete trusted context supplied to the future
    analytical planner.

    ready_for_planning=False means the AI planner MUST NOT be
    invoked for the complete request.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    context_version: str

    ready_for_planning: bool

    blocking_requirements: list[
        str
    ]

    routing_override_reason: (
        str
        | None
    )

    requirements: list[
        AnalyticalPlannerRequirementContext
    ]


# ============================================================
# HELPERS
# ============================================================

def _qualified_name(
    *,
    dataset_id: str,
    column_name: str,
) -> str:

    return (
        f"{dataset_id}.{column_name}"
    )


def _datasets_by_id(
    context: RoutingRelationshipContext,
) -> dict:

    return {
        dataset.dataset_id:
            dataset

        for dataset
        in context.datasets
    }


def _allowed_analytical_tools(
    context: RoutingRelationshipContext,
) -> list[str]:

    result: list[
        str
    ] = []


    seen: set[
        str
    ] = set()


    for tool in context.available_tools:

        normalized = (
            tool
            .strip()
        )


        if (
            normalized
            in STRUCTURAL_TOOLS
        ):
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


# ============================================================
# ANALYTICAL COLUMN CATALOG
# ============================================================

def _build_analytical_columns(
    *,
    structural_requirement: PlannerRequirementStructuralContext,
    context: RoutingRelationshipContext,
) -> list[
    PlannerAnalyticalColumn
]:
    """
    Expose columns only from SEMANTIC datasets.

    Bridge datasets are structurally necessary but do not
    automatically become analytical sources.
    """

    datasets = (
        _datasets_by_id(
            context
        )
    )


    columns: list[
        PlannerAnalyticalColumn
    ] = []


    seen: set[
        str
    ] = set()


    for dataset_id in (
        structural_requirement
        .semantic_dataset_ids
    ):

        dataset = (
            datasets[
                dataset_id
            ]
        )


        for column in dataset.columns:

            qualified = (
                _qualified_name(
                    dataset_id=(
                        dataset_id
                    ),

                    column_name=(
                        column.name
                    ),
                )
            )


            if qualified in seen:
                continue


            seen.add(
                qualified
            )


            columns.append(
                PlannerAnalyticalColumn(
                    qualified_name=(
                        qualified
                    ),

                    dataset_id=(
                        dataset_id
                    ),

                    column_name=(
                        column.name
                    ),

                    analytical_type=(
                        column.analytical_type
                    ),

                    semantic_role=(
                        column.semantic_role
                    ),
                )
            )


    return columns


# ============================================================
# STRUCTURAL KEY CATALOG
# ============================================================

def _build_structural_keys(
    *,
    structural_requirement: PlannerRequirementStructuralContext,
) -> list[
    PlannerStructuralKey
]:
    """
    Extract exact keys used by deterministic traversal steps.

    Both sides of every traversal are included.

    Duplicate qualified keys are collapsed.
    """

    keys: list[
        PlannerStructuralKey
    ] = []


    seen: set[
        str
    ] = set()


    def add_key(
        *,
        dataset_id: str,
        column_name: str,
    ) -> None:

        qualified = (
            _qualified_name(
                dataset_id=(
                    dataset_id
                ),

                column_name=(
                    column_name
                ),
            )
        )


        if qualified in seen:
            return


        seen.add(
            qualified
        )


        keys.append(
            PlannerStructuralKey(
                qualified_name=(
                    qualified
                ),

                dataset_id=(
                    dataset_id
                ),

                column_name=(
                    column_name
                ),
            )
        )


    for step in (
        structural_requirement
        .traversal_steps
    ):

        for column_name in (
            step.from_keys
        ):

            add_key(
                dataset_id=(
                    step.from_dataset_id
                ),

                column_name=(
                    column_name
                ),
            )


        for column_name in (
            step.to_keys
        ):

            add_key(
                dataset_id=(
                    step.to_dataset_id
                ),

                column_name=(
                    column_name
                ),
            )


    return keys


# ============================================================
# ONE REQUIREMENT
# ============================================================

def _build_requirement_context(
    *,
    structural_requirement: PlannerRequirementStructuralContext,
    context: RoutingRelationshipContext,
) -> AnalyticalPlannerRequirementContext:

    return (
        AnalyticalPlannerRequirementContext(
            requirement_id=(
                structural_requirement
                .requirement_id
            ),

            semantic_dataset_ids=list(
                structural_requirement
                .semantic_dataset_ids
            ),

            bridge_dataset_ids=list(
                structural_requirement
                .bridge_dataset_ids
            ),

            materialization_dataset_ids=list(
                structural_requirement
                .materialization_dataset_ids
            ),

            analytical_columns=(
                _build_analytical_columns(
                    structural_requirement=(
                        structural_requirement
                    ),

                    context=context,
                )
            ),

            structural_keys=(
                _build_structural_keys(
                    structural_requirement=(
                        structural_requirement
                    ),
                )
            ),

            relationship_ids=list(
                structural_requirement
                .relationship_ids
            ),

            traversal_steps=list(
                structural_requirement
                .traversal_steps
            ),

            allowed_analytical_tools=(
                _allowed_analytical_tools(
                    context
                )
            ),
        )
    )


# ============================================================
# PUBLIC BUILDER
# ============================================================

def build_analytical_planner_context(
    *,
    candidate: DatasetDependencyCandidate,
    context: RoutingRelationshipContext,
) -> AnalyticalPlannerContext:
    """
    Build the complete trusted input for the analytical
    planner.

    The builder first executes the deterministic structural
    handoff.

    Therefore the planner context cannot bypass:

    - dependency validation;
    - relationship validation;
    - bridge resolution;
    - structural blocking.
    """

    structural_handoff = (
        build_planner_structural_handoff(
            candidate=candidate,
            context=context,
        )
    )


    requirements = [
        _build_requirement_context(
            structural_requirement=(
                requirement
            ),

            context=context,
        )

        for requirement
        in structural_handoff.requirements
    ]


    return (
        AnalyticalPlannerContext(
            context_version=(
                ANALYTICAL_PLANNER_CONTEXT_VERSION
            ),

            ready_for_planning=(
                structural_handoff
                .ready_for_planning
            ),

            blocking_requirements=list(
                structural_handoff
                .blocking_requirements
            ),

            routing_override_reason=(
                structural_handoff
                .routing_override_reason
            ),

            requirements=(
                requirements
            ),
        )
    )


# ============================================================
# READY GUARD
# ============================================================

def require_ready_planner_context(
    planner_context: AnalyticalPlannerContext,
) -> AnalyticalPlannerContext:
    """
    Final deterministic guard before invoking an AI planner.

    A caller must pass through this function before model
    inference.

    This prevents a blocked structural context from silently
    reaching the planner.
    """

    if not (
        planner_context.ready_for_planning
    ):

        raise ValueError(
            "Analytical planner context is blocked. "
            "Blocking requirement(s): "
            f"{planner_context.blocking_requirements}. "
            "Routing override: "
            f"{planner_context.routing_override_reason}"
        )


    return planner_context