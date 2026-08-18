from __future__ import annotations

from pydantic import (
    BaseModel,
    ConfigDict,
)

from app.planning.analytical_v1.dependency import (
    DatasetDependencyCandidate,
    DatasetDependencyGateResult,
    evaluate_dataset_dependencies,
)

from app.planning.analytical_v1.relationship_paths import (
    RelationshipTraversalStep,
    resolve_validated_relationship_plan,
)

from app.planning.analytical_v1.relationships import (
    RoutingRelationshipContext,
)


# ============================================================
# VERSION
# ============================================================

PLANNER_STRUCTURAL_HANDOFF_VERSION = (
    "planner_structural_handoff_v0.9"
)


# ============================================================
# ONE REQUIREMENT
# ============================================================

class PlannerRequirementStructuralContext(
    BaseModel
):
    """
    Deterministic structural information supplied to the
    analytical planner for one semantic analytical
    requirement.

    semantic_dataset_ids:
        Datasets selected semantically by the AI dependency
        extractor.

    bridge_dataset_ids:
        Additional datasets required only to connect the
        semantic datasets.

    materialization_dataset_ids:
        Complete set of datasets that may need to participate
        in structural materialization.

    relationship_ids:
        Validated relationships selected by Python.

    traversal_steps:
        Oriented relationship steps including join keys.

    The analytical planner does not invent any of these.
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

    relationship_ids: list[
        str
    ]

    traversal_steps: list[
        RelationshipTraversalStep
    ]


# ============================================================
# HANDOFF
# ============================================================

class PlannerStructuralHandoff(
    BaseModel
):
    """
    Structural contract passed to the analytical planner.

    ready_for_planning=False means at least one semantic
    requirement is structurally impossible and the analytical
    planner must not be invoked for the complete request.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    handoff_version: str

    ready_for_planning: bool

    blocking_requirements: list[
        str
    ]

    routing_override_reason: (
        str
        | None
    )

    requirements: list[
        PlannerRequirementStructuralContext
    ]


# ============================================================
# BUILD HANDOFF
# ============================================================

def build_planner_structural_handoff(
    *,
    candidate: DatasetDependencyCandidate,
    context: RoutingRelationshipContext,
) -> PlannerStructuralHandoff:
    """
    Convert semantic dependency extraction into deterministic
    structural context for the analytical planner.

    Processing
    ----------

    1. Validate structural feasibility using the existing
       dependency gate.

    2. If any requirement is blocked:
           ready_for_planning = False

       No partial planning is allowed yet.

    3. For each executable requirement:
       resolve exact validated relationship paths.

    4. Return only deterministic structural evidence.

    IMPORTANT
    ---------

    This function does NOT:

    - choose statistical methods;
    - choose analytical metrics;
    - choose aggregation functions;
    - invent joins;
    - infer missing relationships;
    - execute transformations.
    """

    gate_result = (
        evaluate_dataset_dependencies(
            candidate=candidate,
            context=context,
        )
    )


    # ========================================================
    # BLOCKED REQUEST
    #
    # For v0.9 the request is atomic:
    #
    # if one requirement is structurally blocked, the complete
    # analytical planner is not invoked.
    #
    # We still return structural context for requirements that
    # are individually executable so diagnostics remain useful.
    # ========================================================

    requirement_contexts: list[
        PlannerRequirementStructuralContext
    ] = []


    gate_by_requirement_id = {
        requirement.requirement_id:
            requirement

        for requirement
        in gate_result.requirements
    }


    for requirement in candidate.requirements:

        gate_requirement = (
            gate_by_requirement_id[
                requirement.requirement_id
            ]
        )


        if not (
            gate_requirement.executable
        ):
            continue


        resolution = (
            resolve_validated_relationship_plan(
                context=context,

                required_dataset_ids=(
                    requirement.dataset_ids
                ),
            )
        )


        if not resolution.connected:

            raise RuntimeError(
                "Dependency gate marked requirement as "
                "executable but relationship resolver could "
                "not connect it: "
                f"{requirement.requirement_id}"
            )


        traversal_steps: list[
            RelationshipTraversalStep
        ] = []


        seen_relationship_steps: set[
            tuple[
                str,
                str,
                str,
            ]
        ] = set()


        for path in resolution.paths:

            for step in path.steps:

                step_key = (
                    step.relationship_id,
                    step.from_dataset_id,
                    step.to_dataset_id,
                )


                if (
                    step_key
                    in seen_relationship_steps
                ):
                    continue


                seen_relationship_steps.add(
                    step_key
                )


                traversal_steps.append(
                    step
                )


        requirement_contexts.append(
            PlannerRequirementStructuralContext(
                requirement_id=(
                    requirement.requirement_id
                ),

                semantic_dataset_ids=list(
                    requirement.dataset_ids
                ),

                bridge_dataset_ids=list(
                    resolution.bridge_dataset_ids
                ),

                materialization_dataset_ids=list(
                    resolution.all_dataset_ids
                ),

                relationship_ids=list(
                    resolution.relationship_ids
                ),

                traversal_steps=(
                    traversal_steps
                ),
            )
        )


    return PlannerStructuralHandoff(
        handoff_version=(
            PLANNER_STRUCTURAL_HANDOFF_VERSION
        ),

        ready_for_planning=(
            gate_result.executable
        ),

        blocking_requirements=list(
            gate_result.blocking_requirements
        ),

        routing_override_reason=(
            gate_result.routing_override_reason
        ),

        requirements=(
            requirement_contexts
        ),
    )


# ============================================================
# HELPER
# ============================================================

def structural_handoff_from_gate(
    *,
    candidate: DatasetDependencyCandidate,
    context: RoutingRelationshipContext,
    gate_result: DatasetDependencyGateResult,
) -> PlannerStructuralHandoff:
    """
    Variant for callers that already evaluated the dependency
    gate.

    The supplied gate result is checked against a fresh
    deterministic evaluation before creating the handoff.

    This prevents a stale or externally modified gate result
    from becoming trusted planner context.
    """

    verified_gate = (
        evaluate_dataset_dependencies(
            candidate=candidate,
            context=context,
        )
    )


    if (
        verified_gate.model_dump(
            mode="json",
        )
        != gate_result.model_dump(
            mode="json",
        )
    ):
        raise ValueError(
            "Provided dependency gate result does not match "
            "the deterministic structural evaluation."
        )


    return (
        build_planner_structural_handoff(
            candidate=candidate,
            context=context,
        )
    )