from __future__ import annotations

from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)

from app.planning.analytical_v1.context import (
    AnalyticalPlannerContext,
    require_ready_planner_context,
)

from app.planning.analytical_v1.relationships import (
    RoutingRelationshipContext,
)


# ============================================================
# VERSION
# ============================================================

ANALYTICAL_PLANNER_INPUT_VERSION = (
    "analytical_planner_input_v0.9"
)


# ============================================================
# DATASET ROLE
# ============================================================

PlannerDatasetRole = Literal[
    "semantic",
    "bridge",
]


# ============================================================
# DATASET SCOPE
# ============================================================

class PlannerDatasetScope(
    BaseModel
):
    """
    Dataset metadata explicitly available to the analytical
    planner.

    role = semantic
        The dataset contains analytical information required
        by the user request.

    role = bridge
        The dataset exists only because Python selected it as
        part of a validated structural connection path.

        Bridge columns do NOT automatically become analytical
        variables.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    dataset_id: str = Field(
        min_length=1,
    )

    role: PlannerDatasetRole

    grain: str = Field(
        min_length=1,
    )

    entity_columns: list[
        str
    ]


# ============================================================
# ONE REQUIREMENT INPUT
# ============================================================

class AnalyticalPlannerRequirementInput(
    BaseModel
):
    """
    Complete trusted planner input for one analytical
    requirement.

    The planner receives:

    - the requirement identifier;
    - semantic and bridge dataset metadata;
    - analytical columns it may use;
    - structural keys and traversal path selected by Python;
    - analytical tools it may select.

    It does not receive permission to invent datasets,
    relationships, joins, or structural columns.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    requirement_id: str

    datasets: list[
        PlannerDatasetScope
    ]

    analytical_columns: list[
        dict
    ]

    structural_keys: list[
        dict
    ]

    relationship_ids: list[
        str
    ]

    traversal_steps: list[
        dict
    ]

    allowed_analytical_tools: list[
        str
    ]


# ============================================================
# COMPLETE INPUT
# ============================================================

class AnalyticalPlannerInput(
    BaseModel
):
    """
    Model-visible envelope for the future analytical planner.

    This object must only be created from an already validated
    and ready AnalyticalPlannerContext.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    input_version: str

    user_request: str = Field(
        min_length=1,
    )

    requirements: list[
        AnalyticalPlannerRequirementInput
    ] = Field(
        min_length=1,
    )


# ============================================================
# HELPERS
# ============================================================

def _datasets_by_id(
    context: RoutingRelationshipContext,
) -> dict:

    return {
        dataset.dataset_id:
            dataset

        for dataset
        in context.datasets
    }


def _build_dataset_scopes(
    *,
    semantic_dataset_ids: list[str],
    bridge_dataset_ids: list[str],
    context: RoutingRelationshipContext,
) -> list[
    PlannerDatasetScope
]:

    datasets = (
        _datasets_by_id(
            context
        )
    )


    scopes: list[
        PlannerDatasetScope
    ] = []


    seen: set[
        str
    ] = set()


    # ========================================================
    # SEMANTIC DATASETS FIRST
    # ========================================================

    for dataset_id in semantic_dataset_ids:

        if dataset_id in seen:
            continue


        seen.add(
            dataset_id
        )


        dataset = (
            datasets[
                dataset_id
            ]
        )


        scopes.append(
            PlannerDatasetScope(
                dataset_id=(
                    dataset_id
                ),

                role=(
                    "semantic"
                ),

                grain=(
                    dataset.grain
                ),

                entity_columns=list(
                    dataset.entity_columns
                ),
            )
        )


    # ========================================================
    # BRIDGE DATASETS SECOND
    # ========================================================

    for dataset_id in bridge_dataset_ids:

        if dataset_id in seen:
            continue


        seen.add(
            dataset_id
        )


        dataset = (
            datasets[
                dataset_id
            ]
        )


        scopes.append(
            PlannerDatasetScope(
                dataset_id=(
                    dataset_id
                ),

                role=(
                    "bridge"
                ),

                grain=(
                    dataset.grain
                ),

                entity_columns=list(
                    dataset.entity_columns
                ),
            )
        )


    return scopes


# ============================================================
# PUBLIC BUILDER
# ============================================================

def build_analytical_planner_input(
    *,
    user_request: str,
    planner_context: AnalyticalPlannerContext,
    structural_context: RoutingRelationshipContext,
) -> AnalyticalPlannerInput:
    """
    Build the exact payload that may later be serialized for
    an analytical planning model.

    SECURITY / TRUST BOUNDARY
    -------------------------

    The planner context must already be structurally ready.

    Therefore a blocked request cannot be converted into model
    input.
    """

    normalized_request = (
        user_request.strip()
    )


    if not normalized_request:

        raise ValueError(
            "user_request must not be empty."
        )


    verified_context = (
        require_ready_planner_context(
            planner_context
        )
    )


    known_dataset_ids = {
        dataset.dataset_id

        for dataset
        in structural_context.datasets
    }


    requirement_inputs: list[
        AnalyticalPlannerRequirementInput
    ] = []


    for requirement in (
        verified_context.requirements
    ):

        referenced_dataset_ids = {
            *requirement.semantic_dataset_ids,
            *requirement.bridge_dataset_ids,
        }


        unknown = (
            referenced_dataset_ids
            - known_dataset_ids
        )


        if unknown:

            raise ValueError(
                "Planner context references dataset(s) "
                "missing from structural context: "
                f"{sorted(unknown)}"
            )


        requirement_inputs.append(
            AnalyticalPlannerRequirementInput(
                requirement_id=(
                    requirement.requirement_id
                ),

                datasets=(
                    _build_dataset_scopes(
                        semantic_dataset_ids=(
                            requirement
                            .semantic_dataset_ids
                        ),

                        bridge_dataset_ids=(
                            requirement
                            .bridge_dataset_ids
                        ),

                        context=(
                            structural_context
                        ),
                    )
                ),

                analytical_columns=[
                    column.model_dump(
                        mode="json",
                    )

                    for column
                    in requirement.analytical_columns
                ],

                structural_keys=[
                    key.model_dump(
                        mode="json",
                    )

                    for key
                    in requirement.structural_keys
                ],

                relationship_ids=list(
                    requirement.relationship_ids
                ),

                traversal_steps=[
                    step.model_dump(
                        mode="json",
                    )

                    for step
                    in requirement.traversal_steps
                ],

                allowed_analytical_tools=list(
                    requirement.allowed_analytical_tools
                ),
            )
        )


    if not requirement_inputs:

        raise ValueError(
            "Ready planner context contains no "
            "analytical requirements."
        )


    return AnalyticalPlannerInput(
        input_version=(
            ANALYTICAL_PLANNER_INPUT_VERSION
        ),

        user_request=(
            normalized_request
        ),

        requirements=(
            requirement_inputs
        ),
    )