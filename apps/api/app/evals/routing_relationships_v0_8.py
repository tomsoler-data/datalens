from __future__ import annotations

from collections import deque
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

from app.evals.schemas import (
    DatasetContext,
)


# ============================================================
# VERSION
# ============================================================

ROUTING_RELATIONSHIP_CONTEXT_VERSION = (
    "routing_relationship_context_v0.8"
)


# ============================================================
# VOCABULARY
# ============================================================

RelationshipKind = Literal[
    "join",
    "temporal_alignment",
]


CrossDatasetFeasibility = Literal[
    "not_required",
    "supported",
    "missing_combination_capability",
    "missing_validated_relationship",
]


# ============================================================
# COMBINATION CAPABILITIES
# ============================================================

DEFAULT_COMBINATION_TOOLS = {
    "join_datasets",
}


# ============================================================
# DATASET RELATIONSHIP
# ============================================================

class DatasetRelationshipSpec(
    BaseModel
):
    """
    A relationship that has already been validated during
    DataLens preparation.

    IMPORTANT:

    The analytical router does NOT invent these relationships.

    They come from the preparation / combination / validation
    workflow.

    A relationship therefore represents structural evidence,
    not an LLM hypothesis.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    relationship_id: str = Field(
        min_length=1,
    )

    left_dataset_id: str = Field(
        min_length=1,
    )

    right_dataset_id: str = Field(
        min_length=1,
    )

    kind: RelationshipKind

    left_keys: list[
        str
    ] = Field(
        min_length=1,
    )

    right_keys: list[
        str
    ] = Field(
        min_length=1,
    )

    validated: Literal[
        True
    ] = True


    @model_validator(
        mode="after",
    )
    def validate_relationship(
        self,
    ) -> "DatasetRelationshipSpec":

        if (
            self.left_dataset_id
            == self.right_dataset_id
        ):
            raise ValueError(
                "A dataset relationship must connect "
                "two different datasets."
            )


        if (
            len(
                self.left_keys
            )
            != len(
                self.right_keys
            )
        ):
            raise ValueError(
                "left_keys and right_keys must have "
                "the same number of columns."
            )


        if (
            len(
                set(
                    self.left_keys
                )
            )
            != len(
                self.left_keys
            )
        ):
            raise ValueError(
                "left_keys must not contain duplicates."
            )


        if (
            len(
                set(
                    self.right_keys
                )
            )
            != len(
                self.right_keys
            )
        ):
            raise ValueError(
                "right_keys must not contain duplicates."
            )


        return self


# ============================================================
# ROUTING STRUCTURAL CONTEXT
# ============================================================

class RoutingRelationshipContext(
    BaseModel
):
    """
    Structural context made available to the routing layer.

    datasets:
        Schema/grain information already known by DataLens.

    relationships:
        Only relationships that were previously validated by
        the preparation workflow.

    available_tools:
        Analytical capabilities currently available.

    The LLM may reason about which datasets are required.

    Python remains responsible for determining whether those
    datasets can actually be combined.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    datasets: list[
        DatasetContext
    ] = Field(
        min_length=1,
    )

    relationships: list[
        DatasetRelationshipSpec
    ] = Field(
        default_factory=list,
    )

    available_tools: list[
        str
    ] = Field(
        min_length=1,
    )


    @model_validator(
        mode="after",
    )
    def validate_context(
        self,
    ) -> "RoutingRelationshipContext":

        # ====================================================
        # UNIQUE DATASETS
        # ====================================================

        dataset_ids = [
            dataset.dataset_id
            for dataset
            in self.datasets
        ]


        if (
            len(
                dataset_ids
            )
            != len(
                set(
                    dataset_ids
                )
            )
        ):
            raise ValueError(
                "dataset_id values must be unique."
            )


        datasets_by_id = {
            dataset.dataset_id:
                dataset

            for dataset
            in self.datasets
        }


        # ====================================================
        # UNIQUE RELATIONSHIP IDS
        # ====================================================

        relationship_ids = [
            relationship.relationship_id

            for relationship
            in self.relationships
        ]


        if (
            len(
                relationship_ids
            )
            != len(
                set(
                    relationship_ids
                )
            )
        ):
            raise ValueError(
                "relationship_id values must be unique."
            )


        # ====================================================
        # RELATIONSHIP REFERENCES
        # ====================================================

        for relationship in self.relationships:

            if (
                relationship.left_dataset_id
                not in datasets_by_id
            ):
                raise ValueError(
                    "Unknown left_dataset_id in relationship "
                    f"{relationship.relationship_id}: "
                    f"{relationship.left_dataset_id}"
                )


            if (
                relationship.right_dataset_id
                not in datasets_by_id
            ):
                raise ValueError(
                    "Unknown right_dataset_id in relationship "
                    f"{relationship.relationship_id}: "
                    f"{relationship.right_dataset_id}"
                )


            left_dataset = (
                datasets_by_id[
                    relationship.left_dataset_id
                ]
            )


            right_dataset = (
                datasets_by_id[
                    relationship.right_dataset_id
                ]
            )


            left_columns = {
                column.name
                for column
                in left_dataset.columns
            }


            right_columns = {
                column.name
                for column
                in right_dataset.columns
            }


            unknown_left_keys = (
                set(
                    relationship.left_keys
                )
                - left_columns
            )


            if unknown_left_keys:
                raise ValueError(
                    "Unknown left relationship key(s) "
                    f"in {relationship.relationship_id}: "
                    f"{sorted(unknown_left_keys)}"
                )


            unknown_right_keys = (
                set(
                    relationship.right_keys
                )
                - right_columns
            )


            if unknown_right_keys:
                raise ValueError(
                    "Unknown right relationship key(s) "
                    f"in {relationship.relationship_id}: "
                    f"{sorted(unknown_right_keys)}"
                )


        return self


# ============================================================
# HELPERS
# ============================================================

def _normalize_tool_name(
    value: str,
) -> str:
    return (
        value
        .strip()
        .lower()
    )


def _validate_required_dataset_ids(
    *,
    context: RoutingRelationshipContext,
    required_dataset_ids: list[str],
) -> list[str]:

    known_dataset_ids = {
        dataset.dataset_id
        for dataset
        in context.datasets
    }


    normalized_required: list[
        str
    ] = []


    seen: set[
        str
    ] = set()


    for dataset_id in required_dataset_ids:

        if (
            dataset_id
            not in known_dataset_ids
        ):
            raise ValueError(
                "Unknown required dataset: "
                f"{dataset_id}"
            )


        if dataset_id in seen:
            continue


        seen.add(
            dataset_id
        )


        normalized_required.append(
            dataset_id
        )


    if not normalized_required:
        raise ValueError(
            "At least one required dataset "
            "must be supplied."
        )


    return normalized_required


# ============================================================
# RELATIONSHIP GRAPH
# ============================================================

def build_relationship_graph(
    context: RoutingRelationshipContext,
) -> dict[
    str,
    set[str],
]:
    """
    Build an undirected graph from validated relationships.

    A → B means DataLens already has validated structural
    evidence allowing those datasets to participate in a
    combination path.

    Relationship direction is not used for graph reachability.
    Join execution details remain the responsibility of the
    downstream planner/executor.
    """

    graph: dict[
        str,
        set[str],
    ] = {
        dataset.dataset_id:
            set()

        for dataset
        in context.datasets
    }


    for relationship in context.relationships:

        graph[
            relationship.left_dataset_id
        ].add(
            relationship.right_dataset_id
        )


        graph[
            relationship.right_dataset_id
        ].add(
            relationship.left_dataset_id
        )


    return graph


# ============================================================
# RELATIONSHIP PATH
# ============================================================

def has_validated_relationship_path(
    *,
    context: RoutingRelationshipContext,
    required_dataset_ids: list[str],
) -> bool:
    """
    Return True when every required dataset belongs to the same
    connected component of the validated relationship graph.

    One dataset never requires a relationship path.
    """

    required = (
        _validate_required_dataset_ids(
            context=context,
            required_dataset_ids=(
                required_dataset_ids
            ),
        )
    )


    if (
        len(
            required
        )
        <= 1
    ):
        return True


    graph = (
        build_relationship_graph(
            context
        )
    )


    start = (
        required[
            0
        ]
    )


    visited = {
        start
    }


    queue = deque(
        [
            start,
        ]
    )


    while queue:

        current = (
            queue.popleft()
        )


        for neighbour in graph[
            current
        ]:

            if neighbour in visited:
                continue


            visited.add(
                neighbour
            )


            queue.append(
                neighbour
            )


    return all(
        dataset_id
        in visited

        for dataset_id
        in required
    )


# ============================================================
# COMBINATION CAPABILITY
# ============================================================

def has_combination_capability(
    *,
    context: RoutingRelationshipContext,
    combination_tools: set[str] | None = None,
) -> bool:

    supported_tools = (
        combination_tools
        if combination_tools
        is not None
        else DEFAULT_COMBINATION_TOOLS
    )


    normalized_supported = {
        _normalize_tool_name(
            tool
        )

        for tool
        in supported_tools
    }


    normalized_available = {
        _normalize_tool_name(
            tool
        )

        for tool
        in context.available_tools
    }


    return bool(
        normalized_supported
        & normalized_available
    )


# ============================================================
# CROSS-DATASET FEASIBILITY
# ============================================================

def evaluate_cross_dataset_feasibility(
    *,
    context: RoutingRelationshipContext,
    required_dataset_ids: list[str],
    combination_tools: set[str] | None = None,
) -> CrossDatasetFeasibility:
    """
    Deterministically evaluate whether a set of datasets can
    participate in one analytical result.

    IMPORTANT:

    This function does NOT decide which datasets are required.

    That semantic task may belong to the AI.

    Python only verifies structural feasibility after those
    dataset dependencies have been identified.
    """

    required = (
        _validate_required_dataset_ids(
            context=context,
            required_dataset_ids=(
                required_dataset_ids
            ),
        )
    )


    # ========================================================
    # ONE DATASET
    # ========================================================

    if (
        len(
            required
        )
        <= 1
    ):
        return "not_required"


    # ========================================================
    # CAPABILITY
    # ========================================================

    if not has_combination_capability(
        context=context,
        combination_tools=(
            combination_tools
        ),
    ):
        return (
            "missing_combination_capability"
        )


    # ========================================================
    # VALIDATED RELATIONSHIP PATH
    # ========================================================

    if not has_validated_relationship_path(
        context=context,
        required_dataset_ids=required,
    ):
        return (
            "missing_validated_relationship"
        )


    # ========================================================
    # SUPPORTED
    # ========================================================

    return "supported"