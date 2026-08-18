from __future__ import annotations

from collections import deque

from pydantic import (
    BaseModel,
    ConfigDict,
)

from app.evals.routing_relationships_v0_8 import (
    DatasetRelationshipSpec,
    RoutingRelationshipContext,
)


# ============================================================
# VERSION
# ============================================================

RELATIONSHIP_PATH_RESOLVER_VERSION = (
    "relationship_path_resolver_v0.8"
)


# ============================================================
# OUTPUT MODELS
# ============================================================

class RelationshipTraversalStep(
    BaseModel
):
    """
    One validated relationship traversal.

    The relationship may originally have been declared in the
    opposite direction.

    from_keys / to_keys are therefore oriented according to
    the traversal being performed.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    relationship_id: str

    kind: str

    from_dataset_id: str

    to_dataset_id: str

    from_keys: list[
        str
    ]

    to_keys: list[
        str
    ]


class DatasetRelationshipPath(
    BaseModel
):
    """
    One path from the anchor dataset to another required
    dataset.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    source_dataset_id: str

    target_dataset_id: str

    dataset_ids: list[
        str
    ]

    relationship_ids: list[
        str
    ]

    steps: list[
        RelationshipTraversalStep
    ]


class RelationshipResolution(
    BaseModel
):
    """
    Deterministic relationship resolution for one analytical
    requirement.

    required_dataset_ids:
        Semantic datasets requested by the AI extractor.

    bridge_dataset_ids:
        Additional datasets required only to connect the
        semantic datasets structurally.

    all_dataset_ids:
        Union of semantic + bridge datasets.

    relationship_ids:
        Validated relationships required by the resolved
        connection plan.

    unresolved_dataset_ids:
        Required datasets that cannot be reached from the
        anchor dataset.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    resolver_version: str

    required_dataset_ids: list[
        str
    ]

    bridge_dataset_ids: list[
        str
    ]

    all_dataset_ids: list[
        str
    ]

    relationship_ids: list[
        str
    ]

    paths: list[
        DatasetRelationshipPath
    ]

    unresolved_dataset_ids: list[
        str
    ]

    connected: bool


# ============================================================
# HELPERS
# ============================================================

def _deduplicate_preserve_order(
    values: list[str],
) -> list[str]:

    result: list[
        str
    ] = []


    seen: set[
        str
    ] = set()


    for value in values:

        normalized = (
            value.strip()
        )


        if normalized in seen:
            continue


        seen.add(
            normalized
        )


        result.append(
            normalized
        )


    return result


def _validate_required_datasets(
    *,
    context: RoutingRelationshipContext,
    required_dataset_ids: list[str],
) -> list[str]:

    required = (
        _deduplicate_preserve_order(
            required_dataset_ids
        )
    )


    if not required:

        raise ValueError(
            "At least one required dataset "
            "must be supplied."
        )


    known = {
        dataset.dataset_id

        for dataset
        in context.datasets
    }


    unknown = (
        set(
            required
        )
        - known
    )


    if unknown:

        raise ValueError(
            "Unknown required dataset(s): "
            f"{sorted(unknown)}"
        )


    return required


# ============================================================
# ADJACENCY
# ============================================================

def _build_adjacency(
    context: RoutingRelationshipContext,
) -> dict[
    str,
    list[
        tuple[
            str,
            DatasetRelationshipSpec,
        ]
    ],
]:
    """
    Build deterministic undirected adjacency.

    Sorting matters because multiple valid paths may exist.
    Given identical structural metadata, DataLens must always
    resolve the same path.
    """

    adjacency: dict[
        str,
        list[
            tuple[
                str,
                DatasetRelationshipSpec,
            ]
        ],
    ] = {
        dataset.dataset_id:
            []

        for dataset
        in context.datasets
    }


    for relationship in context.relationships:

        adjacency[
            relationship.left_dataset_id
        ].append(
            (
                relationship.right_dataset_id,
                relationship,
            )
        )


        adjacency[
            relationship.right_dataset_id
        ].append(
            (
                relationship.left_dataset_id,
                relationship,
            )
        )


    for dataset_id in adjacency:

        adjacency[
            dataset_id
        ].sort(
            key=lambda item: (
                item[
                    0
                ],
                item[
                    1
                ].relationship_id,
            )
        )


    return adjacency


# ============================================================
# ORIENTED STEP
# ============================================================

def _make_traversal_step(
    *,
    relationship: DatasetRelationshipSpec,
    from_dataset_id: str,
    to_dataset_id: str,
) -> RelationshipTraversalStep:

    if (
        relationship.left_dataset_id
        == from_dataset_id

        and relationship.right_dataset_id
        == to_dataset_id
    ):

        from_keys = (
            relationship.left_keys
        )


        to_keys = (
            relationship.right_keys
        )


    elif (
        relationship.right_dataset_id
        == from_dataset_id

        and relationship.left_dataset_id
        == to_dataset_id
    ):

        from_keys = (
            relationship.right_keys
        )


        to_keys = (
            relationship.left_keys
        )


    else:

        raise ValueError(
            "Relationship does not connect traversal "
            f"{from_dataset_id} -> {to_dataset_id}: "
            f"{relationship.relationship_id}"
        )


    return RelationshipTraversalStep(
        relationship_id=(
            relationship.relationship_id
        ),

        kind=(
            relationship.kind
        ),

        from_dataset_id=(
            from_dataset_id
        ),

        to_dataset_id=(
            to_dataset_id
        ),

        from_keys=list(
            from_keys
        ),

        to_keys=list(
            to_keys
        ),
    )


# ============================================================
# SHORTEST VALIDATED PATH
# ============================================================

def _find_path(
    *,
    context: RoutingRelationshipContext,
    source_dataset_id: str,
    target_dataset_id: str,
) -> DatasetRelationshipPath | None:

    if (
        source_dataset_id
        == target_dataset_id
    ):

        return DatasetRelationshipPath(
            source_dataset_id=(
                source_dataset_id
            ),

            target_dataset_id=(
                target_dataset_id
            ),

            dataset_ids=[
                source_dataset_id,
            ],

            relationship_ids=[],

            steps=[],
        )


    adjacency = (
        _build_adjacency(
            context
        )
    )


    # --------------------------------------------------------
    # child -> (parent, relationship used)
    # --------------------------------------------------------

    parent: dict[
        str,
        tuple[
            str,
            DatasetRelationshipSpec,
        ]
        | None,
    ] = {
        source_dataset_id:
            None
    }


    queue = deque(
        [
            source_dataset_id,
        ]
    )


    while queue:

        current = (
            queue.popleft()
        )


        if (
            current
            == target_dataset_id
        ):
            break


        for (
            neighbour,
            relationship,
        ) in adjacency[
            current
        ]:

            if neighbour in parent:
                continue


            parent[
                neighbour
            ] = (
                current,
                relationship,
            )


            queue.append(
                neighbour
            )


    if (
        target_dataset_id
        not in parent
    ):

        return None


    # ========================================================
    # RECONSTRUCT BACKWARDS
    # ========================================================

    reversed_nodes = [
        target_dataset_id,
    ]


    reversed_edges: list[
        tuple[
            str,
            str,
            DatasetRelationshipSpec,
        ]
    ] = []


    current = (
        target_dataset_id
    )


    while (
        current
        != source_dataset_id
    ):

        parent_record = (
            parent[
                current
            ]
        )


        if parent_record is None:

            raise RuntimeError(
                "Broken relationship path reconstruction."
            )


        (
            previous,
            relationship,
        ) = parent_record


        reversed_edges.append(
            (
                previous,
                current,
                relationship,
            )
        )


        reversed_nodes.append(
            previous
        )


        current = (
            previous
        )


    dataset_ids = list(
        reversed(
            reversed_nodes
        )
    )


    edges = list(
        reversed(
            reversed_edges
        )
    )


    steps = [
        _make_traversal_step(
            relationship=(
                relationship
            ),

            from_dataset_id=(
                from_dataset_id
            ),

            to_dataset_id=(
                to_dataset_id
            ),
        )

        for (
            from_dataset_id,
            to_dataset_id,
            relationship,
        ) in edges
    ]


    return DatasetRelationshipPath(
        source_dataset_id=(
            source_dataset_id
        ),

        target_dataset_id=(
            target_dataset_id
        ),

        dataset_ids=(
            dataset_ids
        ),

        relationship_ids=[
            step.relationship_id

            for step
            in steps
        ],

        steps=(
            steps
        ),
    )


# ============================================================
# PUBLIC RESOLVER
# ============================================================

def resolve_validated_relationship_plan(
    *,
    context: RoutingRelationshipContext,
    required_dataset_ids: list[str],
) -> RelationshipResolution:
    """
    Resolve the deterministic structural paths needed to
    connect datasets belonging to one semantic requirement.

    Strategy
    --------

    The first semantic dataset is used as a deterministic
    anchor.

    A shortest validated path is resolved from that anchor to
    every other required semantic dataset.

    Paths may contain bridge datasets that were NOT part of the
    semantic dependency extracted by the LLM.

    Example
    -------

    AI semantic requirement:

        ["consultations", "care_costs"]

    Validated graph:

        consultations
            |
        patients
            |
        care_costs

    Resolution:

        required:
            consultations, care_costs

        bridge:
            patients

        all:
            consultations, care_costs, patients

        relationships:
            patients_consultations
            patients_costs

    IMPORTANT:

    This resolver never invents relationships.

    It can only traverse relationships already present in the
    validated RoutingRelationshipContext.
    """

    required = (
        _validate_required_datasets(
            context=context,

            required_dataset_ids=(
                required_dataset_ids
            ),
        )
    )


    # ========================================================
    # SINGLE DATASET
    # ========================================================

    if (
        len(
            required
        )
        == 1
    ):

        return RelationshipResolution(
            resolver_version=(
                RELATIONSHIP_PATH_RESOLVER_VERSION
            ),

            required_dataset_ids=(
                required
            ),

            bridge_dataset_ids=[],

            all_dataset_ids=(
                required
            ),

            relationship_ids=[],

            paths=[],

            unresolved_dataset_ids=[],

            connected=True,
        )


    # ========================================================
    # MULTI DATASET
    # ========================================================

    anchor = (
        required[
            0
        ]
    )


    paths: list[
        DatasetRelationshipPath
    ] = []


    unresolved: list[
        str
    ] = []


    for target in required[
        1:
    ]:

        path = (
            _find_path(
                context=context,

                source_dataset_id=(
                    anchor
                ),

                target_dataset_id=(
                    target
                ),
            )
        )


        if path is None:

            unresolved.append(
                target
            )

            continue


        paths.append(
            path
        )


    # ========================================================
    # MATERIALIZATION DATASETS
    # ========================================================

    all_dataset_ids: list[
        str
    ] = []


    seen_datasets: set[
        str
    ] = set()


    def add_dataset(
        dataset_id: str,
    ) -> None:

        if dataset_id in seen_datasets:
            return


        seen_datasets.add(
            dataset_id
        )


        all_dataset_ids.append(
            dataset_id
        )


    # Semantic datasets first.
    for dataset_id in required:

        add_dataset(
            dataset_id
        )


    # Then deterministic bridge datasets.
    for path in paths:

        for dataset_id in path.dataset_ids:

            add_dataset(
                dataset_id
            )


    required_set = set(
        required
    )


    bridge_dataset_ids = [
        dataset_id

        for dataset_id
        in all_dataset_ids

        if (
            dataset_id
            not in required_set
        )
    ]


    # ========================================================
    # RELATIONSHIPS
    # ========================================================

    relationship_ids: list[
        str
    ] = []


    seen_relationships: set[
        str
    ] = set()


    for path in paths:

        for relationship_id in path.relationship_ids:

            if (
                relationship_id
                in seen_relationships
            ):
                continue


            seen_relationships.add(
                relationship_id
            )


            relationship_ids.append(
                relationship_id
            )


    return RelationshipResolution(
        resolver_version=(
            RELATIONSHIP_PATH_RESOLVER_VERSION
        ),

        required_dataset_ids=(
            required
        ),

        bridge_dataset_ids=(
            bridge_dataset_ids
        ),

        all_dataset_ids=(
            all_dataset_ids
        ),

        relationship_ids=(
            relationship_ids
        ),

        paths=(
            paths
        ),

        unresolved_dataset_ids=(
            unresolved
        ),

        connected=(
            len(
                unresolved
            )
            == 0
        ),
    )