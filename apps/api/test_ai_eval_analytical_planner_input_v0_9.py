from __future__ import annotations

from app.evals.analytical_planner_context_v0_9 import (
    build_analytical_planner_context,
)

from app.evals.analytical_planner_input_v0_9 import (
    ANALYTICAL_PLANNER_INPUT_VERSION,
    build_analytical_planner_input,
)

from app.evals.dataset_dependency_contract_v0_8 import (
    DatasetDependencyCandidate,
)

from app.evals.routing_relationships_v0_8 import (
    RoutingRelationshipContext,
)


# ============================================================
# STRUCTURAL CONTEXT
# ============================================================

def make_context(
    *,
    with_join_tool: bool = True,
) -> RoutingRelationshipContext:

    tools = [
        "aggregate",
        "measure_association",
        "compare_groups",
        "analyze_distribution",
    ]


    if with_join_tool:

        tools.append(
            "join_datasets"
        )


    return (
        RoutingRelationshipContext
        .model_validate(
            {
                "datasets": [

                    # ========================================
                    # CUSTOMERS — BRIDGE
                    # ========================================

                    {
                        "dataset_id":
                            "customers",

                        "filename":
                            "customers.csv",

                        "grain":
                            "customer",

                        "entity_columns": [
                            "customer_id",
                        ],

                        "columns": [
                            {
                                "name":
                                    "customer_id",

                                "analytical_type":
                                    "identifier",
                            },

                            {
                                "name":
                                    "segment",

                                "analytical_type":
                                    "categorical",
                            },
                        ],
                    },


                    # ========================================
                    # SALES
                    # ========================================

                    {
                        "dataset_id":
                            "sales",

                        "filename":
                            "sales.csv",

                        "grain":
                            "customer_month",

                        "entity_columns": [
                            "customer_id",
                        ],

                        "columns": [
                            {
                                "name":
                                    "customer_id",

                                "analytical_type":
                                    "identifier",
                            },

                            {
                                "name":
                                    "month",

                                "analytical_type":
                                    "temporal",
                            },

                            {
                                "name":
                                    "revenue",

                                "analytical_type":
                                    "quantitative",
                            },
                        ],
                    },


                    # ========================================
                    # SUPPORT
                    # ========================================

                    {
                        "dataset_id":
                            "support",

                        "filename":
                            "support.csv",

                        "grain":
                            "customer_month",

                        "entity_columns": [
                            "customer_id",
                        ],

                        "columns": [
                            {
                                "name":
                                    "customer_id",

                                "analytical_type":
                                    "identifier",
                            },

                            {
                                "name":
                                    "month",

                                "analytical_type":
                                    "temporal",
                            },

                            {
                                "name":
                                    "ticket_count",

                                "analytical_type":
                                    "quantitative",
                            },
                        ],
                    },


                    # ========================================
                    # MACHINES — DISCONNECTED
                    # ========================================

                    {
                        "dataset_id":
                            "machines",

                        "filename":
                            "machines.csv",

                        "grain":
                            "machine_day",

                        "entity_columns": [
                            "machine_id",
                        ],

                        "columns": [
                            {
                                "name":
                                    "machine_id",

                                "analytical_type":
                                    "identifier",
                            },

                            {
                                "name":
                                    "defect_count",

                                "analytical_type":
                                    "quantitative",
                            },
                        ],
                    },
                ],

                "relationships": [

                    {
                        "relationship_id":
                            "customers_sales",

                        "left_dataset_id":
                            "customers",

                        "right_dataset_id":
                            "sales",

                        "kind":
                            "join",

                        "left_keys": [
                            "customer_id",
                        ],

                        "right_keys": [
                            "customer_id",
                        ],

                        "validated":
                            True,
                    },


                    {
                        "relationship_id":
                            "customers_support",

                        "left_dataset_id":
                            "customers",

                        "right_dataset_id":
                            "support",

                        "kind":
                            "join",

                        "left_keys": [
                            "customer_id",
                        ],

                        "right_keys": [
                            "customer_id",
                        ],

                        "validated":
                            True,
                    },
                ],

                "available_tools":
                    tools,
            }
        )
    )


# ============================================================
# 1. SINGLE DATASET
# ============================================================

def test_single_dataset_input() -> None:

    structural_context = (
        make_context(
            with_join_tool=False,
        )
    )


    candidate = (
        DatasetDependencyCandidate
        .model_validate(
            {
                "requirements": [
                    {
                        "requirement_id":
                            "total_revenue",

                        "dataset_ids": [
                            "sales",
                        ],
                    }
                ],
            }
        )
    )


    planner_context = (
        build_analytical_planner_context(
            candidate=candidate,
            context=structural_context,
        )
    )


    planner_input = (
        build_analytical_planner_input(
            user_request=(
                "Quel est le chiffre d'affaires total ?"
            ),

            planner_context=(
                planner_context
            ),

            structural_context=(
                structural_context
            ),
        )
    )


    assert (
        planner_input.input_version
        == ANALYTICAL_PLANNER_INPUT_VERSION
    )


    assert (
        len(
            planner_input.requirements
        )
        == 1
    )


    requirement = (
        planner_input.requirements[
            0
        ]
    )


    assert (
        len(
            requirement.datasets
        )
        == 1
    )


    dataset = (
        requirement.datasets[
            0
        ]
    )


    assert (
        dataset.dataset_id
        == "sales"
    )


    assert (
        dataset.role
        == "semantic"
    )


    assert (
        dataset.grain
        == "customer_month"
    )


    assert (
        dataset.entity_columns
        == [
            "customer_id",
        ]
    )


    print(
        "Single-dataset planner input: PASS"
    )


# ============================================================
# 2. MULTI-HOP DATASET ROLES
# ============================================================

def test_multihop_dataset_roles() -> None:

    structural_context = (
        make_context()
    )


    candidate = (
        DatasetDependencyCandidate
        .model_validate(
            {
                "requirements": [
                    {
                        "requirement_id":
                            "support_revenue_association",

                        "dataset_ids": [
                            "sales",
                            "support",
                        ],
                    }
                ],
            }
        )
    )


    planner_context = (
        build_analytical_planner_context(
            candidate=candidate,
            context=structural_context,
        )
    )


    planner_input = (
        build_analytical_planner_input(
            user_request=(
                "Le nombre de tickets support "
                "est-il associé au chiffre d'affaires ?"
            ),

            planner_context=(
                planner_context
            ),

            structural_context=(
                structural_context
            ),
        )
    )


    requirement = (
        planner_input.requirements[
            0
        ]
    )


    roles = {
        dataset.dataset_id:
            dataset.role

        for dataset
        in requirement.datasets
    }


    assert (
        roles
        == {
            "sales":
                "semantic",

            "support":
                "semantic",

            "customers":
                "bridge",
        }
    )


    print(
        "Multi-hop dataset roles: PASS"
    )


# ============================================================
# 3. GRAINS EXPOSED
# ============================================================

def test_grains_exposed() -> None:

    structural_context = (
        make_context()
    )


    candidate = (
        DatasetDependencyCandidate
        .model_validate(
            {
                "requirements": [
                    {
                        "requirement_id":
                            "support_revenue_association",

                        "dataset_ids": [
                            "sales",
                            "support",
                        ],
                    }
                ],
            }
        )
    )


    planner_context = (
        build_analytical_planner_context(
            candidate=candidate,
            context=structural_context,
        )
    )


    planner_input = (
        build_analytical_planner_input(
            user_request=(
                "Le nombre de tickets est-il associé "
                "au chiffre d'affaires ?"
            ),

            planner_context=planner_context,
            structural_context=structural_context,
        )
    )


    requirement = (
        planner_input.requirements[
            0
        ]
    )


    grains = {
        dataset.dataset_id:
            dataset.grain

        for dataset
        in requirement.datasets
    }


    assert (
        grains
        == {
            "sales":
                "customer_month",

            "support":
                "customer_month",

            "customers":
                "customer",
        }
    )


    print(
        "Dataset grains exposed: PASS"
    )


# ============================================================
# 4. BRIDGE ANALYTICAL COLUMN STILL HIDDEN
# ============================================================

def test_bridge_column_remains_hidden() -> None:

    structural_context = (
        make_context()
    )


    candidate = (
        DatasetDependencyCandidate
        .model_validate(
            {
                "requirements": [
                    {
                        "requirement_id":
                            "support_revenue_association",

                        "dataset_ids": [
                            "sales",
                            "support",
                        ],
                    }
                ],
            }
        )
    )


    planner_context = (
        build_analytical_planner_context(
            candidate=candidate,
            context=structural_context,
        )
    )


    planner_input = (
        build_analytical_planner_input(
            user_request=(
                "Les tickets support sont-ils associés "
                "au chiffre d'affaires ?"
            ),

            planner_context=planner_context,
            structural_context=structural_context,
        )
    )


    analytical_names = {
        column[
            "qualified_name"
        ]

        for column
        in (
            planner_input
            .requirements[
                0
            ]
            .analytical_columns
        )
    }


    assert (
        "sales.revenue"
        in analytical_names
    )


    assert (
        "support.ticket_count"
        in analytical_names
    )


    assert (
        "customers.segment"
        not in analytical_names
    )


    print(
        "Bridge analytical column remains hidden: PASS"
    )


# ============================================================
# 5. STRUCTURAL KEYS RETAINED
# ============================================================

def test_structural_keys_retained() -> None:

    structural_context = (
        make_context()
    )


    candidate = (
        DatasetDependencyCandidate
        .model_validate(
            {
                "requirements": [
                    {
                        "requirement_id":
                            "support_revenue_association",

                        "dataset_ids": [
                            "sales",
                            "support",
                        ],
                    }
                ],
            }
        )
    )


    planner_context = (
        build_analytical_planner_context(
            candidate=candidate,
            context=structural_context,
        )
    )


    planner_input = (
        build_analytical_planner_input(
            user_request=(
                "Analyse la relation entre tickets "
                "et chiffre d'affaires."
            ),

            planner_context=planner_context,
            structural_context=structural_context,
        )
    )


    structural_names = {
        key[
            "qualified_name"
        ]

        for key
        in (
            planner_input
            .requirements[
                0
            ]
            .structural_keys
        )
    }


    assert (
        structural_names
        == {
            "sales.customer_id",
            "customers.customer_id",
            "support.customer_id",
        }
    )


    print(
        "Structural keys retained: PASS"
    )


# ============================================================
# 6. JOIN TOOL REMAINS HIDDEN
# ============================================================

def test_join_tool_remains_hidden() -> None:

    structural_context = (
        make_context()
    )


    candidate = (
        DatasetDependencyCandidate
        .model_validate(
            {
                "requirements": [
                    {
                        "requirement_id":
                            "support_revenue_association",

                        "dataset_ids": [
                            "sales",
                            "support",
                        ],
                    }
                ],
            }
        )
    )


    planner_context = (
        build_analytical_planner_context(
            candidate=candidate,
            context=structural_context,
        )
    )


    planner_input = (
        build_analytical_planner_input(
            user_request=(
                "Analyse la relation entre tickets "
                "et chiffre d'affaires."
            ),

            planner_context=planner_context,
            structural_context=structural_context,
        )
    )


    tools = (
        planner_input
        .requirements[
            0
        ]
        .allowed_analytical_tools
    )


    assert (
        "join_datasets"
        not in tools
    )


    assert (
        "measure_association"
        in tools
    )


    print(
        "Join tool remains hidden: PASS"
    )


# ============================================================
# 7. BLOCKED REQUEST CANNOT BECOME INPUT
# ============================================================

def test_blocked_request_rejected() -> None:

    structural_context = (
        make_context()
    )


    candidate = (
        DatasetDependencyCandidate
        .model_validate(
            {
                "requirements": [
                    {
                        "requirement_id":
                            "sales_machine_association",

                        "dataset_ids": [
                            "sales",
                            "machines",
                        ],
                    }
                ],
            }
        )
    )


    planner_context = (
        build_analytical_planner_context(
            candidate=candidate,
            context=structural_context,
        )
    )


    assert not (
        planner_context.ready_for_planning
    )


    try:

        build_analytical_planner_input(
            user_request=(
                "Analyse la relation entre ventes "
                "et défauts machine."
            ),

            planner_context=planner_context,
            structural_context=structural_context,
        )


    except ValueError as error:

        assert (
            "sales_machine_association"
            in str(
                error
            )
        )


        print(
            "Blocked planner input rejected: PASS"
        )


    else:

        raise AssertionError(
            "Blocked request must not become "
            "model-visible planner input."
        )


# ============================================================
# 8. EMPTY REQUEST REJECTED
# ============================================================

def test_empty_request_rejected() -> None:

    structural_context = (
        make_context(
            with_join_tool=False,
        )
    )


    candidate = (
        DatasetDependencyCandidate
        .model_validate(
            {
                "requirements": [
                    {
                        "requirement_id":
                            "total_revenue",

                        "dataset_ids": [
                            "sales",
                        ],
                    }
                ],
            }
        )
    )


    planner_context = (
        build_analytical_planner_context(
            candidate=candidate,
            context=structural_context,
        )
    )


    try:

        build_analytical_planner_input(
            user_request=(
                "   "
            ),

            planner_context=planner_context,
            structural_context=structural_context,
        )


    except ValueError:

        print(
            "Empty user request rejected: PASS"
        )


    else:

        raise AssertionError(
            "Empty user request must be rejected."
        )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print(
        "=== DATALENS ANALYTICAL PLANNER INPUT v0.9 ==="
    )


    print(
        "Input:",
        ANALYTICAL_PLANNER_INPUT_VERSION,
    )


    print()


    test_single_dataset_input()

    test_multihop_dataset_roles()

    test_grains_exposed()

    test_bridge_column_remains_hidden()

    test_structural_keys_retained()

    test_join_tool_remains_hidden()

    test_blocked_request_rejected()

    test_empty_request_rejected()


    print()

    print(
        "Analytical Planner Input v0.9: PASS"
    )


if __name__ == "__main__":
    main()