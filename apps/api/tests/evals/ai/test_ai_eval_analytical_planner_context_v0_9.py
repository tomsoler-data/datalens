from __future__ import annotations

from app.evals.analytical_planner_context_v0_9 import (
    ANALYTICAL_PLANNER_CONTEXT_VERSION,
    STRUCTURAL_TOOLS,
    build_analytical_planner_context,
    require_ready_planner_context,
)

from app.evals.dataset_dependency_contract_v0_8 import (
    DatasetDependencyCandidate,
)

from app.evals.routing_relationships_v0_8 import (
    RoutingRelationshipContext,
)


# ============================================================
# CONTEXT
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
# HELPERS
# ============================================================

def qualified_names(
    requirement,
) -> set[str]:

    return {
        column.qualified_name

        for column
        in requirement.analytical_columns
    }


def structural_key_names(
    requirement,
) -> set[str]:

    return {
        key.qualified_name

        for key
        in requirement.structural_keys
    }


# ============================================================
# 1. SINGLE DATASET CONTEXT
# ============================================================

def test_single_dataset_context() -> None:

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

            context=(
                make_context(
                    with_join_tool=False,
                )
            ),
        )
    )


    assert (
        planner_context.ready_for_planning
    )


    assert (
        len(
            planner_context.requirements
        )
        == 1
    )


    requirement = (
        planner_context.requirements[
            0
        ]
    )


    assert (
        requirement.semantic_dataset_ids
        == [
            "sales",
        ]
    )


    assert (
        requirement.bridge_dataset_ids
        == []
    )


    assert (
        qualified_names(
            requirement
        )
        == {
            "sales.customer_id",
            "sales.month",
            "sales.revenue",
        }
    )


    assert (
        requirement.structural_keys
        == []
    )


    print(
        "Single-dataset planner context: PASS"
    )


# ============================================================
# 2. MULTI-HOP SEMANTIC COLUMNS
# ============================================================

def test_multihop_analytical_columns() -> None:

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
            context=(
                make_context()
            ),
        )
    )


    assert (
        planner_context.ready_for_planning
    )


    requirement = (
        planner_context.requirements[
            0
        ]
    )


    assert (
        requirement.bridge_dataset_ids
        == [
            "customers",
        ]
    )


    names = (
        qualified_names(
            requirement
        )
    )


    # ========================================================
    # SEMANTIC DATASET COLUMNS ARE AVAILABLE
    # ========================================================

    assert (
        "sales.revenue"
        in names
    )


    assert (
        "support.ticket_count"
        in names
    )


    # ========================================================
    # BRIDGE-ONLY ANALYTICAL COLUMN MUST NOT LEAK
    # ========================================================

    assert (
        "customers.segment"
        not in names
    )


    print(
        "Multi-hop analytical columns isolated: PASS"
    )


# ============================================================
# 3. STRUCTURAL BRIDGE KEYS
# ============================================================

def test_structural_keys_exposed() -> None:

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
            context=(
                make_context()
            ),
        )
    )


    requirement = (
        planner_context.requirements[
            0
        ]
    )


    keys = (
        structural_key_names(
            requirement
        )
    )


    assert (
        keys
        == {
            "sales.customer_id",
            "customers.customer_id",
            "support.customer_id",
        }
    )


    print(
        "Structural keys exposed separately: PASS"
    )


# ============================================================
# 4. STRUCTURAL TOOL REMOVED
# ============================================================

def test_join_tool_hidden_from_planner() -> None:

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
            context=(
                make_context()
            ),
        )
    )


    requirement = (
        planner_context.requirements[
            0
        ]
    )


    assert (
        "join_datasets"
        not in (
            requirement
            .allowed_analytical_tools
        )
    )


    assert all(
        structural_tool
        not in requirement.allowed_analytical_tools

        for structural_tool
        in STRUCTURAL_TOOLS
    )


    assert (
        "measure_association"
        in requirement.allowed_analytical_tools
    )


    print(
        "Structural tool hidden from planner: PASS"
    )


# ============================================================
# 5. INDEPENDENT REQUIREMENTS
# ============================================================

def test_independent_requirements() -> None:

    candidate = (
        DatasetDependencyCandidate
        .model_validate(
            {
                "requirements": [

                    {
                        "requirement_id":
                            "sales_total",

                        "dataset_ids": [
                            "sales",
                        ],
                    },

                    {
                        "requirement_id":
                            "support_total",

                        "dataset_ids": [
                            "support",
                        ],
                    },
                ],
            }
        )
    )


    planner_context = (
        build_analytical_planner_context(
            candidate=candidate,

            context=(
                make_context(
                    with_join_tool=False,
                )
            ),
        )
    )


    assert (
        planner_context.ready_for_planning
    )


    assert (
        len(
            planner_context.requirements
        )
        == 2
    )


    first = (
        planner_context.requirements[
            0
        ]
    )


    second = (
        planner_context.requirements[
            1
        ]
    )


    assert (
        qualified_names(
            first
        )
        == {
            "sales.customer_id",
            "sales.month",
            "sales.revenue",
        }
    )


    assert (
        qualified_names(
            second
        )
        == {
            "support.customer_id",
            "support.month",
            "support.ticket_count",
        }
    )


    print(
        "Independent planner contexts: PASS"
    )


# ============================================================
# 6. BLOCKED REQUEST
# ============================================================

def test_blocked_context() -> None:

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
            context=(
                make_context()
            ),
        )
    )


    assert not (
        planner_context.ready_for_planning
    )


    assert (
        planner_context.blocking_requirements
        == [
            "sales_machine_association",
        ]
    )


    assert (
        planner_context.routing_override_reason
        == "unsupported_analysis"
    )


    assert (
        planner_context.requirements
        == []
    )


    print(
        "Blocked planner context: PASS"
    )


# ============================================================
# 7. READY GUARD
# ============================================================

def test_ready_guard() -> None:

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
            context=(
                make_context()
            ),
        )
    )


    try:

        require_ready_planner_context(
            planner_context
        )


    except ValueError as error:

        assert (
            "sales_machine_association"
            in str(
                error
            )
        )


        print(
            "Blocked planner invocation rejected: PASS"
        )


    else:

        raise AssertionError(
            "Blocked planner context must never "
            "reach model inference."
        )


# ============================================================
# 8. READY GUARD ACCEPTS VALID CONTEXT
# ============================================================

def test_ready_guard_accepts_valid_context() -> None:

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

            context=(
                make_context(
                    with_join_tool=False,
                )
            ),
        )
    )


    verified = (
        require_ready_planner_context(
            planner_context
        )
    )


    assert (
        verified
        is planner_context
    )


    print(
        "Ready planner context accepted: PASS"
    )


# ============================================================
# 9. MIXED BLOCKING REMAINS GLOBAL
# ============================================================

def test_mixed_request_remains_blocked() -> None:

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
                    },

                    {
                        "requirement_id":
                            "sales_machine_association",

                        "dataset_ids": [
                            "sales",
                            "machines",
                        ],
                    },
                ],
            }
        )
    )


    planner_context = (
        build_analytical_planner_context(
            candidate=candidate,
            context=(
                make_context()
            ),
        )
    )


    assert not (
        planner_context.ready_for_planning
    )


    assert (
        planner_context.blocking_requirements
        == [
            "sales_machine_association",
        ]
    )


    # ========================================================
    # Executable diagnostic context may remain visible,
    # but global planner invocation is forbidden.
    # ========================================================

    assert (
        len(
            planner_context.requirements
        )
        == 1
    )


    assert (
        planner_context
        .requirements[
            0
        ]
        .requirement_id
        == "total_revenue"
    )


    try:

        require_ready_planner_context(
            planner_context
        )


    except ValueError:

        print(
            "Mixed request global blocking preserved: PASS"
        )


    else:

        raise AssertionError(
            "Mixed blocked request must not "
            "reach the analytical planner."
        )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print(
        "=== DATALENS ANALYTICAL PLANNER CONTEXT v0.9 ==="
    )


    print(
        "Context:",
        ANALYTICAL_PLANNER_CONTEXT_VERSION,
    )


    print()


    test_single_dataset_context()

    test_multihop_analytical_columns()

    test_structural_keys_exposed()

    test_join_tool_hidden_from_planner()

    test_independent_requirements()

    test_blocked_context()

    test_ready_guard()

    test_ready_guard_accepts_valid_context()

    test_mixed_request_remains_blocked()


    print()

    print(
        "Analytical Planner Context v0.9: PASS"
    )


if __name__ == "__main__":
    main()