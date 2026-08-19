from __future__ import annotations

from app.evals.dataset_dependency_contract_v0_8 import (
    DatasetDependencyCandidate,
    evaluate_dataset_dependencies,
)

from app.evals.planner_structural_handoff_v0_9 import (
    PLANNER_STRUCTURAL_HANDOFF_VERSION,
    build_planner_structural_handoff,
    structural_handoff_from_gate,
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
                    # CUSTOMERS — structural bridge
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
                    # MACHINES — disconnected
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
# 1. SINGLE DATASET HANDOFF
# ============================================================

def test_single_dataset_handoff() -> None:

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


    result = (
        build_planner_structural_handoff(
            candidate=candidate,
            context=(
                make_context(
                    with_join_tool=False,
                )
            ),
        )
    )


    assert (
        result.ready_for_planning
    )


    assert (
        result.blocking_requirements
        == []
    )


    assert (
        result.routing_override_reason
        is None
    )


    assert (
        len(
            result.requirements
        )
        == 1
    )


    requirement = (
        result.requirements[
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
        requirement.materialization_dataset_ids
        == [
            "sales",
        ]
    )


    assert (
        requirement.relationship_ids
        == []
    )


    assert (
        requirement.traversal_steps
        == []
    )


    print(
        "Single-dataset planner handoff: PASS"
    )


# ============================================================
# 2. MULTI-HOP HANDOFF
# ============================================================

def test_multihop_handoff() -> None:

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


    result = (
        build_planner_structural_handoff(
            candidate=candidate,
            context=(
                make_context()
            ),
        )
    )


    assert (
        result.ready_for_planning
    )


    requirement = (
        result.requirements[
            0
        ]
    )


    assert (
        requirement.semantic_dataset_ids
        == [
            "sales",
            "support",
        ]
    )


    assert (
        requirement.bridge_dataset_ids
        == [
            "customers",
        ]
    )


    assert (
        requirement.materialization_dataset_ids
        == [
            "sales",
            "support",
            "customers",
        ]
    )


    assert (
        requirement.relationship_ids
        == [
            "customers_sales",
            "customers_support",
        ]
    )


    assert (
        len(
            requirement.traversal_steps
        )
        == 2
    )


    first = (
        requirement.traversal_steps[
            0
        ]
    )


    second = (
        requirement.traversal_steps[
            1
        ]
    )


    assert (
        first.from_dataset_id
        == "sales"
    )


    assert (
        first.to_dataset_id
        == "customers"
    )


    assert (
        first.from_keys
        == [
            "customer_id",
        ]
    )


    assert (
        first.to_keys
        == [
            "customer_id",
        ]
    )


    assert (
        second.from_dataset_id
        == "customers"
    )


    assert (
        second.to_dataset_id
        == "support"
    )


    print(
        "Multi-hop planner handoff: PASS"
    )


# ============================================================
# 3. INDEPENDENT REQUIREMENTS
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


    result = (
        build_planner_structural_handoff(
            candidate=candidate,
            context=(
                make_context(
                    with_join_tool=False,
                )
            ),
        )
    )


    assert (
        result.ready_for_planning
    )


    assert (
        len(
            result.requirements
        )
        == 2
    )


    assert all(
        requirement.bridge_dataset_ids
        == []

        for requirement
        in result.requirements
    )


    assert all(
        requirement.relationship_ids
        == []

        for requirement
        in result.requirements
    )


    print(
        "Independent planner requirements: PASS"
    )


# ============================================================
# 4. BLOCKED — NO COMBINATION CAPABILITY
# ============================================================

def test_missing_combination_capability_blocks_planning() -> None:

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


    result = (
        build_planner_structural_handoff(
            candidate=candidate,
            context=(
                make_context(
                    with_join_tool=False,
                )
            ),
        )
    )


    assert not (
        result.ready_for_planning
    )


    assert (
        result.blocking_requirements
        == [
            "support_revenue_association",
        ]
    )


    assert (
        result.routing_override_reason
        == "unsupported_analysis"
    )


    # Blocked requirements are not handed to planner.
    assert (
        result.requirements
        == []
    )


    print(
        "Missing combination capability blocks planner: PASS"
    )


# ============================================================
# 5. BLOCKED — NO VALIDATED RELATIONSHIP
# ============================================================

def test_missing_relationship_blocks_planning() -> None:

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


    result = (
        build_planner_structural_handoff(
            candidate=candidate,
            context=(
                make_context()
            ),
        )
    )


    assert not (
        result.ready_for_planning
    )


    assert (
        result.blocking_requirements
        == [
            "sales_machine_association",
        ]
    )


    assert (
        result.routing_override_reason
        == "unsupported_analysis"
    )


    assert (
        result.requirements
        == []
    )


    print(
        "Missing relationship blocks planner: PASS"
    )


# ============================================================
# 6. MIXED REQUEST
# ============================================================

def test_mixed_request_is_atomic() -> None:

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


    result = (
        build_planner_structural_handoff(
            candidate=candidate,
            context=(
                make_context()
            ),
        )
    )


    # --------------------------------------------------------
    # Whole request is blocked in v0.9.
    # --------------------------------------------------------

    assert not (
        result.ready_for_planning
    )


    assert (
        result.routing_override_reason
        == "unsupported_analysis"
    )


    assert (
        result.blocking_requirements
        == [
            "sales_machine_association",
        ]
    )


    # --------------------------------------------------------
    # Diagnostic context for the executable requirement remains
    # available, but planner invocation is globally forbidden.
    # --------------------------------------------------------

    assert (
        len(
            result.requirements
        )
        == 1
    )


    assert (
        result
        .requirements[
            0
        ]
        .requirement_id
        == "total_revenue"
    )


    print(
        "Mixed request atomic blocking: PASS"
    )


# ============================================================
# 7. STALE GATE RESULT REJECTED
# ============================================================

def test_stale_gate_rejected() -> None:

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


    context = (
        make_context(
            with_join_tool=False,
        )
    )


    gate = (
        evaluate_dataset_dependencies(
            candidate=candidate,
            context=context,
        )
    )


    # Simulate an externally modified / stale verdict.
    tampered_gate = (
        gate.model_copy(
            update={
                "executable":
                    False,

                "routing_override_reason":
                    "unsupported_analysis",
            }
        )
    )


    try:

        structural_handoff_from_gate(
            candidate=candidate,
            context=context,
            gate_result=(
                tampered_gate
            ),
        )


    except ValueError:

        print(
            "Stale/tampered gate rejected: PASS"
        )


    else:

        raise AssertionError(
            "A stale or tampered gate result "
            "must not be trusted."
        )


# ============================================================
# 8. VERIFIED EXISTING GATE ACCEPTED
# ============================================================

def test_verified_gate_accepted() -> None:

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


    context = (
        make_context(
            with_join_tool=False,
        )
    )


    gate = (
        evaluate_dataset_dependencies(
            candidate=candidate,
            context=context,
        )
    )


    result = (
        structural_handoff_from_gate(
            candidate=candidate,
            context=context,
            gate_result=gate,
        )
    )


    assert (
        result.ready_for_planning
    )


    assert (
        len(
            result.requirements
        )
        == 1
    )


    print(
        "Verified existing gate accepted: PASS"
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print(
        "=== DATALENS PLANNER STRUCTURAL HANDOFF v0.9 ==="
    )


    print(
        "Handoff:",
        PLANNER_STRUCTURAL_HANDOFF_VERSION,
    )


    print()


    test_single_dataset_handoff()

    test_multihop_handoff()

    test_independent_requirements()

    test_missing_combination_capability_blocks_planning()

    test_missing_relationship_blocks_planning()

    test_mixed_request_is_atomic()

    test_stale_gate_rejected()

    test_verified_gate_accepted()


    print()

    print(
        "Planner Structural Handoff v0.9: PASS"
    )


if __name__ == "__main__":
    main()