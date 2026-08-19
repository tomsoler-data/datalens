from __future__ import annotations

from app.evals.analytical_planner_context_v0_9 import (
    build_analytical_planner_context,
)

from app.evals.analytical_planner_contract_v0_9 import (
    AnalyticalPlannerCandidate,
)

from app.evals.analytical_planner_input_v0_9 import (
    build_analytical_planner_input,
)

from app.evals.analytical_planner_validator_v0_9 import (
    ANALYTICAL_PLANNER_VALIDATOR_VERSION,
    require_valid_analytical_plan,
    validate_analytical_planner_candidate,
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
    tools: list[str] | None = None,
) -> RoutingRelationshipContext:

    if tools is None:

        tools = [
            "aggregate",
            "build_entity_view",
            "derive_metric",
            "analyze_distribution",
            "detect_outliers",
            "detect_entity_outliers",
            "compare_groups",
            "measure_association",
            "analyze_time_series",
            "join_datasets",
        ]


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

                            {
                                "name":
                                    "order_count",

                                "analytical_type":
                                    "quantitative",
                            },

                            {
                                "name":
                                    "channel",

                                "analytical_type":
                                    "categorical",
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
# INPUT BUILDERS
# ============================================================

def make_single_sales_input(
    *,
    tools: list[str] | None = None,
):

    structural_context = (
        make_context(
            tools=tools,
        )
    )


    dependency_candidate = (
        DatasetDependencyCandidate
        .model_validate(
            {
                "requirements": [
                    {
                        "requirement_id":
                            "sales_analysis",

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
            candidate=(
                dependency_candidate
            ),
            context=(
                structural_context
            ),
        )
    )


    return (
        build_analytical_planner_input(
            user_request=(
                "Analyse les ventes."
            ),

            planner_context=(
                planner_context
            ),

            structural_context=(
                structural_context
            ),
        )
    )


def make_multihop_input():

    structural_context = (
        make_context()
    )


    dependency_candidate = (
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
            candidate=(
                dependency_candidate
            ),
            context=(
                structural_context
            ),
        )
    )


    return (
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


def make_two_requirement_input():

    structural_context = (
        make_context()
    )


    dependency_candidate = (
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
            candidate=(
                dependency_candidate
            ),
            context=(
                structural_context
            ),
        )
    )


    return (
        build_analytical_planner_input(
            user_request=(
                "Calcule séparément les ventes "
                "et le volume de tickets support."
            ),

            planner_context=(
                planner_context
            ),

            structural_context=(
                structural_context
            ),
        )
    )


# ============================================================
# ISSUE HELPER
# ============================================================

def issue_codes(
    result,
) -> set[str]:

    return {
        issue.code

        for issue
        in result.issues
    }


# ============================================================
# 1. VALID ASSOCIATION
# ============================================================

def test_valid_association() -> None:

    planner_input = (
        make_multihop_input()
    )


    candidate = (
        AnalyticalPlannerCandidate
        .model_validate(
            {
                "plans": [
                    {
                        "requirement_id":
                            "support_revenue_association",

                        "intent":
                            "measure_relationship",

                        "family":
                            "association",

                        "target_grain":
                            "customer_month",

                        "steps": [
                            {
                                "step_id":
                                    "measure",

                                "action": {
                                    "name":
                                        "measure_association",

                                    "target":
                                        "support.ticket_count",

                                    "value":
                                        "sales.revenue",
                                },
                            }
                        ],
                    }
                ],
            }
        )
    )


    result = (
        validate_analytical_planner_candidate(
            candidate=candidate,
            planner_input=planner_input,
        )
    )


    assert result.valid


    assert (
        result.validated_requirement_ids
        == [
            "support_revenue_association",
        ]
    )


    assert (
        result.issues
        == []
    )


    print(
        "Valid association plan: PASS"
    )


# ============================================================
# 2. MISSING REQUIREMENT
# ============================================================

def test_missing_requirement() -> None:

    planner_input = (
        make_two_requirement_input()
    )


    candidate = (
        AnalyticalPlannerCandidate
        .model_validate(
            {
                "plans": [
                    {
                        "requirement_id":
                            "sales_total",

                        "intent":
                            "aggregate_metric",

                        "family":
                            "aggregation",

                        "target_grain":
                            "global",

                        "steps": [
                            {
                                "step_id":
                                    "aggregate_sales",

                                "action": {
                                    "name":
                                        "aggregate",

                                    "metrics": [
                                        "sales.revenue",
                                    ],

                                    "group_by":
                                        None,
                                },
                            }
                        ],
                    }
                ],
            }
        )
    )


    result = (
        validate_analytical_planner_candidate(
            candidate=candidate,
            planner_input=planner_input,
        )
    )


    assert not result.valid


    assert (
        "missing_requirement"
        in issue_codes(
            result
        )
    )


    print(
        "Missing input requirement detected: PASS"
    )


# ============================================================
# 3. UNKNOWN REQUIREMENT
# ============================================================

def test_unknown_requirement() -> None:

    planner_input = (
        make_single_sales_input()
    )


    candidate = (
        AnalyticalPlannerCandidate
        .model_validate(
            {
                "plans": [
                    {
                        "requirement_id":
                            "invented_requirement",

                        "intent":
                            "aggregate_metric",

                        "family":
                            "aggregation",

                        "target_grain":
                            "global",

                        "steps": [
                            {
                                "step_id":
                                    "aggregate",

                                "action": {
                                    "name":
                                        "aggregate",

                                    "metrics": [
                                        "sales.revenue",
                                    ],

                                    "group_by":
                                        None,
                                },
                            }
                        ],
                    }
                ],
            }
        )
    )


    result = (
        validate_analytical_planner_candidate(
            candidate=candidate,
            planner_input=planner_input,
        )
    )


    codes = (
        issue_codes(
            result
        )
    )


    assert (
        "unknown_requirement"
        in codes
    )


    assert (
        "missing_requirement"
        in codes
    )


    print(
        "Unknown planner requirement detected: PASS"
    )


# ============================================================
# 4. TOOL NOT ALLOWED
# ============================================================

def test_tool_not_allowed() -> None:

    planner_input = (
        make_single_sales_input(
            tools=[
                "aggregate",
                "analyze_distribution",
            ]
        )
    )


    candidate = (
        AnalyticalPlannerCandidate
        .model_validate(
            {
                "plans": [
                    {
                        "requirement_id":
                            "sales_analysis",

                        "intent":
                            "compare_groups",

                        "family":
                            "group_comparison",

                        "target_grain":
                            "customer_month",

                        "steps": [
                            {
                                "step_id":
                                    "compare",

                                "action": {
                                    "name":
                                        "compare_groups",

                                    "target":
                                        "sales.revenue",

                                    "group_by":
                                        "sales.channel",
                                },
                            }
                        ],
                    }
                ],
            }
        )
    )


    result = (
        validate_analytical_planner_candidate(
            candidate=candidate,
            planner_input=planner_input,
        )
    )


    assert (
        "tool_not_allowed"
        in issue_codes(
            result
        )
    )


    print(
        "Unauthorized analytical tool detected: PASS"
    )


# ============================================================
# 5. BRIDGE COLUMN LEAK
# ============================================================

def test_bridge_column_leak() -> None:

    planner_input = (
        make_multihop_input()
    )


    candidate = (
        AnalyticalPlannerCandidate
        .model_validate(
            {
                "plans": [
                    {
                        "requirement_id":
                            "support_revenue_association",

                        "intent":
                            "measure_relationship",

                        "family":
                            "association",

                        "target_grain":
                            "customer_month",

                        "steps": [
                            {
                                "step_id":
                                    "bad_association",

                                "action": {
                                    "name":
                                        "measure_association",

                                    "target":
                                        "customers.segment",

                                    "value":
                                        "sales.revenue",
                                },
                            }
                        ],
                    }
                ],
            }
        )
    )


    result = (
        validate_analytical_planner_candidate(
            candidate=candidate,
            planner_input=planner_input,
        )
    )


    assert (
        "bridge_column_reference"
        in issue_codes(
            result
        )
    )


    print(
        "Bridge analytical leakage blocked: PASS"
    )


# ============================================================
# 6. FUTURE DERIVED METRIC
# ============================================================

def test_derived_metric_used_before_definition() -> None:

    planner_input = (
        make_single_sales_input()
    )


    candidate = (
        AnalyticalPlannerCandidate
        .model_validate(
            {
                "plans": [
                    {
                        "requirement_id":
                            "sales_analysis",

                        "intent":
                            "distribution_analysis",

                        "family":
                            "distribution",

                        "target_grain":
                            "customer_month",

                        "steps": [

                            {
                                "step_id":
                                    "analyze_first",

                                "action": {
                                    "name":
                                        "analyze_distribution",

                                    "target":
                                        "average_order_value",
                                },
                            },


                            {
                                "step_id":
                                    "derive_later",

                                "action": {
                                    "name":
                                        "derive_metric",

                                    "inputs": [
                                        "sales.revenue",
                                        "sales.order_count",
                                    ],

                                    "output":
                                        "average_order_value",

                                    "formula":
                                        "revenue / order_count",
                                },
                            },
                        ],
                    }
                ],
            }
        )
    )


    result = (
        validate_analytical_planner_candidate(
            candidate=candidate,
            planner_input=planner_input,
        )
    )


    assert (
        "derived_metric_used_before_definition"
        in issue_codes(
            result
        )
    )


    print(
        "Future derived metric reference blocked: PASS"
    )


# ============================================================
# 7. VALID DERIVED METRIC
# ============================================================

def test_valid_derived_metric_sequence() -> None:

    planner_input = (
        make_single_sales_input()
    )


    candidate = (
        AnalyticalPlannerCandidate
        .model_validate(
            {
                "plans": [
                    {
                        "requirement_id":
                            "sales_analysis",

                        "intent":
                            "distribution_analysis",

                        "family":
                            "distribution",

                        "target_grain":
                            "customer_month",

                        "steps": [

                            {
                                "step_id":
                                    "derive_aov",

                                "action": {
                                    "name":
                                        "derive_metric",

                                    "inputs": [
                                        "sales.revenue",
                                        "sales.order_count",
                                    ],

                                    "output":
                                        "average_order_value",

                                    "formula":
                                        "revenue / order_count",
                                },
                            },


                            {
                                "step_id":
                                    "analyze_aov",

                                "action": {
                                    "name":
                                        "analyze_distribution",

                                    "target":
                                        "average_order_value",
                                },
                            },
                        ],
                    }
                ],
            }
        )
    )


    result = (
        validate_analytical_planner_candidate(
            candidate=candidate,
            planner_input=planner_input,
        )
    )


    assert result.valid


    print(
        "Valid derived metric sequence: PASS"
    )


# ============================================================
# 8. DERIVED METRIC COLLISION
# ============================================================

def test_derived_metric_collision() -> None:

    planner_input = (
        make_single_sales_input()
    )


    candidate = (
        AnalyticalPlannerCandidate
        .model_validate(
            {
                "plans": [
                    {
                        "requirement_id":
                            "sales_analysis",

                        "intent":
                            "distribution_analysis",

                        "family":
                            "distribution",

                        "target_grain":
                            "customer_month",

                        "steps": [

                            {
                                "step_id":
                                    "bad_derive",

                                "action": {
                                    "name":
                                        "derive_metric",

                                    "inputs": [
                                        "sales.revenue",
                                        "sales.order_count",
                                    ],

                                    "output":
                                        "sales.revenue",

                                    "formula":
                                        "revenue / order_count",
                                },
                            },


                            {
                                "step_id":
                                    "distribution",

                                "action": {
                                    "name":
                                        "analyze_distribution",

                                    "target":
                                        "sales.revenue",
                                },
                            },
                        ],
                    }
                ],
            }
        )
    )


    result = (
        validate_analytical_planner_candidate(
            candidate=candidate,
            planner_input=planner_input,
        )
    )


    assert (
        "derived_metric_collision"
        in issue_codes(
            result
        )
    )


    print(
        "Derived metric collision detected: PASS"
    )


# ============================================================
# 9. INTENT / FAMILY MISMATCH
# ============================================================

def test_intent_family_mismatch() -> None:

    planner_input = (
        make_single_sales_input()
    )


    candidate = (
        AnalyticalPlannerCandidate
        .model_validate(
            {
                "plans": [
                    {
                        "requirement_id":
                            "sales_analysis",

                        "intent":
                            "measure_relationship",

                        "family":
                            "aggregation",

                        "target_grain":
                            "global",

                        "steps": [
                            {
                                "step_id":
                                    "aggregate",

                                "action": {
                                    "name":
                                        "aggregate",

                                    "metrics": [
                                        "sales.revenue",
                                    ],

                                    "group_by":
                                        None,
                                },
                            }
                        ],
                    }
                ],
            }
        )
    )


    result = (
        validate_analytical_planner_candidate(
            candidate=candidate,
            planner_input=planner_input,
        )
    )


    assert (
        "intent_family_mismatch"
        in issue_codes(
            result
        )
    )


    print(
        "Intent/family mismatch detected: PASS"
    )


# ============================================================
# 10. INVALID TARGET GRAIN
# ============================================================

def test_invalid_target_grain() -> None:

    planner_input = (
        make_single_sales_input()
    )


    candidate = (
        AnalyticalPlannerCandidate
        .model_validate(
            {
                "plans": [
                    {
                        "requirement_id":
                            "sales_analysis",

                        "intent":
                            "aggregate_metric",

                        "family":
                            "aggregation",

                        "target_grain":
                            "invented_grain",

                        "steps": [
                            {
                                "step_id":
                                    "aggregate",

                                "action": {
                                    "name":
                                        "aggregate",

                                    "metrics": [
                                        "sales.revenue",
                                    ],

                                    "group_by":
                                        None,
                                },
                            }
                        ],
                    }
                ],
            }
        )
    )


    result = (
        validate_analytical_planner_candidate(
            candidate=candidate,
            planner_input=planner_input,
        )
    )


    assert (
        "invalid_target_grain"
        in issue_codes(
            result
        )
    )


    print(
        "Invalid target grain detected: PASS"
    )


# ============================================================
# 11. ENTITY VIEW REQUIRED
# ============================================================

def test_entity_view_required() -> None:

    planner_input = (
        make_single_sales_input()
    )


    candidate = (
        AnalyticalPlannerCandidate
        .model_validate(
            {
                "plans": [
                    {
                        "requirement_id":
                            "sales_analysis",

                        "intent":
                            "entity_anomaly_analysis",

                        "family":
                            "entity_outlier",

                        "target_grain":
                            "customer",

                        "steps": [
                            {
                                "step_id":
                                    "detect",

                                "action": {
                                    "name":
                                        "detect_entity_outliers",

                                    "entity":
                                        "sales.customer_id",

                                    "metrics": [
                                        "sales.revenue",
                                        "sales.order_count",
                                    ],
                                },
                            }
                        ],
                    }
                ],
            }
        )
    )


    result = (
        validate_analytical_planner_candidate(
            candidate=candidate,
            planner_input=planner_input,
        )
    )


    assert (
        "entity_view_required"
        in issue_codes(
            result
        )
    )


    print(
        "Entity view requirement enforced: PASS"
    )


# ============================================================
# 12. VALID ENTITY PLAN
# ============================================================

def test_valid_entity_plan() -> None:

    planner_input = (
        make_single_sales_input()
    )


    candidate = (
        AnalyticalPlannerCandidate
        .model_validate(
            {
                "plans": [
                    {
                        "requirement_id":
                            "sales_analysis",

                        "intent":
                            "entity_anomaly_analysis",

                        "family":
                            "entity_outlier",

                        "target_grain":
                            "customer",

                        "steps": [

                            {
                                "step_id":
                                    "build",

                                "action": {
                                    "name":
                                        "build_entity_view",

                                    "entity":
                                        "sales.customer_id",
                                },
                            },


                            {
                                "step_id":
                                    "detect",

                                "action": {
                                    "name":
                                        "detect_entity_outliers",

                                    "entity":
                                        "sales.customer_id",

                                    "metrics": [
                                        "sales.revenue",
                                        "sales.order_count",
                                    ],
                                },
                            },
                        ],
                    }
                ],
            }
        )
    )


    result = (
        validate_analytical_planner_candidate(
            candidate=candidate,
            planner_input=planner_input,
        )
    )


    assert result.valid


    print(
        "Valid entity outlier plan: PASS"
    )


# ============================================================
# 13. TIME SERIES REQUIRES TEMPORAL COLUMN
# ============================================================

def test_time_series_requires_temporal_column() -> None:

    planner_input = (
        make_single_sales_input()
    )


    candidate = (
        AnalyticalPlannerCandidate
        .model_validate(
            {
                "plans": [
                    {
                        "requirement_id":
                            "sales_analysis",

                        "intent":
                            "time_series_analysis",

                        "family":
                            "time_series",

                        "target_grain":
                            "month",

                        "steps": [
                            {
                                "step_id":
                                    "trend",

                                "action": {
                                    "name":
                                        "analyze_time_series",

                                    "date":
                                        "sales.revenue",

                                    "target":
                                        "sales.order_count",
                                },
                            }
                        ],
                    }
                ],
            }
        )
    )


    result = (
        validate_analytical_planner_candidate(
            candidate=candidate,
            planner_input=planner_input,
        )
    )


    assert (
        "temporal_column_required"
        in issue_codes(
            result
        )
    )


    print(
        "Temporal column requirement enforced: PASS"
    )


# ============================================================
# 14. MISSING FAMILY ANCHOR
# ============================================================

def test_missing_family_anchor() -> None:

    planner_input = (
        make_single_sales_input()
    )


    candidate = (
        AnalyticalPlannerCandidate
        .model_validate(
            {
                "plans": [
                    {
                        "requirement_id":
                            "sales_analysis",

                        "intent":
                            "measure_relationship",

                        "family":
                            "association",

                        "target_grain":
                            "customer_month",

                        "steps": [
                            {
                                "step_id":
                                    "derive_only",

                                "action": {
                                    "name":
                                        "derive_metric",

                                    "inputs": [
                                        "sales.revenue",
                                        "sales.order_count",
                                    ],

                                    "output":
                                        "average_order_value",

                                    "formula":
                                        "revenue / order_count",
                                },
                            }
                        ],
                    }
                ],
            }
        )
    )


    result = (
        validate_analytical_planner_candidate(
            candidate=candidate,
            planner_input=planner_input,
        )
    )


    assert (
        "missing_family_anchor"
        in issue_codes(
            result
        )
    )


    print(
        "Family anchor requirement enforced: PASS"
    )


# ============================================================
# 15. EXECUTION GUARD
# ============================================================

def test_execution_guard_rejects_invalid_plan() -> None:

    planner_input = (
        make_multihop_input()
    )


    candidate = (
        AnalyticalPlannerCandidate
        .model_validate(
            {
                "plans": [
                    {
                        "requirement_id":
                            "support_revenue_association",

                        "intent":
                            "measure_relationship",

                        "family":
                            "association",

                        "target_grain":
                            "customer_month",

                        "steps": [
                            {
                                "step_id":
                                    "bad",

                                "action": {
                                    "name":
                                        "measure_association",

                                    "target":
                                        "customers.segment",

                                    "value":
                                        "sales.revenue",
                                },
                            }
                        ],
                    }
                ],
            }
        )
    )


    try:

        require_valid_analytical_plan(
            candidate=candidate,
            planner_input=planner_input,
        )


    except ValueError as error:

        assert (
            "bridge_column_reference"
            in str(
                error
            )
        )


        print(
            "Invalid plan execution guard: PASS"
        )


    else:

        raise AssertionError(
            "Invalid analytical plan must not "
            "proceed toward execution."
        )


# ============================================================
# 16. EXECUTION GUARD ACCEPTS VALID PLAN
# ============================================================

def test_execution_guard_accepts_valid_plan() -> None:

    planner_input = (
        make_multihop_input()
    )


    candidate = (
        AnalyticalPlannerCandidate
        .model_validate(
            {
                "plans": [
                    {
                        "requirement_id":
                            "support_revenue_association",

                        "intent":
                            "measure_relationship",

                        "family":
                            "association",

                        "target_grain":
                            "customer_month",

                        "steps": [
                            {
                                "step_id":
                                    "measure",

                                "action": {
                                    "name":
                                        "measure_association",

                                    "target":
                                        "support.ticket_count",

                                    "value":
                                        "sales.revenue",
                                },
                            }
                        ],
                    }
                ],
            }
        )
    )


    verified = (
        require_valid_analytical_plan(
            candidate=candidate,
            planner_input=planner_input,
        )
    )


    assert (
        verified
        is candidate
    )


    print(
        "Valid plan execution guard: PASS"
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print(
        "=== DATALENS ANALYTICAL PLANNER VALIDATOR v0.9 ==="
    )


    print(
        "Validator:",
        ANALYTICAL_PLANNER_VALIDATOR_VERSION,
    )


    print()


    test_valid_association()

    test_missing_requirement()

    test_unknown_requirement()

    test_tool_not_allowed()

    test_bridge_column_leak()

    test_derived_metric_used_before_definition()

    test_valid_derived_metric_sequence()

    test_derived_metric_collision()

    test_intent_family_mismatch()

    test_invalid_target_grain()

    test_entity_view_required()

    test_valid_entity_plan()

    test_time_series_requires_temporal_column()

    test_missing_family_anchor()

    test_execution_guard_rejects_invalid_plan()

    test_execution_guard_accepts_valid_plan()


    print()

    print(
        "Analytical Planner Validator v0.9: PASS"
    )


if __name__ == "__main__":
    main()