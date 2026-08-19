from __future__ import annotations

from app.evals.analytical_planner_contract_v0_9 import (
    AnalyticalPlannerCandidate,
)

from app.evals.analytical_planner_input_v0_9 import (
    AnalyticalPlannerInput,
)

from app.evals.analytical_planner_scorer_v0_9 import (
    ANALYTICAL_PLANNER_SCORER_VERSION,
    score_analytical_planner_candidate,
)


# ============================================================
# INPUT HELPERS
# ============================================================

def make_single_input() -> AnalyticalPlannerInput:

    return (
        AnalyticalPlannerInput
        .model_validate(
            {
                "input_version":
                    "analytical_planner_input.v0.9",

                "user_request":
                    "Analyse les ventes.",

                "requirements": [
                    {
                        "requirement_id":
                            "sales_analysis",

                        "datasets": [
                            {
                                "dataset_id":
                                    "sales",

                                "role":
                                    "semantic",

                                "grain":
                                    "customer_month",

                                "entity_columns": [
                                    "customer_id",
                                ],
                            }
                        ],

                        "analytical_columns": [
                            {
                                "qualified_name":
                                    "sales.customer_id",

                                "dataset_id":
                                    "sales",

                                "column_name":
                                    "customer_id",

                                "analytical_type":
                                    "identifier",

                                "semantic_role":
                                    None,
                            },

                            {
                                "qualified_name":
                                    "sales.month",

                                "dataset_id":
                                    "sales",

                                "column_name":
                                    "month",

                                "analytical_type":
                                    "temporal",

                                "semantic_role":
                                    None,
                            },

                            {
                                "qualified_name":
                                    "sales.revenue",

                                "dataset_id":
                                    "sales",

                                "column_name":
                                    "revenue",

                                "analytical_type":
                                    "quantitative",

                                "semantic_role":
                                    None,
                            },

                            {
                                "qualified_name":
                                    "sales.order_count",

                                "dataset_id":
                                    "sales",

                                "column_name":
                                    "order_count",

                                "analytical_type":
                                    "quantitative",

                                "semantic_role":
                                    None,
                            },

                            {
                                "qualified_name":
                                    "sales.channel",

                                "dataset_id":
                                    "sales",

                                "column_name":
                                    "channel",

                                "analytical_type":
                                    "categorical",

                                "semantic_role":
                                    None,
                            },
                        ],

                        "structural_keys":
                            [],

                        "relationship_ids":
                            [],

                        "traversal_steps":
                            [],

                        "allowed_analytical_tools": [
                            "aggregate",
                            "build_entity_view",
                            "derive_metric",
                            "analyze_distribution",
                            "detect_outliers",
                            "detect_entity_outliers",
                            "compare_groups",
                            "measure_association",
                            "analyze_time_series",
                        ],
                    }
                ],
            }
        )
    )


def make_two_requirement_input() -> AnalyticalPlannerInput:

    return (
        AnalyticalPlannerInput
        .model_validate(
            {
                "input_version":
                    "analytical_planner_input.v0.9",

                "user_request":
                    (
                        "Calcule séparément les ventes "
                        "et les tickets support."
                    ),

                "requirements": [

                    {
                        "requirement_id":
                            "sales_total",

                        "datasets": [
                            {
                                "dataset_id":
                                    "sales",

                                "role":
                                    "semantic",

                                "grain":
                                    "order",

                                "entity_columns": [
                                    "order_id",
                                ],
                            }
                        ],

                        "analytical_columns": [
                            {
                                "qualified_name":
                                    "sales.revenue",

                                "dataset_id":
                                    "sales",

                                "column_name":
                                    "revenue",

                                "analytical_type":
                                    "quantitative",

                                "semantic_role":
                                    None,
                            }
                        ],

                        "structural_keys":
                            [],

                        "relationship_ids":
                            [],

                        "traversal_steps":
                            [],

                        "allowed_analytical_tools": [
                            "aggregate",
                        ],
                    },


                    {
                        "requirement_id":
                            "support_total",

                        "datasets": [
                            {
                                "dataset_id":
                                    "support",

                                "role":
                                    "semantic",

                                "grain":
                                    "ticket",

                                "entity_columns": [
                                    "ticket_id",
                                ],
                            }
                        ],

                        "analytical_columns": [
                            {
                                "qualified_name":
                                    "support.ticket_count",

                                "dataset_id":
                                    "support",

                                "column_name":
                                    "ticket_count",

                                "analytical_type":
                                    "quantitative",

                                "semantic_role":
                                    None,
                            }
                        ],

                        "structural_keys":
                            [],

                        "relationship_ids":
                            [],

                        "traversal_steps":
                            [],

                        "allowed_analytical_tools": [
                            "aggregate",
                        ],
                    },
                ],
            }
        )
    )


# ============================================================
# EXPECTED HELPERS
# ============================================================

def make_distribution_expected() -> AnalyticalPlannerCandidate:

    return (
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
                                    "expected_distribution",

                                "action": {
                                    "name":
                                        "analyze_distribution",

                                    "target":
                                        "sales.revenue",
                                },
                            }
                        ],
                    }
                ],
            }
        )
    )


def make_two_requirement_expected() -> AnalyticalPlannerCandidate:

    return (
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
                                    "expected_sales",

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
                    },


                    {
                        "requirement_id":
                            "support_total",

                        "intent":
                            "aggregate_metric",

                        "family":
                            "aggregation",

                        "target_grain":
                            "global",

                        "steps": [
                            {
                                "step_id":
                                    "expected_support",

                                "action": {
                                    "name":
                                        "aggregate",

                                    "metrics": [
                                        "support.ticket_count",
                                    ],

                                    "group_by":
                                        None,
                                },
                            }
                        ],
                    },
                ],
            }
        )
    )


# ============================================================
# 1. EXACT PLAN
# ============================================================

def test_exact_plan() -> None:

    planner_input = (
        make_single_input()
    )


    expected = (
        make_distribution_expected()
    )


    candidate = (
        make_distribution_expected()
    )


    score = (
        score_analytical_planner_candidate(
            candidate=candidate,
            expected=expected,
            planner_input=planner_input,
        )
    )


    assert (
        score.overall
        == 1.0
    )


    assert (
        score.metrics.requirement_coverage_f1
        == 1.0
    )


    assert (
        score.metrics.intent_accuracy
        == 1.0
    )


    assert (
        score.metrics.family_accuracy
        == 1.0
    )


    assert (
        score.metrics.target_grain_accuracy
        == 1.0
    )


    assert (
        score.metrics.tool_sequence_score
        == 1.0
    )


    assert (
        score.metrics.tool_argument_score
        == 1.0
    )


    assert (
        score.metrics.validator_acceptance
        == 1.0
    )


    print(
        "Exact analytical plan: PASS"
    )


# ============================================================
# 2. STEP IDS IGNORED
# ============================================================

def test_step_ids_ignored() -> None:

    planner_input = (
        make_single_input()
    )


    expected = (
        make_distribution_expected()
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
                                    "completely_different_id",

                                "action": {
                                    "name":
                                        "analyze_distribution",

                                    "target":
                                        "sales.revenue",
                                },
                            }
                        ],
                    }
                ],
            }
        )
    )


    score = (
        score_analytical_planner_candidate(
            candidate=candidate,
            expected=expected,
            planner_input=planner_input,
        )
    )


    assert (
        score.overall
        == 1.0
    )


    print(
        "Step IDs ignored: PASS"
    )


# ============================================================
# 3. REQUIREMENT ORDER IGNORED
# ============================================================

def test_requirement_order_ignored() -> None:

    planner_input = (
        make_two_requirement_input()
    )


    expected = (
        make_two_requirement_expected()
    )


    candidate = (
        AnalyticalPlannerCandidate
        .model_validate(
            {
                "plans": [
                    expected
                    .plans[
                        1
                    ]
                    .model_dump(
                        mode="json",
                    ),

                    expected
                    .plans[
                        0
                    ]
                    .model_dump(
                        mode="json",
                    ),
                ],
            }
        )
    )


    score = (
        score_analytical_planner_candidate(
            candidate=candidate,
            expected=expected,
            planner_input=planner_input,
        )
    )


    assert (
        score.overall
        == 1.0
    )


    print(
        "Requirement order ignored: PASS"
    )


# ============================================================
# 4. ASSOCIATION ARGUMENTS SYMMETRIC
# ============================================================

def test_association_arguments_symmetric() -> None:

    planner_input = (
        make_single_input()
    )


    expected = (
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
                                    "association",

                                "action": {
                                    "name":
                                        "measure_association",

                                    "target":
                                        "sales.revenue",

                                    "value":
                                        "sales.order_count",
                                },
                            }
                        ],
                    }
                ],
            }
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
                            "measure_relationship",

                        "family":
                            "association",

                        "target_grain":
                            "customer_month",

                        "steps": [
                            {
                                "step_id":
                                    "association_reverse",

                                "action": {
                                    "name":
                                        "measure_association",

                                    "target":
                                        "sales.order_count",

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


    score = (
        score_analytical_planner_candidate(
            candidate=candidate,
            expected=expected,
            planner_input=planner_input,
        )
    )


    assert (
        score.overall
        == 1.0
    )


    print(
        "Symmetric association arguments: PASS"
    )


# ============================================================
# 5. AGGREGATE LIST ORDER IGNORED
# ============================================================

def test_aggregate_metric_order_ignored() -> None:

    planner_input = (
        make_single_input()
    )


    expected = (
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
                            "customer_month",

                        "steps": [
                            {
                                "step_id":
                                    "aggregate",

                                "action": {
                                    "name":
                                        "aggregate",

                                    "metrics": [
                                        "sales.revenue",
                                        "sales.order_count",
                                    ],

                                    "group_by": [
                                        "sales.channel",
                                        "sales.month",
                                    ],
                                },
                            }
                        ],
                    }
                ],
            }
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
                            "aggregate_metric",

                        "family":
                            "aggregation",

                        "target_grain":
                            "customer_month",

                        "steps": [
                            {
                                "step_id":
                                    "aggregate_reordered",

                                "action": {
                                    "name":
                                        "aggregate",

                                    "metrics": [
                                        "sales.order_count",
                                        "sales.revenue",
                                    ],

                                    "group_by": [
                                        "sales.month",
                                        "sales.channel",
                                    ],
                                },
                            }
                        ],
                    }
                ],
            }
        )
    )


    score = (
        score_analytical_planner_candidate(
            candidate=candidate,
            expected=expected,
            planner_input=planner_input,
        )
    )


    assert (
        score.metrics.tool_argument_score
        == 1.0
    )


    assert (
        score.overall
        == 1.0
    )


    print(
        "Aggregate list order ignored: PASS"
    )


# ============================================================
# 6. MISSING REQUIREMENT PENALIZED
# ============================================================

def test_missing_requirement_penalized() -> None:

    planner_input = (
        make_two_requirement_input()
    )


    expected = (
        make_two_requirement_expected()
    )


    candidate = (
        AnalyticalPlannerCandidate
        .model_validate(
            {
                "plans": [
                    expected
                    .plans[
                        0
                    ]
                    .model_dump(
                        mode="json",
                    )
                ],
            }
        )
    )


    score = (
        score_analytical_planner_candidate(
            candidate=candidate,
            expected=expected,
            planner_input=planner_input,
        )
    )


    assert (
        score.overall
        < 1.0
    )


    assert (
        score.metrics.requirement_coverage_f1
        < 1.0
    )


    assert (
        score.metrics.validator_acceptance
        == 0.0
    )


    assert (
        score.diagnostics.missing_requirement_ids
        == [
            "support_total",
        ]
    )


    print(
        "Missing requirement penalized: PASS"
    )


# ============================================================
# 7. EXTRA REQUIREMENT PENALIZED
# ============================================================

def test_extra_requirement_penalized() -> None:

    planner_input = (
        make_single_input()
    )


    expected = (
        make_distribution_expected()
    )


    candidate = (
        AnalyticalPlannerCandidate
        .model_validate(
            {
                "plans": [

                    expected
                    .plans[
                        0
                    ]
                    .model_dump(
                        mode="json",
                    ),


                    {
                        "requirement_id":
                            "invented_requirement",

                        "intent":
                            "distribution_analysis",

                        "family":
                            "distribution",

                        "target_grain":
                            "customer_month",

                        "steps": [
                            {
                                "step_id":
                                    "invented",

                                "action": {
                                    "name":
                                        "analyze_distribution",

                                    "target":
                                        "sales.revenue",
                                },
                            }
                        ],
                    },
                ],
            }
        )
    )


    score = (
        score_analytical_planner_candidate(
            candidate=candidate,
            expected=expected,
            planner_input=planner_input,
        )
    )


    assert (
        score.overall
        < 1.0
    )


    assert (
        score.metrics.validator_acceptance
        == 0.0
    )


    assert (
        score.diagnostics.extra_requirement_ids
        == [
            "invented_requirement",
        ]
    )


    print(
        "Extra requirement penalized: PASS"
    )


# ============================================================
# 8. WRONG INTENT / FAMILY / GRAIN
# ============================================================

def test_wrong_semantics_penalized() -> None:

    planner_input = (
        make_single_input()
    )


    expected = (
        make_distribution_expected()
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


    score = (
        score_analytical_planner_candidate(
            candidate=candidate,
            expected=expected,
            planner_input=planner_input,
        )
    )


    assert (
        score.metrics.intent_accuracy
        == 0.0
    )


    assert (
        score.metrics.family_accuracy
        == 0.0
    )


    assert (
        score.metrics.target_grain_accuracy
        == 0.0
    )


    assert (
        score.metrics.tool_sequence_score
        == 0.0
    )


    assert (
        score.overall
        < 1.0
    )


    print(
        "Wrong planner semantics penalized: PASS"
    )


# ============================================================
# 9. TOOL SEQUENCE PARTIAL CREDIT
# ============================================================

def test_tool_sequence_partial_credit() -> None:

    planner_input = (
        make_single_input()
    )


    expected = (
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
                                    "derive",

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
                                    "distribution",

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
                                    "distribution_only",

                                "action": {
                                    "name":
                                        "analyze_distribution",

                                    "target":
                                        "sales.revenue",
                                },
                            }
                        ],
                    }
                ],
            }
        )
    )


    score = (
        score_analytical_planner_candidate(
            candidate=candidate,
            expected=expected,
            planner_input=planner_input,
        )
    )


    assert (
        score.metrics.tool_sequence_score
        > 0.0
    )


    assert (
        score.metrics.tool_sequence_score
        < 1.0
    )


    print(
        "Tool sequence partial credit: PASS"
    )


# ============================================================
# 10. WRONG ARGUMENTS PENALIZED
# ============================================================

def test_wrong_arguments_penalized() -> None:

    planner_input = (
        make_single_input()
    )


    expected = (
        make_distribution_expected()
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
                                    "wrong_target",

                                "action": {
                                    "name":
                                        "analyze_distribution",

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


    score = (
        score_analytical_planner_candidate(
            candidate=candidate,
            expected=expected,
            planner_input=planner_input,
        )
    )


    assert (
        score.metrics.tool_sequence_score
        == 1.0
    )


    assert (
        score.metrics.tool_argument_score
        == 0.0
    )


    assert (
        score.overall
        < 1.0
    )


    print(
        "Wrong tool arguments penalized: PASS"
    )


# ============================================================
# 11. VALIDATOR FAILURE PENALIZED
# ============================================================

def test_validator_failure_penalized() -> None:

    planner_input = (
        make_single_input()
    )


    expected = (
        make_distribution_expected()
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
                                    "hallucinated_column",

                                "action": {
                                    "name":
                                        "analyze_distribution",

                                    "target":
                                        "sales.invented_column",
                                },
                            }
                        ],
                    }
                ],
            }
        )
    )


    score = (
        score_analytical_planner_candidate(
            candidate=candidate,
            expected=expected,
            planner_input=planner_input,
        )
    )


    assert (
        score.metrics.validator_acceptance
        == 0.0
    )


    assert (
        "unknown_analytical_reference"
        in (
            score
            .diagnostics
            .validator_issue_codes
        )
    )


    print(
        "Validator rejection penalized: PASS"
    )


# ============================================================
# 12. VALID DERIVED METRIC PLAN
# ============================================================

def test_valid_derived_metric_exact() -> None:

    planner_input = (
        make_single_input()
    )


    expected = (
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
                                    "derive_expected",

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
                                    "distribution_expected",

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
                                    "different_derive_id",

                                "action": {
                                    "name":
                                        "derive_metric",

                                    "inputs": [
                                        "sales.order_count",
                                        "sales.revenue",
                                    ],

                                    "output":
                                        "average_order_value",

                                    "formula":
                                        "revenue / order_count",
                                },
                            },


                            {
                                "step_id":
                                    "different_analysis_id",

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


    score = (
        score_analytical_planner_candidate(
            candidate=candidate,
            expected=expected,
            planner_input=planner_input,
        )
    )


    assert (
        score.overall
        == 1.0
    )


    assert (
        score.metrics.validator_acceptance
        == 1.0
    )


    print(
        "Valid derived metric plan scored exactly: PASS"
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print(
        "=== DATALENS ANALYTICAL PLANNER SCORER v0.9 ==="
    )


    print(
        "Scorer:",
        ANALYTICAL_PLANNER_SCORER_VERSION,
    )


    print()


    test_exact_plan()

    test_step_ids_ignored()

    test_requirement_order_ignored()

    test_association_arguments_symmetric()

    test_aggregate_metric_order_ignored()

    test_missing_requirement_penalized()

    test_extra_requirement_penalized()

    test_wrong_semantics_penalized()

    test_tool_sequence_partial_credit()

    test_wrong_arguments_penalized()

    test_validator_failure_penalized()

    test_valid_derived_metric_exact()


    print()

    print(
        "Analytical Planner Scorer v0.9: PASS"
    )


if __name__ == "__main__":
    main()