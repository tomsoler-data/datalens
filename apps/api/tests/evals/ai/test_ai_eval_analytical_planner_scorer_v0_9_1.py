from __future__ import annotations

from app.evals.analytical_planner_contract_v0_9 import (
    AnalyticalPlannerCandidate,
)

from app.evals.analytical_planner_input_v0_9 import (
    AnalyticalPlannerInput,
)

from app.evals.analytical_planner_scorer_v0_9_1 import (
    ANALYTICAL_PLANNER_SCORER_VERSION,
    score_analytical_planner_candidate,
)


# ============================================================
# MARKETING INPUT
# ============================================================

def make_marketing_input() -> AnalyticalPlannerInput:

    return (
        AnalyticalPlannerInput
        .model_validate(
            {
                "input_version":
                    "analytical_planner_input_v0.9",

                "user_request":
                    (
                        "Compare le taux de conversion "
                        "entre les canaux marketing."
                    ),

                "requirements": [
                    {
                        "requirement_id":
                            "conversion_by_channel",

                        "datasets": [
                            {
                                "dataset_id":
                                    "marketing",

                                "role":
                                    "semantic",

                                "grain":
                                    "campaign_day",

                                "entity_columns": [
                                    "campaign_id",
                                ],
                            }
                        ],

                        "analytical_columns": [

                            {
                                "qualified_name":
                                    "marketing.campaign_id",

                                "dataset_id":
                                    "marketing",

                                "column_name":
                                    "campaign_id",

                                "analytical_type":
                                    "identifier",

                                "semantic_role":
                                    None,
                            },

                            {
                                "qualified_name":
                                    "marketing.channel",

                                "dataset_id":
                                    "marketing",

                                "column_name":
                                    "channel",

                                "analytical_type":
                                    "categorical",

                                "semantic_role":
                                    None,
                            },

                            {
                                "qualified_name":
                                    "marketing.visits",

                                "dataset_id":
                                    "marketing",

                                "column_name":
                                    "visits",

                                "analytical_type":
                                    "quantitative",

                                "semantic_role":
                                    None,
                            },

                            {
                                "qualified_name":
                                    "marketing.conversions",

                                "dataset_id":
                                    "marketing",

                                "column_name":
                                    "conversions",

                                "analytical_type":
                                    "quantitative",

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
                            "derive_metric",
                            "compare_groups",
                        ],
                    }
                ],
            }
        )
    )


# ============================================================
# ENTITY INPUT
# ============================================================

def make_entity_input() -> AnalyticalPlannerInput:

    return (
        AnalyticalPlannerInput
        .model_validate(
            {
                "input_version":
                    "analytical_planner_input_v0.9",

                "user_request":
                    (
                        "Quels clients ont un comportement "
                        "commercial inhabituel selon leur "
                        "revenu et leur nombre de commandes ?"
                    ),

                "requirements": [
                    {
                        "requirement_id":
                            "unusual_customers",

                        "datasets": [
                            {
                                "dataset_id":
                                    "customer_activity",

                                "role":
                                    "semantic",

                                "grain":
                                    "customer_order",

                                "entity_columns": [
                                    "customer_id",
                                ],
                            }
                        ],

                        "analytical_columns": [

                            {
                                "qualified_name":
                                    (
                                        "customer_activity"
                                        ".customer_id"
                                    ),

                                "dataset_id":
                                    "customer_activity",

                                "column_name":
                                    "customer_id",

                                "analytical_type":
                                    "identifier",

                                "semantic_role":
                                    None,
                            },

                            {
                                "qualified_name":
                                    (
                                        "customer_activity"
                                        ".revenue"
                                    ),

                                "dataset_id":
                                    "customer_activity",

                                "column_name":
                                    "revenue",

                                "analytical_type":
                                    "quantitative",

                                "semantic_role":
                                    None,
                            },

                            {
                                "qualified_name":
                                    (
                                        "customer_activity"
                                        ".order_count"
                                    ),

                                "dataset_id":
                                    "customer_activity",

                                "column_name":
                                    "order_count",

                                "analytical_type":
                                    "quantitative",

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
                            "build_entity_view",
                            "derive_metric",
                            "detect_outliers",
                            "detect_entity_outliers",
                        ],
                    }
                ],
            }
        )
    )


# ============================================================
# EXPECTED MARKETING PLAN
# ============================================================

def expected_marketing_plan() -> AnalyticalPlannerCandidate:

    return (
        AnalyticalPlannerCandidate
        .model_validate(
            {
                "plans": [
                    {
                        "requirement_id":
                            "conversion_by_channel",

                        "intent":
                            "compare_groups",

                        "family":
                            "group_comparison",

                        "target_grain":
                            "campaign_day",

                        "steps": [

                            {
                                "step_id":
                                    "derive_conversion_rate",

                                "action": {
                                    "name":
                                        "derive_metric",

                                    "inputs": [
                                        "marketing.conversions",
                                        "marketing.visits",
                                    ],

                                    "output":
                                        "conversion_rate",

                                    "formula":
                                        "conversions / visits",
                                },
                            },

                            {
                                "step_id":
                                    "compare_conversion_rate",

                                "action": {
                                    "name":
                                        "compare_groups",

                                    "target":
                                        "conversion_rate",

                                    "group_by":
                                        "marketing.channel",
                                },
                            },
                        ],
                    }
                ],
            }
        )
    )


# ============================================================
# EXPECTED ENTITY PLAN
# ============================================================

def expected_entity_plan() -> AnalyticalPlannerCandidate:

    return (
        AnalyticalPlannerCandidate
        .model_validate(
            {
                "plans": [
                    {
                        "requirement_id":
                            "unusual_customers",

                        "intent":
                            "entity_anomaly_analysis",

                        "family":
                            "entity_outlier",

                        "target_grain":
                            "customer",

                        "steps": [

                            {
                                "step_id":
                                    "build_customer",

                                "action": {
                                    "name":
                                        "build_entity_view",

                                    "entity":
                                        (
                                            "customer_activity"
                                            ".customer_id"
                                        ),
                                },
                            },

                            {
                                "step_id":
                                    "detect_customer",

                                "action": {
                                    "name":
                                        "detect_entity_outliers",

                                    "entity":
                                        (
                                            "customer_activity"
                                            ".customer_id"
                                        ),

                                    "metrics": [
                                        (
                                            "customer_activity"
                                            ".revenue"
                                        ),
                                        (
                                            "customer_activity"
                                            ".order_count"
                                        ),
                                    ],
                                },
                            },
                        ],
                    }
                ],
            }
        )
    )


# ============================================================
# 1. EXACT PLAN
# ============================================================

def test_exact_plan() -> None:

    expected = (
        expected_marketing_plan()
    )


    candidate = (
        expected_marketing_plan()
    )


    score = (
        score_analytical_planner_candidate(
            candidate=candidate,
            expected=expected,
            planner_input=(
                make_marketing_input()
            ),
        )
    )


    assert (
        score.overall
        == 1.0
    )


    assert (
        score.metrics.parsimony_score
        == 1.0
    )


    assert (
        score.metrics.validator_acceptance
        == 1.0
    )


    print(
        "Exact v0.9.1 plan: PASS"
    )


# ============================================================
# 2. QUALIFIED FORMULA EQUIVALENT
# ============================================================

def test_qualified_formula_equivalent() -> None:

    expected = (
        expected_marketing_plan()
    )


    candidate_payload = (
        expected.model_dump(
            mode="json",
        )
    )


    candidate_payload[
        "plans"
    ][
        0
    ][
        "steps"
    ][
        0
    ][
        "action"
    ][
        "formula"
    ] = (
        "marketing.conversions "
        "/ marketing.visits"
    )


    candidate = (
        AnalyticalPlannerCandidate
        .model_validate(
            candidate_payload
        )
    )


    score = (
        score_analytical_planner_candidate(
            candidate=candidate,
            expected=expected,
            planner_input=(
                make_marketing_input()
            ),
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
        "Qualified/unqualified formula equivalence: PASS"
    )


# ============================================================
# 3. REAL QWEN REGRESSION
# ============================================================

def test_qwen_extra_aggregate_regression() -> None:

    expected = (
        expected_marketing_plan()
    )


    candidate = (
        AnalyticalPlannerCandidate
        .model_validate(
            {
                "plans": [
                    {
                        "requirement_id":
                            "conversion_by_channel",

                        "intent":
                            "compare_groups",

                        "family":
                            "group_comparison",

                        "target_grain":
                            "campaign_day",

                        "steps": [

                            {
                                "step_id":
                                    "step_1",

                                "action": {
                                    "name":
                                        "aggregate",

                                    "metrics": [
                                        "marketing.conversions",
                                        "marketing.visits",
                                    ],

                                    "group_by": [
                                        "marketing.channel",
                                    ],
                                },
                            },

                            {
                                "step_id":
                                    "step_2",

                                "action": {
                                    "name":
                                        "derive_metric",

                                    "inputs": [
                                        "marketing.conversions",
                                        "marketing.visits",
                                    ],

                                    "output":
                                        "conversion_rate",

                                    "formula":
                                        (
                                            "marketing.conversions "
                                            "/ marketing.visits"
                                        ),
                                },
                            },

                            {
                                "step_id":
                                    "step_3",

                                "action": {
                                    "name":
                                        "compare_groups",

                                    "target":
                                        "conversion_rate",

                                    "group_by":
                                        "marketing.channel",
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
            planner_input=(
                make_marketing_input()
            ),
        )
    )


    # ========================================================
    # Corresponding actions still receive full argument credit.
    # ========================================================

    assert (
        score.metrics.tool_argument_score
        == 1.0
    )


    # ========================================================
    # Sequence detects the extra action.
    # ========================================================

    assert (
        score.metrics.tool_sequence_score
        == 0.8
    )


    # ========================================================
    # Explicit parsimony penalty:
    #
    # expected = 2
    # actual   = 3
    # ========================================================

    assert (
        score.metrics.parsimony_score
        == round(
            2 / 3,
            12,
        )
    )


    # ========================================================
    # Validator intentionally continues to accept the plan.
    # ========================================================

    assert (
        score.metrics.validator_acceptance
        == 1.0
    )


    assert (
        score.overall
        < 1.0
    )


    print(
        "Qwen extra-step regression scored correctly: PASS"
    )


# ============================================================
# 4. REAL MINISTRAL CONVERSION REGRESSION
# ============================================================

def test_ministral_conversion_regression() -> None:

    expected = (
        expected_marketing_plan()
    )


    candidate = (
        AnalyticalPlannerCandidate
        .model_validate(
            {
                "plans": [
                    {
                        "requirement_id":
                            "conversion_by_channel",

                        "intent":
                            "compare_groups",

                        "family":
                            "group_comparison",

                        "target_grain":
                            "campaign_day",

                        "steps": [

                            {
                                "step_id":
                                    "derive_conversion_rate",

                                "action": {
                                    "name":
                                        "derive_metric",

                                    "inputs": [
                                        "marketing.visits",
                                        "marketing.conversions",
                                    ],

                                    "output":
                                        "conversion_rate",

                                    "formula":
                                        (
                                            "marketing.conversions "
                                            "/ marketing.visits"
                                        ),
                                },
                            },

                            {
                                "step_id":
                                    "compare_channels",

                                "action": {
                                    "name":
                                        "compare_groups",

                                    "target":
                                        "marketing.channel",

                                    "group_by":
                                        "marketing.channel",
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
            planner_input=(
                make_marketing_input()
            ),
        )
    )


    # Derived formula now receives full semantic credit.
    assert (
        score.metrics.tool_sequence_score
        == 1.0
    )


    assert (
        score.metrics.tool_argument_score
        == 0.75
    )


    # But the real semantic error is now caught by Python.
    assert (
        score.metrics.validator_acceptance
        == 0.0
    )


    assert (
        "compare_groups_requires_distinct_references"
        in (
            score
            .diagnostics
            .validator_issue_codes
        )
    )


    assert (
        "invalid_compare_target_type"
        in (
            score
            .diagnostics
            .validator_issue_codes
        )
    )


    print(
        "Ministral conversion regression rescored: PASS"
    )


# ============================================================
# 5. REAL MINISTRAL ENTITY REGRESSION
# ============================================================

def test_ministral_entity_regression() -> None:

    expected = (
        expected_entity_plan()
    )


    candidate = (
        AnalyticalPlannerCandidate
        .model_validate(
            {
                "plans": [
                    {
                        "requirement_id":
                            "unusual_customers",

                        "intent":
                            "entity_anomaly_analysis",

                        "family":
                            "entity_outlier",

                        "target_grain":
                            "customer_order",

                        "steps": [

                            {
                                "step_id":
                                    "build_entity_view",

                                "action": {
                                    "name":
                                        "build_entity_view",

                                    "entity":
                                        (
                                            "customer_activity"
                                            ".customer_id"
                                        ),
                                },
                            },

                            {
                                "step_id":
                                    "derive_metrics",

                                "action": {
                                    "name":
                                        "derive_metric",

                                    "inputs": [
                                        (
                                            "customer_activity"
                                            ".revenue"
                                        ),
                                        (
                                            "customer_activity"
                                            ".order_count"
                                        ),
                                    ],

                                    "output":
                                        "revenue_per_order",

                                    "formula":
                                        "revenue / order_count",
                                },
                            },

                            {
                                "step_id":
                                    "detect_outliers",

                                "action": {
                                    "name":
                                        "detect_outliers",

                                    "target":
                                        "revenue_per_order",
                                },
                            },

                            {
                                "step_id":
                                    "detect_entity_outliers",

                                "action": {
                                    "name":
                                        "detect_entity_outliers",

                                    "entity":
                                        (
                                            "customer_activity"
                                            ".customer_id"
                                        ),

                                    "metrics": [
                                        "revenue_per_order",
                                    ],
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
            planner_input=(
                make_entity_input()
            ),
        )
    )


    assert (
        score.metrics.target_grain_accuracy
        == 0.0
    )


    assert (
        score.metrics.tool_sequence_score
        == round(
            4 / 6,
            12,
        )
    )


    assert (
        score.metrics.parsimony_score
        == 0.5
    )


    assert (
        score.metrics.validator_acceptance
        == 0.0
    )


    assert (
        "entity_target_grain_mismatch"
        in (
            score
            .diagnostics
            .validator_issue_codes
        )
    )


    assert (
        score.diagnostics.extra_step_count
        == 2
    )


    print(
        "Ministral entity regression rescored: PASS"
    )


# ============================================================
# 6. ASSOCIATION SYMMETRY
# ============================================================

def test_association_symmetry() -> None:

    planner_input = (
        AnalyticalPlannerInput
        .model_validate(
            {
                "input_version":
                    "analytical_planner_input_v0.9",

                "user_request":
                    "Analyse la relation entre x et y.",

                "requirements": [
                    {
                        "requirement_id":
                            "relationship",

                        "datasets": [
                            {
                                "dataset_id":
                                    "data",

                                "role":
                                    "semantic",

                                "grain":
                                    "row",

                                "entity_columns":
                                    [],
                            }
                        ],

                        "analytical_columns": [

                            {
                                "qualified_name":
                                    "data.x",

                                "dataset_id":
                                    "data",

                                "column_name":
                                    "x",

                                "analytical_type":
                                    "quantitative",

                                "semantic_role":
                                    None,
                            },

                            {
                                "qualified_name":
                                    "data.y",

                                "dataset_id":
                                    "data",

                                "column_name":
                                    "y",

                                "analytical_type":
                                    "quantitative",

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
                            "measure_association",
                        ],
                    }
                ],
            }
        )
    )


    expected = (
        AnalyticalPlannerCandidate
        .model_validate(
            {
                "plans": [
                    {
                        "requirement_id":
                            "relationship",

                        "intent":
                            "measure_relationship",

                        "family":
                            "association",

                        "target_grain":
                            "row",

                        "steps": [
                            {
                                "step_id":
                                    "expected",

                                "action": {
                                    "name":
                                        "measure_association",

                                    "target":
                                        "data.x",

                                    "value":
                                        "data.y",
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
                            "relationship",

                        "intent":
                            "measure_relationship",

                        "family":
                            "association",

                        "target_grain":
                            "row",

                        "steps": [
                            {
                                "step_id":
                                    "actual",

                                "action": {
                                    "name":
                                        "measure_association",

                                    "target":
                                        "data.y",

                                    "value":
                                        "data.x",
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
        "Association symmetry preserved: PASS"
    )


# ============================================================
# 7. STEP IDS STILL IGNORED
# ============================================================

def test_step_ids_ignored() -> None:

    expected = (
        expected_marketing_plan()
    )


    payload = (
        expected.model_dump(
            mode="json",
        )
    )


    payload[
        "plans"
    ][
        0
    ][
        "steps"
    ][
        0
    ][
        "step_id"
    ] = "anything"


    payload[
        "plans"
    ][
        0
    ][
        "steps"
    ][
        1
    ][
        "step_id"
    ] = "something_else"


    candidate = (
        AnalyticalPlannerCandidate
        .model_validate(
            payload
        )
    )


    score = (
        score_analytical_planner_candidate(
            candidate=candidate,
            expected=expected,
            planner_input=(
                make_marketing_input()
            ),
        )
    )


    assert (
        score.overall
        == 1.0
    )


    print(
        "Step IDs remain ignored: PASS"
    )


# ============================================================
# 8. DIFFERENT FORMULA IS NOT EQUIVALENT
# ============================================================

def test_wrong_formula_penalized() -> None:

    expected = (
        expected_marketing_plan()
    )


    payload = (
        expected.model_dump(
            mode="json",
        )
    )


    payload[
        "plans"
    ][
        0
    ][
        "steps"
    ][
        0
    ][
        "action"
    ][
        "formula"
    ] = (
        "visits / conversions"
    )


    candidate = (
        AnalyticalPlannerCandidate
        .model_validate(
            payload
        )
    )


    score = (
        score_analytical_planner_candidate(
            candidate=candidate,
            expected=expected,
            planner_input=(
                make_marketing_input()
            ),
        )
    )


    assert (
        score.metrics.tool_argument_score
        < 1.0
    )


    assert (
        score.overall
        < 1.0
    )


    print(
        "Semantically different formula penalized: PASS"
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print(
        "=== DATALENS ANALYTICAL PLANNER SCORER v0.9.1 ==="
    )


    print(
        "Scorer:",
        ANALYTICAL_PLANNER_SCORER_VERSION,
    )


    print()


    test_exact_plan()

    test_qualified_formula_equivalent()

    test_qwen_extra_aggregate_regression()

    test_ministral_conversion_regression()

    test_ministral_entity_regression()

    test_association_symmetry()

    test_step_ids_ignored()

    test_wrong_formula_penalized()


    print()

    print(
        "Analytical Planner Scorer v0.9.1: PASS"
    )


if __name__ == "__main__":
    main()