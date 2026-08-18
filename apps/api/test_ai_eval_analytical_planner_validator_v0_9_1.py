from __future__ import annotations

from app.evals.analytical_planner_contract_v0_9 import (
    AnalyticalPlannerCandidate,
)

from app.evals.analytical_planner_input_v0_9 import (
    AnalyticalPlannerInput,
)

from app.evals.analytical_planner_validator_v0_9_1 import (
    ANALYTICAL_PLANNER_VALIDATOR_VERSION,
    require_valid_analytical_plan,
    validate_analytical_planner_candidate,
)


# ============================================================
# INPUT — MARKETING
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
# INPUT — ENTITY OUTLIERS
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
                                    "customer_activity.customer_id",

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
                                    "customer_activity.revenue",

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
                                    "customer_activity.order_count",

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
# 1. VALID GROUP COMPARISON
# ============================================================

def test_valid_group_comparison() -> None:

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
                                    "derive",

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
                                    "compare",

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


    result = (
        validate_analytical_planner_candidate(
            candidate=candidate,
            planner_input=(
                make_marketing_input()
            ),
        )
    )


    assert result.valid


    print(
        "Valid quantitative group comparison: PASS"
    )


# ============================================================
# 2. REAL MINISTRAL REGRESSION — CONVERSION
# ============================================================

def test_ministral_conversion_regression() -> None:

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


    result = (
        validate_analytical_planner_candidate(
            candidate=candidate,
            planner_input=(
                make_marketing_input()
            ),
        )
    )


    codes = (
        issue_codes(
            result
        )
    )


    assert not result.valid


    assert (
        "compare_groups_requires_distinct_references"
        in codes
    )


    assert (
        "invalid_compare_target_type"
        in codes
    )


    print(
        "Ministral conversion regression blocked: PASS"
    )


# ============================================================
# 3. QUANTITATIVE GROUP_BY REJECTED
# ============================================================

def test_quantitative_group_by_rejected() -> None:

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
                                    "bad_group",

                                "action": {
                                    "name":
                                        "compare_groups",

                                    "target":
                                        "marketing.conversions",

                                    "group_by":
                                        "marketing.visits",
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
            planner_input=(
                make_marketing_input()
            ),
        )
    )


    assert (
        "invalid_compare_group_type"
        in issue_codes(
            result
        )
    )


    print(
        "Quantitative group_by rejected: PASS"
    )


# ============================================================
# 4. REAL QWEN REGRESSION REMAINS STRUCTURALLY VALID
# ============================================================

def test_qwen_conversion_regression_remains_valid() -> None:
    """
    Qwen's baseline plan contains an arguably unnecessary
    aggregation step.

    That is a scorer/parsimony question, not a deterministic
    safety violation.

    The validator must therefore continue to accept it.
    """

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


    result = (
        validate_analytical_planner_candidate(
            candidate=candidate,
            planner_input=(
                make_marketing_input()
            ),
        )
    )


    assert result.valid


    assert (
        result.issues
        == []
    )


    print(
        "Qwen alternative conversion plan remains valid: PASS"
    )


# ============================================================
# 5. VALID ENTITY GRAIN
# ============================================================

def test_valid_entity_grain() -> None:

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
                            "customer",

                        "steps": [

                            {
                                "step_id":
                                    "build",

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
                                    "detect",

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


    result = (
        validate_analytical_planner_candidate(
            candidate=candidate,
            planner_input=(
                make_entity_input()
            ),
        )
    )


    assert result.valid


    print(
        "Valid entity target grain: PASS"
    )


# ============================================================
# 6. REAL MINISTRAL REGRESSION — ENTITY GRAIN
# ============================================================

def test_ministral_entity_regression() -> None:

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


    result = (
        validate_analytical_planner_candidate(
            candidate=candidate,
            planner_input=(
                make_entity_input()
            ),
        )
    )


    assert not result.valid


    assert (
        "entity_target_grain_mismatch"
        in issue_codes(
            result
        )
    )


    print(
        "Ministral entity-grain regression blocked: PASS"
    )


# ============================================================
# 7. ENTITY COLUMN NAME IS ALSO ACCEPTED AS GRAIN
# ============================================================

def test_entity_id_grain_alias() -> None:

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
                            "customer_id",

                        "steps": [

                            {
                                "step_id":
                                    "build",

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
                                    "detect",

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
            planner_input=(
                make_entity_input()
            ),
        )
    )


    assert result.valid


    print(
        "Entity-id grain alias accepted: PASS"
    )


# ============================================================
# 8. EXECUTION GUARD USES v0.9.1
# ============================================================

def test_execution_guard_rejects_semantic_error() -> None:

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
                                    "bad",

                                "action": {
                                    "name":
                                        "compare_groups",

                                    "target":
                                        "marketing.channel",

                                    "group_by":
                                        "marketing.channel",
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
            planner_input=(
                make_marketing_input()
            ),
        )


    except ValueError as error:

        assert (
            "compare_groups_requires_distinct_references"
            in str(
                error
            )
        )


        print(
            "v0.9.1 execution guard rejects semantic error: PASS"
        )


    else:

        raise AssertionError(
            "Semantic inconsistency must not proceed "
            "toward execution."
        )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print(
        "=== DATALENS ANALYTICAL PLANNER VALIDATOR v0.9.1 ==="
    )


    print(
        "Validator:",
        ANALYTICAL_PLANNER_VALIDATOR_VERSION,
    )


    print()


    test_valid_group_comparison()

    test_ministral_conversion_regression()

    test_quantitative_group_by_rejected()

    test_qwen_conversion_regression_remains_valid()

    test_valid_entity_grain()

    test_ministral_entity_regression()

    test_entity_id_grain_alias()

    test_execution_guard_rejects_semantic_error()


    print()

    print(
        "Analytical Planner Validator v0.9.1: PASS"
    )


if __name__ == "__main__":
    main()