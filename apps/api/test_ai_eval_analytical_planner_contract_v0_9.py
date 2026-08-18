from __future__ import annotations

import json

from pydantic import (
    ValidationError,
)

from app.evals.analytical_planner_contract_v0_9 import (
    ANALYTICAL_PLANNER_CONTRACT_VERSION,
    AnalyticalPlannerCandidate,
)


# ============================================================
# 1. SIMPLE AGGREGATION
# ============================================================

def test_simple_aggregation() -> None:

    candidate = (
        AnalyticalPlannerCandidate
        .model_validate(
            {
                "plans": [
                    {
                        "requirement_id":
                            "total_revenue",

                        "intent":
                            "aggregate_metric",

                        "family":
                            "aggregation",

                        "target_grain":
                            "global",

                        "steps": [
                            {
                                "step_id":
                                    "aggregate_revenue",

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


    assert (
        len(
            candidate.plans
        )
        == 1
    )


    plan = (
        candidate.plans[
            0
        ]
    )


    assert (
        plan.requirement_id
        == "total_revenue"
    )


    assert (
        plan.intent
        == "aggregate_metric"
    )


    assert (
        plan.family
        == "aggregation"
    )


    assert (
        plan.steps[
            0
        ]
        .action
        .name
        == "aggregate"
    )


    print(
        "Simple aggregation contract: PASS"
    )


# ============================================================
# 2. ASSOCIATION
# ============================================================

def test_association() -> None:

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
                                    "measure_relationship",

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


    action = (
        candidate
        .plans[
            0
        ]
        .steps[
            0
        ]
        .action
    )


    assert (
        action.name
        == "measure_association"
    )


    assert (
        action.target
        == "support.ticket_count"
    )


    assert (
        action.value
        == "sales.revenue"
    )


    print(
        "Association contract: PASS"
    )


# ============================================================
# 3. MULTI-STEP ENTITY PLAN
# ============================================================

def test_entity_outlier_plan() -> None:

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
                                    "build_customer_view",

                                "action": {
                                    "name":
                                        "build_entity_view",

                                    "entity":
                                        "sales.customer_id",
                                },
                            },


                            {
                                "step_id":
                                    "detect_customer_outliers",

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


    plan = (
        candidate.plans[
            0
        ]
    )


    assert (
        len(
            plan.steps
        )
        == 2
    )


    assert (
        plan.steps[
            0
        ]
        .action
        .name
        == "build_entity_view"
    )


    assert (
        plan.steps[
            1
        ]
        .action
        .name
        == "detect_entity_outliers"
    )


    print(
        "Multi-step entity plan contract: PASS"
    )


# ============================================================
# 4. DERIVED METRIC
# ============================================================

def test_derived_metric_plan() -> None:

    candidate = (
        AnalyticalPlannerCandidate
        .model_validate(
            {
                "plans": [
                    {
                        "requirement_id":
                            "conversion_rate_analysis",

                        "intent":
                            "compare_groups",

                        "family":
                            "group_comparison",

                        "target_grain":
                            "channel",

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


    assert (
        candidate
        .plans[
            0
        ]
        .steps[
            0
        ]
        .action
        .name
        == "derive_metric"
    )


    print(
        "Derived metric contract: PASS"
    )


# ============================================================
# 5. TIME SERIES
# ============================================================

def test_time_series() -> None:

    candidate = (
        AnalyticalPlannerCandidate
        .model_validate(
            {
                "plans": [
                    {
                        "requirement_id":
                            "weekly_revenue_trend",

                        "intent":
                            "time_series_analysis",

                        "family":
                            "time_series",

                        "target_grain":
                            "week",

                        "steps": [
                            {
                                "step_id":
                                    "analyze_revenue_over_time",

                                "action": {
                                    "name":
                                        "analyze_time_series",

                                    "date":
                                        "sales.week",

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


    assert (
        candidate
        .plans[
            0
        ]
        .steps[
            0
        ]
        .action
        .name
        == "analyze_time_series"
    )


    print(
        "Time-series contract: PASS"
    )


# ============================================================
# 6. DUPLICATE REQUIREMENT ID REJECTED
# ============================================================

def test_duplicate_requirement_id_rejected() -> None:

    try:

        AnalyticalPlannerCandidate.model_validate(
            {
                "plans": [

                    {
                        "requirement_id":
                            "same_requirement",

                        "intent":
                            "aggregate_metric",

                        "family":
                            "aggregation",

                        "target_grain":
                            "global",

                        "steps": [
                            {
                                "step_id":
                                    "first",

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
                            "same_requirement",

                        "intent":
                            "distribution_analysis",

                        "family":
                            "distribution",

                        "target_grain":
                            "order",

                        "steps": [
                            {
                                "step_id":
                                    "second",

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


    except ValidationError:

        print(
            "Duplicate requirement id rejected: PASS"
        )


    else:

        raise AssertionError(
            "Duplicate planner requirement IDs "
            "must be rejected."
        )


# ============================================================
# 7. DUPLICATE STEP ID REJECTED
# ============================================================

def test_duplicate_step_id_rejected() -> None:

    try:

        AnalyticalPlannerCandidate.model_validate(
            {
                "plans": [
                    {
                        "requirement_id":
                            "distribution",

                        "intent":
                            "distribution_analysis",

                        "family":
                            "distribution",

                        "target_grain":
                            "order",

                        "steps": [

                            {
                                "step_id":
                                    "same_step",

                                "action": {
                                    "name":
                                        "analyze_distribution",

                                    "target":
                                        "sales.revenue",
                                },
                            },


                            {
                                "step_id":
                                    "same_step",

                                "action": {
                                    "name":
                                        "detect_outliers",

                                    "target":
                                        "sales.revenue",
                                },
                            },
                        ],
                    }
                ],
            }
        )


    except ValidationError:

        print(
            "Duplicate step id rejected: PASS"
        )


    else:

        raise AssertionError(
            "Duplicate step IDs must be rejected."
        )


# ============================================================
# 8. JOIN TOOL REJECTED BY SCHEMA
# ============================================================

def test_join_tool_rejected() -> None:

    try:

        AnalyticalPlannerCandidate.model_validate(
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
                            "customer_month",

                        "steps": [
                            {
                                "step_id":
                                    "invent_join",

                                "action": {
                                    "name":
                                        "join_datasets",

                                    "left":
                                        "sales",

                                    "right":
                                        "support",
                                },
                            }
                        ],
                    }
                ],
            }
        )


    except ValidationError:

        print(
            "join_datasets rejected by planner schema: PASS"
        )


    else:

        raise AssertionError(
            "The analytical planner must never "
            "control join_datasets."
        )


# ============================================================
# 9. UNKNOWN TOOL REJECTED
# ============================================================

def test_unknown_tool_rejected() -> None:

    try:

        AnalyticalPlannerCandidate.model_validate(
            {
                "plans": [
                    {
                        "requirement_id":
                            "analysis",

                        "intent":
                            "distribution_analysis",

                        "family":
                            "distribution",

                        "target_grain":
                            "order",

                        "steps": [
                            {
                                "step_id":
                                    "bad_tool",

                                "action": {
                                    "name":
                                        "invented_magic_tool",

                                    "target":
                                        "sales.revenue",
                                },
                            }
                        ],
                    }
                ],
            }
        )


    except ValidationError:

        print(
            "Unknown analytical tool rejected: PASS"
        )


    else:

        raise AssertionError(
            "Unknown analytical tools must be rejected."
        )


# ============================================================
# 10. EXTRA TOOL ARGUMENT REJECTED
# ============================================================

def test_extra_argument_rejected() -> None:

    try:

        AnalyticalPlannerCandidate.model_validate(
            {
                "plans": [
                    {
                        "requirement_id":
                            "distribution",

                        "intent":
                            "distribution_analysis",

                        "family":
                            "distribution",

                        "target_grain":
                            "order",

                        "steps": [
                            {
                                "step_id":
                                    "distribution",

                                "action": {
                                    "name":
                                        "analyze_distribution",

                                    "target":
                                        "sales.revenue",

                                    "invented_argument":
                                        True,
                                },
                            }
                        ],
                    }
                ],
            }
        )


    except ValidationError:

        print(
            "Extra tool argument rejected: PASS"
        )


    else:

        raise AssertionError(
            "Unexpected tool arguments must be rejected."
        )


# ============================================================
# 11. REQUIRED ARGUMENT REJECTED WHEN MISSING
# ============================================================

def test_missing_required_argument_rejected() -> None:

    try:

        AnalyticalPlannerCandidate.model_validate(
            {
                "plans": [
                    {
                        "requirement_id":
                            "total_revenue",

                        "intent":
                            "aggregate_metric",

                        "family":
                            "aggregation",

                        "target_grain":
                            "global",

                        "steps": [
                            {
                                "step_id":
                                    "aggregate_revenue",

                                "action": {
                                    "name":
                                        "aggregate",

                                    "metrics": [
                                        "sales.revenue",
                                    ],

                                    # group_by is intentionally
                                    # missing.
                                },
                            }
                        ],
                    }
                ],
            }
        )


    except ValidationError:

        print(
            "Missing required tool argument rejected: PASS"
        )


    else:

        raise AssertionError(
            "Missing required tool arguments "
            "must be rejected."
        )


# ============================================================
# 12. EMPTY PLAN REJECTED
# ============================================================

def test_empty_plan_rejected() -> None:

    try:

        AnalyticalPlannerCandidate.model_validate(
            {
                "plans":
                    [],
            }
        )


    except ValidationError:

        print(
            "Empty planner candidate rejected: PASS"
        )


    else:

        raise AssertionError(
            "Planner candidate must contain "
            "at least one plan."
        )


# ============================================================
# 13. EMPTY STEPS REJECTED
# ============================================================

def test_empty_steps_rejected() -> None:

    try:

        AnalyticalPlannerCandidate.model_validate(
            {
                "plans": [
                    {
                        "requirement_id":
                            "bad_plan",

                        "intent":
                            "aggregate_metric",

                        "family":
                            "aggregation",

                        "target_grain":
                            "global",

                        "steps":
                            [],
                    }
                ],
            }
        )


    except ValidationError:

        print(
            "Empty analytical steps rejected: PASS"
        )


    else:

        raise AssertionError(
            "Each analytical plan must contain "
            "at least one step."
        )


# ============================================================
# 14. JSON SCHEMA HAS TOOL DISCRIMINATOR
# ============================================================

def test_json_schema_discriminator() -> None:

    schema = (
        AnalyticalPlannerCandidate
        .model_json_schema()
    )


    serialized = (
        json.dumps(
            schema,
            ensure_ascii=False,
        )
    )


    assert (
        '"discriminator"'
        in serialized
    )


    assert (
        '"name"'
        in serialized
    )


    assert (
        "measure_association"
        in serialized
    )


    assert (
        "analyze_time_series"
        in serialized
    )


    assert (
        "detect_entity_outliers"
        in serialized
    )


    # Structural tool must not appear anywhere in the
    # analytical planner JSON schema.
    assert (
        "join_datasets"
        not in serialized
    )


    print(
        "Planner JSON Schema discriminator: PASS"
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print(
        "=== DATALENS ANALYTICAL PLANNER CONTRACT v0.9 ==="
    )


    print(
        "Contract:",
        ANALYTICAL_PLANNER_CONTRACT_VERSION,
    )


    print()


    test_simple_aggregation()

    test_association()

    test_entity_outlier_plan()

    test_derived_metric_plan()

    test_time_series()

    test_duplicate_requirement_id_rejected()

    test_duplicate_step_id_rejected()

    test_join_tool_rejected()

    test_unknown_tool_rejected()

    test_extra_argument_rejected()

    test_missing_required_argument_rejected()

    test_empty_plan_rejected()

    test_empty_steps_rejected()

    test_json_schema_discriminator()


    print()

    print(
        "Analytical Planner Contract v0.9: PASS"
    )


if __name__ == "__main__":
    main()