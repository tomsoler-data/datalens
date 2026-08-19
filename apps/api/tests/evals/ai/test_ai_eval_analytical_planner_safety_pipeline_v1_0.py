from __future__ import annotations

from app.evals.analytical_planner_contract_v0_9 import (
    AnalyticalPlannerCandidate,
)

from app.evals.analytical_planner_input_v0_9 import (
    AnalyticalPlannerInput,
)

from app.evals.analytical_planner_safety_pipeline_v1_0 import (
    ANALYTICAL_PLANNER_SAFETY_PIPELINE_VERSION,
    analytical_planner_safety_summary,
    evaluate_analytical_planner_safety,
    require_safe_analytical_plan,
)


# ============================================================
# COLUMN HELPER
# ============================================================

def analytical_column(
    *,
    dataset_id: str,
    column_name: str,
    analytical_type: str,
) -> dict:

    return {
        "qualified_name":
            f"{dataset_id}.{column_name}",

        "dataset_id":
            dataset_id,

        "column_name":
            column_name,

        "analytical_type":
            analytical_type,

        "semantic_role":
            None,
    }


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
                        "Compare le taux de clic "
                        "entre les canaux publicitaires."
                    ),

                "requirements": [
                    {
                        "requirement_id":
                            "click_rate_by_channel",

                        "datasets": [
                            {
                                "dataset_id":
                                    "ad_performance",

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

                            analytical_column(
                                dataset_id=(
                                    "ad_performance"
                                ),

                                column_name=(
                                    "campaign_id"
                                ),

                                analytical_type=(
                                    "identifier"
                                ),
                            ),

                            analytical_column(
                                dataset_id=(
                                    "ad_performance"
                                ),

                                column_name=(
                                    "channel"
                                ),

                                analytical_type=(
                                    "categorical"
                                ),
                            ),

                            analytical_column(
                                dataset_id=(
                                    "ad_performance"
                                ),

                                column_name=(
                                    "impressions"
                                ),

                                analytical_type=(
                                    "quantitative"
                                ),
                            ),

                            analytical_column(
                                dataset_id=(
                                    "ad_performance"
                                ),

                                column_name=(
                                    "clicks"
                                ),

                                analytical_type=(
                                    "quantitative"
                                ),
                            ),
                        ],

                        "structural_keys":
                            [],

                        "relationship_ids":
                            [],

                        "traversal_steps":
                            [],

                        "allowed_analytical_tools": [
                            "derive_metric",
                            "compare_groups",
                        ],
                    }
                ],
            }
        )
    )


# ============================================================
# HOSPITAL INPUT
# ============================================================

def make_hospital_input() -> AnalyticalPlannerInput:

    return (
        AnalyticalPlannerInput
        .model_validate(
            {
                "input_version":
                    "analytical_planner_input_v0.9",

                "user_request":
                    (
                        "Quel est le nombre total "
                        "d'admissions ?"
                    ),

                "requirements": [
                    {
                        "requirement_id":
                            "total_admissions",

                        "datasets": [
                            {
                                "dataset_id":
                                    "hospital_activity",

                                "role":
                                    "semantic",

                                "grain":
                                    "day",

                                "entity_columns":
                                    [],
                            }
                        ],

                        "analytical_columns": [

                            analytical_column(
                                dataset_id=(
                                    "hospital_activity"
                                ),

                                column_name=(
                                    "date"
                                ),

                                analytical_type=(
                                    "temporal"
                                ),
                            ),

                            analytical_column(
                                dataset_id=(
                                    "hospital_activity"
                                ),

                                column_name=(
                                    "admission_count"
                                ),

                                analytical_type=(
                                    "quantitative"
                                ),
                            ),
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
                    }
                ],
            }
        )
    )


# ============================================================
# MACHINE INPUT
# ============================================================

def make_machine_input() -> AnalyticalPlannerInput:

    return (
        AnalyticalPlannerInput
        .model_validate(
            {
                "input_version":
                    "analytical_planner_input_v0.9",

                "user_request":
                    (
                        "Quelles machines ont un comportement "
                        "inhabituel selon leur nombre de "
                        "défauts et leur temps d'arrêt ?"
                    ),

                "requirements": [
                    {
                        "requirement_id":
                            "unusual_machines",

                        "datasets": [
                            {
                                "dataset_id":
                                    "machine_activity",

                                "role":
                                    "semantic",

                                "grain":
                                    "machine_shift",

                                "entity_columns": [
                                    "machine_id",
                                ],
                            }
                        ],

                        "analytical_columns": [

                            analytical_column(
                                dataset_id=(
                                    "machine_activity"
                                ),

                                column_name=(
                                    "machine_id"
                                ),

                                analytical_type=(
                                    "identifier"
                                ),
                            ),

                            analytical_column(
                                dataset_id=(
                                    "machine_activity"
                                ),

                                column_name=(
                                    "defect_count"
                                ),

                                analytical_type=(
                                    "quantitative"
                                ),
                            ),

                            analytical_column(
                                dataset_id=(
                                    "machine_activity"
                                ),

                                column_name=(
                                    "downtime_minutes"
                                ),

                                analytical_type=(
                                    "quantitative"
                                ),
                            ),
                        ],

                        "structural_keys":
                            [],

                        "relationship_ids":
                            [],

                        "traversal_steps":
                            [],

                        "allowed_analytical_tools": [
                            "build_entity_view",
                            "detect_entity_outliers",
                        ],
                    }
                ],
            }
        )
    )


# ============================================================
# AMBIGUOUS INPUT
# ============================================================

def make_ambiguous_input() -> AnalyticalPlannerInput:

    return (
        AnalyticalPlannerInput
        .model_validate(
            {
                "input_version":
                    "analytical_planner_input_v0.9",

                "user_request":
                    "Analyse la distribution du montant.",

                "requirements": [
                    {
                        "requirement_id":
                            "amount_distribution",

                        "datasets": [

                            {
                                "dataset_id":
                                    "sales",

                                "role":
                                    "semantic",

                                "grain":
                                    "transaction",

                                "entity_columns":
                                    [],
                            },

                            {
                                "dataset_id":
                                    "refunds",

                                "role":
                                    "semantic",

                                "grain":
                                    "refund",

                                "entity_columns":
                                    [],
                            },
                        ],

                        "analytical_columns": [

                            analytical_column(
                                dataset_id=(
                                    "sales"
                                ),

                                column_name=(
                                    "amount"
                                ),

                                analytical_type=(
                                    "quantitative"
                                ),
                            ),

                            analytical_column(
                                dataset_id=(
                                    "refunds"
                                ),

                                column_name=(
                                    "amount"
                                ),

                                analytical_type=(
                                    "quantitative"
                                ),
                            ),
                        ],

                        "structural_keys":
                            [],

                        "relationship_ids":
                            [],

                        "traversal_steps":
                            [],

                        "allowed_analytical_tools": [
                            "analyze_distribution",
                        ],
                    }
                ],
            }
        )
    )


# ============================================================
# CANDIDATE — FROZEN 008 PATTERN
# ============================================================

def make_marketing_alias_candidate() -> (
    AnalyticalPlannerCandidate
):

    return (
        AnalyticalPlannerCandidate
        .model_validate(
            {
                "plans": [
                    {
                        "requirement_id":
                            "click_rate_by_channel",

                        "intent":
                            "compare_groups",

                        "family":
                            "group_comparison",

                        "target_grain":
                            "campaign_day",

                        "steps": [

                            {
                                "step_id":
                                    "derive_click_rate",

                                "action": {
                                    "name":
                                        "derive_metric",

                                    "inputs": [
                                        "ad_performance.clicks",
                                        "ad_performance.impressions",
                                    ],

                                    "output":
                                        "click_rate",

                                    "formula":
                                        "clicks / impressions",
                                },
                            },

                            {
                                "step_id":
                                    "compare",

                                "action": {
                                    "name":
                                        "compare_groups",

                                    "target":
                                        "click_rate",

                                    "group_by":
                                        "channel",
                                },
                            },
                        ],
                    }
                ],
            }
        )
    )


# ============================================================
# CANDIDATE — FROZEN 001 PATTERN
# ============================================================

def make_sum_candidate() -> (
    AnalyticalPlannerCandidate
):

    return (
        AnalyticalPlannerCandidate
        .model_validate(
            {
                "plans": [
                    {
                        "requirement_id":
                            "total_admissions",

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
                                        "sum",
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


# ============================================================
# CANDIDATE — FROZEN 012 PATTERN
# ============================================================

def make_machine_wrong_grain_candidate() -> (
    AnalyticalPlannerCandidate
):

    return (
        AnalyticalPlannerCandidate
        .model_validate(
            {
                "plans": [
                    {
                        "requirement_id":
                            "unusual_machines",

                        "intent":
                            "entity_anomaly_analysis",

                        "family":
                            "entity_outlier",

                        "target_grain":
                            "machine_shift",

                        "steps": [

                            {
                                "step_id":
                                    "build",

                                "action": {
                                    "name":
                                        "build_entity_view",

                                    "entity":
                                        (
                                            "machine_activity"
                                            ".machine_id"
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
                                            "machine_activity"
                                            ".machine_id"
                                        ),

                                    "metrics": [
                                        "defect_count",
                                        "downtime_minutes",
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
# 1. EXACT VALID PLAN
# ============================================================

def test_exact_valid_plan_ready() -> None:

    candidate = (
        AnalyticalPlannerCandidate
        .model_validate(
            {
                "plans": [
                    {
                        "requirement_id":
                            "click_rate_by_channel",

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
                                        "ad_performance.clicks",
                                        "ad_performance.impressions",
                                    ],

                                    "output":
                                        "click_rate",

                                    "formula":
                                        "clicks / impressions",
                                },
                            },

                            {
                                "step_id":
                                    "compare",

                                "action": {
                                    "name":
                                        "compare_groups",

                                    "target":
                                        "click_rate",

                                    "group_by":
                                        "ad_performance.channel",
                                },
                            },
                        ],
                    }
                ],
            }
        )
    )


    result = (
        evaluate_analytical_planner_safety(
            candidate=candidate,
            planner_input=(
                make_marketing_input()
            ),
        )
    )


    assert result.ready_for_execution

    assert (
        result.blocking_stage
        == "none"
    )

    assert (
        result.blocking_codes
        == []
    )

    assert (
        result.validation
        is not None
    )

    assert (
        result.validation.valid
    )

    assert (
        result.execution_candidate
        is not None
    )

    assert (
        result.canonicalization.rewrites
        == []
    )


    print(
        "Exact valid plan ready for execution: PASS"
    )


# ============================================================
# 2. FROZEN 008 — UNIQUE ALIAS THEN READY
# ============================================================

def test_frozen_008_alias_then_ready() -> None:

    result = (
        evaluate_analytical_planner_safety(
            candidate=(
                make_marketing_alias_candidate()
            ),

            planner_input=(
                make_marketing_input()
            ),
        )
    )


    assert result.ready_for_execution

    assert (
        result.blocking_stage
        == "none"
    )

    assert (
        len(
            result
            .canonicalization
            .rewrites
        )
        == 1
    )


    rewrite = (
        result
        .canonicalization
        .rewrites[
            0
        ]
    )


    assert (
        rewrite.original_reference
        == "channel"
    )

    assert (
        rewrite.canonical_reference
        == "ad_performance.channel"
    )


    assert (
        result.execution_candidate
        is not None
    )


    compare_action = (
        result
        .execution_candidate
        .plans[
            0
        ]
        .steps[
            1
        ]
        .action
    )


    assert (
        compare_action.group_by
        == "ad_performance.channel"
    )


    print(
        "Frozen 008 alias canonicalized then validated: PASS"
    )


# ============================================================
# 3. FROZEN 001 — UNKNOWN REFERENCE BLOCKS BEFORE VALIDATOR
# ============================================================

def test_frozen_001_blocks_at_reference_stage() -> None:

    result = (
        evaluate_analytical_planner_safety(
            candidate=(
                make_sum_candidate()
            ),

            planner_input=(
                make_hospital_input()
            ),
        )
    )


    assert not (
        result.ready_for_execution
    )


    assert (
        result.blocking_stage
        == "reference_canonicalization"
    )


    assert (
        result.blocking_codes
        == [
            "unknown_reference",
        ]
    )


    assert (
        result.validation
        is None
    )


    assert (
        result.execution_candidate
        is None
    )


    print(
        "Frozen 001 blocked before validator: PASS"
    )


# ============================================================
# 4. FROZEN 012 — REFERENCES FIXED, GRAIN STILL BLOCKED
# ============================================================

def test_frozen_012_blocks_at_validation_stage() -> None:

    result = (
        evaluate_analytical_planner_safety(
            candidate=(
                make_machine_wrong_grain_candidate()
            ),

            planner_input=(
                make_machine_input()
            ),
        )
    )


    # ========================================================
    # References themselves are safe.
    # ========================================================

    assert (
        result.canonicalization.safe
    )


    assert (
        len(
            result
            .canonicalization
            .rewrites
        )
        == 2
    )


    # ========================================================
    # But genuine reasoning error remains blocked.
    # ========================================================

    assert not (
        result.ready_for_execution
    )


    assert (
        result.blocking_stage
        == "planner_validation"
    )


    assert (
        "entity_target_grain_mismatch"
        in result.blocking_codes
    )


    assert (
        "unknown_analytical_reference"
        not in result.blocking_codes
    )


    assert (
        result.validation
        is not None
    )


    assert not (
        result.validation.valid
    )


    assert (
        result.execution_candidate
        is None
    )


    print(
        "Frozen 012 refs fixed but grain still blocked: PASS"
    )


# ============================================================
# 5. AMBIGUOUS REFERENCE BLOCKS
# ============================================================

def test_ambiguous_reference_blocks() -> None:

    candidate = (
        AnalyticalPlannerCandidate
        .model_validate(
            {
                "plans": [
                    {
                        "requirement_id":
                            "amount_distribution",

                        "intent":
                            "distribution_analysis",

                        "family":
                            "distribution",

                        "target_grain":
                            "transaction",

                        "steps": [
                            {
                                "step_id":
                                    "distribution",

                                "action": {
                                    "name":
                                        "analyze_distribution",

                                    "target":
                                        "amount",
                                },
                            }
                        ],
                    }
                ],
            }
        )
    )


    result = (
        evaluate_analytical_planner_safety(
            candidate=candidate,
            planner_input=(
                make_ambiguous_input()
            ),
        )
    )


    assert not (
        result.ready_for_execution
    )


    assert (
        result.blocking_stage
        == "reference_canonicalization"
    )


    assert (
        result.blocking_codes
        == [
            "ambiguous_reference",
        ]
    )


    assert (
        result.validation
        is None
    )


    print(
        "Ambiguous reference blocked deterministically: PASS"
    )


# ============================================================
# 6. REQUIRE GUARD RETURNS CANONICAL CANDIDATE
# ============================================================

def test_require_guard_returns_canonical_candidate() -> None:

    candidate = (
        require_safe_analytical_plan(
            candidate=(
                make_marketing_alias_candidate()
            ),

            planner_input=(
                make_marketing_input()
            ),
        )
    )


    compare_action = (
        candidate
        .plans[
            0
        ]
        .steps[
            1
        ]
        .action
    )


    assert (
        compare_action.group_by
        == "ad_performance.channel"
    )


    print(
        "Execution guard returns canonical candidate: PASS"
    )


# ============================================================
# 7. REQUIRE GUARD REJECTS REFERENCE ERROR
# ============================================================

def test_require_guard_rejects_reference_error() -> None:

    try:

        require_safe_analytical_plan(
            candidate=(
                make_sum_candidate()
            ),

            planner_input=(
                make_hospital_input()
            ),
        )


    except ValueError as error:

        message = str(
            error
        )


        assert (
            "reference_canonicalization"
            in message
        )


        assert (
            "unknown_reference"
            in message
        )


        print(
            "Execution guard rejects reference error: PASS"
        )


    else:

        raise AssertionError(
            "Unsafe reference plan must not be executable."
        )


# ============================================================
# 8. REQUIRE GUARD REJECTS VALIDATION ERROR
# ============================================================

def test_require_guard_rejects_reasoning_error() -> None:

    try:

        require_safe_analytical_plan(
            candidate=(
                make_machine_wrong_grain_candidate()
            ),

            planner_input=(
                make_machine_input()
            ),
        )


    except ValueError as error:

        message = str(
            error
        )


        assert (
            "planner_validation"
            in message
        )


        assert (
            "entity_target_grain_mismatch"
            in message
        )


        print(
            "Execution guard rejects reasoning error: PASS"
        )


    else:

        raise AssertionError(
            "Invalid entity-grain plan must not execute."
        )


# ============================================================
# 9. OBSERVABILITY SUMMARY
# ============================================================

def test_pipeline_summary() -> None:

    result = (
        evaluate_analytical_planner_safety(
            candidate=(
                make_marketing_alias_candidate()
            ),

            planner_input=(
                make_marketing_input()
            ),
        )
    )


    summary = (
        analytical_planner_safety_summary(
            result
        )
    )


    assert (
        summary[
            "pipeline_version"
        ]
        == (
            ANALYTICAL_PLANNER_SAFETY_PIPELINE_VERSION
        )
    )


    assert (
        summary[
            "ready_for_execution"
        ]
        is True
    )


    assert (
        summary[
            "blocking_stage"
        ]
        == "none"
    )


    assert (
        summary[
            "reference_rewrite_count"
        ]
        == 1
    )


    assert (
        summary[
            "reference_issue_count"
        ]
        == 0
    )


    assert (
        summary[
            "validator_issue_count"
        ]
        == 0
    )


    print(
        "Safety pipeline observability summary: PASS"
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print(
        "=== DATALENS ANALYTICAL PLANNER SAFETY PIPELINE v1.0 ==="
    )


    print(
        "Pipeline:",
        ANALYTICAL_PLANNER_SAFETY_PIPELINE_VERSION,
    )


    print()


    test_exact_valid_plan_ready()

    test_frozen_008_alias_then_ready()

    test_frozen_001_blocks_at_reference_stage()

    test_frozen_012_blocks_at_validation_stage()

    test_ambiguous_reference_blocks()

    test_require_guard_returns_canonical_candidate()

    test_require_guard_rejects_reference_error()

    test_require_guard_rejects_reasoning_error()

    test_pipeline_summary()


    print()


    print(
        "Analytical Planner Safety Pipeline v1.0: PASS"
    )


if __name__ == "__main__":
    main()