from __future__ import annotations

from app.evals.analytical_planner_contract_v0_9 import (
    AnalyticalPlannerCandidate,
)

from app.evals.analytical_planner_input_v0_9 import (
    AnalyticalPlannerInput,
)

from app.evals.analytical_planner_validator_v0_9_1 import (
    validate_analytical_planner_candidate,
)

from app.evals.analytical_reference_canonicalizer_v1_0 import (
    ANALYTICAL_REFERENCE_CANONICALIZER_VERSION,
    canonicalize_analytical_references,
    require_safe_reference_canonicalization,
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
            (
                f"{dataset_id}"
                f".{column_name}"
            ),

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
                            "aggregate",
                            "derive_metric",
                            "compare_groups",
                            "analyze_distribution",
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
                    (
                        "Analyse la distribution "
                        "du montant."
                    ),

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
# ISSUE CODES
# ============================================================

def issue_codes(
    result,
) -> list[str]:

    return [
        issue.code

        for issue
        in result.issues
    ]


# ============================================================
# 1. EXACT QUALIFIED REFERENCE
# ============================================================

def test_exact_reference_unchanged() -> None:

    candidate = (
        AnalyticalPlannerCandidate
        .model_validate(
            {
                "plans": [
                    {
                        "requirement_id":
                            "click_rate_by_channel",

                        "intent":
                            "distribution_analysis",

                        "family":
                            "distribution",

                        "target_grain":
                            "campaign_day",

                        "steps": [
                            {
                                "step_id":
                                    "distribution",

                                "action": {
                                    "name":
                                        "analyze_distribution",

                                    "target":
                                        "ad_performance.clicks",
                                },
                            }
                        ],
                    }
                ],
            }
        )
    )


    result = (
        canonicalize_analytical_references(
            candidate=candidate,
            planner_input=(
                make_marketing_input()
            ),
        )
    )


    assert result.safe


    assert (
        result.rewrites
        == []
    )


    target = (
        result
        .canonicalized_candidate
        .plans[
            0
        ]
        .steps[
            0
        ]
        .action
        .target
    )


    assert (
        target
        == "ad_performance.clicks"
    )


    print(
        "Exact qualified reference unchanged: PASS"
    )


# ============================================================
# 2. FROZEN 008 REGRESSION
#
# channel
# -> ad_performance.channel
# ============================================================

def test_frozen_008_unique_alias() -> None:

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
                                    (
                                        "compare_click_rate_"
                                        "by_channel"
                                    ),

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


    result = (
        canonicalize_analytical_references(
            candidate=candidate,
            planner_input=(
                make_marketing_input()
            ),
        )
    )


    assert result.safe


    assert (
        len(
            result.rewrites
        )
        == 1
    )


    rewrite = (
        result.rewrites[
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


    # ========================================================
    # Derived output must remain a derived reference.
    # ========================================================

    compare_action = (
        result
        .canonicalized_candidate
        .plans[
            0
        ]
        .steps[
            1
        ]
        .action
    )


    assert (
        compare_action.target
        == "click_rate"
    )


    assert (
        compare_action.group_by
        == "ad_performance.channel"
    )


    # ========================================================
    # Formula belongs to another safety layer and must remain
    # untouched.
    # ========================================================

    derive_action = (
        result
        .canonicalized_candidate
        .plans[
            0
        ]
        .steps[
            0
        ]
        .action
    )


    assert (
        derive_action.formula
        == "clicks / impressions"
    )


    # ========================================================
    # After safe canonicalization the existing validator can
    # now correctly validate this plan.
    # ========================================================

    validation = (
        validate_analytical_planner_candidate(
            candidate=(
                result
                .canonicalized_candidate
            ),

            planner_input=(
                make_marketing_input()
            ),
        )
    )


    assert validation.valid


    print(
        "Frozen 008 unique alias canonicalized: PASS"
    )


# ============================================================
# 3. FROZEN 001 REGRESSION
#
# "sum" MUST NOT become admission_count.
# ============================================================

def test_frozen_001_sum_not_repaired() -> None:

    candidate = (
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
                                    "aggregate_admission_count",

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


    result = (
        canonicalize_analytical_references(
            candidate=candidate,
            planner_input=(
                make_hospital_input()
            ),
        )
    )


    assert not result.safe


    assert (
        issue_codes(
            result
        )
        == [
            "unknown_reference",
        ]
    )


    issue = (
        result.issues[
            0
        ]
    )


    assert (
        issue.reference
        == "sum"
    )


    assert (
        issue.candidates
        == []
    )


    metric = (
        result
        .canonicalized_candidate
        .plans[
            0
        ]
        .steps[
            0
        ]
        .action
        .metrics[
            0
        ]
    )


    assert (
        metric
        == "sum"
    )


    print(
        "Frozen 001 semantic error not auto-repaired: PASS"
    )


# ============================================================
# 4. AMBIGUOUS ALIAS
# ============================================================

def test_ambiguous_alias_rejected() -> None:

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
        canonicalize_analytical_references(
            candidate=candidate,
            planner_input=(
                make_ambiguous_input()
            ),
        )
    )


    assert not result.safe


    assert (
        issue_codes(
            result
        )
        == [
            "ambiguous_reference",
        ]
    )


    issue = (
        result.issues[
            0
        ]
    )


    assert (
        issue.reference
        == "amount"
    )


    assert (
        issue.candidates
        == [
            "refunds.amount",
            "sales.amount",
        ]
    )


    # ========================================================
    # No arbitrary choice.
    # ========================================================

    target = (
        result
        .canonicalized_candidate
        .plans[
            0
        ]
        .steps[
            0
        ]
        .action
        .target
    )


    assert (
        target
        == "amount"
    )


    print(
        "Ambiguous alias rejected without guessing: PASS"
    )


# ============================================================
# 5. UNKNOWN QUALIFIED REFERENCE
#
# Explicitly wrong qualification must not be suffix-repaired.
# ============================================================

def test_unknown_qualified_reference_not_repaired() -> None:

    candidate = (
        AnalyticalPlannerCandidate
        .model_validate(
            {
                "plans": [
                    {
                        "requirement_id":
                            "click_rate_by_channel",

                        "intent":
                            "distribution_analysis",

                        "family":
                            "distribution",

                        "target_grain":
                            "campaign_day",

                        "steps": [
                            {
                                "step_id":
                                    "bad_reference",

                                "action": {
                                    "name":
                                        "analyze_distribution",

                                    "target":
                                        "wrong_dataset.clicks",
                                },
                            }
                        ],
                    }
                ],
            }
        )
    )


    result = (
        canonicalize_analytical_references(
            candidate=candidate,
            planner_input=(
                make_marketing_input()
            ),
        )
    )


    assert not result.safe


    assert (
        issue_codes(
            result
        )
        == [
            "unknown_reference",
        ]
    )


    assert (
        result
        .canonicalized_candidate
        .plans[
            0
        ]
        .steps[
            0
        ]
        .action
        .target
        == "wrong_dataset.clicks"
    )


    print(
        "Unknown qualified reference not suffix-repaired: PASS"
    )


# ============================================================
# 6. FROZEN 012 REGRESSION
#
# References are repairable.
# Grain reasoning error is NOT.
# ============================================================

def test_frozen_012_references_only() -> None:

    candidate = (
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
                                    "build_entity_view",

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
                                    "detect_entity_outliers",

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


    result = (
        canonicalize_analytical_references(
            candidate=candidate,
            planner_input=(
                make_machine_input()
            ),
        )
    )


    # ========================================================
    # All references are deterministically resolvable.
    # ========================================================

    assert result.safe


    assert (
        len(
            result.rewrites
        )
        == 2
    )


    action = (
        result
        .canonicalized_candidate
        .plans[
            0
        ]
        .steps[
            1
        ]
        .action
    )


    assert (
        action.metrics
        == [
            "machine_activity.defect_count",
            "machine_activity.downtime_minutes",
        ]
    )


    # ========================================================
    # Canonicalizer MUST NOT repair reasoning.
    # ========================================================

    plan = (
        result
        .canonicalized_candidate
        .plans[
            0
        ]
    )


    assert (
        plan.target_grain
        == "machine_shift"
    )


    # ========================================================
    # Existing validator must still reject the genuine grain
    # reasoning error.
    # ========================================================

    validation = (
        validate_analytical_planner_candidate(
            candidate=(
                result
                .canonicalized_candidate
            ),

            planner_input=(
                make_machine_input()
            ),
        )
    )


    assert not validation.valid


    validator_codes = [
        issue.code

        for issue
        in validation.issues
    ]


    assert (
        "entity_target_grain_mismatch"
        in validator_codes
    )


    assert (
        "unknown_analytical_reference"
        not in validator_codes
    )


    print(
        "Frozen 012 refs repaired but grain error preserved: PASS"
    )


# ============================================================
# 7. UNQUALIFIED ENTITY REFERENCE
# ============================================================

def test_unique_entity_alias_canonicalized() -> None:

    candidate = (
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
                            "machine",

                        "steps": [

                            {
                                "step_id":
                                    "build",

                                "action": {
                                    "name":
                                        "build_entity_view",

                                    "entity":
                                        "machine_id",
                                },
                            },

                            {
                                "step_id":
                                    "detect",

                                "action": {
                                    "name":
                                        "detect_entity_outliers",

                                    "entity":
                                        "machine_id",

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


    result = (
        canonicalize_analytical_references(
            candidate=candidate,
            planner_input=(
                make_machine_input()
            ),
        )
    )


    assert result.safe


    canonical_candidate = (
        result.canonicalized_candidate
    )


    assert (
        canonical_candidate
        .plans[
            0
        ]
        .steps[
            0
        ]
        .action
        .entity
        == "machine_activity.machine_id"
    )


    assert (
        canonical_candidate
        .plans[
            0
        ]
        .steps[
            1
        ]
        .action
        .entity
        == "machine_activity.machine_id"
    )


    validation = (
        validate_analytical_planner_candidate(
            candidate=(
                canonical_candidate
            ),

            planner_input=(
                make_machine_input()
            ),
        )
    )


    assert validation.valid


    print(
        "Unique entity alias canonicalized: PASS"
    )


# ============================================================
# 8. SAFE EXECUTION GUARD
# ============================================================

def test_execution_guard_rejects_unknown_reference() -> None:

    candidate = (
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


    try:

        require_safe_reference_canonicalization(
            candidate=candidate,
            planner_input=(
                make_hospital_input()
            ),
        )


    except ValueError as error:

        assert (
            "unknown_reference"
            in str(
                error
            )
        )


        assert (
            "sum"
            in str(
                error
            )
        )


        print(
            "Unsafe reference execution guard: PASS"
        )


    else:

        raise AssertionError(
            "Unknown analytical references must not "
            "proceed toward validation/execution."
        )


# ============================================================
# 9. SAFE EXECUTION GUARD RETURNS CANONICAL CANDIDATE
# ============================================================

def test_execution_guard_returns_canonical_candidate() -> None:

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
                                        "clicks",
                                        "impressions",
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


    canonical_candidate = (
        require_safe_reference_canonicalization(
            candidate=candidate,
            planner_input=(
                make_marketing_input()
            ),
        )
    )


    derive = (
        canonical_candidate
        .plans[
            0
        ]
        .steps[
            0
        ]
        .action
    )


    compare = (
        canonical_candidate
        .plans[
            0
        ]
        .steps[
            1
        ]
        .action
    )


    assert (
        derive.inputs
        == [
            "ad_performance.clicks",
            "ad_performance.impressions",
        ]
    )


    assert (
        compare.target
        == "click_rate"
    )


    assert (
        compare.group_by
        == "ad_performance.channel"
    )


    print(
        "Safe execution guard returns canonical candidate: PASS"
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print(
        "=== DATALENS ANALYTICAL REFERENCE CANONICALIZER v1.0 ==="
    )


    print(
        "Canonicalizer:",
        ANALYTICAL_REFERENCE_CANONICALIZER_VERSION,
    )


    print()


    test_exact_reference_unchanged()

    test_frozen_008_unique_alias()

    test_frozen_001_sum_not_repaired()

    test_ambiguous_alias_rejected()

    test_unknown_qualified_reference_not_repaired()

    test_frozen_012_references_only()

    test_unique_entity_alias_canonicalized()

    test_execution_guard_rejects_unknown_reference()

    test_execution_guard_returns_canonical_candidate()


    print()


    print(
        "Analytical Reference Canonicalizer v1.0: PASS"
    )


if __name__ == "__main__":
    main()