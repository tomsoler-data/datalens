from __future__ import annotations

from pydantic import (
    ValidationError,
)

from app.evals.dataset_dependency_contract_v0_8 import (
    DATASET_DEPENDENCY_CONTRACT_VERSION,
    DATASET_DEPENDENCY_GATE_VERSION,
    DatasetDependencyCandidate,
    dependency_gate_summary,
    evaluate_dataset_dependencies,
)

from app.evals.routing_relationships_v0_8 import (
    RoutingRelationshipContext,
)


# ============================================================
# DATA
# ============================================================

def make_context(
    *,
    with_join_tool: bool,
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
                    # MACHINES
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
                            "sales_support",

                        "left_dataset_id":
                            "sales",

                        "right_dataset_id":
                            "support",

                        "kind":
                            "join",

                        "left_keys": [
                            "customer_id",
                            "month",
                        ],

                        "right_keys": [
                            "customer_id",
                            "month",
                        ],

                        "validated":
                            True,
                    }
                ],

                "available_tools":
                    tools,
            }
        )
    )


# ============================================================
# 1. VALID SINGLE REQUIREMENT
# ============================================================

def test_valid_candidate() -> None:

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
                    }
                ],
            }
        )
    )


    assert (
        len(
            candidate.requirements
        )
        == 1
    )


    assert (
        candidate
        .requirements[
            0
        ]
        .dataset_ids
        == [
            "sales",
        ]
    )


    print(
        "Valid dependency candidate: PASS"
    )


# ============================================================
# 2. DUPLICATE DATASET IN GROUP REJECTED
# ============================================================

def test_duplicate_dataset_rejected() -> None:

    try:

        DatasetDependencyCandidate.model_validate(
            {
                "requirements": [
                    {
                        "requirement_id":
                            "bad",

                        "dataset_ids": [
                            "sales",
                            "sales",
                        ],
                    }
                ],
            }
        )


    except ValidationError:

        print(
            "Duplicate dataset dependency rejected: PASS"
        )


    else:

        raise AssertionError(
            "Duplicate dataset IDs must be rejected."
        )


# ============================================================
# 3. DUPLICATE REQUIREMENT ID REJECTED
# ============================================================

def test_duplicate_requirement_id_rejected() -> None:

    try:

        DatasetDependencyCandidate.model_validate(
            {
                "requirements": [
                    {
                        "requirement_id":
                            "same",

                        "dataset_ids": [
                            "sales",
                        ],
                    },

                    {
                        "requirement_id":
                            "same",

                        "dataset_ids": [
                            "support",
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
            "Duplicate requirement IDs must be rejected."
        )


# ============================================================
# 4. UNKNOWN DATASET REJECTED BY PYTHON
# ============================================================

def test_unknown_dataset_rejected() -> None:

    context = (
        make_context(
            with_join_tool=True,
        )
    )


    candidate = (
        DatasetDependencyCandidate
        .model_validate(
            {
                "requirements": [
                    {
                        "requirement_id":
                            "invented",

                        "dataset_ids": [
                            "unknown_dataset",
                        ],
                    }
                ],
            }
        )
    )


    try:

        evaluate_dataset_dependencies(
            candidate=candidate,
            context=context,
        )


    except ValueError as error:

        assert (
            "unknown_dataset"
            in str(
                error
            )
        )


        print(
            "Invented dataset rejected by Python: PASS"
        )


    else:

        raise AssertionError(
            "Python must reject invented dataset IDs."
        )


# ============================================================
# 5. SINGLE DATASET
# ============================================================

def test_single_dataset_executable() -> None:

    context = (
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
                            "sales_total",

                        "dataset_ids": [
                            "sales",
                        ],
                    }
                ],
            }
        )
    )


    result = (
        evaluate_dataset_dependencies(
            candidate=candidate,
            context=context,
        )
    )


    assert result.executable


    assert (
        result
        .requirements[
            0
        ]
        .feasibility
        == "not_required"
    )


    assert (
        result.routing_override_reason
        is None
    )


    print(
        "Single-dataset dependency: PASS"
    )


# ============================================================
# 6. INDEPENDENT DATASETS
# ============================================================

def test_independent_analyses_without_join() -> None:
    """
    Two datasets are needed by the user request,
    but never in the SAME analytical result.

    Therefore no combination capability is required.
    """

    context = (
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
        evaluate_dataset_dependencies(
            candidate=candidate,
            context=context,
        )
    )


    assert result.executable


    assert all(
        requirement.feasibility
        == "not_required"

        for requirement
        in result.requirements
    )


    print(
        "Independent dataset analyses: PASS"
    )


# ============================================================
# 7. COMBINATION REQUIRED BUT TOOL ABSENT
# ============================================================

def test_combination_tool_missing() -> None:

    context = (
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
                            "support_revenue_relationship",

                        "dataset_ids": [
                            "support",
                            "sales",
                        ],
                    }
                ],
            }
        )
    )


    result = (
        evaluate_dataset_dependencies(
            candidate=candidate,
            context=context,
        )
    )


    assert not result.executable


    assert (
        result
        .requirements[
            0
        ]
        .feasibility
        == "missing_combination_capability"
    )


    assert (
        result.routing_override_reason
        == "unsupported_analysis"
    )


    assert (
        result.blocking_requirements
        == [
            "support_revenue_relationship",
        ]
    )


    print(
        "Missing combination capability blocked: PASS"
    )


# ============================================================
# 8. VALID COMBINATION
# ============================================================

def test_valid_combination() -> None:

    context = (
        make_context(
            with_join_tool=True,
        )
    )


    candidate = (
        DatasetDependencyCandidate
        .model_validate(
            {
                "requirements": [
                    {
                        "requirement_id":
                            "support_revenue_relationship",

                        "dataset_ids": [
                            "support",
                            "sales",
                        ],
                    }
                ],
            }
        )
    )


    result = (
        evaluate_dataset_dependencies(
            candidate=candidate,
            context=context,
        )
    )


    assert result.executable


    assert (
        result
        .requirements[
            0
        ]
        .feasibility
        == "supported"
    )


    assert (
        result.routing_override_reason
        is None
    )


    print(
        "Validated cross-dataset dependency: PASS"
    )


# ============================================================
# 9. RELATIONSHIP PATH MISSING
# ============================================================

def test_relationship_path_missing() -> None:

    context = (
        make_context(
            with_join_tool=True,
        )
    )


    candidate = (
        DatasetDependencyCandidate
        .model_validate(
            {
                "requirements": [
                    {
                        "requirement_id":
                            "sales_machine_relationship",

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
        evaluate_dataset_dependencies(
            candidate=candidate,
            context=context,
        )
    )


    assert not result.executable


    assert (
        result
        .requirements[
            0
        ]
        .feasibility
        == "missing_validated_relationship"
    )


    assert (
        result.routing_override_reason
        == "unsupported_analysis"
    )


    print(
        "Missing relationship path blocked: PASS"
    )


# ============================================================
# 10. ONE BLOCKED REQUIREMENT BLOCKS WHOLE REQUEST
# ============================================================

def test_partial_request_blocked() -> None:

    context = (
        make_context(
            with_join_tool=True,
        )
    )


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
                            "sales_machine_relationship",

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
        evaluate_dataset_dependencies(
            candidate=candidate,
            context=context,
        )
    )


    assert not result.executable


    assert (
        result.blocking_requirements
        == [
            "sales_machine_relationship",
        ]
    )


    assert (
        result.routing_override_reason
        == "unsupported_analysis"
    )


    print(
        "Blocked sub-requirement blocks request: PASS"
    )


# ============================================================
# 11. COMPACT SUMMARY
# ============================================================

def test_summary() -> None:

    context = (
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
                            "relationship",

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
        evaluate_dataset_dependencies(
            candidate=candidate,
            context=context,
        )
    )


    summary = (
        dependency_gate_summary(
            result
        )
    )


    assert (
        summary[
            "executable"
        ]
        is False
    )


    assert (
        summary[
            "routing_override_reason"
        ]
        == "unsupported_analysis"
    )


    assert (
        summary[
            "requirements"
        ][
            0
        ][
            "feasibility"
        ]
        == "missing_combination_capability"
    )


    print(
        "Dependency gate summary: PASS"
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print(
        "=== DATALENS DATASET DEPENDENCY CONTRACT v0.8 ==="
    )


    print(
        "Contract:",
        DATASET_DEPENDENCY_CONTRACT_VERSION,
    )


    print(
        "Gate:",
        DATASET_DEPENDENCY_GATE_VERSION,
    )


    print()


    test_valid_candidate()

    test_duplicate_dataset_rejected()

    test_duplicate_requirement_id_rejected()

    test_unknown_dataset_rejected()

    test_single_dataset_executable()

    test_independent_analyses_without_join()

    test_combination_tool_missing()

    test_valid_combination()

    test_relationship_path_missing()

    test_partial_request_blocked()

    test_summary()


    print()

    print(
        "Dataset dependency contract v0.8: PASS"
    )


if __name__ == "__main__":
    main()