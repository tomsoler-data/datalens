from __future__ import annotations

from pydantic import (
    ValidationError,
)

from app.evals.routing_relationships_v0_8 import (
    ROUTING_RELATIONSHIP_CONTEXT_VERSION,
    RoutingRelationshipContext,
    build_relationship_graph,
    evaluate_cross_dataset_feasibility,
    has_combination_capability,
    has_validated_relationship_path,
)


# ============================================================
# DATASETS
# ============================================================

def make_datasets() -> list[dict]:

    return [

        # ====================================================
        # CUSTOMERS
        # ====================================================

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


        # ====================================================
        # SALES
        # ====================================================

        {
            "dataset_id":
                "sales",

            "filename":
                "sales.csv",

            "grain":
                "order",

            "entity_columns": [
                "order_id",
            ],

            "columns": [
                {
                    "name":
                        "order_id",

                    "analytical_type":
                        "identifier",
                },

                {
                    "name":
                        "customer_id",

                    "analytical_type":
                        "identifier",
                },

                {
                    "name":
                        "revenue",

                    "analytical_type":
                        "quantitative",
                },
            ],
        },


        # ====================================================
        # SUPPORT
        # ====================================================

        {
            "dataset_id":
                "support",

            "filename":
                "support.csv",

            "grain":
                "ticket",

            "entity_columns": [
                "ticket_id",
            ],

            "columns": [
                {
                    "name":
                        "ticket_id",

                    "analytical_type":
                        "identifier",
                },

                {
                    "name":
                        "customer_id",

                    "analytical_type":
                        "identifier",
                },

                {
                    "name":
                        "resolution_minutes",

                    "analytical_type":
                        "quantitative",
                },
            ],
        },


        # ====================================================
        # MACHINES
        #
        # Deliberately disconnected.
        # ====================================================

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
    ]


# ============================================================
# RELATIONSHIPS
# ============================================================

def make_relationships() -> list[dict]:

    return [

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
    ]


# ============================================================
# CONTEXT
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
                "datasets":
                    make_datasets(),

                "relationships":
                    make_relationships(),

                "available_tools":
                    tools,
            }
        )
    )


# ============================================================
# 1. VALID CONTEXT
# ============================================================

def test_valid_context() -> None:

    context = (
        make_context(
            with_join_tool=True,
        )
    )


    assert (
        len(
            context.datasets
        )
        == 4
    )


    assert (
        len(
            context.relationships
        )
        == 2
    )


    print(
        "Valid relationship context: PASS"
    )


# ============================================================
# 2. UNKNOWN DATASET REJECTED
# ============================================================

def test_unknown_dataset_rejected() -> None:

    relationships = (
        make_relationships()
    )


    relationships.append(
        {
            "relationship_id":
                "bad_relation",

            "left_dataset_id":
                "sales",

            "right_dataset_id":
                "unknown_dataset",

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
        }
    )


    try:

        RoutingRelationshipContext.model_validate(
            {
                "datasets":
                    make_datasets(),

                "relationships":
                    relationships,

                "available_tools": [
                    "join_datasets",
                ],
            }
        )


    except ValidationError:

        print(
            "Unknown dataset rejected: PASS"
        )


    else:

        raise AssertionError(
            "Unknown relationship dataset must be rejected."
        )


# ============================================================
# 3. UNKNOWN KEY REJECTED
# ============================================================

def test_unknown_key_rejected() -> None:

    try:

        RoutingRelationshipContext.model_validate(
            {
                "datasets":
                    make_datasets(),

                "relationships": [
                    {
                        "relationship_id":
                            "bad_key",

                        "left_dataset_id":
                            "customers",

                        "right_dataset_id":
                            "sales",

                        "kind":
                            "join",

                        "left_keys": [
                            "missing_customer_key",
                        ],

                        "right_keys": [
                            "customer_id",
                        ],

                        "validated":
                            True,
                    }
                ],

                "available_tools": [
                    "join_datasets",
                ],
            }
        )


    except ValidationError:

        print(
            "Unknown relationship key rejected: PASS"
        )


    else:

        raise AssertionError(
            "Unknown relationship keys must be rejected."
        )


# ============================================================
# 4. KEY ARITY REJECTED
# ============================================================

def test_key_arity_rejected() -> None:

    try:

        RoutingRelationshipContext.model_validate(
            {
                "datasets":
                    make_datasets(),

                "relationships": [
                    {
                        "relationship_id":
                            "bad_arity",

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
                            "order_id",
                        ],

                        "validated":
                            True,
                    }
                ],

                "available_tools": [
                    "join_datasets",
                ],
            }
        )


    except ValidationError:

        print(
            "Relationship key arity rejected: PASS"
        )


    else:

        raise AssertionError(
            "Relationship key arity mismatch "
            "must be rejected."
        )


# ============================================================
# 5. GRAPH
# ============================================================

def test_relationship_graph() -> None:

    context = (
        make_context(
            with_join_tool=True,
        )
    )


    graph = (
        build_relationship_graph(
            context
        )
    )


    assert (
        graph[
            "customers"
        ]
        == {
            "sales",
            "support",
        }
    )


    assert (
        graph[
            "machines"
        ]
        == set()
    )


    print(
        "Relationship graph: PASS"
    )


# ============================================================
# 6. SINGLE DATASET
# ============================================================

def test_single_dataset_not_required() -> None:

    context = (
        make_context(
            with_join_tool=False,
        )
    )


    result = (
        evaluate_cross_dataset_feasibility(
            context=context,

            required_dataset_ids=[
                "sales",
            ],
        )
    )


    assert (
        result
        == "not_required"
    )


    print(
        "Single-dataset feasibility: PASS"
    )


# ============================================================
# 7. NO COMBINATION CAPABILITY
# ============================================================

def test_missing_combination_capability() -> None:

    context = (
        make_context(
            with_join_tool=False,
        )
    )


    assert not (
        has_combination_capability(
            context=context,
        )
    )


    result = (
        evaluate_cross_dataset_feasibility(
            context=context,

            required_dataset_ids=[
                "sales",
                "support",
            ],
        )
    )


    assert (
        result
        == "missing_combination_capability"
    )


    print(
        "Missing combination capability: PASS"
    )


# ============================================================
# 8. DIRECT VALIDATED PATH
# ============================================================

def test_direct_validated_path() -> None:

    context = (
        make_context(
            with_join_tool=True,
        )
    )


    assert (
        has_validated_relationship_path(
            context=context,

            required_dataset_ids=[
                "customers",
                "sales",
            ],
        )
    )


    result = (
        evaluate_cross_dataset_feasibility(
            context=context,

            required_dataset_ids=[
                "customers",
                "sales",
            ],
        )
    )


    assert (
        result
        == "supported"
    )


    print(
        "Direct validated relationship: PASS"
    )


# ============================================================
# 9. MULTI-HOP VALIDATED PATH
# ============================================================

def test_multihop_validated_path() -> None:

    context = (
        make_context(
            with_join_tool=True,
        )
    )


    # --------------------------------------------------------
    # sales -> customers -> support
    # --------------------------------------------------------

    assert (
        has_validated_relationship_path(
            context=context,

            required_dataset_ids=[
                "sales",
                "support",
            ],
        )
    )


    result = (
        evaluate_cross_dataset_feasibility(
            context=context,

            required_dataset_ids=[
                "sales",
                "support",
            ],
        )
    )


    assert (
        result
        == "supported"
    )


    print(
        "Multi-hop relationship path: PASS"
    )


# ============================================================
# 10. NO VALIDATED PATH
# ============================================================

def test_missing_validated_relationship() -> None:

    context = (
        make_context(
            with_join_tool=True,
        )
    )


    assert not (
        has_validated_relationship_path(
            context=context,

            required_dataset_ids=[
                "sales",
                "machines",
            ],
        )
    )


    result = (
        evaluate_cross_dataset_feasibility(
            context=context,

            required_dataset_ids=[
                "sales",
                "machines",
            ],
        )
    )


    assert (
        result
        == "missing_validated_relationship"
    )


    print(
        "Missing validated relationship: PASS"
    )


# ============================================================
# 11. THREE REQUIRED DATASETS
# ============================================================

def test_three_dataset_component() -> None:

    context = (
        make_context(
            with_join_tool=True,
        )
    )


    result = (
        evaluate_cross_dataset_feasibility(
            context=context,

            required_dataset_ids=[
                "customers",
                "sales",
                "support",
            ],
        )
    )


    assert (
        result
        == "supported"
    )


    print(
        "Three-dataset relationship component: PASS"
    )


# ============================================================
# 12. DISCONNECTED THIRD DATASET
# ============================================================

def test_disconnected_third_dataset() -> None:

    context = (
        make_context(
            with_join_tool=True,
        )
    )


    result = (
        evaluate_cross_dataset_feasibility(
            context=context,

            required_dataset_ids=[
                "customers",
                "sales",
                "machines",
            ],
        )
    )


    assert (
        result
        == "missing_validated_relationship"
    )


    print(
        "Disconnected required dataset detected: PASS"
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print(
        "=== DATALENS ROUTING RELATIONSHIPS v0.8 ==="
    )


    print(
        "Context:",
        ROUTING_RELATIONSHIP_CONTEXT_VERSION,
    )


    print()


    test_valid_context()

    test_unknown_dataset_rejected()

    test_unknown_key_rejected()

    test_key_arity_rejected()

    test_relationship_graph()

    test_single_dataset_not_required()

    test_missing_combination_capability()

    test_direct_validated_path()

    test_multihop_validated_path()

    test_missing_validated_relationship()

    test_three_dataset_component()

    test_disconnected_third_dataset()


    print()

    print(
        "Routing relationship context v0.8: PASS"
    )


if __name__ == "__main__":
    main()