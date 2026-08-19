from __future__ import annotations

import json

from types import SimpleNamespace

from app.ai.analytical_reasoning_pipeline_v1 import (
    ANALYTICAL_REASONING_PIPELINE_VERSION,
    analytical_reasoning_pipeline_summary,
    require_ready_analytical_reasoning,
    run_analytical_reasoning_pipeline,
)

from app.planning.analytical_v1.dataset_context import (
    DatasetColumnSpec,
    DatasetContext,
)

from app.planning.analytical_v1.relationships import (
    DatasetRelationshipSpec,
    RoutingRelationshipContext,
)


# ============================================================
# FAKE OLLAMA CLIENT
# ============================================================

class FakeChatClient:

    def __init__(
        self,
        content: str,
    ) -> None:

        self.content = (
            content
        )

        self.calls: list[
            dict
        ] = []


    def chat(
        self,
        **kwargs,
    ):

        self.calls.append(
            kwargs
        )


        return (
            SimpleNamespace(
                message=(
                    SimpleNamespace(
                        content=(
                            self.content
                        )
                    )
                )
            )
        )


# ============================================================
# DATASET HELPERS
# ============================================================

def column(
    name: str,
    analytical_type: str,
) -> DatasetColumnSpec:

    return (
        DatasetColumnSpec(
            name=(
                name
            ),

            analytical_type=(
                analytical_type
            ),

            semantic_role=None,
        )
    )


def build_sales_dataset() -> DatasetContext:

    return (
        DatasetContext(
            dataset_id=(
                "sales"
            ),

            filename=(
                "sales.csv"
            ),

            grain=(
                "order"
            ),

            entity_columns=[
                "customer_id",
            ],

            columns=[
                column(
                    "order_id",
                    "identifier",
                ),

                column(
                    "customer_id",
                    "identifier",
                ),

                column(
                    "revenue",
                    "quantitative",
                ),
            ],
        )
    )


def build_fuel_dataset() -> DatasetContext:

    return (
        DatasetContext(
            dataset_id=(
                "fuel"
            ),

            filename=(
                "fuel.csv"
            ),

            grain=(
                "vehicle"
            ),

            entity_columns=[
                "vehicle_id",
            ],

            columns=[
                column(
                    "vehicle_id",
                    "identifier",
                ),

                column(
                    "fuel_consumption",
                    "quantitative",
                ),
            ],
        )
    )


def build_maintenance_dataset() -> DatasetContext:

    return (
        DatasetContext(
            dataset_id=(
                "maintenance"
            ),

            filename=(
                "maintenance.csv"
            ),

            grain=(
                "vehicle"
            ),

            entity_columns=[
                "vehicle_id",
            ],

            columns=[
                column(
                    "vehicle_id",
                    "identifier",
                ),

                column(
                    "maintenance_cost",
                    "quantitative",
                ),
            ],
        )
    )


# ============================================================
# STRUCTURAL CONTEXTS
# ============================================================

def build_single_dataset_context() -> RoutingRelationshipContext:

    return (
        RoutingRelationshipContext(
            datasets=[
                build_sales_dataset(),
            ],

            relationships=[],

            available_tools=[
                "join_datasets",
                "aggregate",
                "analyze_distribution",
                "detect_outliers",
                "measure_association",
            ],
        )
    )


def build_connected_context() -> RoutingRelationshipContext:

    relationship = (
        DatasetRelationshipSpec(
            relationship_id=(
                "fuel_maintenance_vehicle"
            ),

            left_dataset_id=(
                "fuel"
            ),

            right_dataset_id=(
                "maintenance"
            ),

            kind=(
                "join"
            ),

            left_keys=[
                "vehicle_id",
            ],

            right_keys=[
                "vehicle_id",
            ],

            validated=True,
        )
    )


    return (
        RoutingRelationshipContext(
            datasets=[
                build_fuel_dataset(),
                build_maintenance_dataset(),
            ],

            relationships=[
                relationship,
            ],

            available_tools=[
                "join_datasets",
                "aggregate",
                "measure_association",
                "analyze_distribution",
                "detect_outliers",
            ],
        )
    )


def build_unconnected_context() -> RoutingRelationshipContext:

    return (
        RoutingRelationshipContext(
            datasets=[
                build_fuel_dataset(),
                build_maintenance_dataset(),
            ],

            relationships=[],

            available_tools=[
                "join_datasets",
                "aggregate",
                "measure_association",
                "analyze_distribution",
                "detect_outliers",
            ],
        )
    )


# ============================================================
# CANDIDATES
# ============================================================

def single_dependency_candidate() -> str:

    return (
        json.dumps(
            {
                "requirements": [
                    {
                        "requirement_id":
                            "revenue_total",

                        "dataset_ids": [
                            "sales",
                        ],
                    }
                ]
            }
        )
    )


def single_planner_candidate() -> str:

    return (
        json.dumps(
            {
                "plans": [
                    {
                        "requirement_id":
                            "revenue_total",

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
                ]
            }
        )
    )


def cross_dependency_candidate() -> str:

    return (
        json.dumps(
            {
                "requirements": [
                    {
                        "requirement_id":
                            "fuel_vs_maintenance",

                        "dataset_ids": [
                            "fuel",
                            "maintenance",
                        ],
                    }
                ]
            }
        )
    )


def cross_planner_candidate() -> str:

    return (
        json.dumps(
            {
                "plans": [
                    {
                        "requirement_id":
                            "fuel_vs_maintenance",

                        "intent":
                            "measure_relationship",

                        "family":
                            "association",

                        "target_grain":
                            "vehicle",

                        "steps": [
                            {
                                "step_id":
                                    "measure_relationship",

                                "action": {
                                    "name":
                                        "measure_association",

                                    "target":
                                        (
                                            "fuel."
                                            "fuel_consumption"
                                        ),

                                    "value":
                                        (
                                            "maintenance."
                                            "maintenance_cost"
                                        ),
                                },
                            }
                        ],
                    }
                ]
            }
        )
    )


def unsafe_planner_candidate() -> str:

    return (
        json.dumps(
            {
                "plans": [
                    {
                        "requirement_id":
                            "revenue_total",

                        "intent":
                            "aggregate_metric",

                        "family":
                            "aggregation",

                        "target_grain":
                            "global",

                        "steps": [
                            {
                                "step_id":
                                    "bad_aggregate",

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
                ]
            }
        )
    )


# ============================================================
# 1. SINGLE DATASET → READY
# ============================================================

def test_single_dataset_ready() -> None:

    dependency_client = (
        FakeChatClient(
            single_dependency_candidate()
        )
    )


    planner_client = (
        FakeChatClient(
            single_planner_candidate()
        )
    )


    result = (
        run_analytical_reasoning_pipeline(
            user_request=(
                "Quel est le chiffre d'affaires total ?"
            ),

            structural_context=(
                build_single_dataset_context()
            ),

            dependency_chat_client=(
                dependency_client
            ),

            planner_chat_client=(
                planner_client
            ),
        )
    )


    assert (
        result.pipeline_version
        == ANALYTICAL_REASONING_PIPELINE_VERSION
    )


    assert (
        result.status
        == "ready"
    )


    assert (
        result.ready_for_execution
    )


    assert (
        result.dependency_gate
        is not None
    )


    assert (
        result.dependency_gate.executable
    )


    assert (
        result.planner_context
        is not None
    )


    assert (
        result.planner_context.ready_for_planning
    )


    assert (
        result.planner_input
        is not None
    )


    assert (
        result.planner
        is not None
    )


    assert (
        result.planner.execution_candidate
        is not None
    )


    assert (
        len(
            dependency_client.calls
        )
        == 1
    )


    assert (
        len(
            planner_client.calls
        )
        == 1
    )


    print(
        "Single-dataset reasoning pipeline reaches READY: PASS"
    )


# ============================================================
# 2. VALIDATED CROSS-DATASET PATH → READY
# ============================================================

def test_connected_cross_dataset_ready() -> None:

    dependency_client = (
        FakeChatClient(
            cross_dependency_candidate()
        )
    )


    planner_client = (
        FakeChatClient(
            cross_planner_candidate()
        )
    )


    result = (
        run_analytical_reasoning_pipeline(
            user_request=(
                "La consommation de carburant est-elle "
                "associée au coût de maintenance ?"
            ),

            structural_context=(
                build_connected_context()
            ),

            dependency_chat_client=(
                dependency_client
            ),

            planner_chat_client=(
                planner_client
            ),
        )
    )


    assert (
        result.status
        == "ready"
    )


    assert (
        result.ready_for_execution
    )


    assert (
        result.dependency_gate
        is not None
    )


    assert (
        result.dependency_gate.executable
    )


    assert (
        result.planner_context
        is not None
    )


    assert (
        result.planner_input
        is not None
    )


    # join_datasets is a structural capability.
    # It must not become an analytical planner action.
    for requirement in (
        result.planner_input.requirements
    ):

        assert (
            "join_datasets"
            not in (
                requirement
                .allowed_analytical_tools
            )
        )


    assert (
        result.planner
        is not None
    )


    assert (
        result.planner.ready_for_execution
    )


    print(
        "Validated cross-dataset reasoning path reaches "
        "READY: PASS"
    )


# ============================================================
# 3. NO VALIDATED RELATIONSHIP → STOP BEFORE PLANNER
# ============================================================

def test_missing_relationship_blocks_before_planner() -> None:

    dependency_client = (
        FakeChatClient(
            cross_dependency_candidate()
        )
    )


    planner_client = (
        FakeChatClient(
            cross_planner_candidate()
        )
    )


    result = (
        run_analytical_reasoning_pipeline(
            user_request=(
                "La consommation de carburant est-elle "
                "associée au coût de maintenance ?"
            ),

            structural_context=(
                build_unconnected_context()
            ),

            dependency_chat_client=(
                dependency_client
            ),

            planner_chat_client=(
                planner_client
            ),
        )
    )


    assert (
        result.status
        == "structurally_blocked"
    )


    assert not (
        result.ready_for_execution
    )


    assert (
        result.dependency_gate
        is not None
    )


    assert not (
        result.dependency_gate.executable
    )


    assert (
        result.planner_context
        is None
    )


    assert (
        result.planner_input
        is None
    )


    assert (
        result.planner
        is None
    )


    assert (
        len(
            dependency_client.calls
        )
        == 1
    )


    # Critical security property:
    # no planner inference occurs after structural blocking.
    assert (
        len(
            planner_client.calls
        )
        == 0
    )


    feasibility_values = [
        requirement.feasibility

        for requirement
        in (
            result
            .dependency_gate
            .requirements
        )
    ]


    assert (
        "missing_validated_relationship"
        in feasibility_values
    )


    print(
        "Missing validated relationship blocks BEFORE "
        "planner invocation: PASS"
    )


# ============================================================
# 4. HALLUCINATED DATASET → STOP BEFORE GATE/PLANNER
# ============================================================

def test_hallucinated_dataset_stops_early() -> None:

    dependency_client = (
        FakeChatClient(
            json.dumps(
                {
                    "requirements": [
                        {
                            "requirement_id":
                                "invented_source",

                            "dataset_ids": [
                                "sales",
                                "secret_crm",
                            ],
                        }
                    ]
                }
            )
        )
    )


    planner_client = (
        FakeChatClient(
            single_planner_candidate()
        )
    )


    result = (
        run_analytical_reasoning_pipeline(
            user_request=(
                "Analyse les ventes avec le CRM secret."
            ),

            structural_context=(
                build_single_dataset_context()
            ),

            dependency_chat_client=(
                dependency_client
            ),

            planner_chat_client=(
                planner_client
            ),
        )
    )


    assert (
        result.status
        == "dependency_invalid_candidate"
    )


    assert not (
        result.ready_for_execution
    )


    assert (
        result.dependency_gate
        is None
    )


    assert (
        result.planner
        is None
    )


    assert (
        len(
            planner_client.calls
        )
        == 0
    )


    print(
        "Hallucinated dependency stops before structural "
        "planning: PASS"
    )


# ============================================================
# 5. DEPENDENCY GENERATION ERROR → STOP
# ============================================================

def test_dependency_generation_error_stops_pipeline() -> None:

    dependency_client = (
        FakeChatClient(
            "not-json"
        )
    )


    planner_client = (
        FakeChatClient(
            single_planner_candidate()
        )
    )


    result = (
        run_analytical_reasoning_pipeline(
            user_request=(
                "Quel est le chiffre d'affaires total ?"
            ),

            structural_context=(
                build_single_dataset_context()
            ),

            dependency_chat_client=(
                dependency_client
            ),

            planner_chat_client=(
                planner_client
            ),
        )
    )


    assert (
        result.status
        == "dependency_generation_error"
    )


    assert (
        result.planner
        is None
    )


    assert (
        len(
            planner_client.calls
        )
        == 0
    )


    print(
        "Dependency generation error stops pipeline: PASS"
    )


# ============================================================
# 6. PLANNER SAFETY BLOCK
# ============================================================

def test_planner_safety_block() -> None:

    dependency_client = (
        FakeChatClient(
            single_dependency_candidate()
        )
    )


    planner_client = (
        FakeChatClient(
            unsafe_planner_candidate()
        )
    )


    result = (
        run_analytical_reasoning_pipeline(
            user_request=(
                "Quel est le chiffre d'affaires total ?"
            ),

            structural_context=(
                build_single_dataset_context()
            ),

            dependency_chat_client=(
                dependency_client
            ),

            planner_chat_client=(
                planner_client
            ),
        )
    )


    assert (
        result.status
        == "planner_blocked"
    )


    assert not (
        result.ready_for_execution
    )


    assert (
        result.planner
        is not None
    )


    assert (
        result.planner.safety
        is not None
    )


    assert (
        result.planner.safety.blocking_stage
        == "reference_canonicalization"
    )


    assert (
        "unknown_reference"
        in (
            result
            .planner
            .safety
            .blocking_codes
        )
    )


    assert (
        result.planner.execution_candidate
        is None
    )


    print(
        "Unsafe planner candidate remains BLOCKED: PASS"
    )


# ============================================================
# 7. FINAL EXECUTION GUARD
# ============================================================

def test_final_execution_guard() -> None:

    dependency_client = (
        FakeChatClient(
            single_dependency_candidate()
        )
    )


    planner_client = (
        FakeChatClient(
            single_planner_candidate()
        )
    )


    candidate = (
        require_ready_analytical_reasoning(
            user_request=(
                "Quel est le chiffre d'affaires total ?"
            ),

            structural_context=(
                build_single_dataset_context()
            ),

            dependency_chat_client=(
                dependency_client
            ),

            planner_chat_client=(
                planner_client
            ),
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
        .metrics
        == [
            "sales.revenue",
        ]
    )


    print(
        "Final execution guard returns only canonical "
        "READY plan: PASS"
    )


# ============================================================
# 8. OBSERVABILITY
# ============================================================

def test_observability_summary() -> None:

    result = (
        run_analytical_reasoning_pipeline(
            user_request=(
                "Quel est le chiffre d'affaires total ?"
            ),

            structural_context=(
                build_single_dataset_context()
            ),

            dependency_chat_client=(
                FakeChatClient(
                    single_dependency_candidate()
                )
            ),

            planner_chat_client=(
                FakeChatClient(
                    single_planner_candidate()
                )
            ),
        )
    )


    summary = (
        analytical_reasoning_pipeline_summary(
            result
        )
    )


    assert (
        summary[
            "pipeline_version"
        ]
        == ANALYTICAL_REASONING_PIPELINE_VERSION
    )


    assert (
        summary[
            "status"
        ]
        == "ready"
    )


    assert (
        summary[
            "ready_for_execution"
        ]
        is True
    )


    assert (
        summary[
            "dependency"
        ][
            "status"
        ]
        == "valid"
    )


    assert (
        summary[
            "dependency_gate"
        ][
            "executable"
        ]
        is True
    )


    assert (
        summary[
            "planner"
        ][
            "status"
        ]
        == "ready"
    )


    print(
        "Reasoning pipeline observability summary: PASS"
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print(
        "=== DATALENS ANALYTICAL REASONING PIPELINE v1.0 ==="
    )


    print()


    test_single_dataset_ready()

    test_connected_cross_dataset_ready()

    test_missing_relationship_blocks_before_planner()

    test_hallucinated_dataset_stops_early()

    test_dependency_generation_error_stops_pipeline()

    test_planner_safety_block()

    test_final_execution_guard()

    test_observability_summary()


    print()


    print(
        "NO OLLAMA INFERENCE WAS PERFORMED."
    )


    print()


    print(
        "Analytical Reasoning Pipeline v1.0: PASS"
    )


if __name__ == "__main__":
    main()