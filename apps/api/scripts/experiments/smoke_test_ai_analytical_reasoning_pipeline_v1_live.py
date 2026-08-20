from __future__ import annotations

import json

from app.ai.analytical_reasoning_pipeline_v1 import (
    ANALYTICAL_REASONING_PIPELINE_VERSION,
    analytical_reasoning_pipeline_summary,
    run_analytical_reasoning_pipeline,
)

from app.planning.analytical_v1.dataset_context import (
    DatasetColumnSpec,
    DatasetContext,
)

from app.planning.analytical_v1.dependency import (
    dependency_gate_summary,
)

from app.planning.analytical_v1.relationships import (
    DatasetRelationshipSpec,
    RoutingRelationshipContext,
)

from app.planning.analytical_v1.safety import (
    analytical_planner_safety_summary,
)


# ============================================================
# LIVE SMOKE TEST
# ============================================================

USER_REQUEST = (
    "La consommation d'électricité des entrepôts "
    "est-elle associée à leur coût annuel de maintenance ?"
)


# ============================================================
# DISPLAY
# ============================================================

def print_json(
    title: str,
    payload,
) -> None:

    print()
    print(
        title
    )

    print(
        "-" * len(
            title
        )
    )

    print(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    )


# ============================================================
# COLUMN HELPERS
# ============================================================

def identifier_column(
    name: str,
) -> DatasetColumnSpec:

    return (
        DatasetColumnSpec(
            name=(
                name
            ),

            analytical_type=(
                "identifier"
            ),

            semantic_role=None,
        )
    )


def quantitative_column(
    name: str,
) -> DatasetColumnSpec:

    return (
        DatasetColumnSpec(
            name=(
                name
            ),

            analytical_type=(
                "quantitative"
            ),

            semantic_role=None,
        )
    )


# ============================================================
# PRODUCTION-LIKE STRUCTURAL CONTEXT
# ============================================================

def build_structural_context(
) -> RoutingRelationshipContext:
    """
    Synthetic production-like metadata.

    Important:
    the relationship below represents a relationship that
    DataLens Preparation would already have validated.

    The AI is not allowed to invent this relationship.
    """

    warehouse_energy = (
        DatasetContext(
            dataset_id=(
                "warehouse_energy"
            ),

            filename=(
                "warehouse_energy.csv"
            ),

            grain=(
                "warehouse"
            ),

            entity_columns=[
                "warehouse_id",
            ],

            columns=[
                identifier_column(
                    "warehouse_id"
                ),

                quantitative_column(
                    "electricity_consumption_kwh"
                ),

                quantitative_column(
                    "surface_m2"
                ),
            ],
        )
    )


    warehouse_maintenance = (
        DatasetContext(
            dataset_id=(
                "warehouse_maintenance"
            ),

            filename=(
                "warehouse_maintenance.csv"
            ),

            grain=(
                "warehouse"
            ),

            entity_columns=[
                "warehouse_id",
            ],

            columns=[
                identifier_column(
                    "warehouse_id"
                ),

                quantitative_column(
                    "annual_maintenance_cost"
                ),

                quantitative_column(
                    "maintenance_intervention_count"
                ),
            ],
        )
    )


    validated_relationship = (
        DatasetRelationshipSpec(
            relationship_id=(
                "warehouse_energy_maintenance_by_warehouse"
            ),

            left_dataset_id=(
                "warehouse_energy"
            ),

            right_dataset_id=(
                "warehouse_maintenance"
            ),

            kind=(
                "join"
            ),

            left_keys=[
                "warehouse_id",
            ],

            right_keys=[
                "warehouse_id",
            ],

            validated=True,
        )
    )


    return (
        RoutingRelationshipContext(
            datasets=[
                warehouse_energy,
                warehouse_maintenance,
            ],

            relationships=[
                validated_relationship,
            ],

            available_tools=[
                "join_datasets",
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
        )
    )


# ============================================================
# LIVE RUN
# ============================================================

def main() -> None:

    print(
        "=== DATALENS ANALYTICAL REASONING PIPELINE v1 "
        "— LIVE TWO-STAGE AI SMOKE TEST ==="
    )

    print()

    print(
        "WARNING:"
    )

    print(
        "This script may perform TWO real local "
        "Ollama inferences."
    )

    print(
        "Qwen #1 = Dataset Dependency Extractor"
    )

    print(
        "Qwen #2 = Analytical Planner"
    )

    print()

    print(
        "No benchmark score will be computed."
    )

    print(
        "No Frozen case will be used."
    )

    print(
        "No analytical tool will be executed."
    )

    print()


    structural_context = (
        build_structural_context()
    )


    print(
        "Pipeline:",
        ANALYTICAL_REASONING_PIPELINE_VERSION,
    )

    print(
        "Request:",
        USER_REQUEST,
    )

    print(
        "Datasets:",
        [
            dataset.dataset_id

            for dataset
            in structural_context.datasets
        ],
    )

    print(
        "Validated relationships:",
        [
            relationship.relationship_id

            for relationship
            in structural_context.relationships
        ],
    )


    # ========================================================
    # RUN COMPLETE REASONING PIPELINE
    # ========================================================

    print()

    print(
        "Running real multi-stage reasoning..."
    )

    print(
        "1/2 — semantic dependency extraction"
    )

    print(
        "2/2 — analytical planning if Python authorizes it"
    )


    result = (
        run_analytical_reasoning_pipeline(
            user_request=(
                USER_REQUEST
            ),

            structural_context=(
                structural_context
            ),
        )
    )


    # ========================================================
    # GLOBAL RESULT
    # ========================================================

    print()

    print(
        "=== GLOBAL RESULT ==="
    )

    print(
        "Status:",
        result.status,
    )

    print(
        "Ready for execution:",
        result.ready_for_execution,
    )


    print_json(
        "PIPELINE SUMMARY",
        analytical_reasoning_pipeline_summary(
            result
        ),
    )


    # ========================================================
    # STAGE 1 — DEPENDENCY EXTRACTION
    # ========================================================

    print()

    print(
        "=== STAGE 1 — DATASET DEPENDENCIES ==="
    )

    print(
        "Status:",
        result.dependency.status,
    )

    print(
        "Inference ms:",
        round(
            result.dependency.inference_ms,
            1,
        ),
    )


    if (
        result.dependency.candidate
        is not None
    ):

        print_json(
            "DEPENDENCY CANDIDATE",
            (
                result
                .dependency
                .candidate
                .model_dump(
                    mode="json"
                )
            ),
        )

    else:

        print()

        print(
            "DEPENDENCY CANDIDATE: NONE"
        )


    if (
        result.dependency.error
        is not None
    ):

        print()

        print(
            "DEPENDENCY ERROR:"
        )

        print(
            result.dependency.error
        )


    # ========================================================
    # STAGE 2 — STRUCTURAL GATE
    # ========================================================

    print()

    print(
        "=== STAGE 2 — PYTHON STRUCTURAL GATE ==="
    )


    if (
        result.dependency_gate
        is not None
    ):

        print_json(
            "DEPENDENCY GATE",
            dependency_gate_summary(
                result.dependency_gate
            ),
        )

    else:

        print(
            "DEPENDENCY GATE: NOT REACHED"
        )


    # ========================================================
    # STAGE 3 — PLANNER CONTEXT
    # ========================================================

    print()

    print(
        "=== STAGE 3 — TRUSTED PLANNER CONTEXT ==="
    )


    if (
        result.planner_context
        is not None
    ):

        print(
            "Ready for planning:",
            result.planner_context.ready_for_planning,
        )

        print_json(
            "PLANNER CONTEXT",
            result.planner_context.model_dump(
                mode="json"
            ),
        )

    else:

        print(
            "PLANNER CONTEXT: NOT CREATED"
        )


    # ========================================================
    # STAGE 4 — MODEL-VISIBLE PLANNER INPUT
    # ========================================================

    print()

    print(
        "=== STAGE 4 — ANALYTICAL PLANNER INPUT ==="
    )


    if (
        result.planner_input
        is not None
    ):

        planner_payload = (
            result
            .planner_input
            .model_dump(
                mode="json"
            )
        )


        print_json(
            "ANALYTICAL PLANNER INPUT",
            planner_payload,
        )


        # ----------------------------------------------------
        # SECURITY ASSERTION:
        # join_datasets is structural and must never become
        # an analytical action available to Qwen Planner.
        # ----------------------------------------------------

        for requirement in (
            result.planner_input.requirements
        ):

            if (
                "join_datasets"
                in requirement.allowed_analytical_tools
            ):

                raise RuntimeError(
                    "SECURITY FAILURE: join_datasets leaked "
                    "into Analytical Planner tools."
                )


        print()

        print(
            "Structural join tool hidden from "
            "Analytical Planner: PASS"
        )

    else:

        print(
            "ANALYTICAL PLANNER INPUT: NOT CREATED"
        )


    # ========================================================
    # STAGE 5 — ANALYTICAL PLANNER
    # ========================================================

    print()

    print(
        "=== STAGE 5 — ANALYTICAL PLANNER ==="
    )


    if (
        result.planner
        is not None
    ):

        print(
            "Status:",
            result.planner.status,
        )

        print(
            "Inference ms:",
            round(
                result.planner.inference_ms,
                1,
            ),
        )


        if (
            result.planner.raw_candidate
            is not None
        ):

            print_json(
                "RAW ANALYTICAL PLAN",
                (
                    result
                    .planner
                    .raw_candidate
                    .model_dump(
                        mode="json"
                    )
                ),
            )


        if (
            result.planner.safety
            is not None
        ):

            print_json(
                "ANALYTICAL PLAN SAFETY",
                analytical_planner_safety_summary(
                    result.planner.safety
                ),
            )


        if (
            result.planner.execution_candidate
            is not None
        ):

            print_json(
                "AUTHORIZED EXECUTION CANDIDATE",
                (
                    result
                    .planner
                    .execution_candidate
                    .model_dump(
                        mode="json"
                    )
                ),
            )

        else:

            print()

            print(
                "AUTHORIZED EXECUTION CANDIDATE: NONE"
            )


        if (
            result.planner.error
            is not None
        ):

            print()

            print(
                "PLANNER ERROR:"
            )

            print(
                result.planner.error
            )

    else:

        print(
            "ANALYTICAL PLANNER: NOT INVOKED"
        )


    # ========================================================
    # FINAL INTERPRETATION
    # ========================================================

    print()

    print(
        "=== INTERPRETATION ==="
    )


    if (
        result.status
        == "ready"
    ):

        if (
            result.planner
            is None
        ):

            raise RuntimeError(
                "READY pipeline unexpectedly has no planner."
            )


        if (
            result.planner.execution_candidate
            is None
        ):

            raise RuntimeError(
                "READY pipeline unexpectedly has no "
                "execution candidate."
            )


        print(
            "PASS — the real two-stage AI reasoning chain "
            "crossed every deterministic safety boundary."
        )

        print()

        print(
            "Qwen #1 identified semantic dataset "
            "dependencies."
        )

        print(
            "Python independently authorized the validated "
            "relationship path."
        )

        print(
            "Qwen #2 produced the analytical plan."
        )

        print(
            "Python independently authorized the final "
            "canonical plan."
        )


    elif (
        result.status
        == "structurally_blocked"
    ):

        print(
            "SAFE BLOCK — semantic dependencies were "
            "identified, but Python refused structural "
            "planning."
        )

        raise SystemExit(
            2
        )


    elif (
        result.status
        in {
            "dependency_generation_error",
            "planner_generation_error",
        }
    ):

        print(
            "GENERATION ERROR — inspect the corresponding "
            "real Ollama stage."
        )

        raise SystemExit(
            1
        )


    else:

        print(
            "SAFE BLOCK — an AI candidate was rejected by "
            "a deterministic validation boundary."
        )

        raise SystemExit(
            2
        )


    # ========================================================
    # TIMING
    # ========================================================

    dependency_ms = (
        result.dependency.inference_ms
    )


    planner_ms = (
        result.planner.inference_ms

        if (
            result.planner
            is not None
        )

        else 0.0
    )


    print()

    print(
        "Dependency inference ms:",
        round(
            dependency_ms,
            1,
        ),
    )

    print(
        "Planner inference ms:",
        round(
            planner_ms,
            1,
        ),
    )

    print(
        "Total model inference ms:",
        round(
            dependency_ms
            + planner_ms,
            1,
        ),
    )


    print()

    print(
        "No benchmark score was computed."
    )

    print(
        "No Frozen case was used."
    )

    print(
        "No evaluation result file was written."
    )

    print(
        "No analytical tool was executed."
    )

    print()

    print(
        "Analytical Reasoning Pipeline v1 "
        "live two-stage smoke test: PASS"
    )


if __name__ == "__main__":
    main()