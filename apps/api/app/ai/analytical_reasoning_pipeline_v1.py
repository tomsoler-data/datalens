from __future__ import annotations

from typing import (
    Any,
    Literal,
)

from pydantic import (
    BaseModel,
    ConfigDict,
)

from app.ai.analytical_planner_v1 import (
    AIAnalyticalPlannerResult,
    generate_analytical_plan,
)

from app.ai.dataset_dependency_extractor_v1 import (
    DatasetDependencyExtractionResult,
    extract_dataset_dependencies,
)

from app.planning.analytical_v1.context import (
    AnalyticalPlannerContext,
    build_analytical_planner_context,
    require_ready_planner_context,
)

from app.planning.analytical_v1.dependency import (
    DatasetDependencyGateResult,
    dependency_gate_summary,
    evaluate_dataset_dependencies,
)

from app.planning.analytical_v1.input import (
    AnalyticalPlannerInput,
    build_analytical_planner_input,
)

from app.planning.analytical_v1.relationships import (
    RoutingRelationshipContext,
)

from app.planning.analytical_v1.safety import (
    analytical_planner_safety_summary,
)


# ============================================================
# VERSION
# ============================================================

ANALYTICAL_REASONING_PIPELINE_VERSION = (
    "analytical_reasoning_pipeline_v1.0"
)


# ============================================================
# STATUS
# ============================================================

AnalyticalReasoningPipelineStatus = Literal[
    "ready",
    "dependency_generation_error",
    "dependency_invalid_candidate",
    "structurally_blocked",
    "planner_generation_error",
    "planner_blocked",
]


# ============================================================
# RESULT
# ============================================================

class AnalyticalReasoningPipelineResult(
    BaseModel
):
    """
    Complete production result for the DataLens analytical
    reasoning pipeline.

    This pipeline performs reasoning and deterministic safety
    checks only.

    It does NOT execute analytical tools.

    ready_for_execution=True means only that the final
    canonicalized AnalyticalPlannerCandidate is authorized to
    enter a future analytical execution layer.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    pipeline_version: str

    user_request: str

    status: AnalyticalReasoningPipelineStatus

    ready_for_execution: bool

    dependency: DatasetDependencyExtractionResult

    dependency_gate: (
        DatasetDependencyGateResult
        | None
    )

    planner_context: (
        AnalyticalPlannerContext
        | None
    )

    planner_input: (
        AnalyticalPlannerInput
        | None
    )

    planner: (
        AIAnalyticalPlannerResult
        | None
    )


# ============================================================
# NORMALIZATION
# ============================================================

def _normalize_user_request(
    user_request: str,
) -> str:

    normalized = (
        user_request.strip()
    )


    if not (
        normalized
    ):

        raise ValueError(
            "Analytical reasoning requires a non-empty "
            "user_request."
        )


    return (
        normalized
    )


# ============================================================
# PUBLIC PIPELINE
# ============================================================

def run_analytical_reasoning_pipeline(
    *,
    user_request: str,
    structural_context: RoutingRelationshipContext,
    dependency_chat_client: (
        Any
        | None
    ) = None,
    planner_chat_client: (
        Any
        | None
    ) = None,
) -> AnalyticalReasoningPipelineResult:
    """
    Execute the production analytical reasoning chain.

    Pipeline
    --------

    1. Semantic dataset dependency extraction
       - AI identifies which datasets are needed together.

    2. Deterministic dependency validation
       - Python rejects invented dataset IDs.

    3. Deterministic structural feasibility gate
       - Python decides whether required datasets can be
         combined using validated relationships.

    4. Analytical planner context construction
       - this internally executes the trusted structural
         handoff and relationship-path resolution;
       - bridge datasets remain structural-only.

    5. Analytical planner input construction
       - only trusted semantic columns, grains and tools are
         made model-visible.

    6. Analytical planning
       - AI chooses analytical actions.

    7. Deterministic planner safety
       - references are canonicalized safely;
       - the planner validator authorizes or blocks the plan.

    No analytical tool is executed by this function.
    """

    normalized_request = (
        _normalize_user_request(
            user_request
        )
    )


    # ========================================================
    # 1. SEMANTIC DATASET DEPENDENCIES
    # ========================================================

    dependency = (
        extract_dataset_dependencies(
            user_request=(
                normalized_request
            ),

            context=(
                structural_context
            ),

            chat_client=(
                dependency_chat_client
            ),
        )
    )


    # ========================================================
    # DEPENDENCY GENERATION ERROR
    # ========================================================

    if (
        dependency.status
        == "generation_error"
    ):

        return (
            AnalyticalReasoningPipelineResult(
                pipeline_version=(
                    ANALYTICAL_REASONING_PIPELINE_VERSION
                ),

                user_request=(
                    normalized_request
                ),

                status=(
                    "dependency_generation_error"
                ),

                ready_for_execution=False,

                dependency=(
                    dependency
                ),

                dependency_gate=None,

                planner_context=None,

                planner_input=None,

                planner=None,
            )
        )


    # ========================================================
    # DEPENDENCY INVALID
    #
    # Example:
    # model invents "crm_secret" although that dataset was
    # never supplied by DataLens.
    # ========================================================

    if (
        dependency.status
        == "invalid_candidate"
    ):

        return (
            AnalyticalReasoningPipelineResult(
                pipeline_version=(
                    ANALYTICAL_REASONING_PIPELINE_VERSION
                ),

                user_request=(
                    normalized_request
                ),

                status=(
                    "dependency_invalid_candidate"
                ),

                ready_for_execution=False,

                dependency=(
                    dependency
                ),

                dependency_gate=None,

                planner_context=None,

                planner_input=None,

                planner=None,
            )
        )


    # ========================================================
    # INTERNAL CONSISTENCY
    # ========================================================

    if (
        not dependency.valid_for_feasibility_gate
        or dependency.candidate
        is None
    ):

        raise RuntimeError(
            "Dataset Dependency Extractor reached an "
            "inconsistent VALID state."
        )


    dependency_candidate = (
        dependency.candidate
    )


    # ========================================================
    # 2. DETERMINISTIC STRUCTURAL FEASIBILITY GATE
    #
    # AI does NOT decide whether joins are possible.
    # ========================================================

    dependency_gate = (
        evaluate_dataset_dependencies(
            candidate=(
                dependency_candidate
            ),

            context=(
                structural_context
            ),
        )
    )


    # ========================================================
    # STRUCTURALLY BLOCKED
    #
    # Crucial property:
    # the Analytical Planner is NOT invoked here.
    # ========================================================

    if not (
        dependency_gate.executable
    ):

        return (
            AnalyticalReasoningPipelineResult(
                pipeline_version=(
                    ANALYTICAL_REASONING_PIPELINE_VERSION
                ),

                user_request=(
                    normalized_request
                ),

                status=(
                    "structurally_blocked"
                ),

                ready_for_execution=False,

                dependency=(
                    dependency
                ),

                dependency_gate=(
                    dependency_gate
                ),

                planner_context=None,

                planner_input=None,

                planner=None,
            )
        )


    # ========================================================
    # 3. STRUCTURAL HANDOFF + RELATIONSHIP PATHS
    #
    # build_analytical_planner_context() deliberately rebuilds
    # and verifies the structural handoff.
    #
    # This means a stale or modified gate result cannot bypass
    # the trusted Python structural layer.
    # ========================================================

    planner_context = (
        build_analytical_planner_context(
            candidate=(
                dependency_candidate
            ),

            context=(
                structural_context
            ),
        )
    )


    planner_context = (
        require_ready_planner_context(
            planner_context
        )
    )


    # ========================================================
    # 4. MODEL-VISIBLE PLANNER INPUT
    # ========================================================

    planner_input = (
        build_analytical_planner_input(
            user_request=(
                normalized_request
            ),

            planner_context=(
                planner_context
            ),

            structural_context=(
                structural_context
            ),
        )
    )


    # ========================================================
    # 5. ANALYTICAL PLANNER
    # ========================================================

    planner = (
        generate_analytical_plan(
            planner_input=(
                planner_input
            ),

            chat_client=(
                planner_chat_client
            ),
        )
    )


    # ========================================================
    # PLANNER GENERATION ERROR
    # ========================================================

    if (
        planner.status
        == "generation_error"
    ):

        return (
            AnalyticalReasoningPipelineResult(
                pipeline_version=(
                    ANALYTICAL_REASONING_PIPELINE_VERSION
                ),

                user_request=(
                    normalized_request
                ),

                status=(
                    "planner_generation_error"
                ),

                ready_for_execution=False,

                dependency=(
                    dependency
                ),

                dependency_gate=(
                    dependency_gate
                ),

                planner_context=(
                    planner_context
                ),

                planner_input=(
                    planner_input
                ),

                planner=(
                    planner
                ),
            )
        )


    # ========================================================
    # PLANNER BLOCKED
    # ========================================================

    if not (
        planner.ready_for_execution
    ):

        return (
            AnalyticalReasoningPipelineResult(
                pipeline_version=(
                    ANALYTICAL_REASONING_PIPELINE_VERSION
                ),

                user_request=(
                    normalized_request
                ),

                status=(
                    "planner_blocked"
                ),

                ready_for_execution=False,

                dependency=(
                    dependency
                ),

                dependency_gate=(
                    dependency_gate
                ),

                planner_context=(
                    planner_context
                ),

                planner_input=(
                    planner_input
                ),

                planner=(
                    planner
                ),
            )
        )


    # ========================================================
    # READY CONSISTENCY
    # ========================================================

    if (
        planner.execution_candidate
        is None
    ):

        raise RuntimeError(
            "Analytical Planner reached READY without an "
            "execution candidate."
        )


    # ========================================================
    # READY
    # ========================================================

    return (
        AnalyticalReasoningPipelineResult(
            pipeline_version=(
                ANALYTICAL_REASONING_PIPELINE_VERSION
            ),

            user_request=(
                normalized_request
            ),

            status=(
                "ready"
            ),

            ready_for_execution=True,

            dependency=(
                dependency
            ),

            dependency_gate=(
                dependency_gate
            ),

            planner_context=(
                planner_context
            ),

            planner_input=(
                planner_input
            ),

            planner=(
                planner
            ),
        )
    )


# ============================================================
# EXECUTION GUARD
# ============================================================

def require_ready_analytical_reasoning(
    *,
    user_request: str,
    structural_context: RoutingRelationshipContext,
    dependency_chat_client: (
        Any
        | None
    ) = None,
    planner_chat_client: (
        Any
        | None
    ) = None,
):
    """
    Return only the canonicalized analytical execution
    candidate produced by a completely READY reasoning run.

    This will later form the final boundary between AI
    reasoning and analytical tool execution.
    """

    result = (
        run_analytical_reasoning_pipeline(
            user_request=(
                user_request
            ),

            structural_context=(
                structural_context
            ),

            dependency_chat_client=(
                dependency_chat_client
            ),

            planner_chat_client=(
                planner_chat_client
            ),
        )
    )


    if not (
        result.ready_for_execution
    ):

        raise ValueError(
            "Analytical reasoning is not authorized for "
            "execution. "
            f"status={result.status}"
        )


    if (
        result.planner
        is None
        or result.planner.execution_candidate
        is None
    ):

        raise RuntimeError(
            "Analytical reasoning reached READY without a "
            "canonical execution candidate."
        )


    return (
        result.planner.execution_candidate
    )


# ============================================================
# OBSERVABILITY
# ============================================================

def analytical_reasoning_pipeline_summary(
    result: AnalyticalReasoningPipelineResult,
) -> dict[str, Any]:
    """
    Compact representation suitable for API preview,
    tracing and observability.
    """

    dependency_gate = (
        dependency_gate_summary(
            result.dependency_gate
        )

        if (
            result.dependency_gate
            is not None
        )

        else None
    )


    planner_safety = (
        analytical_planner_safety_summary(
            result.planner.safety
        )

        if (
            result.planner
            is not None
            and result.planner.safety
            is not None
        )

        else None
    )


    return {
        "pipeline_version":
            result.pipeline_version,

        "status":
            result.status,

        "ready_for_execution":
            result.ready_for_execution,

        "dependency": {
            "status":
                result.dependency.status,

            "valid_for_feasibility_gate":
                (
                    result
                    .dependency
                    .valid_for_feasibility_gate
                ),

            "inference_ms":
                result.dependency.inference_ms,

            "requirement_count":
                (
                    len(
                        result
                        .dependency
                        .candidate
                        .requirements
                    )

                    if (
                        result
                        .dependency
                        .candidate
                        is not None
                    )

                    else 0
                ),
        },

        "dependency_gate":
            dependency_gate,

        "planner": (
            {
                "status":
                    result.planner.status,

                "ready_for_execution":
                    result.planner.ready_for_execution,

                "inference_ms":
                    result.planner.inference_ms,

                "safety":
                    planner_safety,
            }

            if (
                result.planner
                is not None
            )

            else None
        ),
    }