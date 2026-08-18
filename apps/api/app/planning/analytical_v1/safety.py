from __future__ import annotations

from typing import (
    Literal,
)

from pydantic import (
    BaseModel,
    ConfigDict,
)

from app.planning.analytical_v1.contract import (
    AnalyticalPlannerCandidate,
)

from app.planning.analytical_v1.input import (
    AnalyticalPlannerInput,
)

from app.planning.analytical_v1.validator import (
    AnalyticalPlannerValidationResultV091,
    validate_analytical_planner_candidate,
)

from app.planning.analytical_v1.reference_canonicalizer import (
    AnalyticalReferenceCanonicalizationResult,
    canonicalize_analytical_references,
)


# ============================================================
# VERSION
# ============================================================

ANALYTICAL_PLANNER_SAFETY_PIPELINE_VERSION = (
    "analytical_planner_safety_pipeline_v1.0"
)


# ============================================================
# STATUS TYPES
# ============================================================

PlannerSafetyBlockingStage = Literal[
    "none",
    "reference_canonicalization",
    "planner_validation",
]


# ============================================================
# RESULT
# ============================================================

class AnalyticalPlannerSafetyPipelineResult(
    BaseModel
):
    """
    Deterministic safety result between an LLM planner
    candidate and analytical execution.

    ready_for_execution=True means:

        1. every reference was resolved safely;
        2. the canonicalized plan passed Validator v0.9.1;
        3. execution_candidate is the canonical candidate
           that downstream execution is allowed to consume.

    The original LLM candidate is never silently executed.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    pipeline_version: str

    ready_for_execution: bool

    blocking_stage: PlannerSafetyBlockingStage

    blocking_codes: list[
        str
    ]

    canonicalization: (
        AnalyticalReferenceCanonicalizationResult
    )

    validation: (
        AnalyticalPlannerValidationResultV091
        | None
    )

    execution_candidate: (
        AnalyticalPlannerCandidate
        | None
    )


# ============================================================
# PUBLIC PIPELINE
# ============================================================

def evaluate_analytical_planner_safety(
    *,
    candidate: AnalyticalPlannerCandidate,
    planner_input: AnalyticalPlannerInput,
) -> AnalyticalPlannerSafetyPipelineResult:
    """
    Execute the deterministic planner safety boundary.

    Pipeline:

        LLM candidate
            ↓
        reference canonicalization
            ↓
        planner validation
            ↓
        ready_for_execution

    Important:
    ----------
    This function never:

    - calls an LLM;
    - repairs intent;
    - changes family;
    - changes target grain;
    - adds or removes analytical steps;
    - changes tool selection;
    - invents columns;
    - invents relationships;
    - infers joins;
    - rewrites formulas.

    Reference canonicalization is limited to deterministic,
    unique aliases.
    """

    # ========================================================
    # 1. REFERENCE CANONICALIZATION
    # ========================================================

    canonicalization = (
        canonicalize_analytical_references(
            candidate=candidate,
            planner_input=planner_input,
        )
    )


    # ========================================================
    # BLOCK IMMEDIATELY ON UNKNOWN / AMBIGUOUS REFERENCES
    #
    # We deliberately do not send an unsafe reference plan
    # further toward analytical validation/execution.
    # ========================================================

    if not canonicalization.safe:

        blocking_codes = [
            issue.code

            for issue
            in canonicalization.issues
        ]


        return (
            AnalyticalPlannerSafetyPipelineResult(
                pipeline_version=(
                    ANALYTICAL_PLANNER_SAFETY_PIPELINE_VERSION
                ),

                ready_for_execution=False,

                blocking_stage=(
                    "reference_canonicalization"
                ),

                blocking_codes=(
                    blocking_codes
                ),

                canonicalization=(
                    canonicalization
                ),

                validation=None,

                execution_candidate=None,
            )
        )


    canonical_candidate = (
        canonicalization
        .canonicalized_candidate
    )


    # ========================================================
    # 2. DETERMINISTIC PLANNER VALIDATION
    # ========================================================

    validation = (
        validate_analytical_planner_candidate(
            candidate=(
                canonical_candidate
            ),

            planner_input=(
                planner_input
            ),
        )
    )


    # ========================================================
    # BLOCK ON ANALYTICAL / SEMANTIC VALIDATION ERROR
    # ========================================================

    if not validation.valid:

        blocking_codes = [
            issue.code

            for issue
            in validation.issues
        ]


        return (
            AnalyticalPlannerSafetyPipelineResult(
                pipeline_version=(
                    ANALYTICAL_PLANNER_SAFETY_PIPELINE_VERSION
                ),

                ready_for_execution=False,

                blocking_stage=(
                    "planner_validation"
                ),

                blocking_codes=(
                    blocking_codes
                ),

                canonicalization=(
                    canonicalization
                ),

                validation=(
                    validation
                ),

                execution_candidate=None,
            )
        )


    # ========================================================
    # 3. READY
    # ========================================================

    return (
        AnalyticalPlannerSafetyPipelineResult(
            pipeline_version=(
                ANALYTICAL_PLANNER_SAFETY_PIPELINE_VERSION
            ),

            ready_for_execution=True,

            blocking_stage="none",

            blocking_codes=[],

            canonicalization=(
                canonicalization
            ),

            validation=(
                validation
            ),

            execution_candidate=(
                canonical_candidate
            ),
        )
    )


# ============================================================
# EXECUTION GUARD
# ============================================================

def require_safe_analytical_plan(
    *,
    candidate: AnalyticalPlannerCandidate,
    planner_input: AnalyticalPlannerInput,
) -> AnalyticalPlannerCandidate:
    """
    Return the canonicalized planner candidate only when the
    entire deterministic safety pipeline succeeds.

    Downstream analytical execution should consume the value
    returned by this function rather than the raw LLM output.
    """

    result = (
        evaluate_analytical_planner_safety(
            candidate=candidate,
            planner_input=planner_input,
        )
    )


    if not (
        result.ready_for_execution
    ):

        raise ValueError(
            "Analytical planner candidate is not safe "
            "for execution. "
            f"blocking_stage={result.blocking_stage}, "
            f"blocking_codes={result.blocking_codes}"
        )


    if (
        result.execution_candidate
        is None
    ):

        raise RuntimeError(
            "Planner safety pipeline reached a logically "
            "inconsistent state: ready_for_execution=True "
            "without an execution candidate."
        )


    return (
        result.execution_candidate
    )


# ============================================================
# SUMMARY
# ============================================================

def analytical_planner_safety_summary(
    result: AnalyticalPlannerSafetyPipelineResult,
) -> dict:
    """
    Compact observability-oriented representation.

    Useful later for API responses, tracing and eval logs.
    """

    return {
        "pipeline_version":
            result.pipeline_version,

        "ready_for_execution":
            result.ready_for_execution,

        "blocking_stage":
            result.blocking_stage,

        "blocking_codes":
            list(
                result.blocking_codes
            ),

        "reference_rewrite_count":
            len(
                result
                .canonicalization
                .rewrites
            ),

        "reference_issue_count":
            len(
                result
                .canonicalization
                .issues
            ),

        "validator_issue_count":
            (
                len(
                    result
                    .validation
                    .issues
                )

                if (
                    result.validation
                    is not None
                )

                else 0
            ),
    }