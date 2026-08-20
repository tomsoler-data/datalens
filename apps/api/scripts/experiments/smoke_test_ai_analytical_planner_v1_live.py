from __future__ import annotations

import json

from pathlib import Path

from app.ai.analytical_planner_v1 import (
    AI_ANALYTICAL_PLANNER_VERSION,
    DEFAULT_ANALYTICAL_PLANNER_MODEL,
    generate_analytical_plan,
)

from app.ai.analytical_planner_prompt_v1 import (
    ANALYTICAL_PLANNER_PROMPT_VERSION,
)

from app.evals.analytical_planner_benchmark_v0_9 import (
    AnalyticalPlannerEvalCase,
    build_planner_input_for_case,
)

from app.planning.analytical_v1.input import (
    AnalyticalPlannerInput,
)

from app.planning.analytical_v1.safety import (
    analytical_planner_safety_summary,
)


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = (
    Path(
        __file__
    )
    .resolve()
    .parents[2]
)


DEVELOPMENT_BENCHMARK_PATH = (
    BASE_DIR
    / "evals"
    / "analytical_planner_development_v0_9.jsonl"
)


TARGET_CASE_ID = (
    "planner_v0_9_validation_005"
)


# ============================================================
# LOAD ONE DEVELOPMENT CASE
# ============================================================

def load_development_case(
    case_id: str,
) -> AnalyticalPlannerEvalCase:
    """
    Load exactly one case from the NON-FROZEN development
    benchmark.

    This smoke test must never read the Frozen planner
    benchmark.
    """

    if not (
        DEVELOPMENT_BENCHMARK_PATH.exists()
    ):

        raise FileNotFoundError(
            "Development benchmark not found: "
            f"{DEVELOPMENT_BENCHMARK_PATH}"
        )


    for line in (
        DEVELOPMENT_BENCHMARK_PATH
        .read_text(
            encoding="utf-8",
        )
        .splitlines()
    ):

        if not (
            line.strip()
        ):

            continue


        payload = (
            json.loads(
                line
            )
        )


        if (
            payload.get(
                "case_id"
            )
            != case_id
        ):

            continue


        # ====================================================
        # HARD GUARDS
        # ====================================================

        if (
            payload.get(
                "frozen"
            )
            is True
        ):

            raise ValueError(
                "Live smoke test refuses to use "
                "a frozen case."
            )


        if (
            payload.get(
                "split"
            )
            == "test"
        ):

            raise ValueError(
                "Live smoke test refuses to use "
                "a test-split case."
            )


        return (
            AnalyticalPlannerEvalCase
            .model_validate(
                payload
            )
        )


    raise ValueError(
        "Development planner case not found: "
        f"{case_id}"
    )


# ============================================================
# PRODUCTION INPUT
# ============================================================

def build_production_input(
    case: AnalyticalPlannerEvalCase,
) -> AnalyticalPlannerInput:
    """
    Use the historical development fixture builder only to
    reconstruct the already-known trusted test context.

    The resulting object is immediately converted into the
    autonomous production AnalyticalPlannerInput contract.

    The AI runtime itself therefore receives only production
    types.
    """

    development_input = (
        build_planner_input_for_case(
            case
        )
    )


    return (
        AnalyticalPlannerInput
        .model_validate(
            development_input.model_dump(
                mode="json",
            )
        )
    )


# ============================================================
# DISPLAY HELPERS
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
# LIVE SMOKE TEST
# ============================================================

def main() -> None:

    print(
        "=== DATALENS AI ANALYTICAL PLANNER v1 "
        "— LIVE SMOKE TEST ==="
    )


    print()


    print(
        "WARNING:"
    )


    print(
        "This script performs ONE real local Ollama inference."
    )


    print(
        "It does NOT score a benchmark and does NOT write "
        "evaluation results."
    )


    print()


    # ========================================================
    # CASE
    # ========================================================

    case = (
        load_development_case(
            TARGET_CASE_ID
        )
    )


    planner_input = (
        build_production_input(
            case
        )
    )


    print(
        "Case:",
        case.case_id,
    )


    print(
        "Split:",
        case.split,
    )


    print(
        "Domain:",
        case.domain,
    )


    print(
        "Request:",
        case.user_request,
    )


    print()


    print(
        "Planner:",
        AI_ANALYTICAL_PLANNER_VERSION,
    )


    print(
        "Prompt:",
        ANALYTICAL_PLANNER_PROMPT_VERSION,
    )


    print(
        "Model:",
        DEFAULT_ANALYTICAL_PLANNER_MODEL,
    )


    # ========================================================
    # MODEL-VISIBLE INPUT
    # ========================================================

    print_json(
        "PRODUCTION PLANNER INPUT",
        planner_input.model_dump(
            mode="json",
        ),
    )


    # ========================================================
    # ONE REAL INFERENCE
    # ========================================================

    print()
    print(
        "Running ONE real Qwen inference..."
    )


    result = (
        generate_analytical_plan(
            planner_input=(
                planner_input
            )
        )
    )


    # ========================================================
    # RESULT
    # ========================================================

    print()
    print(
        "=== RESULT ==="
    )


    print(
        "Status:",
        result.status,
    )


    print(
        "Ready for execution:",
        result.ready_for_execution,
    )


    print(
        "Inference ms:",
        round(
            result.inference_ms,
            1,
        ),
    )


    # ========================================================
    # RAW CANDIDATE
    # ========================================================

    if (
        result.raw_candidate
        is not None
    ):

        print_json(
            "RAW MODEL CANDIDATE",
            (
                result
                .raw_candidate
                .model_dump(
                    mode="json",
                )
            ),
        )


    else:

        print()
        print(
            "RAW MODEL CANDIDATE: NONE"
        )


    # ========================================================
    # SAFETY
    # ========================================================

    if (
        result.safety
        is not None
    ):

        print_json(
            "DETERMINISTIC SAFETY",
            analytical_planner_safety_summary(
                result.safety
            ),
        )


        print_json(
            "REFERENCE REWRITES",
            [
                rewrite.model_dump(
                    mode="json",
                )

                for rewrite
                in (
                    result
                    .safety
                    .canonicalization
                    .rewrites
                )
            ],
        )


    else:

        print()
        print(
            "DETERMINISTIC SAFETY: NOT REACHED"
        )


    # ========================================================
    # EXECUTION CANDIDATE
    # ========================================================

    if (
        result.execution_candidate
        is not None
    ):

        print_json(
            "AUTHORIZED EXECUTION CANDIDATE",
            (
                result
                .execution_candidate
                .model_dump(
                    mode="json",
                )
            ),
        )


    else:

        print()
        print(
            "AUTHORIZED EXECUTION CANDIDATE: NONE"
        )


    # ========================================================
    # GENERATION ERROR
    # ========================================================

    if (
        result.error
        is not None
    ):

        print()
        print(
            "ERROR:"
        )


        print(
            result.error
        )


    # ========================================================
    # FINAL INTERPRETATION
    # ========================================================

    print()
    print(
        "=== SMOKE TEST INTERPRETATION ==="
    )


    if (
        result.status
        == "ready"
    ):

        print(
            "PASS — real Qwen structured generation crossed "
            "the deterministic production safety boundary."
        )


    elif (
        result.status
        == "blocked"
    ):

        print(
            "BLOCKED — Qwen produced a structured plan, "
            "but Python correctly refused to authorize it."
        )


        print(
            "This is a safe system outcome, but the raw "
            "candidate must be inspected before production "
            "integration."
        )


        raise SystemExit(
            2
        )


    else:

        print(
            "GENERATION ERROR — the real Ollama call or "
            "structured-output parsing failed."
        )


        raise SystemExit(
            1
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


    print()


    print(
        "AI Analytical Planner v1 live smoke test: PASS"
    )


if __name__ == "__main__":
    main()