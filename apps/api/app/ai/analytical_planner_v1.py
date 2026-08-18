from __future__ import annotations

import json

from time import perf_counter
from typing import (
    Any,
    Literal,
)

from pydantic import (
    BaseModel,
    ConfigDict,
)

from app.ai.analytical_planner_prompt_v1 import (
    ANALYTICAL_PLANNER_PROMPT_VERSION,
    ANALYTICAL_PLANNER_SYSTEM_PROMPT,
)

from app.ai.provider import (
    client,
)

from app.planning.analytical_v1.contract import (
    AnalyticalPlannerCandidate,
)

from app.planning.analytical_v1.input import (
    AnalyticalPlannerInput,
)

from app.planning.analytical_v1.safety import (
    AnalyticalPlannerSafetyPipelineResult,
    evaluate_analytical_planner_safety,
)


# ============================================================
# VERSION
# ============================================================

AI_ANALYTICAL_PLANNER_VERSION = (
    "ai_analytical_planner_v1.0"
)


# ============================================================
# MODEL CONFIGURATION
#
# This configuration is intentionally aligned with the model
# configuration selected before the historical Frozen run.
# ============================================================

DEFAULT_ANALYTICAL_PLANNER_MODEL = (
    "qwen3:4b-instruct"
)


ANALYTICAL_PLANNER_TEMPERATURE = 0


ANALYTICAL_PLANNER_THINKING = False


# ============================================================
# STATUS
# ============================================================

AIAnalyticalPlannerStatus = Literal[
    "ready",
    "blocked",
    "generation_error",
]


# ============================================================
# RESULT
# ============================================================

class AIAnalyticalPlannerResult(
    BaseModel
):
    """
    Production envelope for one AI analytical planning call.

    status = ready
        The model produced a valid structured candidate and
        the deterministic safety pipeline authorized the
        canonicalized candidate for execution.

    status = blocked
        Structured generation succeeded, but deterministic
        safety rejected the candidate.

    status = generation_error
        The model call failed or its output could not be
        parsed as AnalyticalPlannerCandidate.

    Only execution_candidate may be sent to downstream
    analytical execution.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    planner_version: str

    prompt_version: str

    model: str

    temperature: int

    thinking: bool

    status: AIAnalyticalPlannerStatus

    ready_for_execution: bool

    inference_ms: float

    raw_content: (
        str
        | None
    )

    raw_candidate: (
        AnalyticalPlannerCandidate
        | None
    )

    safety: (
        AnalyticalPlannerSafetyPipelineResult
        | None
    )

    execution_candidate: (
        AnalyticalPlannerCandidate
        | None
    )

    error: (
        str
        | None
    )


# ============================================================
# EXACT MODEL-VISIBLE USER PROMPT
# ============================================================

def build_analytical_planner_user_prompt(
    planner_input: AnalyticalPlannerInput,
) -> str:
    """
    Build the exact model-visible payload used by the
    development-selected Analytical Planner prompt.

    No benchmark expectation, score, note or frozen metadata
    is included.
    """

    payload = (
        planner_input.model_dump(
            mode="json",
        )
    )


    return (
        "ANALYTICAL PLANNER INPUT:\n\n"
        + json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        )
        + "\n\n"
        + (
            "Construis uniquement le plan analytique "
            "nécessaire pour chaque requirement."
        )
    )


# ============================================================
# OLLAMA RESPONSE
# ============================================================

def _extract_response_content(
    response: Any,
) -> str:
    """
    Extract response.message.content from an Ollama response.

    The historical planner uses the same response interface.
    """

    message = getattr(
        response,
        "message",
        None,
    )


    if (
        message
        is None
    ):

        raise ValueError(
            "The analytical planner model response "
            "contains no message."
        )


    content = getattr(
        message,
        "content",
        None,
    )


    if (
        content
        is None
    ):

        raise ValueError(
            "The analytical planner model response "
            "contains no message content."
        )


    content = (
        str(
            content
        )
    )


    if not (
        content.strip()
    ):

        raise ValueError(
            "The analytical planner model returned "
            "empty structured content."
        )


    return (
        content
    )


# ============================================================
# RESULT HELPERS
# ============================================================

def _generation_error_result(
    *,
    model: str,
    inference_ms: float,
    raw_content: (
        str
        | None
    ),
    error: Exception,
) -> AIAnalyticalPlannerResult:

    return (
        AIAnalyticalPlannerResult(
            planner_version=(
                AI_ANALYTICAL_PLANNER_VERSION
            ),

            prompt_version=(
                ANALYTICAL_PLANNER_PROMPT_VERSION
            ),

            model=(
                model
            ),

            temperature=(
                ANALYTICAL_PLANNER_TEMPERATURE
            ),

            thinking=(
                ANALYTICAL_PLANNER_THINKING
            ),

            status=(
                "generation_error"
            ),

            ready_for_execution=False,

            inference_ms=(
                inference_ms
            ),

            raw_content=(
                raw_content
            ),

            raw_candidate=None,

            safety=None,

            execution_candidate=None,

            error=(
                f"{type(error).__name__}: "
                f"{error}"
            ),
        )
    )


# ============================================================
# MODEL CALL
# ============================================================

def generate_analytical_plan(
    *,
    planner_input: AnalyticalPlannerInput,
    model: str = (
        DEFAULT_ANALYTICAL_PLANNER_MODEL
    ),
    chat_client: (
        Any
        | None
    ) = None,
) -> AIAnalyticalPlannerResult:
    """
    Generate one analytical plan and apply the deterministic
    production safety boundary.

    Pipeline:

        AnalyticalPlannerInput
            ↓
        Qwen structured generation
            ↓
        AnalyticalPlannerCandidate
            ↓
        deterministic reference canonicalization
            ↓
        deterministic planner validation
            ↓
        READY / BLOCKED

    The raw model candidate is never authorized directly for
    execution.

    chat_client is injectable so deterministic tests can run
    without contacting Ollama.
    """

    active_client = (
        client

        if (
            chat_client
            is None
        )

        else (
            chat_client
        )
    )


    raw_content: (
        str
        | None
    ) = None


    started_at = (
        perf_counter()
    )


    # ========================================================
    # LOCAL MODEL + STRUCTURED OUTPUT
    # ========================================================

    try:

        response = (
            active_client.chat(
                model=(
                    model
                ),

                messages=[
                    {
                        "role":
                            "system",

                        "content":
                            ANALYTICAL_PLANNER_SYSTEM_PROMPT,
                    },

                    {
                        "role":
                            "user",

                        "content":
                            build_analytical_planner_user_prompt(
                                planner_input
                            ),
                    },
                ],

                format=(
                    AnalyticalPlannerCandidate
                    .model_json_schema()
                ),

                options={
                    "temperature":
                        ANALYTICAL_PLANNER_TEMPERATURE,
                },

                think=(
                    ANALYTICAL_PLANNER_THINKING
                ),
            )
        )


        raw_content = (
            _extract_response_content(
                response
            )
        )


        raw_candidate = (
            AnalyticalPlannerCandidate
            .model_validate_json(
                raw_content
            )
        )


    except Exception as error:

        inference_ms = (
            (
                perf_counter()
                - started_at
            )
            *
            1000.0
        )


        return (
            _generation_error_result(
                model=(
                    model
                ),

                inference_ms=(
                    inference_ms
                ),

                raw_content=(
                    raw_content
                ),

                error=(
                    error
                ),
            )
        )


    inference_ms = (
        (
            perf_counter()
            - started_at
        )
        *
        1000.0
    )


    # ========================================================
    # DETERMINISTIC SAFETY BOUNDARY
    #
    # We intentionally do not catch programming errors from
    # this layer as "generation errors".
    #
    # If the deterministic safety pipeline itself crashes,
    # that is an application defect and should remain visible.
    # ========================================================

    safety = (
        evaluate_analytical_planner_safety(
            candidate=(
                raw_candidate
            ),

            planner_input=(
                planner_input
            ),
        )
    )


    # ========================================================
    # BLOCKED
    # ========================================================

    if not (
        safety.ready_for_execution
    ):

        return (
            AIAnalyticalPlannerResult(
                planner_version=(
                    AI_ANALYTICAL_PLANNER_VERSION
                ),

                prompt_version=(
                    ANALYTICAL_PLANNER_PROMPT_VERSION
                ),

                model=(
                    model
                ),

                temperature=(
                    ANALYTICAL_PLANNER_TEMPERATURE
                ),

                thinking=(
                    ANALYTICAL_PLANNER_THINKING
                ),

                status=(
                    "blocked"
                ),

                ready_for_execution=False,

                inference_ms=(
                    inference_ms
                ),

                raw_content=(
                    raw_content
                ),

                raw_candidate=(
                    raw_candidate
                ),

                safety=(
                    safety
                ),

                execution_candidate=None,

                error=None,
            )
        )


    # ========================================================
    # INTERNAL CONSISTENCY GUARD
    # ========================================================

    if (
        safety.execution_candidate
        is None
    ):

        raise RuntimeError(
            "Analytical Planner safety pipeline returned "
            "ready_for_execution=True without an "
            "execution_candidate."
        )


    # ========================================================
    # READY
    # ========================================================

    return (
        AIAnalyticalPlannerResult(
            planner_version=(
                AI_ANALYTICAL_PLANNER_VERSION
            ),

            prompt_version=(
                ANALYTICAL_PLANNER_PROMPT_VERSION
            ),

            model=(
                model
            ),

            temperature=(
                ANALYTICAL_PLANNER_TEMPERATURE
            ),

            thinking=(
                ANALYTICAL_PLANNER_THINKING
            ),

            status=(
                "ready"
            ),

            ready_for_execution=True,

            inference_ms=(
                inference_ms
            ),

            raw_content=(
                raw_content
            ),

            raw_candidate=(
                raw_candidate
            ),

            safety=(
                safety
            ),

            execution_candidate=(
                safety.execution_candidate
            ),

            error=None,
        )
    )


# ============================================================
# EXECUTION GUARD
# ============================================================

def require_generated_analytical_plan(
    *,
    planner_input: AnalyticalPlannerInput,
    model: str = (
        DEFAULT_ANALYTICAL_PLANNER_MODEL
    ),
    chat_client: (
        Any
        | None
    ) = None,
) -> AnalyticalPlannerCandidate:
    """
    Generate and return only the canonical candidate that has
    successfully crossed the deterministic safety boundary.

    This is the function a future executor may call.

    Preview / observability endpoints should use
    generate_analytical_plan() instead so BLOCKED states remain
    inspectable.
    """

    result = (
        generate_analytical_plan(
            planner_input=(
                planner_input
            ),

            model=(
                model
            ),

            chat_client=(
                chat_client
            ),
        )
    )


    # ========================================================
    # GENERATION FAILURE
    # ========================================================

    if (
        result.status
        == "generation_error"
    ):

        raise RuntimeError(
            "Analytical Planner generation failed. "
            f"{result.error}"
        )


    # ========================================================
    # DETERMINISTIC BLOCK
    # ========================================================

    if not (
        result.ready_for_execution
    ):

        if (
            result.safety
            is None
        ):

            raise RuntimeError(
                "Analytical Planner returned a blocked state "
                "without safety information."
            )


        raise ValueError(
            "Generated analytical plan is not authorized "
            "for execution. "
            f"blocking_stage="
            f"{result.safety.blocking_stage}, "
            f"blocking_codes="
            f"{result.safety.blocking_codes}"
        )


    # ========================================================
    # READY CONSISTENCY
    # ========================================================

    if (
        result.execution_candidate
        is None
    ):

        raise RuntimeError(
            "Analytical Planner reached READY without an "
            "execution candidate."
        )


    return (
        result.execution_candidate
    )