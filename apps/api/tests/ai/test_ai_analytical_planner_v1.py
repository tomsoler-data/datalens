from __future__ import annotations

import json

from types import SimpleNamespace

from app.ai.analytical_planner_prompt_v1 import (
    ANALYTICAL_PLANNER_PROMPT_VERSION,
    ANALYTICAL_PLANNER_SYSTEM_PROMPT,
)

from app.ai.analytical_planner_v1 import (
    AI_ANALYTICAL_PLANNER_VERSION,
    ANALYTICAL_PLANNER_TEMPERATURE,
    ANALYTICAL_PLANNER_THINKING,
    DEFAULT_ANALYTICAL_PLANNER_MODEL,
    build_analytical_planner_user_prompt,
    generate_analytical_plan,
    require_generated_analytical_plan,
)

from app.planning.analytical_v1.contract import (
    AnalyticalPlannerCandidate,
)

from tests.analysis.test_analytical_v1_production_parity import (
    build_production_planner_input,
    candidate_from_historical_result,
    get_case_by_id,
    load_historical_results,
)


# ============================================================
# FAKE OLLAMA CLIENT
# ============================================================

class FakeChatClient:
    """
    Deterministic substitute for ollama.Client.

    No model and no network are used.
    """

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
# FIXTURES
# ============================================================

def frozen_case(
    number: int,
):

    return (
        get_case_by_id(
            (
                "planner_frozen_v1_0_"
                f"{number:03d}"
            )
        )
    )


def historical_candidate(
    number: int,
) -> AnalyticalPlannerCandidate:

    case_id = (
        "planner_frozen_v1_0_"
        f"{number:03d}"
    )


    historical = (
        load_historical_results()
    )


    return (
        candidate_from_historical_result(
            historical[
                case_id
            ]
        )
    )


def candidate_json(
    candidate: AnalyticalPlannerCandidate,
) -> str:

    return (
        candidate.model_dump_json()
    )


# ============================================================
# 1. LOCKED CONFIGURATION
# ============================================================

def test_locked_configuration() -> None:

    assert (
        AI_ANALYTICAL_PLANNER_VERSION
        == "ai_analytical_planner_v1.0"
    )


    assert (
        ANALYTICAL_PLANNER_PROMPT_VERSION
        == "analytical_planner_prompt_v0.9_baseline"
    )


    assert (
        DEFAULT_ANALYTICAL_PLANNER_MODEL
        == "qwen3:4b-instruct"
    )


    assert (
        ANALYTICAL_PLANNER_TEMPERATURE
        == 0
    )


    assert (
        ANALYTICAL_PLANNER_THINKING
        is False
    )


    assert (
        "Tu es l'Analytical Planner de DataLens."
        in ANALYTICAL_PLANNER_SYSTEM_PROMPT
    )


    assert (
        "join_datasets"
        in ANALYTICAL_PLANNER_SYSTEM_PROMPT
    )


    print(
        "Development-selected configuration locked: PASS"
    )


# ============================================================
# 2. EXACT USER PROMPT SERIALIZATION
# ============================================================

def test_user_prompt_serialization() -> None:

    case = (
        frozen_case(
            8
        )
    )


    planner_input = (
        build_production_planner_input(
            case
        )
    )


    payload = (
        planner_input.model_dump(
            mode="json",
        )
    )


    expected = (
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


    actual = (
        build_analytical_planner_user_prompt(
            planner_input
        )
    )


    assert (
        actual
        == expected
    )


    assert (
        '"expected"'
        not in actual
    )


    assert (
        '"score"'
        not in actual
    )


    assert (
        '"frozen"'
        not in actual
    )


    print(
        "Historical user prompt serialization preserved: PASS"
    )


# ============================================================
# 3. EXACT VALID PLAN → READY
# ============================================================

def test_exact_valid_plan_ready() -> None:

    case = (
        frozen_case(
            8
        )
    )


    planner_input = (
        build_production_planner_input(
            case
        )
    )


    expected_candidate = (
        AnalyticalPlannerCandidate
        .model_validate(
            case.expected.model_dump(
                mode="json",
            )
        )
    )


    fake = (
        FakeChatClient(
            candidate_json(
                expected_candidate
            )
        )
    )


    result = (
        generate_analytical_plan(
            planner_input=(
                planner_input
            ),

            chat_client=(
                fake
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
        result.raw_candidate
        is not None
    )


    assert (
        result.safety
        is not None
    )


    assert (
        result.safety.ready_for_execution
    )


    assert (
        result.execution_candidate
        is not None
    )


    assert (
        len(
            fake.calls
        )
        == 1
    )


    print(
        "Exact valid structured plan reaches READY: PASS"
    )


# ============================================================
# 4. MODEL CALL CONFIGURATION
# ============================================================

def test_model_call_configuration() -> None:

    case = (
        frozen_case(
            8
        )
    )


    planner_input = (
        build_production_planner_input(
            case
        )
    )


    expected_candidate = (
        AnalyticalPlannerCandidate
        .model_validate(
            case.expected.model_dump(
                mode="json",
            )
        )
    )


    fake = (
        FakeChatClient(
            candidate_json(
                expected_candidate
            )
        )
    )


    generate_analytical_plan(
        planner_input=(
            planner_input
        ),

        chat_client=(
            fake
        ),
    )


    assert (
        len(
            fake.calls
        )
        == 1
    )


    call = (
        fake.calls[
            0
        ]
    )


    assert (
        call[
            "model"
        ]
        == "qwen3:4b-instruct"
    )


    assert (
        call[
            "options"
        ][
            "temperature"
        ]
        == 0
    )


    assert (
        call[
            "think"
        ]
        is False
    )


    assert (
        len(
            call[
                "messages"
            ]
        )
        == 2
    )


    assert (
        call[
            "messages"
        ][
            0
        ][
            "role"
        ]
        == "system"
    )


    assert (
        call[
            "messages"
        ][
            0
        ][
            "content"
        ]
        == ANALYTICAL_PLANNER_SYSTEM_PROMPT
    )


    assert (
        call[
            "messages"
        ][
            1
        ][
            "role"
        ]
        == "user"
    )


    assert (
        call[
            "messages"
        ][
            1
        ][
            "content"
        ]
        == build_analytical_planner_user_prompt(
            planner_input
        )
    )


    assert (
        call[
            "format"
        ]
        == (
            AnalyticalPlannerCandidate
            .model_json_schema()
        )
    )


    print(
        "Model call configuration preserved: PASS"
    )


# ============================================================
# 5. HISTORICAL 008
#
# The historical model used an unqualified unique alias:
#
#     channel
#
# Production safety must canonicalize it before execution.
# ============================================================

def test_historical_008_alias_ready() -> None:

    case = (
        frozen_case(
            8
        )
    )


    planner_input = (
        build_production_planner_input(
            case
        )
    )


    candidate = (
        historical_candidate(
            8
        )
    )


    fake = (
        FakeChatClient(
            candidate_json(
                candidate
            )
        )
    )


    result = (
        generate_analytical_plan(
            planner_input=(
                planner_input
            ),

            chat_client=(
                fake
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
        result.raw_candidate
        is not None
    )


    assert (
        result.execution_candidate
        is not None
    )


    assert (
        result.safety
        is not None
    )


    assert (
        len(
            result
            .safety
            .canonicalization
            .rewrites
        )
        == 1
    )


    raw_group_by = (
        result
        .raw_candidate
        .plans[
            0
        ]
        .steps[
            1
        ]
        .action
        .group_by
    )


    execution_group_by = (
        result
        .execution_candidate
        .plans[
            0
        ]
        .steps[
            1
        ]
        .action
        .group_by
    )


    assert (
        raw_group_by
        == "channel"
    )


    assert (
        execution_group_by
        == "ad_performance.channel"
    )


    print(
        "Historical 008 unique alias canonicalized: PASS"
    )


# ============================================================
# 6. HISTORICAL 012
#
# References are safely canonicalized, but the model's real
# target-grain reasoning error must remain BLOCKED.
# ============================================================

def test_historical_012_grain_error_blocked() -> None:

    case = (
        frozen_case(
            12
        )
    )


    planner_input = (
        build_production_planner_input(
            case
        )
    )


    candidate = (
        historical_candidate(
            12
        )
    )


    fake = (
        FakeChatClient(
            candidate_json(
                candidate
            )
        )
    )


    result = (
        generate_analytical_plan(
            planner_input=(
                planner_input
            ),

            chat_client=(
                fake
            ),
        )
    )


    assert (
        result.status
        == "blocked"
    )


    assert not (
        result.ready_for_execution
    )


    assert (
        result.raw_candidate
        is not None
    )


    assert (
        result.safety
        is not None
    )


    assert (
        result.safety.canonicalization.safe
    )


    assert (
        len(
            result
            .safety
            .canonicalization
            .rewrites
        )
        == 2
    )


    assert (
        result.safety.blocking_stage
        == "planner_validation"
    )


    assert (
        "entity_target_grain_mismatch"
        in result.safety.blocking_codes
    )


    assert (
        result.execution_candidate
        is None
    )


    print(
        "Historical 012 reasoning error remains BLOCKED: PASS"
    )


# ============================================================
# 7. HISTORICAL 001
#
# "sum" is not a column reference and must never be silently
# repaired.
# ============================================================

def test_historical_001_unknown_reference_blocked() -> None:

    case = (
        frozen_case(
            1
        )
    )


    planner_input = (
        build_production_planner_input(
            case
        )
    )


    candidate = (
        historical_candidate(
            1
        )
    )


    fake = (
        FakeChatClient(
            candidate_json(
                candidate
            )
        )
    )


    result = (
        generate_analytical_plan(
            planner_input=(
                planner_input
            ),

            chat_client=(
                fake
            ),
        )
    )


    assert (
        result.status
        == "blocked"
    )


    assert not (
        result.ready_for_execution
    )


    assert (
        result.safety
        is not None
    )


    assert (
        result.safety.blocking_stage
        == "reference_canonicalization"
    )


    assert (
        "unknown_reference"
        in result.safety.blocking_codes
    )


    assert (
        result.execution_candidate
        is None
    )


    print(
        "Historical 001 unsafe reference remains BLOCKED: PASS"
    )


# ============================================================
# 8. INVALID STRUCTURED OUTPUT
# ============================================================

def test_invalid_json_generation_error() -> None:

    case = (
        frozen_case(
            8
        )
    )


    planner_input = (
        build_production_planner_input(
            case
        )
    )


    fake = (
        FakeChatClient(
            "this is not valid JSON"
        )
    )


    result = (
        generate_analytical_plan(
            planner_input=(
                planner_input
            ),

            chat_client=(
                fake
            ),
        )
    )


    assert (
        result.status
        == "generation_error"
    )


    assert not (
        result.ready_for_execution
    )


    assert (
        result.raw_candidate
        is None
    )


    assert (
        result.safety
        is None
    )


    assert (
        result.execution_candidate
        is None
    )


    assert (
        result.error
        is not None
    )


    print(
        "Invalid structured output becomes generation_error: PASS"
    )


# ============================================================
# 9. EMPTY RESPONSE
# ============================================================

def test_empty_response_generation_error() -> None:

    case = (
        frozen_case(
            8
        )
    )


    planner_input = (
        build_production_planner_input(
            case
        )
    )


    fake = (
        FakeChatClient(
            ""
        )
    )


    result = (
        generate_analytical_plan(
            planner_input=(
                planner_input
            ),

            chat_client=(
                fake
            ),
        )
    )


    assert (
        result.status
        == "generation_error"
    )


    assert (
        result.execution_candidate
        is None
    )


    assert (
        result.error
        is not None
    )


    print(
        "Empty model response becomes generation_error: PASS"
    )


# ============================================================
# 10. EXECUTION GUARD RETURNS CANONICAL CANDIDATE
# ============================================================

def test_execution_guard_returns_safe_candidate() -> None:

    case = (
        frozen_case(
            8
        )
    )


    planner_input = (
        build_production_planner_input(
            case
        )
    )


    fake = (
        FakeChatClient(
            candidate_json(
                historical_candidate(
                    8
                )
            )
        )
    )


    candidate = (
        require_generated_analytical_plan(
            planner_input=(
                planner_input
            ),

            chat_client=(
                fake
            ),
        )
    )


    assert (
        candidate
        .plans[
            0
        ]
        .steps[
            1
        ]
        .action
        .group_by
        == "ad_performance.channel"
    )


    print(
        "Execution guard returns canonical safe candidate: PASS"
    )


# ============================================================
# 11. EXECUTION GUARD REJECTS BLOCKED CANDIDATE
# ============================================================

def test_execution_guard_rejects_blocked_candidate() -> None:

    case = (
        frozen_case(
            12
        )
    )


    planner_input = (
        build_production_planner_input(
            case
        )
    )


    fake = (
        FakeChatClient(
            candidate_json(
                historical_candidate(
                    12
                )
            )
        )
    )


    try:

        require_generated_analytical_plan(
            planner_input=(
                planner_input
            ),

            chat_client=(
                fake
            ),
        )


    except ValueError as error:

        message = (
            str(
                error
            )
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
            "Execution guard rejects BLOCKED candidate: PASS"
        )


    else:

        raise AssertionError(
            "A blocked model candidate must never reach "
            "analytical execution."
        )


# ============================================================
# 12. EXECUTION GUARD REJECTS GENERATION ERROR
# ============================================================

def test_execution_guard_rejects_generation_error() -> None:

    case = (
        frozen_case(
            8
        )
    )


    planner_input = (
        build_production_planner_input(
            case
        )
    )


    fake = (
        FakeChatClient(
            "invalid-json"
        )
    )


    try:

        require_generated_analytical_plan(
            planner_input=(
                planner_input
            ),

            chat_client=(
                fake
            ),
        )


    except RuntimeError as error:

        assert (
            "generation failed"
            in str(
                error
            )
        )


        print(
            "Execution guard rejects generation_error: PASS"
        )


    else:

        raise AssertionError(
            "A generation error must never produce an "
            "execution candidate."
        )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print(
        "=== DATALENS AI ANALYTICAL PLANNER v1.0 ==="
    )


    print()


    test_locked_configuration()

    test_user_prompt_serialization()

    test_exact_valid_plan_ready()

    test_model_call_configuration()

    test_historical_008_alias_ready()

    test_historical_012_grain_error_blocked()

    test_historical_001_unknown_reference_blocked()

    test_invalid_json_generation_error()

    test_empty_response_generation_error()

    test_execution_guard_returns_safe_candidate()

    test_execution_guard_rejects_blocked_candidate()

    test_execution_guard_rejects_generation_error()


    print()


    print(
        "NO OLLAMA INFERENCE WAS PERFORMED."
    )


    print()


    print(
        "AI Analytical Planner v1.0: PASS"
    )


if __name__ == "__main__":
    main()