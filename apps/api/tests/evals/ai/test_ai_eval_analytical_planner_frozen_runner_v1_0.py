from __future__ import annotations

from app.evals.analytical_planner_frozen_benchmark_v1_0 import (
    build_planner_input_for_frozen_case,
    load_frozen_analytical_planner_benchmark,
)

from app.evals.analytical_planner_frozen_runner_v1_0 import (
    ANALYTICAL_PLANNER_FROZEN_RUNNER_VERSION,
    ANALYTICAL_PLANNER_PROMPT_VERSION,
    ANALYTICAL_PLANNER_SCORER_VERSION,
    ANALYTICAL_PLANNER_VALIDATOR_VERSION,
    BENCHMARK_PATH,
    EXPECTED_BENCHMARK_SHA256,
    FINAL_PATH,
    FROZEN_MODEL,
    build_frozen_planner_user_prompt,
    frozen_runner_metadata,
    preflight_frozen_runner,
    verify_frozen_benchmark_lock,
)


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print(
        "=== DATALENS ANALYTICAL PLANNER FROZEN RUNNER PREFLIGHT v1.0 ==="
    )


    print()


    print(
        "Runner:",
        ANALYTICAL_PLANNER_FROZEN_RUNNER_VERSION,
    )


    print(
        "Model:",
        FROZEN_MODEL,
    )


    print(
        "Prompt:",
        ANALYTICAL_PLANNER_PROMPT_VERSION,
    )


    print(
        "Validator:",
        ANALYTICAL_PLANNER_VALIDATOR_VERSION,
    )


    print(
        "Scorer:",
        ANALYTICAL_PLANNER_SCORER_VERSION,
    )


    print()


    # ========================================================
    # 1. HASH
    # ========================================================

    actual_hash = (
        verify_frozen_benchmark_lock()
    )


    assert (
        actual_hash
        == EXPECTED_BENCHMARK_SHA256
    )


    print(
        "Historical SHA-256 lock: PASS"
    )


    # ========================================================
    # 2. CASES
    # ========================================================

    cases = (
        load_frozen_analytical_planner_benchmark(
            BENCHMARK_PATH
        )
    )


    assert (
        len(
            cases
        )
        == 12
    )


    assert all(
        case.frozen

        for case
        in cases
    )


    assert all(
        case.split
        == "test"

        for case
        in cases
    )


    print(
        "12 frozen test cases: PASS"
    )


    # ========================================================
    # 3. MODEL LOCK
    # ========================================================

    assert (
        FROZEN_MODEL
        == "qwen3:4b-instruct"
    )


    print(
        "Selected model lock: PASS"
    )


    # ========================================================
    # 4. PROMPT LOCK
    # ========================================================

    assert (
        ANALYTICAL_PLANNER_PROMPT_VERSION
        == "analytical_planner_prompt_v0.9_baseline"
    )


    print(
        "Development-selected prompt lock: PASS"
    )


    # ========================================================
    # 5. MODEL-VISIBLE INPUT
    # ========================================================

    for case in cases:

        planner_input = (
            build_planner_input_for_frozen_case(
                case
            )
        )


        prompt = (
            build_frozen_planner_user_prompt(
                case
            )
        )


        # ----------------------------------------------------
        # Expected benchmark plan must never be serialized as
        # a top-level model-visible benchmark field.
        # ----------------------------------------------------

        assert (
            '"expected"'
            not in prompt
        )


        assert (
            '"notes"'
            not in prompt
        )


        assert (
            '"frozen"'
            not in prompt
        )


        # ----------------------------------------------------
        # Structural join capability must remain invisible to
        # the analytical planner.
        # ----------------------------------------------------

        for requirement in (
            planner_input.requirements
        ):

            assert (
                "join_datasets"
                not in (
                    requirement
                    .allowed_analytical_tools
                )
            )


    print(
        "Ground truth hidden from model-visible prompts: PASS"
    )


    print(
        "Structural join tool hidden from planner: PASS"
    )


    # ========================================================
    # 6. METADATA
    # ========================================================

    metadata = (
        frozen_runner_metadata(
            benchmark_hash=(
                actual_hash
            )
        )
    )


    assert (
        metadata[
            "historical_first_run"
        ]
        is True
    )


    assert (
        metadata[
            "frozen"
        ]
        is True
    )


    assert (
        metadata[
            "split"
        ]
        == "test"
    )


    assert (
        metadata[
            "model"
        ]
        == FROZEN_MODEL
    )


    print(
        "Historical run metadata: PASS"
    )


    # ========================================================
    # 7. PREFLIGHT
    #
    # This performs zero model inference.
    # ========================================================

    preflight = (
        preflight_frozen_runner(
            require_unused_final=False,
        )
    )


    assert (
        preflight[
            "benchmark_hash"
        ]
        == EXPECTED_BENCHMARK_SHA256
    )


    assert (
        preflight[
            "case_count"
        ]
        == 12
    )


    print(
        "Runner preflight without inference: PASS"
    )


    # ========================================================
    # 8. FIRST-RUN STATE
    # ========================================================

    print()


    print(
        "Final result already exists:",
        FINAL_PATH.exists(),
    )


    print(
        "Checkpoint completed cases:",
        preflight[
            "checkpoint_completed_cases"
        ],
    )


    print()


    print(
        "NO MODEL INFERENCE WAS PERFORMED BY THIS TEST."
    )


    print()


    print(
        "Analytical Planner Frozen Runner preflight v1.0: PASS"
    )


if __name__ == "__main__":
    main()