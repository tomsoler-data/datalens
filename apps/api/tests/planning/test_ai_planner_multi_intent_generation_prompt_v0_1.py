from __future__ import annotations


from app.planning.ai_analytical_planner import (
    SYSTEM_PROMPT,
    build_user_prompt,
)

from tests.planning.test_objective_coverage_multi_intent_same_metric_v0_1 import (
    OBJECTIVE,
    build_catalog,
)


def main() -> None:
    print()
    print("=" * 80)
    print(
        "DATALENS AI PLANNER MULTI-INTENT "
        "GENERATION PROMPT v0.1"
    )
    print("=" * 80)
    print()

    user_prompt = build_user_prompt(
        objective=OBJECTIVE,
        catalog=build_catalog(),
    )

    combined = (
        SYSTEM_PROMPT
        +
        "\n"
        +
        user_prompt
    )

    # ========================================================
    # INDEPENDENT INTENT DECOMPOSITION
    # ========================================================

    assert (
        "une proposition distincte par intention analytique"
        in
        combined.casefold()
    ), (
        "The planner prompt must explicitly require one distinct "
        "proposal per independently requested analytical intent."
    )

    assert (
        "même métrique"
        in
        combined.casefold()
    ), (
        "The prompt must explicitly explain that several intents "
        "may legitimately reuse the same business metric."
    )

    assert (
        "ne fusionne pas"
        in
        combined.casefold()
    ), (
        "The planner must be explicitly told not to merge "
        "independent analytical families into one proposal."
    )


    # ========================================================
    # BENCHMARK NON-INFERENCE
    # ========================================================

    assert (
        "n'infère jamais un benchmark"
        in
        combined.casefold()
    ), (
        "The prompt must explicitly forbid inferring a benchmark "
        "when the objective does not request one."
    )

    assert (
        "benchmark_reference = null"
        in
        combined
    )

    assert (
        "benchmark_operator = null"
        in
        combined
    )

    assert (
        "benchmark_selection = null"
        in
        combined
    )


    print(
        "[PASS] planner prompt requires independent "
        "intent decomposition"
    )

    print(
        "[PASS] same metric may be reused across "
        "multiple proposals"
    )

    print(
        "[PASS] independent families must not be merged"
    )

    print(
        "[PASS] benchmark inference is explicitly forbidden"
    )

    print()
    print(
        "PASS - AI Planner multi-intent "
        "generation prompt v0.1"
    )


if __name__ == "__main__":
    main()
