from __future__ import annotations


from pathlib import (
    Path,
)


def main() -> None:
    source_path = (
        Path(__file__)
        .resolve()
        .parents[2]
        /
        "app"
        /
        "api"
        /
        "analysis_run.py"
    )

    source = source_path.read_text(
        encoding="utf-8"
    )


    print()
    print("=" * 80)
    print(
        "DATALENS OBJECTIVE COVERAGE EXECUTION GATE v0.1"
    )
    print("=" * 80)
    print()


    # ========================================================
    # IMPORT
    # ========================================================

    assert (
        source.count(
            "from app.planning.objective_coverage import ("
        )
        ==
        1
    )

    assert (
        source.count(
            "    require_objective_coverage,"
        )
        ==
        1
    )


    print(
        "[PASS] Objective Coverage guard imported"
    )


    # ========================================================
    # ROUTE SEGMENTS
    # ========================================================

    tool_start = source.index(
        "def run_ai_analytical_tool("
    )

    native_start = source.index(
        "def run_ai_native_pipeline("
    )

    if not (
        tool_start
        <
        native_start
    ):
        raise AssertionError(
            "Unexpected endpoint ordering."
        )


    tool_segment = source[
        tool_start:
        native_start
    ]

    native_segment = source[
        native_start:
    ]


    # ========================================================
    # EXPLICIT AI TOOL RUN
    # ========================================================

    tool_plan = (
        tool_segment.index(
            "plan_analyses_with_intent_routing("
        )
    )

    tool_guard = (
        tool_segment.index(
            "require_objective_coverage("
        )
    )

    tool_execute = (
        tool_segment.index(
            "execute_ai_planner_report("
        )
    )


    assert (
        tool_plan
        <
        tool_guard
        <
        tool_execute
    )


    assert (
        tool_segment.count(
            "require_objective_coverage("
        )
        ==
        1
    )


    print(
        "[PASS] ai-tool-run is fail-closed before execution"
    )


    # ========================================================
    # NATIVE AI PIPELINE
    # ========================================================

    native_plan = (
        native_segment.index(
            "plan_analyses_with_intent_routing("
        )
    )

    native_guard = (
        native_segment.index(
            "require_objective_coverage("
        )
    )

    native_execute = (
        native_segment.index(
            "execute_native_ai_pipeline("
        )
    )


    assert (
        native_plan
        <
        native_guard
        <
        native_execute
    )


    assert (
        native_segment.count(
            "require_objective_coverage("
        )
        ==
        1
    )


    print(
        "[PASS] ai-native-run is fail-closed before tool calling"
    )


    # ========================================================
    # PREVIEW REMAINS READ-ONLY
    # ========================================================

    preview_start = source.index(
        '"/planning/ai-preview"'
    )

    tool_route_start = source.index(
        '"/planning/ai-tool-run"'
    )

    preview_segment = source[
        preview_start:
        tool_route_start
    ]


    assert (
        "plan_analyses_with_intent_routing("
        in
        preview_segment
    )

    assert (
        "require_objective_coverage("
        not in
        preview_segment
    )


    print(
        "[PASS] read-only planner preview remains non-executing"
    )


    # ========================================================
    # GLOBAL CONTRACT
    # ========================================================

    assert (
        source.count(
            "DATALENS_OBJECTIVE_COVERAGE_EXECUTION_GATE_V0_1"
        )
        ==
        2
    )

    assert (
        source.count(
            "require_objective_coverage("
        )
        ==
        2
    )


    print(
        "[PASS] exactly two execution boundaries are protected"
    )


    print()
    print(
        "PASS - Objective Coverage execution gate v0.1"
    )


if __name__ == "__main__":
    main()
