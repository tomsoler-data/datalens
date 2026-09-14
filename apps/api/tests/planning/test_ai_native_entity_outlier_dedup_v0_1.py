from pathlib import Path
import ast


ROOT = Path(__file__).resolve().parents[2]

API_PATH = (
    ROOT
    / "app"
    / "api"
    / "analysis_run.py"
)


print()
print("=" * 80)
print(
    "DATALENS AI-NATIVE ENTITY OUTLIER "
    "AI-FIRST AUTHORITY v0.1"
)
print("=" * 80)
print()


source = API_PATH.read_text(
    encoding="utf-8"
)

tree = ast.parse(
    source
)


# ============================================================
# 1. LEGACY HELPER MAY REMAIN, BUT IS NOT AUTHORITATIVE
# ============================================================

helper = next(
    (
        node
        for node
        in ast.walk(
            tree
        )
        if (
            isinstance(
                node,
                (
                    ast.FunctionDef,
                    ast.AsyncFunctionDef,
                ),
            )
            and
            node.name
            ==
            "remove_specialized_entity_outlier_duplicate"
        )
    ),
    None,
)


assert (
    helper
    is not None
), (
    "The historical dedup helper may remain during migration "
    "for backward compatibility."
)

print(
    "[PASS] legacy dedup helper remains available "
    "outside native authority"
)


# ============================================================
# 2. EXTRACT LIVE AI-NATIVE ROUTE
# ============================================================

route_start = source.index(
    '"/planning/ai-native-run"'
)

route_end = source.index(
    "# AI OBSERVABILITY",
    route_start,
)

route_segment = source[
    route_start:
    route_end
]


# ============================================================
# 3. QWEN PLANNER IS FIRST SEMANTIC AUTHORITY
# ============================================================

assert (
    "plan_analyses_with_ai("
    in route_segment
), (
    "/planning/ai-native-run must use the direct AI planner."
)


assert (
    "plan_analyses_with_intent_routing("
    not in route_segment
), (
    "/planning/ai-native-run must not run the deterministic "
    "generic resolver before Qwen."
)


print(
    "[PASS] Qwen planner is the native route semantic authority"
)


# ============================================================
# 4. NO SPECIALIZED DEDUP IN EXECUTION AUTHORITY
# ============================================================

assert (
    "remove_specialized_entity_outlier_duplicate("
    not in route_segment
), (
    "The native route must not delete a validated generic "
    "entity_outlier contract because a historical specialized "
    "finding exists."
)


assert (
    "execution_planner_report"
    not in route_segment
), (
    "The native route must execute the original validated "
    "Qwen planner report directly."
)


print(
    "[PASS] generic entity_outlier contract is not removed "
    "by specialized dedup"
)


# ============================================================
# 5. NO PLANNER FAILURE BYPASS
# ============================================================

for forbidden in (
    "entity_outlier_planner_bypass_v0.1",
    "specialized_entity_outlier_planner_bypass",
    "specialized_entity_outlier_fallback_allowed",
):
    assert (
        forbidden
        not in route_segment
    ), (
        "The AI-native route must fail closed when the Qwen "
        f"planner fails. Legacy bypass found: {forbidden}"
    )


print(
    "[PASS] native route contains no Python planner bypass"
)


# ============================================================
# 6. COMPATIBILITY FINDING IS AFTER QWEN PLANNING
# ============================================================

planner_pos = route_segment.index(
    "plan_analyses_with_ai("
)

coverage_pos = route_segment.index(
    "require_objective_coverage("
)

compatibility_pos = route_segment.index(
    "build_entity_outlier_finding_if_requested("
)

pipeline_pos = route_segment.index(
    "execute_native_ai_pipeline("
)


assert (
    planner_pos
    <
    coverage_pos
    <
    compatibility_pos
    <
    pipeline_pos
), (
    "Required order: Qwen planning -> objective coverage -> "
    "legacy compatibility finding -> native execution."
)


print(
    "[PASS] compatibility finding is resolved after Qwen planning"
)


# ============================================================
# 7. ORIGINAL QWEN REPORT ENTERS NATIVE PIPELINE
# ============================================================

native_call_segment = route_segment[
    pipeline_pos:
]


assert (
    "planner_report=\n                    planner_report,"
    in native_call_segment
), (
    "execute_native_ai_pipeline() must receive the original "
    "validated Qwen planner report."
)


print(
    "[PASS] native pipeline receives original Qwen planner report"
)


# ============================================================
# FINAL VERDICT
# ============================================================

print()
print(
    "PASS - AI-native entity outlier AI-first authority v0.1"
)
