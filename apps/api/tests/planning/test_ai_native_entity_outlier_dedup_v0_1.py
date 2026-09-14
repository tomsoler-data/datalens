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
    "AI-FIRST AUTHORITY v0.2"
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
# 1. LEGACY HELPER MAY REMAIN OUTSIDE NATIVE AUTHORITY
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
    "Historical helper may remain while non-native legacy "
    "paths still exist."
)

print(
    "[PASS] historical helper remains isolated outside "
    "native authority"
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
# 4. NO SPECIALIZED DEDUP IN NATIVE EXECUTION AUTHORITY
# ============================================================

assert (
    "remove_specialized_entity_outlier_duplicate("
    not in route_segment
), (
    "The native route must not delete a validated generic "
    "entity_outlier contract because of a historical finding."
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
# 6. NO LEGACY ENTITY-OUTLIER PRODUCER
# ============================================================

assert (
    "build_entity_outlier_finding_if_requested("
    not in route_segment
), (
    "The native route must no longer calculate the historical "
    "customer-oriented entity_outlier_finding."
)

assert (
    "entity_outlier_finding="
    not in route_segment
), (
    "The native route must no longer inject the historical "
    "entity_outlier_finding into the native pipeline."
)

print(
    "[PASS] native route contains no legacy entity-outlier producer"
)


# ============================================================
# 7. ORDER: QWEN -> COVERAGE -> NATIVE PIPELINE
# ============================================================

planner_pos = route_segment.index(
    "plan_analyses_with_ai("
)

coverage_pos = route_segment.index(
    "require_objective_coverage("
)

pipeline_pos = route_segment.index(
    "execute_native_ai_pipeline("
)

assert (
    planner_pos
    <
    coverage_pos
    <
    pipeline_pos
), (
    "Required order: Qwen planning -> objective coverage -> "
    "native execution."
)

print(
    "[PASS] native route order is Qwen -> coverage -> execution"
)


# ============================================================
# 8. ORIGINAL QWEN REPORT ENTERS NATIVE PIPELINE
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
    "PASS - AI-native entity outlier AI-first authority v0.2"
)
