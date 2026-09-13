from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

API_PATH = (
    ROOT
    / "app"
    / "api"
    / "analysis_run.py"
)


print()
print("=" * 80)
print("DATALENS AI-NATIVE ENTITY OUTLIER DEDUP v0.1")
print("=" * 80)
print()


source = API_PATH.read_text(
    encoding="utf-8"
)

tree = ast.parse(
    source
)


# ============================================================
# 1. DEDUP HELPER MUST EXIST
# ============================================================

helper = next(
    (
        node
        for node in ast.walk(tree)
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
    helper is not None
), (
    "A deterministic helper must remove only the generic "
    "distribution proposal that duplicates an already-resolved "
    "specialized entity-outlier finding."
)

print(
    "[PASS] specialized entity-outlier dedup helper exists"
)


# ============================================================
# 2. HELPER MUST BE NARROWLY SCOPED
# ============================================================

helper_segment = ast.get_source_segment(
    source,
    helper,
)

assert (
    helper_segment
    is not None
)

assert (
    '"distribution"'
    in helper_segment
    or
    "'distribution'"
    in helper_segment
), (
    "Dedup must explicitly target the generic distribution family."
)

assert (
    "entity_outlier_finding"
    in helper_segment
), (
    "Dedup must depend on an already-resolved specialized finding."
)

assert (
    "dataset_id"
    in helper_segment
), (
    "Dedup must compare the specialized entity dataset with "
    "the generic proposal dataset instead of removing arbitrary "
    "distribution analyses."
)

print(
    "[PASS] dedup is scoped to matching distribution/entity dataset"
)


# ============================================================
# 3. LIVE ROUTE MUST APPLY DEDUP BEFORE NATIVE EXECUTION
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


assert (
    "remove_specialized_entity_outlier_duplicate("
    in route_segment
), (
    "/planning/ai-native-run must remove the generic duplicate "
    "after resolving the specialized entity-outlier finding."
)

assert (
    route_segment.index(
        "build_entity_outlier_finding_if_requested("
    )
    <
    route_segment.index(
        "remove_specialized_entity_outlier_duplicate("
    )
    <
    route_segment.index(
        "execute_native_ai_pipeline("
    )
), (
    "Required order: specialized finding resolution -> "
    "deduplication -> native execution."
)

print(
    "[PASS] live route deduplicates before native execution"
)


# ============================================================
# 4. OBJECTIVE COVERAGE MUST REMAIN BEFORE DEDUP
# ============================================================

assert (
    route_segment.index(
        "require_objective_coverage("
    )
    <
    route_segment.index(
        "remove_specialized_entity_outlier_duplicate("
    )
), (
    "Dedup must not weaken the existing objective-coverage gate."
)

print(
    "[PASS] objective coverage gate remains intact before dedup"
)


print()
print(
    "PASS - AI-native entity outlier dedup v0.1"
)
