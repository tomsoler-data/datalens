from __future__ import annotations

import ast
import inspect
from pathlib import Path

from app.ai.ai_native_pipeline import (
    AINativePipelineReport,
    execute_native_ai_pipeline,
)


ROOT = Path(__file__).resolve().parents[2]

PIPELINE_PATH = (
    ROOT
    / "app"
    / "ai"
    / "ai_native_pipeline.py"
)

API_PATH = (
    ROOT
    / "app"
    / "api"
    / "analysis_run.py"
)


print()
print("=" * 80)
print("DATALENS AI-NATIVE ENTITY OUTLIER WIRING v0.1")
print("=" * 80)
print()


# ============================================================
# 1. RESPONSE CONTRACT
# ============================================================

assert (
    "entity_outlier_finding"
    in AINativePipelineReport.model_fields
), (
    "AINativePipelineReport must expose an optional "
    "entity_outlier_finding field."
)

print(
    "[PASS] AI-native response contract carries entity-outlier finding"
)


# ============================================================
# 2. PIPELINE INPUT CONTRACT
# ============================================================

pipeline_signature = (
    inspect.signature(
        execute_native_ai_pipeline
    )
)

assert (
    "entity_outlier_finding"
    in pipeline_signature.parameters
), (
    "execute_native_ai_pipeline() must receive the already "
    "resolved entity_outlier_finding before persistence."
)

print(
    "[PASS] native pipeline accepts entity-outlier finding before persistence"
)


# ============================================================
# 3. PIPELINE REPORT CONSTRUCTION
# ============================================================

pipeline_source = (
    PIPELINE_PATH.read_text(
        encoding="utf-8"
    )
)

pipeline_tree = ast.parse(
    pipeline_source
)

pipeline_function = next(
    node
    for node in ast.walk(
        pipeline_tree
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
        "execute_native_ai_pipeline"
    )
)

pipeline_segment = ast.get_source_segment(
    pipeline_source,
    pipeline_function,
)

assert (
    pipeline_segment
    is not None
)

assert (
    "entity_outlier_finding="
    in pipeline_segment
), (
    "execute_native_ai_pipeline() must inject the finding "
    "into AINativePipelineReport before "
    "register_native_pipeline_result() serializes it."
)

assert (
    pipeline_segment.index(
        "entity_outlier_finding="
    )
    <
    pipeline_segment.index(
        "register_native_pipeline_result("
    )
), (
    "entity_outlier_finding must be attached before "
    "server-owned persistence."
)

print(
    "[PASS] entity-outlier finding is attached before artifact persistence"
)


# ============================================================
# 4. LIVE AI-NATIVE ROUTE
# ============================================================

api_source = (
    API_PATH.read_text(
        encoding="utf-8"
    )
)

route_start = (
    api_source.index(
        '"/planning/ai-native-run"'
    )
)

route_end = (
    api_source.index(
        "# AI OBSERVABILITY",
        route_start,
    )
)

route_segment = (
    api_source[
        route_start:
        route_end
    ]
)

assert (
    "build_entity_outlier_finding_if_requested("
    in route_segment
), (
    "/planning/ai-native-run must resolve the explicit "
    "entity-outlier branch in addition to the normal "
    "multi-intent analytical planner."
)

assert (
    route_segment.index(
        "build_entity_outlier_finding_if_requested("
    )
    <
    route_segment.index(
        "execute_native_ai_pipeline("
    )
), (
    "The specialized finding must be resolved before "
    "the native pipeline is executed."
)

assert (
    "entity_outlier_finding="
    in route_segment
), (
    "/planning/ai-native-run must pass the specialized "
    "finding into execute_native_ai_pipeline()."
)

print(
    "[PASS] live AI-native route preserves generic analyses plus entity outlier"
)


# ============================================================
# FINAL VERDICT
# ============================================================

print()
print(
    "PASS - AI-native entity outlier wiring v0.1"
)
