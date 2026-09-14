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
print(
    "DATALENS AI-NATIVE ENTITY OUTLIER "
    "LEGACY PRODUCER REMOVAL v0.1"
)
print("=" * 80)
print()


# ============================================================
# 1. HISTORICAL RESPONSE CONTRACT REMAINS READABLE
# ============================================================

assert (
    "entity_outlier_finding"
    in AINativePipelineReport.model_fields
), (
    "AINativePipelineReport must keep the optional historical "
    "entity_outlier_finding field while old artifacts exist."
)

field = (
    AINativePipelineReport
    .model_fields[
        "entity_outlier_finding"
    ]
)

assert (
    field.default
    is None
), (
    "Historical entity_outlier_finding compatibility must "
    "default to None for new native reports."
)

print(
    "[PASS] historical response field remains readable "
    "and defaults to None"
)


# ============================================================
# 2. NATIVE PIPELINE NO LONGER ACCEPTS LEGACY FINDING
# ============================================================

pipeline_signature = (
    inspect.signature(
        execute_native_ai_pipeline
    )
)

assert (
    "entity_outlier_finding"
    not in pipeline_signature.parameters
), (
    "execute_native_ai_pipeline() must no longer accept "
    "the historical specialized finding as an input."
)

print(
    "[PASS] native pipeline has no legacy finding input"
)


# ============================================================
# 3. NATIVE PIPELINE DOES NOT PRODUCE LEGACY FINDING
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
    for node
    in ast.walk(
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
    "entity_outlier_finding"
    not in pipeline_segment
), (
    "New native pipeline execution must not construct "
    "or inject the historical specialized finding."
)

print(
    "[PASS] native pipeline does not produce legacy finding"
)


# ============================================================
# 4. LIVE AI-NATIVE ROUTE HAS NO LEGACY PRODUCER
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
    "execute_native_ai_pipeline("
    in route_segment
)

assert (
    "build_entity_outlier_finding_if_requested("
    not in route_segment
), (
    "/planning/ai-native-run must not execute the "
    "historical customer-oriented entity-outlier producer."
)

assert (
    "entity_outlier_finding="
    not in route_segment
), (
    "/planning/ai-native-run must not pass a historical "
    "entity-outlier compatibility payload into the native "
    "pipeline."
)

print(
    "[PASS] live AI-native route produces only native "
    "AI-first analytical results"
)


# ============================================================
# FINAL VERDICT
# ============================================================

print()
print(
    "PASS - AI-native entity outlier legacy producer removal v0.1"
)
