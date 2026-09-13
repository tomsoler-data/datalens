from __future__ import annotations

from app.ai.ai_native_pipeline import (
    AINativePipelineReport,
)


print()
print("=" * 80)
print("DATALENS AI-NATIVE ENTITY OUTLIER SERIALIZATION v0.1")
print("=" * 80)
print()


finding = {
    "analysis_id":
        "analysis:entity-outlier:test",

    "title":
        "Clients atypiques",

    "summary":
        "Des profils clients atypiques ont été détectés.",

    "details": {
        "entity_kind":
            "customer",

        "entity_column":
            "client_id",

        "priority_profile_count":
            10,
    },
}


report = (
    AINativePipelineReport.model_construct(
        status=
            "ready",

        trace_id=
            "trace:test-entity-outlier",

        analysis_id=
            None,

        analysis_source_type=
            None,

        entity_outlier_finding=
            finding,

        planner_model=
            "qwen3.5:4b",

        tool_model=
            "qwen3.5:4b",

        supported_native_families=
            [],

        validated_contract_count=
            0,

        pipeline_item_count=
            0,

        executed_count=
            0,

        not_supported_count=
            0,

        rejected_count=
            0,

        items=
            [],

        notes=
            [],

        pipeline_rule_version=
            "test",
    )
)


payload = (
    report.model_dump(
        mode="json"
    )
)


assert (
    "entity_outlier_finding"
    in payload
), (
    "Serialized AI-native payload lost "
    "entity_outlier_finding."
)


assert (
    payload[
        "entity_outlier_finding"
    ]
    ==
    finding
), (
    "Serialized entity-outlier finding "
    "does not match the server-owned finding."
)


assert (
    payload[
        "entity_outlier_finding"
    ][
        "details"
    ][
        "entity_column"
    ]
    ==
    "client_id"
)


assert (
    payload[
        "entity_outlier_finding"
    ][
        "details"
    ][
        "priority_profile_count"
    ]
    ==
    10
)


print(
    "[PASS] entity-outlier finding survives "
    "AINativePipelineReport.model_dump(mode='json')"
)

print(
    "[PASS] entity grain remains client_id"
)

print(
    "[PASS] specialized finding remains distinct "
    "from generic pipeline items"
)

print()
print(
    "PASS - AI-native entity outlier serialization v0.1"
)
