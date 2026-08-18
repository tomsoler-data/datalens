from __future__ import annotations

from pydantic import (
    BaseModel,
    Field,
)


# ============================================================
# CAPABILITY SLICE
# ============================================================

class CapabilitySliceSummary(
    BaseModel
):
    label: str

    fields: list[
        str
    ] = Field(
        default_factory=list,
    )

    assertion_count: int = Field(
        ge=0,
    )

    raw_correct_count: int = Field(
        ge=0,
    )

    datalens_correct_count: int = Field(
        ge=0,
    )

    raw_accuracy: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    datalens_accuracy: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    correct_count_delta: int

    accuracy_delta: float | None = None


# ============================================================
# CAPABILITY DECOMPOSITION
# ============================================================

class CapabilityDecompositionReport(
    BaseModel
):
    benchmark_id: str

    shared: CapabilitySliceSummary

    system_extensions: CapabilitySliceSummary

    end_to_end: CapabilitySliceSummary

    total_gain_count: int

    shared_gain_count: int

    extension_gain_count: int

    shared_gain_share: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    extension_gain_share: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    extension_fields: list[
        str
    ] = Field(
        default_factory=list,
    )

    decomposition_rule_version: str = (
        "evaluation_capability_decomposition_v0.1"
    )
