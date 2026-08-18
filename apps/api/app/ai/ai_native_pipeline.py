from __future__ import annotations


from time import (
    perf_counter,
)

from typing import (
    Any,
    Literal,
)


from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)


from app.ai.native_tool_calling import (
    DEFAULT_NATIVE_TOOL_MODEL,
    NativeToolCallingReport,
    SUPPORTED_NATIVE_FAMILIES,
    run_native_tool_call,
)

from app.planning.ai_analytical_planner import (
    AIPlannerReport,
)


# ============================================================
# VERSION
# ============================================================

AI_NATIVE_PIPELINE_RULE_VERSION = (
    "ai_native_pipeline_v0.4"
)


# ============================================================
# PIPELINE RESULT
# ============================================================

PipelineItemStatus = Literal[
    "executed",
    "not_supported",
    "rejected",
]


class AINativePipelineItem(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid"
    )


    contract_id: str

    family: str

    pipeline_status: (
        PipelineItemStatus
    )

    native_tool: (
        NativeToolCallingReport
        | None
    ) = None

    errors: list[
        str
    ] = Field(
        default_factory=list
    )

    warnings: list[
        str
    ] = Field(
        default_factory=list
    )


class AINativePipelineTiming(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid"
    )


    tool_prompt_construction_ms: float = Field(
        default=0.0,
        ge=0.0,
    )

    tool_model_inference_ms: float = Field(
        default=0.0,
        ge=0.0,
    )

    tool_response_parse_ms: float = Field(
        default=0.0,
        ge=0.0,
    )

    tool_python_validation_ms: float = Field(
        default=0.0,
        ge=0.0,
    )

    deterministic_execution_ms: float = Field(
        default=0.0,
        ge=0.0,
    )

    total_ms: float = Field(
        default=0.0,
        ge=0.0,
    )


class AINativePipelineReport(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid"
    )


    status: Literal[
        "ready"
    ] = "ready"

    trace_id: (
        str
        | None
    ) = None

    planner: AIPlannerReport

    planner_model: str

    tool_model: str

    supported_native_families: list[
        str
    ] = Field(
        default_factory=list
    )

    validated_contract_count: int

    pipeline_item_count: int

    executed_count: int

    not_supported_count: int

    rejected_count: int

    items: list[
        AINativePipelineItem
    ]

    timing: AINativePipelineTiming = Field(
        default_factory=
            AINativePipelineTiming
    )

    notes: list[
        str
    ] = Field(
        default_factory=list
    )

    pipeline_rule_version: str = (
        AI_NATIVE_PIPELINE_RULE_VERSION
    )


# ============================================================
# PIPELINE
# ============================================================

def execute_native_ai_pipeline(
    *,
    planner_report: AIPlannerReport,
    datasets: list[
        dict[
            str,
            Any,
        ]
    ],
    tool_model: str = (
        DEFAULT_NATIVE_TOOL_MODEL
    ),
    trace_id: (
        str
        | None
    ) = None,
) -> AINativePipelineReport:
    pipeline_started_at = (
        perf_counter()
    )


    validated_contracts = [
        item.contract

        for item
        in planner_report.items

        if (
            item.validation_status ==
            "validated"
            and
            item.contract is not None
        )
    ]


    items: list[
        AINativePipelineItem
    ] = []


    for contract in (
        validated_contracts
    ):
        if (
            contract.family
            not in SUPPORTED_NATIVE_FAMILIES
        ):
            items.append(
                AINativePipelineItem(
                    contract_id=(
                        contract
                        .contract_id
                    ),
                    family=(
                        contract
                        .family
                    ),
                    pipeline_status=(
                        "not_supported"
                    ),
                    native_tool=None,
                    warnings=[
                        (
                            "The validated analytical family "
                            "does not yet have a native "
                            "function-calling implementation."
                        ),
                    ],
                )
            )

            continue


        native_report = (
            run_native_tool_call(
                contract=(
                    contract
                ),
                datasets=(
                    datasets
                ),
                model=(
                    tool_model
                ),
            )
        )


        if (
            native_report
            .validation_status
            !=
            "validated"
        ):
            items.append(
                AINativePipelineItem(
                    contract_id=(
                        contract
                        .contract_id
                    ),
                    family=(
                        contract
                        .family
                    ),
                    pipeline_status=(
                        "rejected"
                    ),
                    native_tool=(
                        native_report
                    ),
                    errors=list(
                        native_report
                        .validation_errors
                    ),
                )
            )

            continue


        execution = (
            native_report
            .execution
        )


        if (
            execution is None
            or
            execution
            .execution_status
            !=
            "executed"
        ):
            items.append(
                AINativePipelineItem(
                    contract_id=(
                        contract
                        .contract_id
                    ),
                    family=(
                        contract
                        .family
                    ),
                    pipeline_status=(
                        "rejected"
                    ),
                    native_tool=(
                        native_report
                    ),
                    errors=[
                        (
                            "The native tool call was "
                            "validated, but deterministic "
                            "execution did not complete."
                        ),
                    ],
                )
            )

            continue


        items.append(
            AINativePipelineItem(
                contract_id=(
                    contract
                    .contract_id
                ),
                family=(
                    contract
                    .family
                ),
                pipeline_status=(
                    "executed"
                ),
                native_tool=(
                    native_report
                ),
                warnings=[
                    (
                        "The planner model and native "
                        "tool model had separate roles. "
                        "Python validated both boundaries "
                        "before deterministic execution."
                    ),
                ],
            )
        )


    native_reports = [
        item.native_tool
        for item
        in items
        if item.native_tool is not None
    ]


    timing = (
        AINativePipelineTiming(
            tool_prompt_construction_ms=sum(
                report.timing.prompt_construction_ms
                for report
                in native_reports
            ),
            tool_model_inference_ms=sum(
                report.timing.model_inference_ms
                for report
                in native_reports
            ),
            tool_response_parse_ms=sum(
                report.timing.response_parse_ms
                for report
                in native_reports
            ),
            tool_python_validation_ms=sum(
                report.timing.python_validation_ms
                for report
                in native_reports
            ),
            deterministic_execution_ms=sum(
                report.timing.deterministic_execution_ms
                for report
                in native_reports
            ),
            total_ms=(
                (
                    perf_counter()
                    -
                    pipeline_started_at
                )
                *
                1000.0
            ),
        )
    )


    return (
        AINativePipelineReport(
            trace_id=(
                trace_id
            ),
            planner=(
                planner_report
            ),
            planner_model=(
                planner_report
                .model
            ),
            tool_model=(
                tool_model
            ),
            supported_native_families=(
                sorted(
                    SUPPORTED_NATIVE_FAMILIES
                )
            ),
            validated_contract_count=(
                len(
                    validated_contracts
                )
            ),
            pipeline_item_count=(
                len(
                    items
                )
            ),
            executed_count=sum(
                1
                for item
                in items
                if (
                    item.pipeline_status ==
                    "executed"
                )
            ),
            not_supported_count=sum(
                1
                for item
                in items
                if (
                    item.pipeline_status ==
                    "not_supported"
                )
            ),
            rejected_count=sum(
                1
                for item
                in items
                if (
                    item.pipeline_status ==
                    "rejected"
                )
            ),
            items=(
                items
            ),
            timing=(
                timing
            ),
            notes=[
                (
                    "Stage 1: Gemma proposes a structured "
                    "analytical contract from the user "
                    "objective and dataset catalog."
                ),
                (
                    "Stage 2: Python validates dataset ids, "
                    "column names, analytical types and "
                    "contract invariants."
                ),
                (
                    "Stage 3: Qwen selects one matching "
                    "function from the native DataLens "
                    "tool catalog."
                ),
                (
                    "Stage 4: Python verifies the selected "
                    "tool against the validated family and "
                    "checks every argument."
                ),
                (
                    "Stage 5: the deterministic DataLens "
                    "executor computes the result."
                ),
            ],
        )
    )
