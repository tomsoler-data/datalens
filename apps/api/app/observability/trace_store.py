from __future__ import annotations


import json
import math

from collections import (
    Counter,
)

from pathlib import Path
from typing import (
    Any,
)


from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)


from app.observability.ai_trace import (
    AITraceRecord,
    resolve_ai_trace_path,
)


AI_TRACE_STORE_RULE_VERSION = (
    "ai_trace_store_v0.4"
)


# ============================================================
# RESPONSE SCHEMAS
# ============================================================

class AITraceSummary(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid"
    )


    trace_id: str

    created_at_utc: str

    trace_rule_version: str

    objective: str

    dataset_filenames: list[
        str
    ] = Field(
        default_factory=list
    )

    planner_status: (
        str
        | None
    ) = None

    planner_model: (
        str
        | None
    ) = None

    planner_rule_version: (
        str
        | None
    ) = None

    planner_attempt_count: (
        int
        | None
    ) = None

    planner_retry_count: (
        int
        | None
    ) = None

    planner_normalization_count: (
        int
        | None
    ) = None

    pipeline_status: (
        str
        | None
    ) = None

    pipeline_rule_version: (
        str
        | None
    ) = None

    tool_model: (
        str
        | None
    ) = None

    executed_count: (
        int
        | None
    ) = None

    total_ms: float


class AITraceListResponse(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid"
    )


    trace_store_rule_version: str = (
        AI_TRACE_STORE_RULE_VERSION
    )

    path: str

    trace_count: int

    malformed_line_count: int

    returned_count: int

    traces: list[
        AITraceSummary
    ] = Field(
        default_factory=list
    )



class AITraceLatencyMetrics(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid"
    )


    median_ms: float

    p95_ms: float

    mean_ms: float


class AITraceCategoryCount(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid"
    )


    name: str

    count: int


class AITraceMetricsResponse(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid"
    )


    trace_store_rule_version: str = (
        AI_TRACE_STORE_RULE_VERSION
    )

    path: str

    trace_count: int

    malformed_line_count: int

    analyzed_trace_count: int

    detailed_trace_count: int

    executed_trace_count: int

    execution_rate: float

    planner_retry_trace_count: int

    planner_retry_rate: float

    planner_normalized_trace_count: int

    planner_normalization_rate: float

    planner_share_median: float

    total_latency: AITraceLatencyMetrics

    planner_latency: AITraceLatencyMetrics

    native_pipeline_latency: AITraceLatencyMetrics

    ingestion_latency: AITraceLatencyMetrics

    planner_prompt_latency: AITraceLatencyMetrics

    planner_model_inference_latency: AITraceLatencyMetrics

    planner_structured_parse_latency: AITraceLatencyMetrics

    planner_python_validation_latency: AITraceLatencyMetrics

    planner_retry_feedback_latency: AITraceLatencyMetrics

    tool_prompt_latency: AITraceLatencyMetrics

    tool_model_inference_latency: AITraceLatencyMetrics

    tool_response_parse_latency: AITraceLatencyMetrics

    tool_python_validation_latency: AITraceLatencyMetrics

    deterministic_execution_latency: AITraceLatencyMetrics

    planner_model_share_median: float

    tool_model_share_median: float

    deterministic_execution_share_median: float

    planner_models: list[
        AITraceCategoryCount
    ] = Field(
        default_factory=list
    )

    tool_models: list[
        AITraceCategoryCount
    ] = Field(
        default_factory=list
    )

    families: list[
        AITraceCategoryCount
    ] = Field(
        default_factory=list
    )

    requested_tools: list[
        AITraceCategoryCount
    ] = Field(
        default_factory=list
    )

    pipeline_statuses: list[
        AITraceCategoryCount
    ] = Field(
        default_factory=list
    )


# ============================================================
# PRIVATE HELPERS
# ============================================================

def _read_trace_lines(
    path: Path,
) -> tuple[
    list[
        AITraceRecord
    ],
    int,
]:
    if not path.exists():
        return (
            [],
            0,
        )


    traces: list[
        AITraceRecord
    ] = []


    malformed_line_count = 0


    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        for line in handle:
            normalized = (
                line.strip()
            )


            if not normalized:
                continue


            try:
                payload = (
                    json.loads(
                        normalized
                    )
                )


                trace = (
                    AITraceRecord
                    .model_validate(
                        payload
                    )
                )


            except Exception:
                # Reading observability data must remain robust
                # even if one historical line is malformed.
                malformed_line_count += 1

                continue


            traces.append(
                trace
            )


    return (
        traces,
        malformed_line_count,
    )


def _summary_from_trace(
    trace: AITraceRecord,
) -> AITraceSummary:
    planner = (
        trace.planner
        or {}
    )


    pipeline = (
        trace.native_pipeline
        or {}
    )


    dataset_filenames = [
        str(
            dataset.get(
                "filename",
                "",
            )
        )

        for dataset
        in trace.datasets

        if str(
            dataset.get(
                "filename",
                "",
            )
        )
    ]


    return (
        AITraceSummary(
            trace_id=
                trace.trace_id,

            created_at_utc=
                trace.created_at_utc,

            trace_rule_version=
                trace.trace_rule_version,

            objective=
                trace.objective,

            dataset_filenames=
                dataset_filenames,

            planner_status=
                planner.get(
                    "status"
                ),

            planner_model=
                planner.get(
                    "model"
                ),

            planner_rule_version=
                planner.get(
                    "planner_rule_version"
                ),

            planner_attempt_count=
                planner.get(
                    "attempt_count"
                ),

            planner_retry_count=
                planner.get(
                    "retry_count"
                ),

            planner_normalization_count=
                planner.get(
                    "normalization_count"
                ),

            pipeline_status=
                pipeline.get(
                    "status"
                ),

            pipeline_rule_version=
                pipeline.get(
                    "pipeline_rule_version"
                ),

            tool_model=
                pipeline.get(
                    "tool_model"
                ),

            executed_count=
                pipeline.get(
                    "executed_count"
                ),

            total_ms=
                trace.timings.total_ms,
        )
    )



def _percentile_linear(
    values: list[
        float
    ],
    percentile: float,
) -> float:
    if not values:
        return 0.0


    ordered = sorted(
        values
    )


    if len(
        ordered
    ) == 1:
        return float(
            ordered[
                0
            ]
        )


    position = (
        (
            len(
                ordered
            )
            -
            1
        )
        *
        percentile
    )


    lower_index = int(
        math.floor(
            position
        )
    )


    upper_index = int(
        math.ceil(
            position
        )
    )


    if (
        lower_index
        ==
        upper_index
    ):
        return float(
            ordered[
                lower_index
            ]
        )


    weight = (
        position
        -
        lower_index
    )


    return float(
        ordered[
            lower_index
        ]
        *
        (
            1.0
            -
            weight
        )
        +
        ordered[
            upper_index
        ]
        *
        weight
    )


def _latency_metrics(
    values: list[
        float
    ],
) -> AITraceLatencyMetrics:
    normalized = [
        float(
            value
        )

        for value
        in values

        if (
            isinstance(
                value,
                (
                    int,
                    float,
                ),
            )
            and
            math.isfinite(
                float(
                    value
                )
            )
            and
            float(
                value
            ) >=
            0.0
        )
    ]


    if not normalized:
        return (
            AITraceLatencyMetrics(
                median_ms=
                    0.0,

                p95_ms=
                    0.0,

                mean_ms=
                    0.0,
            )
        )


    ordered = sorted(
        normalized
    )


    middle = len(
        ordered
    ) // 2


    if (
        len(
            ordered
        )
        %
        2
    ):
        median = float(
            ordered[
                middle
            ]
        )


    else:
        median = float(
            (
                ordered[
                    middle -
                    1
                ]
                +
                ordered[
                    middle
                ]
            )
            /
            2.0
        )


    return (
        AITraceLatencyMetrics(
            median_ms=
                round(
                    median,
                    3,
                ),

            p95_ms=
                round(
                    _percentile_linear(
                        normalized,
                        0.95,
                    ),
                    3,
                ),

            mean_ms=
                round(
                    sum(
                        normalized
                    )
                    /
                    len(
                        normalized
                    ),
                    3,
                ),
        )
    )


def _category_counts(
    counter: Counter[
        str
    ],
) -> list[
    AITraceCategoryCount
]:
    return [
        AITraceCategoryCount(
            name=
                name,

            count=
                count,
        )

        for (
            name,
            count,
        )
        in sorted(
            counter.items(),
            key=lambda item: (
                -item[
                    1
                ],
                item[
                    0
                ],
            ),
        )
    ]


# ============================================================
# PUBLIC READ API
# ============================================================

def get_ai_trace_metrics(
    *,
    limit: int = 200,
    path: (
        Path
        | None
    ) = None,
) -> AITraceMetricsResponse:
    if (
        limit < 1
        or
        limit > 5000
    ):
        raise ValueError(
            "limit must be between 1 and 5000."
        )


    resolved_path = (
        (
            path.resolve()
        )
        if path is not None
        else
        resolve_ai_trace_path()
    )


    (
        traces,
        malformed_line_count,
    ) = _read_trace_lines(
        resolved_path
    )


    selected = (
        traces[
            -limit:
        ]
    )


    planner_model_counter: Counter[
        str
    ] = Counter()


    tool_model_counter: Counter[
        str
    ] = Counter()


    family_counter: Counter[
        str
    ] = Counter()


    requested_tool_counter: Counter[
        str
    ] = Counter()


    pipeline_status_counter: Counter[
        str
    ] = Counter()


    total_latencies: list[
        float
    ] = []


    planner_latencies: list[
        float
    ] = []


    native_latencies: list[
        float
    ] = []


    ingestion_latencies: list[
        float
    ] = []


    planner_prompt_latencies: list[float] = []
    planner_model_inference_latencies: list[float] = []
    planner_structured_parse_latencies: list[float] = []
    planner_python_validation_latencies: list[float] = []
    planner_retry_feedback_latencies: list[float] = []

    tool_prompt_latencies: list[float] = []
    tool_model_inference_latencies: list[float] = []
    tool_response_parse_latencies: list[float] = []
    tool_python_validation_latencies: list[float] = []
    deterministic_execution_latencies: list[float] = []


    planner_shares: list[
        float
    ] = []


    planner_model_shares: list[float] = []
    tool_model_shares: list[float] = []
    deterministic_execution_shares: list[float] = []


    executed_trace_count = 0

    planner_retry_trace_count = 0

    planner_normalized_trace_count = 0

    detailed_trace_count = 0


    for trace in selected:
        planner = (
            trace.planner
            or {}
        )


        pipeline = (
            trace.native_pipeline
            or {}
        )


        planner_model = (
            planner.get(
                "model"
            )
            or
            pipeline.get(
                "planner_model"
            )
        )


        tool_model = (
            pipeline.get(
                "tool_model"
            )
        )


        if isinstance(
            planner_model,
            str,
        ) and planner_model:
            planner_model_counter[
                planner_model
            ] += 1


        if isinstance(
            tool_model,
            str,
        ) and tool_model:
            tool_model_counter[
                tool_model
            ] += 1


        pipeline_status = (
            pipeline.get(
                "status"
            )
        )


        if isinstance(
            pipeline_status,
            str,
        ) and pipeline_status:
            pipeline_status_counter[
                pipeline_status
            ] += 1


        executed_count = (
            pipeline.get(
                "executed_count"
            )
            or 0
        )


        if (
            isinstance(
                executed_count,
                int,
            )
            and
            executed_count >
            0
        ):
            executed_trace_count += 1


        retry_count = (
            planner.get(
                "retry_count"
            )
            or 0
        )


        if (
            isinstance(
                retry_count,
                int,
            )
            and
            retry_count >
            0
        ):
            planner_retry_trace_count += 1


        normalization_count = (
            planner.get(
                "normalization_count"
            )
            or 0
        )


        if (
            isinstance(
                normalization_count,
                int,
            )
            and
            normalization_count >
            0
        ):
            planner_normalized_trace_count += 1


        for item in (
            pipeline.get(
                "items",
                [],
            )
            or []
        ):
            if not isinstance(
                item,
                dict,
            ):
                continue


            family = (
                item.get(
                    "family"
                )
            )


            if isinstance(
                family,
                str,
            ) and family:
                family_counter[
                    family
                ] += 1


            tool_call = (
                item.get(
                    "tool_call"
                )
                or {}
            )


            if isinstance(
                tool_call,
                dict,
            ):
                requested_tool = (
                    tool_call.get(
                        "requested_tool"
                    )
                )


                if isinstance(
                    requested_tool,
                    str,
                ) and requested_tool:
                    requested_tool_counter[
                        requested_tool
                    ] += 1


        total_ms = float(
            trace.timings.total_ms
        )

        planner_ms = float(
            trace.timings.planner_ms
        )

        native_ms = float(
            trace.timings.native_pipeline_ms
        )

        ingestion_ms = float(
            trace.timings.ingestion_ms
        )


        total_latencies.append(
            total_ms
        )

        planner_latencies.append(
            planner_ms
        )

        native_latencies.append(
            native_ms
        )

        ingestion_latencies.append(
            ingestion_ms
        )


        planner_prompt_ms = float(
            trace.timings.planner_prompt_construction_ms
        )
        planner_model_inference_ms = float(
            trace.timings.planner_model_inference_ms
        )
        planner_structured_parse_ms = float(
            trace.timings.planner_structured_parse_ms
        )
        planner_python_validation_ms = float(
            trace.timings.planner_python_validation_ms
        )
        planner_retry_feedback_ms = float(
            trace.timings.planner_retry_feedback_ms
        )

        tool_prompt_ms = float(
            trace.timings.tool_prompt_construction_ms
        )
        tool_model_inference_ms = float(
            trace.timings.tool_model_inference_ms
        )
        tool_response_parse_ms = float(
            trace.timings.tool_response_parse_ms
        )
        tool_python_validation_ms = float(
            trace.timings.tool_python_validation_ms
        )
        deterministic_execution_ms = float(
            trace.timings.deterministic_execution_ms
        )


        detailed_total_ms = (
            planner_prompt_ms
            +
            planner_model_inference_ms
            +
            planner_structured_parse_ms
            +
            planner_python_validation_ms
            +
            planner_retry_feedback_ms
            +
            tool_prompt_ms
            +
            tool_model_inference_ms
            +
            tool_response_parse_ms
            +
            tool_python_validation_ms
            +
            deterministic_execution_ms
        )


        if detailed_total_ms > 0.0:
            detailed_trace_count += 1

            planner_prompt_latencies.append(
                planner_prompt_ms
            )
            planner_model_inference_latencies.append(
                planner_model_inference_ms
            )
            planner_structured_parse_latencies.append(
                planner_structured_parse_ms
            )
            planner_python_validation_latencies.append(
                planner_python_validation_ms
            )
            planner_retry_feedback_latencies.append(
                planner_retry_feedback_ms
            )

            tool_prompt_latencies.append(
                tool_prompt_ms
            )
            tool_model_inference_latencies.append(
                tool_model_inference_ms
            )
            tool_response_parse_latencies.append(
                tool_response_parse_ms
            )
            tool_python_validation_latencies.append(
                tool_python_validation_ms
            )
            deterministic_execution_latencies.append(
                deterministic_execution_ms
            )


        if total_ms > 0.0:
            planner_shares.append(
                (
                    planner_ms /
                    total_ms
                )
                *
                100.0
            )


        if (
            detailed_total_ms > 0.0
            and
            total_ms > 0.0
        ):
            planner_model_shares.append(
                (
                    planner_model_inference_ms /
                    total_ms
                )
                *
                100.0
            )

            tool_model_shares.append(
                (
                    tool_model_inference_ms /
                    total_ms
                )
                *
                100.0
            )

            deterministic_execution_shares.append(
                (
                    deterministic_execution_ms /
                    total_ms
                )
                *
                100.0
            )


    analyzed_trace_count = len(
        selected
    )


    execution_rate = (
        (
            executed_trace_count /
            analyzed_trace_count
        )
        if analyzed_trace_count
        else
        0.0
    )


    planner_retry_rate = (
        (
            planner_retry_trace_count /
            analyzed_trace_count
        )
        if analyzed_trace_count
        else
        0.0
    )


    planner_normalization_rate = (
        (
            planner_normalized_trace_count /
            analyzed_trace_count
        )
        if analyzed_trace_count
        else
        0.0
    )


    planner_share_median = (
        _latency_metrics(
            planner_shares
        )
        .median_ms
    )


    return (
        AITraceMetricsResponse(
            path=str(
                resolved_path
            ),

            trace_count=len(
                traces
            ),

            malformed_line_count=
                malformed_line_count,

            analyzed_trace_count=
                analyzed_trace_count,

            detailed_trace_count=
                detailed_trace_count,

            executed_trace_count=
                executed_trace_count,

            execution_rate=
                round(
                    execution_rate,
                    6,
                ),

            planner_retry_trace_count=
                planner_retry_trace_count,

            planner_retry_rate=
                round(
                    planner_retry_rate,
                    6,
                ),

            planner_normalized_trace_count=
                planner_normalized_trace_count,

            planner_normalization_rate=
                round(
                    planner_normalization_rate,
                    6,
                ),

            planner_share_median=
                round(
                    planner_share_median,
                    3,
                ),

            total_latency=
                _latency_metrics(
                    total_latencies
                ),

            planner_latency=
                _latency_metrics(
                    planner_latencies
                ),

            native_pipeline_latency=
                _latency_metrics(
                    native_latencies
                ),

            ingestion_latency=
                _latency_metrics(
                    ingestion_latencies
                ),

            planner_prompt_latency=
                _latency_metrics(
                    planner_prompt_latencies
                ),

            planner_model_inference_latency=
                _latency_metrics(
                    planner_model_inference_latencies
                ),

            planner_structured_parse_latency=
                _latency_metrics(
                    planner_structured_parse_latencies
                ),

            planner_python_validation_latency=
                _latency_metrics(
                    planner_python_validation_latencies
                ),

            planner_retry_feedback_latency=
                _latency_metrics(
                    planner_retry_feedback_latencies
                ),

            tool_prompt_latency=
                _latency_metrics(
                    tool_prompt_latencies
                ),

            tool_model_inference_latency=
                _latency_metrics(
                    tool_model_inference_latencies
                ),

            tool_response_parse_latency=
                _latency_metrics(
                    tool_response_parse_latencies
                ),

            tool_python_validation_latency=
                _latency_metrics(
                    tool_python_validation_latencies
                ),

            deterministic_execution_latency=
                _latency_metrics(
                    deterministic_execution_latencies
                ),

            planner_model_share_median=
                round(
                    _latency_metrics(
                        planner_model_shares
                    ).median_ms,
                    3,
                ),

            tool_model_share_median=
                round(
                    _latency_metrics(
                        tool_model_shares
                    ).median_ms,
                    3,
                ),

            deterministic_execution_share_median=
                round(
                    _latency_metrics(
                        deterministic_execution_shares
                    ).median_ms,
                    3,
                ),

            planner_models=
                _category_counts(
                    planner_model_counter
                ),

            tool_models=
                _category_counts(
                    tool_model_counter
                ),

            families=
                _category_counts(
                    family_counter
                ),

            requested_tools=
                _category_counts(
                    requested_tool_counter
                ),

            pipeline_statuses=
                _category_counts(
                    pipeline_status_counter
                ),
        )
    )


def list_ai_traces(
    *,
    limit: int = 20,
    path: (
        Path
        | None
    ) = None,
) -> AITraceListResponse:
    if (
        limit < 1
        or
        limit > 200
    ):
        raise ValueError(
            "limit must be between 1 and 200."
        )


    resolved_path = (
        (
            path.resolve()
        )
        if path is not None
        else
        resolve_ai_trace_path()
    )


    (
        traces,
        malformed_line_count,
    ) = _read_trace_lines(
        resolved_path
    )


    selected = list(
        reversed(
            traces[
                -limit:
            ]
        )
    )


    summaries = [
        _summary_from_trace(
            trace
        )

        for trace
        in selected
    ]


    return (
        AITraceListResponse(
            path=str(
                resolved_path
            ),

            trace_count=len(
                traces
            ),

            malformed_line_count=
                malformed_line_count,

            returned_count=len(
                summaries
            ),

            traces=
                summaries,
        )
    )


def get_latest_ai_trace(
    *,
    path: (
        Path
        | None
    ) = None,
) -> (
    AITraceRecord
    | None
):
    resolved_path = (
        (
            path.resolve()
        )
        if path is not None
        else
        resolve_ai_trace_path()
    )


    (
        traces,
        _,
    ) = _read_trace_lines(
        resolved_path
    )


    if not traces:
        return None


    return traces[
        -1
    ]


def get_ai_trace(
    trace_id: str,
    *,
    path: (
        Path
        | None
    ) = None,
) -> (
    AITraceRecord
    | None
):
    normalized_trace_id = (
        trace_id.strip()
    )


    if not normalized_trace_id:
        return None


    resolved_path = (
        (
            path.resolve()
        )
        if path is not None
        else
        resolve_ai_trace_path()
    )


    (
        traces,
        _,
    ) = _read_trace_lines(
        resolved_path
    )


    for trace in reversed(
        traces
    ):
        if (
            trace.trace_id
            ==
            normalized_trace_id
        ):
            return trace


    return None
