from __future__ import annotations


import json

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
    "ai_trace_store_v0.1"
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


# ============================================================
# PUBLIC READ API
# ============================================================

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
