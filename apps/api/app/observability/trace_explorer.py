from __future__ import annotations

import json
import re

from pathlib import Path

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)

from app.observability.ai_trace import (
    AITraceRecord,
    resolve_ai_trace_path,
)

from app.observability.runtime_trace import (
    RuntimeTraceRecord,
    resolve_runtime_trace_path,
)


TRACE_EXPLORER_RULE_VERSION = (
    "trace_explorer_v0.1"
)


_SAFE_RUNTIME_REQUEST_ID_RE = re.compile(
    r"^http:[0-9a-f]{32}$"
)


class TraceExplorerReadError(
    RuntimeError
):
    """
    Internal read failure for local observability data.

    Public HTTP handlers must translate this exception to a
    static message and must never expose filesystem details.
    """


class TraceExplorerRuntimeSummary(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid"
    )

    created_at_utc: str

    trace_rule_version: str

    method: str

    route_template: str

    status_code: int = Field(
        ge=100,
        le=599,
    )

    duration_ms: float = Field(
        ge=0.0
    )

    workflow_id: (
        str
        | None
    ) = None

    run_status: str

    failure_kind: (
        str
        | None
    ) = None


class TraceExplorerAISummary(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid"
    )

    trace_id: str

    created_at_utc: str

    trace_rule_version: str

    workflow_id: (
        str
        | None
    ) = None

    analysis_id: (
        str
        | None
    ) = None

    analysis_source_type: (
        str
        | None
    ) = None

    run_status: str

    failure_stage: (
        str
        | None
    ) = None

    total_ms: float = Field(
        ge=0.0
    )


class TraceExplorerResponse(
    BaseModel
):
    model_config = ConfigDict(
        extra="forbid"
    )

    explorer_rule_version: str = (
        TRACE_EXPLORER_RULE_VERSION
    )

    request_id: str

    runtime_found: bool

    runtime: (
        TraceExplorerRuntimeSummary
        | None
    ) = None

    runtime_malformed_line_count: int = Field(
        ge=0
    )

    ai_trace_count: int = Field(
        ge=0
    )

    ai_malformed_line_count: int = Field(
        ge=0
    )

    ai_traces: list[
        TraceExplorerAISummary
    ] = Field(
        default_factory=list
    )


def normalize_runtime_request_id(
    request_id: str,
) -> str:
    if not isinstance(
        request_id,
        str,
    ):
        raise ValueError(
            (
                "request_id must be a DataLens "
                "server-owned HTTP request identifier."
            )
        )

    normalized = (
        request_id
        .strip()
    )

    if not _SAFE_RUNTIME_REQUEST_ID_RE.fullmatch(
        normalized
    ):
        raise ValueError(
            (
                "request_id must match the DataLens "
                "server-owned HTTP request id format."
            )
        )

    return normalized


def _read_runtime_match(
    *,
    request_id: str,
    path: Path,
) -> tuple[
    RuntimeTraceRecord
    | None,
    int,
]:
    latest_match: (
        RuntimeTraceRecord
        | None
    ) = None

    malformed_line_count = 0

    try:
        if not path.exists():
            return (
                None,
                0,
            )

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
                    payload = json.loads(
                        normalized
                    )

                    trace = (
                        RuntimeTraceRecord
                        .model_validate(
                            payload
                        )
                    )

                except Exception:
                    malformed_line_count += 1
                    continue

                if (
                    trace.request_id
                    ==
                    request_id
                ):
                    latest_match = (
                        trace
                    )

    except OSError as error:
        raise TraceExplorerReadError(
            (
                "Local runtime observability "
                "data could not be read."
            )
        ) from error

    return (
        latest_match,
        malformed_line_count,
    )


def _read_ai_matches(
    *,
    request_id: str,
    path: Path,
) -> tuple[
    list[
        AITraceRecord
    ],
    int,
]:
    matches: list[
        AITraceRecord
    ] = []

    malformed_line_count = 0

    try:
        if not path.exists():
            return (
                [],
                0,
            )

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
                    payload = json.loads(
                        normalized
                    )

                    trace = (
                        AITraceRecord
                        .model_validate(
                            payload
                        )
                    )

                except Exception:
                    malformed_line_count += 1
                    continue

                if (
                    trace.request_id
                    ==
                    request_id
                ):
                    matches.append(
                        trace
                    )

    except OSError as error:
        raise TraceExplorerReadError(
            (
                "Local AI observability "
                "data could not be read."
            )
        ) from error

    return (
        matches,
        malformed_line_count,
    )


def _runtime_summary(
    trace: RuntimeTraceRecord,
) -> TraceExplorerRuntimeSummary:
    return (
        TraceExplorerRuntimeSummary(
            created_at_utc=
                trace.created_at_utc,

            trace_rule_version=
                trace.trace_rule_version,

            method=
                trace.method,

            route_template=
                trace.route_template,

            status_code=
                trace.status_code,

            duration_ms=
                trace.duration_ms,

            workflow_id=
                trace.workflow_id,

            run_status=
                trace.run_status,

            failure_kind=
                trace.failure_kind,
        )
    )


def _ai_summary(
    trace: AITraceRecord,
) -> TraceExplorerAISummary:
    return (
        TraceExplorerAISummary(
            trace_id=
                trace.trace_id,

            created_at_utc=
                trace.created_at_utc,

            trace_rule_version=
                trace.trace_rule_version,

            workflow_id=
                trace.workflow_id,

            analysis_id=
                trace.analysis_id,

            analysis_source_type=
                trace.analysis_source_type,

            run_status=
                trace.run_status,

            failure_stage=(
                trace.failure.stage

                if (
                    trace.failure
                    is not None
                )

                else
                None
            ),

            total_ms=
                trace.timings.total_ms,
        )
    )


def get_request_trace_explorer(
    request_id: str,
    *,
    runtime_path: (
        Path
        | None
    ) = None,
    ai_path: (
        Path
        | None
    ) = None,
) -> (
    TraceExplorerResponse
    | None
):
    normalized_request_id = (
        normalize_runtime_request_id(
            request_id
        )
    )

    resolved_runtime_path = (
        runtime_path.resolve()

        if runtime_path is not None

        else
        resolve_runtime_trace_path()
    )

    resolved_ai_path = (
        ai_path.resolve()

        if ai_path is not None

        else
        resolve_ai_trace_path()
    )

    (
        runtime_trace,
        runtime_malformed_line_count,
    ) = _read_runtime_match(
        request_id=
            normalized_request_id,

        path=
            resolved_runtime_path,
    )

    (
        ai_traces,
        ai_malformed_line_count,
    ) = _read_ai_matches(
        request_id=
            normalized_request_id,

        path=
            resolved_ai_path,
    )

    if (
        runtime_trace
        is None
        and
        not ai_traces
    ):
        return None

    ai_summaries = [
        _ai_summary(
            trace
        )

        for trace
        in reversed(
            ai_traces
        )
    ]

    return (
        TraceExplorerResponse(
            request_id=
                normalized_request_id,

            runtime_found=(
                runtime_trace
                is not None
            ),

            runtime=(
                _runtime_summary(
                    runtime_trace
                )

                if runtime_trace
                is not None

                else
                None
            ),

            runtime_malformed_line_count=
                runtime_malformed_line_count,

            ai_trace_count=
                len(
                    ai_summaries
                ),

            ai_malformed_line_count=
                ai_malformed_line_count,

            ai_traces=
                ai_summaries,
        )
    )
