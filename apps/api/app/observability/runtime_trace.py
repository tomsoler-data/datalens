from __future__ import annotations

import json
import os
import re
import threading
import uuid

from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.observability.request_context import (
    bind_runtime_request_id,
    reset_runtime_request_id,
)


RUNTIME_TRACE_RULE_VERSION = "runtime_trace_v0.1"
RUNTIME_TRACE_PRIVACY_RULE_VERSION = "runtime_trace_privacy_v0.1"

RUNTIME_REQUEST_ID_HEADER = "X-DataLens-Request-ID"

RUNTIME_WORKFLOW_ID_STATE_KEY = (
    "datalens_workflow_id"
)

DEFAULT_RUNTIME_TRACE_RELATIVE_PATH = (
    Path("data")
    / "observability"
    / "runtime_requests.jsonl"
)

_RUNTIME_TRACE_WRITE_LOCK = threading.Lock()

_SAFE_WORKFLOW_ID_RE = re.compile(
    r"^prep:[0-9a-fA-F-]{8,64}$"
)


class RuntimeTracePrivacy(BaseModel):
    model_config = ConfigDict(
        extra="forbid"
    )

    privacy_rule_version: str = (
        RUNTIME_TRACE_PRIVACY_RULE_VERSION
    )

    storage_scope: str = "local_jsonl"

    contains_request_body: bool = False
    contains_response_body: bool = False
    contains_query_string: bool = False
    contains_request_headers: bool = False
    contains_client_ip: bool = False
    contains_raw_dataset_rows: bool = False
    contains_uploaded_file_contents: bool = False
    contains_document_chunks: bool = False
    contains_exception_message: bool = False
    contains_filesystem_path: bool = False

    note: str = (
        "DataLens runtime observability stores only "
        "server-generated request correlation, HTTP method, "
        "server-owned route template, status, duration and a "
        "validated workflow identifier when safely available. "
        "Request and response bodies, query strings, incoming "
        "headers, client IPs, dataset rows, uploaded contents, "
        "document chunks, exception messages and filesystem "
        "paths are not persisted."
    )


RuntimeTraceRunStatus = Literal[
    "completed",
    "failed",
]


class RuntimeTraceRecord(BaseModel):
    model_config = ConfigDict(
        extra="forbid"
    )

    request_id: str = Field(
        min_length=1
    )

    created_at_utc: str

    trace_rule_version: str = (
        RUNTIME_TRACE_RULE_VERSION
    )

    method: str = Field(
        min_length=1
    )

    route_template: str = Field(
        min_length=1
    )

    status_code: int = Field(
        ge=100,
        le=599,
    )

    duration_ms: float = Field(
        ge=0.0
    )

    workflow_id: str | None = None

    run_status: RuntimeTraceRunStatus = (
        "completed"
    )

    failure_kind: str | None = None

    privacy: RuntimeTracePrivacy = Field(
        default_factory=RuntimeTracePrivacy
    )


class RuntimeTraceWriteResult(BaseModel):
    model_config = ConfigDict(
        extra="forbid"
    )

    enabled: bool
    written: bool

    path: str | None = None
    error: str | None = None


def new_runtime_request_id() -> str:
    return (
        "http:"
        + uuid.uuid4().hex
    )


def default_api_root() -> Path:
    return (
        Path(__file__)
        .resolve()
        .parents[2]
    )


def runtime_trace_enabled() -> bool:
    raw = (
        os.getenv(
            "DATALENS_RUNTIME_TRACE_ENABLED",
            "1",
        )
        .strip()
        .casefold()
    )

    return raw not in {
        "0",
        "false",
        "no",
        "off",
    }


def resolve_runtime_trace_path() -> Path:
    configured = (
        os.getenv(
            "DATALENS_RUNTIME_TRACE_PATH",
            "",
        )
        .strip()
    )

    if configured:
        configured_path = (
            Path(configured)
            .expanduser()
        )

        if configured_path.is_absolute():
            return configured_path

        return (
            default_api_root()
            / configured_path
        ).resolve()

    return (
        default_api_root()
        / DEFAULT_RUNTIME_TRACE_RELATIVE_PATH
    ).resolve()


def _route_template_from_scope(
    scope: Scope,
) -> str:
    route = scope.get(
        "route"
    )

    path = getattr(
        route,
        "path",
        None,
    )

    if (
        isinstance(path, str)
        and path.strip()
    ):
        return path.strip()

    # Never persist the raw URL path for unmatched routes.
    return "__unmatched__"


def _workflow_id_from_scope(
    scope: Scope,
) -> str | None:
    """
    Read workflow correlation only from trusted ASGI state.

    URL path parameters are intentionally ignored because they
    originate from the HTTP client and therefore cannot prove
    that a Preparation workflow exists or is server-owned.

    Application code may stamp RUNTIME_WORKFLOW_ID_STATE_KEY
    only after resolving the workflow through authoritative
    server state.
    """

    state = (
        scope.get("state")
        or {}
    )

    if not isinstance(
        state,
        dict,
    ):
        return None

    raw = state.get(
        RUNTIME_WORKFLOW_ID_STATE_KEY
    )

    if not isinstance(
        raw,
        str,
    ):
        return None

    normalized = raw.strip()

    if not _SAFE_WORKFLOW_ID_RE.fullmatch(
        normalized
    ):
        return None

    return normalized

def stamp_validated_runtime_workflow_id(
    *,
    scope: Scope,
    workflow_id: str,
) -> str:
    """
    Publish an already server-validated Preparation workflow
    identifier into trusted ASGI state for RuntimeTrace
    correlation.

    This helper does NOT prove workflow existence.

    The caller must invoke it only after authoritative
    server-side workflow validation.

    RuntimeTraceMiddleware remains independent from:
    - request bodies;
    - multipart/form-data;
    - query strings;
    - path parameters;
    - incoming headers.
    """

    if not isinstance(
        workflow_id,
        str,
    ):
        raise ValueError(
            "workflow_id must be a string."
        )

    normalized = (
        workflow_id.strip()
    )

    if not _SAFE_WORKFLOW_ID_RE.fullmatch(
        normalized
    ):
        raise ValueError(
            "workflow_id is not valid for runtime correlation."
        )

    state = (
        scope.get(
            "state"
        )
    )

    if state is None:
        state = {}

        scope[
            "state"
        ] = state

    if not isinstance(
        state,
        dict,
    ):
        raise TypeError(
            "ASGI scope state must be a mutable mapping."
        )

    state[
        RUNTIME_WORKFLOW_ID_STATE_KEY
    ] = normalized

    return normalized



def build_runtime_trace(
    *,
    request_id: str,
    scope: Scope,
    status_code: int,
    duration_ms: float,
    run_status: RuntimeTraceRunStatus = "completed",
    failure_kind: str | None = None,
) -> RuntimeTraceRecord:
    method = str(
        scope.get(
            "method",
            "UNKNOWN",
        )
    ).upper()

    return RuntimeTraceRecord(
        request_id=request_id,
        created_at_utc=(
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
        method=method,
        route_template=(
            _route_template_from_scope(
                scope
            )
        ),
        status_code=status_code,
        duration_ms=round(
            max(
                0.0,
                float(duration_ms),
            ),
            3,
        ),
        workflow_id=(
            _workflow_id_from_scope(
                scope
            )
        ),
        run_status=run_status,
        failure_kind=failure_kind,
    )


def write_runtime_trace(
    trace: RuntimeTraceRecord,
) -> RuntimeTraceWriteResult:
    if not runtime_trace_enabled():
        return RuntimeTraceWriteResult(
            enabled=False,
            written=False,
        )

    path = resolve_runtime_trace_path()

    try:
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        serialized = (
            trace.model_dump_json()
            + "\n"
        )

        with _RUNTIME_TRACE_WRITE_LOCK:
            prefix = ""

            if (
                path.exists()
                and path.stat().st_size > 0
            ):
                with path.open(
                    "rb"
                ) as existing:
                    existing.seek(
                        -1,
                        2,
                    )

                    last_byte = existing.read(
                        1
                    )

                if last_byte not in {
                    b"\n",
                    b"\r",
                }:
                    prefix = "\n"

            with path.open(
                "a",
                encoding="utf-8",
                newline="\n",
            ) as handle:
                handle.write(
                    prefix
                    + serialized
                )

        return RuntimeTraceWriteResult(
            enabled=True,
            written=True,
            path=str(path),
        )

    except OSError as error:
        # Runtime observability must never break the API path.
        return RuntimeTraceWriteResult(
            enabled=True,
            written=False,
            path=str(path),
            error=(
                f"{type(error).__name__}: "
                f"{error}"
            ),
        )


def _write_runtime_trace_best_effort(
    trace: RuntimeTraceRecord,
) -> None:
    try:
        write_runtime_trace(
            trace
        )

    except Exception:
        # Even an unexpected observability defect must not
        # alter the HTTP execution contract.
        return


class RuntimeTraceMiddleware:
    """
    Privacy-safe HTTP request tracing.

    The middleware intentionally never reads:
    - request bodies;
    - query strings;
    - incoming request headers;
    - client addresses.

    A request identifier is always generated by DataLens and
    therefore cannot be spoofed by an incoming header.
    """

    def __init__(
        self,
        app: ASGIApp,
    ) -> None:
        self.app = app

    async def __call__(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        if scope["type"] != "http":
            await self.app(
                scope,
                receive,
                send,
            )
            return

        request_id = (
            new_runtime_request_id()
        )

        request_context_token = (
            bind_runtime_request_id(
                request_id
            )
        )

        started_at = perf_counter()

        status_code = 500

        async def send_with_request_id(
            message: Message,
        ) -> None:
            nonlocal status_code

            if (
                message["type"]
                == "http.response.start"
            ):
                status_code = int(
                    message["status"]
                )

                headers = MutableHeaders(
                    scope=message
                )

                # Replace any downstream value with the
                # server-owned correlation identifier.
                headers[
                    RUNTIME_REQUEST_ID_HEADER
                ] = request_id

            await send(
                message
            )

        try:
            await self.app(
                scope,
                receive,
                send_with_request_id,
            )

        except Exception:
            duration_ms = (
                perf_counter()
                - started_at
            ) * 1000.0

            trace = build_runtime_trace(
                request_id=request_id,
                scope=scope,
                status_code=500,
                duration_ms=duration_ms,
                run_status="failed",
                failure_kind=(
                    "unhandled_exception"
                ),
            )

            _write_runtime_trace_best_effort(
                trace
            )

            raise

        finally:
            reset_runtime_request_id(
                request_context_token
            )

        duration_ms = (
            perf_counter()
            - started_at
        ) * 1000.0

        trace = build_runtime_trace(
            request_id=request_id,
            scope=scope,
            status_code=status_code,
            duration_ms=duration_ms,
            run_status="completed",
        )

        _write_runtime_trace_best_effort(
            trace
        )