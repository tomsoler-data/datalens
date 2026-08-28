from __future__ import annotations

import re

from contextvars import (
    ContextVar,
    Token,
)


RUNTIME_REQUEST_CONTEXT_RULE_VERSION = (
    "runtime_request_context_v0.1"
)


_SAFE_RUNTIME_REQUEST_ID_RE = re.compile(
    r"^http:[0-9a-f]{32}$"
)


_CURRENT_RUNTIME_REQUEST_ID: ContextVar[
    str | None
] = ContextVar(
    "datalens_runtime_request_id",
    default=None,
)


def bind_runtime_request_id(
    request_id: str,
) -> Token:
    normalized = (
        request_id
        .strip()
    )

    if not _SAFE_RUNTIME_REQUEST_ID_RE.fullmatch(
        normalized
    ):
        raise ValueError(
            (
                "Invalid DataLens runtime "
                "request identifier."
            )
        )

    return (
        _CURRENT_RUNTIME_REQUEST_ID
        .set(
            normalized
        )
    )


def reset_runtime_request_id(
    token: Token,
) -> None:
    _CURRENT_RUNTIME_REQUEST_ID.reset(
        token
    )


def current_runtime_request_id(
) -> str | None:
    return (
        _CURRENT_RUNTIME_REQUEST_ID
        .get()
    )
