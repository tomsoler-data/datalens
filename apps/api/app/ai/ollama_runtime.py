from __future__ import annotations


import os

from urllib.parse import (
    urlsplit,
)

from app.security.llm_egress import (
    LocalLLMEgressError,
    require_local_llm_url,
)


# ============================================================
# VERSION
# ============================================================


OLLAMA_RUNTIME_RULE_VERSION = (
    "ollama_runtime_v0.1"
)


# ============================================================
# CONFIGURATION
# ============================================================


OLLAMA_HOST_ENV = (
    "DATALENS_OLLAMA_HOST"
)

DEFAULT_OLLAMA_HOST = (
    "http://localhost:11434"
)


# ============================================================
# RESOLUTION
# ============================================================


def resolve_ollama_host(
) -> str:
    """
    Resolve the single Ollama base URL used by DataLens.

    The environment may override the default, but every
    configured destination still passes through the shared
    local-only LLM egress guard.

    A base Ollama host may not contain an application path.
    """

    configured = os.getenv(
        OLLAMA_HOST_ENV
    )


    candidate = (
        DEFAULT_OLLAMA_HOST
        if configured is None
        else configured.strip()
    )


    if not candidate:
        raise LocalLLMEgressError(
            (
                "Configured Ollama host must "
                "not be empty."
            )
        )


    validated = (
        require_local_llm_url(
            candidate
        )
    )


    parsed = urlsplit(
        validated
    )


    if parsed.path not in (
        "",
        "/",
    ):
        raise LocalLLMEgressError(
            (
                "Configured Ollama host must "
                "not contain an application path."
            )
        )


    return validated.rstrip(
        "/"
    )


def resolve_ollama_chat_url(
) -> str:
    """
    Resolve the Ollama /api/chat endpoint from the single
    guarded Ollama base URL.
    """

    chat_url = (
        f"{resolve_ollama_host()}"
        "/api/chat"
    )


    return require_local_llm_url(
        chat_url
    )