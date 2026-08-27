from __future__ import annotations


import os

from ipaddress import (
    ip_address,
)

from typing import (
    Any,
)

from urllib.parse import (
    urlsplit,
)

from urllib.request import (
    HTTPRedirectHandler,
    ProxyHandler,
    Request,
    build_opener,
)

from app.security.llm_payload import (
    LLMPayloadClass,
    require_allowed_llm_payload_class,
)


# ============================================================
# VERSION
# ============================================================


LLM_EGRESS_RULE_VERSION = (
    "llm_egress_v0.2"
)


# ============================================================
# DOCKER BRIDGE CONTRACT
# ============================================================


LLM_DOCKER_BRIDGE_ENV = (
    "DATALENS_LLM_DOCKER_BRIDGE_ENABLED"
)

OLLAMA_DOCKER_BRIDGE_HOST = (
    "host.docker.internal"
)

OLLAMA_DOCKER_BRIDGE_PORT = (
    11434
)


def is_llm_docker_bridge_enabled(
) -> bool:
    """
    Return True only for an explicit Docker bridge opt-in.

    Any missing, empty, malformed or alternate value keeps
    the bridge disabled.
    """

    configured = (
        os.getenv(
            LLM_DOCKER_BRIDGE_ENV,
            "",
        )
        .strip()
    )

    return configured == "1"


# ============================================================
# ERROR
# ============================================================


class LocalLLMEgressError(
    ValueError
):
    """
    Raised when an LLM destination would leave the local
    machine or violates the local-only URL contract.
    """


# ============================================================
# FAIL-CLOSED LOCAL TRANSPORT
# ============================================================


class _RejectLLMRedirectHandler(
    HTTPRedirectHandler
):
    """
    Reject every HTTP redirect before urllib can perform
    a second network request.

    Even local-to-local redirects are forbidden.
    DataLens model-service URLs must be direct.
    """

    def redirect_request(
        self,
        req,
        fp,
        code,
        msg,
        headers,
        newurl,
    ):
        del (
            req,
            fp,
            code,
            msg,
            headers,
            newurl,
        )

        raise LocalLLMEgressError(
            (
                "LLM egress redirects "
                "are forbidden."
            )
        )


_LOCAL_LLM_OPENER = (
    build_opener(
        # Never inherit HTTP(S)_PROXY or related
        # environment configuration for local LLM I/O.
        ProxyHandler(
            {}
        ),

        _RejectLLMRedirectHandler(),
    )
)


# ============================================================
# LOCAL DESTINATION VALIDATION
# ============================================================


def require_local_llm_url(
    url: str,
) -> str:
    """
    Validate one model-service URL before any network egress.

    DataLens local-first contract:

    - HTTP only;
    - explicit URL syntax;
    - no credentials;
    - no query string or fragment;
    - localhost is allowed;
    - literal loopback IP addresses are allowed;
    - host.docker.internal is allowed only when the Docker
      bridge is explicitly enabled and only on Ollama port
      11434.

    Other DNS hostnames remain forbidden.

    The Docker exception is deliberately exact. It does not
    authorize LAN addresses, arbitrary private networks or
    arbitrary Docker hostnames.
    """

    if not isinstance(
        url,
        str,
    ):
        raise LocalLLMEgressError(
            (
                "LLM egress destination must "
                "be a URL string."
            )
        )


    normalized = (
        url
        .strip()
    )


    if not normalized:
        raise LocalLLMEgressError(
            (
                "LLM egress destination must "
                "not be empty."
            )
        )


    parsed = urlsplit(
        normalized
    )


    if (
        parsed.scheme
        .lower()
        !=
        "http"
    ):
        raise LocalLLMEgressError(
            (
                "LLM egress destination must "
                "use HTTP on the local machine."
            )
        )


    if not parsed.netloc:
        raise LocalLLMEgressError(
            (
                "LLM egress destination must "
                "contain an explicit local host."
            )
        )


    if (
        parsed.username
        is not None
        or
        parsed.password
        is not None
    ):
        raise LocalLLMEgressError(
            (
                "LLM egress destination must "
                "not contain URL credentials."
            )
        )


    if (
        parsed.query
        or
        parsed.fragment
    ):
        raise LocalLLMEgressError(
            (
                "LLM egress destination must "
                "not contain a query string "
                "or fragment."
            )
        )


    hostname = (
        parsed.hostname
    )


    if hostname is None:
        raise LocalLLMEgressError(
            (
                "LLM egress destination has "
                "no valid hostname."
            )
        )


    # Force urllib to validate an explicitly supplied port.
    #
    # parsed.port raises ValueError for malformed or
    # out-of-range ports.
    try:
        port = parsed.port

    except ValueError as error:
        raise LocalLLMEgressError(
            (
                "LLM egress destination contains "
                "an invalid port."
            )
        ) from error


    normalized_hostname = (
        hostname
        .lower()
    )


    if (
        normalized_hostname
        ==
        "localhost"
    ):
        return normalized


    if (
        normalized_hostname
        ==
        OLLAMA_DOCKER_BRIDGE_HOST
    ):
        if not is_llm_docker_bridge_enabled():
            raise LocalLLMEgressError(
                (
                    "Docker LLM bridge destination "
                    "is disabled."
                )
            )

        if (
            port
            !=
            OLLAMA_DOCKER_BRIDGE_PORT
        ):
            raise LocalLLMEgressError(
                (
                    "Docker LLM bridge destination "
                    "must use Ollama port 11434."
                )
            )

        return normalized


    try:
        address = ip_address(
            normalized_hostname
        )

    except ValueError as error:
        raise LocalLLMEgressError(
            (
                "LLM egress destination must use "
                "localhost, a literal loopback IP "
                "address, or the explicitly enabled "
                "Docker Ollama bridge."
            )
        ) from error


    if not address.is_loopback:
        raise LocalLLMEgressError(
            (
                "LLM egress destination is "
                "not a loopback address."
            )
        )


    return normalized


# ============================================================
# GUARDED URLLIB EGRESS
# ============================================================


def open_local_llm_request(
    request: Request,
    *,
    payload_class: LLMPayloadClass | str,
    timeout: float,
) -> Any:
    """
    Validate the final Request URL immediately before urllib
    performs network I/O.

    This protects Preparation paths that historically call
    Ollama through urllib rather than app.ai.provider.client.
    """

    require_allowed_llm_payload_class(
        payload_class
    )

    require_local_llm_url(
        request.full_url
    )


    return (
        _LOCAL_LLM_OPENER
        .open(
            request,
            timeout=timeout,
        )
    )