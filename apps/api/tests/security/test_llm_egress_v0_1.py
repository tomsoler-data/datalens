from __future__ import annotations


import os


from unittest.mock import (
    patch,
)

from urllib.request import (
    Request,
)


import app.ai.ollama_runtime as ollama_runtime_module

import app.ai.provider as provider_module

import app.preparation.analysis_output_explanation as output_explanation_module

import app.preparation.dataset_identity_explanation as identity_explanation_module

import app.preparation.semantic_review as semantic_review_module

import app.security.llm_egress as llm_egress_module


# ============================================================
# VERSION
# ============================================================


TEST_RULE_VERSION = (
    "llm_egress_test_v0.2"
)


# ============================================================
# ASSERTIONS
# ============================================================


def assert_equal(
    actual,
    expected,
    message: str,
) -> None:
    if actual != expected:
        raise AssertionError(
            (
                f"{message}\n"
                f"expected={expected!r}\n"
                f"actual={actual!r}"
            )
        )


def assert_true(
    value,
    message: str,
) -> None:
    if not value:
        raise AssertionError(
            message
        )


def assert_rejected(
    url: str,
) -> None:
    captured = None

    try:
        (
            llm_egress_module
            .require_local_llm_url(
                url
            )
        )

    except (
        llm_egress_module
        .LocalLLMEgressError
    ) as error:
        captured = error


    assert_true(
        captured is not None,
        (
            "Expected LLM destination "
            f"to be rejected: {url}"
        ),
    )


# ============================================================
# TEST 1
# VERSION
# ============================================================


def test_version(
) -> None:
    assert_equal(
        (
            llm_egress_module
            .LLM_EGRESS_RULE_VERSION
        ),
        "llm_egress_v0.2",
        "Unexpected LLM egress rule version.",
    )


# ============================================================
# TEST 2
# LOCAL DESTINATIONS
# ============================================================


def test_loopback_destinations_are_allowed(
) -> None:
    allowed = [
        "http://localhost:11434",
        "http://localhost:11434/api/chat",
        "http://127.0.0.1:11434",
        "http://127.42.8.9:11434/api/chat",
        "http://[::1]:11434/api/chat",
    ]


    for url in allowed:
        assert_equal(
            (
                llm_egress_module
                .require_local_llm_url(
                    url
                )
            ),
            url,
            (
                "Expected local LLM "
                "destination to be allowed."
            ),
        )


# ============================================================
# TEST 3
# REMOTE / MALFORMED DESTINATIONS
# ============================================================


def test_remote_destinations_are_rejected(
) -> None:
    rejected = [
        "https://api.example.com",
        "http://192.168.1.20:11434",
        "http://10.0.0.5:11434",
        "http://ollama.company.local:11434",
        "http://localhost.evil.example:11434",
        "ftp://localhost:11434",
        "localhost:11434",
        "http://user:password@localhost:11434",
        "http://localhost:11434/api/chat?remote=true",
        "http://localhost:11434/api/chat#fragment",
    ]


    for url in rejected:
        assert_rejected(
            url
        )


# ============================================================
# TEST 4
# NETWORK CALL IS BLOCKED BEFORE EGRESS
# ============================================================


def test_remote_request_never_reaches_urlopen(
) -> None:
    request = Request(
        "https://api.example.com/v1/chat",
        method="POST",
    )


    with patch.object(
        llm_egress_module._LOCAL_LLM_OPENER,
        "open",
    ) as mocked_transport:
        captured = None

        try:
            (
                llm_egress_module
                .open_local_llm_request(
                    request,
                    payload_class=(
                        llm_egress_module
                        .LLMPayloadClass
                        .METADATA_ONLY
                    ),
                    timeout=1.0,
                )
            )

        except (
            llm_egress_module
            .LocalLLMEgressError
        ) as error:
            captured = error


        assert_true(
            captured is not None,
            (
                "Remote request must fail "
                "before network I/O."
            ),
        )

        mocked_transport.assert_not_called()


# ============================================================
# TEST 5
# LOCAL REQUEST MAY REACH TRANSPORT
# ============================================================


def test_local_request_reaches_urlopen(
) -> None:
    request = Request(
        "http://127.0.0.1:11434/api/chat",
        method="POST",
    )

    sentinel = object()


    with patch.object(
        llm_egress_module._LOCAL_LLM_OPENER,
        "open",
        return_value=sentinel,
    ) as mocked_transport:
        result = (
            llm_egress_module
            .open_local_llm_request(
                request,
                payload_class=(
                    llm_egress_module
                    .LLMPayloadClass
                    .METADATA_ONLY
                ),
                timeout=2.5,
            )
        )


    assert_true(
        result is sentinel,
        "Local request did not reach transport.",
    )

    mocked_transport.assert_called_once_with(
        request,
        timeout=2.5,
    )


# ============================================================
# TEST 6
# CURRENT PRODUCTION DEFAULTS
# ============================================================


def test_current_ollama_defaults_are_local(
) -> None:
    defaults = [
        provider_module.OLLAMA_HOST,
        (
            output_explanation_module
            .DEFAULT_OLLAMA_CHAT_URL
        ),
        (
            identity_explanation_module
            .DEFAULT_OLLAMA_CHAT_URL
        ),
        (
            semantic_review_module
            .DEFAULT_OLLAMA_CHAT_URL
        ),
    ]


    for url in defaults:
        (
            llm_egress_module
            .require_local_llm_url(
                url
            )
        )


# ============================================================
# TEST 7
# PREPARATION WIRING
# ============================================================


def test_preparation_urllib_paths_use_shared_guard(
) -> None:
    modules = [
        output_explanation_module,
        identity_explanation_module,
        semantic_review_module,
    ]


    for module in modules:
        assert_true(
            (
                module
                .open_local_llm_request
                is
                llm_egress_module
                .open_local_llm_request
            ),
            (
                f"{module.__name__} does not "
                "use the shared LLM egress guard."
            ),
        )


        assert_true(
            not hasattr(
                module,
                "urlopen",
            ),
            (
                f"{module.__name__} still exposes "
                "a direct urlopen bypass."
            ),
        )



# ============================================================
# TEST 8
# DOCKER BRIDGE REQUIRES EXPLICIT OPT-IN
# ============================================================


def test_docker_bridge_requires_explicit_opt_in(
) -> None:
    url = (
        "http://"
        "host.docker.internal"
        ":11434/api/chat"
    )


    with patch.dict(
        os.environ,
        {
            (
                llm_egress_module
                .LLM_DOCKER_BRIDGE_ENV
            ):
                "0",
        },
        clear=False,
    ):
        assert_rejected(
            url
        )


# ============================================================
# TEST 9
# DOCKER BRIDGE SCOPE
# ============================================================


def test_docker_bridge_is_exactly_scoped(
) -> None:
    allowed = [
        (
            "http://"
            "host.docker.internal"
            ":11434"
        ),
        (
            "http://"
            "host.docker.internal"
            ":11434/api/chat"
        ),
    ]

    rejected = [
        "http://host.docker.internal",
        "http://host.docker.internal:11435",
        "https://host.docker.internal:11434",
        (
            "http://user:password@"
            "host.docker.internal:11434"
        ),
        (
            "http://host.docker.internal:"
            "11434/api/chat?remote=true"
        ),
        (
            "http://host.docker.internal:"
            "11434/api/chat#fragment"
        ),
        (
            "http://host.docker.internal."
            "evil.example:11434"
        ),
        "http://192.168.65.254:11434",
        "http://192.168.1.20:11434",
        "http://10.0.0.5:11434",
    ]


    with patch.dict(
        os.environ,
        {
            (
                llm_egress_module
                .LLM_DOCKER_BRIDGE_ENV
            ):
                "1",
        },
        clear=False,
    ):
        for url in allowed:
            assert_equal(
                (
                    llm_egress_module
                    .require_local_llm_url(
                        url
                    )
                ),
                url,
                (
                    "Explicit Docker Ollama "
                    "bridge should be allowed."
                ),
            )


        for url in rejected:
            assert_rejected(
                url
            )


# ============================================================
# TEST 10
# DOCKER BRIDGE MAY REACH GUARDED TRANSPORT
# ============================================================


def test_docker_bridge_request_reaches_transport_when_enabled(
) -> None:
    request = Request(
        (
            "http://"
            "host.docker.internal"
            ":11434/api/chat"
        ),
        method="POST",
    )

    sentinel = object()


    with patch.dict(
        os.environ,
        {
            (
                llm_egress_module
                .LLM_DOCKER_BRIDGE_ENV
            ):
                "1",
        },
        clear=False,
    ):
        with patch.object(
            llm_egress_module._LOCAL_LLM_OPENER,
            "open",
            return_value=sentinel,
        ) as mocked_transport:
            result = (
                llm_egress_module
                .open_local_llm_request(
                    request,
                    payload_class=(
                        llm_egress_module
                        .LLMPayloadClass
                        .METADATA_ONLY
                    ),
                    timeout=2.5,
                )
            )


    assert_true(
        result is sentinel,
        (
            "Enabled Docker Ollama bridge "
            "did not reach guarded transport."
        ),
    )

    mocked_transport.assert_called_once_with(
        request,
        timeout=2.5,
    )


# ============================================================
# TEST 11
# CENTRAL OLLAMA RUNTIME RESOLUTION
# ============================================================


def test_ollama_runtime_resolver_uses_guarded_bridge(
) -> None:
    bridge_host = (
        "http://"
        "host.docker.internal"
        ":11434"
    )


    with patch.dict(
        os.environ,
        {
            (
                llm_egress_module
                .LLM_DOCKER_BRIDGE_ENV
            ):
                "1",

            (
                ollama_runtime_module
                .OLLAMA_HOST_ENV
            ):
                bridge_host,
        },
        clear=False,
    ):
        assert_equal(
            (
                ollama_runtime_module
                .resolve_ollama_host()
            ),
            bridge_host,
            (
                "Central Ollama host resolver "
                "did not select Docker bridge."
            ),
        )

        assert_equal(
            (
                ollama_runtime_module
                .resolve_ollama_chat_url()
            ),
            (
                bridge_host
                + "/api/chat"
            ),
            (
                "Central Ollama chat resolver "
                "did not derive /api/chat."
            ),
        )


    with patch.dict(
        os.environ,
        {
            (
                llm_egress_module
                .LLM_DOCKER_BRIDGE_ENV
            ):
                "1",

            (
                ollama_runtime_module
                .OLLAMA_HOST_ENV
            ):
                "http://192.168.1.20:11434",
        },
        clear=False,
    ):
        captured = None

        try:
            (
                ollama_runtime_module
                .resolve_ollama_host()
            )

        except (
            llm_egress_module
            .LocalLLMEgressError
        ) as error:
            captured = error


        assert_true(
            captured is not None,
            (
                "Central Ollama resolver must "
                "reject arbitrary LAN hosts."
            ),
        )
# ============================================================
# RUNNER
# ============================================================


TESTS = [
    (
        "LLM egress rule version",
        test_version,
    ),
    (
        "Loopback destinations allowed",
        test_loopback_destinations_are_allowed,
    ),
    (
        "Remote destinations rejected",
        test_remote_destinations_are_rejected,
    ),
    (
        "Remote request blocked before network",
        test_remote_request_never_reaches_urlopen,
    ),
    (
        "Local request reaches transport",
        test_local_request_reaches_urlopen,
    ),
    (
        "Production Ollama defaults are local",
        test_current_ollama_defaults_are_local,
    ),
    (
        "Preparation urllib paths share guard",
        test_preparation_urllib_paths_use_shared_guard,
    ),
    (
        "Docker bridge requires explicit opt-in",
        test_docker_bridge_requires_explicit_opt_in,
    ),
    (
        "Docker bridge scope is exact",
        test_docker_bridge_is_exactly_scoped,
    ),
    (
        "Docker bridge reaches guarded transport",
        test_docker_bridge_request_reaches_transport_when_enabled,
    ),
    (
        "Central Ollama runtime uses guarded bridge",
        test_ollama_runtime_resolver_uses_guarded_bridge,
    ),
]


def main(
) -> None:
    print(
        "=== DATALENS LLM EGRESS BOUNDARY v0.2 ==="
    )

    print()


    passed = 0


    for (
        label,
        test,
    ) in TESTS:
        try:
            test()

        except Exception as error:
            print(
                f"[FAIL] {label}"
            )

            print(
                f"       {type(error).__name__}: "
                f"{error}"
            )

            raise


        passed += 1

        print(
            f"[PASS] {label}"
        )


    print()

    print(
        (
            f"PASS - {passed}/{len(TESTS)} "
            "LLM egress security checks"
        )
    )

    print(
        f"Rule: {TEST_RULE_VERSION}"
    )


if __name__ == "__main__":
    main()
