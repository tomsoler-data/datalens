from __future__ import annotations


from unittest.mock import (
    patch,
)

from urllib.request import (
    Request,
)


import app.ai.provider as provider_module

import app.preparation.analysis_output_explanation as output_explanation_module

import app.preparation.dataset_identity_explanation as identity_explanation_module

import app.preparation.semantic_review as semantic_review_module

import app.security.llm_egress as llm_egress_module


# ============================================================
# VERSION
# ============================================================


TEST_RULE_VERSION = (
    "llm_egress_test_v0.1"
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
        "llm_egress_v0.1",
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
]


def main(
) -> None:
    print(
        "=== DATALENS LLM EGRESS BOUNDARY v0.1 ==="
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
