from __future__ import annotations

from app.security.llm_payload import (
    LLM_PAYLOAD_PRIVACY_RULE_VERSION,
    LLMPayloadClass,
    LLMPayloadPrivacyError,
    classified_llm_chat,
    classified_llm_embed,
    require_allowed_llm_payload_class,
)


TEST_RULE_VERSION = "llm_payload_privacy_test_v0.1"


class FakeClient:
    def __init__(self) -> None:
        self.chat_calls = []
        self.embed_calls = []

    def chat(self, **kwargs):
        self.chat_calls.append(kwargs)
        return {"kind": "chat"}

    def embed(self, **kwargs):
        self.embed_calls.append(kwargs)
        return {"kind": "embed"}


def assert_true(value, message: str) -> None:
    if not value:
        raise AssertionError(message)


def test_version() -> None:
    assert (
        LLM_PAYLOAD_PRIVACY_RULE_VERSION
        == "llm_payload_privacy_v0.1"
    )


def test_allowed_classes() -> None:
    allowed = [
        LLMPayloadClass.METADATA_ONLY,
        LLMPayloadClass.DETERMINISTIC_EVIDENCE,
        LLMPayloadClass.SEMANTIC_VALUE_SAMPLE,
        LLMPayloadClass.DOCUMENT_CONTENT,
    ]

    for payload_class in allowed:
        assert (
            require_allowed_llm_payload_class(payload_class)
            is payload_class
        )


def test_raw_rows_rejected() -> None:
    try:
        require_allowed_llm_payload_class(
            LLMPayloadClass.TABULAR_RAW_ROWS
        )
    except LLMPayloadPrivacyError:
        return

    raise AssertionError(
        "TABULAR_RAW_ROWS must be rejected."
    )


def test_unknown_class_rejected() -> None:
    try:
        require_allowed_llm_payload_class(
            "unclassified"
        )
    except LLMPayloadPrivacyError:
        return

    raise AssertionError(
        "Unknown payload class must fail closed."
    )


def test_classified_chat() -> None:
    client = FakeClient()

    result = classified_llm_chat(
        client,
        payload_class=LLMPayloadClass.METADATA_ONLY,
        model="test-model",
        messages=[
            {
                "role": "user",
                "content": "metadata",
            }
        ],
    )

    assert result == {"kind": "chat"}
    assert len(client.chat_calls) == 1

    assert_true(
        "payload_class" not in client.chat_calls[0],
        "payload_class leaked to the model client.",
    )


def test_classified_embed() -> None:
    client = FakeClient()

    result = classified_llm_embed(
        client,
        payload_class=LLMPayloadClass.DOCUMENT_CONTENT,
        model="embedding-model",
        input=["document text"],
    )

    assert result == {"kind": "embed"}
    assert len(client.embed_calls) == 1

    assert_true(
        "payload_class" not in client.embed_calls[0],
        "payload_class leaked to the embedding client.",
    )


def test_raw_rows_blocked_before_client() -> None:
    client = FakeClient()

    try:
        classified_llm_chat(
            client,
            payload_class=LLMPayloadClass.TABULAR_RAW_ROWS,
            model="test-model",
            messages=[],
        )
    except LLMPayloadPrivacyError:
        pass
    else:
        raise AssertionError(
            "Raw-row chat should have been blocked."
        )

    assert client.chat_calls == []


TESTS = [
    ("Payload privacy rule version", test_version),
    ("Allowed payload classes", test_allowed_classes),
    ("Raw tabular rows rejected", test_raw_rows_rejected),
    ("Unknown class fails closed", test_unknown_class_rejected),
    ("Classified chat forwarding", test_classified_chat),
    ("Classified embedding forwarding", test_classified_embed),
    (
        "Raw rows blocked before model client",
        test_raw_rows_blocked_before_client,
    ),
]


def main() -> None:
    print(
        "=== DATALENS LLM PAYLOAD PRIVACY v0.1 ==="
    )
    print()

    passed = 0

    for label, test in TESTS:
        test()
        passed += 1
        print(f"[PASS] {label}")

    print()
    print(
        f"PASS - {passed}/{len(TESTS)} "
        "LLM payload privacy checks"
    )
    print(f"Rule: {TEST_RULE_VERSION}")


if __name__ == "__main__":
    main()
