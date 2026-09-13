from __future__ import annotations


from types import (
    SimpleNamespace,
)


import app.planning.semantic_outlier_scope as scope


# ============================================================
# HELPERS
# ============================================================


def _response(
    content: str,
):
    return SimpleNamespace(
        message=
            SimpleNamespace(
                content=
                    content,
            )
    )


# ============================================================
# LABEL VALIDATION
# ============================================================


def test_exact_supported_labels() -> None:
    assert (
        scope.parse_semantic_outlier_scope_label(
            "customer_entity_outlier_detection"
        )
        ==
        "customer_entity_outlier_detection"
    )

    assert (
        scope.parse_semantic_outlier_scope_label(
            "variable_or_record_outlier_detection"
        )
        ==
        "variable_or_record_outlier_detection"
    )

    assert (
        scope.parse_semantic_outlier_scope_label(
            "none"
        )
        ==
        "none"
    )


def test_unexpected_output_fails_closed() -> None:
    assert (
        scope.parse_semantic_outlier_scope_label(
            "1. customer_entity_outlier_detection"
        )
        is None
    )

    assert (
        scope.parse_semantic_outlier_scope_label(
            '{"intent":"customer_entity_outlier_detection"}'
        )
        is None
    )

    assert (
        scope.parse_semantic_outlier_scope_label(
            "customer"
        )
        is None
    )


# ============================================================
# LOCAL MODEL BOUNDARY
# ============================================================


def test_customer_scope_resolution() -> None:
    original = (
        scope.classified_llm_chat
    )

    captured = {}

    def fake_chat(
        client,
        **kwargs,
    ):
        captured.update(
            kwargs
        )

        return _response(
            "customer_entity_outlier_detection"
        )

    try:
        scope.classified_llm_chat = (
            fake_chat
        )

        result = (
            scope.resolve_semantic_outlier_scope(
                (
                    "Quels clients ont un comportement "
                    "tr?s diff?rent du reste ?"
                )
            )
        )

    finally:
        scope.classified_llm_chat = (
            original
        )

    assert (
        result
        ==
        "customer_entity_outlier_detection"
    )

    assert (
        captured[
            "payload_class"
        ]
        ==
        scope.LLMPayloadClass.METADATA_ONLY
    )

    assert (
        captured[
            "model"
        ]
        ==
        "qwen3.5:4b"
    )

    assert (
        captured[
            "think"
        ]
        is False
    )

    assert (
        captured[
            "stream"
        ]
        is False
    )

    assert (
        captured[
            "options"
        ][
            "temperature"
        ]
        ==
        0
    )

    assert (
        captured[
            "options"
        ][
            "seed"
        ]
        ==
        42
    )

    assert (
        captured[
            "options"
        ][
            "num_ctx"
        ]
        ==
        8192
    )


def test_non_customer_scope_is_preserved() -> None:
    original = (
        scope.classified_llm_chat
    )

    def fake_chat(
        client,
        **kwargs,
    ):
        return _response(
            "variable_or_record_outlier_detection"
        )

    try:
        scope.classified_llm_chat = (
            fake_chat
        )

        result = (
            scope.resolve_semantic_outlier_scope(
                "D?tecte les prix atypiques."
            )
        )

    finally:
        scope.classified_llm_chat = (
            original
        )

    assert (
        result
        ==
        "variable_or_record_outlier_detection"
    )


def test_runtime_failure_fails_closed() -> None:
    original = (
        scope.classified_llm_chat
    )

    def fake_chat(
        client,
        **kwargs,
    ):
        raise RuntimeError(
            "synthetic local model failure"
        )

    try:
        scope.classified_llm_chat = (
            fake_chat
        )

        result = (
            scope.resolve_semantic_outlier_scope(
                "D?tecte les clients atypiques."
            )
        )

    finally:
        scope.classified_llm_chat = (
            original
        )

    assert result is None


def test_empty_objective_does_not_call_model() -> None:
    original = (
        scope.classified_llm_chat
    )

    called = False

    def fake_chat(
        client,
        **kwargs,
    ):
        nonlocal called
        called = True

        return _response(
            "none"
        )

    try:
        scope.classified_llm_chat = (
            fake_chat
        )

        result = (
            scope.resolve_semantic_outlier_scope(
                "   "
            )
        )

    finally:
        scope.classified_llm_chat = (
            original
        )

    assert result is None
    assert called is False


# ============================================================
# RUNNER
# ============================================================


def main() -> None:
    tests = [
        test_exact_supported_labels,
        test_unexpected_output_fails_closed,
        test_customer_scope_resolution,
        test_non_customer_scope_is_preserved,
        test_runtime_failure_fails_closed,
        test_empty_objective_does_not_call_model,
    ]

    print(
        "=== SEMANTIC OUTLIER SCOPE v0.1 ==="
    )

    for test in tests:
        test()

        print(
            f"[PASS] {test.__name__}"
        )

    print()
    print(
        "PASS - semantic outlier scope v0.1"
    )


if __name__ == "__main__":
    main()
