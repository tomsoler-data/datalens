from __future__ import annotations


import app.analysis.entity_outlier_requests as entity


# ============================================================
# FAST PATH
# ============================================================


def test_explicit_customer_outlier_uses_fast_path() -> None:
    original = (
        entity.resolve_semantic_outlier_scope
    )

    called = False

    def forbidden_semantic_call(
        objective,
        *,
        model,
    ):
        nonlocal called
        called = True

        raise AssertionError(
            "Semantic resolver must not run for an "
            "already explicit customer-outlier request."
        )

    try:
        entity.resolve_semantic_outlier_scope = (
            forbidden_semantic_call
        )

        result = (
            entity
            .resolve_entity_outlier_intent_with_semantic_fallback(
                "Find anomalous customers."
            )
        )

    finally:
        entity.resolve_semantic_outlier_scope = (
            original
        )

    assert result.status == "matched"
    assert (
        result.intent
        ==
        "customer_entity_outlier_detection"
    )
    assert result.entity_kind == "customer"
    assert (
        result.resolution_source
        ==
        "deterministic"
    )
    assert result.resolution_model is None
    assert called is False


# ============================================================
# SEMANTIC CUSTOMER PARAPHRASE
# ============================================================


def test_customer_paraphrase_uses_semantic_fallback() -> None:
    original = (
        entity.resolve_semantic_outlier_scope
    )

    call_count = 0

    def fake_semantic_call(
        objective,
        *,
        model,
    ):
        nonlocal call_count
        call_count += 1

        return (
            "customer_entity_outlier_detection"
        )

    try:
        entity.resolve_semantic_outlier_scope = (
            fake_semantic_call
        )

        result = (
            entity
            .resolve_entity_outlier_intent_with_semantic_fallback(
                (
                    "Which customers behave very differently "
                    "from the rest of the customer population?"
                )
            )
        )

    finally:
        entity.resolve_semantic_outlier_scope = (
            original
        )

    assert call_count == 1
    assert result.status == "matched"
    assert (
        result.intent
        ==
        "customer_entity_outlier_detection"
    )
    assert result.entity_kind == "customer"
    assert (
        result.resolution_source
        ==
        "semantic"
    )
    assert (
        result.resolution_model
        ==
        "qwen3.5:4b"
    )


# ============================================================
# CUSTOMER NON-OUTLIER REQUEST
# ============================================================


def test_customer_ranking_does_not_promote_to_entity_outlier() -> None:
    original = (
        entity.resolve_semantic_outlier_scope
    )

    call_count = 0

    def fake_semantic_call(
        objective,
        *,
        model,
    ):
        nonlocal call_count
        call_count += 1

        return "none"

    try:
        entity.resolve_semantic_outlier_scope = (
            fake_semantic_call
        )

        result = (
            entity
            .resolve_entity_outlier_intent_with_semantic_fallback(
                "Rank the top 10 customers by revenue."
            )
        )

    finally:
        entity.resolve_semantic_outlier_scope = (
            original
        )

    assert call_count == 1
    assert result.status == "not_matched"
    assert result.intent is None
    assert result.entity_kind is None
    assert (
        result.resolution_source
        ==
        "semantic"
    )


# ============================================================
# NON-CUSTOMER OUTLIER REQUEST
# ============================================================


def test_variable_outlier_does_not_call_customer_semantic_fallback() -> None:
    original = (
        entity.resolve_semantic_outlier_scope
    )

    called = False

    def forbidden_semantic_call(
        objective,
        *,
        model,
    ):
        nonlocal called
        called = True

        raise AssertionError(
            "Customer semantic resolver must not run for "
            "a non-customer anomaly request."
        )

    try:
        entity.resolve_semantic_outlier_scope = (
            forbidden_semantic_call
        )

        result = (
            entity
            .resolve_entity_outlier_intent_with_semantic_fallback(
                "Detect unusual prices in transactions."
            )
        )

    finally:
        entity.resolve_semantic_outlier_scope = (
            original
        )

    assert result.status == "not_matched"
    assert result.intent is None
    assert result.entity_kind is None
    assert called is False


# ============================================================
# SEMANTIC FAILURE
# ============================================================


def test_semantic_failure_fails_closed() -> None:
    original = (
        entity.resolve_semantic_outlier_scope
    )

    call_count = 0

    def fake_semantic_call(
        objective,
        *,
        model,
    ):
        nonlocal call_count
        call_count += 1

        return None

    try:
        entity.resolve_semantic_outlier_scope = (
            fake_semantic_call
        )

        result = (
            entity
            .resolve_entity_outlier_intent_with_semantic_fallback(
                (
                    "Which customers behave differently "
                    "from the rest?"
                )
            )
        )

    finally:
        entity.resolve_semantic_outlier_scope = (
            original
        )

    assert call_count == 1
    assert result.status == "not_matched"
    assert result.intent is None
    assert result.entity_kind is None


# ============================================================
# RUNNER
# ============================================================


def main() -> None:
    tests = [
        test_explicit_customer_outlier_uses_fast_path,
        test_customer_paraphrase_uses_semantic_fallback,
        test_customer_ranking_does_not_promote_to_entity_outlier,
        test_variable_outlier_does_not_call_customer_semantic_fallback,
        test_semantic_failure_fails_closed,
    ]

    print(
        "=== ENTITY OUTLIER SEMANTIC FALLBACK v0.1 ==="
    )

    for test in tests:
        test()

        print(
            f"[PASS] {test.__name__}"
        )

    print()
    print(
        "PASS - entity outlier semantic fallback v0.1"
    )


if __name__ == "__main__":
    main()
