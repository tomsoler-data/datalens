from __future__ import annotations


from app.analysis.entity_outlier_requests import (
    EntityOutlierIntentResolution,
)

import app.planning.analytical_request_router as router


def test_provided_resolution_is_not_reclassified() -> None:
    original_resolver = (
        router
        .resolve_entity_outlier_intent_with_semantic_fallback
    )

    original_runner = (
        router.run_entity_outlier_request
    )

    semantic_called = False
    received_resolution = None

    resolution = (
        EntityOutlierIntentResolution(
            status="matched",
            objective=(
                "Which customers behave very differently "
                "from the rest?"
            ),
            normalized_objective=(
                "which customers behave very differently "
                "from the rest"
            ),
            intent=(
                "customer_entity_outlier_detection"
            ),
            entity_kind="customer",
            reason="synthetic semantic resolution",
            resolution_source="semantic",
            resolution_model="qwen3.5:4b",
        )
    )

    def forbidden_resolver(
        objective,
    ):
        nonlocal semantic_called
        semantic_called = True

        raise AssertionError(
            "Already validated resolution was reclassified."
        )

    def fake_runner(
        *,
        objective,
        source_dataset_records,
        top_profile_limit,
        resolution,
    ):
        nonlocal received_resolution
        received_resolution = resolution

        class Result:
            status = "ready"

        return Result()

    try:
        (
            router
            .resolve_entity_outlier_intent_with_semantic_fallback
        ) = forbidden_resolver

        router.run_entity_outlier_request = (
            fake_runner
        )

        result = (
            router._try_entity_outlier_route(
                objective=
                    resolution.objective,

                source_dataset_records=
                    [],

                top_profile_limit=
                    50,

                resolution=
                    resolution,
            )
        )

    finally:
        (
            router
            .resolve_entity_outlier_intent_with_semantic_fallback
        ) = original_resolver

        router.run_entity_outlier_request = (
            original_runner
        )

    assert result is not None
    assert semantic_called is False
    assert received_resolution is resolution


def test_router_resolves_only_once_when_not_precomputed() -> None:
    original_resolver = (
        router
        .resolve_entity_outlier_intent_with_semantic_fallback
    )

    original_runner = (
        router.run_entity_outlier_request
    )

    resolution_count = 0
    received_resolution = None

    resolution = (
        EntityOutlierIntentResolution(
            status="matched",
            objective="Natural customer anomaly request",
            normalized_objective=(
                "natural customer anomaly request"
            ),
            intent=(
                "customer_entity_outlier_detection"
            ),
            entity_kind="customer",
            reason="synthetic semantic resolution",
            resolution_source="semantic",
            resolution_model="qwen3.5:4b",
        )
    )

    def fake_resolver(
        objective,
    ):
        nonlocal resolution_count
        resolution_count += 1

        return resolution

    def fake_runner(
        *,
        objective,
        source_dataset_records,
        top_profile_limit,
        resolution,
    ):
        nonlocal received_resolution
        received_resolution = resolution

        class Result:
            status = "ready"

        return Result()

    try:
        (
            router
            .resolve_entity_outlier_intent_with_semantic_fallback
        ) = fake_resolver

        router.run_entity_outlier_request = (
            fake_runner
        )

        result = (
            router._try_entity_outlier_route(
                objective=
                    resolution.objective,

                source_dataset_records=
                    [],

                top_profile_limit=
                    50,
            )
        )

    finally:
        (
            router
            .resolve_entity_outlier_intent_with_semantic_fallback
        ) = original_resolver

        router.run_entity_outlier_request = (
            original_runner
        )

    assert result is not None
    assert resolution_count == 1
    assert received_resolution is resolution


def main() -> None:
    tests = [
        test_provided_resolution_is_not_reclassified,
        test_router_resolves_only_once_when_not_precomputed,
    ]

    print(
        "=== ENTITY OUTLIER SINGLE RESOLUTION WIRING v0.1 ==="
    )

    for test in tests:
        test()

        print(
            f"[PASS] {test.__name__}"
        )

    print()
    print(
        "PASS - entity outlier single resolution wiring v0.1"
    )


if __name__ == "__main__":
    main()
