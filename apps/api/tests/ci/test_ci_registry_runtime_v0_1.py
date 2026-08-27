from __future__ import annotations


from pathlib import (
    Path,
)


# ============================================================
# VERSION
# ============================================================


CI_REGISTRY_RUNTIME_TEST_VERSION = (
    "ci_registry_runtime_test_v0.1"
)


# ============================================================
# PATHS
# ============================================================


REPOSITORY_ROOT = (
    Path(__file__)
    .resolve()
    .parents[4]
)


COMPOSE_PATH = (
    REPOSITORY_ROOT
    / "compose.registry.yaml"
)


WORKFLOW_PATH = (
    REPOSITORY_ROOT
    / ".github"
    / "workflows"
    / "datalens-runtime.yml"
)


# ============================================================
# HELPERS
# ============================================================


def assert_true(
    value,
    message: str,
) -> None:
    if not value:
        raise AssertionError(
            message
        )


def assert_contains(
    text: str,
    fragment: str,
    message: str,
) -> None:
    assert_true(
        fragment in text,
        (
            f"{message}\n"
            f"missing={fragment!r}"
        ),
    )


def assert_not_contains(
    text: str,
    fragment: str,
    message: str,
) -> None:
    assert_true(
        fragment not in text,
        (
            f"{message}\n"
            f"unexpected={fragment!r}"
        ),
    )


def load_compose(
) -> str:
    assert_true(
        COMPOSE_PATH.is_file(),
        (
            "Registry Runtime Compose file missing: "
            f"{COMPOSE_PATH}"
        ),
    )

    return COMPOSE_PATH.read_text(
        encoding="utf-8",
    )


def load_workflow(
) -> str:
    assert_true(
        WORKFLOW_PATH.is_file(),
        (
            "Runtime workflow missing: "
            f"{WORKFLOW_PATH}"
        ),
    )

    return WORKFLOW_PATH.read_text(
        encoding="utf-8",
    )


# ============================================================
# TEST 1
# IDENTITY
# ============================================================


def test_registry_identity(
) -> None:
    compose = load_compose()

    assert_contains(
        compose,
        (
            "Contract version: "
            "datalens_registry_runtime_v0.1"
        ),
        "Registry Runtime contract version missing.",
    )


# ============================================================
# TEST 2
# GHCR IMAGE CONTRACT
# ============================================================


def test_registry_images(
) -> None:
    compose = load_compose()

    for fragment in (
        (
            "${DATALENS_API_IMAGE:-"
            "ghcr.io/tomsoler-data/"
            "datalens-api:integration}"
        ),
        (
            "${DATALENS_WEB_IMAGE:-"
            "ghcr.io/tomsoler-data/"
            "datalens-web:integration}"
        ),
        "pull_policy: always",
    ):
        assert_contains(
            compose,
            fragment,
            "Registry image contract incomplete.",
        )


# ============================================================
# TEST 3
# NO LOCAL BUILD
# ============================================================


def test_no_local_build(
) -> None:
    compose = load_compose()

    for forbidden in (
        "build:",
        "dockerfile:",
        "context: ./apps/api",
        "context: ./apps/web",
    ):
        assert_not_contains(
            compose,
            forbidden,
            (
                "Registry Runtime must not "
                "contain local image builds."
            ),
        )


# ============================================================
# TEST 4
# LOOPBACK PORTS
# ============================================================


def test_loopback_ports(
) -> None:
    compose = load_compose()

    assert_contains(
        compose,
        '"127.0.0.1:8000:8000"',
        "API must bind only to host loopback.",
    )

    assert_contains(
        compose,
        '"127.0.0.1:3000:3000"',
        "Web must bind only to host loopback.",
    )


# ============================================================
# TEST 5
# PERSISTENCE
# ============================================================


def test_persistence(
) -> None:
    compose = load_compose()

    for fragment in (
        "datalens-api-var:/app/var",
        "name: datalens-api-var-v0-1",
    ):
        assert_contains(
            compose,
            fragment,
            "Registry persistence contract incomplete.",
        )


# ============================================================
# TEST 6
# OLLAMA BRIDGE
# ============================================================


def test_ollama_bridge(
) -> None:
    compose = load_compose()

    assert_contains(
        compose,
        (
            'DATALENS_LLM_DOCKER_BRIDGE_ENABLED: '
            '"1"'
        ),
        "Docker LLM bridge opt-in missing.",
    )

    assert_contains(
        compose,
        (
            'DATALENS_OLLAMA_HOST: '
            '"http://host.docker.internal:11434"'
        ),
        "Exact Docker Ollama endpoint missing.",
    )


# ============================================================
# TEST 7
# SERVICE ORDER / INIT
# ============================================================


def test_service_contract(
) -> None:
    compose = load_compose()

    assert_contains(
        compose,
        "condition: service_healthy",
        "Web must wait for healthy API.",
    )

    assert_true(
        compose.count(
            "init: true"
        ) == 2,
        "Both registry services must enable init.",
    )


# ============================================================
# TEST 8
# WORKFLOW TRIGGER
# ============================================================


def test_workflow_trigger(
) -> None:
    workflow = load_workflow()

    assert_true(
        workflow.count(
            '"compose.registry.yaml"'
        ) >= 2,
        (
            "Registry Compose changes must trigger "
            "push and PR Runtime Gate events."
        ),
    )

    assert_contains(
        workflow,
        (
            "python -m tests.ci."
            "test_ci_registry_runtime_v0_1"
        ),
        (
            "Registry Runtime CI contract must "
            "run before publication."
        ),
    )


# ============================================================
# TEST 9
# POST-PUBLISH ORDER
# ============================================================


def test_post_publish_order(
) -> None:
    workflow = load_workflow()

    for fragment in (
        "verify-published-images:",
        "name: Verify Published Registry Runtime",
        "- publish-images",
        "github.event_name == 'push'",
        "packages: read",
    ):
        assert_contains(
            workflow,
            fragment,
            (
                "Post-publication Registry Runtime "
                "gate is incomplete."
            ),
        )


# ============================================================
# TEST 10
# IMMUTABLE SHA IMAGES
# ============================================================


def test_immutable_image_resolution(
) -> None:
    workflow = load_workflow()

    for fragment in (
        'short_sha="${GITHUB_SHA::12}"',
        (
            "DATALENS_API_IMAGE="
            "ghcr.io/${GITHUB_REPOSITORY_OWNER}/"
            "datalens-api:sha-${short_sha}"
        ),
        (
            "DATALENS_WEB_IMAGE="
            "ghcr.io/${GITHUB_REPOSITORY_OWNER}/"
            "datalens-web:sha-${short_sha}"
        ),
    ):
        assert_contains(
            workflow,
            fragment,
            (
                "Published runtime verification "
                "must use immutable commit images."
            ),
        )


# ============================================================
# TEST 11
# REGISTRY SMOKE
# ============================================================


def test_registry_smoke(
) -> None:
    workflow = load_workflow()

    for fragment in (
        "-f ./compose.registry.yaml",
        "pull",
        "--no-build",
        "wait_for_registry_healthy",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8000/health",
        "id -u",
        "id -g",
        "DATALENS_LLM_DOCKER_BRIDGE_ENABLED",
        "DATALENS_OLLAMA_HOST",
    ):
        assert_contains(
            workflow,
            fragment,
            (
                "Published Registry Runtime smoke "
                "contract is incomplete."
            ),
        )


# ============================================================
# TEST 12
# REGISTRY PERSISTENCE / CLEANUP
# ============================================================


def test_registry_persistence_cleanup(
) -> None:
    workflow = load_workflow()

    for fragment in (
        "ci-registry-persistence-sentinel.txt",
        "DATALENS_REGISTRY_CI_V0_1",
        "registry_before_hash",
        "registry_after_hash",
        "if: ${{ always() }}",
        "--volumes",
        "--remove-orphans",
    ):
        assert_contains(
            workflow,
            fragment,
            (
                "Published Registry Runtime "
                "persistence/cleanup incomplete."
            ),
        )


# ============================================================
# RUNNER
# ============================================================


TESTS = [
    (
        "Registry Runtime identity",
        test_registry_identity,
    ),
    (
        "Registry GHCR images",
        test_registry_images,
    ),
    (
        "Registry no local build",
        test_no_local_build,
    ),
    (
        "Registry loopback ports",
        test_loopback_ports,
    ),
    (
        "Registry persistence",
        test_persistence,
    ),
    (
        "Registry Ollama bridge",
        test_ollama_bridge,
    ),
    (
        "Registry service contract",
        test_service_contract,
    ),
    (
        "Registry workflow trigger",
        test_workflow_trigger,
    ),
    (
        "Registry post-publish ordering",
        test_post_publish_order,
    ),
    (
        "Registry immutable SHA images",
        test_immutable_image_resolution,
    ),
    (
        "Registry runtime smoke",
        test_registry_smoke,
    ),
    (
        "Registry persistence cleanup",
        test_registry_persistence_cleanup,
    ),
]


def main(
) -> None:
    print(
        "=== DATALENS REGISTRY RUNTIME CONTRACT v0.1 ==="
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
                (
                    f"       {type(error).__name__}: "
                    f"{error}"
                )
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
            "Registry Runtime contract checks"
        )
    )

    print(
        f"Rule: {CI_REGISTRY_RUNTIME_TEST_VERSION}"
    )


if __name__ == "__main__":
    main()
