from __future__ import annotations


from pathlib import (
    Path,
)


# ============================================================
# VERSION
# ============================================================


CI_RUNTIME_WORKFLOW_TEST_VERSION = (
    "ci_runtime_workflow_test_v0.1"
)


# ============================================================
# PATHS
# ============================================================


REPOSITORY_ROOT = (
    Path(__file__)
    .resolve()
    .parents[4]
)


WORKFLOW_PATH = (
    REPOSITORY_ROOT
    / ".github"
    / "workflows"
    / "datalens-runtime.yml"
)


# ============================================================
# ASSERTIONS
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


def load_workflow(
) -> str:
    assert_true(
        WORKFLOW_PATH.is_file(),
        (
            "Runtime CI workflow is missing: "
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


def test_workflow_identity(
) -> None:
    workflow = load_workflow()

    assert_contains(
        workflow,
        (
            "Contract version: "
            "datalens_ci_runtime_gate_v0.1"
        ),
        "Runtime CI contract version missing.",
    )

    assert_contains(
        workflow,
        "name: DataLens Runtime Gate",
        "Unexpected Runtime CI workflow name.",
    )

    assert_contains(
        workflow,
        "name: Docker Runtime Gate",
        "Unexpected Runtime CI job name.",
    )


# ============================================================
# TEST 2
# TRIGGERS
# ============================================================


def test_workflow_triggers(
) -> None:
    workflow = load_workflow()

    for fragment in (
        "push:",
        "pull_request:",
        "workflow_dispatch:",
        "- main",
        (
            "- chore/"
            "organize-artifact-store-wiring"
        ),
        '"apps/api/**"',
        '"apps/web/**"',
        '"compose.yaml"',
        '".github/workflows/datalens-publish.yml"',
    ):
        assert_contains(
            workflow,
            fragment,
            (
                "Required Runtime CI trigger "
                "contract is missing."
            ),
        )


# ============================================================
# TEST 3
# MINIMAL PERMISSIONS
# ============================================================


def test_minimal_permissions(
) -> None:
    workflow = load_workflow()

    assert_contains(
        workflow,
        "permissions:\n  contents: read",
        (
            "Runtime CI must use read-only "
            "repository contents permission."
        ),
    )

    for forbidden in (
        "contents: write",
        "id-token: write",
    ):
        assert_not_contains(
            workflow,
            forbidden,
            (
                "Runtime validation workflow "
                "must not request this permission."
            ),
        )

    assert_true(
        workflow.count(
            "packages: write"
        ) == 1,
        (
            "packages: write must exist exactly "
            "once, on the gated publish job."
        ),
    )


# ============================================================
# TEST 4
# TOOLCHAIN
# ============================================================


def test_toolchain_versions(
) -> None:
    workflow = load_workflow()

    assert_contains(
        workflow,
        'python-version: "3.13"',
        "Runtime CI must validate Python 3.13.",
    )

    assert_contains(
        workflow,
        'node-version: "22"',
        "Runtime CI must validate Node.js 22.",
    )

    assert_contains(
        workflow,
        "docker compose version",
        "Compose environment diagnostic missing.",
    )


# ============================================================
# TEST 5
# SECURITY REGRESSION
# ============================================================


def test_security_regression_contract(
) -> None:
    workflow = load_workflow()

    assert_contains(
        workflow,
        "tests/security/test_*.py",
        (
            "Runtime CI must discover the "
            "security regression scripts."
        ),
    )

    assert_contains(
        workflow,
        'test_module="${test_file%.py}"',
        (
            "Runtime CI must derive a Python "
            "module from each security test path."
        ),
    )

    assert_contains(
        workflow,
        r'test_module="${test_module//\//.}"',
        (
            "Runtime CI must convert security "
            "test paths to dotted modules."
        ),
    )

    assert_contains(
        workflow,
        'python -m "${test_module}"',
        (
            "Runtime CI must execute security "
            "tests as importable Python modules."
        ),
    )

    assert_not_contains(
        workflow,
        'python "${test_file}"',
        (
            "Runtime CI must not execute security "
            "tests by filesystem path."
        ),
    )

    assert_contains(
        workflow,
        "./app/security/llm_egress.py",
        "LLM egress compilation missing.",
    )

    assert_contains(
        workflow,
        "./app/ai/ollama_runtime.py",
        "Ollama runtime compilation missing.",
    )


# ============================================================
# TEST 6
# DOCKER BUILD / COMPOSE
# ============================================================


def test_docker_runtime_contract(
) -> None:
    workflow = load_workflow()

    for fragment in (
        "config",
        "--quiet",
        "build",
        "--pull",
        "up",
        "--detach",
        "ps",
    ):
        assert_contains(
            workflow,
            fragment,
            (
                "Required Docker runtime "
                "operation missing."
            ),
        )


# ============================================================
# TEST 7
# HEALTH / HTTP
# ============================================================


def test_health_contract(
) -> None:
    workflow = load_workflow()

    assert_contains(
        workflow,
        "wait_for_healthy",
        "Container health wait contract missing.",
    )

    assert_contains(
        workflow,
        "http://127.0.0.1:3000",
        "Frontend smoke test missing.",
    )

    assert_contains(
        workflow,
        "http://127.0.0.1:8000/health",
        "API health smoke test missing.",
    )

    assert_contains(
        workflow,
        '"status": "ok"',
        "API health response assertion missing.",
    )

    assert_contains(
        workflow,
        '"service": "datalens-api"',
        "API service identity assertion missing.",
    )


# ============================================================
# TEST 8
# NON-ROOT
# ============================================================


def test_non_root_contract(
) -> None:
    workflow = load_workflow()

    assert_contains(
        workflow,
        "id -u",
        "Runtime UID assertion missing.",
    )

    assert_contains(
        workflow,
        '"10001"',
        "Expected DataLens runtime UID missing.",
    )


# ============================================================
# TEST 9
# PERSISTENCE
# ============================================================


def test_persistence_contract(
) -> None:
    workflow = load_workflow()

    for fragment in (
        "ci-runtime-persistence-sentinel.txt",
        "DATALENS_RUNTIME_CI_V0_1",
        "docker volume inspect",
        "datalens-api-var-v0-1",
        "before_hash",
        "after_hash",
    ):
        assert_contains(
            workflow,
            fragment,
            (
                "Persistent volume validation "
                "contract is incomplete."
            ),
        )


# ============================================================
# TEST 10
# OLLAMA CI BOUNDARY
# ============================================================


def test_no_real_model_dependency(
) -> None:
    workflow = load_workflow()

    assert_contains(
        workflow,
        "DATALENS_LLM_DOCKER_BRIDGE_ENABLED",
        "Ollama bridge configuration check missing.",
    )

    assert_contains(
        workflow,
        "DATALENS_OLLAMA_HOST",
        "Ollama host configuration check missing.",
    )

    for forbidden in (
        "ollama pull",
        "/api/chat",
        "gemma3:4b",
        "classified_llm_chat(",
    ):
        assert_not_contains(
            workflow,
            forbidden,
            (
                "Hosted Runtime CI must not "
                "depend on a real local model."
            ),
        )


# ============================================================
# TEST 11
# FAILURE DIAGNOSTICS
# ============================================================


def test_failure_diagnostics(
) -> None:
    workflow = load_workflow()

    assert_contains(
        workflow,
        "if: ${{ failure() }}",
        "Failure diagnostics step missing.",
    )

    assert_contains(
        workflow,
        "--tail 250",
        "Bounded failure logs missing.",
    )


# ============================================================
# TEST 12
# CLEANUP
# ============================================================


def test_cleanup_contract(
) -> None:
    workflow = load_workflow()

    assert_contains(
        workflow,
        "if: ${{ always() }}",
        "Always-run cleanup is missing.",
    )

    assert_contains(
        workflow,
        "--volumes",
        "CI cleanup must remove test volumes.",
    )

    assert_contains(
        workflow,
        "--remove-orphans",
        "CI cleanup must remove orphan containers.",
    )


# ============================================================
# TEST 13
# GATED GHCR PUBLICATION HANDOFF
# ============================================================


def test_publish_handoff_contract(
) -> None:
    workflow = load_workflow()

    for fragment in (
        "publish-images:",
        "name: Publish GHCR Images",
        "needs:",
        "- runtime-gate",
        "github.event_name == 'push'",
        "refs/heads/main",
        (
            "refs/heads/chore/"
            "organize-artifact-store-wiring"
        ),
        "packages: write",
        (
            "uses: ./.github/workflows/"
            "datalens-publish.yml"
        ),
        (
            "python -m tests.ci."
            "test_ci_publish_workflow_v0_1"
        ),
    ):
        assert_contains(
            workflow,
            fragment,
            (
                "Runtime-to-publication gated "
                "handoff is incomplete."
            ),
        )

# ============================================================
# RUNNER
# ============================================================


TESTS = [
    (
        "Runtime workflow identity",
        test_workflow_identity,
    ),
    (
        "Runtime workflow triggers",
        test_workflow_triggers,
    ),
    (
        "Runtime minimal permissions",
        test_minimal_permissions,
    ),
    (
        "Runtime toolchain versions",
        test_toolchain_versions,
    ),
    (
        "Runtime security regressions",
        test_security_regression_contract,
    ),
    (
        "Runtime Docker contract",
        test_docker_runtime_contract,
    ),
    (
        "Runtime health contract",
        test_health_contract,
    ),
    (
        "Runtime non-root contract",
        test_non_root_contract,
    ),
    (
        "Runtime persistence contract",
        test_persistence_contract,
    ),
    (
        "Runtime Ollama hosted-CI boundary",
        test_no_real_model_dependency,
    ),
    (
        "Runtime failure diagnostics",
        test_failure_diagnostics,
    ),
    (
        "Runtime cleanup contract",
        test_cleanup_contract,
    ),
    (
        "Runtime gated GHCR publication",
        test_publish_handoff_contract,
    ),
]


def main(
) -> None:
    print(
        "=== DATALENS RUNTIME CI CONTRACT v0.1 ==="
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
            "Runtime CI contract checks"
        )
    )

    print(
        f"Rule: {CI_RUNTIME_WORKFLOW_TEST_VERSION}"
    )


if __name__ == "__main__":
    main()