from __future__ import annotations


from pathlib import (
    Path,
)


# ============================================================
# VERSION
# ============================================================


CI_PUBLISH_WORKFLOW_TEST_VERSION = (
    "ci_publish_workflow_test_v0.1"
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
    / "datalens-publish.yml"
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
            "GHCR publication workflow missing: "
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


def test_identity(
) -> None:
    workflow = load_workflow()

    assert_contains(
        workflow,
        (
            "Contract version: "
            "datalens_ci_publish_v0.1"
        ),
        "Publish contract version missing.",
    )

    assert_contains(
        workflow,
        "name: DataLens Image Publish",
        "Unexpected publication workflow name.",
    )

    assert_contains(
        workflow,
        "name: Publish Runtime Images",
        "Unexpected publication job name.",
    )


# ============================================================
# TEST 2
# REUSABLE-ONLY ENTRYPOINT
# ============================================================


def test_reusable_only_entrypoint(
) -> None:
    workflow = load_workflow()

    event_contract = workflow.split(
        "permissions:",
        1,
    )[0]

    assert_contains(
        event_contract,
        "workflow_call:",
        (
            "Publication workflow must be invoked "
            "through workflow_call."
        ),
    )

    assert_not_contains(
        event_contract,
        "pull_request:",
        "PRs must never directly publish images.",
    )

    assert_not_contains(
        event_contract,
        "workflow_dispatch:",
        (
            "Manual dispatch must not bypass "
            "the Runtime Gate."
        ),
    )

    assert_not_contains(
        event_contract,
        "  push:",
        (
            "Publication workflow must not "
            "independently publish on push."
        ),
    )


# ============================================================
# TEST 3
# PERMISSIONS
# ============================================================


def test_permissions(
) -> None:
    workflow = load_workflow()

    assert_contains(
        workflow,
        "contents: read",
        "Publication requires read-only contents.",
    )

    assert_contains(
        workflow,
        "packages: write",
        "GHCR publication permission missing.",
    )

    for forbidden in (
        "contents: write",
        "actions: write",
        "id-token: write",
    ):
        assert_not_contains(
            workflow,
            forbidden,
            "Unexpected publication permission.",
        )


# ============================================================
# TEST 4
# AUTHORIZED REFS
# ============================================================


def test_authorized_refs(
) -> None:
    workflow = load_workflow()

    assert_contains(
        workflow,
        "refs/heads/main",
        "Main publication ref missing.",
    )

    assert_contains(
        workflow,
        (
            "refs/heads/chore/"
            "organize-artifact-store-wiring"
        ),
        "Integration publication ref missing.",
    )

    assert_contains(
        workflow,
        (
            "::error::Ref is not authorized "
            "for GHCR publication."
        ),
        "Unexpected refs must fail closed.",
    )


# ============================================================
# TEST 5
# REGISTRY / IMAGE NAMES
# ============================================================


def test_registry_contract(
) -> None:
    workflow = load_workflow()

    assert_contains(
        workflow,
        "REGISTRY: ghcr.io",
        "Registry must be GHCR.",
    )

    assert_contains(
        workflow,
        (
            "ghcr.io/${{ github.repository_owner }}"
            "/datalens-api"
        ),
        "API GHCR image name missing.",
    )

    assert_contains(
        workflow,
        (
            "ghcr.io/${{ github.repository_owner }}"
            "/datalens-web"
        ),
        "Web GHCR image name missing.",
    )


# ============================================================
# TEST 6
# AUTHENTICATION
# ============================================================


def test_authentication_contract(
) -> None:
    workflow = load_workflow()

    assert_contains(
        workflow,
        "uses: docker/login-action@v4",
        "Current GHCR login action missing.",
    )

    assert_contains(
        workflow,
        "password: ${{ github.token }}",
        (
            "Publication must use the ephemeral "
            "GitHub workflow token."
        ),
    )

    assert_not_contains(
        workflow,
        "secrets.PAT",
        "A long-lived PAT must not be required.",
    )

    assert_not_contains(
        workflow,
        "docker login",
        (
            "Registry login must remain in the "
            "auditable Docker login action."
        ),
    )


# ============================================================
# TEST 7
# ACTION VERSIONS
# ============================================================


def test_action_versions(
) -> None:
    workflow = load_workflow()

    for action in (
        "actions/checkout@v7",
        "docker/setup-buildx-action@v4",
        "docker/login-action@v4",
        "docker/metadata-action@v6",
        "docker/build-push-action@v7",
    ):
        assert_contains(
            workflow,
            action,
            (
                "Expected current action major "
                f"missing: {action}"
            ),
        )


# ============================================================
# TEST 8
# TAG POLICY
# ============================================================


def test_tag_policy(
) -> None:
    workflow = load_workflow()

    for fragment in (
        'channel="latest"',
        'channel="integration"',
        'short_sha="${GITHUB_SHA::12}"',
        (
            "type=raw,value=${{ "
            "steps.release.outputs.channel }}"
        ),
        (
            "type=raw,value=sha-${{ "
            "steps.release.outputs.short_sha }}"
        ),
    ):
        assert_contains(
            workflow,
            fragment,
            "GHCR tag policy is incomplete.",
        )


# ============================================================
# TEST 9
# OCI SOURCE LABELS
# ============================================================


def test_oci_labels(
) -> None:
    workflow = load_workflow()

    assert_contains(
        workflow,
        "org.opencontainers.image.source=",
        "OCI repository source label missing.",
    )

    assert_contains(
        workflow,
        "org.opencontainers.image.revision=",
        "OCI Git revision label missing.",
    )


# ============================================================
# TEST 10
# BUILD CONTRACT
# ============================================================


def test_build_contract(
) -> None:
    workflow = load_workflow()

    for fragment in (
        "context: ./apps/api",
        "file: ./apps/api/Dockerfile",
        "context: ./apps/web",
        "file: ./apps/web/Dockerfile",
        "platforms: linux/amd64",
        "push: true",
        (
            "NEXT_PUBLIC_DATALENS_API_URL="
            "http://127.0.0.1:8000"
        ),
    ):
        assert_contains(
            workflow,
            fragment,
            "Container publication build contract incomplete.",
        )


# ============================================================
# TEST 11
# POST-PUSH VERIFICATION
# ============================================================


def test_registry_verification(
) -> None:
    workflow = load_workflow()

    assert_contains(
        workflow,
        "docker buildx imagetools inspect",
        "Published manifest verification missing.",
    )

    assert_contains(
        workflow,
        "steps.api_build.outputs.digest",
        "API pushed digest verification missing.",
    )

    assert_contains(
        workflow,
        "steps.web_build.outputs.digest",
        "Web pushed digest verification missing.",
    )


# ============================================================
# RUNNER
# ============================================================


TESTS = [
    (
        "Publish workflow identity",
        test_identity,
    ),
    (
        "Publish reusable-only entrypoint",
        test_reusable_only_entrypoint,
    ),
    (
        "Publish minimal permissions",
        test_permissions,
    ),
    (
        "Publish authorized refs",
        test_authorized_refs,
    ),
    (
        "Publish GHCR image names",
        test_registry_contract,
    ),
    (
        "Publish ephemeral authentication",
        test_authentication_contract,
    ),
    (
        "Publish action versions",
        test_action_versions,
    ),
    (
        "Publish tag policy",
        test_tag_policy,
    ),
    (
        "Publish OCI labels",
        test_oci_labels,
    ),
    (
        "Publish build contract",
        test_build_contract,
    ),
    (
        "Publish registry verification",
        test_registry_verification,
    ),
]


def main(
) -> None:
    print(
        "=== DATALENS GHCR PUBLISH CONTRACT v0.1 ==="
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
            "GHCR publication contract checks"
        )
    )

    print(
        f"Rule: {CI_PUBLISH_WORKFLOW_TEST_VERSION}"
    )


if __name__ == "__main__":
    main()