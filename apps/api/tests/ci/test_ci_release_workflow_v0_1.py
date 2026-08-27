from __future__ import annotations


from pathlib import (
    Path,
)


# ============================================================
# VERSION
# ============================================================


CI_RELEASE_WORKFLOW_TEST_VERSION = (
    "ci_release_workflow_test_v0.1"
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
    / "datalens-release.yml"
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
            "Release Promotion workflow missing: "
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

    for fragment in (
        (
            "Contract version: "
            "datalens_release_promotion_v0.1"
        ),
        "name: DataLens Release Promotion",
        "name: Validate Release Promotion",
        "name: Promote Runtime Release",
    ):
        assert_contains(
            workflow,
            fragment,
            "Release workflow identity incomplete.",
        )


# ============================================================
# TEST 2
# MANUAL-ONLY ENTRYPOINT
# ============================================================


def test_manual_only_entrypoint(
) -> None:
    workflow = load_workflow()

    event_contract = workflow.split(
        "permissions:",
        1,
    )[0]

    assert_contains(
        event_contract,
        "workflow_dispatch:",
        (
            "Release Promotion must use "
            "manual workflow_dispatch."
        ),
    )

    for forbidden in (
        "pull_request:",
        "workflow_call:",
        "  push:",
    ):
        assert_not_contains(
            event_contract,
            forbidden,
            (
                "Release Promotion must not "
                "have an automatic entrypoint."
            ),
        )


# ============================================================
# TEST 3
# VERSION / EXPLICIT CONFIRMATION
# ============================================================


def test_version_contract(
) -> None:
    workflow = load_workflow()

    for fragment in (
        "version:",
        "confirm:",
        'version="${{ inputs.version }}"',
        'confirm="${{ inputs.confirm }}"',
        (
            "^v(0|[1-9][0-9]*)"
            r"\.(0|[1-9][0-9]*)"
            r"\.(0|[1-9][0-9]*)$"
        ),
        '"RELEASE ${version}"',
        (
            "Release version must use stable "
            "SemVer vMAJOR.MINOR.PATCH."
        ),
        "Release confirmation mismatch.",
    ):
        assert_contains(
            workflow,
            fragment,
            (
                "Version / confirmation "
                "contract incomplete."
            ),
        )


# ============================================================
# TEST 4
# LEAST PRIVILEGE
# ============================================================


def test_permissions_contract(
) -> None:
    workflow = load_workflow()

    assert_contains(
        workflow,
        "permissions:\n  contents: read",
        (
            "Default workflow permission must "
            "be read-only contents."
        ),
    )

    assert_contains(
        workflow,
        "packages: read",
        (
            "Preflight job must have "
            "read-only package access."
        ),
    )

    assert_true(
        workflow.count(
            "contents: write"
        ) == 1,
        (
            "contents: write must exist exactly "
            "once on the promotion job."
        ),
    )

    assert_true(
        workflow.count(
            "packages: write"
        ) == 1,
        (
            "packages: write must exist exactly "
            "once on the promotion job."
        ),
    )

    for forbidden in (
        "actions: write",
        "id-token: write",
    ):
        assert_not_contains(
            workflow,
            forbidden,
            (
                "Unexpected privileged "
                "Release Promotion permission."
            ),
        )


# ============================================================
# TEST 5
# CURRENT MAIN ONLY
# ============================================================


def test_main_source_contract(
) -> None:
    workflow = load_workflow()

    for fragment in (
        "refs/heads/main",
        "git ls-remote",
        "refs/heads/main",
        "remote_main",
        "GITHUB_SHA",
        (
            "Release Promotion is authorized "
            "only from main."
        ),
        (
            "Workflow SHA is not the "
            "current origin/main."
        ),
        (
            "origin/main moved after "
            "release preflight."
        ),
    ):
        assert_contains(
            workflow,
            fragment,
            (
                "Current-main source "
                "contract incomplete."
            ),
        )

    assert_not_contains(
        workflow,
        (
            "refs/heads/chore/"
            "organize-artifact-store-wiring"
        ),
        (
            "Integration branch must never "
            "be an authorized release source."
        ),
    )


# ============================================================
# TEST 6
# IMMUTABLE SHA SOURCE / OCI PROVENANCE
# ============================================================


def test_immutable_source_contract(
) -> None:
    workflow = load_workflow()

    for fragment in (
        (
            "ghcr.io/${{ github.repository_owner }}"
            "/datalens-api"
        ),
        (
            "ghcr.io/${{ github.repository_owner }}"
            "/datalens-web"
        ),
        'short_sha="${GITHUB_SHA::12}"',
        "sha-${short_sha}",
        "docker buildx imagetools inspect",
        "docker pull",
        "--platform linux/amd64",
        "org.opencontainers.image.revision",
        (
            "API OCI revision does not match "
            "the release commit."
        ),
        (
            "Web OCI revision does not match "
            "the release commit."
        ),
        (
            "Immutable release source "
            "revalidated."
        ),
    ):
        assert_contains(
            workflow,
            fragment,
            (
                "Immutable release source "
                "contract incomplete."
            ),
        )


# ============================================================
# TEST 7
# NO REBUILD
# ============================================================


def test_no_rebuild_contract(
) -> None:
    workflow = load_workflow()

    for forbidden in (
        "docker/build-push-action@",
        "docker compose build",
        "docker build ",
        "context: ./apps/api",
        "context: ./apps/web",
        "file: ./apps/api/Dockerfile",
        "file: ./apps/web/Dockerfile",
    ):
        assert_not_contains(
            workflow,
            forbidden,
            (
                "Release Promotion must "
                "never rebuild runtime images."
            ),
        )

    assert_contains(
        workflow,
        "Docker rebuild: \\`no\\`",
        (
            "Release summary must state "
            "the no-rebuild policy."
        ),
    )


# ============================================================
# TEST 8
# CONFLICT / RECOVERY GUARDS
# ============================================================


def test_conflict_guards(
) -> None:
    workflow = load_workflow()

    for fragment in (
        (
            "release tag already exists "
            "with a different digest."
        ),
        (
            "Release target already exists "
            "with an incompatible digest."
        ),
        (
            "Git tag already exists "
            "on another commit."
        ),
        "GitHub Release already exists.",
        (
            "already points to the "
            "expected digest."
        ),
        "already promoted.",
    ):
        assert_contains(
            workflow,
            fragment,
            (
                "Release conflict / recovery "
                "guard incomplete."
            ),
        )


# ============================================================
# TEST 9
# DIGEST-PRESERVING PROMOTION
# ============================================================


def test_digest_promotion_contract(
) -> None:
    workflow = load_workflow()

    for fragment in (
        "docker buildx imagetools create",
        '--tag "${target}"',
        '"${image}@${expected_digest}"',
        "Release tag digest mismatch.",
        "promoted without rebuild.",
        "api_digest",
        "web_digest",
        "API source digest changed after preflight.",
        "Web source digest changed after preflight.",
    ):
        assert_contains(
            workflow,
            fragment,
            (
                "Digest-preserving promotion "
                "contract incomplete."
            ),
        )


# ============================================================
# TEST 10
# GIT TAG / GITHUB RELEASE
# ============================================================


def test_git_release_contract(
) -> None:
    workflow = load_workflow()

    for fragment in (
        "git tag",
        "-a",
        '"${version}"',
        '"${GITHUB_SHA}"',
        "git push",
        '"refs/tags/${version}"',
        "gh release create",
        "--verify-tag",
        '--title "DataLens ${version}"',
        "--notes-file release-notes.md",
        (
            "Release tags reuse the already "
            "validated immutable GHCR manifests"
        ),
    ):
        assert_contains(
            workflow,
            fragment,
            (
                "Git tag / GitHub Release "
                "contract incomplete."
            ),
        )


# ============================================================
# TEST 11
# FINAL VERIFICATION
# ============================================================


def test_final_verification_contract(
) -> None:
    workflow = load_workflow()

    for fragment in (
        "Final API release digest mismatch.",
        "Final Web release digest mismatch.",
        "Final Git tag target mismatch.",
        "Release Promotion fully verified.",
        "gh release view",
        "name: Write release summary",
    ):
        assert_contains(
            workflow,
            fragment,
            (
                "Final Release Promotion "
                "verification incomplete."
            ),
        )


# ============================================================
# RUNNER
# ============================================================


TESTS = [
    (
        "Release workflow identity",
        test_identity,
    ),
    (
        "Release manual-only entrypoint",
        test_manual_only_entrypoint,
    ),
    (
        "Release stable version contract",
        test_version_contract,
    ),
    (
        "Release least privilege",
        test_permissions_contract,
    ),
    (
        "Release current-main source",
        test_main_source_contract,
    ),
    (
        "Release immutable source provenance",
        test_immutable_source_contract,
    ),
    (
        "Release no-rebuild policy",
        test_no_rebuild_contract,
    ),
    (
        "Release conflict guards",
        test_conflict_guards,
    ),
    (
        "Release digest-preserving promotion",
        test_digest_promotion_contract,
    ),
    (
        "Release Git tag and GitHub Release",
        test_git_release_contract,
    ),
    (
        "Release final verification",
        test_final_verification_contract,
    ),
]


def main(
) -> None:
    print(
        "=== DATALENS RELEASE PROMOTION CONTRACT v0.1 ==="
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
            "Release Promotion contract checks"
        )
    )

    print(
        f"Rule: {CI_RELEASE_WORKFLOW_TEST_VERSION}"
    )


if __name__ == "__main__":
    main()