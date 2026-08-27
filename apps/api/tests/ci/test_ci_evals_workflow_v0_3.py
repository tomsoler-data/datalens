from __future__ import annotations

from pathlib import Path


CI_EVALS_GATE_RULE_VERSION = (
    "datalens_ci_evals_gate_v0.3"
)

CI_PYTHON_VERSION = (
    "3.13"
)

PROMPT_ANALYSIS_E2E_MODULE = (
    "tests.analysis."
    "test_analysis_prompt_e2e_deterministic_v0_4"
)


# ============================================================
# REPOSITORY
# ============================================================


def repository_root(
) -> Path:
    """
    Resolve the DataLens repository root without depending on
    the physical depth of this test file.

    This keeps the CI contract test stable when tests are
    reorganized under domain-specific directories.
    """

    current_file = (
        Path(
            __file__
        )
        .resolve()
    )


    for parent in current_file.parents:

        candidate = (
            parent
            /
            ".github"
            /
            "workflows"
            /
            "datalens-evals.yml"
        )


        if candidate.exists():

            return parent


    raise AssertionError(
        "Could not locate the DataLens repository root "
        "from test_ci_evals_workflow_v0_3.py."
    )


def workflow_path(
) -> Path:
    return (
        repository_root()
        /
        ".github"
        /
        "workflows"
        /
        "datalens-evals.yml"
    )


def workflow_text(
) -> str:
    path = (
        workflow_path()
    )


    assert path.exists(), (
        "GitHub Actions workflow missing: "
        f"{path}"
    )


    return path.read_text(
        encoding="utf-8"
    )


# ============================================================
# VERSION
# ============================================================


def test_workflow_is_versioned(
) -> None:
    text = workflow_text()


    assert (
        CI_EVALS_GATE_RULE_VERSION
        in text
    )


    print(
        "CI eval workflow contract is versioned: PASS"
    )


# ============================================================
# PERMISSIONS
# ============================================================


def test_workflow_uses_read_only_repository_permission(
) -> None:
    text = workflow_text()


    assert (
        "permissions:\n  contents: read"
        in text
    )


    print(
        "CI eval workflow uses read-only repository permission: PASS"
    )


# ============================================================
# TRIGGERS
# ============================================================


def test_workflow_runs_on_main_push_pull_request_and_manual_dispatch(
) -> None:
    text = workflow_text()


    assert "\n  push:" in text
    assert "\n  pull_request:" in text
    assert "\n  workflow_dispatch:" in text

    assert (
        "push:\n"
        "    branches:\n"
        "      - main"
        in text
    )

    assert (
        "pull_request:\n"
        "    branches:\n"
        "      - main"
        in text
    )


    print(
        "CI eval workflow targets main pushes and pull requests: PASS"
    )


# ============================================================
# PYTHON CONTRACT
# ============================================================


def test_workflow_pins_backend_python_contract(
) -> None:
    text = workflow_text()


    assert (
        f'python-version: "{CI_PYTHON_VERSION}"'
        in text
    )


    print(
        "CI eval workflow mirrors the backend Python contract: PASS"
    )


# ============================================================
# TEST LAYOUT
# ============================================================


def test_workflow_validates_its_contract(
) -> None:
    text = workflow_text()


    assert (
        "python -m tests.ci."
        "test_ci_evals_workflow_v0_3"
        in text
    )


    print(
        "CI eval workflow validates its own contract: PASS"
    )


# ============================================================
# EVAL LAYERS
# ============================================================


def test_workflow_executes_all_eval_layers(
) -> None:
    text = workflow_text()


    required_commands = [
        (
            "python -m tests.evals."
            "test_analysis_benchmark_v0_1"
        ),
        (
            "python -m tests.evals."
            "test_eval_suite_runner_v0_1"
        ),
        (
            "python -m tests.evals."
            "test_eval_coverage_v0_1"
        ),
        (
            "python -m tests.evals."
            "test_eval_regression_gate_v0_1"
        ),
        (
            "python -m "
            + PROMPT_ANALYSIS_E2E_MODULE
        ),
        "python -m app.evals.suite_runner",
        "python -m app.evals.regression_gate",
    ]


    missing = [
        command

        for command
        in required_commands

        if command not in text
    ]


    assert not missing, (
        "Missing eval CI commands: "
        +
        ", ".join(
            missing
        )
    )


    print(
        "CI eval workflow executes benchmark, suite, "
        "coverage, prompt E2E and gate: PASS"
    )


def test_workflow_executes_deterministic_prompt_analysis_e2e(
) -> None:
    text = workflow_text()


    command = (
        "python -m "
        + PROMPT_ANALYSIS_E2E_MODULE
    )


    assert (
        command
        in text
    )


    assert (
        "Run deterministic prompt-analysis E2E regression"
        in text
    )


    print(
        "CI eval workflow executes deterministic prompt-analysis "
        "E2E regression: PASS"
    )


def test_prompt_analysis_e2e_runs_before_real_eval_suite(
) -> None:
    text = workflow_text()


    prompt_command = (
        "python -m "
        + PROMPT_ANALYSIS_E2E_MODULE
    )

    real_eval_command = (
        "python -m app.evals.suite_runner"
    )


    prompt_index = text.find(
        prompt_command
    )

    real_eval_index = text.find(
        real_eval_command
    )


    assert (
        prompt_index
        >=
        0
    )

    assert (
        real_eval_index
        >=
        0
    )

    assert (
        prompt_index
        <
        real_eval_index
    ), (
        "Deterministic prompt-analysis E2E regression must "
        "run before the real eval suite."
    )


    print(
        "Deterministic prompt-analysis E2E runs before "
        "the real eval suite: PASS"
    )


# ============================================================
# BASELINE
# ============================================================


def test_workflow_enforces_frozen_baseline(
) -> None:
    text = workflow_text()


    assert (
        "./app/evals/baselines/"
        "discovery_prioritization_v0_1.json"
        in text
    )


    print(
        "CI eval workflow enforces the frozen regression baseline: PASS"
    )


# ============================================================
# EVIDENCE
# ============================================================


def test_workflow_persists_machine_readable_evidence(
) -> None:
    text = workflow_text()


    assert (
        "--json-output ./evals_report.json"
        in text
    )

    assert (
        "--json-output ./eval_regression_gate_report.json"
        in text
    )

    assert (
        "actions/upload-artifact@v7"
        in text
    )

    assert (
        "if: ${{ always() }}"
        in text
    )


    print(
        "CI eval workflow preserves JSON evidence as an artifact: PASS"
    )


# ============================================================
# FAILURE SEMANTICS
# ============================================================


def test_workflow_failure_propagates_to_ci(
) -> None:
    text = workflow_text()


    assert (
        "continue-on-error:"
        not in text
    )


    print(
        "Eval failures propagate to the GitHub Actions job: PASS"
    )


# ============================================================
# SCOPE
# ============================================================


def test_workflow_scopes_execution_to_backend_changes(
) -> None:
    text = workflow_text()


    assert (
        '- "apps/api/**"'
        in text
    )

    assert (
        "working-directory: apps/api"
        in text
    )


    print(
        "CI eval workflow is scoped to DataLens backend changes: PASS"
    )


# ============================================================
# MAIN
# ============================================================


def main(
) -> None:
    print(
        "=== DATALENS CI EVALS GATE v0.3 ==="
    )

    print()


    test_workflow_is_versioned()

    test_workflow_uses_read_only_repository_permission()

    test_workflow_runs_on_main_push_pull_request_and_manual_dispatch()

    test_workflow_pins_backend_python_contract()

    test_workflow_validates_its_contract()

    test_workflow_executes_all_eval_layers()

    test_workflow_executes_deterministic_prompt_analysis_e2e()

    test_prompt_analysis_e2e_runs_before_real_eval_suite()

    test_workflow_enforces_frozen_baseline()

    test_workflow_persists_machine_readable_evidence()

    test_workflow_failure_propagates_to_ci()

    test_workflow_scopes_execution_to_backend_changes()


    print()

    print(
        "CI Evals Gate v0.3: PASS"
    )


if __name__ == "__main__":
    main()
