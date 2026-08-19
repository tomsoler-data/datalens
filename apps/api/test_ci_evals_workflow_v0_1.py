from __future__ import annotations

from pathlib import Path


CI_EVALS_GATE_RULE_VERSION = (
    "datalens_ci_evals_gate_v0.1"
)


def repository_root(
) -> Path:
    """
    test_ci_evals_workflow_v0_1.py lives in:

        datalens/apps/api/

    parents[0] -> api
    parents[1] -> apps
    parents[2] -> datalens
    """

    return (
        Path(
            __file__
        )
        .resolve()
        .parents[
            2
        ]
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


def test_workflow_runs_on_push_pull_request_and_manual_dispatch(
) -> None:
    text = workflow_text()


    assert "\n  push:" in text
    assert "\n  pull_request:" in text
    assert "\n  workflow_dispatch:" in text


    print(
        "CI eval workflow has push, pull_request and manual triggers: PASS"
    )


def test_workflow_pins_backend_python_contract(
) -> None:
    text = workflow_text()


    assert (
        'python-version: "3.9.13"'
        in text
    )


    print(
        "CI eval workflow mirrors the backend Python contract: PASS"
    )


def test_workflow_executes_all_eval_layers(
) -> None:
    text = workflow_text()


    required_commands = [
        "test_analysis_benchmark_v0_1.py",
        "test_eval_suite_runner_v0_1.py",
        "test_eval_coverage_v0_1.py",
        "test_eval_regression_gate_v0_1.py",
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
        "CI eval workflow executes benchmark, suite, coverage and gate: PASS"
    )


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


def test_workflow_failure_propagates_to_ci(
) -> None:
    text = workflow_text()


    # The suite and gate commands are deliberately run without
    # continue-on-error. Their existing non-zero exit codes must
    # therefore fail the GitHub Actions job.
    assert (
        "continue-on-error:"
        not in text
    )


    print(
        "Eval failures propagate to the GitHub Actions job: PASS"
    )


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


def main() -> None:
    print(
        "=== DATALENS CI EVALS GATE v0.1 ==="
    )

    print()


    test_workflow_is_versioned()

    test_workflow_uses_read_only_repository_permission()

    test_workflow_runs_on_push_pull_request_and_manual_dispatch()

    test_workflow_pins_backend_python_contract()

    test_workflow_executes_all_eval_layers()

    test_workflow_enforces_frozen_baseline()

    test_workflow_persists_machine_readable_evidence()

    test_workflow_failure_propagates_to_ci()

    test_workflow_scopes_execution_to_backend_changes()


    print()

    print(
        "CI Evals Gate v0.1: PASS"
    )


if __name__ == "__main__":
    main()
