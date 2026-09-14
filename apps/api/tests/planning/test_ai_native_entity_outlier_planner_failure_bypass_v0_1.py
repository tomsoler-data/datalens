from __future__ import annotations

from types import SimpleNamespace

import pytest

from fastapi import HTTPException

import app.api.analysis_run as analysis_run


def run_ai_first_planner_failure_case(
    monkeypatch,
    *,
    workflow_id: str,
    objective: str,
) -> None:
    """
    The AI-native route is AI-first.

    If the semantic planner fails:
    - no specialized entity-outlier fallback may run;
    - no native tool pipeline may run;
    - the endpoint must fail closed with HTTP 503.
    """

    events: list[str] = []

    source_dataset_records = [
        {
            "dataset_id":
                "dataset:test",
        }
    ]


    handoff = (
        SimpleNamespace(
            ingestion=
                SimpleNamespace(),

            dataset_records=
                source_dataset_records,
        )
    )


    analysis_datasets = [
        {
            "dataset_id":
                "derived:test:entity",
        }
    ]


    catalog = (
        SimpleNamespace()
    )


    def fake_load_validated_analysis_input_for_http(
        *,
        workflow_id: str,
    ):
        return (
            handoff
        )


    def fake_prepare_ai_planner_dataset_universe(
        *,
        source_dataset_records,
        objective,
    ):
        return (
            analysis_datasets,
            catalog,
        )


    def failing_ai_planner(
        *,
        objective,
        catalog,
        model,
    ):
        events.append(
            "planner"
        )

        raise RuntimeError(
            "synthetic AI planner failure"
        )


    def forbidden_compatibility_finding(
        **kwargs,
    ):
        events.append(
            "entity_outlier"
        )

        raise AssertionError(
            "Legacy entity-outlier compatibility finding "
            "must not run after an AI planner failure."
        )


    def forbidden_native_pipeline(
        **kwargs,
    ):
        events.append(
            "pipeline"
        )

        raise AssertionError(
            "Native pipeline must not execute after "
            "an AI planner failure."
        )


    monkeypatch.setattr(
        analysis_run,
        "load_validated_analysis_input_for_http",
        fake_load_validated_analysis_input_for_http,
    )


    monkeypatch.setattr(
        analysis_run,
        "prepare_ai_planner_dataset_universe",
        fake_prepare_ai_planner_dataset_universe,
    )


    monkeypatch.setattr(
        analysis_run,
        "plan_analyses_with_ai",
        failing_ai_planner,
    )


    monkeypatch.setattr(
        analysis_run,
        "build_entity_outlier_finding_if_requested",
        forbidden_compatibility_finding,
    )


    monkeypatch.setattr(
        analysis_run,
        "execute_native_ai_pipeline",
        forbidden_native_pipeline,
    )


    monkeypatch.setattr(
        analysis_run,
        "build_ai_trace",
        lambda **kwargs:
            SimpleNamespace(),
    )


    monkeypatch.setattr(
        analysis_run,
        "write_ai_trace",
        lambda trace:
            SimpleNamespace(
                enabled=False,
                written=False,
            ),
    )


    with pytest.raises(
        HTTPException
    ) as captured:
        analysis_run.run_ai_native_pipeline(
            request=None,

            dataset_files=None,

            workflow_id=
                workflow_id,

            objective=
                objective,

            planner_model=
                "qwen3.5:4b",

            tool_model=
                "qwen2.5:1.5b-instruct",

            approved_action_ids_json=None,

            semantic_decisions_json=None,

            approved_semantic_choices_json=None,
        )


    assert (
        captured.value.status_code
        ==
        503
    )


    assert (
        events
        ==
        [
            "planner",
        ]
    )


def test_entity_outlier_does_not_bypass_ai_planner_failure(
    monkeypatch,
):
    run_ai_first_planner_failure_case(
        monkeypatch,

        workflow_id=
            "prep:test-ai-first-entity-outlier",

        objective=
            "quels sont les clients atypiques ?",
    )


def test_mixed_request_does_not_bypass_ai_planner_failure(
    monkeypatch,
):
    run_ai_first_planner_failure_case(
        monkeypatch,

        workflow_id=
            "prep:test-ai-first-mixed-request",

        objective=(
            "quels sont les clients atypiques "
            "et quel est le chiffre d'affaires ?"
        ),
    )


def test_mixed_ranking_does_not_bypass_ai_planner_failure(
    monkeypatch,
):
    run_ai_first_planner_failure_case(
        monkeypatch,

        workflow_id=
            "prep:test-ai-first-mixed-ranking",

        objective=(
            "Identifie les 10 meilleurs clients et "
            "signale les éventuels clients atypiques."
        ),
    )
