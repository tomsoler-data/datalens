from __future__ import annotations

from types import SimpleNamespace

import app.api.analysis_run as analysis_run


WORKFLOW_ID = (
    "prep:test-entity-outlier-planner-failure"
)

OBJECTIVE = (
    "quels sont les clients atypiques ?"
)


def test_entity_outlier_survives_general_planner_failure(
    monkeypatch,
):
    events: list[str] = []

    source_dataset_records = [
        {
            "dataset_id":
                "dataset:test",
        }
    ]

    handoff = SimpleNamespace(
        ingestion=
            SimpleNamespace(),

        dataset_records=
            source_dataset_records,
    )

    analysis_datasets = [
        {
            "dataset_id":
                "derived:test:customer",
        }
    ]

    catalog = (
        SimpleNamespace()
    )

    finding_payload = {
        "status":
            "ready",

        "kind":
            "customer_entity_outlier_detection",

        "dataset_id":
            "derived:test:customer",

        "profiles":
            [],
    }

    finding = SimpleNamespace(
        status=
            "ready",

        kind=
            "customer_entity_outlier_detection",

        dataset_id=
            "derived:test:customer",

        model_dump=lambda **kwargs:
            dict(
                finding_payload
            ),
    )

    expected_pipeline_report = (
        SimpleNamespace(
            notes=
                [],

            entity_outlier_finding=
                finding_payload,
        )
    )


    def fake_load_validated_analysis_input_for_http(
        *,
        workflow_id,
    ):
        assert (
            workflow_id
            ==
            WORKFLOW_ID
        )

        return handoff


    def fake_prepare_ai_planner_dataset_universe(
        *,
        source_dataset_records,
        objective,
    ):
        assert (
            objective
            ==
            OBJECTIVE
        )

        return (
            analysis_datasets,
            catalog,
        )


    def fake_build_entity_outlier_finding_if_requested(
        *,
        objective,
        source_dataset_records,
    ):
        events.append(
            "entity_outlier"
        )

        assert (
            objective
            ==
            OBJECTIVE
        )

        return finding


    def fake_plan_analyses_with_intent_routing(
        *,
        objective,
        catalog,
        model,
    ):
        events.append(
            "planner"
        )

        raise RuntimeError(
            "synthetic planner failure"
        )


    def fake_execute_native_ai_pipeline(
        *,
        planner_report,
        datasets,
        tool_model,
        trace_id,
        entity_outlier_finding,
    ):
        events.append(
            "pipeline"
        )

        assert (
            planner_report
            is not None
        )

        assert (
            datasets
            ==
            analysis_datasets
        )

        assert (
            entity_outlier_finding
            ==
            finding_payload
        )

        return (
            expected_pipeline_report
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
        "build_entity_outlier_finding_if_requested",
        fake_build_entity_outlier_finding_if_requested,
    )

    monkeypatch.setattr(
        analysis_run,
        "plan_analyses_with_intent_routing",
        fake_plan_analyses_with_intent_routing,
    )

    monkeypatch.setattr(
        analysis_run,
        "require_objective_coverage",
        lambda **kwargs:
            None,
    )

    monkeypatch.setattr(
        analysis_run,
        "execute_native_ai_pipeline",
        fake_execute_native_ai_pipeline,
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


    result = (
        analysis_run
        .run_ai_native_pipeline(
            request=None,

            dataset_files=None,

            workflow_id=
                WORKFLOW_ID,

            objective=
                OBJECTIVE,

            planner_model=
                "qwen3.5:4b",

            tool_model=
                "qwen2.5:1.5b-instruct",

            approved_action_ids_json=None,

            semantic_decisions_json=None,

            approved_semantic_choices_json=None,
        )
    )


    assert (
        result
        is
        expected_pipeline_report
    )

    assert (
        events
        ==
        [
            "entity_outlier",
            "planner",
            "pipeline",
        ]
    )


    assert (
        result.entity_outlier_finding
        ==
        finding_payload
    )



def test_entity_outlier_does_not_hide_mixed_request_planner_failure(
    monkeypatch,
):
    import pytest

    from fastapi import (
        HTTPException,
    )


    workflow_id = (
        "prep:test-entity-outlier-mixed-request"
    )

    objective = (
        "quels sont les clients atypiques "
        "et quel est le chiffre d'affaires ?"
    )


    handoff = SimpleNamespace(
        ingestion=
            SimpleNamespace(),

        dataset_records=[
            {
                "dataset_id":
                    "dataset:test",
            }
        ],
    )


    finding = SimpleNamespace(
        status=
            "ready",

        kind=
            "customer_entity_outlier_detection",

        dataset_id=
            "derived:test:customer",

        model_dump=lambda **kwargs:
            {
                "status":
                    "ready",

                "kind":
                    "customer_entity_outlier_detection",

                "dataset_id":
                    "derived:test:customer",
            },
    )


    monkeypatch.setattr(
        analysis_run,
        "load_validated_analysis_input_for_http",
        lambda **kwargs:
            handoff,
    )


    monkeypatch.setattr(
        analysis_run,
        "prepare_ai_planner_dataset_universe",
        lambda **kwargs:
            (
                [
                    {
                        "dataset_id":
                            "derived:test:customer",
                    }
                ],
                SimpleNamespace(),
            ),
    )


    monkeypatch.setattr(
        analysis_run,
        "build_entity_outlier_finding_if_requested",
        lambda **kwargs:
            finding,
    )


    # Simulate one additional explicit analytical requirement.
    # This makes the specialized-only fallback unsafe.
    monkeypatch.setattr(
        analysis_run,
        "extract_objective_requirements",
        lambda **kwargs:
            [
                SimpleNamespace(
                    concept=
                        "revenue_total"
                )
            ],
    )


    monkeypatch.setattr(
        analysis_run,
        "plan_analyses_with_intent_routing",
        lambda **kwargs:
            (
                (_ for _ in ())
                .throw(
                    RuntimeError(
                        "synthetic planner failure"
                    )
                )
            ),
    )


    pipeline_called = {
        "value":
            False,
    }


    def fail_if_pipeline_runs(
        **kwargs,
    ):
        pipeline_called[
            "value"
        ] = True

        raise AssertionError(
            "Native pipeline must not execute "
            "after a mixed-request planner failure."
        )


    monkeypatch.setattr(
        analysis_run,
        "execute_native_ai_pipeline",
        fail_if_pipeline_runs,
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
        pipeline_called[
            "value"
        ]
        is False
    )



def test_entity_outlier_does_not_hide_mixed_ranking_planner_failure(
    monkeypatch,
):
    import pytest

    from fastapi import (
        HTTPException,
    )


    workflow_id = (
        "prep:test-entity-outlier-mixed-ranking"
    )

    objective = (
        "Identifie les 10 meilleurs clients et "
        "signale les ?ventuels clients atypiques."
    )


    handoff = SimpleNamespace(
        ingestion=
            SimpleNamespace(),

        dataset_records=[
            {
                "dataset_id":
                    "dataset:test",
            }
        ],
    )


    finding = SimpleNamespace(
        status=
            "ready",

        kind=
            "customer_entity_outlier_detection",

        dataset_id=
            "derived:test:customer",

        model_dump=lambda **kwargs:
            {
                "status":
                    "ready",

                "kind":
                    "customer_entity_outlier_detection",

                "dataset_id":
                    "derived:test:customer",
            },
    )


    monkeypatch.setattr(
        analysis_run,
        "load_validated_analysis_input_for_http",
        lambda **kwargs:
            handoff,
    )


    monkeypatch.setattr(
        analysis_run,
        "prepare_ai_planner_dataset_universe",
        lambda **kwargs:
            (
                [
                    {
                        "dataset_id":
                            "derived:test:customer",
                    }
                ],
                SimpleNamespace(),
            ),
    )


    monkeypatch.setattr(
        analysis_run,
        "build_entity_outlier_finding_if_requested",
        lambda **kwargs:
            finding,
    )


    monkeypatch.setattr(
        analysis_run,
        "plan_analyses_with_intent_routing",
        lambda **kwargs:
            (
                (_ for _ in ())
                .throw(
                    RuntimeError(
                        "synthetic planner failure"
                    )
                )
            ),
    )


    pipeline_called = {
        "value":
            False,
    }


    def fail_if_pipeline_runs(
        **kwargs,
    ):
        pipeline_called[
            "value"
        ] = True

        raise AssertionError(
            "Native pipeline must not execute "
            "after a mixed ranking + entity-outlier "
            "planner failure."
        )


    monkeypatch.setattr(
        analysis_run,
        "execute_native_ai_pipeline",
        fail_if_pipeline_runs,
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
        pipeline_called[
            "value"
        ]
        is False
    )
