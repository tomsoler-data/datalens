from __future__ import annotations


from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


from app.reporting import analysis_artifact_store as artifact_store
from app.reporting import unified_report_artifacts as unified_artifacts


def _assert(
    condition: bool,
    message: str,
) -> None:
    if not condition:
        raise AssertionError(
            message
        )


def _fake_native_pipeline(
    *,
    trace_id: str,
    objective: str,
    executed_count: int = 1,
):
    planner = SimpleNamespace(
        objective=
            objective,
    )

    return SimpleNamespace(
        trace_id=
            trace_id,

        planner=
            planner,

        executed_count=
            executed_count,

        model_dump=lambda mode="json": {
            "trace_id":
                trace_id,

            "planner":
                {
                    "objective":
                        objective,
                },

            "executed_count":
                executed_count,
        },
    )


def test_prompt_native_execution_does_not_auto_select() -> None:
    datasets = [
        {
            "preparation_workflow_id":
                "prep:manual-report-selection",
        }
    ]


    first_pipeline = (
        _fake_native_pipeline(
            trace_id=
                "ai:first",

            objective=
                "CA par catégorie",
        )
    )


    second_pipeline = (
        _fake_native_pipeline(
            trace_id=
                "ai:second",

            objective=
                "Top 3 marques par chiffre d'affaires",
        )
    )


    initial_metadata = []


    follow_up_metadata = [
        {
            "analysis_id":
                "analysis:ai:first",

            "workflow_id":
                "prep:manual-report-selection",

            "source_type":
                "initial_request",
        }
    ]


    with (
        patch.object(
            artifact_store,
            "_ensure_store_initialized",
            return_value=
                None,
        ),
        patch.object(
            artifact_store,
            "resolve_analysis_artifact_store_path",
            return_value=
                Path("ignored.json"),
        ),
        patch.object(
            artifact_store,
            "get_analysis_artifact_index_entry",
            return_value=
                None,
        ),
        patch.object(
            artifact_store,
            "_persist_record",
            side_effect=lambda record: record,
        ),
        patch(
            "app.reporting.report_selection_store.add_analysis_to_report"
        ) as add_to_report,
    ):
        with patch.object(
            artifact_store,
            "load_analysis_artifact_index_scope",
            return_value=
                initial_metadata,
        ):
            initial = (
                artifact_store
                .register_native_pipeline_result(
                    datasets=
                        datasets,

                    pipeline_report=
                        first_pipeline,
                )
            )


        with patch.object(
            artifact_store,
            "load_analysis_artifact_index_scope",
            return_value=
                follow_up_metadata,
        ):
            follow_up = (
                artifact_store
                .register_native_pipeline_result(
                    datasets=
                        datasets,

                    pipeline_report=
                        second_pipeline,
                )
            )


    _assert(
        initial is not None
        and
        initial.source_type
        ==
        "initial_request",
        (
            "The first prompt must still be classified as "
            "initial_request."
        ),
    )


    _assert(
        follow_up is not None
        and
        follow_up.source_type
        ==
        "follow_up_prompt",
        (
            "A later prompt must still be classified as "
            "follow_up_prompt."
        ),
    )


    _assert(
        add_to_report.call_count
        ==
        0,
        (
            "Prompt-native execution must not add initial or "
            "follow-up analyses to the report automatically."
        ),
    )


def test_server_owned_sources_ignore_legacy_auto_select_flag() -> None:
    source_types = [
        "initial_request",
        "follow_up_prompt",
        "document_request",
        "automatic",
    ]


    with (
        patch.object(
            artifact_store,
            "_ensure_store_initialized",
            return_value=
                None,
        ),
        patch.object(
            artifact_store,
            "resolve_analysis_artifact_store_path",
            return_value=
                Path("ignored.json"),
        ),
        patch.object(
            artifact_store,
            "get_analysis_artifact_index_entry",
            return_value=
                None,
        ),
        patch.object(
            artifact_store,
            "_persist_record",
            side_effect=lambda record: record,
        ),
        patch(
            "app.reporting.report_selection_store.add_analysis_to_report"
        ) as add_to_report,
        patch(
            "app.reporting.report_selection_store.remove_analysis_from_report"
        ) as remove_from_report,
    ):
        for index, source_type in enumerate(
            source_types,
            start=1,
        ):
            record = (
                artifact_store
                .register_server_owned_analysis(
                    workflow_id=
                        "prep:manual-report-selection",

                    analysis_id=
                        f"analysis:server:{index}",

                    trace_id=
                        f"trace:{index}",

                    source_type=
                        source_type,

                    objective=
                        f"Objective {index}",

                    executed=
                        True,

                    executed_count=
                        1,

                    pipeline_payload=
                        {},

                    # Older call sites may still pass True.
                    # The v0.3 policy must ignore it.
                    select_by_default=
                        True,
                )
            )


            _assert(
                record.executed,
                (
                    "The artifact must remain executable and "
                    "available."
                ),
            )


    _assert(
        add_to_report.call_count
        ==
        0,
        (
            "No source type may mutate report selection merely "
            "because an analysis was registered or executed."
        ),
    )


    _assert(
        remove_from_report.call_count
        ==
        0,
        (
            "Executed artifacts must not be removed from an "
            "existing manual selection."
        ),
    )


def test_non_executable_artifact_is_removed_from_selection() -> None:
    with (
        patch.object(
            artifact_store,
            "_ensure_store_initialized",
            return_value=
                None,
        ),
        patch.object(
            artifact_store,
            "resolve_analysis_artifact_store_path",
            return_value=
                Path("ignored.json"),
        ),
        patch.object(
            artifact_store,
            "get_analysis_artifact_index_entry",
            return_value=
                None,
        ),
        patch.object(
            artifact_store,
            "_persist_record",
            side_effect=lambda record: record,
        ),
        patch(
            "app.reporting.report_selection_store.remove_analysis_from_report"
        ) as remove_from_report,
    ):
        artifact_store.register_server_owned_analysis(
            workflow_id=
                "prep:manual-report-selection",

            analysis_id=
                "analysis:blocked",

            trace_id=
                "trace:blocked",

            source_type=
                "document_request",

            objective=
                "Blocked request",

            executed=
                False,

            executed_count=
                0,

            pipeline_payload=
                {},

            select_by_default=
                True,
        )


    _assert(
        remove_from_report.call_count
        ==
        1,
        (
            "A non-executable artifact must still be removed "
            "from report selection to preserve integrity."
        ),
    )


def test_unified_artifacts_never_request_default_selection() -> None:
    captured: list[
        tuple[
            str,
            bool,
        ]
    ] = []


    def fake_register_finding(
        *,
        workflow_id,
        source_type,
        finding,
        select_by_default,
        requested_plan=None,
    ):
        captured.append(
            (
                source_type,
                select_by_default,
            )
        )

        return SimpleNamespace(
            workflow_id=
                workflow_id,

            source_type=
                source_type,

            finding=
                finding,

            requested_plan=
                requested_plan,
        )


    report = {
        "requested_findings":
            [
                {
                    "analysis_id":
                        "requested:1",

                    "title":
                        "CA par catégorie",
                }
            ],

        "main_findings":
            [
                {
                    "analysis_id":
                        "automatic:1",

                    "title":
                        "Distribution",
                }
            ],

        "additional_findings":
            [],

        "context_analyses":
            [],
    }


    with patch.object(
        unified_artifacts,
        "_register_finding",
        side_effect=
            fake_register_finding,
    ):
        registered = (
            unified_artifacts
            .register_unified_report_artifacts(
                workflow_id=
                    "prep:manual-report-selection",

                report=
                    report,
            )
        )


    _assert(
        len(
            registered
        )
        ==
        2,
        (
            "Both document-request and automatic findings must "
            "still be persisted as available analyses."
        ),
    )


    _assert(
        captured
        ==
        [
            (
                "document_request",
                False,
            ),
            (
                "automatic",
                False,
            ),
        ],
        (
            "Unified report artifacts must never request "
            "automatic report selection."
        ),
    )


def test_requested_promotion_ignores_legacy_true_flag() -> None:
    finding = {
        "analysis_id":
            "requested:promotion",

        "title":
            "Requested finding",

        "request_text":
            "Requested finding",
    }


    with patch.object(
        unified_artifacts,
        "_register_finding",
    ) as register_finding:
        register_finding.return_value = (
            SimpleNamespace(
                analysis_id=
                    "analysis:dummy",
            )
        )


        unified_artifacts.register_requested_report_finding(
            workflow_id=
                "prep:manual-report-selection",

            finding=
                finding,

            # requested_resolution.py may still pass True.
            # The public registration helper must neutralize it.
            select_by_default=
                True,
        )


    _assert(
        register_finding.call_args
        is not None,
        (
            "The requested finding must still be forwarded for "
            "artifact registration."
        ),
    )


    _assert(
        register_finding.call_args.kwargs[
            "select_by_default"
        ]
        is False,
        (
            "A document-request promotion must remain available "
            "without becoming selected automatically."
        ),
    )


def test_rule_versions() -> None:
    _assert(
        artifact_store
        .ANALYSIS_ARTIFACT_STORE_RULE_VERSION
        ==
        "analysis_artifact_store_v0.3",
        "Unexpected analysis artifact store rule version.",
    )


    _assert(
        unified_artifacts
        .UNIFIED_REPORT_ARTIFACT_RULE_VERSION
        ==
        "unified_report_artifacts_v0.2",
        "Unexpected unified report artifact rule version.",
    )


def main() -> None:
    print(
        "=== DATALENS MANUAL REPORT SELECTION v0.1 ==="
    )


    test_prompt_native_execution_does_not_auto_select()

    print(
        "[PASS] prompt-native analyses remain available without auto-selection"
    )


    test_server_owned_sources_ignore_legacy_auto_select_flag()

    print(
        "[PASS] all server-owned source types ignore legacy default-selection requests"
    )


    test_non_executable_artifact_is_removed_from_selection()

    print(
        "[PASS] non-executable artifacts are still removed from report selection"
    )


    test_unified_artifacts_never_request_default_selection()

    print(
        "[PASS] deterministic/document artifacts are available but not auto-selected"
    )


    test_requested_promotion_ignores_legacy_true_flag()

    print(
        "[PASS] requested-analysis promotion cannot auto-select through a legacy True flag"
    )


    test_rule_versions()

    print(
        "[PASS] manual report-selection rule versions"
    )


    print()
    print(
        "PASS - manual report selection v0.1"
    )


if __name__ == "__main__":
    main()
