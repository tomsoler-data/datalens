from __future__ import annotations

import os
import tempfile
from pathlib import Path


with tempfile.TemporaryDirectory() as directory:
    root = Path(
        directory
    )


    os.environ[
        "DATALENS_ANALYSIS_ARTIFACT_STORE_PATH"
    ] = str(
        root
        /
        "analysis_artifacts.json"
    )


    os.environ[
        "DATALENS_REPORT_SELECTION_STORE_PATH"
    ] = str(
        root
        /
        "report_selection.json"
    )


    from app.reporting.analysis_artifact_store import (
        get_analysis_artifact,
        register_server_owned_analysis,
    )

    from app.reporting.report_selection_store import (
        ReportSelectionNotExecutableError,
        add_analysis_to_report,
        get_report_selection,
    )


    workflow_id = "prep:test-lifecycle"

    analysis_id = (
        "analysis:report:test-request"
    )


    common = {
        "workflow_id":
            workflow_id,

        "analysis_id":
            analysis_id,

        "trace_id":
            "report:requested:test-request",

        "source_type":
            "document_request",

        "objective":
            "top produits",
    }


    print()
    print(
        "===== REQUEST ARTIFACT LIFECYCLE v0.1 ====="
    )
    print()


    # ========================================================
    # 1. UNRESOLVED
    # ========================================================

    unresolved = (
        register_server_owned_analysis(
            **common,

            executed=
                False,

            executed_count=
                0,

            pipeline_payload=
                {
                    "status":
                        "not_executed",
                },

            select_by_default=
                True,
        )
    )


    assert (
        unresolved.executed
        is False
    )


    selection = (
        get_report_selection(
            workflow_id=
                workflow_id
        )
    )


    assert (
        selection.selected_count
        ==
        0
    )


    print(
        "[PASS] unresolved artifact persisted "
        "without report selection"
    )


    # Manual selection must also fail.
    try:
        add_analysis_to_report(
            workflow_id=
                workflow_id,

            analysis_id=
                analysis_id,
        )

    except ReportSelectionNotExecutableError:
        pass

    else:
        raise AssertionError(
            "Non-executed artifact became selectable."
        )


    print(
        "[PASS] unresolved artifact cannot be "
        "selected manually"
    )


    # ========================================================
    # 2. FALSE -> TRUE
    # ========================================================

    resolved = (
        register_server_owned_analysis(
            **common,

            executed=
                True,

            executed_count=
                1,

            pipeline_payload=
                {
                    "status":
                        "ready",
                },

            select_by_default=
                True,
        )
    )


    assert (
        resolved.executed
        is True
    )


    selection = (
        get_report_selection(
            workflow_id=
                workflow_id
        )
    )


    assert (
        selection.selected_count
        ==
        1
    )


    assert (
        selection.analyses[
            0
        ].analysis_id
        ==
        analysis_id
    )


    print(
        "[PASS] False -> True transition "
        "selects newly executable request"
    )


    # ========================================================
    # 3. TRUE -> TRUE MUST PRESERVE USER DECISION
    # ========================================================

    from app.reporting.report_selection_store import (
        remove_analysis_from_report,
    )


    remove_analysis_from_report(
        workflow_id=
            workflow_id,

        analysis_id=
            analysis_id,
    )


    selection = (
        get_report_selection(
            workflow_id=
                workflow_id
        )
    )


    assert (
        selection.selected_count
        ==
        0
    )


    register_server_owned_analysis(
        **common,

        executed=
            True,

        executed_count=
            1,

        pipeline_payload=
            {
                "status":
                    "ready",
            },

        select_by_default=
            True,
    )


    selection = (
        get_report_selection(
            workflow_id=
                workflow_id
        )
    )


    assert (
        selection.selected_count
        ==
        0
    )


    print(
        "[PASS] True -> True preserves "
        "user deselection"
    )


    # ========================================================
    # 4. RESELECT THEN TRUE -> FALSE
    # ========================================================

    add_analysis_to_report(
        workflow_id=
            workflow_id,

        analysis_id=
            analysis_id,
    )


    assert (
        get_report_selection(
            workflow_id=
                workflow_id
        )
        .selected_count
        ==
        1
    )


    reverted = (
        register_server_owned_analysis(
            **common,

            executed=
                False,

            executed_count=
                0,

            pipeline_payload=
                {
                    "status":
                        "not_executed",
                },

            select_by_default=
                True,
        )
    )


    assert (
        reverted.executed
        is False
    )


    selection = (
        get_report_selection(
            workflow_id=
                workflow_id
        )
    )


    assert (
        selection.selected_count
        ==
        0
    )


    stored = (
        get_analysis_artifact(
            workflow_id=
                workflow_id,

            analysis_id=
                analysis_id,
        )
    )


    assert (
        stored.executed
        is False
    )


    print(
        "[PASS] True -> False removes stale "
        "report selection"
    )


    print()
    print(
        "PASS - request artifact lifecycle v0.1"
    )
