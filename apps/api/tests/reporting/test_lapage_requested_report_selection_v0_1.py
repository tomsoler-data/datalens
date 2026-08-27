from __future__ import annotations


import os


from pathlib import (
    Path,
)

from tempfile import (
    TemporaryDirectory,
)

from unittest.mock import (
    patch,
)


import app.reporting.analysis_artifact_store as analysis_artifact_store_module

import app.reporting.report_selection_store as report_selection_store_module


from app.reporting.unified_report_artifacts import (
    register_unified_report_artifacts,
)

from app.reporting.report_selection_store import (
    add_analysis_to_report,
    get_report_selection,
    get_report_selection_details,
)

from tests.reporting.test_lapage_requested_report_adapter_v0_1 import (
    build_lapage_requested_findings,
)


# ============================================================
# ASSERTIONS
# ============================================================

def assert_equal(
    actual,
    expected,
    message: str,
) -> None:
    if (
        actual
        !=
        expected
    ):
        raise AssertionError(
            (
                f"{message}\n"
                f"Expected: {expected!r}\n"
                f"Actual:   {actual!r}"
            )
        )


def assert_true(
    value,
    message: str,
) -> None:
    if not value:
        raise AssertionError(
            message
        )


def pass_test(
    message: str,
) -> None:
    print(
        f"[PASS] {message}"
    )


# ============================================================
# REPORT FIXTURE
# ============================================================

def build_report_payload():
    (
        _,
        _,
        findings,
    ) = build_lapage_requested_findings()


    return {
        "requested_findings": [
            finding.model_dump(
                mode="python",
            )

            for finding
            in findings
        ],

        "main_findings": [],

        "additional_findings": [],

        "context_analyses": [],
    }


# ============================================================
# ISOLATED STORE EXECUTION
# ============================================================

def build_isolated_selection():
    workflow_id = (
        "prep:test-lapage-request-selection-v0-1"
    )


    temporary_directory = (
        TemporaryDirectory()
    )


    root = Path(
        temporary_directory.name
    )


    artifact_store_path = (
        root
        /
        "analysis_artifacts.json"
    )


    selection_store_path = (
        root
        /
        "report_selection.json"
    )


    sqlite_path = (
        root
        /
        "datalens.sqlite3"
    )


    sqlite_environment_patch = (
        patch.dict(
            os.environ,
            {
                "DATALENS_SQLITE_PATH":
                    str(
                        sqlite_path
                    ),
            },
            clear=False,
        )
    )


    # Keep the environment patch alive for the entire
    # lifecycle returned by this fixture. Later tests call
    # add_analysis_to_report() after build_isolated_selection()
    # has returned, so SQLite isolation must remain active
    # until cleanup_isolated_selection().
    setattr(
        temporary_directory,
        "_datalens_sqlite_environment_patch",
        sqlite_environment_patch,
    )


    report = (
        build_report_payload()
    )


    artifact_path_patch = (
        patch.object(
            analysis_artifact_store_module,
            "resolve_analysis_artifact_store_path",
            return_value=
                artifact_store_path,
        )
    )


    selection_path_patch = (
        patch.object(
            report_selection_store_module,
            "resolve_report_selection_store_path",
            return_value=
                selection_store_path,
        )
    )


    sqlite_environment_patch.start()

    artifact_path_patch.start()

    selection_path_patch.start()


    try:
        registered = (
            register_unified_report_artifacts(
                workflow_id=
                    workflow_id,

                report=
                    report,
            )
        )


        selection = (
            get_report_selection(
                workflow_id=
                    workflow_id,
            )
        )


        details = (
            get_report_selection_details(
                workflow_id=
                    workflow_id,
            )
        )


        return (
            temporary_directory,
            artifact_path_patch,
            selection_path_patch,
            workflow_id,
            report,
            registered,
            selection,
            details,
            artifact_store_path,
            selection_store_path,
        )


    except Exception:
        selection_path_patch.stop()

        artifact_path_patch.stop()

        sqlite_environment_patch.stop()

        temporary_directory.cleanup()

        raise


def cleanup_isolated_selection(
    result,
) -> None:
    (
        temporary_directory,
        artifact_path_patch,
        selection_path_patch,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
    ) = result


    selection_path_patch.stop()

    artifact_path_patch.stop()


    sqlite_environment_patch = (
        getattr(
            temporary_directory,
            "_datalens_sqlite_environment_patch",
            None,
        )
    )


    if (
        sqlite_environment_patch
        is not None
    ):
        sqlite_environment_patch.stop()


    temporary_directory.cleanup()


# ============================================================
# TEST 1
# ============================================================

def test_eleven_requested_findings_are_registered(
) -> None:
    result = (
        build_isolated_selection()
    )


    try:
        (
            _,
            _,
            _,
            _,
            report,
            registered,
            _,
            _,
            _,
            _,
        ) = result


        assert_equal(
            len(
                report[
                    "requested_findings"
                ]
            ),
            11,
            (
                "Lapage report fixture must "
                "contain eleven requested findings."
            ),
        )


        assert_equal(
            len(
                registered
            ),
            11,
            (
                "Every requested finding should "
                "be registered as one server-owned "
                "analysis artifact."
            ),
        )


        registered_ids = [
            record.analysis_id

            for record
            in registered
        ]


        assert_equal(
            len(
                set(
                    registered_ids
                )
            ),
            11,
            (
                "Registered requested analyses "
                "must have eleven unique artifact IDs."
            ),
        )


        pass_test(
            (
                "11 requested findings become "
                "11 server-owned analysis artifacts"
            )
        )


    finally:
        cleanup_isolated_selection(
            result
        )


# ============================================================
# TEST 2
# ============================================================

def test_requested_artifacts_are_not_selected_by_default(
) -> None:
    result = (
        build_isolated_selection()
    )


    try:
        (
            _,
            _,
            _,
            _,
            _,
            registered,
            selection,
            details,
            _,
            _,
        ) = result


        assert_equal(
            len(
                registered
            ),
            11,
            (
                "All eleven requested findings "
                "must remain available as artifacts."
            ),
        )


        assert_equal(
            selection.selected_count,
            0,
            (
                "Registering analysis artifacts "
                "must not add them to the report "
                "automatically."
            ),
        )


        assert_equal(
            len(
                selection.analyses
            ),
            0,
            (
                "Fresh report selection must contain "
                "no automatically selected analyses."
            ),
        )


        assert_equal(
            details.selected_count,
            0,
            (
                "Selection details must also start "
                "with zero selected analyses."
            ),
        )


        assert_equal(
            len(
                details.analyses
            ),
            0,
            (
                "No analysis details should appear "
                "before an explicit report-selection "
                "action."
            ),
        )


        pass_test(
            (
                "11 artifacts are available but "
                "none is selected by default"
            )
        )


    finally:
        cleanup_isolated_selection(
            result
        )


# ============================================================
# TEST 3
# ============================================================

def test_explicit_document_request_selection_preserves_source_type(
) -> None:
    result = (
        build_isolated_selection()
    )


    try:
        (
            _,
            _,
            _,
            workflow_id,
            _,
            registered,
            _,
            _,
            _,
            _,
        ) = result


        target = (
            registered[
                0
            ]
        )


        selection = (
            add_analysis_to_report(
                workflow_id=
                    workflow_id,

                analysis_id=
                    target.analysis_id,
            )
        )


        assert_equal(
            selection.selected_count,
            1,
            (
                "One explicit add action should "
                "select exactly one analysis."
            ),
        )


        assert_equal(
            len(
                selection.analyses
            ),
            1,
            (
                "Selection should contain exactly "
                "the explicitly added analysis."
            ),
        )


        selected = (
            selection.analyses[
                0
            ]
        )


        assert_equal(
            selected.analysis_id,
            target.analysis_id,
            (
                "Explicit selection changed the "
                "requested analysis identity."
            ),
        )


        assert_equal(
            selected.source_type,
            "document_request",
            (
                "Explicitly selected requested "
                "analysis must preserve "
                "source_type=document_request."
            ),
        )


        pass_test(
            (
                "explicit report selection preserves "
                "source_type=document_request"
            )
        )


    finally:
        cleanup_isolated_selection(
            result
        )


# ============================================================
# TEST 4
# ============================================================

def test_selection_details_preserve_explicit_selection(
) -> None:
    result = (
        build_isolated_selection()
    )


    try:
        (
            _,
            _,
            _,
            workflow_id,
            _,
            registered,
            _,
            _,
            _,
            _,
        ) = result


        target_ids = [
            registered[
                0
            ].analysis_id,

            registered[
                1
            ].analysis_id,
        ]


        for analysis_id in target_ids:
            add_analysis_to_report(
                workflow_id=
                    workflow_id,

                analysis_id=
                    analysis_id,
            )


        selection = (
            get_report_selection(
                workflow_id=
                    workflow_id,
            )
        )


        details = (
            get_report_selection_details(
                workflow_id=
                    workflow_id,
            )
        )


        assert_equal(
            selection.selected_count,
            2,
            (
                "Exactly two explicitly added "
                "analyses should be selected."
            ),
        )


        assert_equal(
            details.selected_count,
            2,
            (
                "Detailed report selection should "
                "contain exactly the two explicitly "
                "selected analyses."
            ),
        )


        selection_ids = [
            item.analysis_id

            for item
            in selection.analyses
        ]


        detail_ids = [
            detail.selection.analysis_id

            for detail
            in details.analyses
        ]


        assert_equal(
            detail_ids,
            selection_ids,
            (
                "Detailed report selection changed "
                "the selected analysis ordering."
            ),
        )


        assert_equal(
            set(
                selection_ids
            ),
            set(
                target_ids
            ),
            (
                "Report selection contains analyses "
                "that were not explicitly selected."
            ),
        )


        pass_test(
            (
                "selection details contain only "
                "explicitly selected analyses"
            )
        )


    finally:
        cleanup_isolated_selection(
            result
        )


# ============================================================
# TEST 5
# ============================================================

def test_reporting_storage_is_isolated(
) -> None:
    result = (
        build_isolated_selection()
    )


    try:
        (
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            artifact_store_path,
            selection_store_path,
        ) = result


        sqlite_path = (
            artifact_store_path.parent
            /
            "datalens.sqlite3"
        )


        assert_true(
            sqlite_path.exists(),
            (
                "Isolated SQLite control-plane "
                "database was not created."
            ),
        )


        assert_equal(
            os.environ.get(
                "DATALENS_SQLITE_PATH"
            ),
            str(
                sqlite_path
            ),
            (
                "Reporting regression test is not "
                "using its temporary SQLite database."
            ),
        )


        assert_equal(
            artifact_store_path.parent,
            selection_store_path.parent,
            (
                "Analysis-artifact and report-selection "
                "scopes should share the isolated "
                "temporary reporting root."
            ),
        )


        assert_equal(
            sqlite_path.parent,
            artifact_store_path.parent,
            (
                "SQLite control plane and reporting "
                "data-plane scopes should share the "
                "same temporary test root."
            ),
        )


        assert_true(
            "var\reporting"
            not in
            str(
                artifact_store_path
            ),
            (
                "The regression test unexpectedly "
                "used the production analysis-artifact "
                "reporting scope."
            ),
        )


        assert_true(
            "var\reporting"
            not in
            str(
                selection_store_path
            ),
            (
                "The regression test unexpectedly "
                "used the production report-selection "
                "scope."
            ),
        )


        pass_test(
            (
                "regression test uses an isolated "
                "temporary SQLite control plane and "
                "reporting scopes"
            )
        )


    finally:
        cleanup_isolated_selection(
            result
        )


# ============================================================
# TEST 6
# ============================================================

def test_reregistration_preserves_explicit_selection_only(
) -> None:
    result = (
        build_isolated_selection()
    )


    try:
        (
            _,
            _,
            _,
            workflow_id,
            report,
            registered,
            initial_selection,
            _,
            _,
            _,
        ) = result


        assert_equal(
            initial_selection.selected_count,
            0,
            (
                "Initial registration must not "
                "auto-select requested analyses."
            ),
        )


        target = (
            registered[
                0
            ]
        )


        first_selection = (
            add_analysis_to_report(
                workflow_id=
                    workflow_id,

                analysis_id=
                    target.analysis_id,
            )
        )


        first_ids = [
            item.analysis_id

            for item
            in first_selection.analyses
        ]


        assert_equal(
            first_selection.selected_count,
            1,
            (
                "Explicit add should create one "
                "selected report analysis."
            ),
        )


        register_unified_report_artifacts(
            workflow_id=
                workflow_id,

            report=
                report,
        )


        second_selection = (
            get_report_selection(
                workflow_id=
                    workflow_id,
            )
        )


        second_ids = [
            item.analysis_id

            for item
            in second_selection.analyses
        ]


        assert_equal(
            second_selection.selected_count,
            1,
            (
                "Re-registering available artifacts "
                "must not auto-select additional "
                "analyses."
            ),
        )


        assert_equal(
            second_ids,
            first_ids,
            (
                "Re-registering artifacts changed "
                "the explicit report selection."
            ),
        )


        assert_equal(
            second_ids,
            [
                target.analysis_id
            ],
            (
                "Report selection must still contain "
                "only the explicitly selected "
                "analysis."
            ),
        )


        pass_test(
            (
                "re-registration preserves explicit "
                "selection without auto-selecting "
                "other artifacts"
            )
        )


    finally:
        cleanup_isolated_selection(
            result
        )


# ============================================================
# DIAGNOSTIC
# ============================================================

def print_selection_diagnostic(
) -> None:
    result = (
        build_isolated_selection()
    )


    try:
        (
            _,
            _,
            _,
            workflow_id,
            _,
            registered,
            selection,
            details,
            artifact_store_path,
            selection_store_path,
        ) = result


        print()

        print(
            "===== LAPAGE REQUESTED REPORT SELECTION ====="
        )

        print()


        print(
            f"workflow: {workflow_id}"
        )

        print(
            (
                "registered artifacts: "
                f"{len(registered)}"
            )
        )

        print(
            (
                "selected count: "
                f"{selection.selected_count}"
            )
        )

        print(
            (
                "detail count: "
                f"{details.selected_count}"
            )
        )

        print(
            (
                "selection revision: "
                f"{selection.revision}"
            )
        )


        print()

        print(
            "===== SELECTED ANALYSES ====="
        )

        print()


        for (
            index,
            item,
        ) in enumerate(
            selection.analyses,
            start=1,
        ):
            print(
                (
                    f"{index:02d}. "
                    f"[{item.source_type}] "
                    f"{item.analysis_id}"
                )
            )

            print(
                (
                    f"    objective: "
                    f"{item.objective}"
                )
            )


        print()

        print(
            "===== ISOLATED STORES ====="
        )

        print()


        print(
            (
                "artifact store: "
                f"{artifact_store_path}"
            )
        )

        print(
            (
                "selection store: "
                f"{selection_store_path}"
            )
        )


    finally:
        cleanup_isolated_selection(
            result
        )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print(
        (
            "=== DATALENS LAPAGE REQUESTED "
            "REPORT SELECTION v0.1 ==="
        )
    )

    print()


    test_eleven_requested_findings_are_registered()

    test_requested_artifacts_are_not_selected_by_default()

    test_explicit_document_request_selection_preserves_source_type()

    test_selection_details_preserve_explicit_selection()

    test_reporting_storage_is_isolated()

    test_reregistration_preserves_explicit_selection_only()


    print_selection_diagnostic()


    print()

    print(
        (
            "PASS - Lapage requested "
            "report selection v0.1"
        )
    )


if (
    __name__
    ==
    "__main__"
):
    main()