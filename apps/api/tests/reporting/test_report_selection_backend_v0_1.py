from __future__ import annotations


import os

from pathlib import (
    Path,
)

from tempfile import (
    TemporaryDirectory,
)

from types import (
    SimpleNamespace,
)


from fastapi import (
    FastAPI,
)

from fastapi.testclient import (
    TestClient,
)


from app.api.report_selection import (
    router as report_selection_router,
)

from app.reporting.analysis_artifact_store import (
    delete_analysis_artifacts,
    register_native_pipeline_result,
)

from app.reporting.report_selection_store import (
    delete_report_selection,
    get_report_selection,
)


class FakePipelineReport:
    def __init__(
        self,
        *,
        trace_id: str,
        objective: str,
        executed_count: int,
    ) -> None:
        self.trace_id = (
            trace_id
        )

        self.planner = (
            SimpleNamespace(
                objective=
                    objective
            )
        )

        self.executed_count = (
            executed_count
        )


    def model_dump(
        self,
        *,
        mode: str,
    ):
        _ = mode

        return {
            "status":
                "ready",

            "trace_id":
                self.trace_id,

            "planner":
                {
                    "objective":
                        self.planner.objective,
                },

            "executed_count":
                self.executed_count,

            "items":
                [],
        }


def datasets(
    workflow_id: str,
):
    return [
        {
            "dataset_id":
                "dataset:final",

            "filename":
                "final.csv",

            "preparation_workflow_id":
                workflow_id,
        }
    ]


def build_client() -> TestClient:
    app = FastAPI()

    app.include_router(
        report_selection_router
    )

    return (
        TestClient(
            app
        )
    )


def test_server_owned_registration_and_selection() -> None:
    workflow_id = (
        "workflow:test-report-selection"
    )


    first = (
        register_native_pipeline_result(
            datasets=
                datasets(
                    workflow_id
                ),

            pipeline_report=
                FakePipelineReport(
                    trace_id=
                        "trace-initial",

                    objective=
                        "Comparer le coût par catégorie.",

                    executed_count=
                        1,
                ),
        )
    )


    assert (
        first is not None
    )

    assert (
        first.analysis_id
        ==
        "analysis:trace-initial"
    )

    assert (
        first.source_type
        ==
        "initial_request"
    )

    assert (
        first.executed
    )


    # Execution and report composition are separate concerns.
    # Registering an executable analysis must not select it.
    state = (
        get_report_selection(
            workflow_id=
                workflow_id
        )
    )


    assert (
        state.selected_count
        ==
        0
    )

    assert (
        state.analyses
        ==
        []
    )


    second = (
        register_native_pipeline_result(
            datasets=
                datasets(
                    workflow_id
                ),

            pipeline_report=
                FakePipelineReport(
                    trace_id=
                        "trace-follow-up",

                    objective=
                        "Classer les catégories par prix moyen.",

                    executed_count=
                        1,
                ),
        )
    )


    assert (
        second is not None
    )

    assert (
        second.source_type
        ==
        "follow_up_prompt"
    )


    # Follow-up execution must also remain outside report
    # composition until the user explicitly selects it.
    state = (
        get_report_selection(
            workflow_id=
                workflow_id
        )
    )


    assert (
        state.selected_count
        ==
        0
    )

    assert (
        state.analyses
        ==
        []
    )


    client = (
        build_client()
    )


    # --------------------------------------------------------
    # Explicitly select the first analysis.
    # --------------------------------------------------------

    first_add_response = (
        client.post(
            "/report/selection/add",

            json={
                "workflow_id":
                    workflow_id,

                "analysis_id":
                    first.analysis_id,
            },
        )
    )


    assert (
        first_add_response.status_code
        ==
        200
    )


    first_added = (
        first_add_response.json()
    )


    assert (
        first_added[
            "selected_count"
        ]
        ==
        1
    )

    assert (
        first_added[
            "analyses"
        ][
            0
        ][
            "analysis_id"
        ]
        ==
        first.analysis_id
    )


    # --------------------------------------------------------
    # Explicitly select the follow-up analysis.
    # --------------------------------------------------------

    add_response = (
        client.post(
            "/report/selection/add",

            json={
                "workflow_id":
                    workflow_id,

                "analysis_id":
                    second.analysis_id,
            },
        )
    )


    assert (
        add_response.status_code
        ==
        200
    )


    added = (
        add_response.json()
    )


    assert (
        added[
            "selected_count"
        ]
        ==
        2
    )


    selected_ids = {
        item[
            "analysis_id"
        ]

        for item
        in added[
            "analyses"
        ]
    }


    assert (
        selected_ids
        ==
        {
            first.analysis_id,
            second.analysis_id,
        }
    )


    # --------------------------------------------------------
    # Repeated explicit add remains idempotent.
    # --------------------------------------------------------

    repeated_add_response = (
        client.post(
            "/report/selection/add",

            json={
                "workflow_id":
                    workflow_id,

                "analysis_id":
                    second.analysis_id,
            },
        )
    )


    assert (
        repeated_add_response.status_code
        ==
        200
    )


    repeated_added = (
        repeated_add_response.json()
    )


    assert (
        repeated_added[
            "selected_count"
        ]
        ==
        2
    )


    # --------------------------------------------------------
    # Server-owned report ordering.
    # --------------------------------------------------------

    reorder_response = (
        client.post(
            "/report/selection/reorder",

            json={
                "workflow_id":
                    workflow_id,

                "analysis_ids":
                    [
                        second.analysis_id,
                        first.analysis_id,
                    ],
            },
        )
    )


    assert (
        reorder_response.status_code
        ==
        200
    )


    reordered = (
        reorder_response.json()
    )


    assert (
        reordered[
            "analyses"
        ][
            0
        ][
            "analysis_id"
        ]
        ==
        second.analysis_id
    )

    assert (
        reordered[
            "analyses"
        ][
            1
        ][
            "analysis_id"
        ]
        ==
        first.analysis_id
    )


    # --------------------------------------------------------
    # Removing from the report must not delete the artifact.
    # --------------------------------------------------------

    remove_response = (
        client.post(
            "/report/selection/remove",

            json={
                "workflow_id":
                    workflow_id,

                "analysis_id":
                    first.analysis_id,
            },
        )
    )


    assert (
        remove_response.status_code
        ==
        200
    )


    removed = (
        remove_response.json()
    )


    assert (
        removed[
            "selected_count"
        ]
        ==
        1
    )

    assert (
        removed[
            "analyses"
        ][
            0
        ][
            "analysis_id"
        ]
        ==
        second.analysis_id
    )


    details_response = (
        client.get(
            "/report/selection/details",

            params={
                "workflow_id":
                    workflow_id,
            },
        )
    )


    assert (
        details_response.status_code
        ==
        200
    )


    details = (
        details_response.json()
    )


    assert (
        details[
            "selected_count"
        ]
        ==
        1
    )

    assert (
        details[
            "analyses"
        ][
            0
        ][
            "pipeline_payload"
        ][
            "planner"
        ][
            "objective"
        ]
        ==
        second.objective
    )


def test_rejected_analysis_cannot_be_selected() -> None:
    workflow_id = (
        "workflow:test-rejected-analysis"
    )


    rejected = (
        register_native_pipeline_result(
            datasets=
                datasets(
                    workflow_id
                ),

            pipeline_report=
                FakePipelineReport(
                    trace_id=
                        "trace-rejected",

                    objective=
                        "Question non exécutée",

                    executed_count=
                        0,
                ),
        )
    )


    assert (
        rejected is not None
    )

    assert not (
        rejected.executed
    )


    client = (
        build_client()
    )


    response = (
        client.post(
            "/report/selection/add",

            json={
                "workflow_id":
                    workflow_id,

                "analysis_id":
                    rejected.analysis_id,
            },
        )
    )


    assert (
        response.status_code
        ==
        409
    )


    assert (
        response.json()[
            "detail"
        ][
            "error"
        ]
        ==
        "analysis_not_executable_for_report"
    )


def test_unknown_browser_analysis_id_is_rejected() -> None:
    client = (
        build_client()
    )


    response = (
        client.post(
            "/report/selection/add",

            json={
                "workflow_id":
                    "workflow:test",

                "analysis_id":
                    "analysis:invented-by-browser",
            },
        )
    )


    assert (
        response.status_code
        ==
        404
    )


def main() -> None:
    with TemporaryDirectory() as temporary:
        root = Path(
            temporary
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


        delete_analysis_artifacts()

        delete_report_selection()


        print(
            "=== DATALENS REPORT SELECTION BACKEND v0.1 ==="
        )


        test_server_owned_registration_and_selection()

        print(
            "[PASS] server-owned analysis registration"
        )

        print(
            "[PASS] initial execution remains manually unselected"
        )

        print(
            "[PASS] follow-up execution remains manually unselected"
        )

        print(
            "[PASS] repeated add remains idempotent"
        )

        print(
            "[PASS] report order is server-owned"
        )

        print(
            "[PASS] remove keeps analysis artifact available"
        )


        test_rejected_analysis_cannot_be_selected()

        print(
            "[PASS] rejected analysis cannot enter report"
        )


        test_unknown_browser_analysis_id_is_rejected()

        print(
            "[PASS] browser cannot invent an analysis result"
        )


        print(
            "PASS - report selection backend v0.1"
        )


if __name__ == "__main__":
    main()
