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
    add_analysis_to_report,
    delete_report_selection,
)


class FakePipelineReport:
    def __init__(
        self,
        *,
        trace_id: str,
        objective: str,
        family: str,
        chart_type: str,
        chart_data: list[
            dict
        ],
        metrics: dict,
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
            1
        )

        self._payload = {
            "status":
                "ready",

            "trace_id":
                trace_id,

            "planner_model":
                "gemma3:4b",

            "tool_model":
                "qwen2.5:1.5b-instruct",

            "planner":
                {
                    "objective":
                        objective,

                    "items":
                        [
                            {
                                "validation_status":
                                    "validated",

                                "contract":
                                    {
                                        "family":
                                            family,

                                        "bindings":
                                            [
                                                {
                                                    "role":
                                                        "group",

                                                    "column":
                                                        "category",
                                                },
                                                {
                                                    "role":
                                                        "value",

                                                    "column":
                                                        "unit_price",
                                                },
                                            ],

                                        "aggregation":
                                            None,

                                        "ranking":
                                            None,
                                    },
                            }
                        ],
                },

            "executed_count":
                1,

            "items":
                [
                    {
                        "pipeline_status":
                            "executed",

                        "native_tool":
                            {
                                "requested_tool":
                                    (
                                        "run_ranking"
                                        if family
                                        ==
                                        "ranking"
                                        else
                                        "run_group_comparison"
                                    ),

                                "validation_status":
                                    "validated",

                                "execution":
                                    {
                                        "execution_status":
                                            "executed",

                                        "result":
                                            {
                                                "title":
                                                    objective,

                                                "family":
                                                    family,

                                                "execution_status":
                                                    "complete",

                                                "chart_type":
                                                    chart_type,

                                                "summary":
                                                    [
                                                        "Résultat déterministe vérifié."
                                                    ],

                                                "metrics":
                                                    metrics,

                                                "chart_data":
                                                    chart_data,

                                                "warnings":
                                                    [],

                                                "limitations":
                                                    [],
                                            },
                                    },
                            },
                    }
                ],
        }


    def model_dump(
        self,
        *,
        mode: str,
    ):
        _ = mode

        return dict(
            self._payload
        )


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


def test_server_owned_pdf_uses_report_selection_only() -> None:
    workflow_id = (
        "workflow:pdf-v0-1"
    )


    initial = (
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
                        (
                            "Étudier la relation entre les "
                            "coûts unitaires et la catégorie."
                        ),

                    family=
                        "group_comparison",

                    chart_type=
                        "boxplot",

                    metrics=
                        {
                            "valid_observations":
                                36,

                            "group_count":
                                3,
                        },

                    chart_data=
                        [
                            {
                                "group":
                                    "Accessories",

                                "min":
                                    10.0,

                                "q1":
                                    12.0,

                                "median":
                                    15.0,

                                "q3":
                                    19.0,

                                "max":
                                    24.0,
                            },
                            {
                                "group":
                                    "Electronics",

                                "min":
                                    15.0,

                                "q1":
                                    25.0,

                                "median":
                                    80.0,

                                "q3":
                                    150.0,

                                "max":
                                    520.0,
                            },
                            {
                                "group":
                                    "Furniture",

                                "min":
                                    45.0,

                                "q1":
                                    90.0,

                                "median":
                                    130.0,

                                "q3":
                                    230.0,

                                "max":
                                    360.0,
                            },
                        ],
                ),
        )
    )


    assert (
        initial is not None
    )


    follow_up = (
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
                        (
                            "Analyser les deux catégories "
                            "ayant le prix catalogue moyen "
                            "le plus élevé."
                        ),

                    family=
                        "ranking",

                    chart_type=
                        "bar",

                    metrics=
                        {
                            "source_observation_count":
                                39,

                            "available_group_count":
                                3,

                            "result_count":
                                2,

                            "ranking_limit":
                                2,

                            "top_value":
                                273.13,
                        },

                    chart_data=
                        [
                            {
                                "category":
                                    "Furniture",

                                "value":
                                    273.13,

                                "rank":
                                    1,
                            },
                            {
                                "category":
                                    "Electronics",

                                "value":
                                    255.67,

                                "rank":
                                    2,
                            },
                        ],
                ),
        )
    )


    assert (
        follow_up is not None
    )


    # Initial is automatically selected. Add only this follow-up.
    add_analysis_to_report(
        workflow_id=
            workflow_id,

        analysis_id=
            follow_up.analysis_id,
    )


    client = (
        build_client()
    )


    response = (
        client.post(
            "/report/export-pdf",

            json={
                "workflow_id":
                    workflow_id,
            },
        )
    )


    assert (
        response.status_code
        ==
        200
    )


    assert (
        response.headers[
            "content-type"
        ]
        ==
        "application/pdf"
    )


    assert (
        response.headers[
            "x-datalens-report-selection-count"
        ]
        ==
        "2"
    )


    assert (
        response.content[
            :
            4
        ]
        ==
        b"%PDF"
    )


    assert (
        len(
            response.content
        )
        >
        3000
    )


def test_empty_selection_is_rejected() -> None:
    client = (
        build_client()
    )


    response = (
        client.post(
            "/report/export-pdf",

            json={
                "workflow_id":
                    "workflow:empty-pdf",
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
        "report_selection_empty"
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
            "=== DATALENS SERVER-OWNED PDF v0.1 ==="
        )


        test_server_owned_pdf_uses_report_selection_only()

        print(
            "[PASS] PDF generated from server selection"
        )

        print(
            "[PASS] initial + selected follow-up included"
        )

        print(
            "[PASS] browser sends workflow_id only"
        )


        test_empty_selection_is_rejected()

        print(
            "[PASS] empty report selection rejected"
        )


        print(
            "PASS - server-owned PDF v0.1"
        )


if __name__ == "__main__":
    main()
