from __future__ import annotations


import json
import tempfile
import unittest

from pathlib import Path
from unittest.mock import patch


from fastapi import FastAPI
from fastapi.testclient import TestClient


from app.api import analysis_run

from app.observability.ai_trace import (
    build_ai_trace,
)

from app.observability.trace_store import (
    get_ai_trace as store_get_ai_trace,
    get_ai_trace_metrics as store_get_ai_trace_metrics,
    get_latest_ai_trace as store_get_latest_ai_trace,
    list_ai_traces as store_list_ai_traces,
)


class AITraceHTTPV04Tests(
    unittest.TestCase
):
    def setUp(
        self,
    ) -> None:
        self.temporary_directory = (
            tempfile.TemporaryDirectory()
        )

        self.path = (
            Path(
                self.temporary_directory.name
            )
            /
            "ai-traces.jsonl"
        )


        completed = build_ai_trace(
            trace_id=
                "ai:http-completed-v04",

            workflow_id=
                "prep:http-completed-v04",

            objective=
                "Revenue by category",

            catalog={
                "datasets": [
                    {
                        "filename":
                            "orders.csv",
                    },
                ],
            },

            planner_report={
                "status":
                    "ready",

                "model":
                    "gemma3:4b",

                "items":
                    [],
            },

            pipeline_report={
                "analysis_id":
                    "analysis:ai:http-completed-v04",

                "analysis_source_type":
                    "initial_request",

                "status":
                    "ready",

                "tool_model":
                    "qwen2.5:1.5b-instruct",

                "executed_count":
                    1,

                "items":
                    [],
            },

            ingestion_ms=
                1.0,

            planner_ms=
                2.0,

            native_pipeline_ms=
                3.0,

            total_ms=
                6.0,
        )


        failed = build_ai_trace(
            trace_id=
                "ai:http-failed-v04",

            workflow_id=
                "prep:http-failed-v04",

            objective=
                "Revenue by category",

            catalog={
                "datasets": [],
            },

            planner_report=
                None,

            pipeline_report=
                None,

            ingestion_ms=
                1.0,

            planner_ms=
                2.0,

            native_pipeline_ms=
                0.0,

            total_ms=
                3.0,

            run_status=
                "failed",

            failure={
                "stage":
                    "planner",

                "error_type":
                    "RuntimeError",

                "message_safe":
                    (
                        "DataLens AI execution "
                        "failed before completion."
                    ),
            },
        )


        with self.path.open(
            "w",
            encoding="utf-8",
        ) as handle:
            handle.write(
                json.dumps(
                    completed.model_dump(
                        mode="json"
                    )
                )
                +
                "\n"
            )

            handle.write(
                json.dumps(
                    failed.model_dump(
                        mode="json"
                    )
                )
                +
                "\n"
            )


        app = FastAPI()

        app.include_router(
            analysis_run.router
        )

        self.client = TestClient(
            app
        )


    def tearDown(
        self,
    ) -> None:
        self.client.close()

        self.temporary_directory.cleanup()


    def _patch_store(
        self,
    ):
        return (
            patch.object(
                analysis_run,
                "list_ai_traces",
                side_effect=(
                    lambda *,
                    limit=20:
                        store_list_ai_traces(
                            limit=
                                limit,

                            path=
                                self.path,
                        )
                ),
            ),

            patch.object(
                analysis_run,
                "get_ai_trace_metrics",
                side_effect=(
                    lambda *,
                    limit=200:
                        store_get_ai_trace_metrics(
                            limit=
                                limit,

                            path=
                                self.path,
                        )
                ),
            ),

            patch.object(
                analysis_run,
                "get_latest_ai_trace",
                side_effect=(
                    lambda:
                        store_get_latest_ai_trace(
                            path=
                                self.path
                        )
                ),
            ),

            patch.object(
                analysis_run,
                "get_ai_trace",
                side_effect=(
                    lambda trace_id:
                        store_get_ai_trace(
                            trace_id,
                            path=
                                self.path,
                        )
                ),
            ),
        )


    def test_trace_list_exposes_v04_summary_fields(
        self,
    ) -> None:
        patches = self._patch_store()

        with (
            patches[0],
            patches[1],
            patches[2],
            patches[3],
        ):
            response = self.client.get(
                "/observability/traces"
            )


        self.assertEqual(
            response.status_code,
            200,
        )

        self.assertEqual(
            response.headers.get(
                "cache-control"
            ),
            "no-store",
        )


        payload = response.json()

        self.assertEqual(
            payload[
                "trace_count"
            ],
            2,
        )


        failed = payload[
            "traces"
        ][0]

        self.assertEqual(
            failed[
                "workflow_id"
            ],
            "prep:http-failed-v04",
        )

        self.assertEqual(
            failed[
                "run_status"
            ],
            "failed",
        )

        self.assertEqual(
            failed[
                "failure_stage"
            ],
            "planner",
        )


    def test_metrics_expose_v04_lifecycle_fields(
        self,
    ) -> None:
        patches = self._patch_store()

        with (
            patches[0],
            patches[1],
            patches[2],
            patches[3],
        ):
            response = self.client.get(
                "/observability/metrics"
            )


        self.assertEqual(
            response.status_code,
            200,
        )

        payload = response.json()

        self.assertEqual(
            payload[
                "completed_trace_count"
            ],
            1,
        )

        self.assertEqual(
            payload[
                "failed_trace_count"
            ],
            1,
        )

        self.assertEqual(
            payload[
                "failure_rate"
            ],
            0.5,
        )

        self.assertEqual(
            payload[
                "failure_stages"
            ],
            [
                {
                    "name":
                        "planner",

                    "count":
                        1,
                },
            ],
        )


    def test_latest_trace_exposes_full_failed_contract(
        self,
    ) -> None:
        patches = self._patch_store()

        with (
            patches[0],
            patches[1],
            patches[2],
            patches[3],
        ):
            response = self.client.get(
                "/observability/traces/latest"
            )


        self.assertEqual(
            response.status_code,
            200,
        )

        payload = response.json()

        self.assertEqual(
            payload[
                "trace_id"
            ],
            "ai:http-failed-v04",
        )

        self.assertEqual(
            payload[
                "workflow_id"
            ],
            "prep:http-failed-v04",
        )

        self.assertEqual(
            payload[
                "run_status"
            ],
            "failed",
        )

        self.assertEqual(
            payload[
                "failure"
            ][
                "stage"
            ],
            "planner",
        )

        self.assertEqual(
            payload[
                "failure"
            ][
                "error_type"
            ],
            "RuntimeError",
        )


    def test_trace_by_id_exposes_completed_correlation(
        self,
    ) -> None:
        patches = self._patch_store()

        with (
            patches[0],
            patches[1],
            patches[2],
            patches[3],
        ):
            response = self.client.get(
                (
                    "/observability/traces/"
                    "ai:http-completed-v04"
                )
            )


        self.assertEqual(
            response.status_code,
            200,
        )

        payload = response.json()

        self.assertEqual(
            payload[
                "workflow_id"
            ],
            "prep:http-completed-v04",
        )

        self.assertEqual(
            payload[
                "analysis_id"
            ],
            "analysis:ai:http-completed-v04",
        )

        self.assertEqual(
            payload[
                "analysis_source_type"
            ],
            "initial_request",
        )

        self.assertEqual(
            payload[
                "run_status"
            ],
            "completed",
        )

        self.assertIsNone(
            payload[
                "failure"
            ]
        )


if __name__ == "__main__":
    print(
        "=== DATALENS AI TRACE HTTP v0.4 ==="
    )

    unittest.main(
        verbosity=2
    )
