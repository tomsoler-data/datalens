from __future__ import annotations

import json
import tempfile
import unittest

from pathlib import Path
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api import (
    analysis_run,
)

from app.observability.ai_trace import (
    AITraceRecord,
    build_ai_trace,
)

from app.observability.runtime_trace import (
    RuntimeTraceRecord,
)

from app.observability.trace_explorer import (
    TRACE_EXPLORER_RULE_VERSION,
    get_request_trace_explorer,
    normalize_runtime_request_id,
)


REQUEST_ID = (
    "http:"
    + (
        "1"
        * 32
    )
)

OTHER_REQUEST_ID = (
    "http:"
    + (
        "2"
        * 32
    )
)


def append_json(
    path: Path,
    payload,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "a",
        encoding="utf-8",
        newline="\n",
    ) as handle:
        handle.write(
            json.dumps(
                payload
            )
            +
            "\n"
        )


def build_runtime_trace(
    request_id: str,
) -> RuntimeTraceRecord:
    return (
        RuntimeTraceRecord(
            request_id=
                request_id,

            created_at_utc=
                "2026-08-28T12:00:00+00:00",

            method=
                "POST",

            route_template=
                "/planning/ai-native-run",

            status_code=
                200,

            duration_ms=
                125.5,

            workflow_id=
                "prep:12345678",

            run_status=
                "completed",
        )
    )


def build_ai_trace_for_request(
    *,
    request_id: str,
    trace_id: str,
    objective: str = (
        "PRIVATE_OBJECTIVE_DO_NOT_EXPOSE"
    ),
    filename: str = (
        "private-raw-orders.csv"
    ),
) -> AITraceRecord:
    trace = (
        build_ai_trace(
            trace_id=
                trace_id,

            workflow_id=
                "prep:12345678",

            objective=
                objective,

            catalog={
                "datasets": [
                    {
                        "filename":
                            filename,
                    },
                ],
            },

            planner_report={
                "status":
                    "ready",

                "model":
                    "gemma3:4b",

                "timing":
                    {},

                "items":
                    [],
            },

            pipeline_report={
                "trace_id":
                    trace_id,

                "analysis_id":
                    (
                        "analysis:"
                        + trace_id
                    ),

                "analysis_source_type":
                    "automatic",

                "status":
                    "ready",

                "planner_model":
                    "gemma3:4b",

                "tool_model":
                    "qwen2.5:1.5b-instruct",

                "validated_contract_count":
                    0,

                "pipeline_item_count":
                    0,

                "executed_count":
                    1,

                "not_supported_count":
                    0,

                "rejected_count":
                    0,

                "timing":
                    {},

                "items":
                    [],
            },

            ingestion_ms=
                10.0,

            planner_ms=
                20.0,

            native_pipeline_ms=
                30.0,

            total_ms=
                60.0,
        )
    )

    payload = (
        trace.model_dump(
            mode="python"
        )
    )

    payload[
        "request_id"
    ] = request_id

    return (
        AITraceRecord
        .model_validate(
            payload
        )
    )


class TraceExplorerV01Tests(
    unittest.TestCase
):
    def setUp(
        self,
    ) -> None:
        self.temporary_directory = (
            tempfile.TemporaryDirectory()
        )

        self.root = Path(
            self.temporary_directory.name
        )

        self.runtime_path = (
            self.root
            / "runtime.jsonl"
        )

        self.ai_path = (
            self.root
            / "ai.jsonl"
        )

        app = FastAPI()

        app.include_router(
            analysis_run.router
        )

        self.client = (
            TestClient(
                app
            )
        )


    def tearDown(
        self,
    ) -> None:
        self.client.close()

        self.temporary_directory.cleanup()


    def explorer(
        self,
        request_id: str,
    ):
        return (
            get_request_trace_explorer(
                request_id,

                runtime_path=
                    self.runtime_path,

                ai_path=
                    self.ai_path,
            )
        )


    def patch_http_explorer(
        self,
    ):
        return (
            patch.object(
                analysis_run,
                "get_request_trace_explorer",
                side_effect=(
                    lambda request_id:
                        self.explorer(
                            request_id
                        )
                ),
            )
        )


    def test_correlates_runtime_and_ai_with_safe_summary(
        self,
    ) -> None:
        runtime = (
            build_runtime_trace(
                REQUEST_ID
            )
        )

        ai_trace = (
            build_ai_trace_for_request(
                request_id=
                    REQUEST_ID,

                trace_id=
                    "ai:trace-explorer-v01",
            )
        )

        append_json(
            self.runtime_path,
            runtime.model_dump(
                mode="json"
            ),
        )

        append_json(
            self.runtime_path,
            {
                "request_id":
                    REQUEST_ID,

                "filesystem_path":
                    "C:/private/runtime.csv",

                "raw_body":
                    "RUNTIME_POISON",
            },
        )

        append_json(
            self.ai_path,
            ai_trace.model_dump(
                mode="json"
            ),
        )

        append_json(
            self.ai_path,
            {
                "request_id":
                    REQUEST_ID,

                "objective":
                    "AI_MALFORMED_POISON",

                "filesystem_path":
                    "C:/private/ai.jsonl",
            },
        )

        result = (
            self.explorer(
                REQUEST_ID
            )
        )

        self.assertIsNotNone(
            result
        )

        assert result is not None

        self.assertEqual(
            result.explorer_rule_version,
            TRACE_EXPLORER_RULE_VERSION,
        )

        self.assertEqual(
            result.request_id,
            REQUEST_ID,
        )

        self.assertTrue(
            result.runtime_found
        )

        self.assertIsNotNone(
            result.runtime
        )

        assert result.runtime is not None

        self.assertEqual(
            result.runtime.method,
            "POST",
        )

        self.assertEqual(
            result.runtime.route_template,
            "/planning/ai-native-run",
        )

        self.assertEqual(
            result.runtime.status_code,
            200,
        )

        self.assertEqual(
            result.runtime_malformed_line_count,
            1,
        )

        self.assertEqual(
            result.ai_malformed_line_count,
            1,
        )

        self.assertEqual(
            result.ai_trace_count,
            1,
        )

        self.assertEqual(
            result.ai_traces[
                0
            ].trace_id,
            "ai:trace-explorer-v01",
        )

        serialized = (
            result.model_dump_json()
        )

        self.assertNotIn(
            "PRIVATE_OBJECTIVE_DO_NOT_EXPOSE",
            serialized,
        )

        self.assertNotIn(
            "private-raw-orders.csv",
            serialized,
        )

        self.assertNotIn(
            "RUNTIME_POISON",
            serialized,
        )

        self.assertNotIn(
            "AI_MALFORMED_POISON",
            serialized,
        )

        self.assertNotIn(
            "C:/private/",
            serialized,
        )


    def test_runtime_only_is_valid_correlation(
        self,
    ) -> None:
        runtime = (
            build_runtime_trace(
                REQUEST_ID
            )
        )

        append_json(
            self.runtime_path,
            runtime.model_dump(
                mode="json"
            ),
        )

        result = (
            self.explorer(
                REQUEST_ID
            )
        )

        self.assertIsNotNone(
            result
        )

        assert result is not None

        self.assertTrue(
            result.runtime_found
        )

        self.assertEqual(
            result.ai_trace_count,
            0,
        )

        self.assertEqual(
            result.ai_traces,
            [],
        )


    def test_ai_only_is_valid_correlation(
        self,
    ) -> None:
        ai_trace = (
            build_ai_trace_for_request(
                request_id=
                    REQUEST_ID,

                trace_id=
                    "ai:orphan-correlation",
            )
        )

        append_json(
            self.ai_path,
            ai_trace.model_dump(
                mode="json"
            ),
        )

        result = (
            self.explorer(
                REQUEST_ID
            )
        )

        self.assertIsNotNone(
            result
        )

        assert result is not None

        self.assertFalse(
            result.runtime_found
        )

        self.assertIsNone(
            result.runtime
        )

        self.assertEqual(
            result.ai_trace_count,
            1,
        )


    def test_unrelated_records_do_not_match(
        self,
    ) -> None:
        runtime = (
            build_runtime_trace(
                OTHER_REQUEST_ID
            )
        )

        ai_trace = (
            build_ai_trace_for_request(
                request_id=
                    OTHER_REQUEST_ID,

                trace_id=
                    "ai:other-request",
            )
        )

        append_json(
            self.runtime_path,
            runtime.model_dump(
                mode="json"
            ),
        )

        append_json(
            self.ai_path,
            ai_trace.model_dump(
                mode="json"
            ),
        )

        result = (
            self.explorer(
                REQUEST_ID
            )
        )

        self.assertIsNone(
            result
        )


    def test_invalid_request_id_is_rejected(
        self,
    ) -> None:
        invalid_values = [
            "",
            "attacker",
            "http:123",
            (
                "http:"
                + (
                    "F"
                    * 32
                )
            ),
            (
                "http:"
                + (
                    "1"
                    * 31
                )
            ),
        ]

        for value in invalid_values:
            with self.subTest(
                value=value
            ):
                with self.assertRaises(
                    ValueError
                ):
                    normalize_runtime_request_id(
                        value
                    )


    def test_http_explorer_returns_no_store(
        self,
    ) -> None:
        runtime = (
            build_runtime_trace(
                REQUEST_ID
            )
        )

        ai_trace = (
            build_ai_trace_for_request(
                request_id=
                    REQUEST_ID,

                trace_id=
                    "ai:http-explorer",
            )
        )

        append_json(
            self.runtime_path,
            runtime.model_dump(
                mode="json"
            ),
        )

        append_json(
            self.ai_path,
            ai_trace.model_dump(
                mode="json"
            ),
        )

        with self.patch_http_explorer():
            response = self.client.get(
                (
                    "/observability/requests/"
                    + REQUEST_ID
                )
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
                "request_id"
            ],
            REQUEST_ID,
        )

        self.assertTrue(
            payload[
                "runtime_found"
            ]
        )

        self.assertEqual(
            payload[
                "ai_trace_count"
            ],
            1,
        )


    def test_http_invalid_request_id_is_static_422(
        self,
    ) -> None:
        attacker_value = (
            "attacker-controlled-id"
        )

        with self.patch_http_explorer():
            response = self.client.get(
                (
                    "/observability/requests/"
                    + attacker_value
                )
            )

        self.assertEqual(
            response.status_code,
            422,
        )

        self.assertEqual(
            response.json()[
                "detail"
            ],
            (
                "Invalid DataLens server-owned "
                "request identifier."
            ),
        )

        self.assertNotIn(
            attacker_value,
            response.text,
        )


    def test_http_missing_request_is_static_404(
        self,
    ) -> None:
        with self.patch_http_explorer():
            response = self.client.get(
                (
                    "/observability/requests/"
                    + REQUEST_ID
                )
            )

        self.assertEqual(
            response.status_code,
            404,
        )

        self.assertEqual(
            response.json()[
                "detail"
            ],
            (
                "Local request observability "
                "correlation was not found."
            ),
        )


if __name__ == "__main__":
    print(
        "=== DATALENS TRACE EXPLORER v0.1 ==="
    )

    unittest.main(
        verbosity=2
    )
