from __future__ import annotations

import json
import os
import tempfile
import unittest

from pathlib import Path
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.observability import runtime_trace
from app.observability.runtime_trace import (
    RUNTIME_REQUEST_ID_HEADER,
    RUNTIME_TRACE_RULE_VERSION,
    RuntimeTraceMiddleware,
)


class RuntimeTraceV01Tests(
    unittest.TestCase
):
    def setUp(
        self,
    ) -> None:
        self.temporary_directory = (
            tempfile.TemporaryDirectory()
        )

        self.trace_path = (
            Path(
                self.temporary_directory.name
            )
            / "runtime_requests.jsonl"
        )

        self.environment = patch.dict(
            os.environ,
            {
                "DATALENS_RUNTIME_TRACE_ENABLED":
                    "1",

                "DATALENS_RUNTIME_TRACE_PATH":
                    str(
                        self.trace_path
                    ),
            },
            clear=False,
        )

        self.environment.start()

        app = FastAPI()

        @app.get(
            "/health"
        )
        async def health():
            return {
                "status":
                    "ok"
            }

        @app.post(
            "/workflows/{workflow_id}/echo"
        )
        async def workflow_echo(
            workflow_id: str,
            payload: dict,
        ):
            return {
                "workflow_id":
                    workflow_id,

                "received":
                    bool(payload),
            }

        @app.get(
            "/explode"
        )
        async def explode():
            raise RuntimeError(
                "SUPER_SECRET_EXCEPTION_MESSAGE"
            )

        app.add_middleware(
            RuntimeTraceMiddleware
        )

        self.client = TestClient(
            app,
            raise_server_exceptions=False,
        )

    def tearDown(
        self,
    ) -> None:
        self.client.close()

        self.environment.stop()

        self.temporary_directory.cleanup()

    def _records(
        self,
    ) -> list[
        dict
    ]:
        if not self.trace_path.exists():
            return []

        records = []

        for line in (
            self.trace_path
            .read_text(
                encoding="utf-8"
            )
            .splitlines()
        ):
            if not line.strip():
                continue

            records.append(
                json.loads(
                    line
                )
            )

        return records

    def test_completed_request_has_server_owned_id_and_trace(
        self,
    ) -> None:
        response = self.client.get(
            "/health",
            headers={
                RUNTIME_REQUEST_ID_HEADER:
                    "attacker-controlled-id",
            },
        )

        self.assertEqual(
            response.status_code,
            200,
        )

        request_id = response.headers.get(
            RUNTIME_REQUEST_ID_HEADER
        )

        self.assertIsNotNone(
            request_id
        )

        self.assertTrue(
            request_id.startswith(
                "http:"
            )
        )

        self.assertNotEqual(
            request_id,
            "attacker-controlled-id",
        )

        records = self._records()

        self.assertEqual(
            len(records),
            1,
        )

        trace = records[0]

        self.assertEqual(
            trace["request_id"],
            request_id,
        )

        self.assertEqual(
            trace["trace_rule_version"],
            RUNTIME_TRACE_RULE_VERSION,
        )

        self.assertEqual(
            trace["method"],
            "GET",
        )

        self.assertEqual(
            trace["route_template"],
            "/health",
        )

        self.assertEqual(
            trace["status_code"],
            200,
        )

        self.assertEqual(
            trace["run_status"],
            "completed",
        )

    def test_body_query_headers_and_raw_path_are_not_persisted(
        self,
    ) -> None:
        workflow_id = (
            "prep:abcdef1234567890"
        )

        response = self.client.post(
            (
                "/workflows/"
                + workflow_id
                + "/echo"
                + "?secret=QUERY_SECRET"
            ),
            headers={
                "Authorization":
                    "Bearer HEADER_SECRET",

                "X-Private-Header":
                    "HEADER_PRIVATE_VALUE",
            },
            json={
                "raw_secret":
                    "BODY_PRIVATE_VALUE",
            },
        )

        self.assertEqual(
            response.status_code,
            200,
        )

        records = self._records()

        self.assertEqual(
            len(records),
            1,
        )

        trace = records[0]

        self.assertEqual(
            trace["route_template"],
            (
                "/workflows/"
                "{workflow_id}/echo"
            ),
        )

        # A syntactically valid workflow id in the URL is still
        # client-controlled and must not be trusted as runtime
        # correlation evidence.
        self.assertIsNone(
            trace["workflow_id"]
        )

        serialized = json.dumps(
            trace,
            sort_keys=True,
        )

        forbidden = (
            "QUERY_SECRET",
            "Bearer HEADER_SECRET",
            "HEADER_PRIVATE_VALUE",
            "BODY_PRIVATE_VALUE",
            "raw_secret",
        )

        for value in forbidden:
            self.assertNotIn(
                value,
                serialized,
            )

        privacy = trace[
            "privacy"
        ]

        self.assertFalse(
            privacy[
                "contains_request_body"
            ]
        )

        self.assertFalse(
            privacy[
                "contains_query_string"
            ]
        )

        self.assertFalse(
            privacy[
                "contains_request_headers"
            ]
        )

        self.assertFalse(
            privacy[
                "contains_client_ip"
            ]
        )

    def test_invalid_workflow_identifier_is_not_persisted(
        self,
    ) -> None:
        response = self.client.post(
            (
                "/workflows/"
                "this-is-not-a-server-workflow-id"
                "/echo"
            ),
            json={
                "ok":
                    True,
            },
        )

        self.assertEqual(
            response.status_code,
            200,
        )

        trace = self._records()[0]

        self.assertIsNone(
            trace["workflow_id"]
        )

    def test_unmatched_route_does_not_store_raw_url(
        self,
    ) -> None:
        response = self.client.get(
            (
                "/missing/"
                "RAW_PRIVATE_PATH_VALUE"
                "?token=PRIVATE_QUERY_TOKEN"
            )
        )

        self.assertEqual(
            response.status_code,
            404,
        )

        trace = self._records()[0]

        self.assertEqual(
            trace["route_template"],
            "__unmatched__",
        )

        serialized = json.dumps(
            trace
        )

        self.assertNotIn(
            "RAW_PRIVATE_PATH_VALUE",
            serialized,
        )

        self.assertNotIn(
            "PRIVATE_QUERY_TOKEN",
            serialized,
        )

    def test_unhandled_exception_trace_contains_no_exception_message(
        self,
    ) -> None:
        response = self.client.get(
            "/explode"
        )

        self.assertEqual(
            response.status_code,
            500,
        )

        trace = self._records()[0]

        self.assertEqual(
            trace["run_status"],
            "failed",
        )

        self.assertEqual(
            trace["status_code"],
            500,
        )

        self.assertEqual(
            trace["failure_kind"],
            "unhandled_exception",
        )

        serialized = json.dumps(
            trace
        )

        self.assertNotIn(
            "SUPER_SECRET_EXCEPTION_MESSAGE",
            serialized,
        )

    def test_main_app_installs_runtime_trace_and_exposes_request_id(
        self,
    ) -> None:
        from fastapi.middleware.cors import (
            CORSMiddleware,
        )

        from app.main import (
            app as datalens_app,
        )

        runtime_middleware = [
            item
            for item in datalens_app.user_middleware
            if item.cls is RuntimeTraceMiddleware
        ]

        self.assertEqual(
            len(runtime_middleware),
            1,
        )

        cors_middleware = [
            item
            for item in datalens_app.user_middleware
            if item.cls is CORSMiddleware
        ]

        self.assertEqual(
            len(cors_middleware),
            1,
        )

        exposed_headers = (
            cors_middleware[0]
            .kwargs
            .get(
                "expose_headers",
                [],
            )
        )

        self.assertIn(
            RUNTIME_REQUEST_ID_HEADER,
            exposed_headers,
        )

    def test_observability_writer_failure_does_not_break_request(
        self,
    ) -> None:
        with patch.object(
            runtime_trace,
            "write_runtime_trace",
            side_effect=RuntimeError(
                "writer failure"
            ),
        ):
            response = self.client.get(
                "/health"
            )

        self.assertEqual(
            response.status_code,
            200,
        )

        request_id = response.headers.get(
            RUNTIME_REQUEST_ID_HEADER
        )

        self.assertIsNotNone(
            request_id
        )


if __name__ == "__main__":
    print(
        "=== DATALENS RUNTIME TRACE v0.1 ==="
    )

    unittest.main(
        verbosity=2
    )