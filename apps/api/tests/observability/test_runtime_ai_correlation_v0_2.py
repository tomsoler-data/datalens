from __future__ import annotations

import asyncio
import json
import os
import unittest

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.observability.ai_trace import (
    build_ai_trace,
    new_ai_trace_id,
    write_ai_trace,
)

from app.observability.request_context import (
    current_runtime_request_id,
)

from app.observability.runtime_trace import (
    RUNTIME_REQUEST_ID_HEADER,
    RuntimeTraceMiddleware,
)


TEST_RULE_VERSION = (
    "runtime_ai_correlation_test_v0.2"
)


def build_probe_ai_trace():
    trace_id = (
        new_ai_trace_id()
    )

    return build_ai_trace(
        trace_id=
            trace_id,

        objective=
            "Runtime correlation probe",

        catalog={
            "datasets":
                [],
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
                0,

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
            1.0,

        planner_ms=
            2.0,

        native_pipeline_ms=
            3.0,

        total_ms=
            6.0,
    )


def read_jsonl(
    path: Path,
) -> list[dict]:
    if not path.exists():
        return []

    return [
        json.loads(
            line
        )

        for line
        in path.read_text(
            encoding="utf-8"
        ).splitlines()

        if line.strip()
    ]


class RuntimeAICorrelationV02Tests(
    unittest.TestCase
):
    def build_app(
        self,
    ) -> FastAPI:
        app = FastAPI()

        @app.post(
            "/analysis-probe"
        )
        async def analysis_probe():
            trace = (
                build_probe_ai_trace()
            )

            write_ai_trace(
                trace
            )

            return {
                "trace_id":
                    trace.trace_id,

                "request_id":
                    trace.request_id,
            }

        app.add_middleware(
            RuntimeTraceMiddleware
        )

        return app


    def test_ai_trace_uses_server_owned_request_id(
        self,
    ) -> None:
        attacker_id = (
            "http:"
            + (
                "f"
                * 32
            )
        )

        with TemporaryDirectory() as directory:
            root = Path(
                directory
            )

            ai_path = (
                root
                / "ai.jsonl"
            )

            runtime_path = (
                root
                / "runtime.jsonl"
            )

            with patch.dict(
                os.environ,
                {
                    "DATALENS_AI_TRACE_ENABLED":
                        "1",

                    "DATALENS_AI_TRACE_PATH":
                        str(
                            ai_path
                        ),

                    "DATALENS_RUNTIME_TRACE_ENABLED":
                        "1",

                    "DATALENS_RUNTIME_TRACE_PATH":
                        str(
                            runtime_path
                        ),
                },
                clear=False,
            ):
                with TestClient(
                    self.build_app()
                ) as client:
                    response = client.post(
                        "/analysis-probe",
                        headers={
                            RUNTIME_REQUEST_ID_HEADER:
                                attacker_id,
                        },
                    )

            self.assertEqual(
                response.status_code,
                200,
            )

            server_id = (
                response.headers[
                    RUNTIME_REQUEST_ID_HEADER
                ]
            )

            self.assertNotEqual(
                server_id,
                attacker_id,
            )

            self.assertTrue(
                server_id.startswith(
                    "http:"
                )
            )

            self.assertEqual(
                len(
                    server_id
                ),
                37,
            )

            self.assertEqual(
                response.json()[
                    "request_id"
                ],
                server_id,
            )

            ai_records = (
                read_jsonl(
                    ai_path
                )
            )

            runtime_records = (
                read_jsonl(
                    runtime_path
                )
            )

            self.assertEqual(
                len(
                    ai_records
                ),
                1,
            )

            self.assertEqual(
                len(
                    runtime_records
                ),
                1,
            )

            self.assertEqual(
                ai_records[
                    0
                ][
                    "request_id"
                ],
                server_id,
            )

            self.assertEqual(
                runtime_records[
                    0
                ][
                    "request_id"
                ],
                server_id,
            )


    def test_context_is_reset_after_asgi_request(
        self,
    ) -> None:
        observed = []

        async def inner_app(
            scope,
            receive,
            send,
        ):
            observed.append(
                current_runtime_request_id()
            )

            await send(
                {
                    "type":
                        "http.response.start",

                    "status":
                        204,

                    "headers":
                        [],
                }
            )

            await send(
                {
                    "type":
                        "http.response.body",

                    "body":
                        b"",
                }
            )

        async def run_probe():
            middleware = (
                RuntimeTraceMiddleware(
                    inner_app
                )
            )

            async def receive():
                return {
                    "type":
                        "http.request",

                    "body":
                        b"",

                    "more_body":
                        False,
                }

            async def send(
                message,
            ):
                return None

            with TemporaryDirectory() as directory:
                with patch.dict(
                    os.environ,
                    {
                        "DATALENS_RUNTIME_TRACE_ENABLED":
                            "1",

                        "DATALENS_RUNTIME_TRACE_PATH":
                            str(
                                Path(
                                    directory
                                )
                                / "runtime.jsonl"
                            ),
                    },
                    clear=False,
                ):
                    await middleware(
                        {
                            "type":
                                "http",

                            "method":
                                "GET",

                            "path":
                                "/probe",
                        },

                        receive,
                        send,
                    )

            return (
                current_runtime_request_id()
            )

        after_request = (
            asyncio.run(
                run_probe()
            )
        )

        self.assertEqual(
            len(
                observed
            ),
            1,
        )

        self.assertIsNotNone(
            observed[
                0
            ]
        )

        self.assertTrue(
            observed[
                0
            ].startswith(
                "http:"
            )
        )

        self.assertIsNone(
            after_request
        )


    def test_context_is_reset_after_unhandled_exception(
        self,
    ) -> None:
        observed = []

        async def failing_app(
            scope,
            receive,
            send,
        ):
            observed.append(
                current_runtime_request_id()
            )

            raise RuntimeError(
                "correlation-reset-probe"
            )

        async def run_probe():
            middleware = (
                RuntimeTraceMiddleware(
                    failing_app
                )
            )

            async def receive():
                return {
                    "type":
                        "http.request",

                    "body":
                        b"",

                    "more_body":
                        False,
                }

            async def send(
                message,
            ):
                return None

            with TemporaryDirectory() as directory:
                with patch.dict(
                    os.environ,
                    {
                        "DATALENS_RUNTIME_TRACE_ENABLED":
                            "1",

                        "DATALENS_RUNTIME_TRACE_PATH":
                            str(
                                Path(
                                    directory
                                )
                                / "runtime.jsonl"
                            ),
                    },
                    clear=False,
                ):
                    with self.assertRaises(
                        RuntimeError
                    ):
                        await middleware(
                            {
                                "type":
                                    "http",

                                "method":
                                    "GET",

                                "path":
                                    "/failure-probe",
                            },

                            receive,
                            send,
                        )

            return (
                current_runtime_request_id()
            )

        after_request = (
            asyncio.run(
                run_probe()
            )
        )

        self.assertEqual(
            len(
                observed
            ),
            1,
        )

        self.assertIsNotNone(
            observed[
                0
            ]
        )

        self.assertTrue(
            observed[
                0
            ].startswith(
                "http:"
            )
        )

        self.assertIsNone(
            after_request
        )


    def test_sequential_requests_are_distinct(
        self,
    ) -> None:
        with TemporaryDirectory() as directory:
            root = Path(
                directory
            )

            ai_path = (
                root
                / "ai.jsonl"
            )

            runtime_path = (
                root
                / "runtime.jsonl"
            )

            with patch.dict(
                os.environ,
                {
                    "DATALENS_AI_TRACE_ENABLED":
                        "1",

                    "DATALENS_AI_TRACE_PATH":
                        str(
                            ai_path
                        ),

                    "DATALENS_RUNTIME_TRACE_ENABLED":
                        "1",

                    "DATALENS_RUNTIME_TRACE_PATH":
                        str(
                            runtime_path
                        ),
                },
                clear=False,
            ):
                with TestClient(
                    self.build_app()
                ) as client:
                    first = client.post(
                        "/analysis-probe"
                    )

                    second = client.post(
                        "/analysis-probe"
                    )

            first_id = (
                first.headers[
                    RUNTIME_REQUEST_ID_HEADER
                ]
            )

            second_id = (
                second.headers[
                    RUNTIME_REQUEST_ID_HEADER
                ]
            )

            self.assertNotEqual(
                first_id,
                second_id,
            )

            ai_records = (
                read_jsonl(
                    ai_path
                )
            )

            runtime_records = (
                read_jsonl(
                    runtime_path
                )
            )

            self.assertEqual(
                [
                    item[
                        "request_id"
                    ]
                    for item
                    in ai_records
                ],
                [
                    first_id,
                    second_id,
                ],
            )

            self.assertEqual(
                [
                    item[
                        "request_id"
                    ]
                    for item
                    in runtime_records
                ],
                [
                    first_id,
                    second_id,
                ],
            )


if __name__ == "__main__":
    print(
        (
            "=== DATALENS RUNTIME / AI "
            "CORRELATION v0.2 ==="
        )
    )

    unittest.main(
        verbosity=2
    )
