from __future__ import annotations


import json
import tempfile
import unittest

from pathlib import Path


from app.observability.ai_trace import (
    build_ai_trace,
)

from app.observability.trace_store import (
    get_ai_trace_metrics,
    list_ai_traces,
)


def completed_trace(
    *,
    trace_id: str,
    workflow_id: str | None,
):
    return build_ai_trace(
        trace_id=
            trace_id,

        workflow_id=
            workflow_id,

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

            "retry_count":
                0,

            "normalization_count":
                0,

            "items":
                [],
        },

        pipeline_report={
            "analysis_id":
                f"analysis:{trace_id}",

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


def failed_trace():
    return build_ai_trace(
        trace_id=
            "ai:store-failed-v04",

        workflow_id=
            "prep:store-failed-v04",

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


class AITraceStoreV04Tests(
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


        legacy = completed_trace(
            trace_id=
                "ai:store-legacy-v03",

            workflow_id=
                None,
        ).model_dump(
            mode="json"
        )


        # Simulate the actual historical v0.3 payload:
        # lifecycle/correlation fields did not exist yet.
        legacy[
            "trace_rule_version"
        ] = (
            "ai_trace_v0.3"
        )

        for key in (
            "workflow_id",
            "analysis_id",
            "analysis_source_type",
            "run_status",
            "failure",
        ):
            legacy.pop(
                key,
                None,
            )


        completed = (
            completed_trace(
                trace_id=
                    "ai:store-completed-v04",

                workflow_id=
                    "prep:store-completed-v04",
            )
            .model_dump(
                mode="json"
            )
        )


        failed = (
            failed_trace()
            .model_dump(
                mode="json"
            )
        )


        with self.path.open(
            "w",
            encoding="utf-8",
        ) as handle:
            handle.write(
                json.dumps(
                    legacy
                )
                +
                "\n"
            )

            handle.write(
                json.dumps(
                    completed
                )
                +
                "\n"
            )

            handle.write(
                json.dumps(
                    failed
                )
                +
                "\n"
            )

            handle.write(
                "{malformed json\n"
            )


    def tearDown(
        self,
    ) -> None:
        self.temporary_directory.cleanup()


    def test_list_exposes_v04_correlation_and_lifecycle(
        self,
    ) -> None:
        result = list_ai_traces(
            limit=
                20,

            path=
                self.path,
        )


        self.assertEqual(
            result.trace_count,
            3,
        )

        self.assertEqual(
            result.malformed_line_count,
            1,
        )


        failed = result.traces[0]

        self.assertEqual(
            failed.trace_id,
            "ai:store-failed-v04",
        )

        self.assertEqual(
            failed.workflow_id,
            "prep:store-failed-v04",
        )

        self.assertIsNone(
            failed.analysis_id
        )

        self.assertIsNone(
            failed.analysis_source_type
        )

        self.assertEqual(
            failed.run_status,
            "failed",
        )

        self.assertEqual(
            failed.failure_stage,
            "planner",
        )


        completed = result.traces[1]

        self.assertEqual(
            completed.workflow_id,
            "prep:store-completed-v04",
        )

        self.assertEqual(
            completed.analysis_id,
            "analysis:ai:store-completed-v04",
        )

        self.assertEqual(
            completed.analysis_source_type,
            "initial_request",
        )

        self.assertEqual(
            completed.run_status,
            "completed",
        )

        self.assertIsNone(
            completed.failure_stage
        )


        legacy = result.traces[2]

        self.assertEqual(
            legacy.trace_rule_version,
            "ai_trace_v0.3",
        )

        self.assertIsNone(
            legacy.workflow_id
        )

        self.assertEqual(
            legacy.run_status,
            "completed",
        )

        self.assertIsNone(
            legacy.failure_stage
        )


    def test_metrics_expose_lifecycle_without_replacing_existing_metrics(
        self,
    ) -> None:
        metrics = get_ai_trace_metrics(
            limit=
                200,

            path=
                self.path,
        )


        self.assertEqual(
            metrics.trace_count,
            3,
        )

        self.assertEqual(
            metrics.malformed_line_count,
            1,
        )

        self.assertEqual(
            metrics.analyzed_trace_count,
            3,
        )


        self.assertEqual(
            metrics.completed_trace_count,
            2,
        )

        self.assertEqual(
            metrics.failed_trace_count,
            1,
        )

        self.assertEqual(
            metrics.failure_rate,
            round(
                1 / 3,
                6,
            ),
        )


        failure_stages = {
            item.name:
                item.count

            for item
            in metrics.failure_stages
        }

        self.assertEqual(
            failure_stages,
            {
                "planner":
                    1,
            },
        )


        # Existing metric semantics remain untouched:
        # 2 of the 3 analyzed traces executed at least one item.
        self.assertEqual(
            metrics.executed_trace_count,
            2,
        )

        self.assertEqual(
            metrics.execution_rate,
            round(
                2 / 3,
                6,
            ),
        )


if __name__ == "__main__":
    print(
        "=== DATALENS AI TRACE STORE v0.4 ==="
    )

    unittest.main(
        verbosity=2
    )
