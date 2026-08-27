from __future__ import annotations


import unittest

from types import (
    SimpleNamespace,
)

from unittest.mock import (
    patch,
)


from fastapi import (
    HTTPException,
)


from app.api import (
    analysis_run,
)


class AITraceRuntimeFailureV04Tests(
    unittest.TestCase
):
    def _run_with_planner_failure(
        self,
        *,
        trace_writer,
    ):
        """
        Execute the real run_ai_native_pipeline() runtime path
        while replacing only its external/stage dependencies.

        The failure is injected specifically inside the planner
        so the runtime must classify it as:

            failure.stage == "planner"
        """

        handoff = SimpleNamespace(
            ingestion=None,
            dataset_records=[],
        )


        catalog = {
            "datasets":
                [],
        }


        def fail_planner(
            *args,
            **kwargs,
        ):
            raise RuntimeError(
                "synthetic planner failure"
            )


        with (
            patch.object(
                analysis_run,
                "new_ai_trace_id",
                return_value=
                    "ai:runtime-failure-v04",
            ),

            patch.object(
                analysis_run,
                "load_validated_analysis_input_for_http",
                return_value=
                    handoff,
            ),

            patch.object(
                analysis_run,
                "reject_post_validation_preparation_overrides",
                return_value=
                    None,
            ),

            patch.object(
                analysis_run,
                "prepare_ai_planner_dataset_universe",
                return_value=(
                    [],
                    catalog,
                ),
            ),

            patch.object(
                analysis_run,
                "plan_analyses_with_intent_routing",
                side_effect=
                    fail_planner,
            ),

            patch.object(
                analysis_run,
                "write_ai_trace",
                side_effect=
                    trace_writer,
            ),
        ):
            return (
                analysis_run.run_ai_native_pipeline(
                    dataset_files=
                        None,

                    workflow_id=
                        "prep:runtime-failure-v04",

                    objective=
                        "Revenue by category",

                    planner_model=
                        "gemma3:4b",

                    tool_model=
                        "qwen2.5:1.5b-instruct",

                    approved_action_ids_json=
                        None,

                    semantic_decisions_json=
                        None,

                    approved_semantic_choices_json=
                        None,
                )
            )


    def test_planner_runtime_failure_writes_correlated_failed_trace(
        self,
    ) -> None:
        written_traces = []


        def capture_trace(
            trace,
        ):
            written_traces.append(
                trace
            )

            return SimpleNamespace(
                enabled=True,
                written=True,
                path=None,
                error=None,
            )


        with self.assertRaises(
            HTTPException
        ) as context:
            self._run_with_planner_failure(
                trace_writer=
                    capture_trace
            )


        error = (
            context.exception
        )


        self.assertEqual(
            error.status_code,
            503,
        )

        self.assertEqual(
            error.detail,
            (
                "Local AI processing is unavailable "
                "or returned an invalid response."
            ),
        )


        self.assertEqual(
            len(
                written_traces
            ),
            1,
        )


        trace = (
            written_traces[
                0
            ]
        )


        self.assertEqual(
            trace.trace_rule_version,
            "ai_trace_v0.4",
        )

        self.assertEqual(
            trace.trace_id,
            "ai:runtime-failure-v04",
        )

        self.assertEqual(
            trace.workflow_id,
            "prep:runtime-failure-v04",
        )

        self.assertEqual(
            trace.run_status,
            "failed",
        )

        self.assertIsNotNone(
            trace.failure
        )

        self.assertEqual(
            trace.failure.stage,
            "planner",
        )

        self.assertEqual(
            trace.failure.error_type,
            "RuntimeError",
        )

        self.assertEqual(
            trace.failure.message_safe,
            (
                "DataLens AI execution "
                "failed before completion."
            ),
        )


        # The planner failed before either of these server-owned
        # analysis-correlation fields could exist.
        self.assertIsNone(
            trace.analysis_id
        )

        self.assertIsNone(
            trace.analysis_source_type
        )


        # The planner itself failed, therefore no successful
        # planner or pipeline payload may be fabricated.
        self.assertEqual(
            trace.planner.get(
                "status"
            ),
            None,
        )

        self.assertEqual(
            trace.native_pipeline.get(
                "status"
            ),
            None,
        )


        # Partial runtime timing must remain valid even though
        # execution did not complete.
        self.assertGreaterEqual(
            trace.timings.ingestion_ms,
            0.0,
        )

        self.assertGreaterEqual(
            trace.timings.planner_ms,
            0.0,
        )

        self.assertGreaterEqual(
            trace.timings.total_ms,
            0.0,
        )


    def test_observability_failure_does_not_mask_runtime_error(
        self,
    ) -> None:
        writer_calls = []


        def fail_trace_writer(
            trace,
        ):
            writer_calls.append(
                trace
            )

            raise OSError(
                "synthetic observability failure"
            )


        with self.assertRaises(
            HTTPException
        ) as context:
            self._run_with_planner_failure(
                trace_writer=
                    fail_trace_writer
            )


        error = (
            context.exception
        )


        # The original analytical failure remains authoritative.
        self.assertEqual(
            error.status_code,
            503,
        )

        self.assertEqual(
            error.detail,
            (
                "Local AI processing is unavailable "
                "or returned an invalid response."
            ),
        )


        # We nevertheless prove that observability was attempted.
        self.assertEqual(
            len(
                writer_calls
            ),
            1,
        )

        attempted_trace = (
            writer_calls[
                0
            ]
        )

        self.assertEqual(
            attempted_trace.workflow_id,
            "prep:runtime-failure-v04",
        )

        self.assertEqual(
            attempted_trace.run_status,
            "failed",
        )

        self.assertEqual(
            attempted_trace.failure.stage,
            "planner",
        )


if __name__ == "__main__":
    print(
        "=== DATALENS AI TRACE RUNTIME FAILURE v0.4 ==="
    )

    unittest.main(
        verbosity=2
    )
