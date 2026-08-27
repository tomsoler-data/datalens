from __future__ import annotations


import unittest


from app.observability import (
    ai_trace as ai_trace_module,
)

from app.observability.ai_trace import (
    AITraceRecord,
)


# ============================================================
# FIXTURES
# ============================================================


def legacy_v03_payload() -> dict[
    str,
    object,
]:
    return {
        "trace_id":
            "ai:legacy-v03",

        "created_at_utc":
            "2026-08-27T10:00:00+00:00",

        "trace_rule_version":
            "ai_trace_v0.3",

        "objective":
            "Analyse le chiffre d'affaires par categorie.",

        "objective_sha256":
            "legacy-objective-sha256",

        "datasets":
            [],

        "planner":
            {},

        "native_pipeline":
            {},

        "timings":
            {
                "ingestion_ms":
                    1.0,

                "planner_ms":
                    2.0,

                "native_pipeline_ms":
                    3.0,

                "total_ms":
                    6.0,
            },

        "privacy":
            {
                "storage_scope":
                    "local_jsonl",

                "contains_raw_dataset_rows":
                    False,

                "contains_uploaded_file_contents":
                    False,

                "contains_document_chunks":
                    False,

                "contains_objective_text":
                    True,

                "note":
                    (
                        "Legacy observability trace "
                        "without v0.4 correlation fields."
                    ),
            },
    }


# ============================================================
# CONTRACT TESTS
# ============================================================


class AITraceContractV04Tests(
    unittest.TestCase
):
    def test_v03_payload_remains_readable(
        self,
    ) -> None:
        trace = (
            AITraceRecord.model_validate(
                legacy_v03_payload()
            )
        )


        self.assertEqual(
            trace.trace_id,
            "ai:legacy-v03",
        )

        self.assertEqual(
            trace.trace_rule_version,
            "ai_trace_v0.3",
        )


    def test_v04_is_current_rule_version(
        self,
    ) -> None:
        self.assertEqual(
            ai_trace_module
            .AI_TRACE_RULE_VERSION,
            "ai_trace_v0.4",
        )


    def test_v04_correlation_and_lifecycle_fields_exist(
        self,
    ) -> None:
        fields = (
            AITraceRecord.model_fields
        )


        expected_fields = {
            "workflow_id",
            "analysis_id",
            "analysis_source_type",
            "run_status",
            "failure",
        }


        self.assertTrue(
            expected_fields.issubset(
                fields
            ),
            (
                "AITraceRecord v0.4 must expose "
                "workflow_id, analysis_id, "
                "analysis_source_type, run_status "
                "and failure."
            ),
        )


    def test_v03_defaults_preserve_backward_compatibility(
        self,
    ) -> None:
        trace = (
            AITraceRecord.model_validate(
                legacy_v03_payload()
            )
        )


        self.assertTrue(
            hasattr(
                trace,
                "workflow_id",
            )
        )

        self.assertTrue(
            hasattr(
                trace,
                "analysis_id",
            )
        )

        self.assertTrue(
            hasattr(
                trace,
                "analysis_source_type",
            )
        )

        self.assertTrue(
            hasattr(
                trace,
                "run_status",
            )
        )

        self.assertTrue(
            hasattr(
                trace,
                "failure",
            )
        )


        self.assertIsNone(
            trace.workflow_id
        )

        self.assertIsNone(
            trace.analysis_id
        )

        self.assertIsNone(
            trace.analysis_source_type
        )

        self.assertEqual(
            trace.run_status,
            "completed",
        )

        self.assertIsNone(
            trace.failure
        )


    def test_v04_completed_trace_accepts_server_owned_correlation(
        self,
    ) -> None:
        payload = (
            legacy_v03_payload()
        )


        payload.update(
            {
                "trace_id":
                    "ai:test-v04",

                "trace_rule_version":
                    "ai_trace_v0.4",

                "workflow_id":
                    "prep:test-v04",

                "analysis_id":
                    "analysis:ai:test-v04",

                "analysis_source_type":
                    "initial_request",

                "run_status":
                    "completed",

                "failure":
                    None,
            }
        )


        trace = (
            AITraceRecord.model_validate(
                payload
            )
        )


        self.assertEqual(
            trace.workflow_id,
            "prep:test-v04",
        )

        self.assertEqual(
            trace.analysis_id,
            "analysis:ai:test-v04",
        )

        self.assertEqual(
            trace.analysis_source_type,
            "initial_request",
        )

        self.assertEqual(
            trace.run_status,
            "completed",
        )

        self.assertIsNone(
            trace.failure
        )


    def test_v04_failure_contract_is_structured(
        self,
    ) -> None:
        self.assertTrue(
            hasattr(
                ai_trace_module,
                "AITraceFailure",
            ),
            (
                "Observability v0.4 must expose "
                "AITraceFailure."
            ),
        )


        failure_type = getattr(
            ai_trace_module,
            "AITraceFailure",
        )


        failure = (
            failure_type.model_validate(
                {
                    "stage":
                        "planner",

                    "error_type":
                        "RuntimeError",

                    "message_safe":
                        (
                            "The local planner "
                            "could not complete."
                        ),
                }
            )
        )


        self.assertEqual(
            failure.stage,
            "planner",
        )

        self.assertEqual(
            failure.error_type,
            "RuntimeError",
        )

        self.assertEqual(
            failure.message_safe,
            (
                "The local planner "
                "could not complete."
            ),
        )


        payload = (
            legacy_v03_payload()
        )


        payload.update(
            {
                "trace_id":
                    "ai:test-v04-failure",

                "trace_rule_version":
                    "ai_trace_v0.4",

                "workflow_id":
                    "prep:test-v04",

                "analysis_id":
                    None,

                "analysis_source_type":
                    None,

                "run_status":
                    "failed",

                "failure":
                    failure.model_dump(
                        mode="json"
                    ),
            }
        )


        trace = (
            AITraceRecord.model_validate(
                payload
            )
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


    def test_build_ai_trace_carries_server_owned_correlation(
        self,
    ) -> None:
        trace = (
            ai_trace_module.build_ai_trace(
                trace_id=
                    "ai:correlation-v04",

                workflow_id=
                    "prep:correlation-v04",

                objective=
                    "Revenue by category",

                catalog=
                    {
                        "datasets":
                            [],
                    },

                planner_report=
                    {
                        "status":
                            "ready",

                        "model":
                            "gemma3:4b",

                        "timing":
                            {},

                        "items":
                            [],
                    },

                pipeline_report=
                    {
                        "trace_id":
                            "ai:correlation-v04",

                        "analysis_id":
                            (
                                "analysis:"
                                "ai:correlation-v04"
                            ),

                        "analysis_source_type":
                            "initial_request",

                        "status":
                            "ready",

                        "planner_model":
                            "gemma3:4b",

                        "tool_model":
                            "qwen2.5:1.5b-instruct",

                        "validated_contract_count":
                            1,

                        "pipeline_item_count":
                            1,

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
                    1.0,

                planner_ms=
                    2.0,

                native_pipeline_ms=
                    3.0,

                total_ms=
                    6.0,
            )
        )


        self.assertEqual(
            trace.trace_id,
            "ai:correlation-v04",
        )

        self.assertEqual(
            trace.workflow_id,
            "prep:correlation-v04",
        )

        self.assertEqual(
            trace.analysis_id,
            "analysis:ai:correlation-v04",
        )

        self.assertEqual(
            trace.analysis_source_type,
            "initial_request",
        )

        self.assertEqual(
            trace.run_status,
            "completed",
        )

        self.assertIsNone(
            trace.failure
        )


if __name__ == "__main__":
    unittest.main(
        verbosity=2
    )
